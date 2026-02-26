#!/usr/bin/env python3
# chi_squared_rm_trl_new.py
#
# Fixes for newer TRL:
# - SFTTrainer no longer accepts `tokenizer=...` in __init__ (use processing_class=tokenizer)
# - warmup_ratio deprecated (use warmup_steps)
# - pairwise chi-squared (SPIN-style) objective + DDP + wandb
#
# Expected dataset format (jsonl):
# {
#   "prompt": "...",
#   "responses": ["<gen_response>", "<solution>"],
#   "true_solution": "..."   # optional
# }

import argparse
import json
import os
import pickle
import re
import shutil
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
import wandb
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


SYSTEM_PROMPT = "You are a helpful assistant."
USER_PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. The last line of your response "
    "should be of the form Answer: $Answer (without quotes) where $Answer is the "
    "answer to the problem.\n\n"
    "{problem}\n\n"
    "Remember to put your answer on its own line after \"Answer:\"."
)


def normalize_prompt(p: Any) -> str:
    if isinstance(p, str):
        return p
    if isinstance(p, (list, tuple)):
        return "\n".join(str(x) for x in p)
    if isinstance(p, dict):
        for k in ["prompt", "question", "content", "text"]:
            if k in p:
                return normalize_prompt(p[k])
        return json.dumps(p, ensure_ascii=False)
    return str(p)


def format_prompt(tokenizer: Any, question: Any) -> str:
    p = normalize_prompt(question)
    has_chat = hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None)
    # if has_chat:
    user_prompt = user_prompt = USER_PROMPT_TEMPLATE.format(problem=p)
    messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    # return f"User: {p}\nAssistant:"


def clean_answer(ans: Optional[str]) -> str:
    if ans is None:
        return ""
    return str(ans).strip()


def maybe_strip_to_answer(solution: str, answer: str, strip_solution_to_answer: bool) -> str:
    if not strip_solution_to_answer or not answer:
        return solution
    idx = solution.find(answer)
    if idx != -1:
        return solution[: idx + len(answer)]
    # fallback: whitespace-normalized search (brittle)
    sol_norm = re.sub(r"\s+", " ", solution)
    ans_norm = re.sub(r"\s+", " ", answer)
    if sol_norm.find(ans_norm) != -1:
        return solution
    return solution


def build_pos_text(
    solution: Any,
    answer: Optional[str],
    use_answer_only_pos: bool,
    append_final_answer: bool,
    final_answer_prefix: str,
    strip_solution_to_answer: bool,
) -> str:
    ans = clean_answer(answer)
    if use_answer_only_pos and ans:
        return ans

    pos = solution if solution is not None else ""
    pos = str(pos)
    pos = maybe_strip_to_answer(pos, ans, strip_solution_to_answer)

    if append_final_answer and ans:
        pos = pos.rstrip() + final_answer_prefix + ans
    return pos


def encode_prompt_response(tokenizer: Any, prompt: str, response: str, max_length: int) -> Dict[str, List[int]]:
    p_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    r_ids = tokenizer(response, add_special_tokens=False).input_ids
    if tokenizer.eos_token_id is not None:
        r_ids = r_ids + [tokenizer.eos_token_id]

    input_ids = p_ids + r_ids
    if len(input_ids) > max_length:
        input_ids = input_ids[-max_length:]

    attn = [1] * len(input_ids)
    p_len = min(len(p_ids), len(input_ids))
    labels = [-100] * p_len + input_ids[p_len:]
    return {
        "input_ids": input_ids,
        "attention_mask": attn,
        "labels": labels,
    }


def distributed_barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def is_valid_saved_dataset_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    needed = ["dataset_info.json", "state.json"]
    return all(os.path.isfile(os.path.join(path, n)) for n in needed)


def sequence_logprob_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    logits: [B, T, V], labels: [B, T] with -100 masked tokens.
    returns: sum log p over non-masked tokens, shape [B]
    """
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]

    loss_mask = labels != -100
    safe_labels = labels.masked_fill(~loss_mask, 0)

    # Compute log-probs in fp32 without creating an explicit fp32 copy first.
    log_probs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    per_token_logps = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    loss_mask = loss_mask.to(per_token_logps.dtype)
    return (per_token_logps * loss_mask).sum(-1)


@torch.inference_mode()
def precompute_ref_logps_for_dataset(
    dataset,
    ref_model,
    device: torch.device,
    pad_token_id: int,
    batch_size: int,
    is_main: bool = False,
):
    ref_model.eval()
    n = len(dataset)
    logp_ref_pos_all: List[float] = []
    logp_ref_neg_all: List[float] = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = dataset[start:end]

        pos_input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x, dtype=torch.long) for x in batch["pos_input_ids"]],
            batch_first=True,
            padding_value=pad_token_id,
        ).to(device, non_blocking=True)
        pos_attention_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x, dtype=torch.long) for x in batch["pos_attention_mask"]],
            batch_first=True,
            padding_value=0,
        ).to(device, non_blocking=True)
        pos_labels = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x, dtype=torch.long) for x in batch["pos_labels"]],
            batch_first=True,
            padding_value=-100,
        ).to(device, non_blocking=True)

        neg_input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x, dtype=torch.long) for x in batch["neg_input_ids"]],
            batch_first=True,
            padding_value=pad_token_id,
        ).to(device, non_blocking=True)
        neg_attention_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x, dtype=torch.long) for x in batch["neg_attention_mask"]],
            batch_first=True,
            padding_value=0,
        ).to(device, non_blocking=True)
        neg_labels = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x, dtype=torch.long) for x in batch["neg_labels"]],
            batch_first=True,
            padding_value=-100,
        ).to(device, non_blocking=True)

        pos_out = ref_model(
            input_ids=pos_input_ids,
            attention_mask=pos_attention_mask,
            use_cache=False,
        )
        logp_ref_pos = sequence_logprob_from_logits(pos_out.logits, pos_labels)
        del pos_out

        neg_out = ref_model(
            input_ids=neg_input_ids,
            attention_mask=neg_attention_mask,
            use_cache=False,
        )
        logp_ref_neg = sequence_logprob_from_logits(neg_out.logits, neg_labels)
        del neg_out

        logp_ref_pos_all.extend(logp_ref_pos.detach().cpu().tolist())
        logp_ref_neg_all.extend(logp_ref_neg.detach().cpu().tolist())

        if is_main and (start == 0 or end == n or ((start // batch_size + 1) % 200 == 0)):
            print(f"[precompute_ref] {end}/{n}", flush=True)

    if "logp_ref_pos" in dataset.column_names:
        dataset = dataset.remove_columns("logp_ref_pos")
    if "logp_ref_neg" in dataset.column_names:
        dataset = dataset.remove_columns("logp_ref_neg")
    dataset = dataset.add_column("logp_ref_pos", logp_ref_pos_all)
    dataset = dataset.add_column("logp_ref_neg", logp_ref_neg_all)
    return dataset


@dataclass
class PairwiseCollatorNew:
    tokenizer: Any
    max_length: int
    use_answer_only_pos: bool = False
    append_final_answer: bool = False
    final_answer_prefix: str = "\n\nFinal Answer: "
    strip_solution_to_answer: bool = False

    def _encode(self, prompt: str, response: str) -> Dict[str, torch.Tensor]:
        enc = encode_prompt_response(self.tokenizer, prompt, response, self.max_length)
        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(enc["labels"], dtype=torch.long),
        }

    @staticmethod
    def _clean_answer(ans: Optional[str]) -> str:
        return clean_answer(ans)

    def _maybe_strip_to_answer(self, solution: str, answer: str) -> str:
        return maybe_strip_to_answer(solution, answer, self.strip_solution_to_answer)

    def _build_pos_text(self, solution: str, answer: Optional[str]) -> str:
        return build_pos_text(
            solution=solution,
            answer=answer,
            use_answer_only_pos=self.use_answer_only_pos,
            append_final_answer=self.append_final_answer,
            final_answer_prefix=self.final_answer_prefix,
            strip_solution_to_answer=self.strip_solution_to_answer,
        )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pos_items, neg_items = [], []
        cached_logp_ref_pos: List[float] = []
        cached_logp_ref_neg: List[float] = []
        has_cached_ref = True
        tok = self.tokenizer

        for f in features:
            if "pos_input_ids" in f and "neg_input_ids" in f:
                pos_items.append(
                    {
                        "input_ids": torch.tensor(f["pos_input_ids"], dtype=torch.long),
                        "attention_mask": torch.tensor(f["pos_attention_mask"], dtype=torch.long),
                        "labels": torch.tensor(f["pos_labels"], dtype=torch.long),
                    }
                )
                neg_items.append(
                    {
                        "input_ids": torch.tensor(f["neg_input_ids"], dtype=torch.long),
                        "attention_mask": torch.tensor(f["neg_attention_mask"], dtype=torch.long),
                        "labels": torch.tensor(f["neg_labels"], dtype=torch.long),
                    }
                )
                if "logp_ref_pos" in f and "logp_ref_neg" in f:
                    cached_logp_ref_pos.append(float(f["logp_ref_pos"]))
                    cached_logp_ref_neg.append(float(f["logp_ref_neg"]))
                else:
                    has_cached_ref = False
                continue

            q = f["prompt"]
            gen_resp = f["responses"][0]   # negative
            solution = f["responses"][1]   # positive
            answer = f.get("true_solution", None)

            prompt = format_prompt(tok, q)
            pos_text = self._build_pos_text(solution=solution, answer=answer)
            neg_text = gen_resp

            pos_items.append(self._encode(prompt, pos_text))
            neg_items.append(self._encode(prompt, neg_text))
            has_cached_ref = False

        pad_id = tok.pad_token_id

        def pad(items, key, pad_val):
            seqs = [it[key] for it in items]
            return torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=pad_val)

        out = {
            "pos_input_ids": pad(pos_items, "input_ids", pad_id),
            "pos_attention_mask": pad(pos_items, "attention_mask", 0),
            "pos_labels": pad(pos_items, "labels", -100),
            "neg_input_ids": pad(neg_items, "input_ids", pad_id),
            "neg_attention_mask": pad(neg_items, "attention_mask", 0),
            "neg_labels": pad(neg_items, "labels", -100),
        }
        if has_cached_ref and len(cached_logp_ref_pos) == len(features):
            out["logp_ref_pos"] = torch.tensor(cached_logp_ref_pos, dtype=torch.float32)
            out["logp_ref_neg"] = torch.tensor(cached_logp_ref_neg, dtype=torch.float32)
        return out


class PairwiseRatioMSETrainer(SFTTrainer):
    """
    Uses new TRL SFTTrainer signature:
      - do NOT pass tokenizer=...
      - pass processing_class=tokenizer (or omit; we use custom collator anyway)

    Objective (aligned with trainer.py):
      0.5 * (beta * log(pi/pi_ref)(y+) - r_max)^2
    + 0.5 * (beta * log(pi/pi_ref)(y-) - r_min)^2
    + reg_coef * ((beta * log(pi/pi_ref)(y+))^2 + (beta * log(pi/pi_ref)(y-))^2)
    """

    def __init__(
        self,
        *args,
        ref_model=None,
        use_precomputed_ref_logps: bool = False,
        beta=0.1,
        r_max=0.5,
        r_min=-0.5,
        logratio_clip=0.0,
        reg_coef=0.005,
        length_normalize=False,
        wandb_log_internal=False,
        **kwargs,
    ):
        # kwargs["skip_prepare_dataset"] = True
        super().__init__(*args, **kwargs)

        self.use_precomputed_ref_logps = bool(use_precomputed_ref_logps)
        self.ref_model = ref_model
        self._ref_device = self.accelerator.device
        self._ref_is_deepspeed = False
        self.beta = beta
        self.r_max = r_max
        self.r_min = r_min
        if logratio_clip and logratio_clip > 0:
            raise ValueError(
                "--logratio_clip is unsupported to stay aligned with trainer.py (SPINTrainer.spin_loss)."
            )
        self.logratio_clip = 0.0
        self.reg_coef = reg_coef
        if length_normalize:
            raise ValueError(
                "--length_normalize is unsupported. log_ps are computed as masked sums: "
                "(per_token_logps * loss_mask).sum(-1)."
            )
        self.length_normalize = False
        self.wandb_log_internal = wandb_log_internal

        if self.use_precomputed_ref_logps:
            self.ref_model = None
        else:
            if self.ref_model is None:
                raise ValueError("ref_model must be provided unless --precompute_ref_logps is enabled.")

            self.ref_model.eval()

            if getattr(self, "is_deepspeed_enabled", False):
                self.ref_model = self._prepare_ref_deepspeed(self.ref_model)
                self._ref_is_deepspeed = True
            else:
                self.ref_model.to(self._ref_device)

            # Freeze reference model params after wrapping.
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad_(False)

    def _prepare_ref_deepspeed(self, model):
        try:
            import deepspeed
        except Exception as exc:
            raise RuntimeError("DeepSpeed is required to ZeRO-wrap reference model.") from exc

        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)
        args = self.args

        def _auto(v, fallback):
            return fallback if isinstance(v, str) and v == "auto" else v

        world_size = int(getattr(self.accelerator, "num_processes", 1))
        micro_bs = int(getattr(args, "per_device_train_batch_size", 1))
        grad_acc = int(getattr(args, "gradient_accumulation_steps", 1))
        total_bs = micro_bs * grad_acc * max(1, world_size)

        # For ref-model ZeRO wrapping, keep config numerically concrete.
        config_kwargs["train_micro_batch_size_per_gpu"] = int(
            _auto(config_kwargs.get("train_micro_batch_size_per_gpu"), micro_bs)
        )
        config_kwargs["gradient_accumulation_steps"] = int(
            _auto(config_kwargs.get("gradient_accumulation_steps"), grad_acc)
        )
        config_kwargs["train_batch_size"] = int(_auto(config_kwargs.get("train_batch_size"), total_bs))
        config_kwargs["gradient_clipping"] = float(
            _auto(config_kwargs.get("gradient_clipping"), float(getattr(args, "max_grad_norm", 1.0)))
        )

        bf16_cfg = config_kwargs.get("bf16")
        if isinstance(bf16_cfg, dict):
            bf16_cfg["enabled"] = bool(_auto(bf16_cfg.get("enabled"), bool(getattr(args, "bf16", False))))

        fp16_cfg = config_kwargs.get("fp16")
        if isinstance(fp16_cfg, dict):
            fp16_cfg["enabled"] = bool(_auto(fp16_cfg.get("enabled"), bool(getattr(args, "fp16", False))))

        opt_cfg = config_kwargs.get("optimizer")
        if isinstance(opt_cfg, dict):
            opt_params = opt_cfg.setdefault("params", {})
            opt_params["lr"] = float(_auto(opt_params.get("lr"), float(getattr(args, "learning_rate", 5e-7))))
            opt_params["betas"] = _auto(
                opt_params.get("betas"),
                [float(getattr(args, "adam_beta1", 0.9)), float(getattr(args, "adam_beta2", 0.999))],
            )
            opt_params["eps"] = float(_auto(opt_params.get("eps"), float(getattr(args, "adam_epsilon", 1e-8))))
            opt_params["weight_decay"] = float(
                _auto(opt_params.get("weight_decay"), float(getattr(args, "weight_decay", 0.0)))
            )

        # Ref model is eval-only; avoid constructing a scheduler from unresolved "auto" fields.
        config_kwargs.pop("scheduler", None)

        zero_opt = config_kwargs.get("zero_optimization", {})
        stage = int(zero_opt.get("stage", 0))

        if hasattr(model, "config") and stage == 3:
            hidden_size = (
                max(model.config.hidden_sizes)
                if getattr(model.config, "hidden_sizes", None)
                else getattr(model.config, "hidden_size", None)
            )
            if hidden_size is not None:
                zero_opt["reduce_bucket_size"] = hidden_size * hidden_size
                zero_opt["stage3_param_persistence_threshold"] = 10 * hidden_size
                zero_opt["stage3_prefetch_bucket_size"] = int(0.9 * hidden_size * hidden_size)

        # Keep parity with trainer.py semantics.
        if stage != 3:
            zero_opt["stage"] = 0

        config_kwargs["zero_optimization"] = zero_opt

        # Keep each param group dtype-homogeneous to avoid ZeRO-3 defragment dtype asserts.
        grouped: Dict[torch.dtype, List[torch.nn.Parameter]] = {}
        for p in model.parameters():
            if not p.requires_grad or not p.is_floating_point():
                continue
            grouped.setdefault(p.dtype, []).append(p)
        model_parameters = [{"params": ps} for ps in grouped.values() if ps]
        if not model_parameters:
            model_parameters = [{"params": [p for p in model.parameters() if p.is_floating_point()]}]

        model_engine, *_ = deepspeed.initialize(
            model=model,
            model_parameters=model_parameters,
            config=config_kwargs,
        )
        model_engine.eval()
        return model_engine

    def _seq_logprob(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if labels.device != logits.device:
            labels = labels.to(logits.device, non_blocking=True)
        return sequence_logprob_from_logits(logits, labels)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        policy_model = model
        wrapped_model = getattr(self, "model_wrapped", None)
        if wrapped_model is not None and wrapped_model is not self.model:
            policy_model = wrapped_model
        deepspeed_model = getattr(self, "deepspeed", None)
        if deepspeed_model is not None and deepspeed_model is not self.model:
            policy_model = deepspeed_model

        pos_input_ids = inputs["pos_input_ids"]
        pos_attention_mask = inputs["pos_attention_mask"]
        pos_labels = inputs["pos_labels"]

        neg_input_ids = inputs["neg_input_ids"]
        neg_attention_mask = inputs["neg_attention_mask"]
        neg_labels = inputs["neg_labels"]

        # Low-VRAM path: compute pos/neg separately to avoid doubling sequence activations.
        pos_policy_out = policy_model(
            input_ids=pos_input_ids,
            attention_mask=pos_attention_mask,
            use_cache=False,
        )
        policy_device = pos_policy_out.logits.device
        logp_pi_pos = self._seq_logprob(pos_policy_out.logits, pos_labels)
        del pos_policy_out

        neg_policy_out = policy_model(
            input_ids=neg_input_ids,
            attention_mask=neg_attention_mask,
            use_cache=False,
        )
        logp_pi_neg = self._seq_logprob(neg_policy_out.logits, neg_labels)
        del neg_policy_out

        if self.use_precomputed_ref_logps:
            if "logp_ref_pos" not in inputs or "logp_ref_neg" not in inputs:
                raise ValueError(
                    "Missing precomputed ref log-probs in batch. "
                    "Ensure cached dataset includes logp_ref_pos/logp_ref_neg."
                )
            logp_ref_pos = inputs["logp_ref_pos"]
            logp_ref_neg = inputs["logp_ref_neg"]
            if not torch.is_tensor(logp_ref_pos):
                logp_ref_pos = torch.tensor(logp_ref_pos, dtype=torch.float32)
            if not torch.is_tensor(logp_ref_neg):
                logp_ref_neg = torch.tensor(logp_ref_neg, dtype=torch.float32)
        else:
            # Under ZeRO/FSDP, inferring from parameter.device is unreliable.
            # Keep ref forward on same device as policy logits when ref is not DeepSpeed-wrapped.
            if not self._ref_is_deepspeed and self._ref_device != policy_device:
                self.ref_model.to(policy_device)
                self._ref_device = policy_device

            with torch.no_grad():
                ref_device = policy_device if self._ref_is_deepspeed else self._ref_device
                pos_input_ids_ref = pos_input_ids.to(ref_device, non_blocking=True)
                pos_attention_mask_ref = pos_attention_mask.to(ref_device, non_blocking=True)
                neg_input_ids_ref = neg_input_ids.to(ref_device, non_blocking=True)
                neg_attention_mask_ref = neg_attention_mask.to(ref_device, non_blocking=True)

                pos_ref_out = self.ref_model(
                    input_ids=pos_input_ids_ref,
                    attention_mask=pos_attention_mask_ref,
                    use_cache=False,
                )
                logp_ref_pos = self._seq_logprob(pos_ref_out.logits, pos_labels)
                del pos_ref_out

                neg_ref_out = self.ref_model(
                    input_ids=neg_input_ids_ref,
                    attention_mask=neg_attention_mask_ref,
                    use_cache=False,
                )
                logp_ref_neg = self._seq_logprob(neg_ref_out.logits, neg_labels)
                del neg_ref_out

        if logp_ref_pos.device != logp_pi_pos.device:
            logp_ref_pos = logp_ref_pos.to(logp_pi_pos.device, non_blocking=True)
        if logp_ref_neg.device != logp_pi_neg.device:
            logp_ref_neg = logp_ref_neg.to(logp_pi_neg.device, non_blocking=True)

        lr_pos = logp_pi_pos - logp_ref_pos
        lr_neg = logp_pi_neg - logp_ref_neg

        pred_pos = self.beta * lr_pos
        pred_neg = self.beta * lr_neg

        # Match trainer.py spin_loss:
        #   0.5*(pred_pos-r_max)^2 + 0.5*(pred_neg-r_min)^2 + reg_coef*(pred_pos^2 + pred_neg^2)
        loss_pos = 0.5 * (pred_pos - self.r_max).pow(2)
        loss_neg = 0.5 * (pred_neg - self.r_min).pow(2)
        loss_reg = self.reg_coef * (pred_pos.pow(2) + pred_neg.pow(2))
        loss = (loss_pos + loss_neg + loss_reg).mean()

        if self.wandb_log_internal and self.state.is_world_process_zero and wandb.run is not None:
            wandb.log(
                {
                    "train/loss": float(loss.item()),
                    "train/loss_pos": float(loss_pos.mean().item()),
                    "train/loss_neg": float(loss_neg.mean().item()),
                    "train/loss_reg": float(loss_reg.mean().item()),
                    "train/lr_pos_mean": float(lr_pos.mean().item()),
                    "train/lr_neg_mean": float(lr_neg.mean().item()),
                    "train/pred_pos_mean": float(pred_pos.mean().item()),
                    "train/pred_neg_mean": float(pred_neg.mean().item()),
                    "train/beta": float(self.beta),
                    "train/r_max": float(self.r_max),
                    "train/r_min": float(self.r_min),
                    "train/reg_coef": float(self.reg_coef),
                    "train/logratio_clip": float(self.logratio_clip),
                    "train/length_normalize": int(self.length_normalize),
                },
                step=int(self.state.global_step),
            )

        if return_outputs:
            return (
                loss,
                {
                    "logp_pi_pos": logp_pi_pos.detach(),
                    "logp_pi_neg": logp_pi_neg.detach(),
                    "logp_ref_pos": logp_ref_pos.detach(),
                    "logp_ref_neg": logp_ref_neg.detach(),
                },
            )
        return loss

    def _prepare_dataset(self, dataset, *args, **kwargs):
        # TRL older versions: SFTTrainer will try to tokenize dataset using dataset_text_field="text".
        # Our training is pairwise and we provide a custom data_collator that already returns tensors.
        # So we bypass TRL's dataset preparation entirely.
        return dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paired_jsonl", required=True)
    ap.add_argument("--model_name_or_path", required=True)
    ap.add_argument("--ref_model_name_or_path", required=True)
    ap.add_argument("--output_dir", required=True)

    ap.add_argument("--beta", type=float, default=0.001)
    ap.add_argument("--r_max", type=float, default=1.0)
    ap.add_argument("--r_min", type=float, default=-1.0)

    ap.add_argument("--max_length", type=int, default=9216)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--dataloader_num_workers", type=int, default=4)
    ap.add_argument("--learning_rate", type=float, default=5.0e-7)
    ap.add_argument("--num_train_epochs", type=float, default=3.0)
    ap.add_argument("--warmup_steps", type=float, default=0)
    ap.add_argument("--logging_steps", type=int, default=1)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        help="Attention backend for from_pretrained (e.g. sdpa, eager, flash_attention_2).",
    )

    ap.add_argument("--logratio_clip", type=float, default=0.0,
                    help="Deprecated/unsupported (kept for CLI compatibility); must remain 0.")
    ap.add_argument("--reg_coef", type=float, default=0.005,
                    help="Quadratic regularization coefficient used in the SPIN-style chi-squared loss.")

    # New dataset usage knobs
    ap.add_argument("--use_answer_only_pos", action="store_true",
                    help="If set, pos uses 'answer' only (fallback to solution if answer missing).")
    ap.add_argument("--append_final_answer", action="store_true",
                    help="If set, append 'Final Answer: {answer}' to pos solution when answer exists.")
    ap.add_argument("--final_answer_prefix", type=str, default="\n\nFinal Answer: ",
                    help="Prefix used when --append_final_answer is enabled.")
    ap.add_argument("--strip_solution_to_answer", action="store_true",
                    help="If set, try to cut solution up to first occurrence of answer (brittle).")

    # Reward-model stability knob
    ap.add_argument("--length_normalize", action="store_true",
                    help="Deprecated/unsupported: sequence logprob is always summed over loss-masked tokens.")
    ap.add_argument(
        "--pretokenize_num_proc",
        type=int,
        default=max(1, min(16, os.cpu_count() or 1)),
        help="If >0, pretokenize entire dataset with datasets.map(num_proc=...).",
    )
    ap.add_argument(
        "--pretokenize_batch_size",
        type=int,
        default=64,
        help="Batch size for pretokenization map.",
    )
    ap.add_argument(
        "--pretokenize_cache_dir",
        type=str,
        default="",
        help="Optional cache dir for pretokenized dataset. Defaults to <output_dir>/tokenized_train_ds.",
    )
    ap.add_argument(
        "--reuse_pretokenized_cache",
        action="store_true",
        help="Reuse existing pretokenized cache directory if present.",
    )
    ap.add_argument(
        "--pretokenize_wait_timeout_sec",
        type=int,
        default=7200,
        help="How long non-main ranks wait for rank-0 pretokenization cache to become ready.",
    )
    ap.add_argument(
        "--precompute_ref_logps",
        action="store_true",
        help=(
            "Precompute logp_ref_pos/logp_ref_neg with the reference model into the tokenized cache, "
            "then train without loading ref model."
        ),
    )
    ap.add_argument(
        "--precompute_ref_batch_size",
        type=int,
        default=1,
        help="Batch size used for precomputing reference log-probs.",
    )

    # wandb knobs
    ap.add_argument("--wandb_project", type=str, default="rm-pairwise-ratio-mse")
    ap.add_argument("--wandb_name", type=str, default="")
    ap.add_argument("--wandb_group", type=str, default="")
    ap.add_argument("--wandb_tags", type=str, default="")
    ap.add_argument("--wandb_entity", type=str, default="")
    ap.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    ap.add_argument("--wandb_log_internal", action="store_true",
                    help="Also log internal RM scalars (lr_pos/lr_neg/etc.) from compute_loss on rank 0.")
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main = (rank == 0)

    if args.precompute_ref_logps and args.pretokenize_num_proc <= 0:
        raise ValueError("--precompute_ref_logps requires --pretokenize_num_proc > 0.")

    if torch.cuda.is_available():
        # Ensure each distributed process binds to its own GPU.
        torch.cuda.set_device(local_rank)

    # ---- dataset ----
    ds = load_dataset("json", data_files=args.paired_jsonl, split="train")

    # ---- tokenizer ----
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    # ---- multiprocess pretokenization + optional distributed ref-logp precompute ----
    if args.pretokenize_num_proc > 0:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.makedirs(args.output_dir, exist_ok=True)
        cache_dir = args.pretokenize_cache_dir.strip() or os.path.join(args.output_dir, "tokenized_train_ds")
        ready_file = cache_dir + ".ready"
        meta_file = cache_dir + ".meta.json"
        base_cache_dir = cache_dir + ".base"
        base_ready_file = base_cache_dir + ".ready"
        rank_shard_file = f"{cache_dir}.ref_logps.rank{rank}.pt"
        rank_done_file = f"{cache_dir}.ref_logps.rank{rank}.done"
        expected_cache_meta = {
            "model_name_or_path": args.model_name_or_path,
            "ref_model_name_or_path": args.ref_model_name_or_path,
            "tokenizer_len": len(tok),
            "max_length": int(args.max_length),
            "use_answer_only_pos": bool(args.use_answer_only_pos),
            "append_final_answer": bool(args.append_final_answer),
            "final_answer_prefix": str(args.final_answer_prefix),
            "strip_solution_to_answer": bool(args.strip_solution_to_answer),
            "precompute_ref_logps": bool(args.precompute_ref_logps),
            "attn_implementation": str(args.attn_implementation),
            "bf16": bool(args.bf16),
            "fp16": bool(args.fp16),
        }

        def _cache_meta_matches() -> bool:
            if not os.path.isfile(meta_file):
                return False
            try:
                with open(meta_file, "r", encoding="utf-8") as rf:
                    meta = json.load(rf)
            except Exception:
                return False
            for k, v in expected_cache_meta.items():
                if meta.get(k) != v:
                    return False
            return True

        def _final_cache_ready() -> bool:
            return is_valid_saved_dataset_dir(cache_dir) and os.path.exists(ready_file) and _cache_meta_matches()

        def _wait_for(check_fn, wait_desc: str) -> None:
            start = time.time()
            announced = False
            while not check_fn():
                if not announced:
                    print(f"[pretokenize][rank{rank}] Waiting for {wait_desc}", flush=True)
                    announced = True
                if time.time() - start > args.pretokenize_wait_timeout_sec:
                    raise TimeoutError(
                        f"Timed out waiting for {wait_desc} for "
                        f"{args.pretokenize_wait_timeout_sec}s"
                    )
                time.sleep(5)

        def _pretokenize_batch(batch: Dict[str, List[Any]]) -> Dict[str, List[List[int]]]:
            prompts = batch["prompt"]
            pairs = batch["responses"]
            answers = batch.get("true_solution")
            if answers is None:
                answers = [None] * len(prompts)

            out: Dict[str, List[List[int]]] = {
                "pos_input_ids": [],
                "pos_attention_mask": [],
                "pos_labels": [],
                "neg_input_ids": [],
                "neg_attention_mask": [],
                "neg_labels": [],
            }

            for q, pair, answer in zip(prompts, pairs, answers):
                pair = pair if isinstance(pair, (list, tuple)) else [pair]
                neg_text = str(pair[0]) if len(pair) > 0 and pair[0] is not None else ""
                solution = str(pair[1]) if len(pair) > 1 and pair[1] is not None else ""

                prompt = format_prompt(tok, q)
                pos_text = build_pos_text(
                    solution=solution,
                    answer=answer,
                    use_answer_only_pos=args.use_answer_only_pos,
                    append_final_answer=args.append_final_answer,
                    final_answer_prefix=args.final_answer_prefix,
                    strip_solution_to_answer=args.strip_solution_to_answer,
                )

                pos_enc = encode_prompt_response(tok, prompt, pos_text, args.max_length)
                neg_enc = encode_prompt_response(tok, prompt, neg_text, args.max_length)

                out["pos_input_ids"].append(pos_enc["input_ids"])
                out["pos_attention_mask"].append(pos_enc["attention_mask"])
                out["pos_labels"].append(pos_enc["labels"])
                out["neg_input_ids"].append(neg_enc["input_ids"])
                out["neg_attention_mask"].append(neg_enc["attention_mask"])
                out["neg_labels"].append(neg_enc["labels"])

            return out

        need_build = False
        if is_main:
            need_build = True
            if args.reuse_pretokenized_cache and _final_cache_ready():
                print(f"[pretokenize] Reusing cache: {cache_dir}", flush=True)
                need_build = False
            elif args.reuse_pretokenized_cache and is_valid_saved_dataset_dir(cache_dir):
                print(
                    f"[pretokenize] Cache metadata mismatch, rebuilding: {cache_dir}",
                    flush=True,
                )

            if need_build:
                if os.path.exists(ready_file):
                    os.remove(ready_file)
                if os.path.exists(meta_file):
                    os.remove(meta_file)
                if os.path.isdir(cache_dir):
                    shutil.rmtree(cache_dir)
                if os.path.exists(base_ready_file):
                    os.remove(base_ready_file)
                if os.path.isdir(base_cache_dir):
                    shutil.rmtree(base_cache_dir)
                for r in range(world_size):
                    shard_path = f"{cache_dir}.ref_logps.rank{r}.pt"
                    done_path = f"{cache_dir}.ref_logps.rank{r}.done"
                    if os.path.exists(shard_path):
                        os.remove(shard_path)
                    if os.path.exists(done_path):
                        os.remove(done_path)
                print(
                    f"[pretokenize] Tokenizing with num_proc={args.pretokenize_num_proc}, "
                    f"batch_size={args.pretokenize_batch_size} ...",
                    flush=True,
                )
                ds_tok = ds.map(
                    _pretokenize_batch,
                    batched=True,
                    batch_size=args.pretokenize_batch_size,
                    num_proc=args.pretokenize_num_proc,
                    remove_columns=ds.column_names,
                    desc="Pretokenizing pairwise RM dataset",
                )
                if args.precompute_ref_logps:
                    ds_tok.save_to_disk(base_cache_dir)
                    with open(base_ready_file, "w", encoding="utf-8") as wf:
                        wf.write("ready\n")
                    print(f"[precompute_ref] Saved tokenized base cache to {base_cache_dir}", flush=True)
                else:
                    ds_tok.save_to_disk(cache_dir)
                    print(f"[pretokenize] Saved cache to {cache_dir}", flush=True)
                    with open(meta_file, "w", encoding="utf-8") as wf:
                        json.dump(expected_cache_meta, wf, ensure_ascii=True, sort_keys=True)
                    with open(ready_file, "w", encoding="utf-8") as wf:
                        wf.write("ready\n")

        if args.precompute_ref_logps and not _final_cache_ready():
            _wait_for(
                lambda: is_valid_saved_dataset_dir(base_cache_dir) and os.path.exists(base_ready_file),
                f"tokenized base cache from rank0: {base_cache_dir}",
            )
            ds_base = load_from_disk(base_cache_dir)
            shard_indices = list(range(rank, len(ds_base), max(1, world_size)))
            if is_main:
                print(
                    f"[precompute_ref] Distributed precompute over {world_size} ranks, "
                    f"batch_size={args.precompute_ref_batch_size}",
                    flush=True,
                )
            print(
                f"[precompute_ref][rank{rank}] shard_size={len(shard_indices)}",
                flush=True,
            )

            shard_payload: Dict[str, Any] = {
                "indices": shard_indices,
                "logp_ref_pos": [],
                "logp_ref_neg": [],
            }
            if shard_indices:
                ds_shard = ds_base.select(shard_indices)

                precompute_kwargs: Dict[str, Any] = {"attn_implementation": args.attn_implementation}
                if args.bf16:
                    precompute_kwargs["torch_dtype"] = torch.bfloat16
                elif args.fp16:
                    precompute_kwargs["torch_dtype"] = torch.float16

                ref_precompute_model = AutoModelForCausalLM.from_pretrained(
                    args.ref_model_name_or_path,
                    **precompute_kwargs,
                )
                ref_vocab = ref_precompute_model.get_input_embeddings().weight.shape[0]
                tok_len = len(tok)
                if ref_vocab < tok_len:
                    print(
                        f"[precompute_ref][rank{rank}] Resizing ref embeddings: {ref_vocab} -> {tok_len}",
                        flush=True,
                    )
                    ref_precompute_model.resize_token_embeddings(tok_len)

                ref_precompute_model.config.use_cache = False
                if torch.cuda.is_available():
                    precompute_device = torch.device(f"cuda:{local_rank}")
                else:
                    precompute_device = torch.device("cpu")
                ref_precompute_model.to(precompute_device)

                ds_shard = precompute_ref_logps_for_dataset(
                    dataset=ds_shard,
                    ref_model=ref_precompute_model,
                    device=precompute_device,
                    pad_token_id=tok.pad_token_id,
                    batch_size=max(1, int(args.precompute_ref_batch_size)),
                    is_main=is_main,
                )
                # Persist only plain Python types so torch.load(weights_only=True) works on PyTorch>=2.6.
                shard_payload["logp_ref_pos"] = [float(v) for v in ds_shard["logp_ref_pos"]]
                shard_payload["logp_ref_neg"] = [float(v) for v in ds_shard["logp_ref_neg"]]
                del ref_precompute_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            torch.save(shard_payload, rank_shard_file)
            with open(rank_done_file, "w", encoding="utf-8") as wf:
                wf.write("done\n")

            if is_main:
                _wait_for(
                    lambda: all(
                        os.path.exists(f"{cache_dir}.ref_logps.rank{r}.done")
                        for r in range(world_size)
                    ),
                    f"all precompute shards ({world_size} ranks)",
                )

                ds_final = load_from_disk(base_cache_dir)
                n = len(ds_final)
                merged_pos: List[Optional[float]] = [None] * n
                merged_neg: List[Optional[float]] = [None] * n
                for r in range(world_size):
                    shard_path = f"{cache_dir}.ref_logps.rank{r}.pt"
                    try:
                        shard_data = torch.load(shard_path, map_location="cpu")
                    except pickle.UnpicklingError:
                        # Backward compatibility with shard files saved before plain-list conversion.
                        shard_data = torch.load(shard_path, map_location="cpu", weights_only=False)
                    idxs = shard_data.get("indices", [])
                    pos_vals = shard_data.get("logp_ref_pos", [])
                    neg_vals = shard_data.get("logp_ref_neg", [])
                    if not (len(idxs) == len(pos_vals) == len(neg_vals)):
                        raise ValueError(
                            f"Invalid shard payload from rank {r}: "
                            f"len(indices)={len(idxs)}, len(pos)={len(pos_vals)}, len(neg)={len(neg_vals)}"
                        )
                    for i, p, ng in zip(idxs, pos_vals, neg_vals):
                        merged_pos[i] = float(p)
                        merged_neg[i] = float(ng)

                if any(v is None for v in merged_pos) or any(v is None for v in merged_neg):
                    raise ValueError("Incomplete merged precomputed ref log-probs across ranks.")

                if "logp_ref_pos" in ds_final.column_names:
                    ds_final = ds_final.remove_columns("logp_ref_pos")
                if "logp_ref_neg" in ds_final.column_names:
                    ds_final = ds_final.remove_columns("logp_ref_neg")
                ds_final = ds_final.add_column("logp_ref_pos", [float(v) for v in merged_pos])
                ds_final = ds_final.add_column("logp_ref_neg", [float(v) for v in merged_neg])

                ds_final.save_to_disk(cache_dir)
                print(f"[precompute_ref] Saved merged cache to {cache_dir}", flush=True)
                with open(meta_file, "w", encoding="utf-8") as wf:
                    json.dump(expected_cache_meta, wf, ensure_ascii=True, sort_keys=True)
                with open(ready_file, "w", encoding="utf-8") as wf:
                    wf.write("ready\n")

                if os.path.exists(base_ready_file):
                    os.remove(base_ready_file)
                if os.path.isdir(base_cache_dir):
                    shutil.rmtree(base_cache_dir)
                for r in range(world_size):
                    shard_path = f"{cache_dir}.ref_logps.rank{r}.pt"
                    done_path = f"{cache_dir}.ref_logps.rank{r}.done"
                    if os.path.exists(shard_path):
                        os.remove(shard_path)
                    if os.path.exists(done_path):
                        os.remove(done_path)

        if not _final_cache_ready():
            _wait_for(
                _final_cache_ready,
                f"final pretokenized cache from rank0: {cache_dir}",
            )

        distributed_barrier()
        ds = load_from_disk(cache_dir)
        if args.precompute_ref_logps:
            needed_cols = {"logp_ref_pos", "logp_ref_neg"}
            missing_cols = needed_cols.difference(set(ds.column_names))
            if missing_cols:
                raise ValueError(
                    f"Precomputed cache missing columns: {sorted(missing_cols)}. "
                    "Delete cache or disable --reuse_pretokenized_cache to rebuild."
                )
        distributed_barrier()

    # ---- models ----
    model_kwargs: Dict[str, Any] = {"attn_implementation": args.attn_implementation}
    if args.bf16:
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif args.fp16:
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    ref_model = None
    if not args.precompute_ref_logps:
        ref_model = AutoModelForCausalLM.from_pretrained(args.ref_model_name_or_path, **model_kwargs)

    tok_len = len(tok)
    policy_vocab = model.get_input_embeddings().weight.shape[0]
    if policy_vocab < tok_len:
        if is_main:
            print(f"[model] Resizing policy embeddings: {policy_vocab} -> {tok_len}", flush=True)
        model.resize_token_embeddings(tok_len)
    if ref_model is not None:
        ref_vocab = ref_model.get_input_embeddings().weight.shape[0]
        if ref_vocab < tok_len:
            if is_main:
                print(f"[model] Resizing ref embeddings: {ref_vocab} -> {tok_len}", flush=True)
            ref_model.resize_token_embeddings(tok_len)

    # Keep model parameter dtypes consistent after possible embedding resize.
    if args.bf16:
        model.to(dtype=torch.bfloat16)
        if ref_model is not None:
            ref_model.to(dtype=torch.bfloat16)
    elif args.fp16:
        model.to(dtype=torch.float16)
        if ref_model is not None:
            ref_model.to(dtype=torch.float16)

    model.gradient_checkpointing_enable()

    # Recommended for checkpointing + DDP
    model.config.use_cache = False
    if ref_model is not None:
        ref_model.config.use_cache = False

    # ---- collator ----
    collator = PairwiseCollatorNew(
        tokenizer=tok,
        max_length=args.max_length,
        use_answer_only_pos=args.use_answer_only_pos,
        append_final_answer=args.append_final_answer,
        final_answer_prefix=args.final_answer_prefix,
        strip_solution_to_answer=args.strip_solution_to_answer,
    )

    # ---- wandb init (rank 0 only) ----

    if args.wandb_mode == "disabled":
        os.environ["WANDB_DISABLED"] = "true"
    else:
        os.environ.pop("WANDB_DISABLED", None)
        os.environ["WANDB_MODE"] = args.wandb_mode

    run_name = args.wandb_name.strip() if args.wandb_name.strip() else os.path.basename(args.output_dir.rstrip("/"))

    if is_main and args.wandb_mode != "disabled":
        wandb_kwargs = dict(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
        )
        if args.wandb_entity.strip():
            wandb_kwargs["entity"] = args.wandb_entity.strip()
        if args.wandb_group.strip():
            wandb_kwargs["group"] = args.wandb_group.strip()
        if args.wandb_tags.strip():
            wandb_kwargs["tags"] = [t for t in args.wandb_tags.split(",") if t.strip()]
        wandb.init(**wandb_kwargs)

    # ---- training args ----
    report_to = ["wandb"] if args.wandb_mode != "disabled" else ["none"]

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to=report_to,
        run_name=run_name,
        seed=args.seed,
        bf16=args.bf16,
        fp16=args.fp16,
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
    )

    # NOTE (NEW TRL): do NOT pass tokenizer=...
    # Use processing_class=tok if you want TRL to know how to process; with custom collator it’s fine either way.
    trainer = PairwiseRatioMSETrainer(
        model=model,
        ref_model=ref_model,
        use_precomputed_ref_logps=args.precompute_ref_logps,
        beta=args.beta,
        r_max=args.r_max,
        r_min=args.r_min,
        logratio_clip=args.logratio_clip,
        reg_coef=args.reg_coef,
        length_normalize=args.length_normalize,
        wandb_log_internal=args.wandb_log_internal,
        args=targs,
        train_dataset=ds,
        data_collator=collator,
        processing_class=tok,   # <- new TRL
        # DO NOT pass dataset_text_field / formatting_func
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)

    if is_main and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
