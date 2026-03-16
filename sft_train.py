#!/usr/bin/env python3

import argparse
import json
import os
import re
from contextlib import nullcontext
from typing import Any, Dict, List, Tuple

import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None

try:
    from trl.models.utils import unwrap_model_for_generation
except Exception:  # pragma: no cover
    unwrap_model_for_generation = None


ANSWER_PATTERN = re.compile(r"(?:final answer|answer)\s*[:：]\s*(.+)$", flags=re.IGNORECASE)
BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]+)\}")

SYSTEM_PROMPT = "You are a helpful assistant."
USER_PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. The last line of your response "
    "should be of the form Answer: $Answer (without quotes) where $Answer is the "
    "answer to the problem.\n\n"
    "{problem}\n\n"
    "Remember to put your answer on its own line after \"Answer:\"."
)


def normalize_prompt(prompt_obj: Any) -> str:
    if isinstance(prompt_obj, str):
        return prompt_obj
    if isinstance(prompt_obj, (list, tuple)):
        parts: List[str] = []
        for item in prompt_obj:
            if isinstance(item, dict) and item.get("role") == "user":
                parts.append(str(item.get("content", "")))
            else:
                parts.append(normalize_prompt(item))
        return "\n".join(x for x in parts if x is not None)
    if isinstance(prompt_obj, dict):
        for key in ["prompt", "question", "content", "text"]:
            if key in prompt_obj:
                return normalize_prompt(prompt_obj[key])
        return json.dumps(prompt_obj, ensure_ascii=False)
    return str(prompt_obj)


def build_prompt_text(tokenizer: Any, prompt_obj: Any) -> str:
    problem = normalize_prompt(prompt_obj)
    user_prompt = USER_PROMPT_TEMPLATE.format(problem=problem)
    has_chat = hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None)
    if has_chat:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return f"System: {SYSTEM_PROMPT}\nUser: {user_prompt}\nAssistant:"


def encode_prompt_response(
    tokenizer: Any,
    prompt_text: str,
    response_text: str,
    max_length: int,
) -> Dict[str, List[int]]:
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    response_ids = tokenizer(response_text, add_special_tokens=False).input_ids
    if tokenizer.eos_token_id is not None:
        response_ids = response_ids + [tokenizer.eos_token_id]

    input_ids = prompt_ids + response_ids
    overflow = max(0, len(input_ids) - max_length)
    if overflow > 0:
        input_ids = input_ids[overflow:]

    kept_prompt_len = max(0, len(prompt_ids) - overflow)
    kept_prompt_len = min(kept_prompt_len, len(input_ids))
    labels = [-100] * kept_prompt_len + input_ids[kept_prompt_len:]
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def normalize_answer_text(text: str) -> str:
    x = str(text).strip()
    x = x.replace("$", "")
    x = x.replace("\\(", "").replace("\\)", "")
    x = x.replace("\\[", "").replace("\\]", "")
    x = x.replace(",", "")
    x = x.replace(" ", "")
    x = x.lower()
    x = re.sub(r"[.;:]+$", "", x)
    return x


def extract_final_answer(text: str) -> str:
    if not text:
        return ""

    boxed = BOXED_PATTERN.findall(text)
    if boxed:
        return boxed[-1].strip()

    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    for ln in reversed(lines):
        match = ANSWER_PATTERN.search(ln)
        if match:
            return match.group(1).strip()

    if lines:
        return lines[-1]
    return text.strip()


def extract_user_content_from_chat(prompt: Any) -> str:
    if not isinstance(prompt, list):
        return str(prompt)
    for msg in prompt:
        if isinstance(msg, dict) and msg.get("role") == "user":
            return str(msg.get("content", ""))
    if prompt and isinstance(prompt[0], dict):
        return str(prompt[0].get("content", ""))
    return str(prompt)


def extract_eval_prompt(example: Dict[str, Any]) -> str:
    if "prompt" in example and example["prompt"] is not None:
        val = example["prompt"]
        if isinstance(val, list):
            return extract_user_content_from_chat(val).strip()
        return str(val).strip()

    for key in ["problem", "question", "input", "text"]:
        if key in example and example[key] is not None:
            val = example[key]
            if isinstance(val, list):
                return extract_user_content_from_chat(val).strip()
            return str(val).strip()
    return ""


def extract_eval_ground_truth(example: Dict[str, Any]) -> str:
    for key in ["answer", "solution", "true_solution", "ground_truth"]:
        if key in example and example[key] is not None:
            val = str(example[key]).strip()
            if val:
                return val
    return ""


def format_prompt_with_system(tokenizer: Any, user_prompt: str, system_prompt: str) -> str:
    has_chat = hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None)
    if has_chat:
        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if system_prompt.strip():
        return f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"
    return f"User: {user_prompt}\nAssistant:"


def trim_generated_tokens(
    token_ids: torch.Tensor,
    eos_token_id: int,
    pad_token_id: int,
) -> torch.Tensor:
    if token_ids.numel() == 0:
        return token_ids

    end = int(token_ids.shape[0])
    if eos_token_id is not None:
        eos_pos = (token_ids == eos_token_id).nonzero(as_tuple=False)
        if eos_pos.numel() > 0:
            end = min(end, int(eos_pos[0].item()))
    if pad_token_id is not None:
        pad_pos = (token_ids == pad_token_id).nonzero(as_tuple=False)
        if pad_pos.numel() > 0:
            end = min(end, int(pad_pos[0].item()))
    return token_ids[:end]


def get_rank_world_size() -> Tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank()), int(dist.get_world_size())
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, max(1, world_size)


def get_module_device(module: Any, fallback: torch.device) -> torch.device:
    if module is None:
        return fallback
    try:
        for p in module.parameters():
            return p.device
    except Exception:
        pass
    try:
        for b in module.buffers():
            return b.device
    except Exception:
        pass
    return fallback


@torch.no_grad()
def evaluate_math500_passk(
    model: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    trainer: Any = None,
    phase: str = "initial",
) -> Dict[str, float]:
    if args.skip_math500_eval:
        return {}

    rank, world_size = get_rank_world_size()
    is_main = rank == 0

    try:
        if args.math500_eval_dataset_config.strip():
            eval_ds = load_dataset(
                args.math500_eval_dataset_name,
                args.math500_eval_dataset_config,
                split=args.math500_eval_split,
            )
        else:
            eval_ds = load_dataset(args.math500_eval_dataset_name, split=args.math500_eval_split)
    except Exception as exc:
        if is_main:
            print(f"[math500] Failed to load dataset: {exc}", flush=True)
        return {}

    if args.math500_eval_max_samples > 0:
        keep = min(int(args.math500_eval_max_samples), len(eval_ds))
        eval_ds = eval_ds.select(range(keep))

    local_indices = list(range(rank, len(eval_ds), world_size))
    if is_main:
        print(
            f"[math500] Running pass@{args.math500_pass_k} ({phase}) on "
            f"{len(eval_ds)} examples across {world_size} ranks.",
            flush=True,
        )

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("Math500 evaluation requires CUDA. No GPU is available.")
    metric_device = torch.device(f"cuda:{local_rank}")
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0

    do_sample = bool(args.math500_eval_do_sample)
    k = max(1, int(args.math500_pass_k))
    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": int(args.math500_eval_max_new_tokens),
        "do_sample": do_sample,
        "num_return_sequences": k,
        "pad_token_id": int(pad_token_id),
        "eos_token_id": tokenizer.eos_token_id,
        "synced_gpus": world_size > 1,
    }
    if do_sample:
        gen_kwargs["temperature"] = float(args.math500_eval_temperature)
        gen_kwargs["top_p"] = float(args.math500_eval_top_p)

    use_unwrap = (
        unwrap_model_for_generation is not None
        and trainer is not None
        and hasattr(trainer, "accelerator")
    )
    if use_unwrap:
        model_context = unwrap_model_for_generation(
            model,
            trainer.accelerator,
            gather_deepspeed3_params=True,
            generation_kwargs=gen_kwargs,
        )
    else:
        model_context = nullcontext(model)

    local_total = 0.0
    local_correct = 0.0
    batch_size = max(1, int(args.math500_eval_batch_size))
    was_training = model.training if hasattr(model, "training") else False
    model.eval()

    with model_context as gen_model:
        model_device = get_module_device(gen_model, fallback=metric_device)
        if model_device != metric_device:
            if is_main:
                print(
                    f"[math500] Moving eval model from {model_device} to {metric_device}.",
                    flush=True,
                )
            gen_model.to(metric_device)
            model_device = get_module_device(gen_model, fallback=metric_device)
        if model_device.type != "cuda":
            raise RuntimeError(f"Eval model must be on CUDA, got {model_device}.")

        for start in range(0, len(local_indices), batch_size):
            idxs = local_indices[start : start + batch_size]
            prompts: List[str] = []
            truths: List[str] = []
            for ds_idx in idxs:
                ex = eval_ds[int(ds_idx)]
                question = extract_eval_prompt(ex)
                if args.math500_eval_append_answer_instruction:
                    question = (
                        f"{question.rstrip()}\n\n{args.math500_eval_answer_instruction_text}"
                    )
                prompts.append(
                    format_prompt_with_system(
                        tokenizer=tokenizer,
                        user_prompt=question,
                        system_prompt=args.math500_eval_system_prompt,
                    )
                )
                truths.append(extract_eval_ground_truth(ex))

            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(args.math500_eval_max_prompt_length),
                add_special_tokens=False,
            )
            input_ids = enc["input_ids"].to(model_device, non_blocking=True)
            attention_mask = enc["attention_mask"].to(model_device, non_blocking=True)
            context_lengths = attention_mask.sum(dim=1).tolist()

            outputs = gen_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

            bsz = len(idxs)
            for i in range(bsz):
                gt_norm = normalize_answer_text(truths[i])
                passed = False
                for sample_idx in range(k):
                    row = outputs[i * k + sample_idx]
                    ctx_len = int(context_lengths[i])
                    response = row[ctx_len:]
                    response = trim_generated_tokens(
                        response,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=pad_token_id,
                    )
                    decoded = tokenizer.decode(response, skip_special_tokens=True).strip()
                    pred = extract_final_answer(decoded)
                    pred_norm = normalize_answer_text(pred)
                    if gt_norm and pred_norm == gt_norm:
                        passed = True
                        break

                local_total += 1.0
                if passed:
                    local_correct += 1.0

    if was_training:
        model.train()

    correct_t = torch.tensor(local_correct, dtype=torch.float32, device=metric_device)
    total_t = torch.tensor(local_total, dtype=torch.float32, device=metric_device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(correct_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_t, op=dist.ReduceOp.SUM)

    global_correct = float(correct_t.item())
    global_total = float(total_t.item())
    metric_name = f"eval/math500_pass@{k}_{phase}"
    total_name = f"eval/math500_total_{phase}"
    pass_k = (global_correct / max(1.0, global_total)) if global_total > 0 else 0.0
    metrics = {metric_name: pass_k, total_name: global_total}

    if is_main:
        print(
            f"[math500] {phase} pass@{k}={pass_k:.4f} ({int(global_correct)}/{int(global_total)})",
            flush=True,
        )
        if wandb is not None and wandb.run is not None:
            wandb.log(metrics)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return metrics


def make_preprocess_fn(tokenizer: Any, max_length: int):
    def _preprocess_batch(batch: Dict[str, List[Any]]) -> Dict[str, List[List[int]]]:
        prompts = batch.get("prompt", [])
        solutions = batch.get("solution")
        if solutions is None:
            solutions = [""] * len(prompts)

        out: Dict[str, List[List[int]]] = {"input_ids": [], "attention_mask": [], "labels": []}
        for prompt_obj, solution in zip(prompts, solutions):
            prompt_text = build_prompt_text(tokenizer, prompt_obj)
            response_text = "" if solution is None else str(solution)
            encoded = encode_prompt_response(
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                response_text=response_text,
                max_length=max_length,
            )
            out["input_ids"].append(encoded["input_ids"])
            out["attention_mask"].append(encoded["attention_mask"])
            out["labels"].append(encoded["labels"])
        return out

    return _preprocess_batch


class SFTDataCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
        return {
            "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id),
            "attention_mask": pad_sequence(attention_mask, batch_first=True, padding_value=0),
            "labels": pad_sequence(labels, batch_first=True, padding_value=-100),
        }


class TokenizedSFTTrainer(SFTTrainer):
    def _prepare_dataset(self, dataset, *args, **kwargs):
        return dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--model_name_or_path", required=True)
    ap.add_argument("--output_dir", required=True)

    ap.add_argument("--max_length", type=int, default=9216)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--dataloader_num_workers", type=int, default=4)
    ap.add_argument("--learning_rate", type=float, default=2.0e-5)
    ap.add_argument("--num_train_epochs", type=float, default=3.0)
    ap.add_argument("--warmup_steps", type=int, default=100)
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

    ap.add_argument(
        "--dataset_num_proc",
        type=int,
        default=max(1, min(16, os.cpu_count() or 1)),
        help="Number of processes used by datasets.map for tokenization.",
    )
    ap.add_argument(
        "--dataset_map_batch_size",
        type=int,
        default=64,
        help="Batch size used during dataset.map tokenization.",
    )

    ap.add_argument("--skip_math500_eval", action="store_true")
    ap.add_argument("--math500_eval_dataset_name", type=str, default="HuggingFaceH4/MATH-500")
    ap.add_argument("--math500_eval_dataset_config", type=str, default="")
    ap.add_argument("--math500_eval_split", type=str, default="test")
    ap.add_argument("--math500_eval_max_samples", type=int, default=500)
    ap.add_argument("--math500_eval_batch_size", type=int, default=1)
    ap.add_argument("--math500_pass_k", type=int, default=2)
    ap.add_argument("--math500_eval_max_prompt_length", type=int, default=2048)
    ap.add_argument("--math500_eval_max_new_tokens", type=int, default=4096)
    ap.add_argument("--math500_eval_temperature", type=float, default=0.7)
    ap.add_argument("--math500_eval_top_p", type=float, default=0.95)
    ap.add_argument("--math500_eval_do_sample", action="store_true")
    ap.add_argument("--math500_eval_greedy", dest="math500_eval_do_sample", action="store_false")
    ap.set_defaults(math500_eval_do_sample=True)
    ap.add_argument(
        "--math500_eval_system_prompt",
        type=str,
        default="You are a helpful assistant. Solve the math problem carefully.",
    )
    ap.add_argument("--math500_eval_append_answer_instruction", action="store_true")
    ap.add_argument(
        "--no_math500_eval_append_answer_instruction",
        dest="math500_eval_append_answer_instruction",
        action="store_false",
    )
    ap.set_defaults(math500_eval_append_answer_instruction=True)
    ap.add_argument(
        "--math500_eval_answer_instruction_text",
        type=str,
        default="Give your final answer on the last line as: Answer: <final answer>",
    )

    ap.add_argument("--wandb_project", type=str, default="sft-dapo")
    ap.add_argument("--wandb_name", type=str, default="")
    ap.add_argument("--wandb_group", type=str, default="")
    ap.add_argument("--wandb_tags", type=str, default="")
    ap.add_argument("--wandb_entity", type=str, default="")
    ap.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main = rank == 0

    if not torch.cuda.is_available():
        raise RuntimeError("This script is configured for GPU-only execution, but CUDA is not available.")
    torch.cuda.set_device(local_rank)
    local_device = torch.device(f"cuda:{local_rank}")

    if args.wandb_mode != "disabled" and wandb is None:
        raise RuntimeError("wandb is not available but wandb logging is enabled.")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.wandb_mode == "disabled":
        os.environ["WANDB_DISABLED"] = "true"
    else:
        os.environ.pop("WANDB_DISABLED", None)
        os.environ["WANDB_MODE"] = args.wandb_mode

    ds = load_dataset("json", data_files=args.train_jsonl, split="train")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    preprocess_fn = make_preprocess_fn(tokenizer=tokenizer, max_length=args.max_length)
    ds = ds.map(
        preprocess_fn,
        batched=True,
        batch_size=args.dataset_map_batch_size,
        num_proc=max(1, int(args.dataset_num_proc)),
        remove_columns=ds.column_names,
        desc="Tokenizing SFT dataset",
    )

    model_kwargs: Dict[str, Any] = {"attn_implementation": args.attn_implementation}
    if args.bf16:
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif args.fp16:
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    vocab_size = model.get_input_embeddings().weight.shape[0]
    tokenizer_len = len(tokenizer)
    if vocab_size < tokenizer_len:
        if is_main:
            print(f"[model] Resizing embeddings: {vocab_size} -> {tokenizer_len}", flush=True)
        model.resize_token_embeddings(tokenizer_len)

    if args.bf16:
        model.to(device=local_device, dtype=torch.bfloat16)
    elif args.fp16:
        model.to(device=local_device, dtype=torch.float16)
    else:
        model.to(device=local_device)

    model.gradient_checkpointing_enable()
    model.config.use_cache = False

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

    report_to = ["wandb"] if args.wandb_mode != "disabled" else ["none"]
    training_args = TrainingArguments(
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

    trainer = TokenizedSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=SFTDataCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
    )

    trainer.train()
    evaluate_math500_passk(
        model=trainer.model,
        tokenizer=tokenizer,
        args=args,
        trainer=trainer,
        phase="final",
    )
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if is_main and wandb is not None and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
