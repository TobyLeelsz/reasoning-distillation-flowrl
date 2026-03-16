# Copyright 2026 Individual Contributor
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import threading
from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from verl.utils.reward_score import default_compute_score

PROMPT_TOKEN_IDS_KEY = "__verl_prompt_token_ids"
RESPONSE_TOKEN_IDS_KEY = "__verl_response_token_ids"

_SCORER = None
_SCORER_CFG = None
_SCORER_LOCK = threading.Lock()
_VERBOSE = os.environ.get("VERL_LOG_RATIO_REWARD_VERBOSE", "0") == "1"
_DEFAULT_MAX_SEQ_LEN = 8192


def _vprint(msg: str):
    if _VERBOSE:
        print(f"[log_ratio_reward] {msg}", flush=True)


def _to_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    if dtype_name is None:
        return torch.bfloat16
    key = str(dtype_name).lower()
    if key not in mapping:
        raise ValueError(f"Unsupported torch dtype '{dtype_name}'.")
    return mapping[key]


def _sequence_logprob_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mirror chi_squared_rm.py sequence logprob computation."""
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]

    loss_mask = labels != -100
    safe_labels = labels.masked_fill(~loss_mask, 0)

    log_probs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    per_token_logps = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    loss_mask = loss_mask.to(per_token_logps.dtype)
    return (per_token_logps * loss_mask).sum(-1)


def _encode_prompt_response_ids(
    prompt_ids: list[int],
    response_ids: list[int],
    eos_token_id: Optional[int],
    max_length: Optional[int],
) -> dict:
    """Mirror chi_squared_rm.py encode_prompt_response semantics for token IDs."""
    p_ids = list(prompt_ids)
    r_ids = list(response_ids)
    # Match text->tokenize->append EOS behavior used in test_log_ratio_reward_pairs.py:
    # keep exactly one terminal EOS regardless of source tokenization path.
    if eos_token_id is not None:
        while len(r_ids) > 0 and r_ids[-1] == eos_token_id:
            r_ids.pop()
        r_ids = r_ids + [eos_token_id]

    input_ids = p_ids + r_ids
    if max_length is not None and max_length > 0 and len(input_ids) > max_length:
        input_ids = input_ids[-max_length:]

    attention_mask = [1] * len(input_ids)
    p_len = min(len(p_ids), len(input_ids))
    labels = [-100] * p_len + input_ids[p_len:]
    response_token_len = sum(1 for x in labels if x != -100)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "response_token_len": int(response_token_len),
        "response_token_ids": input_ids[p_len:],
    }


def _extract_scalar_score(score_obj) -> float:
    if isinstance(score_obj, dict):
        score_obj = score_obj.get("score", 0.0)
    try:
        score = float(score_obj)
    except (TypeError, ValueError):
        score = 0.0
    if not math.isfinite(score):
        score = 0.0
    return score


class LogRatioRewardScorer:
    def __init__(
        self,
        policy_model_path: str,
        reference_model_path: str,
        tokenizer_path: Optional[str],
        beta: float,
        micro_batch_size: int,
        device_map: str,
        torch_dtype: str,
        offload_folder: Optional[str],
        max_gpu_memory: Optional[str],
        max_cpu_memory: Optional[str],
        trust_remote_code: bool,
        use_fast_tokenizer: bool,
        attn_implementation: str,
        max_seq_len: Optional[int],
        normalize_by_length: bool,
        clear_cuda_cache: bool,
        repeat_penalty_weight: float,
        repeat_penalty_ngram_size: Optional[int],
        repeat_penalty_clip_min: Optional[float],
        log_ratio_reward_clip_min: Optional[float],
        reward_clip_min: Optional[float],
    ):
        from transformers import AutoTokenizer

        self.beta = float(beta)
        self.micro_batch_size = max(1, int(micro_batch_size))
        self.device_map = str(device_map).lower()
        self.torch_dtype = _to_dtype(torch_dtype)
        self.offload_folder = offload_folder
        self.max_gpu_memory = max_gpu_memory
        self.max_cpu_memory = max_cpu_memory
        self.trust_remote_code = trust_remote_code
        self.attn_implementation = str(attn_implementation) if attn_implementation is not None else "eager"
        # Align with chi/test script default max length when config does not override it.
        self.max_seq_len = int(max_seq_len) if max_seq_len is not None else _DEFAULT_MAX_SEQ_LEN
        self.normalize_by_length = bool(normalize_by_length)
        self.clear_cuda_cache = bool(clear_cuda_cache)
        repeat_penalty_weight = float(repeat_penalty_weight) if repeat_penalty_weight is not None else 0.0
        repeat_penalty_ngram_size = int(repeat_penalty_ngram_size) if repeat_penalty_ngram_size is not None else None
        repeat_penalty_clip_min = float(repeat_penalty_clip_min) if repeat_penalty_clip_min is not None else None
        self.log_ratio_reward_clip_min = (
            float(log_ratio_reward_clip_min) if log_ratio_reward_clip_min is not None else None
        )
        self.reward_clip_min = float(reward_clip_min) if reward_clip_min is not None else None
        if repeat_penalty_weight != 0.0 or repeat_penalty_ngram_size is not None or repeat_penalty_clip_min is not None:
            _vprint("repeat penalty kwargs are ignored; final reward uses rule-based reward + log-ratio reward.")

        if self.device_map in {"auto", "cuda"} and not torch.cuda.is_available():
            self.device_map = "cpu"

        if self.device_map == "cpu" and self.torch_dtype == torch.float16:
            self.torch_dtype = torch.float32

        if self.offload_folder:
            os.makedirs(self.offload_folder, exist_ok=True)

        tokenizer_path = tokenizer_path or policy_model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=trust_remote_code,
            use_fast=use_fast_tokenizer,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.pad_token_id

        self.policy_model = self._load_model(policy_model_path)
        self.reference_model = self._load_model(reference_model_path)
        _vprint(
            f"initialized scorer: device_map={self.device_map}, dtype={self.torch_dtype}, "
            f"micro_batch_size={self.micro_batch_size}, max_seq_len={self.max_seq_len}, "
            f"log_ratio_reward_clip_min={self.log_ratio_reward_clip_min}, "
            f"reward_clip_min={self.reward_clip_min}"
        )

    def _build_max_memory(self) -> Optional[dict]:
        if self.device_map != "auto":
            return None
        max_memory = {}
        if torch.cuda.is_available() and self.max_gpu_memory:
            for i in range(torch.cuda.device_count()):
                max_memory[i] = self.max_gpu_memory
        if self.max_cpu_memory:
            max_memory["cpu"] = self.max_cpu_memory
        return max_memory if max_memory else None

    def _load_model(self, model_path: str):
        from transformers import AutoModelForCausalLM

        load_kwargs = {
            "torch_dtype": self.torch_dtype,
            "trust_remote_code": self.trust_remote_code,
            "low_cpu_mem_usage": True,
            "attn_implementation": self.attn_implementation,
        }

        if self.device_map == "cpu":
            load_kwargs["device_map"] = {"": "cpu"}
        elif self.device_map == "auto":
            load_kwargs["device_map"] = "auto"
            if self.offload_folder:
                load_kwargs["offload_folder"] = self.offload_folder
            max_memory = self._build_max_memory()
            if max_memory is not None:
                load_kwargs["max_memory"] = max_memory

        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

        if self.device_map == "cuda":
            model = model.to(torch.device("cuda"))

        # Some transformers versions may ignore constructor kwargs and keep sdpa
        # from config; force it here to avoid sdpa+meta edge cases.
        if hasattr(model, "config"):
            if hasattr(model.config, "_attn_implementation"):
                model.config._attn_implementation = self.attn_implementation
            if hasattr(model.config, "attn_implementation"):
                model.config.attn_implementation = self.attn_implementation

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    def _prepare_encoded_sequences(
        self,
        solution_strs: Iterable[str],
        extra_infos: Optional[Iterable[dict]],
    ) -> list[dict]:
        solution_strs = list(solution_strs)
        extra_infos = list(extra_infos) if extra_infos is not None else [None] * len(solution_strs)
        if len(extra_infos) != len(solution_strs):
            extra_infos = [None] * len(solution_strs)

        encoded_list = []
        for solution, extra in zip(solution_strs, extra_infos):
            if isinstance(extra, dict) and PROMPT_TOKEN_IDS_KEY in extra and RESPONSE_TOKEN_IDS_KEY in extra:
                prompt_ids = list(extra[PROMPT_TOKEN_IDS_KEY])
                response_ids = list(extra[RESPONSE_TOKEN_IDS_KEY])
            else:
                prompt_ids = []
                response_ids = self.tokenizer(solution, add_special_tokens=False).input_ids

            encoded = _encode_prompt_response_ids(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                eos_token_id=self.tokenizer.eos_token_id,
                max_length=self.max_seq_len,
            )
            encoded_list.append(encoded)

        return encoded_list

    def _prepare_batch_tensors(
        self,
        encoded_list: list[dict],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x["input_ids"], dtype=torch.long) for x in encoded_list],
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x["attention_mask"], dtype=torch.long) for x in encoded_list],
            batch_first=True,
            padding_value=0,
        )
        label_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x["labels"], dtype=torch.long) for x in encoded_list],
            batch_first=True,
            padding_value=-100,
        )
        response_lens_tensor = torch.tensor([int(x["response_token_len"]) for x in encoded_list], dtype=torch.long)

        return input_ids, attention_mask, label_ids, response_lens_tensor

    def _sequence_log_probs(self, model, input_ids: torch.Tensor, attention_mask: torch.Tensor, label_ids: torch.Tensor) -> torch.Tensor:
        all_logps = []
        has_hf_device_map = hasattr(model, "hf_device_map")
        model_device = torch.device("cpu")
        if not has_hf_device_map:
            model_device = next(model.parameters()).device

        total = input_ids.size(0)
        _vprint(f"start logprob pass: total_seqs={total}, micro_batch_size={self.micro_batch_size}, hf_device_map={has_hf_device_map}")
        with torch.inference_mode():
            for start in range(0, input_ids.size(0), self.micro_batch_size):
                end = min(start + self.micro_batch_size, input_ids.size(0))
                mb_input_ids = input_ids[start:end]
                mb_attention_mask = attention_mask[start:end]
                mb_label_ids = label_ids[start:end]

                if not has_hf_device_map:
                    mb_input_ids = mb_input_ids.to(model_device, non_blocking=True)
                    mb_attention_mask = mb_attention_mask.to(model_device, non_blocking=True)
                    mb_label_ids = mb_label_ids.to(model_device, non_blocking=True)

                try:
                    logits = model(input_ids=mb_input_ids, attention_mask=mb_attention_mask, use_cache=False).logits
                except RuntimeError as e:
                    msg = str(e)
                    if "meta tensors" in msg.lower() and "tensor.item()" in msg.lower():
                        raise RuntimeError(
                            "Model forward hit a meta-tensor path. "
                            "Try `attn_implementation='eager'` and/or avoid `device_map='auto'`."
                        ) from e
                    raise
                if mb_label_ids.device != logits.device:
                    mb_label_ids = mb_label_ids.to(logits.device, non_blocking=True)
                seq_logps = _sequence_logprob_from_logits(logits, mb_label_ids)
                all_logps.append(seq_logps.detach().cpu())

                del logits, seq_logps
                if self.clear_cuda_cache and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if _VERBOSE and (end == total or end % max(self.micro_batch_size * 64, 1) == 0):
                    _vprint(f"logprob progress: {end}/{total}")

        _vprint("finished logprob pass")
        return torch.cat(all_logps, dim=0)

    def score_batch(self, data_sources, solution_strs, ground_truths, extra_infos=None):
        solution_strs = list(solution_strs)
        data_sources = list(data_sources)
        ground_truths = list(ground_truths)
        extra_infos = list(extra_infos) if extra_infos is not None else [None] * len(solution_strs)
        if len(data_sources) != len(solution_strs) or len(ground_truths) != len(solution_strs):
            raise ValueError("`data_sources`, `solution_strs`, and `ground_truths` must have the same length.")
        if len(extra_infos) != len(solution_strs):
            extra_infos = [None] * len(solution_strs)

        encoded_list = self._prepare_encoded_sequences(solution_strs=solution_strs, extra_infos=extra_infos)
        if _VERBOSE:
            avg_len = 0.0
            if len(encoded_list) > 0:
                avg_len = sum(int(x["response_token_len"]) for x in encoded_list) / float(len(encoded_list))
            _vprint(f"score_batch size={len(encoded_list)}, avg_response_len={avg_len:.1f}")
        input_ids, attention_mask, label_ids, response_lens = self._prepare_batch_tensors(encoded_list=encoded_list)

        policy_logps = self._sequence_log_probs(self.policy_model, input_ids, attention_mask, label_ids)
        ref_logps = self._sequence_log_probs(self.reference_model, input_ids, attention_mask, label_ids)
        log_ratio = policy_logps - ref_logps

        if self.normalize_by_length:
            denom = response_lens.clamp_min(1).to(log_ratio.dtype)
            log_ratio = log_ratio / denom

        rewards = self.beta * log_ratio
        log_ratio_mean = float(log_ratio.mean().item())
        if not math.isfinite(log_ratio_mean):
            log_ratio_mean = 0.0

        results = []
        for reward, lr, p_logp, r_logp, resp_len, data_source, solution_str, ground_truth, extra_info in zip(
            rewards,
            log_ratio,
            policy_logps,
            ref_logps,
            response_lens,
            data_sources,
            solution_strs,
            ground_truths,
            extra_infos,
        ):
            if str(data_source) == "warmup":
                rule_reward_raw = 0.0
            else:
                rule_reward_raw = default_compute_score(
                    data_source=data_source,
                    solution_str=solution_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )
            rule_reward = _extract_scalar_score(rule_reward_raw)

            # New FlowRL log-pi-ratio composition:
            # first_term = -1 when clip(rule_reward, 0, 1) == 0;
            # first_term = clip(log_ratio_reward, min=-1) when clip(rule_reward, 0, 1) == 1.
            rule_reward_gate = max(0.0, min(1.0, rule_reward))
            # if rule_reward_gate == 0.0:
            #     log_ratio_reward = -1.0
            # else:
            #     log_ratio_reward = float(reward.item())
            #     log_ratio_clip_min = -1.0
            #     if self.log_ratio_reward_clip_min is not None:
            #         log_ratio_clip_min = max(log_ratio_clip_min, self.log_ratio_reward_clip_min)
            #     if math.isfinite(log_ratio_reward):
            #         log_ratio_reward = max(log_ratio_reward, log_ratio_clip_min)
            #     else:
            #         log_ratio_reward = log_ratio_clip_min
            log_ratio_reward = float(reward.item())
            log_ratio_clip_min = -1.0
            if self.log_ratio_reward_clip_min is not None:
                log_ratio_clip_min = max(log_ratio_clip_min, self.log_ratio_reward_clip_min)
            if math.isfinite(log_ratio_reward):
                log_ratio_reward = max(log_ratio_reward, log_ratio_clip_min)
            else:
                log_ratio_reward = log_ratio_clip_min
            
            if log_ratio_reward > 0.0 and rule_reward == -1.0:
                # If log_ratio_reward is positive but rule_reward is -1, set log_ratio_reward to 0 to avoid positive reward.
                log_ratio_reward = 0.0

            raw_reward_val = log_ratio_reward + rule_reward
            reward_val = raw_reward_val
            if self.reward_clip_min is not None and math.isfinite(reward_val):
                reward_val = max(reward_val, self.reward_clip_min)
            if not math.isfinite(reward_val):
                reward_val = 0.0
            results.append(
                {
                    "score": reward_val,
                    "log_ratio": float(lr.item()),
                    "log_ratio_mean": log_ratio_mean,
                    "pi_logp": float(p_logp.item()),
                    "pi_ref_logp": float(r_logp.item()),
                    "response_token_len": int(resp_len.item()),
                    "log_ratio_reward": log_ratio_reward,
                    "log_ratio_reward_clip_min": self.log_ratio_reward_clip_min,
                    "rule_reward": rule_reward,
                    "rule_reward_gate": rule_reward_gate,
                    "raw_score_before_clip": raw_reward_val,
                    "reward_clip_min": self.reward_clip_min,
                }
            )

        return results


def _get_scorer(**kwargs):
    global _SCORER, _SCORER_CFG
    cfg_key = tuple((k, kwargs[k]) for k in sorted(kwargs.keys()))
    with _SCORER_LOCK:
        if _SCORER is None or _SCORER_CFG != cfg_key:
            _SCORER = LogRatioRewardScorer(**kwargs)
            _SCORER_CFG = cfg_key
    return _SCORER


def compute_score(
    data_sources,
    solution_strs,
    ground_truths,
    extra_infos=None,
    policy_model_path=None,
    reference_model_path=None,
    tokenizer_path=None,
    beta=0.001,
    micro_batch_size=1,
    device_map="auto",
    torch_dtype="bfloat16",
    offload_folder="/tmp/verl_reward_offload",
    max_gpu_memory="2GiB",
    max_cpu_memory=None,
    trust_remote_code=False,
    use_fast_tokenizer=True,
    attn_implementation="eager",
    max_seq_len=None,
    normalize_by_length=False,
    clear_cuda_cache=False,
    repeat_penalty_weight=0.0,
    repeat_penalty_ngram_size=None,
    repeat_penalty_clip_min=None,
    log_ratio_reward_clip_min=None,
    reward_clip_min=None,
):
    if not policy_model_path:
        raise ValueError("`policy_model_path` must be provided for log-ratio reward.")
    if not reference_model_path:
        raise ValueError("`reference_model_path` must be provided for log-ratio reward.")

    scorer = _get_scorer(
        policy_model_path=policy_model_path,
        reference_model_path=reference_model_path,
        tokenizer_path=tokenizer_path,
        beta=beta,
        micro_batch_size=micro_batch_size,
        device_map=device_map,
        torch_dtype=torch_dtype,
        offload_folder=offload_folder,
        max_gpu_memory=max_gpu_memory,
        max_cpu_memory=max_cpu_memory,
        trust_remote_code=trust_remote_code,
        use_fast_tokenizer=use_fast_tokenizer,
        attn_implementation=attn_implementation,
        max_seq_len=max_seq_len,
        normalize_by_length=normalize_by_length,
        clear_cuda_cache=clear_cuda_cache,
        repeat_penalty_weight=repeat_penalty_weight,
        repeat_penalty_ngram_size=repeat_penalty_ngram_size,
        repeat_penalty_clip_min=repeat_penalty_clip_min,
        log_ratio_reward_clip_min=log_ratio_reward_clip_min,
        reward_clip_min=reward_clip_min,
    )
    return scorer.score_batch(
        data_sources=data_sources,
        solution_strs=solution_strs,
        ground_truths=ground_truths,
        extra_infos=extra_infos,
    )
