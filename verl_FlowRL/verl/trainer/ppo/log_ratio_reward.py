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

import json
import math
import os
import random
import threading
from contextlib import contextmanager, nullcontext
from typing import Any, Iterable, Optional

import torch
import torch.nn.functional as F

try:
    import wandb
except Exception:
    wandb = None

from verl.utils import hf_tokenizer
from verl.utils.reward_score import default_compute_score

PROMPT_TOKEN_IDS_KEY = "__verl_prompt_token_ids"
RESPONSE_TOKEN_IDS_KEY = "__verl_response_token_ids"
PROMPT_TEXT_KEY = "__verl_prompt_text"
RESPONSE_TEXT_KEY = "__verl_response_text"

_SCORER = None
_SCORER_CFG = None
_SCORER_LOCK = threading.Lock()
_VERBOSE = os.environ.get("VERL_LOG_RATIO_REWARD_VERBOSE", "0") == "1"
_DEFAULT_MAX_SEQ_LEN = 8192
_DEFAULT_RM_TRAIN_BETA = 0.001
_DEFAULT_RM_TRAIN_LENGTH_NORMALIZE = False
_DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
_DEFAULT_USER_PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. The last line of your response "
    "should be of the form Answer: $Answer (without quotes) where $Answer is the "
    "answer to the problem.\n\n"
    "{problem}\n\n"
    'Remember to put your answer on its own line after "Answer:".'
)


def _vprint(msg: str):
    if _VERBOSE:
        print(f"[log_ratio_reward] {msg}", flush=True)


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off", ""}:
            return False
    return bool(value)


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


def _sequence_logprob_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    length_normalize: bool = False,
    source: str = "sequence_logprob",
) -> torch.Tensor:
    """Mirror chi_squared_rm.py sequence logprob computation."""
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]

    if logits.ndim != 3 or labels.ndim != 2:
        raise RuntimeError(
            f"{source}: expected logits.ndim=3 and labels.ndim=2, got "
            f"logits.ndim={logits.ndim}, labels.ndim={labels.ndim}"
        )
    if tuple(logits.shape[:2]) != tuple(labels.shape):
        raise RuntimeError(
            f"{source}: shape mismatch after shift, logits[:2]={tuple(logits.shape[:2])}, "
            f"labels={tuple(labels.shape)}"
        )

    vocab_size = int(logits.size(-1))
    if vocab_size <= 0:
        raise RuntimeError(f"{source}: invalid logits vocab size {vocab_size}")

    valid_label_mask = labels != -100
    invalid_label_mask = valid_label_mask & ((labels < 0) | (labels >= vocab_size))
    if bool(invalid_label_mask.any().item()):
        invalid_indices = invalid_label_mask.nonzero(as_tuple=False)[:8]
        preview = ", ".join(
            f"{int(i)}:{int(j)}={int(labels[i, j].item())}" for i, j in invalid_indices
        )
        valid_labels = labels[valid_label_mask]
        valid_min = int(valid_labels.min().item()) if valid_labels.numel() > 0 else -100
        valid_max = int(valid_labels.max().item()) if valid_labels.numel() > 0 else -100
        raise RuntimeError(
            f"{source}: labels out of logits vocab range "
            f"(valid_label_min={valid_min}, valid_label_max={valid_max}, "
            f"logits_vocab_size={vocab_size}, invalid_count={int(invalid_label_mask.sum().item())}, "
            f"examples=[{preview}])"
        )

    loss_mask = labels != -100
    safe_labels = labels.masked_fill(~loss_mask, 0)

    log_probs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    per_token_logps = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    loss_mask = loss_mask.to(per_token_logps.dtype)
    seq_logps = (per_token_logps * loss_mask).sum(-1)
    if not length_normalize:
        return seq_logps
    token_counts = loss_mask.sum(-1).clamp_min(1.0)
    return seq_logps / token_counts


def _encode_prompt_response_ids(
    prompt_ids: list[int],
    response_ids: list[int],
    eos_token_id: Optional[int],
    max_length: Optional[int],
) -> dict:
    """Mirror chi_squared_rm.py encode_prompt_response semantics for token IDs."""
    p_ids = list(prompt_ids)
    r_ids = list(response_ids)
    # Keep parity with chi_squared_rm.py: append EOS once, without dedup logic.
    if eos_token_id is not None:
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
    @staticmethod
    def _unwrap_parallel_model(model):
        if isinstance(model, torch.nn.DataParallel):
            return model.module
        return model

    @contextmanager
    def _temporarily_disable_gradient_checkpointing(self, model):
        unwrapped = self._unwrap_parallel_model(model)
        was_enabled = bool(getattr(unwrapped, "is_gradient_checkpointing", False))
        can_toggle = (
            was_enabled
            and hasattr(unwrapped, "gradient_checkpointing_disable")
            and hasattr(unwrapped, "gradient_checkpointing_enable")
        )
        if can_toggle:
            try:
                unwrapped.gradient_checkpointing_disable()
            except Exception:
                can_toggle = False
        try:
            yield
        finally:
            if can_toggle:
                try:
                    unwrapped.gradient_checkpointing_enable()
                except Exception as exc:
                    _vprint(f"failed to re-enable gradient checkpointing after reference forward: {exc}")

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
        online_train_with_rollout: bool,
        online_train_positive_jsonl: Optional[str],
        online_train_beta: float,
        online_train_length_normalize: bool,
        online_train_lr: float,
        online_train_weight_decay: float,
        online_train_micro_batch_size: int,
        online_train_max_pairs: int,
        online_train_r_max: float,
        online_train_r_min: float,
        online_train_reg_coef: float,
        online_train_grad_clip: float,
        online_train_prompt_max_length: Optional[int],
        online_train_seed: int,
        online_train_updates_per_rollout_batch: int,
        online_train_use_dataparallel: bool,
        online_train_rollout_minibatch_size: Optional[int],
        online_train_skip_warmup_data: bool,
        online_train_reference_from_policy: bool,
    ):
        self.beta = float(beta)
        self.micro_batch_size = max(1, int(micro_batch_size))
        self.device_map = str(device_map).lower()
        self.torch_dtype = _to_dtype(torch_dtype)
        self.offload_folder = offload_folder
        self.max_gpu_memory = max_gpu_memory
        self.max_cpu_memory = max_cpu_memory
        self.trust_remote_code = trust_remote_code
        self.use_fast_tokenizer = bool(use_fast_tokenizer)
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

        # Online RM training knobs (separate from inference knobs above).
        self.online_train_with_rollout = _to_bool(online_train_with_rollout)
        if online_train_positive_jsonl in {None, "", "null", "None"}:
            self.online_train_positive_jsonl = None
        else:
            self.online_train_positive_jsonl = str(online_train_positive_jsonl)
        self.online_train_beta = float(
            online_train_beta if online_train_beta is not None else _DEFAULT_RM_TRAIN_BETA
        )
        self.online_train_length_normalize = _to_bool(
            online_train_length_normalize
            if online_train_length_normalize is not None
            else _DEFAULT_RM_TRAIN_LENGTH_NORMALIZE
        )
        self.online_train_lr = float(online_train_lr)
        self.online_train_weight_decay = float(online_train_weight_decay)
        self.online_train_micro_batch_size = max(1, int(online_train_micro_batch_size))
        self.online_train_max_pairs = max(1, int(online_train_max_pairs))
        self.online_train_r_max = float(online_train_r_max)
        self.online_train_r_min = float(online_train_r_min)
        self.online_train_reg_coef = float(online_train_reg_coef)
        self.online_train_grad_clip = float(online_train_grad_clip)
        self.online_train_prompt_max_length = (
            int(online_train_prompt_max_length)
            if online_train_prompt_max_length is not None
            else None
        )
        self.online_train_seed = int(online_train_seed)
        self.online_train_updates_per_rollout_batch = max(1, int(online_train_updates_per_rollout_batch))
        self.online_train_use_dataparallel = _to_bool(online_train_use_dataparallel)
        self.online_train_rollout_minibatch_size = (
            int(online_train_rollout_minibatch_size)
            if online_train_rollout_minibatch_size is not None
            else None
        )
        if self.online_train_rollout_minibatch_size is not None and self.online_train_rollout_minibatch_size <= 0:
            self.online_train_rollout_minibatch_size = None
        self.online_train_skip_warmup_data = _to_bool(online_train_skip_warmup_data)
        self.online_train_reference_from_policy = _to_bool(online_train_reference_from_policy)
        self._online_train_rng = random.Random(self.online_train_seed)
        self._online_train_enabled = False
        self._online_prompt_to_pos_responses: dict[tuple[int, ...], list[list[int]]] = {}
        self._online_all_positive_responses: list[list[int]] = []
        self._online_prompt_keys: list[tuple[int, ...]] = []
        self._online_last_train_pairs: list[tuple[list[int], list[int], list[int]]] = []
        self._online_train_optimizer = None
        self._online_train_updates = 0
        try:
            self.online_train_log_every = max(1, int(os.environ.get("VERL_LOG_RATIO_RM_LOSS_LOG_EVERY", "1")))
        except (TypeError, ValueError):
            self.online_train_log_every = 1

        if repeat_penalty_weight != 0.0 or repeat_penalty_ngram_size is not None or repeat_penalty_clip_min is not None:
            _vprint("repeat penalty kwargs are ignored; final reward uses rule-based reward + log-ratio reward.")

        if self.device_map in {"auto", "cuda"} and not torch.cuda.is_available():
            self.device_map = "cpu"

        if self.device_map == "cpu" and self.torch_dtype == torch.float16:
            self.torch_dtype = torch.float32

        if self.offload_folder:
            os.makedirs(self.offload_folder, exist_ok=True)

        self._tokenizer_path = tokenizer_path or policy_model_path
        self.tokenizer = self._load_tokenizer(self._tokenizer_path)
        tokenizer_token_id_upper_bound = self._infer_tokenizer_upper_bound(self.tokenizer)
        self.pad_token_id = self.tokenizer.pad_token_id

        self.policy_model = self._load_model(
            policy_model_path,
            target_vocab_size=tokenizer_token_id_upper_bound,
        )
        self.reference_model = self._load_model(
            reference_model_path,
            target_vocab_size=tokenizer_token_id_upper_bound,
        )

        model_vocab_sizes = [
            self._infer_model_vocab_size(self.policy_model),
            self._infer_model_vocab_size(self.reference_model),
        ]
        model_vocab_sizes = [int(v) for v in model_vocab_sizes if v is not None and int(v) > 0]
        model_token_id_upper_bound = min(model_vocab_sizes) if model_vocab_sizes else None

        if (
            tokenizer_token_id_upper_bound is not None
            and model_token_id_upper_bound is not None
            and tokenizer_token_id_upper_bound > model_token_id_upper_bound
        ):
            fallback_tokenizer_path = policy_model_path
            fallback_tokenizer = None
            fallback_upper_bound = None
            try:
                same_path = os.path.realpath(str(fallback_tokenizer_path)) == os.path.realpath(str(self._tokenizer_path))
            except Exception:
                same_path = str(fallback_tokenizer_path) == str(self._tokenizer_path)
            if not same_path:
                try:
                    fallback_tokenizer = self._load_tokenizer(fallback_tokenizer_path)
                    fallback_upper_bound = self._infer_tokenizer_upper_bound(fallback_tokenizer)
                except Exception as exc:
                    _vprint(
                        "failed to load fallback tokenizer from policy model path "
                        f"({fallback_tokenizer_path}): {exc}"
                    )

            if (
                fallback_tokenizer is not None
                and (
                    fallback_upper_bound is None
                    or fallback_upper_bound <= model_token_id_upper_bound
                )
            ):
                _vprint(
                    "tokenizer/model vocab mismatch detected; switching tokenizer to policy model path "
                    f"({fallback_tokenizer_path})"
                )
                self.tokenizer = fallback_tokenizer
                self._tokenizer_path = fallback_tokenizer_path
                tokenizer_token_id_upper_bound = fallback_upper_bound
                self.pad_token_id = self.tokenizer.pad_token_id
            else:
                _vprint(
                    "tokenizer/model vocab mismatch detected; raw rollout token IDs may be incompatible "
                    f"(tokenizer_upper={tokenizer_token_id_upper_bound}, model_upper={model_token_id_upper_bound}). "
                    "Will fallback to text retokenization per sample when needed."
                )

        self._model_token_id_upper_bound = model_token_id_upper_bound
        self._token_id_upper_bound = (
            model_token_id_upper_bound if model_token_id_upper_bound is not None else tokenizer_token_id_upper_bound
        )
        if _VERBOSE:
            _vprint(
                f"token_id_upper_bound={self._token_id_upper_bound} "
                f"(tokenizer={tokenizer_token_id_upper_bound}, model={model_token_id_upper_bound}, "
                f"tokenizer_path={self._tokenizer_path})"
            )

        if self.online_train_with_rollout:
            self._init_online_training()
            if not self._online_train_enabled:
                raise RuntimeError(
                    "online_train_with_rollout=True but online RM training failed to initialize; "
                    "aborting to avoid silently skipping RM updates."
                )

        _vprint(
            f"initialized scorer: device_map={self.device_map}, dtype={self.torch_dtype}, "
            f"micro_batch_size={self.micro_batch_size}, max_seq_len={self.max_seq_len}, "
            f"log_ratio_reward_clip_min={self.log_ratio_reward_clip_min}, "
            f"reward_clip_min={self.reward_clip_min}, "
            f"online_train_enabled={self._online_train_enabled}, "
            f"online_train_beta={self.online_train_beta}, "
            f"online_train_length_normalize={int(self.online_train_length_normalize)}, "
            f"online_train_reference_from_policy={int(self.online_train_reference_from_policy)}"
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

    def _load_tokenizer(self, tokenizer_path: str):
        tokenizer = hf_tokenizer(
            tokenizer_path,
            trust_remote_code=self.trust_remote_code,
            use_fast=self.use_fast_tokenizer,
            correct_pad_token=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @staticmethod
    def _infer_tokenizer_upper_bound(tokenizer) -> Optional[int]:
        try:
            return int(len(tokenizer))
        except Exception:
            return None


    @staticmethod
    def _maybe_patch_qwen3_rope_config(config, model_path: str) -> Any:
        if getattr(config, "model_type", None) != "qwen3":
            return config

        rope_params = getattr(config, "rope_parameters", None)
        rope_scaling = getattr(config, "rope_scaling", None)
        patched = False

        if rope_scaling is None and isinstance(rope_params, dict) and rope_params:
            config.rope_scaling = dict(rope_params)
            patched = True

        if isinstance(rope_params, dict):
            rope_theta = rope_params.get("rope_theta", None)
            if rope_theta is not None and hasattr(config, "rope_theta"):
                current_theta = getattr(config, "rope_theta", None)
                if current_theta != rope_theta:
                    config.rope_theta = rope_theta
                    patched = True

        if patched:
            _vprint(
                f"patched qwen3 rope config for {model_path}: "
                f"rope_theta={getattr(config, 'rope_theta', None)} "
                f"rope_scaling={getattr(config, 'rope_scaling', None)}"
            )
        return config

    def _load_model(self, model_path: str, target_vocab_size: Optional[int] = None):
        from transformers import AutoConfig, AutoModelForCausalLM

        config = None
        try:
            config = AutoConfig.from_pretrained(
                model_path,
                trust_remote_code=self.trust_remote_code,
            )
            config = self._maybe_patch_qwen3_rope_config(config, model_path=model_path)
        except Exception as exc:
            _vprint(f"failed to load/patch AutoConfig for {model_path}: {exc}")

        load_kwargs = {
            "torch_dtype": self.torch_dtype,
            "trust_remote_code": self.trust_remote_code,
            "low_cpu_mem_usage": True,
            "attn_implementation": self.attn_implementation,
        }
        if config is not None:
            load_kwargs["config"] = config

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

        if target_vocab_size is not None:
            try:
                target_vocab_size = int(target_vocab_size)
            except (TypeError, ValueError):
                target_vocab_size = None
        if target_vocab_size is not None and target_vocab_size > 0:
            model_vocab_size = self._infer_model_vocab_size(model)
            if model_vocab_size is not None and int(model_vocab_size) != int(target_vocab_size):
                model.resize_token_embeddings(int(target_vocab_size))
                # Keep config vocab consistent with resized embedding/head tables.
                if hasattr(model, "config") and hasattr(model.config, "vocab_size"):
                    model.config.vocab_size = int(target_vocab_size)
                _vprint(
                    f"resized model token embeddings for {model_path}: "
                    f"{model_vocab_size} -> {target_vocab_size}"
                )

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

    @staticmethod
    def _infer_model_vocab_size(model) -> Optional[int]:
        # Use the strictest available bound because scoring uses logits.gather on labels.
        # Some checkpoints can have inconsistent sizes across config/input/output heads.
        candidates: list[int] = []

        try:
            emb = model.get_input_embeddings()
            if emb is not None and hasattr(emb, "weight") and emb.weight is not None:
                vocab_int = int(emb.weight.shape[0])
                if vocab_int > 0:
                    candidates.append(vocab_int)
        except Exception:
            pass

        try:
            out_emb = model.get_output_embeddings()
            if out_emb is not None:
                if hasattr(out_emb, "weight") and out_emb.weight is not None:
                    vocab_int = int(out_emb.weight.shape[0])
                    if vocab_int > 0:
                        candidates.append(vocab_int)
                elif hasattr(out_emb, "out_features"):
                    vocab_int = int(out_emb.out_features)
                    if vocab_int > 0:
                        candidates.append(vocab_int)
        except Exception:
            pass

        cfg = getattr(model, "config", None)
        if cfg is not None:
            vocab_size = getattr(cfg, "vocab_size", None)
            if vocab_size is not None:
                try:
                    vocab_int = int(vocab_size)
                    if vocab_int > 0:
                        candidates.append(vocab_int)
                except (TypeError, ValueError):
                    pass

        if not candidates:
            return None
        return min(candidates)

    @classmethod
    def _infer_model_input_vocab_size(cls, model) -> Optional[int]:
        """Infer input embedding vocab rows from the actual forward model."""
        unwrapped = cls._unwrap_parallel_model(model)
        try:
            emb = unwrapped.get_input_embeddings()
            if emb is not None and hasattr(emb, "weight") and emb.weight is not None:
                rows = int(emb.weight.shape[0])
                if rows > 0:
                    return rows
        except Exception:
            pass
        return cls._infer_model_vocab_size(unwrapped)

    @staticmethod
    def _normalize_prompt_text(prompt: Any) -> str:
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, (list, tuple)):
            return "\n".join(str(x) for x in prompt)
        if isinstance(prompt, dict):
            for key in ["prompt", "question", "content", "text"]:
                if key in prompt:
                    return LogRatioRewardScorer._normalize_prompt_text(prompt[key])
            return json.dumps(prompt, ensure_ascii=False)
        return str(prompt)

    def _format_prompt_for_positive_pool(self, prompt: Any) -> str:
        question = self._normalize_prompt_text(prompt)
        has_chat_template = hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None)
        if has_chat_template:
            messages = [
                {"role": "system", "content": _DEFAULT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _DEFAULT_USER_PROMPT_TEMPLATE.format(problem=question),
                },
            ]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"User: {question}\nAssistant:"

    def _safe_int_list(self, values: Any, *, source: str) -> list[int]:
        """Convert token IDs to ints and fail fast on invalid values.

        We intentionally do not silently drop IDs here because dropping can
        change effective training targets and hide tokenizer/model mismatches.
        """
        if values is None:
            return []
        out: list[int] = []
        for idx, value in enumerate(values):
            try:
                token_id = int(value)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"{source} contains a non-integer token id at index {idx}: {value!r}"
                ) from exc
            out.append(token_id)

        upper = self._token_id_upper_bound
        if upper is None:
            invalid = [(idx, tok) for idx, tok in enumerate(out) if tok < 0]
        else:
            invalid = [(idx, tok) for idx, tok in enumerate(out) if tok < 0 or tok >= upper]
        if invalid:
            preview = ", ".join(f"{idx}:{tok}" for idx, tok in invalid[:8])
            raise RuntimeError(
                f"{source} has out-of-range token ids "
                f"(upper_bound={upper}, invalid_count={len(invalid)}, examples=[{preview}])"
            )
        return out

    def _tokenize_text_to_ids(self, text: str, *, source: str) -> list[int]:
        token_ids = self.tokenizer(text, add_special_tokens=False).input_ids
        return self._safe_int_list(token_ids, source=source)

    def _load_online_positive_pool(self) -> None:
        assert self.online_train_positive_jsonl is not None

        prompt_to_pos: dict[tuple[int, ...], list[list[int]]] = {}
        loaded = 0
        skipped = 0
        with open(self.online_train_positive_jsonl, "r", encoding="utf-8") as f:
            for line_idx, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    if _VERBOSE and skipped <= 3:
                        _vprint(f"skip malformed positive jsonl line {line_idx}")
                    continue

                responses = record.get("responses")
                if not isinstance(responses, (list, tuple)) or len(responses) < 2:
                    skipped += 1
                    continue

                pos_text = responses[1]
                if pos_text is None:
                    skipped += 1
                    continue

                prompt_text = self._format_prompt_for_positive_pool(record.get("prompt", ""))
                prompt_ids = self._tokenize_text_to_ids(prompt_text, source=f"positive_pool.prompt.line_{line_idx}")
                if self.online_train_prompt_max_length is not None and self.online_train_prompt_max_length > 0:
                    if len(prompt_ids) > self.online_train_prompt_max_length:
                        prompt_ids = prompt_ids[-self.online_train_prompt_max_length :]
                if not prompt_ids:
                    skipped += 1
                    continue

                pos_ids = self._tokenize_text_to_ids(str(pos_text), source=f"positive_pool.positive_response.line_{line_idx}")
                if not pos_ids:
                    skipped += 1
                    continue

                key = tuple(int(x) for x in prompt_ids)
                prompt_to_pos.setdefault(key, []).append([int(x) for x in pos_ids])
                loaded += 1

        self._online_prompt_to_pos_responses = prompt_to_pos
        self._online_prompt_keys = list(prompt_to_pos.keys())
        self._online_all_positive_responses = [
            list(response_ids)
            for response_list in prompt_to_pos.values()
            for response_ids in response_list
        ]
        _vprint(
            f"online positive pool loaded: pairs={loaded}, unique_prompts={len(prompt_to_pos)}, "
            f"skipped={skipped}, source={self.online_train_positive_jsonl}"
        )

    def _init_online_training(self) -> None:
        if not self.online_train_positive_jsonl:
            _vprint("online RM training disabled: missing online_train_positive_jsonl")
            return
        if not os.path.exists(self.online_train_positive_jsonl):
            _vprint(
                "online RM training disabled: positive jsonl not found "
                f"({self.online_train_positive_jsonl})"
            )
            return

        if hasattr(self.policy_model, "hf_device_map"):
            _vprint("online RM training disabled: policy model uses hf_device_map/device_map=auto")
            return
        if not self.online_train_reference_from_policy and hasattr(self.reference_model, "hf_device_map"):
            _vprint("online RM training disabled: reference model uses hf_device_map/device_map=auto")
            return

        try:
            policy_device = next(self.policy_model.parameters()).device
            if self.online_train_reference_from_policy:
                reference_device = policy_device
            else:
                reference_device = next(self.reference_model.parameters()).device
        except StopIteration:
            _vprint("online RM training disabled: model has no trainable parameters")
            return

        if policy_device.type != "cuda":
            _vprint(
                "online RM training disabled: policy model is not on CUDA. "
                "Set device_map=cuda for reward workers."
            )
            return

        visible_cuda_devices = torch.cuda.device_count()
        using_dataparallel = False
        if self.online_train_use_dataparallel and visible_cuda_devices > 1:
            self.policy_model = torch.nn.DataParallel(self.policy_model)
            if not self.online_train_reference_from_policy:
                self.reference_model = torch.nn.DataParallel(self.reference_model)
            policy_device = next(self.policy_model.parameters()).device
            if self.online_train_reference_from_policy:
                reference_device = policy_device
            else:
                reference_device = next(self.reference_model.parameters()).device
            using_dataparallel = True
            _vprint(
                f"online RM training using DataParallel over {visible_cuda_devices} visible GPUs"
            )
        if (
            not using_dataparallel
            and not self.online_train_reference_from_policy
            and reference_device != policy_device
        ):
            _vprint(
                "online RM training forcing co-located devices to disable peer copies: "
                f"moving reference from {reference_device} to {policy_device}"
            )
            self.reference_model = self.reference_model.to(policy_device)
            reference_device = next(self.reference_model.parameters()).device
        if (
            not using_dataparallel
            and not self.online_train_reference_from_policy
            and reference_device != policy_device
        ):
            raise RuntimeError(
                "online RM training requires policy/reference on the same device "
                "when DataParallel is disabled; refusing cross-device peer copies "
                f"(policy={policy_device}, reference={reference_device})."
            )

        self._load_online_positive_pool()
        if not self._online_prompt_to_pos_responses:
            _vprint("online RM training disabled: positive pool is empty")
            return

        if not self.online_train_reference_from_policy:
            for parameter in self.reference_model.parameters():
                parameter.requires_grad_(False)
        else:
            _vprint("online RM training uses detached current policy as the reference branch")

        for parameter in self.policy_model.parameters():
            parameter.requires_grad_(True)

        policy_train_model = self.policy_model.module if isinstance(self.policy_model, torch.nn.DataParallel) else self.policy_model
        if hasattr(policy_train_model, "gradient_checkpointing_enable"):
            try:
                policy_train_model.gradient_checkpointing_enable()
            except Exception as exc:
                _vprint(f"online RM gradient checkpointing enable failed: {exc}")
        if hasattr(policy_train_model, "config") and hasattr(policy_train_model.config, "use_cache"):
            policy_train_model.config.use_cache = False

        self._online_train_optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=self.online_train_lr,
            weight_decay=self.online_train_weight_decay,
        )

        self.policy_model.eval()
        self.reference_model.eval()
        self._online_train_enabled = True

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
        for row_idx, (solution, extra) in enumerate(zip(solution_strs, extra_infos)):
            prompt_ids: list[int] = []
            response_ids: list[int] = []

            if isinstance(extra, dict):
                has_prompt_ids = PROMPT_TOKEN_IDS_KEY in extra and extra.get(PROMPT_TOKEN_IDS_KEY) is not None
                has_response_ids = RESPONSE_TOKEN_IDS_KEY in extra and extra.get(RESPONSE_TOKEN_IDS_KEY) is not None
                if has_prompt_ids and has_response_ids:
                    try:
                        prompt_ids = self._safe_int_list(
                            extra.get(PROMPT_TOKEN_IDS_KEY),
                            source=f"rollout.prompt_token_ids.score_batch_row_{row_idx}",
                        )
                        response_ids = self._safe_int_list(
                            extra.get(RESPONSE_TOKEN_IDS_KEY),
                            source=f"rollout.response_token_ids.score_batch_row_{row_idx}",
                        )
                    except RuntimeError as exc:
                        if _VERBOSE:
                            _vprint(
                                f"score_batch row {row_idx}: raw rollout token ids incompatible; "
                                f"fallback to text retokenization ({exc})"
                            )
                        prompt_ids = []
                        response_ids = []

                if not prompt_ids:
                    prompt_text = extra.get(PROMPT_TEXT_KEY)
                    if isinstance(prompt_text, str) and prompt_text:
                        prompt_ids = self._tokenize_text_to_ids(
                            prompt_text,
                            source=f"rollout.prompt_text.score_batch_row_{row_idx}",
                        )

                if not response_ids:
                    response_text = extra.get(RESPONSE_TEXT_KEY)
                    if isinstance(response_text, str) and response_text:
                        response_ids = self._tokenize_text_to_ids(
                            response_text,
                            source=f"rollout.response_text.score_batch_row_{row_idx}",
                        )

            if not response_ids:
                response_ids = self._tokenize_text_to_ids(
                    str(solution),
                    source=f"solution_text.score_batch_row_{row_idx}",
                )

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
        attention_mask = attention_mask.ne(0).to(dtype=torch.long)
        label_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x["labels"], dtype=torch.long) for x in encoded_list],
            batch_first=True,
            padding_value=-100,
        )
        response_lens_tensor = torch.tensor([int(x["response_token_len"]) for x in encoded_list], dtype=torch.long)

        return input_ids, attention_mask, label_ids, response_lens_tensor

    def _prepare_train_io(
        self,
        encoded_list: list[dict],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        attention_mask = attention_mask.ne(0).to(dtype=torch.long)
        labels = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x["labels"], dtype=torch.long) for x in encoded_list],
            batch_first=True,
            padding_value=-100,
        )
        return input_ids, attention_mask, labels

    def _forward_causal_lm_logits(
        self,
        model,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        runtime_vocab_upper_bound = self._infer_model_input_vocab_size(model)
        if runtime_vocab_upper_bound is not None:
            token_min = int(input_ids.min().item())
            token_max = int(input_ids.max().item())
            if token_min < 0 or token_max >= int(runtime_vocab_upper_bound):
                invalid_mask = (input_ids < 0) | (input_ids >= int(runtime_vocab_upper_bound))
                invalid_indices = invalid_mask.nonzero(as_tuple=False)[:8]
                preview = ", ".join(
                    f"{int(i)}:{int(j)}={int(input_ids[i, j].item())}" for i, j in invalid_indices
                )
                raise RuntimeError(
                    "forward input_ids out of embedding range "
                    f"(min={token_min}, max={token_max}, vocab_upper_bound={int(runtime_vocab_upper_bound)}, "
                    f"invalid_count={int(invalid_mask.sum().item())}, examples=[{preview}])"
                )

        # DataParallel can scatter tiny minibatches into empty per-replica chunks;
        # run on the base module when batch is too small for stable per-replica slices.
        if isinstance(model, torch.nn.DataParallel):
            device_ids = list(getattr(model, "device_ids", []) or [])
            replica_count = len(device_ids)
            batch_size = int(input_ids.size(0))
            if replica_count > 1 and batch_size <= replica_count:
                base_model = model.module
                base_device = next(base_model.parameters()).device
                if input_ids.device != base_device:
                    input_ids = input_ids.to(base_device, non_blocking=False)
                    attention_mask = attention_mask.to(base_device, non_blocking=False)
                return base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).logits

        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits

    def _validate_forward_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        source: str,
        vocab_upper_bound: Optional[int],
    ) -> None:
        if input_ids.ndim != 2 or attention_mask.ndim != 2:
            raise RuntimeError(
                f"{source}: expected 2D tensors, got input_ids.ndim={input_ids.ndim}, "
                f"attention_mask.ndim={attention_mask.ndim}"
            )
        if tuple(input_ids.shape) != tuple(attention_mask.shape):
            raise RuntimeError(
                f"{source}: shape mismatch input_ids={tuple(input_ids.shape)} "
                f"attention_mask={tuple(attention_mask.shape)}"
            )
        if input_ids.numel() == 0:
            raise RuntimeError(f"{source}: empty input_ids tensor")

        token_min = int(input_ids.min().item())
        token_max = int(input_ids.max().item())
        if token_min < 0:
            raise RuntimeError(f"{source}: negative token id found (min={token_min})")
        if vocab_upper_bound is not None and token_max >= int(vocab_upper_bound):
            raise RuntimeError(
                f"{source}: token id out of model vocab range "
                f"(max={token_max}, vocab_upper_bound={int(vocab_upper_bound)})"
            )

        mask_min = int(attention_mask.min().item())
        mask_max = int(attention_mask.max().item())
        if mask_min < 0 or mask_max > 1:
            raise RuntimeError(
                f"{source}: attention_mask must be binary 0/1, got min={mask_min}, max={mask_max}"
            )
        valid_counts = attention_mask.sum(dim=-1)
        if bool((valid_counts <= 0).any().item()):
            raise RuntimeError(f"{source}: found sequence with zero valid tokens in attention_mask")

    def _sample_online_positive_response(self, prompt_ids: list[int]) -> list[int]:
        prompt_key = tuple(int(x) for x in prompt_ids)
        prompt_candidates = self._online_prompt_to_pos_responses.get(prompt_key)
        if prompt_candidates:
            return list(self._online_train_rng.choice(prompt_candidates))

        prompt_prefix = ",".join(str(x) for x in prompt_ids[:16])
        raise RuntimeError(
            "online RM prompt mismatch: no positive sample for rollout prompt "
            f"(prompt_len={len(prompt_ids)}, prompt_prefix=[{prompt_prefix}])"
        )

    def _build_synthetic_online_train_pairs(self, num_pairs: int) -> list[tuple[list[int], list[int], list[int]]]:
        if not self._online_prompt_keys:
            return []

        target_pairs = max(1, int(num_pairs))
        synthetic_pairs: list[tuple[list[int], list[int], list[int]]] = []
        for _ in range(target_pairs):
            prompt_key = self._online_train_rng.choice(self._online_prompt_keys)
            prompt_ids = [int(x) for x in prompt_key]
            prompt_candidates = self._online_prompt_to_pos_responses.get(prompt_key, [])
            if prompt_candidates:
                pos_response_ids = list(self._online_train_rng.choice(prompt_candidates))
            elif self._online_all_positive_responses:
                pos_response_ids = list(self._online_train_rng.choice(self._online_all_positive_responses))
            else:
                continue

            if len(prompt_candidates) >= 2:
                neg_response_ids = list(self._online_train_rng.choice(prompt_candidates))
                for _retry in range(4):
                    if neg_response_ids != pos_response_ids:
                        break
                    neg_response_ids = list(self._online_train_rng.choice(prompt_candidates))
            elif self._online_all_positive_responses:
                neg_response_ids = list(self._online_train_rng.choice(self._online_all_positive_responses))
            else:
                neg_response_ids = list(pos_response_ids)

            synthetic_pairs.append((prompt_ids, pos_response_ids, neg_response_ids))
        return synthetic_pairs

    def _build_online_train_pairs(self, data_sources, extra_infos) -> list[tuple[list[int], list[int], list[int]]]:
        data_sources = list(data_sources)
        extra_infos = list(extra_infos) if extra_infos is not None else [None] * len(data_sources)
        if len(extra_infos) != len(data_sources):
            extra_infos = [None] * len(data_sources)

        pairs: list[tuple[list[int], list[int], list[int]]] = []
        non_warmup_count = 0
        dict_extra_count = 0
        valid_token_pair_count = 0
        raw_token_pair_count = 0
        text_retokenized_count = 0
        for row_idx, (data_source, extra) in enumerate(zip(data_sources, extra_infos)):
            if str(data_source) == "warmup":
                continue
            non_warmup_count += 1
            if not isinstance(extra, dict):
                continue
            dict_extra_count += 1

            prompt_ids: list[int] = []
            neg_response_ids: list[int] = []
            used_text_retokenization = False

            has_prompt_ids = PROMPT_TOKEN_IDS_KEY in extra and extra.get(PROMPT_TOKEN_IDS_KEY) is not None
            has_response_ids = RESPONSE_TOKEN_IDS_KEY in extra and extra.get(RESPONSE_TOKEN_IDS_KEY) is not None

            if has_prompt_ids and has_response_ids:
                try:
                    prompt_ids = self._safe_int_list(
                        extra.get(PROMPT_TOKEN_IDS_KEY),
                        source=f"rollout.prompt_token_ids.online_train_row_{row_idx}",
                    )
                    neg_response_ids = self._safe_int_list(
                        extra.get(RESPONSE_TOKEN_IDS_KEY),
                        source=f"rollout.response_token_ids.online_train_row_{row_idx}",
                    )
                    raw_token_pair_count += 1
                except RuntimeError as exc:
                    if _VERBOSE:
                        _vprint(
                            f"online_train row {row_idx}: raw rollout token ids incompatible; "
                            f"fallback to text retokenization ({exc})"
                        )
                    prompt_ids = []
                    neg_response_ids = []

            if not prompt_ids:
                prompt_text = extra.get(PROMPT_TEXT_KEY)
                if isinstance(prompt_text, str) and prompt_text:
                    prompt_ids = self._tokenize_text_to_ids(
                        prompt_text,
                        source=f"rollout.prompt_text.online_train_row_{row_idx}",
                    )
                    used_text_retokenization = True

            if not neg_response_ids:
                neg_response_text = extra.get(RESPONSE_TEXT_KEY)
                if isinstance(neg_response_text, str) and neg_response_text:
                    neg_response_ids = self._tokenize_text_to_ids(
                        neg_response_text,
                        source=f"rollout.response_text.online_train_row_{row_idx}",
                    )
                    used_text_retokenization = True

            if prompt_ids and neg_response_ids and used_text_retokenization:
                text_retokenized_count += 1

            if self.online_train_prompt_max_length is not None and self.online_train_prompt_max_length > 0:
                if len(prompt_ids) > self.online_train_prompt_max_length:
                    prompt_ids = prompt_ids[-self.online_train_prompt_max_length :]

            if not prompt_ids or not neg_response_ids:
                continue
            valid_token_pair_count += 1

            pos_response_ids = self._sample_online_positive_response(prompt_ids)
            pairs.append((prompt_ids, pos_response_ids, neg_response_ids))

        if not pairs:
            raise RuntimeError(
                "online RM found no valid prompt/response pairs in this minibatch; "
                f"check rollout extra_info and prompt alignment "
                f"(batch_size={len(data_sources)}, non_warmup={non_warmup_count}, "
                f"dict_extra={dict_extra_count}, token_pairs={valid_token_pair_count}, "
                f"raw_token_pairs={raw_token_pair_count}, text_retokenized={text_retokenized_count})."
            )

        if len(pairs) > self.online_train_max_pairs:
            pairs = self._online_train_rng.sample(pairs, self.online_train_max_pairs)

        return pairs

    def _online_train_single_update(

        self, data_sources, extra_infos
    ) -> Optional[dict[str, float]]:
        if not self._online_train_enabled or self._online_train_optimizer is None:
            return None

        data_sources = list(data_sources)
        extra_infos = list(extra_infos) if extra_infos is not None else [None] * len(data_sources)
        if len(extra_infos) != len(data_sources):
            extra_infos = [None] * len(data_sources)

        train_pairs = self._build_online_train_pairs(data_sources=data_sources, extra_infos=extra_infos)
        if not train_pairs:
            raise RuntimeError(
                "online RM failed to construct train pairs for update; "
                "positive prompt alignment is required."
            )

        pos_encoded = []
        neg_encoded = []
        for prompt_ids, pos_response_ids, neg_response_ids in train_pairs:
            pos_encoded.append(
                _encode_prompt_response_ids(
                    prompt_ids=prompt_ids,
                    response_ids=pos_response_ids,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_length=self.max_seq_len,
                )
            )
            neg_encoded.append(
                _encode_prompt_response_ids(
                    prompt_ids=prompt_ids,
                    response_ids=neg_response_ids,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_length=self.max_seq_len,
                )
            )

        pos_input_ids, pos_attention_mask, pos_labels = self._prepare_train_io(pos_encoded)
        neg_input_ids, neg_attention_mask, neg_labels = self._prepare_train_io(neg_encoded)

        policy_device = next(self.policy_model.parameters()).device
        online_reference_model = self.policy_model if self.online_train_reference_from_policy else self.reference_model
        reference_device = next(online_reference_model.parameters()).device
        if reference_device != policy_device:
            raise RuntimeError(
                "online RM policy/reference device mismatch detected "
                f"(policy={policy_device}, reference={reference_device}). "
                "Cross-device peer copies are disabled; co-locate models on one "
                "device or enable online_train_use_dataparallel."
            )
        policy_vocab_upper_bound = self._infer_model_input_vocab_size(self.policy_model)
        reference_vocab_upper_bound = self._infer_model_input_vocab_size(online_reference_model)
        if policy_vocab_upper_bound is None:
            policy_vocab_upper_bound = self._model_token_id_upper_bound
        if reference_vocab_upper_bound is None:
            reference_vocab_upper_bound = self._model_token_id_upper_bound

        self.policy_model.train()
        if not self.online_train_reference_from_policy:
            self.reference_model.eval()

        optimizer = self._online_train_optimizer
        optimizer.zero_grad(set_to_none=True)

        total_pairs = int(pos_input_ids.size(0))
        weighted_loss_sum = 0.0
        weighted_loss_pos_sum = 0.0
        weighted_loss_neg_sum = 0.0
        weighted_loss_reg_sum = 0.0

        for start in range(0, total_pairs, self.online_train_micro_batch_size):
            end = min(start + self.online_train_micro_batch_size, total_pairs)
            mb_pairs = end - start
            batch_scale = mb_pairs / float(total_pairs)

            # Keep transfers blocking for deterministic update ordering.
            mb_pos_input_ids = pos_input_ids[start:end].contiguous().to(policy_device, non_blocking=False)
            mb_pos_attention_mask = pos_attention_mask[start:end].contiguous().to(policy_device, non_blocking=False)
            mb_pos_attention_mask = mb_pos_attention_mask.ne(0).to(dtype=torch.long)
            mb_pos_labels = pos_labels[start:end].contiguous().to(policy_device, non_blocking=False)

            mb_neg_input_ids = neg_input_ids[start:end].contiguous().to(policy_device, non_blocking=False)
            mb_neg_attention_mask = neg_attention_mask[start:end].contiguous().to(policy_device, non_blocking=False)
            mb_neg_attention_mask = mb_neg_attention_mask.ne(0).to(dtype=torch.long)
            mb_neg_labels = neg_labels[start:end].contiguous().to(policy_device, non_blocking=False)

            self._validate_forward_inputs(
                mb_pos_input_ids,
                mb_pos_attention_mask,
                source=f"online_train.pos_policy.update_{self._online_train_updates + 1}.mb_{start}_{end}",
                vocab_upper_bound=policy_vocab_upper_bound,
            )
            self._validate_forward_inputs(
                mb_neg_input_ids,
                mb_neg_attention_mask,
                source=f"online_train.neg_policy.update_{self._online_train_updates + 1}.mb_{start}_{end}",
                vocab_upper_bound=policy_vocab_upper_bound,
            )

            # Positive branch: backward immediately to reduce activation peak memory.
            pos_policy_logits = self._forward_causal_lm_logits(
                self.policy_model,
                input_ids=mb_pos_input_ids,
                attention_mask=mb_pos_attention_mask,
            )
            logp_pi_pos = _sequence_logprob_from_logits(
                pos_policy_logits,
                mb_pos_labels,
                length_normalize=self.online_train_length_normalize,
                source=f"online_train.pos_policy_logprob.update_{self._online_train_updates + 1}.mb_{start}_{end}",
            )
            del pos_policy_logits

            reference_gc_ctx = (
                self._temporarily_disable_gradient_checkpointing(online_reference_model)
                if self.online_train_reference_from_policy
                else nullcontext()
            )
            with reference_gc_ctx:
                with torch.no_grad():
                    mb_pos_input_ids_ref = mb_pos_input_ids
                    mb_pos_attention_mask_ref = mb_pos_attention_mask
                    mb_pos_labels_ref = mb_pos_labels

                    mb_pos_attention_mask_ref = mb_pos_attention_mask_ref.ne(0).to(dtype=torch.long)
                    self._validate_forward_inputs(
                        mb_pos_input_ids_ref,
                        mb_pos_attention_mask_ref,
                        source=f"online_train.pos_reference.update_{self._online_train_updates + 1}.mb_{start}_{end}",
                        vocab_upper_bound=reference_vocab_upper_bound,
                    )

                    pos_ref_logits = self._forward_causal_lm_logits(
                        online_reference_model,
                        input_ids=mb_pos_input_ids_ref,
                        attention_mask=mb_pos_attention_mask_ref,
                    )
                    logp_ref_pos = _sequence_logprob_from_logits(
                        pos_ref_logits,
                        mb_pos_labels_ref,
                        length_normalize=self.online_train_length_normalize,
                        source=f"online_train.pos_ref_logprob.update_{self._online_train_updates + 1}.mb_{start}_{end}",
                    )
                    del pos_ref_logits

            if logp_ref_pos.device != logp_pi_pos.device:
                logp_ref_pos = logp_ref_pos.to(logp_pi_pos.device, non_blocking=False)

            lr_pos = logp_pi_pos - logp_ref_pos
            pred_pos = self.online_train_beta * lr_pos
            loss_pos = 0.5 * (pred_pos - self.online_train_r_max).pow(2)
            loss_reg_pos = self.online_train_reg_coef * pred_pos.pow(2)
            loss_pos_mean = loss_pos.mean()
            loss_reg_pos_mean = loss_reg_pos.mean()
            pos_total_loss = loss_pos_mean + loss_reg_pos_mean
            (pos_total_loss * batch_scale).backward()

            loss_pos_scalar = float(loss_pos_mean.detach().item())
            loss_reg_pos_scalar = float(loss_reg_pos_mean.detach().item())

            del (
                logp_pi_pos,
                logp_ref_pos,
                lr_pos,
                pred_pos,
                loss_pos,
                loss_reg_pos,
                loss_pos_mean,
                loss_reg_pos_mean,
                pos_total_loss,
            )
            del mb_pos_input_ids_ref, mb_pos_attention_mask_ref, mb_pos_labels_ref

            # Negative branch: run separately to keep only one branch graph alive at a time.
            neg_policy_logits = self._forward_causal_lm_logits(
                self.policy_model,
                input_ids=mb_neg_input_ids,
                attention_mask=mb_neg_attention_mask,
            )
            logp_pi_neg = _sequence_logprob_from_logits(
                neg_policy_logits,
                mb_neg_labels,
                length_normalize=self.online_train_length_normalize,
                source=f"online_train.neg_policy_logprob.update_{self._online_train_updates + 1}.mb_{start}_{end}",
            )
            del neg_policy_logits

            reference_gc_ctx = (
                self._temporarily_disable_gradient_checkpointing(online_reference_model)
                if self.online_train_reference_from_policy
                else nullcontext()
            )
            with reference_gc_ctx:
                with torch.no_grad():
                    mb_neg_input_ids_ref = mb_neg_input_ids
                    mb_neg_attention_mask_ref = mb_neg_attention_mask
                    mb_neg_labels_ref = mb_neg_labels

                    mb_neg_attention_mask_ref = mb_neg_attention_mask_ref.ne(0).to(dtype=torch.long)
                    self._validate_forward_inputs(
                        mb_neg_input_ids_ref,
                        mb_neg_attention_mask_ref,
                        source=f"online_train.neg_reference.update_{self._online_train_updates + 1}.mb_{start}_{end}",
                        vocab_upper_bound=reference_vocab_upper_bound,
                    )

                    neg_ref_logits = self._forward_causal_lm_logits(
                        online_reference_model,
                        input_ids=mb_neg_input_ids_ref,
                        attention_mask=mb_neg_attention_mask_ref,
                    )
                    logp_ref_neg = _sequence_logprob_from_logits(
                        neg_ref_logits,
                        mb_neg_labels_ref,
                        length_normalize=self.online_train_length_normalize,
                        source=f"online_train.neg_ref_logprob.update_{self._online_train_updates + 1}.mb_{start}_{end}",
                    )
                    del neg_ref_logits

            if logp_ref_neg.device != logp_pi_neg.device:
                logp_ref_neg = logp_ref_neg.to(logp_pi_neg.device, non_blocking=False)

            lr_neg = logp_pi_neg - logp_ref_neg
            pred_neg = self.online_train_beta * lr_neg
            loss_neg = 0.5 * (pred_neg - self.online_train_r_min).pow(2)
            loss_reg_neg = self.online_train_reg_coef * pred_neg.pow(2)
            loss_neg_mean = loss_neg.mean()
            loss_reg_neg_mean = loss_reg_neg.mean()
            neg_total_loss = loss_neg_mean + loss_reg_neg_mean
            (neg_total_loss * batch_scale).backward()

            loss_neg_scalar = float(loss_neg_mean.detach().item())
            loss_reg_neg_scalar = float(loss_reg_neg_mean.detach().item())

            del (
                mb_pos_input_ids,
                mb_pos_attention_mask,
                mb_pos_labels,
                mb_neg_input_ids,
                mb_neg_attention_mask,
                mb_neg_labels,
                logp_pi_neg,
                logp_ref_neg,
                lr_neg,
                pred_neg,
                loss_neg,
                loss_reg_neg,
                loss_neg_mean,
                loss_reg_neg_mean,
                neg_total_loss,
            )
            if reference_device != policy_device:
                del mb_neg_input_ids_ref, mb_neg_attention_mask_ref, mb_neg_labels_ref

            loss_reg_scalar = loss_reg_pos_scalar + loss_reg_neg_scalar
            loss_scalar = loss_pos_scalar + loss_neg_scalar + loss_reg_scalar

            weighted_loss_sum += loss_scalar * mb_pairs
            weighted_loss_pos_sum += loss_pos_scalar * mb_pairs
            weighted_loss_neg_sum += loss_neg_scalar * mb_pairs
            weighted_loss_reg_sum += loss_reg_scalar * mb_pairs

        if self.online_train_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.online_train_grad_clip)
        optimizer.step()

        self.policy_model.eval()
        self._online_train_updates += 1
        self._online_last_train_pairs = [
            (list(prompt_ids), list(pos_response_ids), list(neg_response_ids))
            for prompt_ids, pos_response_ids, neg_response_ids in train_pairs
        ]

        if self.clear_cuda_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()

        avg_loss = weighted_loss_sum / float(max(total_pairs, 1))
        avg_loss_pos = weighted_loss_pos_sum / float(max(total_pairs, 1))
        avg_loss_neg = weighted_loss_neg_sum / float(max(total_pairs, 1))
        avg_loss_reg = weighted_loss_reg_sum / float(max(total_pairs, 1))

        should_log_loss = (
            self.online_train_log_every <= 1
            or self._online_train_updates <= 3
            or self._online_train_updates % self.online_train_log_every == 0
        )
        if should_log_loss:
            print(
                (
                    "[log_ratio_reward] "
                    f"online_rm_update={self._online_train_updates} "
                    f"loss={avg_loss:.6f} "
                    f"loss_pos={avg_loss_pos:.6f} "
                    f"loss_neg={avg_loss_neg:.6f} "
                    f"loss_reg={avg_loss_reg:.6f} "
                    f"pairs={total_pairs}"
                ),
                flush=True,
            )
            if wandb is not None and wandb.run is not None:
                wandb.log(
                    {
                        "online_rm/update": int(self._online_train_updates),
                        "online_rm/loss": float(avg_loss),
                        "online_rm/loss_pos": float(avg_loss_pos),
                        "online_rm/loss_neg": float(avg_loss_neg),
                        "online_rm/loss_reg": float(avg_loss_reg),
                        "online_rm/pairs": float(total_pairs),
                    },
                    commit=False,
                )

        return {
            "online_rm_loss": float(avg_loss),
            "online_rm_loss_pos": float(avg_loss_pos),
            "online_rm_loss_neg": float(avg_loss_neg),
            "online_rm_loss_reg": float(avg_loss_reg),
            "online_rm_pairs": float(total_pairs),
        }

    def _online_train_zero_stats(self, num_updates_target: int, num_updates_actual: int = 0) -> dict[str, float]:
        return {
            "online_rm_loss": 0.0,
            "online_rm_loss_pre_update": 0.0,
            "online_rm_loss_mean_updates": 0.0,
            "online_rm_loss_last": 0.0,
            "online_rm_pairs": 0.0,
            "online_rm_updates_per_batch_actual": float(max(0, int(num_updates_actual))),
            "online_rm_updates_per_batch_target": float(max(0, int(num_updates_target))),
        }

    def _online_train_on_rollout_batch(self, data_sources, extra_infos) -> Optional[dict[str, float]]:
        if not self._online_train_enabled:
            return None

        data_sources = list(data_sources)
        extra_infos = list(extra_infos) if extra_infos is not None else [None] * len(data_sources)
        if len(extra_infos) != len(data_sources):
            extra_infos = [None] * len(data_sources)

        num_updates = self.online_train_updates_per_rollout_batch

        rollout_minibatch_size = self.online_train_rollout_minibatch_size
        if rollout_minibatch_size is None:
            rollout_minibatch_size = len(data_sources)
        rollout_minibatch_size = max(1, int(rollout_minibatch_size))

        rollout_minibatches: list[tuple[list[Any], list[Any]]] = []
        for start in range(0, len(data_sources), rollout_minibatch_size):
            end = min(start + rollout_minibatch_size, len(data_sources))
            rollout_minibatches.append((data_sources[start:end], extra_infos[start:end]))
        if not rollout_minibatches:
            rollout_minibatches = [([], [])]

        update_stats_list: list[dict[str, float]] = []
        for update_idx in range(num_updates):
            mb_data_sources, mb_extra_infos = rollout_minibatches[update_idx % len(rollout_minibatches)]
            update_stats = self._online_train_single_update(data_sources=mb_data_sources, extra_infos=mb_extra_infos)
            if update_stats is None:
                raise RuntimeError("online RM update unexpectedly returned None while online training is enabled")
            update_stats_list.append(update_stats)

        if not update_stats_list:
            return self._online_train_zero_stats(num_updates_target=num_updates)

        first_loss = float(update_stats_list[0]["online_rm_loss"])
        mean_loss = float(sum(s["online_rm_loss"] for s in update_stats_list) / len(update_stats_list))
        mean_pairs = float(sum(s["online_rm_pairs"] for s in update_stats_list) / len(update_stats_list))
        return {
            # Report pre-update loss for this rollout batch as the primary metric to
            # keep parity with RM pretraining interpretation.
            "online_rm_loss": first_loss,
            "online_rm_loss_pre_update": first_loss,
            # Keep mean-over-updates for debugging multi-update behavior.
            "online_rm_loss_mean_updates": mean_loss,
            "online_rm_loss_last": float(update_stats_list[-1]["online_rm_loss"]),
            "online_rm_pairs": mean_pairs,
            "online_rm_updates_per_batch_actual": float(len(update_stats_list)),
            "online_rm_updates_per_batch_target": float(num_updates),
        }

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
                    logits = self._forward_causal_lm_logits(
                        model,
                        input_ids=mb_input_ids,
                        attention_mask=mb_attention_mask,
                    )
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
                seq_logps = _sequence_logprob_from_logits(
                    logits,
                    mb_label_ids,
                    source=f"score_batch.logprob_mb_{start}_{end}",
                )
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
            rule_reward_scalar = _extract_scalar_score(rule_reward_raw)
            rule_reward = 1.0 if rule_reward_scalar > 0.0 else -1.0

            log_ratio_reward = float(reward.item())
            if not math.isfinite(log_ratio_reward):
                log_ratio_reward = 0.0

            # Strict binary log-ratio reward.
            log_ratio_reward = 1.0 if log_ratio_reward > 0.0 else -1.0

            # Direct composition as requested.
            raw_reward_val = rule_reward + log_ratio_reward

            reward_val = raw_reward_val
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
                    "rule_reward_gate": None,
                    "raw_score_before_clip": raw_reward_val,
                    "reward_clip_min": self.reward_clip_min,
                    "online_rm_loss": None,
                    "online_rm_loss_pre_update": None,
                    "online_rm_loss_mean_updates": None,
                    "online_rm_loss_last": None,
                    "online_rm_pairs": None,
                    "online_rm_updates_per_batch_actual": None,
                    "online_rm_updates_per_batch_target": None,
                }
            )

        # Keep inference reward unchanged for the current batch, then update RM using
        # the same rollout batch for future reward computations.
        if self._online_train_enabled:
            has_non_warmup = any(str(data_source) != "warmup" for data_source in data_sources)
            if has_non_warmup:
                online_train_stats = self._online_train_on_rollout_batch(data_sources=data_sources, extra_infos=extra_infos)
            else:
                online_train_stats = self._online_train_zero_stats(
                    num_updates_target=self.online_train_updates_per_rollout_batch
                )
            if online_train_stats is None:
                online_train_stats = self._online_train_zero_stats(
                    num_updates_target=self.online_train_updates_per_rollout_batch
                )
            for result in results:
                result.update(online_train_stats)

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
    online_train_with_rollout=False,
    online_train_positive_jsonl=None,
    online_train_beta=_DEFAULT_RM_TRAIN_BETA,
    online_train_length_normalize=_DEFAULT_RM_TRAIN_LENGTH_NORMALIZE,
    online_train_lr=5e-7,
    online_train_weight_decay=0.0,
    online_train_micro_batch_size=1,
    online_train_max_pairs=8,
    online_train_r_max=1.0,
    online_train_r_min=-1.0,
    online_train_reg_coef=0.005,
    online_train_grad_clip=1.0,
    online_train_prompt_max_length=None,
    online_train_seed=42,
    online_train_updates_per_rollout_batch=1,
    online_train_use_dataparallel=True,
    online_train_rollout_minibatch_size=None,
    online_train_skip_warmup_data=False,
    online_train_reference_from_policy=True,
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
        online_train_with_rollout=online_train_with_rollout,
        online_train_positive_jsonl=online_train_positive_jsonl,
        online_train_beta=online_train_beta,
        online_train_length_normalize=online_train_length_normalize,
        online_train_lr=online_train_lr,
        online_train_weight_decay=online_train_weight_decay,
        online_train_micro_batch_size=online_train_micro_batch_size,
        online_train_max_pairs=online_train_max_pairs,
        online_train_r_max=online_train_r_max,
        online_train_r_min=online_train_r_min,
        online_train_reg_coef=online_train_reg_coef,
        online_train_grad_clip=online_train_grad_clip,
        online_train_prompt_max_length=online_train_prompt_max_length,
        online_train_seed=online_train_seed,
        online_train_updates_per_rollout_batch=online_train_updates_per_rollout_batch,
        online_train_use_dataparallel=online_train_use_dataparallel,
        online_train_rollout_minibatch_size=online_train_rollout_minibatch_size,
        online_train_skip_warmup_data=online_train_skip_warmup_data,
        online_train_reference_from_policy=online_train_reference_from_policy,
    )
    return scorer.score_batch(
        data_sources=data_sources,
        solution_strs=solution_strs,
        ground_truths=ground_truths,
        extra_infos=extra_infos,
    )
