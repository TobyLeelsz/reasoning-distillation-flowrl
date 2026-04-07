#!/usr/bin/env python3
"""Check FlowRL reward preprocessing/logits parity against chi_squared_rm.py.

Default check targets negative responses (`responses[0]`) from the paired JSONL.
"""

import argparse
import importlib.util
import inspect
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare chi_squared_rm.py and FlowRL reward pipeline on the same data. "
            "Checks preprocessing parity first, and optionally logits/logprob parity."
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("/proj/weitongzlab/projects/rm_training/generate/dapo_17k_merged_qwen3.jsonl"),
        help="Paired JSONL path.",
    )
    parser.add_argument(
        "--chi-script-path",
        type=Path,
        default=Path("/proj/weitongzlab/projects/rm_training/chi_squared_rm.py"),
        help="Path to chi_squared_rm.py used for RM pretraining.",
    )
    parser.add_argument(
        "--flow-reward-script-path",
        type=Path,
        default=Path(
            "/proj/weitongzlab/projects/reasoning-distillation-flowrl/verl_FlowRL/verl/trainer/ppo/log_ratio_reward.py"
        ),
        help="Path to FlowRL reward script (log_ratio_reward.py).",
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        required=True,
        help="Model path used for logits parity check.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Tokenizer path. Default: --model-name-or-path.",
    )
    parser.add_argument("--num-samples", type=int, default=16, help="How many valid rows to test.")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many valid rows first.")
    parser.add_argument(
        "--response-index",
        type=int,
        default=0,
        help="Response index in `responses` to test (0 is negative in your dataset).",
    )
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--skip-logits-check", action="store_true")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Execution device for logits check.",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "bfloat16", "float16"],
    )
    parser.add_argument("--attn-implementation", type=str, default="eager")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-fast-tokenizer", action="store_true")
    parser.add_argument("--logits-atol", type=float, default=1e-6)
    parser.add_argument("--seq-logprob-atol", type=float, default=1e-6)
    return parser.parse_args()


def load_module(path: Path, module_name: str):
    if not path.exists():
        raise FileNotFoundError(f"Module path not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to import module from: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_rows(path: Path, num_samples: int, offset: int, response_index: int) -> list[dict[str, Any]]:
    if num_samples <= 0:
        raise ValueError(f"--num-samples must be > 0, got {num_samples}")
    if offset < 0:
        raise ValueError(f"--offset must be >= 0, got {offset}")
    if response_index < 0:
        raise ValueError(f"--response-index must be >= 0, got {response_index}")
    if not path.exists():
        raise FileNotFoundError(f"Data path not found: {path}")

    rows: list[dict[str, Any]] = []
    valid_seen = 0
    with path.open("r", encoding="utf-8") as f:
        for line_idx, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            responses = row.get("responses")
            if not isinstance(responses, list) or len(responses) <= response_index:
                continue
            if valid_seen < offset:
                valid_seen += 1
                continue
            rows.append(row)
            if len(rows) >= num_samples:
                break

    if len(rows) < num_samples:
        raise ValueError(
            f"Requested {num_samples} samples at offset {offset}, "
            f"but only found {len(rows)} valid rows with responses[{response_index}]."
        )
    return rows


def _to_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device=cuda requested but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cpu")


def load_tokenizer(tokenizer_path: str, trust_remote_code: bool, use_fast: bool):
    tok = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=trust_remote_code,
        use_fast=use_fast,
    )
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(
    model_path: str,
    device: torch.device,
    dtype: torch.dtype,
    attn_implementation: str,
    trust_remote_code: bool,
):
    model_dtype = dtype
    if device.type == "cpu" and model_dtype == torch.float16:
        print("[warn] CPU + float16 is unsupported for many models; falling back to float32.", flush=True)
        model_dtype = torch.float32

    base_kwargs = {
        "trust_remote_code": trust_remote_code,
        "attn_implementation": attn_implementation,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=model_dtype,
            **base_kwargs,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=model_dtype,
            **base_kwargs,
        )
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model


def call_chi_sequence_logprob(chi_mod, logits: torch.Tensor, labels: torch.Tensor, length_normalize: bool) -> torch.Tensor:
    fn = chi_mod.sequence_logprob_from_logits
    sig = inspect.signature(fn)
    if "length_normalize" in sig.parameters:
        return fn(logits, labels, length_normalize=length_normalize)
    return fn(logits, labels)


def make_flow_harness(flow_mod, tokenizer: Any, max_length: int):
    harness = flow_mod.LogRatioRewardScorer.__new__(flow_mod.LogRatioRewardScorer)
    harness.tokenizer = tokenizer
    harness.max_seq_len = max_length
    try:
        harness._token_id_upper_bound = flow_mod.LogRatioRewardScorer._infer_tokenizer_upper_bound(tokenizer)
    except Exception:
        harness._token_id_upper_bound = None
    return harness


def compare_encoding(lhs: dict[str, list[int]], rhs: dict[str, list[int]]) -> bool:
    for key in ["input_ids", "attention_mask", "labels"]:
        if lhs.get(key) != rhs.get(key):
            return False
    return True


def pad_batch(items: list[dict[str, list[int]]], pad_token_id: int):
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(x["input_ids"], dtype=torch.long) for x in items],
        batch_first=True,
        padding_value=pad_token_id,
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(x["attention_mask"], dtype=torch.long) for x in items],
        batch_first=True,
        padding_value=0,
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(x["labels"], dtype=torch.long) for x in items],
        batch_first=True,
        padding_value=-100,
    )
    return input_ids, attention_mask, labels


def main() -> None:
    args = parse_args()

    chi_mod = load_module(args.chi_script_path, "chi_rm_module")
    flow_mod = load_module(args.flow_reward_script_path, "flow_log_ratio_module")

    tokenizer_path = args.tokenizer_path or args.model_name_or_path
    tokenizer = load_tokenizer(
        tokenizer_path=tokenizer_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=(not args.no_fast_tokenizer),
    )
    print(f"[info] tokenizer loaded from: {tokenizer_path}", flush=True)

    rows = load_rows(
        path=args.data_path,
        num_samples=args.num_samples,
        offset=args.offset,
        response_index=args.response_index,
    )
    print(
        f"[info] loaded rows: num_samples={len(rows)}, offset={args.offset}, response_index={args.response_index}",
        flush=True,
    )

    prompt_key = getattr(flow_mod, "PROMPT_TOKEN_IDS_KEY", "__verl_prompt_token_ids")
    response_key = getattr(flow_mod, "RESPONSE_TOKEN_IDS_KEY", "__verl_response_token_ids")
    flow_harness = make_flow_harness(flow_mod, tokenizer=tokenizer, max_length=args.max_length)

    chi_encoded: list[dict[str, list[int]]] = []
    encode_helper_match = 0
    encode_prepare_match = 0
    mismatch_examples: list[dict[str, Any]] = []

    for idx, row in enumerate(rows):
        prompt_obj = row.get("prompt", "")
        response_text = str(row["responses"][args.response_index])

        prompt_text = chi_mod.format_prompt(tokenizer, prompt_obj)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        response_ids = tokenizer(response_text, add_special_tokens=False).input_ids

        chi_enc = chi_mod.encode_prompt_response(tokenizer, prompt_text, response_text, args.max_length)
        flow_enc = flow_mod._encode_prompt_response_ids(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            eos_token_id=tokenizer.eos_token_id,
            max_length=args.max_length,
        )
        flow_enc_prepare = flow_harness._prepare_encoded_sequences(
            solution_strs=[response_text],
            extra_infos=[{prompt_key: prompt_ids, response_key: response_ids}],
        )[0]

        chi_encoded.append(chi_enc)

        helper_ok = compare_encoding(chi_enc, flow_enc)
        prepare_ok = compare_encoding(chi_enc, flow_enc_prepare)
        encode_helper_match += int(helper_ok)
        encode_prepare_match += int(prepare_ok)

        if (not helper_ok or not prepare_ok) and len(mismatch_examples) < 3:
            mismatch_examples.append(
                {
                    "row_idx": idx,
                    "helper_match": helper_ok,
                    "prepare_match": prepare_ok,
                    "chi_len": len(chi_enc["input_ids"]),
                    "flow_helper_len": len(flow_enc["input_ids"]),
                    "flow_prepare_len": len(flow_enc_prepare["input_ids"]),
                }
            )

    print("=== Preprocessing Parity ===")
    print(f"chi_vs_flow_encode_helper_match={encode_helper_match}/{len(rows)}")
    print(f"chi_vs_flow_prepare_encoded_sequences_match={encode_prepare_match}/{len(rows)}")
    if mismatch_examples:
        print(f"preprocess_mismatch_examples={json.dumps(mismatch_examples, ensure_ascii=False)}")

    if args.skip_logits_check:
        preprocess_pass = (encode_helper_match == len(rows)) and (encode_prepare_match == len(rows))
        print("=== Verdict ===")
        if preprocess_pass:
            print("PASS: preprocessing is equivalent for the tested samples.")
        else:
            print("FAIL: preprocessing differs for at least one tested sample.")
        return

    device = resolve_device(args.device)
    dtype = _to_dtype(args.torch_dtype)
    print(f"[info] loading model: path={args.model_name_or_path}, device={device}, dtype={dtype}", flush=True)
    model = load_model(
        model_path=args.model_name_or_path,
        device=device,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    input_ids, attention_mask, labels = pad_batch(chi_encoded, pad_token_id=pad_token_id)
    micro_batch_size = max(1, int(args.micro_batch_size))

    model_has_hf_device_map = hasattr(model, "hf_device_map")
    if model_has_hf_device_map:
        model_device = torch.device("cpu")
    else:
        model_device = next(model.parameters()).device

    logits_max_abs_diff = 0.0
    logits_abs_sum = 0.0
    logits_numel = 0

    seq_max_abs_diff = 0.0
    seq_abs_sum = 0.0
    seq_numel = 0

    with torch.inference_mode():
        for start in range(0, input_ids.size(0), micro_batch_size):
            end = min(start + micro_batch_size, input_ids.size(0))
            mb_input = input_ids[start:end]
            mb_attn = attention_mask[start:end]
            mb_labels = labels[start:end]

            if not model_has_hf_device_map:
                mb_input = mb_input.to(model_device, non_blocking=True)
                mb_attn = mb_attn.to(model_device, non_blocking=True)
                mb_labels = mb_labels.to(model_device, non_blocking=True)

            logits_chi = model(input_ids=mb_input, attention_mask=mb_attn, use_cache=False).logits
            logits_flow = flow_harness._forward_causal_lm_logits(
                model,
                input_ids=mb_input,
                attention_mask=mb_attn,
            )

            logits_diff = (logits_chi - logits_flow).abs()
            logits_max_abs_diff = max(logits_max_abs_diff, float(logits_diff.max().item()))
            logits_abs_sum += float(logits_diff.sum().item())
            logits_numel += int(logits_diff.numel())

            if mb_labels.device != logits_chi.device:
                mb_labels = mb_labels.to(logits_chi.device, non_blocking=True)

            chi_seq = call_chi_sequence_logprob(
                chi_mod,
                logits=logits_chi,
                labels=mb_labels,
                length_normalize=False,
            )
            flow_seq = flow_mod._sequence_logprob_from_logits(
                logits_flow,
                mb_labels,
                length_normalize=False,
                source=f"parity_check.mb_{start}_{end}",
            )
            seq_diff = (chi_seq - flow_seq).abs()
            seq_max_abs_diff = max(seq_max_abs_diff, float(seq_diff.max().item()))
            seq_abs_sum += float(seq_diff.sum().item())
            seq_numel += int(seq_diff.numel())

            del logits_chi, logits_flow, logits_diff, chi_seq, flow_seq, seq_diff

    logits_mean_abs_diff = logits_abs_sum / float(max(1, logits_numel))
    seq_mean_abs_diff = seq_abs_sum / float(max(1, seq_numel))

    print("=== Logits/Logprob Parity ===")
    print(f"logits_max_abs_diff={logits_max_abs_diff:.10e}")
    print(f"logits_mean_abs_diff={logits_mean_abs_diff:.10e}")
    print(f"seq_logprob_max_abs_diff={seq_max_abs_diff:.10e}")
    print(f"seq_logprob_mean_abs_diff={seq_mean_abs_diff:.10e}")

    preprocess_pass = (encode_helper_match == len(rows)) and (encode_prepare_match == len(rows))
    logits_pass = logits_max_abs_diff <= float(args.logits_atol)
    seq_pass = seq_max_abs_diff <= float(args.seq_logprob_atol)

    print("=== Verdict ===")
    print(f"preprocess_pass={preprocess_pass}")
    print(f"logits_pass={logits_pass} (atol={args.logits_atol})")
    print(f"seq_logprob_pass={seq_pass} (atol={args.seq_logprob_atol})")
    if preprocess_pass and logits_pass and seq_pass:
        print("PASS: final logits generation is the same on the tested samples (within tolerance).")
    else:
        print("FAIL: at least one parity check failed on the tested samples.")


if __name__ == "__main__":
    main()
