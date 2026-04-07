#!/usr/bin/env python3
"""Compute batch mean rewards using chi_squared_rm.py implementation directly."""

import argparse
import importlib.util
import inspect
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use /proj/weitongzlab/projects/rm_training/chi_squared_rm.py "
            "implementation directly to compute batch mean rewards."
        )
    )
    parser.add_argument(
        "--chi-script-path",
        type=Path,
        default=Path("/proj/weitongzlab/projects/rm_training/chi_squared_rm.py"),
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("/proj/weitongzlab/projects/rm_training/generate/dapo_17k_merged_qwen3.jsonl"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batch-index", type=int, default=0)

    parser.add_argument(
        "--policy-model-path",
        type=str,
        default="/proj/weitongzlab/projects/rm_training/rm_out/checkpoint-1600",
    )
    parser.add_argument(
        "--reference-model-path",
        type=str,
        default="Qwen/Qwen3-1.7B-Base",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="Qwen/Qwen3-1.7B-Base",
        help=(
            "Tokenizer path used for chi preprocessing. "
            "To match chi training setup, keep this as Qwen/Qwen3-1.7B-Base."
        ),
    )

    parser.add_argument("--beta", type=float, default=0.001, help="Matches chi_squared_rm.py training default.")
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--torch-dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--attn-implementation", type=str, default="eager")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-fast-tokenizer", action="store_true")

    parser.add_argument("--chi-length-normalize", action="store_true")
    parser.add_argument("--use_answer_only_pos", action="store_true")
    parser.add_argument("--append_final_answer", action="store_true")
    parser.add_argument("--final_answer_prefix", type=str, default="\n\nFinal Answer: ")
    parser.add_argument("--strip_solution_to_answer", action="store_true")
    return parser.parse_args()


def load_module(path: Path, module_name: str):
    if not path.exists():
        raise FileNotFoundError(f"Module not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to import module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_rows_for_batch(data_path: Path, batch_size: int, batch_index: int) -> tuple[list[dict[str, Any]], int]:
    if batch_size <= 0:
        raise ValueError(f"--batch-size must be > 0, got {batch_size}")
    if batch_index < 0:
        raise ValueError(f"--batch-index must be >= 0, got {batch_index}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Keep dataset ingestion aligned with chi_squared_rm.py:
    # ds = load_dataset("json", data_files=args.paired_jsonl, split="train")
    ds = load_dataset("json", data_files=str(data_path), split="train")
    start_idx = batch_index * batch_size
    end_idx = start_idx + batch_size
    if end_idx > len(ds):
        raise ValueError(
            f"Requested batch_index={batch_index}, batch_size={batch_size}, "
            f"but dataset size is {len(ds)}."
        )
    rows = [ds[i] for i in range(start_idx, end_idx)]
    return rows, start_idx


def load_tokenizer_chi_aligned(tokenizer_path: str):
    # Keep tokenizer loading aligned with chi_squared_rm.py:
    # tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    tok = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda requested but CUDA is unavailable.")
    return torch.device(device_arg)


def to_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    return mapping[dtype_name]


def load_model(
    model_path: str,
    device: torch.device,
    dtype: torch.dtype,
    attn_implementation: str,
    trust_remote_code: bool,
):
    model_dtype = dtype
    if device.type == "cpu" and model_dtype == torch.float16:
        model_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=model_dtype,
        attn_implementation=attn_implementation,
        trust_remote_code=trust_remote_code,
    )
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model


def call_chi_sequence_logprob(chi, logits: torch.Tensor, labels: torch.Tensor, length_normalize: bool) -> torch.Tensor:
    fn = chi.sequence_logprob_from_logits
    sig = inspect.signature(fn)
    if "length_normalize" in sig.parameters:
        return fn(logits, labels, length_normalize=length_normalize)
    return fn(logits, labels)


def compute_logps(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    micro_batch_size: int,
    chi,
    length_normalize: bool,
) -> torch.Tensor:
    out: list[torch.Tensor] = []
    model_device = next(model.parameters()).device
    with torch.inference_mode():
        for start in range(0, input_ids.size(0), micro_batch_size):
            end = min(start + micro_batch_size, input_ids.size(0))
            mb_input = input_ids[start:end].to(model_device, non_blocking=True)
            mb_attn = attention_mask[start:end].to(model_device, non_blocking=True)
            mb_labels = labels[start:end].to(model_device, non_blocking=True)

            logits = model(input_ids=mb_input, attention_mask=mb_attn, use_cache=False).logits
            lp = call_chi_sequence_logprob(chi, logits, mb_labels, length_normalize=length_normalize)
            out.append(lp.detach().cpu())
            del logits, lp
    return torch.cat(out, dim=0)


def mean(xs: list[float]) -> float:
    return sum(xs) / float(len(xs)) if xs else 0.0


def main() -> None:
    args = parse_args()
    chi = load_module(args.chi_script_path, "chi_squared_rm_module")

    rows, start_valid_idx = load_rows_for_batch(args.data_path, args.batch_size, args.batch_index)

    tok_path = args.tokenizer_path
    tokenizer = load_tokenizer_chi_aligned(tok_path)
    print(f"[info] tokenizer loaded from path={tok_path}, use_fast=True", flush=True)

    collator = chi.PairwiseCollatorNew(
        tokenizer=tokenizer,
        max_length=args.max_length,
        use_answer_only_pos=args.use_answer_only_pos,
        append_final_answer=args.append_final_answer,
        final_answer_prefix=args.final_answer_prefix,
        strip_solution_to_answer=args.strip_solution_to_answer,
    )
    batch = collator(rows)

    device = resolve_device(args.device)
    dtype = to_dtype(args.torch_dtype)
    policy_model = load_model(
        args.policy_model_path,
        device=device,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )
    ref_model = load_model(
        args.reference_model_path,
        device=device,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )

    mb = max(1, int(args.micro_batch_size))

    pos_input_ids = batch["pos_input_ids"]
    pos_attention_mask = batch["pos_attention_mask"]
    pos_labels = batch["pos_labels"]
    neg_input_ids = batch["neg_input_ids"]
    neg_attention_mask = batch["neg_attention_mask"]
    neg_labels = batch["neg_labels"]

    logp_pi_pos = compute_logps(
        policy_model,
        pos_input_ids,
        pos_attention_mask,
        pos_labels,
        mb,
        chi,
        length_normalize=args.chi_length_normalize,
    )
    logp_pi_neg = compute_logps(
        policy_model,
        neg_input_ids,
        neg_attention_mask,
        neg_labels,
        mb,
        chi,
        length_normalize=args.chi_length_normalize,
    )
    logp_ref_pos = compute_logps(
        ref_model,
        pos_input_ids,
        pos_attention_mask,
        pos_labels,
        mb,
        chi,
        length_normalize=args.chi_length_normalize,
    )
    logp_ref_neg = compute_logps(
        ref_model,
        neg_input_ids,
        neg_attention_mask,
        neg_labels,
        mb,
        chi,
        length_normalize=args.chi_length_normalize,
    )

    beta = float(args.beta)
    pred_pos = (beta * (logp_pi_pos - logp_ref_pos)).tolist()
    pred_neg = (beta * (logp_pi_neg - logp_ref_neg)).tolist()
    pos_minus_neg = [p - n for p, n in zip(pred_pos, pred_neg)]
    pos_better_ratio = sum(1 for x in pos_minus_neg if x > 0.0) / float(len(pos_minus_neg))

    print(f"batch_index={args.batch_index}")
    print(f"batch_size={len(rows)}")
    print(f"valid_row_start_index={start_valid_idx}")
    print(f"valid_row_end_index={start_valid_idx + len(rows) - 1}")
    print(f"policy_model_path={args.policy_model_path}")
    print(f"reference_model_path={args.reference_model_path}")
    print(f"beta={beta}")
    print(f"chi_length_normalize={int(bool(args.chi_length_normalize))}")
    print(f"mean_pos_reward={mean(pred_pos):.8f}")
    print(f"mean_neg_reward={mean(pred_neg):.8f}")
    print(f"pos_minus_neg_mean={mean(pos_minus_neg):.8f}")
    print(f"pos_better_ratio={pos_better_ratio:.4f}")


if __name__ == "__main__":
    main()
