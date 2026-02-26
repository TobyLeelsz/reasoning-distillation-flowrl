#!/usr/bin/env python3
"""Quick parity-style test for verl/trainer/ppo/log_ratio_reward.py."""

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick test for log_ratio_reward on first-N jsonl pairs.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("/home/azureuser/shangzhe/FlowRL/verl_FlowRL/dapo_17k_merged_new.jsonl"),
    )
    parser.add_argument("--num-pairs", type=int, default=10)
    parser.add_argument(
        "--chi-script-path",
        type=Path,
        default=Path("/home/azureuser/shangzhe/FlowRL/verl_FlowRL/chi_squared_rm.py"),
    )
    parser.add_argument(
        "--reward-script-path",
        type=Path,
        default=Path("/home/azureuser/shangzhe/FlowRL/verl_FlowRL/verl/trainer/ppo/log_ratio_reward.py"),
    )
    parser.add_argument(
        "--policy-model-path",
        type=str,
        default="/home/azureuser/shangzhe/FlowRL/checkpoints/checkpoint-800",
    )
    parser.add_argument(
        "--reference-model-path",
        type=str,
        default="/home/azureuser/shangzhe/FlowRL/downloads/Qwen/Qwen2.5-7B",
    )
    parser.add_argument("--tokenizer-path", type=str, default=None)

    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--torch-dtype", type=str, default="bfloat16")
    parser.add_argument("--attn-implementation", type=str, default="eager")
    parser.add_argument("--offload-folder", type=str, default="/tmp/verl_reward_offload")
    parser.add_argument("--max-gpu-memory", type=str, default="2GiB")
    parser.add_argument("--max-cpu-memory", type=str, default=None)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--normalize-by-length", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-fast-tokenizer", action="store_true")
    parser.add_argument("--clear-cuda-cache", action="store_true")

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


def load_first_rows(path: Path, n: int) -> list[dict[str, Any]]:
    if n <= 0:
        raise ValueError(f"--num-pairs must be > 0, got {n}")
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            if len(rows) >= n:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            responses = row.get("responses")
            if not isinstance(responses, list) or len(responses) < 2:
                raise ValueError(f"Line {line_idx} does not contain at least 2 responses.")
            rows.append(row)
    if len(rows) < n:
        raise ValueError(f"Requested {n} rows but found only {len(rows)}.")
    return rows


def load_tokenizer_with_fallback(
    tokenizer_paths: list[str],
    trust_remote_code: bool,
    prefer_fast: bool,
):
    from transformers import AutoTokenizer

    unique_paths: list[str] = []
    seen = set()
    for path in tokenizer_paths:
        if not path or path in seen:
            continue
        unique_paths.append(path)
        seen.add(path)

    modes = [prefer_fast, not prefer_fast]
    last_error: Exception | None = None
    for tokenizer_path in unique_paths:
        for use_fast in modes:
            try:
                tok = AutoTokenizer.from_pretrained(
                    tokenizer_path,
                    use_fast=use_fast,
                    trust_remote_code=trust_remote_code,
                )
                if tok.pad_token is None and tok.eos_token is not None:
                    tok.pad_token = tok.eos_token
                return tok, tokenizer_path, use_fast
            except Exception as e:  # noqa: BLE001
                last_error = e
                print(
                    f"[warn] tokenizer init failed (path={tokenizer_path}, use_fast={use_fast}): "
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )
    assert last_error is not None
    raise last_error


def split_prompt_response_from_encoding(enc: dict[str, list[int]]) -> tuple[list[int], list[int]]:
    input_ids = enc["input_ids"]
    labels = enc["labels"]
    p_len = 0
    for x in labels:
        if x == -100:
            p_len += 1
        else:
            break
    return input_ids[:p_len], input_ids[p_len:]


def build_inputs(rows: list[dict[str, Any]], tokenizer: Any, chi, args: argparse.Namespace):
    pos_solution_strs: list[str] = []
    neg_solution_strs: list[str] = []
    pos_extra_infos: list[dict[str, Any]] = []
    neg_extra_infos: list[dict[str, Any]] = []

    for row in rows:
        prompt_obj = row.get("prompt", "")
        responses = row["responses"]
        neg_text = str(responses[0])
        solution = responses[1]
        answer = row.get("true_solution", None)

        prompt = chi.format_prompt(tokenizer, prompt_obj)
        pos_text = chi.build_pos_text(
            solution=solution,
            answer=answer,
            use_answer_only_pos=args.use_answer_only_pos,
            append_final_answer=args.append_final_answer,
            final_answer_prefix=args.final_answer_prefix,
            strip_solution_to_answer=args.strip_solution_to_answer,
        )

        pos_enc = chi.encode_prompt_response(tokenizer, prompt, pos_text, args.max_length)
        neg_enc = chi.encode_prompt_response(tokenizer, prompt, neg_text, args.max_length)

        pos_prompt_ids, pos_response_ids = split_prompt_response_from_encoding(pos_enc)
        neg_prompt_ids, neg_response_ids = split_prompt_response_from_encoding(neg_enc)

        pos_extra_infos.append(
            {
                "__verl_prompt_token_ids": pos_prompt_ids,
                "__verl_response_token_ids": pos_response_ids,
            }
        )
        neg_extra_infos.append(
            {
                "__verl_prompt_token_ids": neg_prompt_ids,
                "__verl_response_token_ids": neg_response_ids,
            }
        )
        pos_solution_strs.append(str(pos_text))
        neg_solution_strs.append(neg_text)

    return pos_solution_strs, pos_extra_infos, neg_solution_strs, neg_extra_infos


def main() -> None:
    args = parse_args()
    chi = load_module(args.chi_script_path, "chi_module")
    reward_mod = load_module(args.reward_script_path, "log_ratio_reward_module")

    tokenizer_paths = [
        args.tokenizer_path or args.policy_model_path,
        args.reference_model_path,
    ]
    tokenizer, tok_path, use_fast = load_tokenizer_with_fallback(
        tokenizer_paths=tokenizer_paths,
        trust_remote_code=args.trust_remote_code,
        prefer_fast=(not args.no_fast_tokenizer),
    )
    print(f"[info] tokenizer loaded from path={tok_path}, use_fast={use_fast}", flush=True)

    rows = load_first_rows(args.data_path, args.num_pairs)
    pos_solution_strs, pos_extra_infos, neg_solution_strs, neg_extra_infos = build_inputs(rows, tokenizer, chi, args)

    reward_kwargs = dict(
        policy_model_path=args.policy_model_path,
        reference_model_path=args.reference_model_path,
        tokenizer_path=args.tokenizer_path or tok_path,
        beta=args.beta,
        micro_batch_size=max(1, int(args.micro_batch_size)),
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        offload_folder=args.offload_folder,
        max_gpu_memory=args.max_gpu_memory,
        max_cpu_memory=args.max_cpu_memory,
        trust_remote_code=args.trust_remote_code,
        use_fast_tokenizer=(not args.no_fast_tokenizer),
        max_seq_len=args.max_length,
        normalize_by_length=args.normalize_by_length,
        clear_cuda_cache=args.clear_cuda_cache,
    )

    data_sources = ["math"] * len(rows)
    ground_truths = [row.get("true_solution", None) for row in rows]

    pos_results = reward_mod.compute_score(
        data_sources=data_sources,
        solution_strs=pos_solution_strs,
        ground_truths=ground_truths,
        extra_infos=pos_extra_infos,
        **reward_kwargs,
    )
    neg_results = reward_mod.compute_score(
        data_sources=data_sources,
        solution_strs=neg_solution_strs,
        ground_truths=ground_truths,
        extra_infos=neg_extra_infos,
        **reward_kwargs,
    )

    pos_scores = [float(x["score"]) if isinstance(x, dict) else float(x) for x in pos_results]
    neg_scores = [float(x["score"]) if isinstance(x, dict) else float(x) for x in neg_results]
    pos_minus_neg = [p - n for p, n in zip(pos_scores, neg_scores)]
    pos_better_ratio = sum(1 for x in pos_minus_neg if x > 0.0) / float(len(pos_minus_neg))

    print(f"num_pairs={len(rows)}")
    print(f"mean_pos_reward={sum(pos_scores) / len(pos_scores):.8f}")
    print(f"mean_neg_reward={sum(neg_scores) / len(neg_scores):.8f}")
    print(f"pos_minus_neg_mean={sum(pos_minus_neg) / len(pos_minus_neg):.8f}")
    print(f"pos_better_ratio={pos_better_ratio:.4f}")


if __name__ == "__main__":
    main()
