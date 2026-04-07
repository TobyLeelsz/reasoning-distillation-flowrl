#!/usr/bin/env python3
"""Direct first-batch test for verl/trainer/ppo/log_ratio_reward.py compute_score."""

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


PROMPT_TOKEN_IDS_KEY = "__verl_prompt_token_ids"
RESPONSE_TOKEN_IDS_KEY = "__verl_response_token_ids"

SYSTEM_PROMPT = "You are a helpful assistant."
USER_PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. The last line of your response "
    "should be of the form Answer: $Answer (without quotes) where $Answer is the "
    "answer to the problem.\n\n"
    "{problem}\n\n"
    'Remember to put your answer on its own line after "Answer:".'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Directly call log_ratio_reward.compute_score on first batch pos/neg data."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("/proj/weitongzlab/projects/rm_training/generate/dapo_17k_merged_qwen3.jsonl"),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument(
        "--reward-script-path",
        type=Path,
        default=Path(
            "/proj/weitongzlab/projects/reasoning-distillation-flowrl/verl_FlowRL/verl/trainer/ppo/log_ratio_reward.py"
        ),
    )
    parser.add_argument(
        "--policy-model-path",
        type=str,
        default="/proj/weitongzlab/projects/rm_training/rm_out/checkpoint-1629",
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
        help="Tokenizer used to build prompt/response token IDs passed to log_ratio_reward.",
    )
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--torch-dtype", type=str, default="bfloat16")
    parser.add_argument("--attn-implementation", type=str, default="flash_attention_2")
    parser.add_argument("--offload-folder", type=str, default="/tmp/verl_reward_offload")
    parser.add_argument("--max-gpu-memory", type=str, default="2GiB")
    parser.add_argument("--max-cpu-memory", type=str, default=None)
    parser.add_argument("--max-length", type=int, default=9216)
    parser.add_argument("--normalize-by-length", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-fast-tokenizer", action="store_true")
    parser.add_argument("--clear-cuda-cache", action="store_true")
    parser.add_argument(
        "--data-source",
        type=str,
        default="warmup",
        help="Use warmup to disable rule reward, so score equals pure log-ratio reward.",
    )
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


def normalize_prompt(prompt_obj: Any) -> str:
    if isinstance(prompt_obj, str):
        return prompt_obj
    if isinstance(prompt_obj, (list, tuple)):
        return "\n".join(str(x) for x in prompt_obj)
    if isinstance(prompt_obj, dict):
        for key in ["prompt", "question", "content", "text"]:
            if key in prompt_obj:
                return normalize_prompt(prompt_obj[key])
        return json.dumps(prompt_obj, ensure_ascii=False)
    return str(prompt_obj)


def load_rows_for_batch(data_path: Path, batch_size: int, batch_index: int) -> tuple[list[dict[str, Any]], int]:
    if batch_size <= 0:
        raise ValueError(f"--batch-size must be > 0, got {batch_size}")
    if batch_index < 0:
        raise ValueError(f"--batch-index must be >= 0, got {batch_index}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    start_valid_idx = batch_index * batch_size
    end_valid_idx = start_valid_idx + batch_size

    rows: list[dict[str, Any]] = []
    valid_idx = 0
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            responses = row.get("responses")
            if not isinstance(responses, list) or len(responses) < 2:
                continue

            if valid_idx < start_valid_idx:
                valid_idx += 1
                continue
            if valid_idx >= end_valid_idx:
                break

            rows.append(row)
            valid_idx += 1

    if len(rows) < batch_size:
        raise ValueError(
            f"Requested batch_index={batch_index}, batch_size={batch_size}, "
            f"but only found {len(rows)} rows in that range."
        )
    return rows, start_valid_idx


def build_prompt_text(tokenizer, prompt_obj: Any) -> str:
    question = normalize_prompt(prompt_obj)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(problem=question)},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def build_inputs(rows: list[dict[str, Any]], tokenizer):
    pos_solution_strs: list[str] = []
    neg_solution_strs: list[str] = []
    pos_extra_infos: list[dict[str, Any]] = []
    neg_extra_infos: list[dict[str, Any]] = []

    for row in rows:
        prompt_text = build_prompt_text(tokenizer, row.get("prompt", ""))
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids

        responses = row["responses"]
        neg_text = str(responses[0]) if responses[0] is not None else ""
        pos_text = str(responses[1]) if responses[1] is not None else ""

        pos_ids = tokenizer(pos_text, add_special_tokens=False).input_ids
        neg_ids = tokenizer(neg_text, add_special_tokens=False).input_ids

        pos_solution_strs.append(pos_text)
        neg_solution_strs.append(neg_text)

        pos_extra_infos.append(
            {
                PROMPT_TOKEN_IDS_KEY: list(prompt_ids),
                RESPONSE_TOKEN_IDS_KEY: list(pos_ids),
            }
        )
        neg_extra_infos.append(
            {
                PROMPT_TOKEN_IDS_KEY: list(prompt_ids),
                RESPONSE_TOKEN_IDS_KEY: list(neg_ids),
            }
        )

    return pos_solution_strs, pos_extra_infos, neg_solution_strs, neg_extra_infos


def metric(values: list[Any], key: str) -> list[float]:
    out: list[float] = []
    for value in values:
        if isinstance(value, dict):
            out.append(float(value.get(key, 0.0)))
        else:
            out.append(float(value))
    return out


def mean(values: list[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def main() -> None:
    args = parse_args()

    reward_mod = load_module(args.reward_script_path, "log_ratio_reward_module")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        use_fast=(not args.no_fast_tokenizer),
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[info] tokenizer_path={args.tokenizer_path} tokenizer_cls={tokenizer.__class__.__name__}", flush=True)

    rows, start_idx = load_rows_for_batch(args.data_path, args.batch_size, args.batch_index)
    pos_solution_strs, pos_extra_infos, neg_solution_strs, neg_extra_infos = build_inputs(rows, tokenizer)

    reward_kwargs = dict(
        policy_model_path=args.policy_model_path,
        reference_model_path=args.reference_model_path,
        tokenizer_path=args.tokenizer_path,
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

    data_sources = [args.data_source] * len(rows)
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

    pos_lrr = metric(pos_results, "log_ratio_reward")
    neg_lrr = metric(neg_results, "log_ratio_reward")
    pos_pi = metric(pos_results, "pi_logp")
    pos_ref = metric(pos_results, "pi_ref_logp")
    neg_pi = metric(neg_results, "pi_logp")
    neg_ref = metric(neg_results, "pi_ref_logp")

    print("idx\treward_pos\treward_neg\tlogp_pi_pos\tlogp_ref_pos\tlogp_pi_neg\tlogp_ref_neg")
    for i in range(len(rows)):
        print(
            f"{i}\t"
            f"{pos_lrr[i]:.8f}\t{neg_lrr[i]:.8f}\t"
            f"{pos_pi[i]:.4f}\t{pos_ref[i]:.4f}\t{neg_pi[i]:.4f}\t{neg_ref[i]:.4f}"
        )

    print(f"batch_index={args.batch_index}")
    print(f"batch_size={len(rows)}")
    print(f"valid_row_start_index={start_idx}")
    print(f"valid_row_end_index={start_idx + len(rows) - 1}")
    print(f"data_source={args.data_source}")
    print(f"policy_model_path={args.policy_model_path}")
    print(f"reference_model_path={args.reference_model_path}")
    print(f"beta={args.beta}")
    print(f"normalize_by_length={int(bool(args.normalize_by_length))}")
    print(f"mean_reward_pos={mean(pos_lrr):.8f}")
    print(f"mean_reward_neg={mean(neg_lrr):.8f}")


if __name__ == "__main__":
    main()
