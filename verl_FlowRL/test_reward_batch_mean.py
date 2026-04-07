#!/usr/bin/env python3
"""Compute positive/negative reward means for one JSONL batch."""

import argparse
import importlib.util
import inspect
import json
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load one batch from paired JSONL and compute positive/negative rewards "
            "with FlowRL log_ratio_reward compute_score."
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("/proj/weitongzlab/projects/rm_training/generate/dapo_17k_merged_qwen3.jsonl"),
        help="Paired JSONL path with keys prompt/responses/true_solution.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of valid pairs in one batch.",
    )
    parser.add_argument(
        "--batch-index",
        type=int,
        default=0,
        help="0-based batch index over valid pairs.",
    )

    parser.add_argument(
        "--chi-script-path",
        type=Path,
        default=Path("/proj/weitongzlab/projects/rm_training/chi_squared_rm.py"),
        help="Path to chi_squared_rm.py (preprocessing source of truth).",
    )
    parser.add_argument(
        "--reward-script-path",
        type=Path,
        default=Path(
            "/proj/weitongzlab/projects/reasoning-distillation-flowrl/verl_FlowRL/verl/trainer/ppo/log_ratio_reward.py"
        ),
        help="Path to FlowRL reward script.",
    )

    parser.add_argument(
        "--policy-model-path",
        type=str,
        default="/proj/weitongzlab/projects/rm_training/rm_out/checkpoint-1629",
        help="Policy RM path (the reward model checkpoint).",
    )
    parser.add_argument(
        "--reference-model-path",
        type=str,
        default="Qwen/Qwen3-1.7B-Base",
        help="Reference model path used by log-ratio reward.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Tokenizer path. Default: policy path, then reference path fallback.",
    )

    parser.add_argument("--beta", type=float, default=0.2)
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

    parser.add_argument(
        "--data-source",
        type=str,
        default="warmup",
        help=(
            "Data source passed to compute_score. "
            "Default warmup disables rule reward so score is pure log-ratio reward."
        ),
    )
    parser.add_argument("--use_answer_only_pos", action="store_true")
    parser.add_argument("--append_final_answer", action="store_true")
    parser.add_argument("--final_answer_prefix", type=str, default="\n\nFinal Answer: ")
    parser.add_argument("--strip_solution_to_answer", action="store_true")
    parser.add_argument(
        "--run-chi-reward",
        action="store_true",
        help=(
            "Also compute reward with chi_squared_rm.py logprob path using the same "
            "already-loaded policy/reference models from log_ratio_reward scorer."
        ),
    )
    parser.add_argument(
        "--chi-length-normalize",
        action="store_true",
        help="Use length_normalize=True when calling chi.sequence_logprob_from_logits (if supported).",
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
        for line_no, line in enumerate(f, start=1):
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
    pos_encoded_items: list[dict[str, list[int]]] = []
    neg_encoded_items: list[dict[str, list[int]]] = []

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
        pos_encoded_items.append(pos_enc)
        neg_encoded_items.append(neg_enc)

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

    return (
        pos_solution_strs,
        pos_extra_infos,
        neg_solution_strs,
        neg_extra_infos,
        pos_encoded_items,
        neg_encoded_items,
    )


def extract_metric(values: list[Any], key: str, fallback_to_score: bool = False) -> list[float]:
    out: list[float] = []
    for v in values:
        if isinstance(v, dict):
            if key in v:
                out.append(float(v[key]))
            elif fallback_to_score and "score" in v:
                out.append(float(v["score"]))
            else:
                out.append(0.0)
        else:
            out.append(float(v))
    return out


def mean(xs: list[float]) -> float:
    return sum(xs) / float(len(xs)) if xs else 0.0


def call_chi_sequence_logprob(
    chi,
    logits: torch.Tensor,
    labels: torch.Tensor,
    length_normalize: bool,
) -> torch.Tensor:
    fn = chi.sequence_logprob_from_logits
    sig = inspect.signature(fn)
    if "length_normalize" in sig.parameters:
        return fn(logits, labels, length_normalize=length_normalize)
    return fn(logits, labels)


def pad_batch_items(items: list[dict[str, list[int]]], pad_token_id: int):
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
    attention_mask = attention_mask.ne(0).to(dtype=torch.long)
    return input_ids, attention_mask, labels


def compute_chi_logps_with_model(
    scorer,
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    micro_batch_size: int,
    chi,
    chi_length_normalize: bool,
) -> torch.Tensor:
    has_hf_device_map = hasattr(model, "hf_device_map")
    model_device = torch.device("cpu")
    if not has_hf_device_map:
        model_device = next(model.parameters()).device

    out_logps: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, input_ids.size(0), micro_batch_size):
            end = min(start + micro_batch_size, input_ids.size(0))
            mb_input_ids = input_ids[start:end]
            mb_attention_mask = attention_mask[start:end]
            mb_labels = labels[start:end]

            if not has_hf_device_map:
                mb_input_ids = mb_input_ids.to(model_device, non_blocking=True)
                mb_attention_mask = mb_attention_mask.to(model_device, non_blocking=True)
                mb_labels = mb_labels.to(model_device, non_blocking=True)

            logits = scorer._forward_causal_lm_logits(
                model,
                input_ids=mb_input_ids,
                attention_mask=mb_attention_mask,
            )
            if mb_labels.device != logits.device:
                mb_labels = mb_labels.to(logits.device, non_blocking=True)
            lp = call_chi_sequence_logprob(
                chi,
                logits=logits,
                labels=mb_labels,
                length_normalize=chi_length_normalize,
            )
            out_logps.append(lp.detach().cpu())
            del logits, lp
    return torch.cat(out_logps, dim=0)


def compute_chi_style_rewards_with_same_models(
    chi,
    reward_mod,
    pos_encoded_items: list[dict[str, list[int]]],
    neg_encoded_items: list[dict[str, list[int]]],
    beta: float,
    micro_batch_size: int,
    chi_length_normalize: bool,
):
    scorer = getattr(reward_mod, "_SCORER", None)
    if scorer is None:
        raise RuntimeError("FlowRL scorer is not initialized. Run compute_score first.")

    pad_token_id = scorer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must have a valid pad_token_id.")

    pos_input_ids, pos_attention_mask, pos_labels = pad_batch_items(pos_encoded_items, pad_token_id=pad_token_id)
    neg_input_ids, neg_attention_mask, neg_labels = pad_batch_items(neg_encoded_items, pad_token_id=pad_token_id)

    mb = max(1, int(micro_batch_size))
    logp_pi_pos = compute_chi_logps_with_model(
        scorer=scorer,
        model=scorer.policy_model,
        input_ids=pos_input_ids,
        attention_mask=pos_attention_mask,
        labels=pos_labels,
        micro_batch_size=mb,
        chi=chi,
        chi_length_normalize=chi_length_normalize,
    )
    logp_ref_pos = compute_chi_logps_with_model(
        scorer=scorer,
        model=scorer.reference_model,
        input_ids=pos_input_ids,
        attention_mask=pos_attention_mask,
        labels=pos_labels,
        micro_batch_size=mb,
        chi=chi,
        chi_length_normalize=chi_length_normalize,
    )
    logp_pi_neg = compute_chi_logps_with_model(
        scorer=scorer,
        model=scorer.policy_model,
        input_ids=neg_input_ids,
        attention_mask=neg_attention_mask,
        labels=neg_labels,
        micro_batch_size=mb,
        chi=chi,
        chi_length_normalize=chi_length_normalize,
    )
    logp_ref_neg = compute_chi_logps_with_model(
        scorer=scorer,
        model=scorer.reference_model,
        input_ids=neg_input_ids,
        attention_mask=neg_attention_mask,
        labels=neg_labels,
        micro_batch_size=mb,
        chi=chi,
        chi_length_normalize=chi_length_normalize,
    )

    beta_f = float(beta)
    pred_pos = beta_f * (logp_pi_pos - logp_ref_pos)
    pred_neg = beta_f * (logp_pi_neg - logp_ref_neg)
    return pred_pos.tolist(), pred_neg.tolist()


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

    rows, start_valid_idx = load_rows_for_batch(
        data_path=args.data_path,
        batch_size=args.batch_size,
        batch_index=args.batch_index,
    )
    (
        pos_solution_strs,
        pos_extra_infos,
        neg_solution_strs,
        neg_extra_infos,
        pos_encoded_items,
        neg_encoded_items,
    ) = build_inputs(rows, tokenizer, chi, args)

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

    pos_score = extract_metric(pos_results, key="score")
    neg_score = extract_metric(neg_results, key="score")
    pos_log_ratio_reward = extract_metric(pos_results, key="log_ratio_reward", fallback_to_score=True)
    neg_log_ratio_reward = extract_metric(neg_results, key="log_ratio_reward", fallback_to_score=True)

    pos_minus_neg_score = [p - n for p, n in zip(pos_score, neg_score)]
    pos_minus_neg_lrr = [p - n for p, n in zip(pos_log_ratio_reward, neg_log_ratio_reward)]
    pos_better_score_ratio = sum(1 for x in pos_minus_neg_score if x > 0.0) / float(len(pos_minus_neg_score))
    pos_better_lrr_ratio = sum(1 for x in pos_minus_neg_lrr if x > 0.0) / float(len(pos_minus_neg_lrr))

    print(f"batch_index={args.batch_index}")
    print(f"batch_size={len(rows)}")
    print(f"valid_row_start_index={start_valid_idx}")
    print(f"valid_row_end_index={start_valid_idx + len(rows) - 1}")
    print(f"data_source={args.data_source}")
    print(f"policy_model_path={args.policy_model_path}")
    print(f"reference_model_path={args.reference_model_path}")
    print(f"beta={args.beta}")
    print(f"normalize_by_length={int(bool(args.normalize_by_length))}")

    print(f"mean_pos_score={mean(pos_score):.8f}")
    print(f"mean_neg_score={mean(neg_score):.8f}")
    print(f"pos_minus_neg_score_mean={mean(pos_minus_neg_score):.8f}")
    print(f"pos_better_score_ratio={pos_better_score_ratio:.4f}")

    print(f"mean_pos_log_ratio_reward={mean(pos_log_ratio_reward):.8f}")
    print(f"mean_neg_log_ratio_reward={mean(neg_log_ratio_reward):.8f}")
    print(f"pos_minus_neg_log_ratio_reward_mean={mean(pos_minus_neg_lrr):.8f}")
    print(f"pos_better_log_ratio_reward_ratio={pos_better_lrr_ratio:.4f}")

    if args.run_chi_reward:
        chi_pos_reward, chi_neg_reward = compute_chi_style_rewards_with_same_models(
            chi=chi,
            reward_mod=reward_mod,
            pos_encoded_items=pos_encoded_items,
            neg_encoded_items=neg_encoded_items,
            beta=args.beta,
            micro_batch_size=args.micro_batch_size,
            chi_length_normalize=args.chi_length_normalize,
        )
        chi_pos_minus_neg = [p - n for p, n in zip(chi_pos_reward, chi_neg_reward)]
        chi_pos_better_ratio = sum(1 for x in chi_pos_minus_neg if x > 0.0) / float(len(chi_pos_minus_neg))

        print(f"chi_length_normalize={int(bool(args.chi_length_normalize))}")
        print(f"chi_mean_pos_reward={mean(chi_pos_reward):.8f}")
        print(f"chi_mean_neg_reward={mean(chi_neg_reward):.8f}")
        print(f"chi_pos_minus_neg_reward_mean={mean(chi_pos_minus_neg):.8f}")
        print(f"chi_pos_better_reward_ratio={chi_pos_better_ratio:.4f}")


if __name__ == "__main__":
    main()
