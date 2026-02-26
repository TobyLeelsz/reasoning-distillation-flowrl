#!/usr/bin/env python3
"""Strict chi_squared_rm-style loss check on first-N JSONL pairs."""

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clone chi_squared_rm.py preprocessing + loss and print mean loss/pos/neg reward."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("/home/azureuser/shangzhe/FlowRL/verl_FlowRL/dapo_17k_merged_new.jsonl"),
        help="Path to JSONL with keys: prompt, responses[neg,pos], true_solution(optional).",
    )
    parser.add_argument("--num-pairs", type=int, default=10, help="Number of leading data pairs to evaluate.")
    parser.add_argument(
        "--chi-script-path",
        type=Path,
        default=Path("/home/azureuser/shangzhe/FlowRL/verl_FlowRL/chi_squared_rm.py"),
        help="Path to chi_squared_rm.py (used as source of truth for preprocessing/loss primitives).",
    )

    parser.add_argument("--model-name-or-path", type=str, default="/home/azureuser/shangzhe/FlowRL/checkpoints/checkpoint-800")
    parser.add_argument("--ref-model-name-or-path", type=str, default="/home/azureuser/shangzhe/FlowRL/downloads/Qwen/Qwen2.5-7B")
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Tokenizer path. Default: model-name-or-path, then fallback to ref-model-name-or-path.",
    )
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--r-max", type=float, default=1.0)
    parser.add_argument("--r-min", type=float, default=-1.0)
    parser.add_argument("--reg-coef", type=float, default=0.005)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--attn-implementation", type=str, default="sdpa")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-fast-tokenizer", action="store_true")

    # Keep these aligned with chi_squared_rm.py CLI defaults.
    parser.add_argument("--use_answer_only_pos", action="store_true")
    parser.add_argument("--append_final_answer", action="store_true")
    parser.add_argument("--final_answer_prefix", type=str, default="\n\nFinal Answer: ")
    parser.add_argument("--strip_solution_to_answer", action="store_true")
    return parser.parse_args()


def load_chi_module(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"chi_squared_rm.py not found: {path}")
    spec = importlib.util.spec_from_file_location("chi_squared_rm_module", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to import module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = [
        "format_prompt",
        "build_pos_text",
        "encode_prompt_response",
        "sequence_logprob_from_logits",
    ]
    for name in required:
        if not hasattr(module, name):
            raise AttributeError(f"Missing required symbol '{name}' in {path}")
    return module


def load_first_rows(data_path: Path, num_pairs: int) -> list[dict[str, Any]]:
    if num_pairs <= 0:
        raise ValueError(f"--num-pairs must be > 0, got {num_pairs}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    rows: list[dict[str, Any]] = []
    with data_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            if len(rows) >= num_pairs:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            responses = row.get("responses")
            if not isinstance(responses, list) or len(responses) < 2:
                raise ValueError(f"Line {line_idx} does not contain at least 2 responses.")
            rows.append(row)
    if len(rows) < num_pairs:
        raise ValueError(f"Requested {num_pairs} pairs but only found {len(rows)} valid rows.")
    return rows


def load_tokenizer_with_fallback(
    tokenizer_paths: list[str],
    trust_remote_code: bool,
    prefer_fast: bool,
):
    unique_paths: list[str] = []
    seen = set()
    for path in tokenizer_paths:
        if not path or path in seen:
            continue
        unique_paths.append(path)
        seen.add(path)

    modes = [prefer_fast]
    if not prefer_fast:
        modes.append(True)
    else:
        modes.append(False)

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


def load_model(model_path: str, args: argparse.Namespace, device: torch.device):
    model_kwargs: dict[str, Any] = {
        "attn_implementation": args.attn_implementation,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.bf16:
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif args.fp16:
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.config.use_cache = False
    return model


def build_encoded_pairs(rows: list[dict[str, Any]], tokenizer: Any, chi, args: argparse.Namespace):
    pos_items: list[dict[str, list[int]]] = []
    neg_items: list[dict[str, list[int]]] = []

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
        neg_text = str(neg_text)

        pos_enc = chi.encode_prompt_response(tokenizer, prompt, pos_text, args.max_length)
        neg_enc = chi.encode_prompt_response(tokenizer, prompt, neg_text, args.max_length)
        pos_items.append(pos_enc)
        neg_items.append(neg_enc)

    return pos_items, neg_items


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


def seq_logprob(chi, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if labels.device != logits.device:
        labels = labels.to(logits.device, non_blocking=True)
    return chi.sequence_logprob_from_logits(logits, labels)


def compute_logps(model, items: list[dict[str, list[int]]], pad_token_id: int, micro_batch_size: int, chi):
    input_ids, attention_mask, labels = pad_batch(items, pad_token_id)
    model_device = next(model.parameters()).device
    out_logps: list[torch.Tensor] = []

    with torch.inference_mode():
        for start in range(0, input_ids.size(0), micro_batch_size):
            end = min(start + micro_batch_size, input_ids.size(0))
            mb_input_ids = input_ids[start:end].to(model_device, non_blocking=True)
            mb_attention_mask = attention_mask[start:end].to(model_device, non_blocking=True)
            mb_labels = labels[start:end].to(model_device, non_blocking=True)

            logits = model(input_ids=mb_input_ids, attention_mask=mb_attention_mask, use_cache=False).logits
            lp = seq_logprob(chi, logits, mb_labels)
            out_logps.append(lp.detach().cpu())
            del logits, lp

    return torch.cat(out_logps, dim=0)


def main() -> None:
    args = parse_args()
    chi = load_chi_module(args.chi_script_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer_paths = [
        args.tokenizer_path or args.model_name_or_path,
        args.ref_model_name_or_path,
    ]
    tokenizer, tokenizer_path, use_fast = load_tokenizer_with_fallback(
        tokenizer_paths=tokenizer_paths,
        trust_remote_code=args.trust_remote_code,
        prefer_fast=(not args.no_fast_tokenizer),
    )
    print(f"[info] tokenizer loaded from path={tokenizer_path}, use_fast={use_fast}", flush=True)

    rows = load_first_rows(args.data_path, args.num_pairs)
    pos_items, neg_items = build_encoded_pairs(rows, tokenizer, chi, args)

    policy_model = load_model(args.model_name_or_path, args, device)
    ref_model = load_model(args.ref_model_name_or_path, args, device)

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must have a pad_token_id.")

    micro_batch_size = max(1, int(args.micro_batch_size))
    logp_pi_pos = compute_logps(policy_model, pos_items, pad_token_id, micro_batch_size, chi)
    logp_pi_neg = compute_logps(policy_model, neg_items, pad_token_id, micro_batch_size, chi)
    logp_ref_pos = compute_logps(ref_model, pos_items, pad_token_id, micro_batch_size, chi)
    logp_ref_neg = compute_logps(ref_model, neg_items, pad_token_id, micro_batch_size, chi)

    lr_pos = logp_pi_pos - logp_ref_pos
    lr_neg = logp_pi_neg - logp_ref_neg
    pred_pos = args.beta * lr_pos
    pred_neg = args.beta * lr_neg

    loss_pos = 0.5 * (pred_pos - args.r_max).pow(2)
    loss_neg = 0.5 * (pred_neg - args.r_min).pow(2)
    loss_reg = args.reg_coef * (pred_pos.pow(2) + pred_neg.pow(2))
    loss = loss_pos + loss_neg + loss_reg

    print(f"num_pairs={args.num_pairs}")
    print(f"mean_loss={float(loss.mean().item()):.8f}")
    print(f"mean_pos_reward={float(pred_pos.mean().item()):.8f}")
    print(f"mean_neg_reward={float(pred_neg.mean().item()):.8f}")


if __name__ == "__main__":
    main()
