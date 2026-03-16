#!/usr/bin/env python3
"""
Parallel sampling on multiple GPUs via HuggingFace Accelerate.
Each process (rank) takes a shard of the dataset, generates solutions, and writes a JSONL shard.
Then you merge shards.

Example (4 GPUs):
  accelerate launch --num_processes 4 gen_deepmath_qwen_accel.py \
    --model_id Qwen/Qwen3-4B-Instruct-2507 \
    --dataset trl-lib/DeepMath-103K \
    --split train \
    --out_dir out_deepmath_qwen3 \
    --batch_size 32 \
    --max_new_tokens 1024 \
    --temperature 0.7 --top_p 0.95

Merge:
  python gen_deepmath_qwen_accel.py --merge_only --out_dir out_deepmath_qwen3
"""

import argparse
import glob
import json
import math
import os
from datetime import timedelta
from typing import List, Dict, Any, Optional

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs


def normalize_prompt(p: Any) -> str:
    """
    DeepMath prompts should usually be strings, but some datasets store as list/tuple.
    Keep this small + robust.
    """
    if isinstance(p, str):
        return p
    if isinstance(p, (list, tuple)):
        return "\n".join(str(x) for x in p)
    if isinstance(p, dict):
        # fallback for occasional dict prompts
        for k in ["prompt", "question", "content", "text"]:
            if k in p:
                return normalize_prompt(p[k])
        return json.dumps(p, ensure_ascii=False)
    return str(p)


def load_existing_prompt_set(jsonl_path: str) -> set:
    """
    Load prompts from an existing generated jsonl (merged or shard),
    return a set of normalized prompt strings.
    """
    s = set()
    with open(jsonl_path, "r", encoding="utf-8") as r:
        for line in r:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if "prompt" not in obj:
                continue
            s.add(normalize_prompt(obj["prompt"]).strip())
    return s


def get_first_present_key(example: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in example and example[k] is not None:
            return k
    return None


def filter_missing_prompts(ds, existing_prompts: set, prompt_key: Optional[str] = None):
    """
    Filter a HF dataset to only rows whose prompt (normalized) is NOT in existing_prompts.
    Returns (filtered_dataset, chosen_prompt_key).
    """
    if prompt_key is None:
        # robust candidates for DAPO-like datasets
        cand = ["prompt", "question", "problem", "input", "text", "query", "instruction"]
        prompt_key = get_first_present_key(ds[0], cand)
        if prompt_key is None:
            raise ValueError(f"Cannot infer prompt key. Available columns: {ds.column_names}")

    def _keep(ex):
        p = normalize_prompt(ex[prompt_key]).strip()
        return p not in existing_prompts

    ds2 = ds.filter(_keep)
    return ds2, prompt_key


SYSTEM_PROMPT = "You are a helpful assistant."
USER_PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. The last line of your response "
    "should be of the form Answer: $Answer (without quotes) where $Answer is the "
    "answer to the problem.\n\n"
    "{problem}\n\n"
    "Remember to put your answer on its own line after \"Answer:\"."
)


def build_chat_texts(tokenizer, prompts: List[Any]) -> List[str]:
    has_chat = hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None)
    texts = []
    for p in prompts:
        p = normalize_prompt(p)
        user_prompt = USER_PROMPT_TEMPLATE.format(problem=p)

        if has_chat:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            texts.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        else:
            texts.append(f"System: {SYSTEM_PROMPT}\nUser: {user_prompt}\nAssistant:")

    return texts


@torch.inference_mode()
def generate_batch(
    model,
    tokenizer,
    prompts: List[Any],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    greedy: bool,
) -> List[str]:
    texts = build_chat_texts(tokenizer, prompts)
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    do_sample = not greedy
    gen = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        pad_token_id=tokenizer.eos_token_id,
    )

    input_lens = enc["attention_mask"].sum(dim=1).tolist()
    outs = []
    for i in range(gen.size(0)):
        new_tokens = gen[i, input_lens[i]:]
        outs.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return outs


def merge_shards(out_dir: str, merged_name: str = "deepmath_generated_merged.jsonl") -> str:
    shard_paths = sorted(glob.glob(os.path.join(out_dir, "deepmath_generated_rank*.jsonl")))
    if not shard_paths:
        raise FileNotFoundError(f"No shard files found in {out_dir} (expected deepmath_generated_rank*.jsonl)")

    merged_path = os.path.join(out_dir, merged_name)
    seen = 0
    with open(merged_path, "w", encoding="utf-8") as w:
        for sp in shard_paths:
            with open(sp, "r", encoding="utf-8") as r:
                for line in r:
                    w.write(line)
                    seen += 1
    print(f"[merge] merged {len(shard_paths)} shards, total lines={seen} -> {merged_path}")
    return merged_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="trl-lib/DeepMath-103K")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--model_id", type=str, default="Qwen/Qwen3-0.6B-Instruct")

    ap.add_argument("--out_dir", type=str, default="out_deepmath")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--greedy", action="store_true")

    ap.add_argument("--keep_original_solution", action="store_true")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)

    ap.add_argument("--merge_only", action="store_true")
    ap.add_argument("--merged_name", type=str, default="deepmath_generated_merged.jsonl")
    ap.add_argument("--complement_only", action="store_true",
                    help="Generate only prompts missing from an existing generated JSONL.")
    ap.add_argument("--existing_jsonl", type=str, default="",
                    help="Path to existing generated merged/shard JSONL to avoid duplicates.")
    ap.add_argument("--target_dataset", type=str, default="open-r1/DAPO-Math-17k-Processed",
                    help="Dataset to complement from (default: DAPO-Math-17k-Processed).")
    ap.add_argument("--target_split", type=str, default="train",
                    help="Split for target_dataset.")
    ap.add_argument("--target_prompt_key", type=str, default="",
                    help="Optional explicit prompt key for target_dataset (otherwise auto-detect).")
    ap.add_argument(
        "--dist_timeout_sec",
        type=int,
        default=21600,
        help="Distributed synchronization timeout in seconds (e.g., for wait_for_everyone).",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.merge_only:
        merge_shards(args.out_dir, args.merged_name)
        return

    if args.dist_timeout_sec <= 0:
        raise ValueError("--dist_timeout_sec must be a positive integer")

    pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=args.dist_timeout_sec))
    accelerator = Accelerator(kwargs_handlers=[pg_kwargs])
    rank = accelerator.process_index
    world = accelerator.num_processes

    if args.complement_only:
        if not args.existing_jsonl:
            raise ValueError("--complement_only requires --existing_jsonl")

        existing = load_existing_prompt_set(args.existing_jsonl)
        if rank == 0:
            print(f"[complement] loaded {len(existing)} existing prompts from {args.existing_jsonl}")

        ds = load_dataset(args.target_dataset, split=args.target_split)

        # filter to missing prompts only
        pk = args.target_prompt_key.strip() or None
        ds, pk = filter_missing_prompts(ds, existing, prompt_key=pk)

        total_remaining = len(ds)

        # shard as usual
        ds_shard = ds.shard(
            num_shards=accelerator.num_processes,
            index=accelerator.process_index,
            contiguous=True,
        )

        if accelerator.process_index == 0:
            print(
                f"[complement] Remaining prompts to generate (total): {total_remaining}",
                flush=True,
            )

        print(
            f"[rank{accelerator.process_index}] shard size = {len(ds_shard)}",
            flush=True,
        )

        # standardize to a 'prompt' column expected by the rest of the code
        if pk != "prompt":
            ds = ds.rename_column(pk, "prompt")

        if rank == 0:
            print(f"[complement] target dataset size={len(ds)} after filtering missing prompts")

    else:
        ds = load_dataset(args.dataset, split=args.split)
        if "prompt" not in ds.column_names:
            raise ValueError(f"Dataset missing 'prompt'. Columns: {ds.column_names}")

    # Optional slice
    start = max(args.start, 0)
    end = len(ds) if args.end < 0 else min(args.end, len(ds))
    ds = ds.select(range(start, end))

    # Shard by rank (deterministic partition)
    ds_shard = ds.shard(num_shards=world, index=rank, contiguous=True)
    total = len(ds_shard)
    n_batches = math.ceil(total / args.batch_size)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)

    # IMPORTANT: do NOT use device_map="auto" with multi-process accelerate.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.to(accelerator.device)
    model.eval()

    out_path = os.path.join(args.out_dir, f"deepmath_generated_rank{rank:02d}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for b in range(n_batches):
            lo = b * args.batch_size
            hi = min((b + 1) * args.batch_size, total)
            batch = ds_shard.select(range(lo, hi))

            prompts = batch["prompt"]
            gens = generate_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                greedy=args.greedy,
            )

            for i in range(len(prompts)):
                row: Dict[str, Any] = {
                    "prompt": prompts[i],
                    "solution": gens[i],
                }
                if args.keep_original_solution and "solution" in batch.column_names:
                    row["solution_original"] = batch["solution"][i]
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

            if rank == 0:
                print(f"[rank0] batch {b+1}/{n_batches} done (shard size={total})")

    accelerator.wait_for_everyone()
    if rank == 0:
        print(f"All ranks finished. Shards saved in: {args.out_dir}")
        print(f"Run merge with: python {os.path.basename(__file__)} --merge_only --out_dir {args.out_dir}")


if __name__ == "__main__":
    main()
