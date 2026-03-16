#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge:
- dapo_17k_final.jsonl : {"prompt": <string>, "solution": <model_response>}
- dapo_17k_gpt.jsonl   : {"prompt": <list[{"role","content"}]>, "solution": <model_response>}

into:
{"prompt": <canonical_prompt_from_processed>, "responses": [resp1, resp2], "true_solution": <processed_solution>}

Canonical prompt/true_solution source:
  open-r1/DAPO-Math-17k-Processed (HF datasets)  :contentReference[oaicite:1]{index=1}
"""

import argparse
import json
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out

def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def is_chat_prompt(p: Any) -> bool:
    return isinstance(p, list) and (len(p) == 0 or isinstance(p[0], dict))

def extract_user_content_from_chat(prompt: List[Dict[str, Any]]) -> str:
    """
    Extract a stable "user content" key from a chat-style prompt.
    In open-r1/DAPO-Math-17k-Processed, prompt commonly looks like:
      [{"content": "...", "role": "user"}]
    We take the *first user message* (or first message if no role found).
    """
    if not prompt:
        return ""
    for m in prompt:
        if isinstance(m, dict) and m.get("role") == "user":
            return str(m.get("content", ""))
    # fallback: first message content
    m0 = prompt[0]
    if isinstance(m0, dict):
        return str(m0.get("content", ""))
    return str(m0)

_whitespace_re = re.compile(r"[ \t\r]+")

def normalize_key(text: str) -> str:
    """
    Normalize prompt text into a matching key:
    - unify line endings
    - collapse runs of spaces/tabs
    - strip outer whitespace
    """
    if text is None:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _whitespace_re.sub(" ", text)
    # keep newlines (often meaningful), but strip trailing spaces on each line
    text = "\n".join([ln.strip() for ln in text.split("\n")])
    return text.strip()

def prompt_to_key(prompt_field: Any) -> str:
    if isinstance(prompt_field, str):
        return normalize_key(prompt_field)
    if is_chat_prompt(prompt_field):
        return normalize_key(extract_user_content_from_chat(prompt_field))
    # unknown type
    return normalize_key(str(prompt_field))

def load_processed_dataset(split: str, subset: Optional[str]):
    """
    Load open-r1/DAPO-Math-17k-Processed. The dataset has multiple configs
    in some mirrors/usages (e.g., 'en'), but the safest is to try with subset then fallback.
    """
    from datasets import load_dataset

    ds_name = "open-r1/DAPO-Math-17k-Processed"
    if subset:
        return load_dataset(ds_name, subset, split=split)
    # If no subset specified, try default
    return load_dataset(ds_name, split=split)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final_jsonl", type=str, required=True, help="dapo_17k_final.jsonl")
    ap.add_argument("--gpt_jsonl", type=str, required=True, help="dapo_17k_gpt.jsonl")
    ap.add_argument("--out_jsonl", type=str, required=True, help="merged output jsonl")
    ap.add_argument("--processed_split", type=str, default="train", help="HF split for processed dataset")
    ap.add_argument("--processed_subset", type=str, default=None,
                    help="HF config/subset (often 'en'); if unsure, leave empty and try later.")
    ap.add_argument("--require_both", action="store_true",
                    help="Only keep prompts that have BOTH response_1 and response_2.")
    ap.add_argument("--missing_report", type=str, default=None,
                    help="Optional path to write missing/unmatched keys report as JSON.")
    args = ap.parse_args()

    # 1) Load local JSONLs
    final_rows = read_jsonl(args.final_jsonl)
    gpt_rows = read_jsonl(args.gpt_jsonl)

    # Build maps: key -> response
    final_map: Dict[str, str] = {}
    for r in final_rows:
        k = prompt_to_key(r.get("prompt"))
        # response_1 is the "solution" field from final
        resp = r.get("solution", "")
        if k and resp:
            # If duplicates exist, keep first occurrence
            final_map.setdefault(k, resp)

    gpt_map: Dict[str, str] = {}
    for r in gpt_rows:
        k = prompt_to_key(r.get("prompt"))
        resp = r.get("solution", "")
        if k and resp:
            gpt_map.setdefault(k, resp)

    # 2) Load canonical prompts + true solutions
    processed = load_processed_dataset(args.processed_split, args.processed_subset)

    # Determine which field in processed dataset is the "true solution"
    # Many viewers show 'solution' exists; some also have 'ground_truth'.
    cols = set(processed.column_names)
    if "solution" in cols:
        true_field = "solution"
    elif "ground_truth" in cols:
        true_field = "ground_truth"
    else:
        raise RuntimeError(f"Processed dataset missing expected fields. columns={processed.column_names}")

    # 3) Merge by key based on processed prompts (canonical)
    merged: List[Dict[str, Any]] = []
    missing = {
        "no_final_response": 0,
        "no_gpt_response": 0,
        "kept": 0,
        "processed_total": len(processed),
        "examples_missing_final": [],
        "examples_missing_gpt": [],
    }

    # Keep the canonical prompt as processed["prompt"] (chat-format)
    for ex in processed:
        canonical_prompt = ex["prompt"]
        key = prompt_to_key(canonical_prompt)

        r1 = final_map.get(key)
        r2 = gpt_map.get(key)

        if r1 is None:
            missing["no_final_response"] += 1
            if len(missing["examples_missing_final"]) < 20:
                missing["examples_missing_final"].append(key[:300])
        if r2 is None:
            missing["no_gpt_response"] += 1
            if len(missing["examples_missing_gpt"]) < 20:
                missing["examples_missing_gpt"].append(key[:300])

        if args.require_both and (r1 is None or r2 is None):
            continue

        responses = []
        if r1 is not None:
            responses.append(r1)
        if r2 is not None:
            responses.append(r2)

        # If require_both is False, we still want at least one response.
        if not responses:
            continue

        merged.append({
            "prompt": canonical_prompt,
            "responses": responses,
            "true_solution": ex[true_field],
        })
        missing["kept"] += 1

    # 4) Write outputs
    write_jsonl(args.out_jsonl, merged)

    if args.missing_report:
        with open(args.missing_report, "w", encoding="utf-8") as f:
            json.dump(missing, f, ensure_ascii=False, indent=2)

    # 5) Print a brief summary
    print("=== Merge summary ===")
    print(f"processed_total      : {missing['processed_total']}")
    print(f"kept                : {missing['kept']}")
    print(f"missing final resp  : {missing['no_final_response']}")
    print(f"missing gpt resp    : {missing['no_gpt_response']}")
    print(f"true_solution_field : {true_field}")
    print(f"output              : {args.out_jsonl}")
    if args.missing_report:
        print(f"missing_report       : {args.missing_report}")

if __name__ == "__main__":
    main()
