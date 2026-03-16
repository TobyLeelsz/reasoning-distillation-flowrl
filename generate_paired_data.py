#!/usr/bin/env python3
# build_pairs.py
import argparse
import json
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"JSON parse error in {path}:{line_no}: {e}") from e

_ws_re = re.compile(r"\s+")
def normalize_question(q: str) -> str:
    # Light normalization: collapse whitespace and strip.
    # (Avoid aggressive LaTeX edits; DeepMath questions often contain LaTeX.)
    return _ws_re.sub(" ", q).strip()

def extract_question_from_generated(ex: Dict[str, Any]) -> str:
    # Your format: {"prompt":[{"content": "...", "role":"user"}], "solution": "..."}
    prompt = ex.get("prompt")
    if isinstance(prompt, list) and len(prompt) > 0 and isinstance(prompt[0], dict):
        content = prompt[0].get("content")
        if isinstance(content, str) and content.strip():
            return content
    # Fallbacks if your schema varies
    for k in ["question", "query", "problem", "input"]:
        v = ex.get(k)
        if isinstance(v, str) and v.strip():
            return v
    raise KeyError("Could not find question text in generated example (expected prompt[0].content or a known fallback key).")

def extract_generated_answer(ex: Dict[str, Any]) -> str:
    for k in ["solution", "generated", "answer", "output", "response"]:
        v = ex.get(k)
        if isinstance(v, str):
            return v
    raise KeyError("Could not find generated answer (expected 'solution' or a known fallback key).")

def extract_question(ex, question_keys):
    for qk in question_keys:
        if qk not in ex:
            continue

        v = ex[qk]

        # case 1: string
        if isinstance(v, str) and v.strip():
            return v

        # case 2: chat-style list
        if isinstance(v, list):
            for m in v:
                if isinstance(m, dict) and m.get("role") == "user":
                    c = m.get("content")
                    if isinstance(c, str) and c.strip():
                        return c
    return None

def build_groundtruth_index(
    hf_dataset_name: str,
    hf_split: str,
    question_keys: List[str],
    answer_keys: List[str],
    id_key: Optional[str] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Returns:
      - by_qnorm: map normalized_question -> example (first occurrence)
      - by_id:    map id -> example (if id_key exists)
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as e:
        raise RuntimeError("Please install datasets: pip install datasets") from e

    ds = load_dataset(hf_dataset_name, split=hf_split)

    by_qnorm: Dict[str, Dict[str, Any]] = {}
    by_id: Dict[str, Dict[str, Any]] = {}

    def extract_question(ex: Dict[str, Any]) -> Optional[str]:
        """
        Supports:
          - string question
          - chat-style prompt: list[{"role": ..., "content": ...}]
        """
        for qk in question_keys:
            if qk not in ex:
                continue

            v = ex[qk]

            # Case 1: plain string
            if isinstance(v, str) and v.strip():
                return v.strip()

            # Case 2: chat-style list
            if isinstance(v, list):
                for m in v:
                    if (
                        isinstance(m, dict)
                        and m.get("role") == "user"
                        and isinstance(m.get("content"), str)
                        and m["content"].strip()
                    ):
                        return m["content"].strip()
        return None

    def extract_answer(ex: Dict[str, Any]) -> Optional[str]:
        """
        Answer must be a non-empty string.
        """
        for ak in answer_keys:
            if ak in ex and isinstance(ex[ak], str) and ex[ak].strip():
                return ex[ak].strip()
        return None

    for ex in ds:
        q = extract_question(ex)
        if q is None:
            continue

        a = extract_answer(ex)
        if a is None:
            continue  # keep strict: only index examples with answers

        qn = normalize_question(q)
        if qn not in by_qnorm:
            by_qnorm[qn] = ex

        if id_key and id_key in ex:
            ex_id = ex[id_key]
            if isinstance(ex_id, str) and ex_id and ex_id not in by_id:
                by_id[ex_id] = ex

    return by_qnorm, by_id

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated_jsonl", required=True, help="Your generated JSONL file.")
    ap.add_argument("--out_jsonl", required=True, help="Where to write paired JSONL.")

    # DeepMath dataset options (edit these if needed)
    ap.add_argument("--hf_dataset", default="trl-lib/DeepMath-103K", help="HF dataset name for DeepMath-103K.")
    ap.add_argument("--hf_split", default="train", help="Which split to load.")

    # Which keys DeepMath uses for question/answer (try common ones)
    ap.add_argument("--question_keys", nargs="+", default=["question", "prompt", "problem", "query", "input"],
                    help="Candidate question field names in DeepMath examples.")
    ap.add_argument("--answer_keys", nargs="+", default=["answer", "solution", "final", "output", "target"],
                    help="Candidate answer field names in DeepMath examples.")

    # If both datasets share an ID, set these
    ap.add_argument("--use_id_join", action="store_true", help="Join on IDs instead of question text.")
    ap.add_argument("--generated_id_key", default="id", help="ID field name in your generated JSONL (if any).")
    ap.add_argument("--deepmath_id_key", default="id", help="ID field name in DeepMath examples (if any).")

    ap.add_argument("--skip_unmatched", default=False, help="If set, drop examples without a ground truth match.")
    args = ap.parse_args()

    by_qnorm, by_id = build_groundtruth_index(
    hf_dataset_name="trl-lib/DeepMath-103K",
    hf_split="train",
    question_keys=["prompt"],
    answer_keys=["solution", "answer"],
    id_key=None,
)

    total = 0
    matched = 0
    unmatched = 0

    with open(args.out_jsonl, "w", encoding="utf-8") as out_f:
        for gex in read_jsonl(args.generated_jsonl):
            total += 1
            q = extract_question_from_generated(gex)
            gen_ans = extract_generated_answer(gex)

            gt_ex = None
            if args.use_id_join:
                gid = gex.get(args.generated_id_key)
                if isinstance(gid, str) and gid:
                    gt_ex = by_id.get(gid)
            else:
                gt_ex = by_qnorm.get(normalize_question(q))
            # import pdb; pdb.set_trace()
            if gt_ex is None:
                unmatched += 1
                if args.skip_unmatched:
                    continue
                # Keep but with null ground truth
                record = {"question": q, "pair": [gen_ans, None]}
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue

            # Extract ground truth answer
            gt_ans = None
            for ak in args.answer_keys:
                v = gt_ex.get(ak)
                if isinstance(v, str):
                    gt_ans = v
                    break

            matched += 1
            record = {"question": q, "pair": [gen_ans, gt_ans]}
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Done.\nTotal: {total}\nMatched: {matched}\nUnmatched: {unmatched}\nOutput: {args.out_jsonl}")

if __name__ == "__main__":
    main()
