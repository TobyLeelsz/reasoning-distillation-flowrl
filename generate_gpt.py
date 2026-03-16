import os
import json
import time
import random
import argparse
from typing import Any, Dict, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm
from datasets import load_dataset
from openai import OpenAI

# =====================================================
# Config (recommend moving secrets to env vars)
# =====================================================

ENDPOINT = "https://oai-weitongzlab.openai.azure.com/openai/v1"
DEPLOYMENT_NAME = "gpt-4.1"
API_KEY = os.environ["AZURE_OPENAI_API_KEY"]

DATASET = "trl-lib/DeepMath-103K"
SPLIT = "train"
OUT_JSONL = "out_deepmath_gpt4.1/deepmath_103k_azure_openai.jsonl"

MAX_TOKENS = 8192
TEMPERATURE = 0.2

SYSTEM_PROMPT = """You are a professional mathematician.
Solve the problem carefully and provide a complete solution.
"""

# =====================================================
# Helpers
# =====================================================

def extract_question_text(raw_q: Any) -> str:
    if isinstance(raw_q, str):
        return raw_q

    if isinstance(raw_q, list) and len(raw_q) > 0 and isinstance(raw_q[0], dict):
        for msg in raw_q:
            if isinstance(msg, dict) and msg.get("role") == "user" and isinstance(msg.get("content"), str):
                return msg["content"]
        if isinstance(raw_q[0].get("content"), str):
            return raw_q[0]["content"]

    if isinstance(raw_q, dict) and isinstance(raw_q.get("content"), str):
        return raw_q["content"]

    raise TypeError(f"Unsupported question format: {type(raw_q)} -> {raw_q}")

def load_done_set(out_jsonl: str) -> set:
    done = set()
    if not os.path.exists(out_jsonl):
        return done
    with open(out_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            try:
                ex = json.loads(line)
                # your record format: prompt=[{"content":..., "role":"user"}]
                q = ex["prompt"][0]["content"]
                done.add(q)
            except Exception:
                pass
    return done

# =====================================================
# Worker (runs in subprocess)
# - Create client inside the worker
# - Retry with exponential backoff + jitter
# =====================================================

def _make_client() -> OpenAI:
    if not API_KEY:
        raise RuntimeError(
            "AZURE_OPENAI_API_KEY is not set. "
            "Please export AZURE_OPENAI_API_KEY instead of hardcoding secrets."
        )
    return OpenAI(base_url=ENDPOINT, api_key=API_KEY)

def generate_answer_with_retry(
    question: str,
    max_retries: int,
    base_sleep: float,
    max_sleep: float,
    per_call_sleep: float,
) -> str:
    client = _make_client()

    # small jitter to reduce stampede in multi-proc
    if per_call_sleep > 0:
        time.sleep(per_call_sleep * (0.5 + random.random()))

    last_err: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model=DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            last_err = e
            # exponential backoff with jitter
            sleep_s = min(max_sleep, base_sleep * (2 ** attempt)) * (0.7 + 0.6 * random.random())
            # If rate-limited, backing off is crucial; for other errors also helps.
            time.sleep(sleep_s)

    raise last_err if last_err else RuntimeError("Unknown error in generate_answer_with_retry")

def worker_task(args: Tuple[int, str, Dict[str, Any]]) -> Tuple[int, str, str]:
    """
    Return: (index, question, answer)
    """
    idx, question, cfg = args
    answer = generate_answer_with_retry(
        question=question,
        max_retries=cfg["max_retries"],
        base_sleep=cfg["retry_base_sleep"],
        max_sleep=cfg["retry_max_sleep"],
        per_call_sleep=cfg["per_call_sleep"],
    )
    return idx, question, answer

# =====================================================
# Main
# =====================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    parser.add_argument("--max_rows", type=int, default=-1, help="For debugging: only process first N rows")
    parser.add_argument("--max_retries", type=int, default=6)
    parser.add_argument("--retry_base_sleep", type=float, default=1.0)
    parser.add_argument("--retry_max_sleep", type=float, default=30.0)
    parser.add_argument("--per_call_sleep", type=float, default=0.2, help="Small per-worker jitter sleep before each call")
    args = parser.parse_args()

    ds = load_dataset(DATASET, split=SPLIT)
    if args.max_rows > 0:
        ds = ds.select(range(min(args.max_rows, len(ds))))

    os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)

    done = load_done_set(OUT_JSONL)
    print(f"Already generated: {len(done)}")

    # Build worklist (index used only for debugging / stable identification)
    work_items = []
    for i, ex in enumerate(ds):
        q = extract_question_text(ex["prompt"])
        if q in done:
            continue
        work_items.append((i, q))

    print(f"To generate: {len(work_items)}  | workers={args.workers}")

    cfg = {
        "max_retries": args.max_retries,
        "retry_base_sleep": args.retry_base_sleep,
        "retry_max_sleep": args.retry_max_sleep,
        "per_call_sleep": args.per_call_sleep,
    }

    # Only main process writes to file
    with open(OUT_JSONL, "a", encoding="utf-8") as out:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = []
            for (idx, q) in work_items:
                futures.append(ex.submit(worker_task, (idx, q, cfg)))

            for fut in tqdm(as_completed(futures), total=len(futures)):
                try:
                    idx, question, answer = fut.result()

                    record = {
                        "prompt": [{"content": question, "role": "user"}],
                        "solution": answer,
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out.flush()
                    done.add(question)

                except Exception as e:
                    # You can also log failed questions to a separate file if needed
                    print("Error in future:", repr(e))

if __name__ == "__main__":
    main()
