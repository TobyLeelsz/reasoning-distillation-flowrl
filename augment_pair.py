import json
import re
from collections import defaultdict

PAIR_FILE = "paired.jsonl"
DEEPMATH_FILE = "out_dapo_gpt_4.1_new/deepmath_103k_azure_openai.jsonl"
OUT_FILE = "pair_matched.jsonl"
UNMATCHED_FILE = "pair_unmatched.jsonl"
DUPLICATE_FILE = "deepmath_duplicate_keys.jsonl"

def normalize_q(s: str) -> str:
    """
    轻量归一化：用于“同题目”匹配。
    你可以按需要更激进/更保守。
    """
    if s is None:
        return ""

    # 统一换行
    s = s.replace("\r\n", "\n").replace("\r", "\n")

    # 去掉 latex inline 数学定界符（可选；如果你担心影响匹配，可以注释掉这两行）
    s = s.replace("\\(", "").replace("\\)", "")

    # 去掉多余空白：把所有空白压成单个空格
    s = re.sub(r"\s+", " ", s).strip()

    return s

# 1) 建 deepmath: norm_prompt -> list[solution]
deepmath_map = defaultdict(list)

with open(DEEPMATH_FILE, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        prompt = obj.get("prompt", [])
        if not prompt:
            continue
        # 取第一条 user content 作为题目文本
        prompt_text = prompt[0].get("content", "")
        key = normalize_q(prompt_text)
        deepmath_map[key].append(obj.get("solution", ""))

# 2) 处理 deepmath 重复题目（同 key 多条 solution）
duplicates = {k: v for k, v in deepmath_map.items() if len(v) > 1}
if duplicates:
    with open(DUPLICATE_FILE, "w", encoding="utf-8") as fdup:
        for k, sols in duplicates.items():
            fdup.write(json.dumps({"normalized_question": k, "num": len(sols)}, ensure_ascii=False) + "\n")
    print(f"[Warn] Found {len(duplicates)} duplicate prompt keys in deepmath. Saved summary to {DUPLICATE_FILE}")

# 3) 逐条读 pair.jsonl，匹配并替换
matched = 0
unmatched = 0
used_duplicate = 0  # 统计遇到重复 key 时用了第几条（默认第1条）

with open(PAIR_FILE, "r", encoding="utf-8") as fin, \
     open(OUT_FILE, "w", encoding="utf-8") as fout, \
     open(UNMATCHED_FILE, "w", encoding="utf-8") as funm:

    for line in fin:
        ex = json.loads(line)

        q = ex.get("question", "")
        pair = ex.get("pair", [])
        if not isinstance(pair, list) or len(pair) < 2:
            # 结构异常，直接算 unmatched
            funm.write(json.dumps({"reason": "bad_pair_format", "example": ex}, ensure_ascii=False) + "\n")
            unmatched += 1
            continue

        model_response = pair[0]
        original_answer = pair[1]

        key = normalize_q(q)
        sols = deepmath_map.get(key, [])

        if not sols:
            # 没匹配到
            funm.write(json.dumps({"reason": "no_match", "question": q, "normalized": key}, ensure_ascii=False) + "\n")
            unmatched += 1
            continue

        # 匹配到：如果 duplicates，默认取第1条（你也可以改成轮询/按其它字段区分）
        if len(sols) > 1:
            used_duplicate += 1
        solution = sols[0]

        new_ex = {
            "question": q,
            "pair": [model_response, solution],
            "answer": original_answer
        }
        fout.write(json.dumps(new_ex, ensure_ascii=False) + "\n")
        matched += 1

print(f"Done. matched={matched}, unmatched={unmatched}, duplicate_used={used_duplicate}")
print(f"Output: {OUT_FILE}")
print(f"Unmatched log: {UNMATCHED_FILE}")