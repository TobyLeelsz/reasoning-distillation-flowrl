import json
from datasets import Dataset

# 直接用 Python 的 json 模块加载（规避 pyarrow 的限制）
with open("baselines/rllm/data/deepcoder_train.json", "r") as f:
    data = json.load(f)

# 构造 HuggingFace Dataset
dataset = Dataset.from_list(data)

print(f"Loaded dataset with {len(dataset)} samples.")

# 简单查看前5条
for i in range(5):
    print(dataset[i])
