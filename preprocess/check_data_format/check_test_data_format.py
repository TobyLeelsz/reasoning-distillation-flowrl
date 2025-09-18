from datasets import load_dataset
import os
from tqdm import tqdm
from collections import Counter

parquet_path = os.path.expanduser("data/dapo/dapo-math-17k.parquet")

dataset = load_dataset("parquet", data_files=parquet_path, split="train")

prefix = "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
suffix = "\n\nRemember to put your answer on its own line after \"Answer:\"."


data_source_counter = Counter()

for i, example in enumerate(tqdm(dataset)):
    content = example["prompt"][0]["content"]
    answer = example["reward_model"]["ground_truth"]
    data_source = example["data_source"]

    data_source_counter[data_source] += 1

    if not content.startswith(prefix):
        print(f"[Line {i}] prompt does not start with prefix.")
    if not content.endswith(suffix):
        print(f"[Line {i}] prompt does not end with suffix.")
    if not isinstance(answer, str):
        print(f"[Line {i}] ground_truth is not a string.")
    if "\n" in answer:
        print(f"[Line {i}] ground_truth contains newline.")

print("\n📊 Data Source Distribution:")
for k, v in data_source_counter.items():
    print(f"  - {k}: {v} samples")

print("\n✅ Validation complete.")