from datasets import load_dataset
import pickle
import json
import os

processed_data = []

data_path = "data/math"
train_data = load_dataset('parquet', data_files='{}/train.parquet'.format(data_path))['train']
for iterm in train_data:
    iterm_dict = { "instruction": {},
                    "input": iterm["prompt"][0]["content"],
                    "output": iterm["reward_model"]["ground_truth"],
                    "data_source":"math"}
    processed_data.append(json.dumps(iterm_dict))


data_path = "data/gsm8k"
train_data = load_dataset('parquet', data_files='{}/train.parquet'.format(data_path))['train']
for iterm in train_data:
    iterm_dict = { "instruction": {},
                    "input": iterm["prompt"][0]["content"],
                    "output": iterm["extra_info"]["answer"],
                    "data_source":"gsm8k"}
    processed_data.append(json.dumps(iterm_dict))

save_data_path = "data/math-sft"
os.makedirs(save_data_path, exist_ok=True)
with open(f"{save_data_path}/train.json", "w") as f_in:
    for i in processed_data:
        f_in.write(i+"\n")