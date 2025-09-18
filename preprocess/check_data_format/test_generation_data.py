from datasets import load_dataset
import pickle
from collections import Counter


data_path = "outputs/gfn_important_sampling_step200_v0.1/test-output-16.parquet"

# Load the train.parquet file
dataset = load_dataset('parquet', data_files=data_path, split='train')
print(dataset)

print("\n🧾 Available columns:", dataset.column_names)

first_item = dataset[0]
for k, v in first_item.items():
    if k == "responses":
        print(len(v))
        print(v[:2])  

# invalid_count = 0
for i, example in enumerate(dataset):
    responses = example.get("responses", None)
    if not isinstance(responses, list) or len(responses) != 16:
        print(f"[Line {i}] ⚠️ responses is invalid: type={type(responses)}, len={len(responses) if isinstance(responses, list) else 'N/A'}")
        invalid_count += 1

data_sources = [example['data_source'] for example in dataset]
source_counter = Counter(data_sources)

print("\n📊 Data source counts:")
for source, count in source_counter.items():
    print(f"- {source}: {count} examples")

print(f"\n🔢 Total unique data sources: {len(source_counter)}")        

# if invalid_count == 0:
#     print("\n✅ All responses are valid lists of length 16.")
# else:
#     print(f"\n❗ Found {invalid_count} items with invalid responses.")