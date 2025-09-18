from datasets import load_dataset
from collections import Counter

# 修改为你的 parquet 文件路径
data_path = "data/r1_bench_plus/test.parquet"

# 加载数据集
dataset = load_dataset('parquet', data_files=data_path)['train']

# 提取所有样本的 extra_info.index 字段
indices = []
for item in dataset:
    if 'extra_info' in item and 'index' in item['extra_info']:
        indices.append(str(item['extra_info']['index']))
    # if "__index_level_0__" in item:
    #     indices.append(item['__index_level_0__'])
    #     print(item['__index_level_0__'])
    

# 统计每个 index 的出现次数
index_counts = Counter(indices)

# 找出重复的 index（出现次数 > 1）
repeated_indices = {idx: count for idx, count in index_counts.items() if count > 1}

# 输出统计信息
print(f"📌 总样本数: {len(dataset)}")
print(f"🔁 重复 index 数量: {len(repeated_indices)}")
print(f"🔁 总重复样本数（超出唯一 index 的部分）: {sum(repeated_indices.values()) - len(repeated_indices)}")
print("\n📋 出现重复的 index（部分展示）:")
for idx, count in list(repeated_indices.items())[:10]:
    print(f"Index: {idx}, Count: {count}")