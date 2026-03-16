python generate_paired_data.py \
  --generated_jsonl /home/azureuser/shangzhe/GRPOMath/out_deepmath_qwen2/deepmath_generated_merged.jsonl \
  --out_jsonl paired.jsonl \
  --hf_dataset trl-lib/DeepMath-103K \
  --hf_split train \
#   --skip_unmatched