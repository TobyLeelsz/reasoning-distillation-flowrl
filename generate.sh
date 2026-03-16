DIST_TIMEOUT_SEC=43200

accelerate launch --num_processes 4 gen_deepmath_qwen_accel.py \
  --model_id /home/azureuser/shangzhe/GRPOMath/sft_out_qwen3_1p7b/checkpoint-1632 \
  --dataset open-r1/DAPO-Math-17k-Processed \
  --split train \
  --out_dir out_dapo_qwen3_1p7b_sft \
  --batch_size 36 \
  --max_new_tokens 8192 \
  --temperature 1.0 --top_p 1.0 \
  --dist_timeout_sec "${DIST_TIMEOUT_SEC}"

# accelerate launch --num_processes 4 gen_deepmath_qwen_accel.py \
#   --model_id Qwen/Qwen3-1.7B-Base \
#   --out_dir out_dapo_complement_new_qwen3_1p7b \
#   --batch_size 64 \
#   --max_new_tokens 8192 \
#   --temperature 1.0 --top_p 1.0 \
#   --complement_only \
#   --existing_jsonl /home/azureuser/shangzhe/GRPOMath/out_deepmath_qwen3/deepmath_generated_merged.jsonl \
#   --target_dataset open-r1/DAPO-Math-17k-Processed \
#   --target_split train

# python gen_deepmath_qwen_accel.py --merge_only --out_dir generated_qwen_2.5
