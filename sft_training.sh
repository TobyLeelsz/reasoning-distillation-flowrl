#!/usr/bin/env bash
set -euo pipefail

DATALOADER_WORKERS=2
NUM_PROCESSES=4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-4}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"

accelerate launch \
  --use_deepspeed \
  --num_processes "${NUM_PROCESSES}" \
  --num_machines 1 \
  --mixed_precision bf16 \
  --deepspeed_config_file "${SCRIPT_DIR}/flowrl/deepspeed_zero3_hf.json" \
  "${SCRIPT_DIR}/sft_train.py" \
  --train_jsonl "${SCRIPT_DIR}/dapo_qwen3_1p7b/dapo_17k_gpt_new.jsonl" \
  --model_name_or_path Qwen/Qwen3-1.7B-Base \
  --output_dir "${SCRIPT_DIR}/sft_out_qwen3_1p7b" \
  --bf16 \
  --attn_implementation flash_attention_2 \
  --dataloader_num_workers "${DATALOADER_WORKERS}" \
  --warmup_steps 100 \
  --math500_pass_k 2 \
  --math500_eval_max_samples 500 \
  --math500_eval_batch_size 1 \
  --math500_eval_max_new_tokens 4096 \
  --wandb_project sft-dapo

# Optional knobs:
# NUM_PROCESSES=4 DATALOADER_WORKERS=2 bash sft_training.sh
