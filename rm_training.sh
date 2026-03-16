#!/usr/bin/env bash
set -euo pipefail

# rm -rf /home/azureuser/shangzhe/GRPOMath/rm_out/tokenized_train_ds \
#        /home/azureuser/shangzhe/GRPOMath/rm_out/tokenized_train_ds.ready \
#        /home/azureuser/shangzhe/GRPOMath/rm_out/tokenized_train_ds.meta.json

PRETOKENIZE_NUM_PROC=16
PRETOKENIZE_BATCH_SIZE=128 
PRECOMPUTE_REF_BATCH_SIZE=1
DATALOADER_WORKERS=2
PRETOKENIZE_WAIT_TIMEOUT_SEC=14400
NUM_PROCESSES=4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRETOKENIZE_NUM_PROC="${PRETOKENIZE_NUM_PROC:-8}"
PRETOKENIZE_BATCH_SIZE="${PRETOKENIZE_BATCH_SIZE:-64}"
PRECOMPUTE_REF_BATCH_SIZE="${PRECOMPUTE_REF_BATCH_SIZE:-72}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-4}"
PRETOKENIZE_WAIT_TIMEOUT_SEC="${PRETOKENIZE_WAIT_TIMEOUT_SEC:-7200}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"

accelerate launch \
  --use_deepspeed \
  --num_processes "${NUM_PROCESSES}" \
  --num_machines 1 \
  --mixed_precision bf16 \
  --deepspeed_config_file "${SCRIPT_DIR}/flowrl/deepspeed_zero3_hf.json" \
  "${SCRIPT_DIR}/chi_squared_rm.py" \
  --paired_jsonl "${SCRIPT_DIR}/dapo_qwen3_1p7b_sft/dapo_17k_merged_qwen3_sft_new.jsonl" \
  --model_name_or_path /home/azureuser/shangzhe/GRPOMath/checkpoint_sft/checkpoint-1632 \
  --ref_model_name_or_path /home/azureuser/shangzhe/GRPOMath/checkpoint_sft/checkpoint-1632 \
  --output_dir "${SCRIPT_DIR}/rm_out" \
  --bf16 \
  --attn_implementation flash_attention_2 \
  --pretokenize_num_proc "${PRETOKENIZE_NUM_PROC}" \
  --pretokenize_batch_size "${PRETOKENIZE_BATCH_SIZE}" \
  --pretokenize_wait_timeout_sec "${PRETOKENIZE_WAIT_TIMEOUT_SEC}" \
  --precompute_ref_logps \
  --precompute_ref_batch_size "${PRECOMPUTE_REF_BATCH_SIZE}" \
  --reuse_pretokenized_cache \
  --dataloader_num_workers "${DATALOADER_WORKERS}" \
  --warmup_steps 100 \
  --beta 0.001 \
  --wandb_project rm-dapo \
  --wandb_log_internal \
  # --length_normalize \

# Optional knobs:
# --reg_coef 0.005
# PRETOKENIZE_NUM_PROC=16 PRETOKENIZE_BATCH_SIZE=128 bash rm_training.sh
