#!/bin/bash

USERNAME="xuekai"
REPO_NAME="FlowRL-DeepSeek-Coder-7B-code"
LOCAL_DIR="/mnt/petrelfs/linzhouhan/xuekaizhu/from_huoshan/results_model/results_model/code/deepseek-7B/gfn_qwen_7b_8k_code_global_step_350"

export HF_HUB_ENABLE_HF_TRANSFER=1

huggingface-cli repo create $USERNAME/$REPO_NAME --type model

huggingface-cli upload $USERNAME/$REPO_NAME $LOCAL_DIR --repo-type model
