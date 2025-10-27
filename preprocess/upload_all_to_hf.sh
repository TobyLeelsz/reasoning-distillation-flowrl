#!/bin/bash

USERNAME="xuekai"
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "=========================================="
echo "Uploading FlowRL Models to Hugging Face"
echo "=========================================="

# Model 1: Qwen2.5-7B-math
echo "[1/3] Uploading FlowRL-Qwen2.5-7B-math..."
REPO_NAME="FlowRL-Qwen2.5-7B-math"
LOCAL_DIR="/mnt/petrelfs/linzhouhan/xuekaizhu/from_huoshan/results_model/results_model/ablation_is/is_15_step200"
huggingface-cli repo create $USERNAME/$REPO_NAME --type model
huggingface-cli upload $USERNAME/$REPO_NAME $LOCAL_DIR --repo-type model
echo "✓ Completed: $REPO_NAME"
echo ""

# Model 2: Qwen2.5-32B-math
echo "[2/3] Uploading FlowRL-Qwen2.5-32B-math..."
REPO_NAME="FlowRL-Qwen2.5-32B-math"
LOCAL_DIR="/mnt/petrelfs/linzhouhan/xuekaizhu/from_huoshan/results_model/results_model/gfn/qwen_32B/gfn_is_qwen_32B_0629_global_step_200"
huggingface-cli repo create $USERNAME/$REPO_NAME --type model
huggingface-cli upload $USERNAME/$REPO_NAME $LOCAL_DIR --repo-type model
echo "✓ Completed: $REPO_NAME"
echo ""

# Model 3: DeepSeek-7B-code
echo "[3/3] Uploading FlowRL-DeepSeek-7B-code..."
REPO_NAME="FlowRL-DeepSeek-7B-code"
LOCAL_DIR="/mnt/petrelfs/linzhouhan/xuekaizhu/from_huoshan/results_model/results_model/code/deepseek-7B/gfn_qwen_7b_8k_code_global_step_350"
huggingface-cli repo create $USERNAME/$REPO_NAME --type model
huggingface-cli upload $USERNAME/$REPO_NAME $LOCAL_DIR --repo-type model
echo "✓ Completed: $REPO_NAME"
echo ""

echo "=========================================="
echo "All models uploaded successfully!"
echo "=========================================="
