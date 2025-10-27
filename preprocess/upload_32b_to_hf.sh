#!/bin/bash

USERNAME="xuekai"
REPO_NAME="FlowRL-Qwen2.5-32B-math"
LOCAL_DIR="/mnt/petrelfs/linzhouhan/xuekaizhu/from_huoshan/results_model/results_model/gfn/qwen_32B/gfn_is_qwen_32B_0629_global_step_200"

export HF_HUB_ENABLE_HF_TRANSFER=1

huggingface-cli repo create $USERNAME/$REPO_NAME --type model

huggingface-cli upload $USERNAME/$REPO_NAME $LOCAL_DIR --repo-type model
