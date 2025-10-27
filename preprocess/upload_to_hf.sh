#!/bin/bash

USERNAME="xuekai"
REPO_NAME="FlowRL-Qwen2.5-32B-math"
LOCAL_DIR="results_model/code/deepseek-7B/flowrl_7b_8k_to_16k_global_step_100"

export HF_HUB_ENABLE_HF_TRANSFER=1

huggingface-cli repo create $USERNAME/$REPO_NAME --type model

huggingface-cli upload $USERNAME/$REPO_NAME $LOCAL_DIR --repo-type model
