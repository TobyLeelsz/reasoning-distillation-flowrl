#!/bin/bash

# Simple HuggingFace data upload script
# Usage: run from the FlowRL root directory

PROJECT="xuekai"  # Use your username
REPO_NAME="flowrl-data-collection"
LOCAL_DIR="data/"

# Optional: enable fast upload
export HF_HUB_ENABLE_HF_TRANSFER=1

# Create public dataset repository (if not exists)
# Comment out if repo already exists
# huggingface-cli repo create $PROJECT/$REPO_NAME --type dataset

echo "📤 Uploading $LOCAL_DIR to $PROJECT/$REPO_NAME (public dataset)..."
huggingface-cli upload $PROJECT/$REPO_NAME $LOCAL_DIR --repo-type dataset
echo "✅ Done! Check: https://huggingface.co/datasets/$PROJECT/$REPO_NAME"