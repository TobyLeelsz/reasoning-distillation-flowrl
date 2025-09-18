#!/bin/bash

MODEL_NAME=Qwen/Qwen2.5-7B

huggingface-cli download $MODEL_NAME \
  --repo-type model \
  --resume-download \
  --local-dir pre_trained_model/$MODEL_NAME \
  --local-dir-use-symlinks False \
  --exclude *.pth