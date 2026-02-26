#!/bin/bash

MODEL_NAME=Qwen/Qwen2.5-7B

hf download $MODEL_NAME \
  --repo-type model \
  --local-dir downloads/$MODEL_NAME \
  --exclude *.pth