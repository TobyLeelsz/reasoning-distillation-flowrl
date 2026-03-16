#!/usr/bin/env bash

set -u  # 未定义变量直接报错
CMD="python generate_gpt.py --workers 64"
SLEEP=10   # 崩溃后等待秒数

i=0
while true; do
  i=$((i+1))
  echo "=============================="
  echo "[`date`] Start run #$i"
  echo "Command: $CMD"
  echo "=============================="

  $CMD
  exit_code=$?

  echo "[`date`] Process exited with code $exit_code"

  echo "Sleeping ${SLEEP}s before restart..."
  sleep $SLEEP
done