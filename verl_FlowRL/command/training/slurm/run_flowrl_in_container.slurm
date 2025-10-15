#!/bin/bash
#SBATCH --partition=plm
#SBATCH --job-name=flowrl_run_with_pkgs
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=./logs/flowrl_%j.out
#SBATCH --error=./logs/flowrl_%j.err

# --- 基本设置 ---
mkdir -p ./logs
unset ROCR_VISIBLE_DEVICES
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "CWD: $(pwd)"
echo "=========================================="

# --- 环境变量 ---
SIF=/mnt/petrelfs/linzhouhan/xuekaizhu/containers/verl_hiyouga.sif
WORKDIR=/mnt/petrelfs/linzhouhan/xuekaizhu/dev/FlowRL/verl_FlowRL
RUNFILE=command/training/math/flowrl_1.5B_math_test.sh

# --- 关键：挂载整个 /mnt，以保证路径一致 ---
apptainer exec --nv \
  --bind /mnt:/mnt,/dev/shm:/dev/shm \
  "$SIF" \
  bash --noprofile --norc -c "
    set -e
    cd $WORKDIR

    echo '=== Inside container: installing required packages ==='
    pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple vertexai sentence_transformers

    echo '=== Packages installed. Checking environment ==='
    python3 -c 'import torch; print(\"Torch:\", torch.__version__, \"CUDA:\", torch.cuda.is_available(), \"GPUs:\", torch.cuda.device_count())'

    echo '=== Starting training ==='
    bash $RUNFILE
  "

echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="