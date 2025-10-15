#!/bin/bash
#SBATCH --partition=plm
#SBATCH --job-name=flowrl_run_file
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=./logs/flowrl_%j.out
#SBATCH --error=./logs/flowrl_%j.err

# 打印一些信息
mkdir -p ./logs
unset ROCR_VISIBLE_DEVICES
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "CWD: $(pwd)"
echo "=========================================="
nvidia-smi || true

proxy_on

# 路径
SIF=/mnt/petrelfs/linzhouhan/xuekaizhu/containers/verl_hiyouga.sif
WORKDIR=/mnt/petrelfs/linzhouhan/xuekaizhu/dev/FlowRL/verl_FlowRL
RUNFILE=command/training/math/flowrl_1.5B_math_test.sh

# 可选：把临时编译缓存放内存盘（更稳更快）
export TRITON_CACHE_DIR=/dev/shm/triton_cache_$SLURM_JOB_ID
export XDG_CACHE_HOME=$TRITON_CACHE_DIR
export TMPDIR=/dev/shm/tmp_$SLURM_JOB_ID
mkdir -p "$TRITON_CACHE_DIR" "$TMPDIR"

# 关键点：把 /mnt 原样挂进容器，保证绝对路径在容器里也成立
apptainer exec --nv \
  --bind /mnt:/mnt,/dev/shm:/dev/shm \
  "$SIF" \
  bash --noprofile --norc -c "
    set -e
    cd $WORKDIR
    echo 'Inside container:' \$(hostname)
    nvidia-smi || true
    bash $RUNFILE
  "

echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="