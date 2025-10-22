#!/bin/bash
#SBATCH --partition=plm
#SBATCH --job-name=xuekai_flowrl_v0_1022
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --output=./logs/%j.out
#SBATCH --error=./logs/%j.err

# Unset AMD GPU variable to avoid conflicts with CUDA
unset ROCR_VISIBLE_DEVICES

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

export RAY_TMPDIR=/mnt/hwfile/linzhouhan/rt
mkdir -p "$RAY_TMPDIR"

# Print GPU info
nvidia-smi


# ===== TRITON =====
export TRITON_CACHE_DIR=/dev/shm/triton_${SLURM_JOB_ID}_${HOSTNAME}
export XDG_CACHE_HOME="$TRITON_CACHE_DIR"
export TMPDIR=/dev/shm/tmp_${SLURM_JOB_ID}_${HOSTNAME}
mkdir -p "$TRITON_CACHE_DIR" "$TMPDIR"
# ================================

# Run the training script
cd /mnt/petrelfs/linzhouhan/xuekaizhu/dev/FlowRL/verl_FlowRL/
bash command/training/math/flowrl_7B_math_clip_max.sh