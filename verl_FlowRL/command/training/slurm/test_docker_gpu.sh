#!/bin/bash
#SBATCH --partition=plm
#SBATCH --job-name=test_container_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --output=./logs/container_gpu_%j.out
#SBATCH --error=./logs/container_gpu_%j.err
#SBATCH --time=00:10:00

# 确保日志目录存在
mkdir -p ./logs

unset ROCR_VISIBLE_DEVICES
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="
nvidia-smi

SIF=/mnt/petrelfs/linzhouhan/xuekaizhu/containers/verl_hiyouga.sif
WORKDIR=/mnt/petrelfs/linzhouhan/xuekaizhu

echo "Launching Apptainer container..."
apptainer exec --nv \
  --bind ${WORKDIR}:/workspace \
  "$SIF" \
  bash -c '
    echo "Inside container:"
    nvidia-smi
    python3 - <<PY
import torch
print("torch version:", getattr(torch, "__version__", "unknown"))
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i} name:", torch.cuda.get_device_name(i))
PY
  '

echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="