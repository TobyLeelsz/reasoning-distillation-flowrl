#!/bin/bash
#SBATCH --partition=plm
#SBATCH --job-name=test_container_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --output=./logs/container_gpu_%j.out
#SBATCH --error=./logs/container_gpu_%j.err
#SBATCH --time=00:10:00

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
  $SIF \
  bash -lc "
    echo 'Inside container:'
    nvidia-smi
    python3 -c 'import torch; print(f\"torch version: {torch.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"GPU count: {torch.cuda.device_count()}\")'
  "

echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="