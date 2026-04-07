#!/bin/bash
#SBATCH --job-name=flowrl-generate-1p7b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/proj/weitongzlab/projects/reasoning-distillation-flowrl/generation/slurm-%j.out
#SBATCH --error=/proj/weitongzlab/projects/reasoning-distillation-flowrl/generation/slurm-%j.err
## Optional, if your cluster requires them:
#SBATCH --partition=weitongzlab

set -euo pipefail

PROJECT_ROOT="/proj/weitongzlab/projects/reasoning-distillation-flowrl"
GEN_DIR="${PROJECT_ROOT}/generation"

cd "${PROJECT_ROOT}"

# Initialize conda in non-interactive shells, then use requested activation form.
source /nas/longleaf/home/shangzhe/anaconda3/etc/profile.d/conda.sh
source activate verl_flowrl

echo "[$(date)] Host: $(hostname)"
echo "[$(date)] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[$(date)] Running: ${GEN_DIR}/generate.sh"

bash "${GEN_DIR}/generate.sh"
