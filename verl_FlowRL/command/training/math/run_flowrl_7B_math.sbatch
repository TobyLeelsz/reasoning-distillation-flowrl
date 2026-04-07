#!/bin/bash
#SBATCH --job-name=flowrl-math-1p7b
#SBATCH --account=rc_yzlin_pi
#SBATCH --partition=yzlinlab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_max-q_workstation_edition:4
#SBATCH --time=10-00:00:00
#SBATCH --mem=64G
#SBATCH --output=/proj/weitongzlab/projects/reasoning-distillation-flowrl/verl_FlowRL/command/training/math/slurm-%j.out
#SBATCH --error=/proj/weitongzlab/projects/reasoning-distillation-flowrl/verl_FlowRL/command/training/math/slurm-%j.err

set -euo pipefail

PROJECT_ROOT="/proj/weitongzlab/projects/reasoning-distillation-flowrl"
FLOWRL_ROOT="${PROJECT_ROOT}/verl_FlowRL"
TRAIN_SCRIPT="${FLOWRL_ROOT}/command/training/math/flowrl_7B_math.sh"
BASE_POLICY_MODEL="${PROJECT_ROOT}/downloads/Qwen/sft-dapo-17k-qwen3-1.7b"
REWARD_MODEL="/proj/weitongzlab/projects/rm_training/rm_out/checkpoint-1629"
ALIGNED_MATH_TRAIN_PATH="${PROJECT_ROOT}/data/math_data/dapo-math-17k.prompt_aligned_with_dapo17k_rm.parquet"
BASE_MATH_TRAIN_PATH="${PROJECT_ROOT}/data/math_data/dapo-math-17k.parquet"

if [ "$#" -eq 0 ]; then
    # Default launch profile; override by passing args to sbatch.
    set -- restart-2rollout-2reward
fi

source /nas/longleaf/home/shangzhe/anaconda3/etc/profile.d/conda.sh
conda activate verl_flowrl

cd "${FLOWRL_ROOT}"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
export REWARD_ASYNC_NUM_CPUS="${REWARD_ASYNC_NUM_CPUS:-2}"
export USE_TORCH_COMPILE="${USE_TORCH_COMPILE:-0}"

export PRETRAINED_MODEL="${BASE_POLICY_MODEL}"
export RM_REFERENCE_MODEL="${BASE_POLICY_MODEL}"
export RM_TOKENIZER_PATH="${BASE_POLICY_MODEL}"
export RM_POLICY_MODEL="${REWARD_MODEL}"
export MODEL_TAG="qwen3_1p7b"
export MODEL_SIZE_DIR="1.7B"

# Keep rollout prompts aligned with online RM positive prompts by default.
if [ -f "${ALIGNED_MATH_TRAIN_PATH}" ]; then
    export FLOWRL_MATH_TRAIN_PATH="${FLOWRL_MATH_TRAIN_PATH:-${ALIGNED_MATH_TRAIN_PATH}}"
else
    export FLOWRL_MATH_TRAIN_PATH="${FLOWRL_MATH_TRAIN_PATH:-${BASE_MATH_TRAIN_PATH}}"
fi

export VERL_LOG_RATIO_REWARD_VERBOSE=1
export VERL_LOG_RATIO_RM_LOSS_LOG_EVERY=1

echo "[$(date)] Host: $(hostname)"
echo "[$(date)] CWD: $(pwd)"
echo "[$(date)] PRETRAINED_MODEL=${PRETRAINED_MODEL}"
echo "[$(date)] RM_REFERENCE_MODEL=${RM_REFERENCE_MODEL}"
echo "[$(date)] RM_POLICY_MODEL=${RM_POLICY_MODEL}"
echo "[$(date)] OMP_NUM_THREADS=${OMP_NUM_THREADS} MKL_NUM_THREADS=${MKL_NUM_THREADS}"
echo "[$(date)] DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS} REWARD_ASYNC_NUM_CPUS=${REWARD_ASYNC_NUM_CPUS}"
echo "[$(date)] USE_TORCH_COMPILE=${USE_TORCH_COMPILE}"
echo "[$(date)] FLOWRL_MATH_TRAIN_PATH=${FLOWRL_MATH_TRAIN_PATH}"
echo "[$(date)] Running: bash ${TRAIN_SCRIPT} $*"

bash "${TRAIN_SCRIPT}" "$@"
