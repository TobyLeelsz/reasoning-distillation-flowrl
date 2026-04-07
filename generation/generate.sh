#!/usr/bin/env bash

if [[ -z "${TRITON_CACHE_DIR:-}" ]]; then
  export TRITON_CACHE_DIR="/tmp/${USER:-$(id -un)}/triton-autotune"
fi
mkdir -p "${TRITON_CACHE_DIR}"

# DeepSpeed reads torch.utils.cpp_extension.CUDA_HOME; fix stale CUDA_HOME values.
if [[ -z "${CUDA_HOME:-}" || ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/nvcc" ]]; then
    export CUDA_HOME="${CONDA_PREFIX}"
  else
    NVCC_BIN="$(command -v nvcc 2>/dev/null || true)"
    if [[ -n "${NVCC_BIN}" ]]; then
      export CUDA_HOME="$(dirname "$(dirname "${NVCC_BIN}")")"
    fi
  fi
fi

if [[ -z "${CUDA_HOME:-}" || ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "ERROR: nvcc not found. Set CUDA_HOME to a valid CUDA toolkit root (contains bin/nvcc)." >&2
  exit 1
fi

export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"

DIST_TIMEOUT_SEC=43200
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-${SCRIPT_DIR}/accelerate_generate.yaml}"

if [[ ! -f "${ACCELERATE_CONFIG_FILE}" ]]; then
  echo "ERROR: accelerate config not found: ${ACCELERATE_CONFIG_FILE}" >&2
  exit 1
fi

# Ensure this inference launch doesn't inherit training-time DeepSpeed settings.
unset ACCELERATE_USE_DEEPSPEED
unset ACCELERATE_DEEPSPEED_CONFIG_FILE

accelerate launch --config_file "${ACCELERATE_CONFIG_FILE}" --num_processes 4 "${SCRIPT_DIR}/gen_deepmath_qwen_accel.py" \
  --model_id Qwen/Qwen3-1.7B-Base \
  --dataset open-r1/DAPO-Math-17k-Processed \
  --split train \
  --out_dir "${SCRIPT_DIR}/out_dapo_qwen3_1p7b_base_new" \
  --batch_size 72 \
  --max_new_tokens 8192 \
  --temperature 1.0 --top_p 1.0 \
  --dist_timeout_sec "${DIST_TIMEOUT_SEC}"

# accelerate launch --config_file "${ACCELERATE_CONFIG_FILE}" --num_processes 4 "${SCRIPT_DIR}/gen_deepmath_qwen_accel.py" \
#   --model_id Qwen/Qwen3-1.7B-Base \
#   --out_dir "${SCRIPT_DIR}/out_dapo_complement_new" \
#   --batch_size 64 \
#   --max_new_tokens 8192 \
#   --temperature 1.0 --top_p 0.7 \
#   --complement_only \
#   --existing_jsonl /home/azureuser/shangzhe/GRPOMath/out_dapo_qwen3_1p7b_base_new/deepmath_generated_merged.jsonl \
#   --target_dataset open-r1/DAPO-Math-17k-Processed \
#   --target_split train

# python "${SCRIPT_DIR}/gen_deepmath_qwen_accel.py" --merge_only --out_dir "${SCRIPT_DIR}/generated_qwen_2.5"
