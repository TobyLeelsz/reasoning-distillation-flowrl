#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_DIR}"

# Model selection:
# - Default behavior: evaluate merged model from FlowRL FSDP checkpoint (CKPT_DIR/actor).
# - Set MODEL_PATH to evaluate a direct HF/local model (e.g. Qwen/Qwen2.5-7B) without merge.
MODEL_PATH="${MODEL_PATH:-}"
MODEL_TRUST_REMOTE_CODE="${MODEL_TRUST_REMOTE_CODE:-false}"

# FlowRL checkpoint / model merge
CKPT_DIR="${CKPT_DIR:-${PROJECT_DIR}/checkpoints/FlowRL/math/7B/flowrl_qwen_7b_math_log_ratio_rm/global_step_400}"
ACTOR_DIR="${ACTOR_DIR:-${CKPT_DIR}/actor}"
FORCE_MERGE="${FORCE_MERGE:-0}"

if [ -n "${MODEL_PATH}" ] && [ -z "${EVAL_WORKDIR:-}" ]; then
    SAFE_MODEL_TAG="$(echo "${MODEL_PATH}" | sed 's#[/:]#_#g')"
    EVAL_WORKDIR="${PROJECT_DIR}/checkpoints/eval/${SAFE_MODEL_TAG}_gpqa_pass2"
fi
EVAL_WORKDIR="${EVAL_WORKDIR:-${CKPT_DIR}/eval_gpqa_pass2}"

if ! mkdir -p "${EVAL_WORKDIR}" 2>/dev/null; then
    FALLBACK_TAG="${SAFE_MODEL_TAG:-flowrl_ckpt}"
    EVAL_WORKDIR="/tmp/${FALLBACK_TAG}_gpqa_pass2"
    echo "[WARN] Cannot write EVAL_WORKDIR under project, fallback to: ${EVAL_WORKDIR}"
    mkdir -p "${EVAL_WORKDIR}"
fi

MERGED_MODEL_DIR="${MERGED_MODEL_DIR:-${EVAL_WORKDIR}/merged_hf}"

# GPQA data
AUTO_PREP_GPQA="${AUTO_PREP_GPQA:-1}"
GPQA_DATA_DIR="${GPQA_DATA_DIR:-${EVAL_WORKDIR}/data/gpqa}"
GPQA_PARQUET="${GPQA_PARQUET:-${GPQA_DATA_DIR}/test.parquet}"

# Generation
NNODES="${NNODES:-1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-1}"
TP_SIZE="${TP_SIZE:-1}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-128}"
N_SAMPLES="${N_SAMPLES:-2}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:--1}"
PROMPT_LENGTH="${PROMPT_LENGTH:-2048}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1024}"
ACTOR_PORJ_LAYER="${ACTOR_PORJ_LAYER:-3}"
GEN_OUTPUT_PARQUET="${GEN_OUTPUT_PARQUET:-${EVAL_WORKDIR}/gpqa_gen.parquet}"

# Eval
REWARD_SCRIPT="${REWARD_SCRIPT:-${PROJECT_DIR}/recipe/r1/reward_score.py}"
REWARD_FN_NAME="${REWARD_FN_NAME:-reward_func}"
EVAL_METRIC="${EVAL_METRIC:-pass_at_k}"
PASS_AT_K="${PASS_AT_K:-2}"
METRICS_PATH="${METRICS_PATH:-${EVAL_WORKDIR}/metrics.txt}"

echo "[INFO] PROJECT_DIR=${PROJECT_DIR}"
echo "[INFO] CKPT_DIR=${CKPT_DIR}"
echo "[INFO] ACTOR_DIR=${ACTOR_DIR}"
echo "[INFO] MERGED_MODEL_DIR=${MERGED_MODEL_DIR}"
echo "[INFO] MODEL_PATH=${MODEL_PATH:-<empty>}"
echo "[INFO] GPQA_PARQUET=${GPQA_PARQUET}"

GEN_MODEL_PATH=""

if [ ! -f "${GPQA_PARQUET}" ] && [ "${AUTO_PREP_GPQA}" = "1" ]; then
    echo "[INFO] GPQA parquet missing; generating via recipe.r1.data_process ..."
    mkdir -p "${GPQA_DATA_DIR}"
    PYTHONPATH=. python3 -m recipe.r1.data_process \
        --local_dir "${GPQA_DATA_DIR}" \
        --tasks gpqa_diamond
fi

if [ ! -f "${GPQA_PARQUET}" ]; then
    echo "[ERROR] GPQA parquet not found: ${GPQA_PARQUET}"
    echo "[ERROR] Provide GPQA_PARQUET or set AUTO_PREP_GPQA=1."
    exit 1
fi

if [ "${TEMPERATURE}" = "0" ] || [ "${TEMPERATURE}" = "0.0" ]; then
    if [ "${N_SAMPLES}" != "1" ]; then
        echo "[ERROR] N_SAMPLES must be 1 when TEMPERATURE is 0."
        exit 1
    fi
fi

if [ "${N_SAMPLES}" -lt "${PASS_AT_K}" ]; then
    echo "[ERROR] N_SAMPLES (${N_SAMPLES}) must be >= PASS_AT_K (${PASS_AT_K}) for valid pass@k evaluation."
    exit 1
fi

if [ -n "${MODEL_PATH}" ]; then
    GEN_MODEL_PATH="${MODEL_PATH}"
    echo "[INFO] Using direct model path/id for generation: ${GEN_MODEL_PATH}"
else
    if [ ! -d "${ACTOR_DIR}" ]; then
        echo "[ERROR] Actor checkpoint directory not found: ${ACTOR_DIR}"
        echo "[ERROR] Either provide a valid CKPT_DIR/ACTOR_DIR or set MODEL_PATH (e.g. MODEL_PATH=Qwen/Qwen2.5-7B)."
        exit 1
    fi

    if [ "${FORCE_MERGE}" = "1" ] || [ ! -f "${MERGED_MODEL_DIR}/config.json" ]; then
        echo "[INFO] Merging FSDP checkpoint to HuggingFace format ..."
        mkdir -p "${MERGED_MODEL_DIR}"
        PYTHONPATH=. python3 scripts/model_merger.py merge \
            --backend fsdp \
            --local_dir "${ACTOR_DIR}" \
            --target_dir "${MERGED_MODEL_DIR}"
    else
        echo "[INFO] Reusing merged model: ${MERGED_MODEL_DIR}"
    fi

    GEN_MODEL_PATH="${MERGED_MODEL_DIR}"
fi

echo "[INFO] Running GPQA generation ..."
GEN_ARGS=(
    "trainer.nnodes=${NNODES}"
    "trainer.n_gpus_per_node=${N_GPUS_PER_NODE}"
    "data.path=${GPQA_PARQUET}"
    "data.prompt_key=prompt"
    "data.batch_size=${GEN_BATCH_SIZE}"
    "data.n_samples=${N_SAMPLES}"
    "data.output_path=${GEN_OUTPUT_PARQUET}"
    "model.path=${GEN_MODEL_PATH}"
    "+model.trust_remote_code=${MODEL_TRUST_REMOTE_CODE}"
    "rollout.temperature=${TEMPERATURE}"
    "rollout.top_p=${TOP_P}"
    "rollout.top_k=${TOP_K}"
    "rollout.prompt_length=${PROMPT_LENGTH}"
    "rollout.response_length=${RESPONSE_LENGTH}"
    "rollout.tensor_model_parallel_size=${TP_SIZE}"
    "rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
    "rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}"
    "rollout.max_num_seqs=${MAX_NUM_SEQS}"
    "+actor.porj_layer=${ACTOR_PORJ_LAYER}"
)
PYTHONPATH=. python3 -m verl.trainer.main_generation "${GEN_ARGS[@]}"

echo "[INFO] Running GPQA evaluation (${EVAL_METRIC}, k=${PASS_AT_K}) ..."
mkdir -p "$(dirname "${METRICS_PATH}")"
EVAL_ARGS=(
    "data.path=${GEN_OUTPUT_PARQUET}"
    "data.prompt_key=prompt"
    "data.response_key=responses"
    "data.data_source_key=data_source"
    "data.reward_model_key=reward_model"
    "custom_reward_function.path=${REWARD_SCRIPT}"
    "custom_reward_function.name=${REWARD_FN_NAME}"
    "+eval.metric=${EVAL_METRIC}"
    "+eval.pass_at_k=${PASS_AT_K}"
)
PYTHONPATH=. python3 -m verl.trainer.main_eval "${EVAL_ARGS[@]}" | tee "${METRICS_PATH}"

echo "[INFO] Done."
echo "[INFO] Generated parquet: ${GEN_OUTPUT_PARQUET}"
echo "[INFO] Metrics log: ${METRICS_PATH}"
