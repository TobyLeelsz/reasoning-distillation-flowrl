#!/bin/bash
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

export WANDB_INIT_TIMEOUT=600
export WANDB_HTTP_TIMEOUT=600
export WANDB__SERVICE_WAIT=600

# Built-in restart helper:
#   bash flowrl_7B_math.sh restart-2plus2
# This performs a clean Ray restart and relaunches with:
#   2 GPUs actor+rollout/update + 2 GPUs async reward.
#
#   bash flowrl_7B_math.sh restart-1plus1
# This performs a clean Ray restart and relaunches with:
#   1 GPU actor+rollout/update + 1 GPU async reward.
#
#   bash flowrl_7B_math.sh restart-1plus1-rewardgpu
# This performs a clean Ray restart and relaunches with:
#   1 GPU actor+rollout/update + 1 GPU async reward, forcing reward on CUDA.
#
#   bash flowrl_7B_math.sh restart-minmem
# This performs a clean Ray restart and relaunches with an ultra-low-memory profile.
#
#   bash flowrl_7B_math.sh restart-rule-1gpu
# This performs a clean Ray restart and relaunches with:
#   rule-based reward + single-GPU smoke-test profile.
#
#   bash flowrl_7B_math.sh restart-rule-2gpu
# This performs a clean Ray restart and relaunches with:
#   rule-based reward on 2 GPUs.
#
#   bash flowrl_7B_math.sh restart-2rollout-2reward
# This performs a clean Ray restart and relaunches with:
#   log-ratio RM reward, 2 GPUs for actor+rollout/update, 2 GPUs for async reward.
if [ "${1:-}" = "restart-2plus2" ]; then
    shift
    ray stop --force >/dev/null 2>&1 || true
    exec env \
        REWARD_MODE=log_ratio_rm \
        CUDA_VISIBLE_DEVICES=0,1,2,3 \
        N_GPUS_PER_NODE=4 \
        OPTION1_SPLIT_MODE=1 \
        ACTOR_ROLLOUT_GPUS=2 \
        REWARD_ASYNC_NUM_GPUS=2 \
        TRAIN_BATCH_SIZE=64 \
        ROLLOUT_N=1 \
        MAX_PROMPT_LENGTH=2048 \
        ROLLOUT_GPU_MEMORY_UTIL=0.45 \
        ROLLOUT_MAX_NUM_SEQS=8 \
        ACTOR_PARAM_OFFLOAD=true \
        ACTOR_OPTIMIZER_OFFLOAD=true \
        REF_PARAM_OFFLOAD=true \
        RESUME_MODE=disable \
        USE_RULE_REWARD_FOR_EVAL=1 \
        RM_FORCE_CPU=0 \
        LOG_RATIO_SAFE_MODE=0 \
        DISABLE_VAL=0 \
        SINGLE_GPU_TEST_MODE=0 \
        MIN_MEM_MODE=0 \
        LOGGER_MODE=both \
        RAY_CLEAN_START=1 \
        bash "$0" "$@"
fi

if [ "${1:-}" = "restart-2rollout-2reward" ]; then
    shift
    ray stop --force >/dev/null 2>&1 || true
    exec env \
        REWARD_MODE=log_ratio_rm \
        CUDA_VISIBLE_DEVICES=0,1,2,3 \
        N_GPUS_PER_NODE=4 \
        OPTION1_SPLIT_MODE=1 \
        ACTOR_ROLLOUT_GPUS=2 \
        REWARD_ASYNC_NUM_GPUS=2 \
        TRAIN_BATCH_SIZE=64 \
        ROLLOUT_N=1 \
        MAX_PROMPT_LENGTH=2048 \
        ROLLOUT_GPU_MEMORY_UTIL=0.45 \
        ROLLOUT_MAX_NUM_SEQS=8 \
        ACTOR_PARAM_OFFLOAD=true \
        ACTOR_OPTIMIZER_OFFLOAD=true \
        REF_PARAM_OFFLOAD=true \
        RESUME_MODE=disable \
        USE_RULE_REWARD_FOR_EVAL=1 \
        RM_FORCE_CPU=0 \
        LOG_RATIO_SAFE_MODE=0 \
        DISABLE_VAL=0 \
        SINGLE_GPU_TEST_MODE=0 \
        MIN_MEM_MODE=0 \
        LOGGER_MODE=both \
        RAY_CLEAN_START=1 \
        bash "$0" "$@"
fi

if [ "${1:-}" = "restart-1plus1" ]; then
    shift
    ray stop --force >/dev/null 2>&1 || true
    exec env \
        N_GPUS_PER_NODE=2 \
        OPTION1_SPLIT_MODE=1 \
        ACTOR_ROLLOUT_GPUS=1 \
        REWARD_ASYNC_NUM_GPUS=1 \
        LOGGER_MODE=console \
        RAY_CLEAN_START=1 \
        bash "$0" "$@"
fi

if [ "${1:-}" = "restart-1plus1-rewardgpu" ]; then
    shift
    ray stop --force >/dev/null 2>&1 || true
    exec env \
        N_GPUS_PER_NODE=2 \
        OPTION1_SPLIT_MODE=1 \
        ACTOR_ROLLOUT_GPUS=1 \
        REWARD_ASYNC_NUM_GPUS=1 \
        RM_FORCE_CPU=0 \
        MIN_MEM_MODE=1 \
        MIN_MEM_REWARD_GPU=1 \
        LOGGER_MODE=console \
        RAY_CLEAN_START=1 \
        bash "$0" "$@"
fi

if [ "${1:-}" = "restart-minmem" ]; then
    shift
    ray stop --force >/dev/null 2>&1 || true
    exec env \
        N_GPUS_PER_NODE=2 \
        OPTION1_SPLIT_MODE=1 \
        ACTOR_ROLLOUT_GPUS=1 \
        REWARD_ASYNC_NUM_GPUS=1 \
        MIN_MEM_MODE=1 \
        LOGGER_MODE=console \
        RAY_CLEAN_START=1 \
        bash "$0" "$@"
fi

if [ "${1:-}" = "restart-rule-1gpu" ]; then
    shift
    ray stop --force >/dev/null 2>&1 || true
    exec env \
        REWARD_MODE=rule \
        N_GPUS_PER_NODE=1 \
        OPTION1_SPLIT_MODE=0 \
        SINGLE_GPU_TEST_MODE=1 \
        LOGGER_MODE=console \
        RAY_CLEAN_START=1 \
        bash "$0" "$@"
fi

if [ "${1:-}" = "restart-rule-2gpu" ]; then
    shift
    ray stop --force >/dev/null 2>&1 || true
    exec env \
        REWARD_MODE=rule \
        N_GPUS_PER_NODE=2 \
        OPTION1_SPLIT_MODE=0 \
        SINGLE_GPU_TEST_MODE=0 \
        LOGGER_MODE=console \
        RAY_CLEAN_START=1 \
        bash "$0" "$@"
fi

PRETRAINED_MODEL=../downloads/Qwen/Qwen2.5-7B
n_nodes=${N_NODES:-1}
n_gpus_per_node=${N_GPUS_PER_NODE:-4}
tensor_model_parallel_size=${ROLLOUT_TP_SIZE:-1}
save_freq=${SAVE_FREQ:-200}

# Split mode:
# reserve some GPUs for actor+rollout/update and the rest for async reward computation.
# Default split is 2 (actor/update) + 2 (reward) on a 4-GPU node.
if [ "$n_gpus_per_node" -ge 4 ]; then
    OPTION1_SPLIT_MODE=${OPTION1_SPLIT_MODE:-1}
else
    OPTION1_SPLIT_MODE=${OPTION1_SPLIT_MODE:-0}
fi
ACTOR_ROLLOUT_GPUS=${ACTOR_ROLLOUT_GPUS:-2}
REWARD_ASYNC_NUM_GPUS=${REWARD_ASYNC_NUM_GPUS:-2}
if [ "$OPTION1_SPLIT_MODE" = "1" ]; then
    if [ "$ACTOR_ROLLOUT_GPUS" -le 0 ]; then
        echo "ACTOR_ROLLOUT_GPUS must be >=1 (got $ACTOR_ROLLOUT_GPUS)"
        exit 1
    fi
    if [ "$REWARD_ASYNC_NUM_GPUS" -le 0 ]; then
        echo "REWARD_ASYNC_NUM_GPUS must be >=1 (got $REWARD_ASYNC_NUM_GPUS)"
        exit 1
    fi
    total_split=$((ACTOR_ROLLOUT_GPUS + REWARD_ASYNC_NUM_GPUS))
    if [ "$total_split" -gt "$n_gpus_per_node" ]; then
        echo "Split requires $total_split GPUs, but N_GPUS_PER_NODE=$n_gpus_per_node"
        exit 1
    fi
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        IFS=',' read -r -a _visible_gpu_arr <<< "$CUDA_VISIBLE_DEVICES"
        visible_gpu_count=${#_visible_gpu_arr[@]}
        if [ "$visible_gpu_count" -lt "$total_split" ]; then
            echo "CUDA_VISIBLE_DEVICES exposes $visible_gpu_count GPU(s): '$CUDA_VISIBLE_DEVICES', but split mode requires $total_split."
            echo "Fix CUDA_VISIBLE_DEVICES or reduce ACTOR_ROLLOUT_GPUS/REWARD_ASYNC_NUM_GPUS."
            exit 1
        fi
    fi
    n_gpus_per_node=$ACTOR_ROLLOUT_GPUS
fi

dapo_train_path=../data/math_data/dapo-math-17k.parquet
base_val_path=${FLOWRL_MATH_VAL_PATH:-../data/math_data/validation.parquet}
eval_keep_sources=${FLOWRL_KEEP_EVAL_SOURCES:-aime2024,aime2025,gpqa}
eval_cache_dir=${FLOWRL_EVAL_CACHE_DIR:-/tmp}
filter_eval_sources=${FLOWRL_FILTER_EVAL_SOURCES:-1}
r1_test_path=$base_val_path

if [ "$filter_eval_sources" = "1" ]; then
    if [ ! -f "$base_val_path" ]; then
        echo "Validation file not found: $base_val_path"
        exit 1
    fi

    mkdir -p "$eval_cache_dir"
    eval_mix_path="$eval_cache_dir/flowrl_eval_aime24_aime25_gpqa.parquet"
    FLOWRL_BASE_VAL_PATH="$base_val_path" \
    FLOWRL_KEEP_EVAL_SOURCES="$eval_keep_sources" \
    FLOWRL_EVAL_MIX_PATH="$eval_mix_path" \
    python3 - <<'PY'
import os
import pandas as pd

base_path = os.environ["FLOWRL_BASE_VAL_PATH"]
keep_sources_raw = [s.strip() for s in os.environ.get("FLOWRL_KEEP_EVAL_SOURCES", "aime2024,aime2025,gpqa").split(",") if s.strip()]
output_path = os.environ["FLOWRL_EVAL_MIX_PATH"]

source_aliases = {
    "aime2024": {"aime2024", "aime-2024", "aime_2024"},
    "aime2025": {"aime2025", "aime-2025", "aime_2025"},
    "gpqa": {"gpqa", "gpqa_diamond", "idavidrein/gpqa"},
}
keep_normalized = set()
for source in keep_sources_raw:
    normalized = source.strip().lower()
    if not normalized:
        continue
    keep_normalized.add(normalized)
    keep_normalized.update(source_aliases.get(normalized, set()))

base_df = pd.read_parquet(base_path)
if "data_source" not in base_df.columns:
    raise ValueError(f"Validation parquet is missing required column: data_source ({base_path})")

base_df = base_df.copy()
base_df["__source_norm__"] = base_df["data_source"].astype(str).str.strip().str.lower()
filtered_df = base_df[base_df["__source_norm__"].isin(keep_normalized)].drop(columns=["__source_norm__"]).reset_index(drop=True)

if len(filtered_df) == 0:
    raise ValueError(f"No validation rows left after filtering to sources: {sorted(keep_normalized)}")

filtered_df.to_parquet(output_path, index=False)

counts = filtered_df["data_source"].value_counts(dropna=False)
print("[INFO] Eval data_source mix (filtered):")
for source, count in counts.items():
    print(f"[INFO]   {source}: {count}")
print(f"[INFO] Wrote eval file: {output_path}")
PY
    r1_test_path="$eval_mix_path"
fi

base_experiment_name="flowrl_qwen_7b_math"
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-8192}
train_batch_size=${TRAIN_BATCH_SIZE:-512}
rollout_n=${ROLLOUT_N:-8}
rollout_gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTIL:-0.6}
rollout_max_num_seqs=${ROLLOUT_MAX_NUM_SEQS:-1024}
actor_param_offload=${ACTOR_PARAM_OFFLOAD:-false}
actor_optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD:-false}
ref_param_offload=${REF_PARAM_OFFLOAD:-false}
resume_mode=${RESUME_MODE:-auto}
use_rule_reward_for_eval=${USE_RULE_REWARD_FOR_EVAL:-0}

# Reward switch:
# - REWARD_MODE=rule: original rule-based reward
# - REWARD_MODE=log_ratio_rm: beta * log(pi/pi_ref) using fine-tuned RM checkpoint
REWARD_MODE=${REWARD_MODE:-log_ratio_rm}

# Log-ratio reward model settings (used when REWARD_MODE=log_ratio_rm)
RM_POLICY_MODEL=${RM_POLICY_MODEL:-/workspace/checkpoints/checkpoint-800}
RM_REFERENCE_MODEL=${RM_REFERENCE_MODEL:-$PRETRAINED_MODEL}
RM_TOKENIZER_PATH=${RM_TOKENIZER_PATH:-$PRETRAINED_MODEL}
RM_BETA=${RM_BETA:-0.001} # 0.001
RM_MICRO_BSZ=${RM_MICRO_BSZ:-1}
RM_DEVICE_MAP=${RM_DEVICE_MAP:-auto}
RM_TORCH_DTYPE=${RM_TORCH_DTYPE:-bfloat16}
RM_OFFLOAD_FOLDER=${RM_OFFLOAD_FOLDER:-/tmp/verl_reward_offload}
RM_MAX_GPU_MEMORY=${RM_MAX_GPU_MEMORY:-2GiB}
RM_FORCE_CPU=${RM_FORCE_CPU:-0}
if [ "$RM_FORCE_CPU" = "1" ]; then
    RM_DEVICE_MAP=cpu
    RM_TORCH_DTYPE=float32
fi
# Safety profile for log-ratio RM to avoid "appears stuck" runs.
# Set LOG_RATIO_SAFE_MODE=0 to disable these conservative overrides.
LOG_RATIO_SAFE_MODE=${LOG_RATIO_SAFE_MODE:-1}
SAFE_TRAIN_BATCH_SIZE=${SAFE_TRAIN_BATCH_SIZE:-64}
SAFE_ROLLOUT_N=${SAFE_ROLLOUT_N:-2}
SAFE_RM_MAX_SEQ_LEN=${SAFE_RM_MAX_SEQ_LEN:-2048}
# Single-GPU fallback profile (mainly for smoke testing).
# Auto-enabled when n_gpus_per_node==1 unless explicitly set.
if [ "$n_gpus_per_node" = "1" ]; then
    SINGLE_GPU_TEST_MODE=${SINGLE_GPU_TEST_MODE:-1}
else
    SINGLE_GPU_TEST_MODE=${SINGLE_GPU_TEST_MODE:-0}
fi
SINGLE_GPU_MAX_PROMPT_LENGTH=${SINGLE_GPU_MAX_PROMPT_LENGTH:-1024}
SINGLE_GPU_MAX_RESPONSE_LENGTH=${SINGLE_GPU_MAX_RESPONSE_LENGTH:-1024}
SINGLE_GPU_MAX_TOKENS_PER_GPU=${SINGLE_GPU_MAX_TOKENS_PER_GPU:-2048}
SINGLE_GPU_TRAIN_BATCH_SIZE=${SINGLE_GPU_TRAIN_BATCH_SIZE:-8}
SINGLE_GPU_ROLLOUT_N=${SINGLE_GPU_ROLLOUT_N:-1}
SINGLE_GPU_VLLM_GPU_UTIL=${SINGLE_GPU_VLLM_GPU_UTIL:-0.90}
SINGLE_GPU_MAX_NUM_SEQS=${SINGLE_GPU_MAX_NUM_SEQS:-16}
# Logging switch:
# - LOGGER_MODE=console: no wandb
# - LOGGER_MODE=wandb: wandb only
# - LOGGER_MODE=both: console + wandb
LOGGER_MODE=${LOGGER_MODE:-console}
WANDB_MODE=${WANDB_MODE:-online}
VERL_WANDB_INIT_TIMEOUT=${VERL_WANDB_INIT_TIMEOUT:-600}
VERL_WANDB_INIT_RETRIES=${VERL_WANDB_INIT_RETRIES:-3}
VERL_WANDB_RETRY_SLEEP=${VERL_WANDB_RETRY_SLEEP:-15}
VERL_WANDB_DISABLE_ON_ERROR=${VERL_WANDB_DISABLE_ON_ERROR:-1}
export WANDB_MODE VERL_WANDB_INIT_TIMEOUT VERL_WANDB_INIT_RETRIES VERL_WANDB_RETRY_SLEEP VERL_WANDB_DISABLE_ON_ERROR
# Testing profile:
# TESTING_MODE=1 enables lightweight settings to quickly verify end-to-end flow.
TESTING_MODE=${TESTING_MODE:-0}
# Disable validation (val_before_train and periodic test) for heavy-RM runs.
DISABLE_VAL=${DISABLE_VAL:-0}
# Ultra-low-memory profile switch.
MIN_MEM_MODE=${MIN_MEM_MODE:-0}
# Keep reward on GPU in MIN_MEM_MODE when split mode is enabled.
MIN_MEM_REWARD_GPU=${MIN_MEM_REWARD_GPU:-0}
# Clean up stale Ray processes from previous interrupted runs before launch.
RAY_CLEAN_START=${RAY_CLEAN_START:-1}

case "$LOGGER_MODE" in
  console)
    trainer_logger="['console']"
    ;;
  wandb)
    trainer_logger="['wandb']"
    ;;
  both)
    trainer_logger="['console','wandb']"
    ;;
  *)
    echo "Unsupported LOGGER_MODE: $LOGGER_MODE"
    echo "Use LOGGER_MODE=console or LOGGER_MODE=wandb or LOGGER_MODE=both"
    exit 1
    ;;
esac

case "$REWARD_MODE" in
  rule)
    reward_overrides=(
      reward_model.reward_manager=naive
      custom_reward_function.path=null
      custom_reward_function.name=compute_score
    )
    ;;
  log_ratio_rm)
    reward_overrides=(
      reward_model.reward_manager=batch
      custom_reward_function.path=/workspace/verl_FlowRL/verl/trainer/ppo/log_ratio_reward.py
      custom_reward_function.name=compute_score
      custom_reward_function.reward_kwargs.policy_model_path=$RM_POLICY_MODEL
      custom_reward_function.reward_kwargs.reference_model_path=$RM_REFERENCE_MODEL
      custom_reward_function.reward_kwargs.tokenizer_path=$RM_TOKENIZER_PATH
      custom_reward_function.reward_kwargs.beta=$RM_BETA
      custom_reward_function.reward_kwargs.micro_batch_size=$RM_MICRO_BSZ
      custom_reward_function.reward_kwargs.device_map=$RM_DEVICE_MAP
      custom_reward_function.reward_kwargs.torch_dtype=$RM_TORCH_DTYPE
      custom_reward_function.reward_kwargs.offload_folder=$RM_OFFLOAD_FOLDER
      custom_reward_function.reward_kwargs.max_gpu_memory=$RM_MAX_GPU_MEMORY
      custom_reward_function.reward_kwargs.normalize_by_length=False
      +custom_reward_function.reward_kwargs.reward_clip_min=-1.0
    )
    ;;
  *)
    echo "Unsupported REWARD_MODE: $REWARD_MODE"
    echo "Use REWARD_MODE=rule or REWARD_MODE=log_ratio_rm"
    exit 1
    ;;
esac

experiment_name="${base_experiment_name}_${REWARD_MODE}"
OUTPUT_DIR=checkpoints/FlowRL/math/7B/$experiment_name

safe_overrides=()
if [ "$REWARD_MODE" = "log_ratio_rm" ] && [ "$LOG_RATIO_SAFE_MODE" = "1" ]; then
    safe_overrides=(
        reward_model.launch_reward_fn_async=True
        data.train_batch_size=$SAFE_TRAIN_BATCH_SIZE
        actor_rollout_ref.rollout.n=$SAFE_ROLLOUT_N
        custom_reward_function.reward_kwargs.max_seq_len=$SAFE_RM_MAX_SEQ_LEN
        custom_reward_function.reward_kwargs.device_map=cpu
        custom_reward_function.reward_kwargs.torch_dtype=float32
    )
fi

option1_overrides=()
if [ "$OPTION1_SPLIT_MODE" = "1" ]; then
    option1_overrides=(
        reward_model.launch_reward_fn_async=True
        reward_model.reward_fn_async_num_workers=$REWARD_ASYNC_NUM_GPUS
        reward_model.reward_fn_async_num_gpus=1
        reward_model.reward_fn_async_num_cpus=1
        reward_model.reward_fn_async_warmup=True
        reward_model.reward_fn_async_warmup_timeout_s=600
        actor_rollout_ref.rollout.layered_summon=True
        custom_reward_function.reward_kwargs.device_map=cuda
        custom_reward_function.reward_kwargs.torch_dtype=bfloat16
    )
fi

single_gpu_overrides=()
if [ "$SINGLE_GPU_TEST_MODE" = "1" ]; then
    single_gpu_overrides=(
        data.max_prompt_length=$SINGLE_GPU_MAX_PROMPT_LENGTH
        data.max_response_length=$SINGLE_GPU_MAX_RESPONSE_LENGTH
        data.train_batch_size=$SINGLE_GPU_TRAIN_BATCH_SIZE
        actor_rollout_ref.rollout.n=$SINGLE_GPU_ROLLOUT_N
        actor_rollout_ref.rollout.max_num_batched_tokens=$SINGLE_GPU_MAX_TOKENS_PER_GPU
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$SINGLE_GPU_MAX_TOKENS_PER_GPU
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$SINGLE_GPU_MAX_TOKENS_PER_GPU
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$SINGLE_GPU_MAX_TOKENS_PER_GPU
        actor_rollout_ref.rollout.gpu_memory_utilization=$SINGLE_GPU_VLLM_GPU_UTIL
        actor_rollout_ref.rollout.max_num_seqs=$SINGLE_GPU_MAX_NUM_SEQS
        actor_rollout_ref.actor.fsdp_config.param_offload=True
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
        actor_rollout_ref.ref.fsdp_config.param_offload=True
    )
fi

testing_overrides=()
if [ "$TESTING_MODE" = "1" ]; then
    experiment_name="${experiment_name}_test"
    OUTPUT_DIR=checkpoints/FlowRL/math/7B/$experiment_name
    testing_overrides=(
        trainer.val_before_train=False
        trainer.test_freq=-1
        data.train_batch_size=16
        data.val_batch_size=8
        data.max_response_length=2048
        actor_rollout_ref.rollout.n=1
        actor_rollout_ref.rollout.load_format=dummy_hf
        actor_rollout_ref.rollout.max_num_batched_tokens=4096
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096
    )
    if [ "$REWARD_MODE" = "log_ratio_rm" ]; then
        testing_overrides+=(
            custom_reward_function.reward_kwargs.max_seq_len=2048
            custom_reward_function.reward_kwargs.micro_batch_size=2
            custom_reward_function.reward_kwargs.device_map=cpu
            custom_reward_function.reward_kwargs.torch_dtype=float32
        )
    fi
fi

validation_overrides=()
if [ "$DISABLE_VAL" = "1" ]; then
    validation_overrides=(
        trainer.val_before_train=False
        trainer.test_freq=-1
    )
fi

minmem_overrides=()
if [ "$MIN_MEM_MODE" = "1" ]; then
    minmem_overrides=(
        trainer.val_before_train=False
        trainer.test_freq=-1
        data.max_prompt_length=256
        data.max_response_length=256
        data.train_batch_size=2
        actor_rollout_ref.actor.use_torch_compile=False
        actor_rollout_ref.actor.porj_layer=1
        actor_rollout_ref.rollout.name=hf
        actor_rollout_ref.rollout.n=1
        actor_rollout_ref.actor.ppo_mini_batch_size=4
        actor_rollout_ref.rollout.max_num_batched_tokens=512
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=512
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=512
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=512
        actor_rollout_ref.rollout.max_num_seqs=4
        actor_rollout_ref.rollout.gpu_memory_utilization=0.92
        actor_rollout_ref.model.enable_activation_offload=True
        actor_rollout_ref.actor.fsdp_config.param_offload=False
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
        actor_rollout_ref.ref.fsdp_config.param_offload=False
        custom_reward_function.reward_kwargs.max_seq_len=256
        custom_reward_function.reward_kwargs.micro_batch_size=1
    )

    if [ "$MIN_MEM_REWARD_GPU" = "1" ] && [ "$OPTION1_SPLIT_MODE" = "1" ] && [ "$REWARD_ASYNC_NUM_GPUS" -ge 1 ]; then
        minmem_overrides+=(
            reward_model.launch_reward_fn_async=True
            reward_model.reward_fn_async_num_workers=$REWARD_ASYNC_NUM_GPUS
            reward_model.reward_fn_async_num_gpus=1
            reward_model.reward_fn_async_num_cpus=1
            reward_model.reward_fn_async_warmup=True
            reward_model.reward_fn_async_warmup_timeout_s=600
            custom_reward_function.reward_kwargs.device_map=cuda
            custom_reward_function.reward_kwargs.torch_dtype=bfloat16
        )
    else
        minmem_overrides+=(
            reward_model.launch_reward_fn_async=False
            custom_reward_function.reward_kwargs.device_map=cpu
            custom_reward_function.reward_kwargs.torch_dtype=float32
        )
    fi
fi

set -x
echo "[INFO] REWARD_MODE=$REWARD_MODE LOG_RATIO_SAFE_MODE=$LOG_RATIO_SAFE_MODE TESTING_MODE=$TESTING_MODE MIN_MEM_MODE=$MIN_MEM_MODE MIN_MEM_REWARD_GPU=$MIN_MEM_REWARD_GPU"
echo "[INFO] SINGLE_GPU_TEST_MODE=$SINGLE_GPU_TEST_MODE n_gpus_per_node=$n_gpus_per_node"
echo "[INFO] OPTION1_SPLIT_MODE=$OPTION1_SPLIT_MODE ACTOR_ROLLOUT_GPUS=$n_gpus_per_node REWARD_ASYNC_NUM_GPUS=$REWARD_ASYNC_NUM_GPUS"
echo "[INFO] TRAIN_BATCH_SIZE=$train_batch_size ROLLOUT_N=$rollout_n MAX_PROMPT_LENGTH=$max_prompt_length MAX_RESPONSE_LENGTH=$max_response_length ROLLOUT_GPU_MEMORY_UTIL=$rollout_gpu_memory_utilization ROLLOUT_MAX_NUM_SEQS=$rollout_max_num_seqs"
echo "[INFO] ACTOR_PARAM_OFFLOAD=$actor_param_offload ACTOR_OPTIMIZER_OFFLOAD=$actor_optimizer_offload REF_PARAM_OFFLOAD=$ref_param_offload"
echo "[INFO] RESUME_MODE=$resume_mode"
echo "[INFO] USE_RULE_REWARD_FOR_EVAL=$use_rule_reward_for_eval"
echo "[INFO] EVAL_FILES=$r1_test_path"
echo "[INFO] LOGGER_MODE=$LOGGER_MODE WANDB_MODE=$WANDB_MODE VERL_WANDB_INIT_TIMEOUT=$VERL_WANDB_INIT_TIMEOUT"
echo "[INFO] Output dir: $OUTPUT_DIR"

if [ "$RAY_CLEAN_START" = "1" ]; then
    ray stop --force >/dev/null 2>&1 || true
fi

TRAIN_PID=""
cleanup() {
    echo "[INFO] Caught interrupt. Stopping training and Ray workers..."
    if [ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null; then
        kill -INT "$TRAIN_PID" 2>/dev/null || true
        sleep 1
        kill -TERM "$TRAIN_PID" 2>/dev/null || true
    fi
    ray stop --force >/dev/null 2>&1 || true
}
trap 'cleanup; exit 130' INT TERM

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    +algorithm.adv_clip_min_before_normalization=-5.0 \
    data.train_files=$dapo_train_path \
    data.val_files=$r1_test_path \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.truncation='left' \
    +actor_rollout_ref.actor.tb_type=tempered_important_sampling \
    +actor_rollout_ref.actor.porj_layer=3 \
    actor_rollout_ref.model.path=$PRETRAINED_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=$actor_param_offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$actor_optimizer_offload \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
    actor_rollout_ref.rollout.gpu_memory_utilization=$rollout_gpu_memory_utilization \
    actor_rollout_ref.rollout.max_num_seqs=$rollout_max_num_seqs \
    actor_rollout_ref.rollout.n=$rollout_n \
    actor_rollout_ref.ref.fsdp_config.param_offload=$ref_param_offload \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger="$trainer_logger" \
    trainer.project_name='FlowRL' \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$n_nodes \
    trainer.resume_mode=$resume_mode \
    reward_model.use_rule_reward_for_eval=$use_rule_reward_for_eval \
    trainer.save_freq=$save_freq \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    "${reward_overrides[@]}" \
    "${safe_overrides[@]}" \
    "${single_gpu_overrides[@]}" \
    "${testing_overrides[@]}" \
    "${validation_overrides[@]}" \
    "${option1_overrides[@]}" \
    "${minmem_overrides[@]}" \
    "$@" &

TRAIN_PID=$!
wait "$TRAIN_PID"
train_exit_code=$?
trap - INT TERM
exit "$train_exit_code"
