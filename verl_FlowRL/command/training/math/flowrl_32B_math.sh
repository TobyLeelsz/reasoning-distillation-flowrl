#!/bin/bash

PRETRAINED_MODEL=../pre_trained_model/Qwen/Qwen2.5-32B
n_nodes=4
n_gpus_per_node=8
tensor_model_parallel_size=4
save_freq=50

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

experiment_name="flowrl_qwen_32b_math"
max_prompt_length=2048
max_response_length=8192
OUTPUT_DIR=../checkpoints/FlowRL/math/32B/$experiment_name

echo "[INFO] EVAL_FILES=$r1_test_path"
set -x

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$dapo_train_path \
    data.val_files=$r1_test_path \
    data.train_batch_size=512 \
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
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='FlowRL' \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$n_nodes \
    trainer.resume_mode=auto \
    trainer.save_freq=$save_freq \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    $@
