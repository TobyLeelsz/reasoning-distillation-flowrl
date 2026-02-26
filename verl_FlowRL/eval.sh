# cd /workspace/verl_FlowRL
CUDA_VISIBLE_DEVICES=0,1,2,3 \
N_GPUS_PER_NODE=4 \
TP_SIZE=1 \
GEN_DEVICE_NAME=cuda \
MODEL_PATH=Qwen/Qwen2.5-7B \
PASS_AT_K=2 \
N_SAMPLES=2 \
AUTO_PREP_GPQA=1 \
AUTO_PREP_GSM8K=1 \
bash command/eval/mixed/flowrl_eval_gpqa_gsm8k_math500_pass2.sh
