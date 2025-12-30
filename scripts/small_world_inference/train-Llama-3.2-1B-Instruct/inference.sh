cuda_devices=$1
output_dir_root=$2

task="longbench"
train_dataset="LongAlpaca-12k"
eval_dataset="qasper"
model="Llama-3.2-1B-Instruct"
method="SmallWorld"

export COLOR_PRINT=1
export TRITON_CACHE_DIR="/rhf/allocations/as143/el72"
export CHAT=1

CUDA_VISIBLE_DEVICES=$1 deepspeed  --master_port 28500 pipeline/train_quest/main.py \
    --exp_desc ${task}_${train_dataset}_${model}_${method} \
    --pipeline_config_dir config/pipeline_config/${method}/${model}/${model}-inference-only.json \
    --eval_config_dir config/eval_config/${task}/${eval_dataset}.json \
    --train_config_dir config/train_config/train/${train_dataset}.json \
    --output_folder_dir ${output_dir_root}/${task}/${method}/${model}/${train_dataset} \
    --deepspeed_config config/deepspeed_config/zero3_inference.json

