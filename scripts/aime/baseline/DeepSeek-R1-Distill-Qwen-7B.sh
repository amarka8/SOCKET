cuda_devices=$1
output_dir_root=$2

task="aime"
dataset="aime"
model="DeepSeek-R1-Distill-Qwen-7B"
method="baseline"

export COLOR_PRINT=1
# export CHAT=1

CUDA_VISIBLE_DEVICES=${cuda_devices} python pipeline/baseline/main.py \
    --exp_desc ${task}_${dataset}_${model}_${method} \
    --pipeline_config_dir config/pipeline_config/${method}/${task}/${model}/${model}.json \
    --eval_config_dir config/eval_config/${task}/${dataset}.json \
    --output_folder_dir ${output_dir_root}/${task}/${method}/${model}/${dataset}

