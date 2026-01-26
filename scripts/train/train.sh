cuda_devices=$1
output_dir_root=$2

task="longbench"
dataset="qasper"
model="Meta-Llama-3-8B-Instruct"
method="train"

export COLOR_PRINT=1
# export CHAT=1

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes 1 --nproc_per_node 4 --master_port 28600 pipeline/train_quest/main.py \
    --exp_desc ${task}_${dataset}_${model}_${method} \
    --pipeline_config_dir config/pipeline_config/${method}/${model}/${model}.json \
    --eval_config_dir config/eval_config/${task}/${dataset}.json \
    --output_folder_dir ${output_dir_root}/${task}/${method}/${model}/${dataset}

