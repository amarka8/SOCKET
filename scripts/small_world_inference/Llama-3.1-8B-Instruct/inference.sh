output_dir_root=$1

task="longbench"
train_dataset="LongAlpaca-12k"
eval_dataset="qasper"
model="Llama-3.1-8B-Instruct"
method="SmallWorld"

export COLOR_PRINT=1
# ADD PATH TO TRITON CACHE HERE
export TRITON_CACHE_DIR= # ADD PATH TO TRITON CACHE HERE
export CHAT=1

# CD INTO DIRECTORY FROM WHICH PATHS BELOW ARE RELATIVE (e.g. ../pipeline)
cd /path/to/directory || exit

deepspeed --master_port 27501 pipeline/train_quest/main.py \
    --exp_desc ${task}_${train_dataset}_${model}_${method} \
    --pipeline_config_dir config/pipeline_config/${method}/${model}/${model}-inference-32hadamard.json \
    --eval_config_dir config/eval_config/${task}/${eval_dataset}.json \
    --train_config_dir config/train_config/train/${train_dataset}.json \
    --output_folder_dir ${task}/${method}/${model}/${train_dataset} \
    --deepspeed_config config/deepspeed_config/zero3_inference.json

