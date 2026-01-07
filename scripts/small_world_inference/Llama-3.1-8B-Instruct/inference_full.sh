cuda_devices=$1
output_dir_root=$2

task="longbench"
train_dataset="LongAlpaca-12k"
model="Llama-3.1-8B-Instruct"
method="SmallWorld"

export COLOR_PRINT=1
export TRITON_CACHE_DIR="/rhf/allocations/as143/el72"
export CHAT=1

datasets=(
    "narrativeqa"
    "qasper"
    "multifieldqa_en"
    "hotpotqa"
    "2wikimqa"
    "musique"
    "gov_report"
    "qmsum"
    "multi_news"
    "trec"
    "triviaqa"
    "samsum"
    "passage_retrieval_en"
    "lcc"
    "repobench-p"
    "passage_count"
)

for eval_dataset in "${datasets[@]}"; do
    CUDA_VISIBLE_DEVICES=${cuda_devices} deepspeed --master_port 27600 pipeline/train_quest/main.py \
        --exp_desc ${task}_${train_dataset}_${model}_${method} \
        --pipeline_config_dir config/pipeline_config/${method}/${model}/${model}-inference-32hadamard.json \
        --eval_config_dir config/eval_config/${task}/${eval_dataset}.json \
        --train_config_dir config/train_config/train/${train_dataset}.json \
        --output_folder_dir ${output_dir_root}/${task}/${method}/${model}/${train_dataset}/${eval_dataset} \
        --deepspeed_config config/deepspeed_config/zero3_inference.json
done
