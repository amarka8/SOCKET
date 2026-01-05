cuda_devices=$1
output_dir_root=$2

task="longbench"
model="Llama-3.1-8B-Instruct"
method="baseline"

export COLOR_PRINT=1

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

for dataset in "${datasets[@]}"; do
    CUDA_VISIBLE_DEVICES=${cuda_devices} python pipeline/baseline/main.py \
        --exp_desc ${task}_${dataset}_${model}_${method} \
        --pipeline_config_dir config/pipeline_config/${method}/${task}/${model}/${model}.json \
        --eval_config_dir config/eval_config/${task}/${dataset}.json \
        --output_folder_dir ${output_dir_root}/${task}/${method}/${model}/${dataset}
done
