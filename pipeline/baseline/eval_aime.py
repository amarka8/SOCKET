import json
import os
import sys
import random
import time
import shortuuid
import logging
from tqdm import tqdm
from typing import Dict, List, Optional

logger = logging.getLogger("main")


import torch
from datasets import load_dataset

from pipeline.baseline.utils import (
    initialize_model_tokenizer,

)

def generate_prompt(input):
    INSTRUCTION = f"""Return your final response within \\boxed{{}}. {input}"""
    return INSTRUCTION

def run_eval(
    model_id,
    questions,
    model,
    tokenizer,
    pipeline_config,
    eval_config
):
    num_gpus_total=pipeline_config['num_gpus_total']
    num_gpus_per_model=pipeline_config['num_gpus_per_model']
    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)
    processed_result, raw_result = {}, {}
    for i in range(0, len(questions), chunk_size):
        processed_result, raw_result = get_model_answers(
            model_id,
            questions[i : i + chunk_size],
            model=model,
            tokenizer=tokenizer,
            pipeline_config=pipeline_config,
            eval_config=eval_config
        )

    return processed_result, raw_result

@torch.inference_mode()
def get_model_answers(
    model_id,
    questions,
    model,
    tokenizer,
    pipeline_config,
    eval_config
):  
    do_sample = pipeline_config['do_sample']
    if not do_sample:
        temperature = 0.0 #force greedy
    else:
        temperature = pipeline_config["temperature"]
    step = 1
    outputs = []
    for question_idx, question in enumerate(tqdm(questions['Problem'])):
        torch.manual_seed(step)
        turns = []
        prompts = []
        
        prompt = question
        prompt = generate_prompt(prompt)
        prompts.append(prompt)
        input_ids = tokenizer([prompt]).input_ids
        output_ids = model.generate(
            torch.as_tensor(input_ids).cuda(),
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=eval_config['max_new_tokens'],
            top_p=pipeline_config['top_p']
        )
        
        output_ids = output_ids[0][len(input_ids[0]) :]
        output = tokenizer.decode(
            output_ids,
            spaces_between_special_tokens=False,
        )
        for special_token in tokenizer.special_tokens_map.values():
            if isinstance(special_token, list):
                for special_tok in special_token:
                    output = output.replace(special_tok, "")
            else:
                output = output.replace(special_token, "")
        output = output.strip()
        outputs.append(output)
        logger.info(output)
        import pdb; pdb.set_trace()

    processed_result = {
        "outputs": outputs
    }
    raw_result = ""

    return processed_result, raw_result

def eval_aime(config):
    pipeline_config = config['pipeline_params']
    eval_config = config['eval_params']
    model_id = f"{pipeline_config['model_name']}-level-{pipeline_config['level']}-win-{pipeline_config['window']}-guess-{pipeline_config['num_guesses']}"

    # Load data
    dataset = load_dataset(eval_config['dataset_path'])
    questions = dataset["train"]
    # Load model
    model, tokenizer = initialize_model_tokenizer(pipeline_config)
    model.tokenizer = tokenizer

    processed_result, raw_result = run_eval(
        model_id=model_id,
        questions=questions,
        model=model,
        tokenizer=tokenizer,
        pipeline_config=pipeline_config,
        eval_config=eval_config
    )

    return processed_result, raw_result
