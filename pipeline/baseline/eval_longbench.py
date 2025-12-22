import logging
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

import eval.longbench_utils.eval_long_bench as longbench_eval
from pipeline.baseline.utils import initialize_model_tokenizer

logger = logging.getLogger("main")


def _strip_special_tokens(text: str, tokenizer) -> str:
    for special_token in tokenizer.special_tokens_map.values():
        if isinstance(special_token, list):
            for special_tok in special_token:
                text = text.replace(special_tok, "")
        else:
            text = text.replace(special_token, "")
    return text.strip()


def _truncate_middle(input_ids: List[int], max_len: int) -> List[int]:
    if max_len is None or len(input_ids) <= max_len:
        return input_ids
    keep_head = max_len // 2
    keep_tail = max_len - keep_head
    return input_ids[:keep_head] + input_ids[-keep_tail:]


def _build_input_ids(
    sample: Dict,
    eval_config: Dict,
    tokenizer,
    pipeline_config: Dict,
) -> List[int]:
    instruction = eval_config["instruction"]
    user_content = instruction.format(**sample)
    conversation = [{"role": "system", "content": "You are a useful assistant."}]
    conversation.append({"role": "user", "content": user_content})
    input_ids = tokenizer.apply_chat_template(
        conversation, tokenize=True, add_generation_prompt=True
    )
    max_len = pipeline_config.get("max_model_len")
    if max_len:
        input_ids = _truncate_middle(input_ids, max_len)
    return input_ids

@torch.inference_mode()
def _generate_predictions(
    dataset,
    model,
    tokenizer,
    pipeline_config: Dict,
    eval_config: Dict,
    ground_truths: List,
    all_classes,
) -> Tuple[List[str], float]:
    do_sample = pipeline_config["do_sample"]
    temperature = pipeline_config["temperature"] if do_sample else 0.0
    outputs = []
    total_score = 0.0
    pbar = tqdm(dataset, desc="Generating", dynamic_ncols=True)
    for idx, sample in enumerate(pbar):
        input_ids = _build_input_ids(sample, eval_config, tokenizer, pipeline_config)
        output_ids = model.generate(
            torch.as_tensor([input_ids]).cuda(),
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=eval_config["max_new_tokens"],
            top_p=pipeline_config["top_p"],
        )
        output_ids = output_ids[0][len(input_ids) :]
        output = tokenizer.decode(
            output_ids,
            spaces_between_special_tokens=False,
        )
        output = _strip_special_tokens(output, tokenizer)
        outputs.append(output)
        total_score += longbench_eval.scorer(
            eval_config["dataset"],
            [output],
            [ground_truths[idx]],
            all_classes,
        )
        print(output)
        avg_score = 100.0 * total_score / max(len(outputs), 1)
        pbar.set_description(f"Generating (score: {avg_score:.2f})")
    return outputs, total_score


def eval_longbench(config) -> Tuple[Dict, Dict]:
    pipeline_config = config["pipeline_params"]
    eval_config = config["eval_params"]
    dataset_name = eval_config["dataset"]

    dataset = longbench_eval.load_data(dataset_name)
    ground_truths = [example["answers"] for example in dataset]
    all_classes = dataset[0]["all_classes"]

    model, tokenizer = initialize_model_tokenizer(pipeline_config)
    model.tokenizer = tokenizer

    predictions, total_score = _generate_predictions(
        dataset=dataset,
        model=model,
        tokenizer=tokenizer,
        pipeline_config=pipeline_config,
        eval_config=eval_config,
        ground_truths=ground_truths,
        all_classes=all_classes,
    )

    avg_score = 100.0 * total_score / max(len(predictions), 1)
    processed_result = {
        "dataset": dataset_name,
        "score": avg_score,
        "outputs": predictions,
    }
    raw_result = {
        "total_score": total_score,
        "num_samples": len(predictions),
    }

    return processed_result, raw_result
