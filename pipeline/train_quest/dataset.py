# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import json
import os
import itertools
import random
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from datasets import load_dataset, Dataset

from tqdm import tqdm

import eval.longbench_utils.eval_long_bench as longbench_eval
from eval.longbench_utils.constants import LONGBENCH_DATASET
from pipeline.baseline.utils import initialize_model_tokenizer

def load_longctx_dataset(pipeline_config, dataset_config):
    print(f"[INFO] Loading raw dataset: {dataset_config['dataset_path']} ...")

    if dataset_config['dataset'] == "LongAlpaca-12k":
        ds = load_dataset(dataset_config['dataset_path'], split="train")
    elif dataset_config['dataset'] in LONGBENCH_DATASET:
        ds = longbench_eval.load_data(dataset_config['dataset'])
    else:
        raise ValueError(f"Unknown dataset: {dataset_config['dataset']}")
    _model, tokenizer = initialize_model_tokenizer(
        {
            'model_name': "meta-llama/Llama-3.2-3B-Instruct",
            "use_flash_attn": False
        }
    )

    data = []
    max_len = pipeline_config['max_model_len']
    for ex in tqdm(ds, desc="Preparing samples"):
        if dataset_config['dataset'] == "LongAlpaca-12k":
            prompt = dataset_config['instruction'].format(**ex)
            answer = ex.get("output", "")
        else:
            prompt = dataset_config['instruction'].format(**ex)
            answer = ex.get("answers", ex.get("answers", []))
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if len(prompt_ids) < max_len and len(prompt_ids) >= 300:
            data.append({
                "prompt": prompt,
                "answer": answer,
                "idx": len(data)
            })
    ds = Dataset.from_list(data)
    return ds


def get_dataset(tokenizer, pipeline_config, dataset_config, max_size=1000000000):

    def tokenize_sample(sample):
        if dataset_config['dataset'] in LONGBENCH_DATASET:
            prompt = dataset_config['instruction'].format(**sample)
            answer = sample.get("answer", sample.get("answers", ""))[0]
        elif dataset_config['dataset'] == "LongAlpaca-12k":
            prompt = sample['prompt']
            answer = sample["answer"]

        sample = {
            "prompt": prompt,
            "answer": answer,
            "idx": sample["idx"],
        }
        return sample

    if dataset_config['dataset'] in LONGBENCH_DATASET:
        ds = longbench_eval.load_data(dataset_config['dataset'])
        data = [
            {**dict(example), "idx": idx}
            for idx, example in enumerate(ds)
        ]
    elif dataset_config['dataset'] == "LongAlpaca-12k":
        ds = load_longctx_dataset(pipeline_config, dataset_config)
        max_len = pipeline_config['max_model_len']

        data = []
        for ex in ds:
            prompt_ids = tokenizer(ex.get("prompt", ""), add_special_tokens=False)["input_ids"]
            if len(prompt_ids) < max_len:
                data.append({
                    "prompt": ex.get("prompt", ""),
                    "answer": ex.get("answer", ""),
                    "idx": len(data),  # ensures sequential idx
                })

    else:
        raise ValueError(f"Unknown dataset: {dataset_config['dataset']}")

    keys = data[0].keys()
    dataset = Dataset.from_dict({k: [d[k] for d in data] for k in keys})

    if torch.cuda.device_count() > 1:
        if dist.get_rank() == 0:
            # print("rank: ")
            processed_dataset = [
                dataset.map(
                    tokenize_sample, remove_columns=list(dataset.features), num_proc=32
                )
            ]
            dist.broadcast_object_list(processed_dataset, src=0)
        else:
            processed_dataset = [None]
            dist.broadcast_object_list(processed_dataset, src=0)
        
        dataset = processed_dataset[0]

    else:
        dataset = dataset.map(
            tokenize_sample, remove_columns=list(dataset.features), num_proc=32
        )

    return dataset


@dataclass
class MyCollator:

    tokenizer: PreTrainedTokenizerBase
    latent_id: Optional[int] = None
    label_pad_token_id: Optional[int] = -100

    def __call__(self, features, return_tensors=None):

        assert self.tokenizer.padding_side == "right"

        """
        Pad the batch like this to maximize the reuse of kv cache.
        E.g.,
        
        xxxxxxxxxx<latent><latent>xxxxx--
        -----xxxxx<latent>xxxxxxxx-------
        ---xxxxxxx<latent><latent>xxxxxxx


        ("x" is word token, "-" is pad token)
        """
        max_length = max(len(feature["input_ids"]) for feature in features)

        for feature in features:
            n_tok_pad = max_length - len(feature["input_ids"])
            feature["prompt_len"] += n_tok_pad
            feature["position_ids"] = [0] * n_tok_pad + list(range(len(feature["input_ids"])))
            feature["input_ids"] = [self.tokenizer.pad_token_id] * n_tok_pad + feature["input_ids"]
            if "labels" in feature:
                feature["labels"] = [self.label_pad_token_id] * n_tok_pad + feature["labels"]
            feature["attention_mask"] = [0] * n_tok_pad + feature["attention_mask"]

        return_tensors = "pt"

        label_name = "label" if "label" in features[0].keys() else "labels"

        non_label_position_features = [
            {
                k: v
                for k, v in feature.items()
                if k != label_name and k != "position_ids"
            }
            for feature in features
        ]

        # run through tokenizer without labels to ensure no side effects
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            non_label_position_features,
            padding=True,
            pad_to_multiple_of=None,
            return_tensors=return_tensors,
        )

        labels = (
            [feature[label_name] for feature in features]
            if label_name in features[0].keys()
            else None
        )
        if labels is not None and all(label is None for label in labels):
            labels = None
        position_ids = (
            [feature["position_ids"] for feature in features]
            if "position_ids" in features[0].keys()
            else None
        )
        # we have to pad the labels and position_ids manually as we cannot rely on `tokenizer.pad`

        if labels is not None:
            max_label_length = max(len(l) for l in labels)

            batch["labels"] = [
                label + [self.label_pad_token_id] * (max_label_length - len(label))
                for label in labels
            ]
            batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)

        if position_ids is not None:
            max_pos_length = max(len(l) for l in position_ids)

            batch["position_ids"] = [
                position_id + [0] * (max_pos_length - len(position_id))
                for position_id in position_ids
            ]
            batch["position_ids"] = torch.tensor(
                batch["position_ids"], dtype=torch.int64
            )

        return batch


def get_val_dataset(
    base_dataset_valid,
    config,
    tokenizer
):
    def process_dataset(sample):
        conversation = [{"role": "system", "content": "You are a useful assistant."}]
        conversation.append({"role": "user", "content": sample["prompt"]})
        input_ids = tokenizer.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True
        )
        
        max_len = config['pipeline_params']['max_model_len']
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len // 2] + input_ids[-max_len // 2:]
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": list(range(len(input_ids))),
            "prompt_len": len(input_ids),
            "idx": sample["idx"],
        }

    return base_dataset_valid.map(
        process_dataset, remove_columns=list(base_dataset_valid.features), num_proc=32
    )


def get_train_dataset(
    base_dataset,
    tokenizer,
    config,
    shuffle=False,
):
    def process_dataset(sample):

        # Trim if exceed max_len
        prompt_ids = tokenizer(sample["prompt"], add_special_tokens=False)["input_ids"]
        if len(prompt_ids) > config['pipeline_params']['max_model_len']:
            max_len = config['pipeline_params']['max_model_len']
            prompt_ids = prompt_ids[:max_len // 2] + prompt_ids[-max_len // 2:]
            sample['prompt'] = tokenizer.decode(prompt_ids)
        
        conversation = [{"role": "system", "content": "You are a useful assistant."}]
        conversation.append({"role": "user", "content": sample["prompt"]})
        conversation.append({"role": "assistant", "content": sample["answer"]})
        prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
        input_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        # prefix: everything BEFORE assistant reply
        prefix_txt = tokenizer.apply_chat_template(
            conversation[:-1], tokenize=False, add_generation_prompt=False
        )
        prefix_ids = tokenizer(prefix_txt, add_special_tokens=False)["input_ids"]

        # mask non-assistant tokens
        labels = [-100] * len(prefix_ids) + input_ids[len(prefix_ids):]

        attention_mask = [1] * len(input_ids)
        sample = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "position_ids": list(range(len(input_ids))),
            "prompt_len": len(prefix_ids),
            "idx": sample["idx"],
        }
        
        return sample

    if torch.cuda.device_count() > 1:
        if dist.get_rank() == 0:
            processed_dataset = base_dataset.map(
                process_dataset, remove_columns=list(base_dataset.features), num_proc=32
            )
            if shuffle:
                processed_dataset = processed_dataset.shuffle()
            processed_dataset = [processed_dataset]
        else:
            processed_dataset = [None]
        dist.broadcast_object_list(processed_dataset, src=0)
        dataset = processed_dataset[0]

    else:
        processed_dataset = base_dataset.map(
            process_dataset, remove_columns=list(base_dataset.features), num_proc=32
        )
        if shuffle:
            processed_dataset = processed_dataset.shuffle()
        dataset = processed_dataset

    return dataset
