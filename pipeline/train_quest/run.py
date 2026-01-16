# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
torch.autograd.set_detect_anomaly(True)
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from collections import OrderedDict
from peft import LoraConfig, get_peft_model, PeftModel
import deepspeed

import eval.longbench_utils.eval_long_bench as longbench_eval
from eval.longbench_utils.constants import LONGBENCH_DATASET

from pipeline.train_quest.dataset import (
    get_dataset,
    get_train_dataset,
    get_val_dataset,
    MyCollator,
)
from pipeline.train_quest.modeling.modeling_llama import (
    LlamaForCausalLM
)
from pipeline.train_quest.CE import CE

from tqdm import tqdm
from copy import copy
import os, sys
import yaml
import json
import gc
import argparse
import functools
from pipeline.train_quest.utils import Config, set_seed
from torch.utils.checkpoint import checkpoint, set_checkpoint_debug_enabled

import os
import torch
import torch.distributed as dist

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

def is_rank0():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

@torch.no_grad()
def save_base_causallm(ds_engine, save_path, sub_prefix="base_causallm."):
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    full_sd = None
    if hasattr(ds_engine, "_zero3_consolidated_16bit_state_dict"):
        full_sd = ds_engine._zero3_consolidated_16bit_state_dict()

    if not is_rank0():
        return

    assert full_sd is not None, "Consolidation returned None on rank 0."

    cpu_sd = OrderedDict(
        (k[len(sub_prefix):], v.detach().cpu())
        for k, v in full_sd.items()
        if k.startswith(sub_prefix)
    )
    del full_sd
    torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tmp = save_path + ".tmp"
    torch.save(cpu_sd, tmp)
    os.replace(tmp, save_path)
    print(f"[rank0] Saved {save_path}")


def _model_device(mod):
    return next(mod.parameters()).device


def run(configs, args, logger):
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    rank = int(os.environ["RANK"])
    eval_params = configs['eval_params']
    pipeline_params = configs['pipeline_params']
    train_params = configs['train_params']
    n_gpu = torch.cuda.device_count()
    use_deepspeed = not pipeline_params["only_eval"]
    # init distributed environment
    if use_deepspeed:
        deepspeed.init_distributed(dist_backend='nccl')
    elif dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    ground_truth = None
    if eval_params['dataset'] in LONGBENCH_DATASET:
        ds = longbench_eval.load_data(eval_params['dataset'])
        ground_truth = [
            example['answers']
            for idx, example in enumerate(ds)
        ]
        all_classes = ds[0]["all_classes"]
    
    if rank == 0:
        print("Config:", configs)

    set_seed(41)
    save_dir = os.path.join(pipeline_params["save_path"])

    if not os.path.exists(save_dir) and rank == 0:
        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() == 0:
                os.makedirs(save_dir, exist_ok=True)
            dist.barrier()  # ensure everyone sees the dir
        else:
            os.makedirs(save_dir, exist_ok=True)

    cur_ckpts = os.listdir(save_dir)

    # check if the job is preempted and resumed.

    if len(cur_ckpts) > 0 and not pipeline_params["only_eval"]:
        # if there are previous checkpoints, and only_eval is False
        # it means the previous run was preempted and the program is restarted.
        # need to find the latest checkpoint and resume from that.

        if rank == 0:
            print(
                f"Warning: found previous run and gonna resume from that. the inputted `resume` argument is ignored!"
            )

        checkpoints = [f for f in cur_ckpts if f.startswith("checkpoint_")]
        checkpoints.sort(key=lambda x: int(x.split("_")[1]))

        # Get the last item in the sorted list
        latest_checkpoint = checkpoints[-1] if checkpoints else None
        pipeline_params["resume"] = int(latest_checkpoint.split("_")[1])
        load_dir = os.path.join(pipeline_params["save_path"], latest_checkpoint)

        pipeline_params["load_model_path"] = load_dir
        epoch = pipeline_params["resume"]
        print(f"Loading from previous run epoch_{epoch}!")

    model_name = pipeline_params["model_name"]
    use_smallworld = pipeline_params.get("method") == "smallworld"
    is_llama_instruct = model_name in {
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
    }

    if use_smallworld and is_llama_instruct:
        llama_config = AutoConfig.from_pretrained(model_name)

        llama_config.use_topk_masker   = True
        llama_config.topk_chunk_len    = 32
        llama_config.topk_k            = 64
        llama_config.topk_hidden       = 256
        llama_config.topk_tau          = 1.5
        llama_config.topk_soft_alpha   = 8.0
        llama_config.random_walk_hadamard_dim = pipeline_params.get("random_walk_hadamard_dim", 128)
        model = LlamaForCausalLM.from_pretrained(
            model_name, config=llama_config, torch_dtype=torch.bfloat16
        )
        model.set_masker_mode(configs['pipeline_params']["train_mode"])
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=None, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    loaded = False

    if pipeline_params["load_model_path"] is not None:
        saved_weights = torch.load(
            pipeline_params["load_model_path"], map_location="cpu"
        )

        new_state_dict = OrderedDict()
        for k, v in saved_weights.items():
            name = k.replace("module.", "") # remove `module.`
            new_state_dict[name] = v
        saved_weights = new_state_dict
        loaded = True
        print(model.load_state_dict(saved_weights, strict=False))

    raw_model = CE(model, tokenizer)
    raw_model.train()
    # Prepare optimizer
    def _unique_named_params(mod):
        seen, out = set(), []
        for n, p in mod.named_parameters():
            if not p.requires_grad:
                continue
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            out.append((n, p))
        return out

    if use_deepspeed:
        # 1) find the base LM inside CE(model, tokenizer)
        base = getattr(raw_model, "base_causallm", None) or getattr(raw_model, "model", None)
        if base is None:
            base = raw_model  # fallback

        # 2) unique params from the BASE model
        base_named = _unique_named_params(base)
        base_ids   = {id(p) for _, p in base_named}

        # 3) include CE-specific trainables (if any) that are NOT the same Parameter objects
        ce_extras = []
        for n, p in raw_model.named_parameters():
            if p.requires_grad and id(p) not in base_ids:
                ce_extras.append((f"CE.{n}", p))
                base_ids.add(id(p))

        uniq = base_named + ce_extras

        # 4) split decay / no_decay (bias and 1D params → no_decay)
        no_decay_keys = ('bias',)
        decay = [p for n, p in uniq if not (p.ndim == 1 or any(k in n for k in no_decay_keys))]
        nodec = [p for n, p in uniq if      (p.ndim == 1 or any(k in n for k in no_decay_keys))]

        optimizer_grouped_parameters = [
            {"params": decay, "weight_decay": 0.01},
            {"params": nodec, "weight_decay": 0.0},
        ]

        ids = []
        for g in optimizer_grouped_parameters:
            ids.extend([id(p) for p in g["params"]])
        assert len(ids) == len(set(ids)), "[fatal] duplicate Parameter objects in optimizer_grouped_parameters"

        model_engine, optimizer, _, _ = deepspeed.initialize(
            args=args,
            model=raw_model,
            model_parameters=optimizer_grouped_parameters,
            dist_init_required=True)

        print("propagate deepspeed-config settings to client settings")
        args.train_batch_size = pipeline_params["batch_size_training"] = model_engine.train_micro_batch_size_per_gpu()
        args.gradient_accumulation_steps = model_engine.gradient_accumulation_steps()
        args.fp16 = model_engine.fp16_enabled()
        args.print_steps = model_engine.steps_per_print()
        args.learning_rate = model_engine.get_lr()[0]
        args.wall_clock_breakdown = model_engine.wall_clock_breakdown()

        model_module = model_engine.module
        if rank == 0:
            print(model_engine)
    else:
        model_engine = raw_model
        model_module = raw_model
        optimizer = None
        model_module.to(torch.device(f"cuda:{local_rank}"))

    base_dataset_valid = get_dataset(
        tokenizer, pipeline_params, eval_params, max_size=100000000
    )
    if not pipeline_params["only_eval"]:
        base_dataset_train = get_dataset(
            tokenizer, pipeline_params, train_params, max_size=100000000
        )


    max_new_tokens = eval_params["max_new_tokens"]
    total_train_steps = 0
    best_acc = 0
    collator = MyCollator(tokenizer, label_pad_token_id=-100)
    processed_result = None
    raw_result = None

    for epoch in range(pipeline_params["resume"], pipeline_params["num_epochs"]):

        dataset_gen_val = get_val_dataset(
            base_dataset_valid,
            configs,
            tokenizer
        )

        valid_gen_dataloader = torch.utils.data.DataLoader(
            dataset_gen_val,
            num_workers=1,
            pin_memory=True,
            batch_size=1,
            collate_fn=collator,
            sampler=DistributedSampler(dataset_gen_val, shuffle=False),
        )

        if not pipeline_params["only_eval"]:
            dataset_train = get_train_dataset(
                base_dataset_train,
                tokenizer,
                configs,
                shuffle=True,
            )

            train_dataloader = torch.utils.data.DataLoader(
                dataset_train,
                num_workers=1,
                shuffle=False,
                pin_memory=True,
                batch_size=pipeline_params["batch_size_training"],
                collate_fn=collator,
                sampler=DistributedSampler(dataset_train, shuffle=True),
            )

            total_length = int(len(train_dataloader) / args.train_batch_size)
            pbar = tqdm(
                colour="blue",
                desc=f"Training Epoch: {epoch+1}",
                total=total_length,
                dynamic_ncols=True,
            )
            model_module.train()
            num_step = 0
            # Set up for training
            for step, batch in enumerate(train_dataloader):
                num_step += 1
                if step == 0 and rank == 0:
                    print("logging training data")
                    cur_bs = len(batch["input_ids"])
                    text_str = ""
                    for data_idx in range(cur_bs):
                        for token_idx in range(len(batch["input_ids"][data_idx])):
                            text_str += (
                                str(batch["input_ids"][data_idx][token_idx].item())
                                + " "
                                + str(batch["labels"][data_idx][token_idx].item())
                                + " "
                                + tokenizer.decode(
                                    batch["input_ids"][data_idx][token_idx]
                                )
                                + "\n"
                            )
                        text_str += "====" * 10 + "\n"

                total_train_steps += 1
                device = _model_device(model_module)
                batch = {
                    key: batch[key].to(device) for key in batch.keys() if key != "idx"
                }
                outputs = model_engine(**batch, config=configs)
                loss = outputs.loss
                if n_gpu > 1:
                    loss = loss.mean()

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                model_engine.backward(loss)

                if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    model_engine.step()
                    pbar.update(num_step)
                    num_step = 0

                epochs = pipeline_params["num_epochs"]
                grad_steps = pipeline_params["gradient_accumulation_steps"]

                pbar.set_description(
                    f"Training Epoch: {epoch+1}/{epochs}, batch {step}/{len(train_dataloader)} "
                    f"completed (loss: {round(float(loss.detach().float() * grad_steps), 4)}"
                )
                loss = None
            pbar.close()

            if not pipeline_params["only_eval"] and ((epoch + 1) % pipeline_params['eval_period'] == 0 or epoch == pipeline_params["num_epochs"] - 1):
                save_path = os.path.join(save_dir, f"checkpoint_{epoch+1}")
                save_base_causallm(model_engine, save_path)
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()


        if (epoch + 1) % pipeline_params['eval_period'] == 0 or epoch == pipeline_params["num_epochs"] - 1:
            # val generation accuracy
            total_length = len(valid_gen_dataloader)

            pbar = tqdm(colour="blue", desc=f"Test Accuracy", total=total_length, dynamic_ncols=True)
            score = torch.tensor(0.0, device=_model_device(model_module))
            total = torch.tensor(0.0, device=_model_device(model_module))
            predictions = []
            prediction_indices = []
            with torch.no_grad():
                model_module.eval()
                for idx, batch in enumerate(valid_gen_dataloader):
                    test_idx = batch["idx"][0]

                    batch = {
                        k: v.to(_model_device(model_module))
                        for k, v in batch.items()
                        if v != None and k not in ["idx", "position_ids"]
                    }

                    assert len(batch["input_ids"]) == 1
                    total += 1

                    # synced_gpus=True in FSDP mode, as we need to keep # forward pass the same on each device
                    # Inference has bug: after generate the first token, attention_mask is not correctly generated
                    answer = model_module.generate(
                        **batch,
                        config=configs,
                        max_new_tokens=max_new_tokens,
                        synced_gpus=use_deepspeed,
                    )
                    predictions.append(answer)
                    prediction_indices.append(int(test_idx))
                    score += longbench_eval.scorer(
                        eval_params['dataset'],
                        [answer],
                        [ground_truth[test_idx]],
                        all_classes
                        )
                    if idx < 50 and rank == 0:
                        # print some examples
                        print(
                            f"Question {test_idx}: Answer = '{answer}'"
                        )
                        print(f"Extracted Output: '{answer}'")

                    pbar.update(1)
                    curr_score = (score / total) * 100
                    curr_score = curr_score.detach().cpu().item()
                    pbar.set_description(
                        f"Test accuracy: {round(curr_score, 2)}"
                    )

                pbar.close()
                curr_score = 100 * score / total
                curr_score = curr_score.detach().cpu().item()
                print(f"Device {rank}: Score={round(curr_score, 2)}, Total={total}")

            dist.all_reduce(score, op=dist.ReduceOp.SUM)
            dist.all_reduce(total, op=dist.ReduceOp.SUM)
            all_predictions = None
            if dist.is_available() and dist.is_initialized():
                gathered_preds = [None for _ in range(dist.get_world_size())]
                gathered_indices = [None for _ in range(dist.get_world_size())]
                if rank == 0:
                    dist.gather_object(predictions, gathered_preds, dst=0)
                    dist.gather_object(prediction_indices, gathered_indices, dst=0)                
                else:
                    dist.gather_object(predictions, None, dst=0) # Pass None on other ranks
                    dist.gather_object(prediction_indices, None, dst=0)                

                # dist.gather_object(predictions, gathered_preds, dst=0)
                # dist.gather_object(prediction_indices, gathered_indices, dst=0)
                if rank == 0:
                    pred_map = {}
                    for preds, indices in zip(gathered_preds, gathered_indices):
                        for p, i in zip(preds or [], indices or []):
                            pred_map[int(i)] = p
                    max_idx = max(pred_map.keys()) if pred_map else -1
                    all_predictions = [
                        pred_map.get(i) for i in range(max_idx + 1)
                    ]
            else:
                pred_map = {int(i): p for i, p in zip(prediction_indices, predictions)}
                max_idx = max(pred_map.keys()) if pred_map else -1
                all_predictions = [pred_map.get(i) for i in range(max_idx + 1)]

            final_score = 100 * score / total
            final_score = final_score.detach().cpu().item()
            if rank == 0:
                print(f"Accuracy on validation set: {round(final_score, 2)}")
            sys.stdout.flush()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            processed_result = {
                "dataset": eval_params["dataset"],
                "score": final_score,
                "outputs": all_predictions,
            }
            raw_result = {
                "total_score": score.detach().cpu().item(),
                "total": total.detach().cpu().item(),
            }
            if pipeline_params["only_eval"]:
                return processed_result, raw_result

            if (round(final_score, 2) > best_acc and not pipeline_params["only_eval"]):
                save_path = os.path.join(save_dir, f"checkpoint_{epoch+1}")
                save_base_causallm(model_engine, save_path)
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()

    return processed_result, raw_result
