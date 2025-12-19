# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import gc
import torch
import random
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from collections import namedtuple
from transformers.models.gpt2 import GPT2LMHeadModel

Outputs = namedtuple("Outputs", ["loss"])


class CE(nn.Module):

    def __init__(
        self,
        base_causallm,
        tokenizer
    ):

        super(CE, self).__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id
        # tested with GPT2 and Llama3
        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            self.embedding = self.base_causallm.get_input_embeddings()

    def forward(
            self, 
            input_ids, 
            attention_mask, 
            labels, 
            position_ids, 
            prompt_len,
            config, 
            **kwargs
        ):
        logits = []
        inputs_embeds = self.embedding(input_ids)
        kv_cache = None

        ########### Inference process
        past_key_values = None
        if kv_cache is not None:
            past_key_values = kv_cache
        
        output_hidden_states = False
        # Predict each token, create mask each time
        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_hidden_states=output_hidden_states,
            prompt_lens=prompt_len,
            config=config
        )
        logits.append(outputs.logits)
        loss = outputs.loss

        return Outputs(loss=loss)

    def train(self):
        self.base_causallm.config.use_cache = False
        self.base_causallm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.base_causallm.train()

    def eval(self):
        self.base_causallm.config.use_cache = True
        self.base_causallm.eval()

    def generate(
        self,
        input_ids,
        attention_mask,
        prompt_len,
        config,
        max_new_tokens=16,
        output_embedding=False,
        synced_gpus=False,
        **kwargs
    ):

        self.gen_forward_cnt = 0
        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"
        tokens = input_ids[0].detach().tolist()
        position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device).reshape(1, -1)
        inputs_embeds = self.embedding(input_ids)
        kv_cache = None
        compute_range = (0, input_ids.shape[1])

        for i in range(max_new_tokens):
            self.gen_forward_cnt += 1
            ########### Inference process
            past_key_values = None
            if kv_cache is not None:
                past_key_values = kv_cache
                kv_len = kv_cache[0][0].size(2)
            else:
                kv_len = 0

            output_hidden_states = False
            outputs = self.base_causallm(
                inputs_embeds=inputs_embeds[
                    :, compute_range[0] : compute_range[1], :
                ],
                attention_mask=attention_mask,
                position_ids=position_ids[
                    :, compute_range[0] : compute_range[1]
                ],
                past_key_values=past_key_values,
                output_hidden_states=output_hidden_states,
                synced_gpus=synced_gpus
            )
            kv_cache = outputs.past_key_values

            # get the first token using the current hidden state
            next_token = torch.argmax(outputs.logits[0, -1]).item()
            tokens.append(next_token)
            new_token_embed = self.embedding(
                torch.tensor(next_token, device=input_ids.device)
            ).view(1, 1, -1)
            inputs_embeds = torch.cat((inputs_embeds, new_token_embed), dim=1)
            # Update attention_mask
            new_attention = torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([attention_mask, new_attention], dim=1)
            # Update position_ids
            new_pos = position_ids[:, -1:]  + 1 
            position_ids = torch.cat([position_ids, new_pos], dim=1) 
            compute_range = (compute_range[1], compute_range[1] + 1)
            if next_token == self.eos_token_id:
                break


        if synced_gpus:
            # in FSDP, the number of forward pass need to be the same across devices
            while (
                self.gen_forward_cnt < max_new_tokens
            ):  # leave some room for latent tokens
                self.gen_forward_cnt += 1
                _ = self.base_causallm(inputs_embeds=inputs_embeds)

        answer = tokens[prompt_len:]
        return self.tokenizer.decode(answer)
