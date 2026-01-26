import logging
logger = logging.getLogger("main")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, LlamaForCausalLM


def initialize_model_tokenizer(pipeline_params):
    config = AutoConfig.from_pretrained(pipeline_params['model_name'])
    if 'rope_theta_factor' in pipeline_params and hasattr(config, 'rope_theta'):
        config.rope_theta *= pipeline_params['rope_theta_factor']

    if pipeline_params['use_flash_attn']:
        attn_implementation = 'flash_attention_2'
    else:
        attn_implementation = 'sdpa'

    if 'mamba' in pipeline_params['model_name'].lower():
        from transformers import MambaConfig, MambaForCausalLM
        model = MambaForCausalLM.from_pretrained(pipeline_params['model_name']).to("cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(pipeline_params['model_name'], config=config, attn_implementation=attn_implementation, torch_dtype=torch.bfloat16).to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(pipeline_params['model_name'], padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    logger.info(f'Model {model} and Tokenizer {tokenizer} initialized.')
    return model, tokenizer
