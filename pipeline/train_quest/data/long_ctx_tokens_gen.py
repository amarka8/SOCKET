# long_context_labels_lsd2_fixed.py
# pip install torch transformers

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from pipeline.train_quest.mask_gen.stream_mask import gen_stream_mask


@torch.no_grad()
def log_probs_under_full_context(model, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Per-position log P(x_i | x_<i) for i=1..T-1.
    Returns tensor of shape (T,) where index 0 is 0.0 (unused).
    """
    out = model(input_ids=input_ids)                     # logits: (1, T, V)
    logp = F.log_softmax(out.logits, dim=-1)            # (1, T, V)
    ids  = input_ids[0]                                  # (T,)
    lcl  = torch.zeros(ids.size(0), dtype=logp.dtype, device=logp.device)
    # for each i>=1, take the prob of the true token at i from logits at i-1
    lcl[1:] = logp[0, :-1].gather(-1, ids[1:].unsqueeze(-1)).squeeze(-1)
    return lcl


@torch.no_grad()
def log_probs_under_short_context(
    model,
    ids: torch.Tensor,
    attention_mask: torch.Tensor,
    config: dict,
    dtype=torch.float32,
) -> torch.Tensor:
    """
    Compute per-position log P(x_i | sink + recent context) 

    Args:
        model: causal LM (AutoModelForCausalLM)
        ids: (1, T) input_ids tensor
        attention_mask: (1, T)  1=real, 0=pad
        config: dict with keys {'n_init', 'n_local'} used by gen_stream_mask
        dtype: match model input dtype (e.g., torch.float16)
    Returns:
        short_logprobs: (T,) tensor of log P(x_i | sink + recent context) 
    """
    B, T = ids.shape
    device = ids.device
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "eager"
    elif hasattr(model.config, "attn_implementation"):
        model.config.attn_implementation = "eager"

    # get prompt_len = number of valid tokens per sample (for get_init_local)
    prompt_len = attention_mask.sum(dim=1).to(dtype=torch.int)

    # build additive mask (B, 1, T, T)
    attn_mask = gen_stream_mask(config, attention_mask, prompt_len, dtype).to(device)

    # run model forward with the custom bias
    out = model(input_ids=ids, attention_mask=attn_mask)

    logp = F.log_softmax(out.logits, dim=-1)     # (B, T, V)
    ids = ids[0]                               # (T,)
    short = torch.zeros(T, dtype=torch.float32, device=device)
    short[1:] = logp[0, :-1].gather(-1, ids[1:].unsqueeze(-1)).squeeze(-1)
    return short


import torch

@torch.no_grad()
def is_longctx_token(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    config: dict,
    context_len,
    dtype=torch.float32,
    alpha: float = 1.0,
    beta: float = -2.0,
) -> dict:
    """
    Compute per-position Long-Short Difference (LSD) and mark long-context-sensitive tokens.

    A token x_i is considered long-context-sensitive if:
        LSDθ(x_i) > α  and  LCLθ(x_i) > β

    Args:
        model: causal LM supporting attn_bias
        tokenizer: tokenizer (not used for conversion unless needed)
        input_ids: (1, T)
        attention_mask: (1, T)
        config: dict with keys {'n_init', 'n_local'} used by gen_stream_mask
        dtype: dtype for attention bias (e.g., torch.float16)
        alpha: LSD threshold (default = 2.0)
        beta:  LCL threshold (default = -2.0)

    Returns:
        dict with:
            - input_ids       : Tensor(1, T)
            - lcl             : Tensor(T,) log P(x_i | full context)
            - short           : Tensor(T,) log P(x_i | sink + recent)
            - lsd             : Tensor(T,) difference
            - is_long_context : Bool Tensor(T,) per-token label
    """
    lcl = log_probs_under_full_context(model, input_ids)  # (T,)

    short = log_probs_under_short_context(
        model=model,
        ids=input_ids,
        attention_mask=attention_mask,
        config=config,
        dtype=dtype,
    )  # (T,)

    lsd = (lcl - short).to(torch.float32)
    is_long_context = (lsd > alpha) & (lcl > beta)

    return {
        "input_ids": input_ids,
        "lcl": lcl,
        "short": short,
        "lsd": lsd,
        "is_long_context": is_long_context,
    }
