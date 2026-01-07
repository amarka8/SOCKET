# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections.abc import Callable
from typing import Optional, Union

import torch
# torch.autograd.set_detect_anomaly(True)
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as pt_checkpoint

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import create_causal_mask
from transformers.modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from transformers.models.llama.configuration_llama import LlamaConfig


from deepspeed.runtime.activation_checkpointing.checkpointing import checkpoint as ds_checkpoint


logger = logging.get_logger(__name__)

def _torch_version_gte(target: str) -> bool:
    try:
        from packaging import version as pkg_version
        return pkg_version.parse(torch.__version__) >= pkg_version.parse(target)
    except Exception:
        ver = torch.__version__.split("+", 1)[0]
        parts = []
        for part in ver.split("."):
            num = ""
            for ch in part:
                if ch.isdigit():
                    num += ch
                else:
                    break
            parts.append(int(num) if num else 0)
        target_parts = [int(p) for p in target.split(".")]
        while len(parts) < len(target_parts):
            parts.append(0)
        return tuple(parts[: len(target_parts)]) >= tuple(target_parts)


def _is_torch_xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def _is_torch_npu_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


_is_torch_greater_or_equal_than_2_5 = _torch_version_gte("2.5")
_is_torch_greater_or_equal_than_2_8 = _torch_version_gte("2.8")


@use_kernel_forward_from_hub("RMSNorm")
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class LlamaRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: LlamaConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def sdpa_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    rw_allow_blocks: Optional[torch.Tensor] = None,
    rw_block_size: Optional[int] = None,
    rw_tq: Optional[int] = None,
    rw_tk: Optional[int] = None,
    rw_window_tokens: Optional[int] = None,
    **kwargs: Unpack[TransformersKwargs],
):
    if kwargs.get("output_attentions", False):
        logger.warning_once(
            "`sdpa` attention does not support `output_attentions=True`. "
            "Please set attention to `eager` if you want these features."
        )
    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)

    if _is_torch_npu_available():
        if attention_mask is not None and attention_mask.dtype != torch.bool:
            attention_mask = torch.logical_not(attention_mask.bool()).to(query.device)

    if attention_mask is not None:
        if attention_mask.dim() == 2:
            attention_mask = attention_mask[:, None, None, :]
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)
        elif attention_mask.dim() == 4 and attention_mask.size(1) == 0:
            attention_mask = attention_mask.new_zeros(
                (attention_mask.size(0), 1, attention_mask.size(2), attention_mask.size(3))
            )


    sdpa_kwargs = {}
    is_causal = False

    if query.dim() == 4 and rw_allow_blocks is not None:
        B, H, T_q, D = query.shape
        token_window_mask = None
        if rw_window_tokens is not None and rw_window_tokens > 0:
            q_idx = torch.arange(T_q, device=query.device).view(T_q, 1)
            k_idx = torch.arange(T_q if rw_tk is None else rw_tk, device=query.device).view(1, -1)
            token_window_mask = (k_idx <= q_idx) & (k_idx >= (q_idx - rw_window_tokens + 1))
            sink_mask = k_idx < rw_window_tokens
            token_window_mask = token_window_mask | sink_mask

        block_size = rw_block_size or 1
        allow_blocks = rw_allow_blocks
        num_blocks_q, num_blocks_k = allow_blocks.shape[1], allow_blocks.shape[2]
        allow = allow_blocks[:, :, None, :, None].expand(
            B, num_blocks_q, block_size, num_blocks_k, block_size
        )
        allow = allow.reshape(B, num_blocks_q * block_size, num_blocks_k * block_size)
        allow = allow[:, : (rw_tq or T_q), : (rw_tk or allow.size(-1))]
        if token_window_mask is not None:
            allow = allow | token_window_mask.unsqueeze(0)
        if not module.training:
            q_idx = torch.arange(allow.size(1), device=query.device).view(1, -1, 1)
            k_idx = torch.arange(allow.size(2), device=query.device).view(1, 1, -1)
            causal = k_idx <= q_idx
            allowed_causal = allow & causal
            density = allowed_causal.float().sum() / causal.float().sum()
            print(f"Layer {module.layer_idx}, [rw-mask] causal density={density.item():.2f}")


        mask = attention_mask
        neg_inf = torch.finfo(query.dtype).min
        if mask is None:
            mask = (~allow).unsqueeze(1).to(query.dtype) * neg_inf
        else:
            if mask.dim() == 2:
                mask = mask[:, None, None, :]
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            allow_mask = (~allow).unsqueeze(1).to(mask.dtype) * neg_inf
            mask = mask + allow_mask

        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=dropout,
            scale=scaling,
            is_causal=is_causal,
            **sdpa_kwargs,
        )
    else:
        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=dropout,
            scale=scaling,
            is_causal=is_causal,
            **sdpa_kwargs,
        )
    
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class LlamaAttention(nn.Module):
    """
    Multi-head attention with optional sparse masking controls.
    """

    def __init__(self, config: "LlamaConfig", layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

        # ---- Token attention estimator controls ----
        self.masker_kl_weight: float = getattr(config, "topk_masker_kl_weight", 1.0)

        hidden_gate = getattr(config, "topk_hidden", max(256, self.head_dim))
        self.masker_mode = "joint"
        self.random_walk_alpha = getattr(config, "random_walk_alpha", 0.001)
        self.random_walk_window = getattr(config, "random_walk_window", 0.1)
        self.random_walk_block_size = getattr(config, "random_walk_block_size", 64)
        self.random_walk_hadamard_dim = getattr(config, "random_walk_hadamard_dim", 128)
        print(f"Hadamard sketching dim is: {self.random_walk_hadamard_dim}")

    def _get_random_walk_states(self, past_key_values, runtime_states):
        if past_key_values is not None:
            states = getattr(past_key_values, "random_walk_states", None)
            if states is None:
                states = {}
                setattr(past_key_values, "random_walk_states", states)
            return states
        return runtime_states

    def _prepare_random_walk_state(self, prev, target_len, batch_size, dtype, device):
        if prev is None or prev.size(0) != batch_size:
            base = torch.eye(target_len, device=device, dtype=dtype).unsqueeze(0)
            if batch_size > 1:
                base = base.repeat(batch_size, 1, 1)
            prev = base
        else:
            prev = prev.to(device=device, dtype=dtype)
            prev_len = prev.size(-1)
            if prev_len < target_len:
                pad = target_len - prev_len
                base = torch.eye(target_len, device=device, dtype=dtype).unsqueeze(0)
                if batch_size > 1:
                    base = base.repeat(batch_size, 1, 1)
                base[:, :prev_len, :prev_len] = prev
                prev = base
            elif prev_len > target_len:
                prev = prev[:, :target_len, :target_len]
        return prev

    def _fwht(self, x: torch.Tensor) -> torch.Tensor:
        n = x.size(-1)
        y = x.reshape(-1, n)
        h = 1
        while h < n:
            y = y.view(-1, n // (2 * h), 2 * h)
            a = y[..., :h].clone()
            b = y[..., h : 2 * h]
            y[..., :h] = a + b
            y[..., h : 2 * h] = a - b
            h *= 2
        return y.view(*x.shape)

    def _hadamard_project(self, x: torch.Tensor, proj_dim: int) -> torch.Tensor:
        n = x.size(-1)
        n_pow2 = 1 << (n - 1).bit_length()
        if n_pow2 != n:
            x = F.pad(x, (0, n_pow2 - n), value=0.0)
        sign = torch.randint(0, 2, (1, x.size(1), 1, n_pow2), device=x.device, dtype=torch.int8)
        sign = sign.to(x.dtype) * 2 - 1
        x = x * sign
        x = self._fwht(x)
        proj_dim = min(proj_dim, n_pow2)
        x = x / ((n_pow2 / proj_dim) ** 0.5)
        idx = torch.randperm(n_pow2, device=x.device)[:proj_dim]
        return x.index_select(-1, idx)

    def _hadamard_attention_probs(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if key_states.size(1) != query_states.size(1):
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        proj_dtype = torch.bfloat16
        q = self._hadamard_project(query_states.to(proj_dtype), self.random_walk_hadamard_dim)
        k = self._hadamard_project(key_states.to(proj_dtype), self.random_walk_hadamard_dim)
        scale = q.size(-1) ** -0.5
        logits = torch.einsum("bhqd,bhkd->bhqk", q, k) * scale
        if attention_mask is not None:
            logits = logits + attention_mask
        # If softmax here, many outliers make most of the numbers become 0 -> clamp
        attn_probs = torch.softmax(logits.clamp(-10, 10), dim=-1)
        pt_tok = attn_probs.sum(dim=1)
        # import pdb
        # import torch.distributed as dist
        # dist.barrier()
        # if dist.get_rank() == 0:
        #     import pdb; pdb.set_trace()
        # dist.barrier()
        return pt_tok / (pt_tok.sum(dim=-1, keepdim=True) + 1e-12)

    def _block_pool(self, x: torch.Tensor, block_size: int) -> torch.Tensor:
        # x: (B, H, T, D) -> (B, H, T_blocks, D) via mean pooling
        if block_size <= 1:
            return x
        B, H, T, D = x.shape
        num_blocks = (T + block_size - 1) // block_size
        pad = num_blocks * block_size - T
        if pad:
            x = F.pad(x, (0, 0, 0, pad), value=0.0)
        x = x.view(B, H, num_blocks, block_size, D).mean(dim=3)
        return x

    def _hadamard_attention_probs_block(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        block_size: int,
    ) -> torch.Tensor:
        if key_states.size(1) != query_states.size(1):
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        q = self._block_pool(query_states, block_size)
        k = self._block_pool(key_states, block_size)
        proj_dtype = torch.bfloat16
        q = self._hadamard_project(q.to(proj_dtype), self.random_walk_hadamard_dim)
        k = self._hadamard_project(k.to(proj_dtype), self.random_walk_hadamard_dim)
        scale = q.size(-1) ** -0.5
        logits = torch.einsum("bhqd,bhkd->bhqk", q, k) * scale
        if attention_mask is not None:
            logits = logits + attention_mask
        attn_probs = torch.softmax(logits.clamp(-10, 10), dim=-1)
        pt_blk = attn_probs.sum(dim=1)
        return pt_blk / (pt_blk.sum(dim=-1, keepdim=True) + 1e-12)


    def _update_random_walk(self, attention_probs, past_key_values, runtime_states):
        per_head = attention_probs.dim() == 4
        if per_head:
            B, H, T_q, T_k = attention_probs.shape
            attention_probs = attention_probs.reshape(B * H, T_q, T_k)
        else:
            B, T_q, T_k = attention_probs.shape
        states = self._get_random_walk_states(past_key_values, runtime_states)
        # BUG: when T_q == 1 and layer0
        prev = states.get(self.layer_idx - 1, None) if states is not None else None
        if prev is None:
            state = attention_probs
        else:
            state = self._prepare_random_walk_state(
                prev=prev,
                target_len=T_k,
                batch_size=B,
                dtype=attention_probs.dtype,
                device=attention_probs.device,
            )
        
        for i in range(3):
            state = torch.matmul(state, attention_probs)
        walk = state
        if states is not None:
            states[self.layer_idx] = walk.detach()
        if not per_head:
            return walk
        return walk.view(B, H, T_q, T_k)

    def forward(
        self,
        hidden_states: torch.Tensor,                           # (B, T, hidden_size)
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional["Cache"] = None,
        cache_position: Optional[torch.LongTensor] = None,
        random_walk_states: Optional[dict] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)   # (B,H,T_q,D)
        key_states   = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)   # (B,H_kv,T_k,D)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if self.num_key_value_groups > 1:
            B, H_kv, T_k, D = key_states.shape
            H = query_states.size(1)
            # repeat = H // H_kv
        else:
            B, H, T_k, D = key_states.shape
            H = query_states.size(1)

        B, H, T_q, D = query_states.shape
        device = hidden_states.device

        estimator_log_probs = None
        attn_impl = "sdpa"
        if attn_impl == "sdpa":
            attention_interface = sdpa_attention_forward
        else:
            attention_interface = eager_attention_forward
        # Teacher token-level probs over keys: (B,T_q,T_k)
        teacher_attention_probs = None
        attn_out_masked = None
        attn_weights_masked = None

        if self.masker_mode == "inference_only" and self.random_walk_alpha is not None and T_q > 1 and self.layer_idx > 1:
            block_size = self.random_walk_block_size or 1
            teacher_attention_probs = self._hadamard_attention_probs_block(
                query_states, key_states, attention_mask=None, block_size=block_size
            )
            random_walk_probs = self._update_random_walk(
                teacher_attention_probs.detach(), past_key_values, random_walk_states
            )

            if block_size > 1:
                num_blocks_q = random_walk_probs.shape[1]
                num_blocks_k = random_walk_probs.shape[2]
                block_probs = random_walk_probs
                q_idx = torch.arange(num_blocks_q, device=device).view(num_blocks_q, 1)
                k_idx = torch.arange(num_blocks_k, device=device).view(1, num_blocks_k)
                causal_blocks = (k_idx <= q_idx).view(1, num_blocks_q, num_blocks_k)
                masked_probs = block_probs.float().masked_fill(~causal_blocks, float("nan"))
                th = torch.nanquantile(masked_probs, 0.8, dim=-1, keepdim=True)  # (B, T_qb, 1)
                if isinstance(th, torch.Tensor):
                    tail = min(block_size, th.shape[-2])
                    if tail > 0:
                        th = th.clone()
                        th[:, -5:, :] = 0
                allow_blocks = block_probs > th  # (B, T_qb, T_kb)

                # This only works for prefill
                # SLIDING WINDOW (token-level)
                block_window = self.random_walk_window
                if block_window is not None and 0 < block_window < 1:
                    block_window = max(1, int(block_window * T_k))
            else:
                allow_blocks = random_walk_probs >= 0
                block_window = None

            attn_mask_for_second = attention_mask
            
            attn_out_masked, attn_weights_masked = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attn_mask_for_second,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                rw_allow_blocks=allow_blocks,
                rw_block_size=block_size,
                rw_tq=T_q,
                rw_tk=T_k,
                rw_window_tokens=block_window,
                **kwargs,
            )
        else:
            attn_out_masked, attn_weights_masked = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs,
            )

        extra = {
            "estimator_log_probs": estimator_log_probs,
            "teacher_attention_probs": teacher_attention_probs,
        }
        attn_out = attn_out_masked.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_out)

        return attn_output, attn_weights_masked, extra


class LlamaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.masker_mode = 'joint'

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        random_walk_states: Optional[dict] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _, extra = self.self_attn(
            hidden_states=hidden_states.detach().requires_grad_(True) if self.masker_mode == "only_selector" else hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            random_walk_states=random_walk_states,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, extra


@auto_docstring
class LlamaPreTrainedModel(PreTrainedModel):
    config: LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": LlamaDecoderLayer,
        "attentions": LlamaAttention,
    }



@auto_docstring
class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.masker_mode = "joint"

        # Initialize weights and apply final processing
        self.post_init()

    def _causal_mask(self, attention_mask, input_embeds, past_key_values=None):
        """
        Shapes:
        input_embeds: (B, T_q, D)
        attention_mask (optional): (B, T_k)  -- if provided, T_k = attention_mask.shape[1]
        returns: (B, 1, T_q, T_k)
        """

        device = input_embeds.device
        dtype  = input_embeds.dtype
        neg_inf = torch.finfo(dtype).min
        B, T_q, _ = input_embeds.shape
        if attention_mask is not None:
            T_k = attention_mask.shape[1]
            past_len = max(T_k - T_q, 0)
        else:
            past_len = (past_key_values[0][0].size(2) if past_key_values is not None else 0)
            T_k = past_len + T_q

        base = torch.zeros((1, 1, T_q, T_k), dtype=dtype, device=device)

        # Causal upper-tri blocking ONLY over the current window [past_len : past_len + T_q]
        curr_k_len = max(min(T_q, T_k - past_len), 0)
        if curr_k_len > 0:
            # boolean upper-tri (True above diag -> to be filled with -inf)
            tri_bool = torch.triu(torch.ones((T_q, curr_k_len), dtype=torch.bool, device=device), diagonal=1)
            # slice the current window and masked_fill
            curr_view = base[:, :, :, past_len:past_len + curr_k_len]
            base[:, :, :, past_len:past_len + curr_k_len] = curr_view.masked_fill(tri_bool, neg_inf)

        # Expand to batch then clone to allow masked_fill safely (avoid in-place on a view from expand)
        base = base.expand(B, 1, T_q, T_k).clone()

        # Padding: mask out keys where attention_mask == 0 (no arithmetic with -inf)
        if attention_mask is not None:
            key_pad = (attention_mask == 0).view(B, 1, 1, T_k)
            base = base.masked_fill(key_pad, neg_inf)

        return base.contiguous()


    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        causal_mask = self._causal_mask(
            attention_mask=attention_mask,
            input_embeds=inputs_embeds,
            past_key_values=past_key_values,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        loss_dtype  = hidden_states.dtype
        loss_device = hidden_states.device
        total_selector_loss = torch.tensor(0.0, device=loss_device, dtype=loss_dtype)
        num_contrib = 0

        runtime_random_walk_states = {} if past_key_values is None else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states, extra = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                random_walk_states=runtime_random_walk_states,
                **kwargs,
            )
            log_p = extra.get("estimator_log_probs", None)
            pt_attn = extra.get("teacher_attention_probs", None)

            if log_p is not None and pt_attn is not None:
                log_pt = (pt_attn.clamp_min(1e-12)).log()
                layer_loss = (pt_attn * (log_pt - log_p)).sum(dim=-1).mean()

                # dtype/device adapt without disabling grad
                if layer_loss.dtype != loss_dtype:
                    layer_loss = layer_loss.to(loss_dtype)
                if layer_loss.device != loss_device:
                    layer_loss = layer_loss.to(loss_device)

                total_selector_loss = total_selector_loss + layer_loss
                num_contrib += 1

        if num_contrib > 0:
            total_selector_loss = total_selector_loss / num_contrib
        else:
            total_selector_loss = torch.zeros(1, device=loss_device, dtype=loss_dtype).squeeze(0)

        hidden_states = self.norm(hidden_states)
        return total_selector_loss, BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring
class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def set_masker_mode(
        self,
        mode: str,
        unfreeze_ln_and_embed: bool = False,
    ):
        assert mode in {"joint", "only_selector", "inference_only"}
        self.config.topk_masker_mode = mode
        self._apply_mode_param_grads_model(mode, unfreeze_ln_and_embed=unfreeze_ln_and_embed)

    def _apply_mode_param_grads_model(self, mode: str, *, unfreeze_ln_and_embed: bool):
        for p in self.parameters():
            p.requires_grad = False

        for mod in self.modules():
            if hasattr(mod, "masker_mode"):
                mod.masker_mode = mode

        if mode == "joint":
            for p in self.parameters():
                p.requires_grad = True

        elif mode == "only_selector":
            if unfreeze_ln_and_embed:
                for mod in self.modules():
                    if isinstance(mod, (nn.LayerNorm, nn.Embedding)):
                        for p in mod.parameters():
                            p.requires_grad = True

        elif mode == "inference_only":
            pass

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        prompt_lens: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        if use_cache is None:
            use_cache = self.config.use_cache

        loss, outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        if labels is not None:
            loss += self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
