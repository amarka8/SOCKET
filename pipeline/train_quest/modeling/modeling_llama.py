import os
import sys
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
import itertools
import math
import triton
import triton.language as tl

try:
    from .soft_hash_collision_loader import load_soft_hash_collision
except ImportError:
    # Fallback for running the file outside the package context.
    from soft_hash_collision_loader import load_soft_hash_collision

# Ensure repo root is on sys.path so local modules (e.g., kernels) resolve from any cwd
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from kernels.cra_triton_kernels import (
    attention_mask_to_allowed_prob, sparse_attention_fwd
)

logger = logging.get_logger(__name__)

_SOFT_HASH_EXT = None

def _get_soft_hash_ext():
    global _SOFT_HASH_EXT
    if _SOFT_HASH_EXT is None:
        _SOFT_HASH_EXT = load_soft_hash_collision(3)
    return _SOFT_HASH_EXT

@torch.no_grad()
def build_sparse_list_decode(
    q_probs: torch.Tensor,         # [B,H,L,R] fp16/bf16/fp32
    k_hard_bhlt: torch.Tensor,      # [B,H,L,T] int16/int32
    v_norm_bht: torch.Tensor,       # [B,H,T] fp16/bf16/fp32
    allowed_bht: torch.Tensor,      # [B,H,T] bool
    sink: int,
    window: int,
    M: int,
    KC: int = 8,
    BLOCK_N: int = 512,
    num_warps: int = 8,
    num_stages: int = 2,
):
    assert q_probs.is_cuda and k_hard_bhlt.is_cuda and v_norm_bht.is_cuda and allowed_bht.is_cuda
    assert allowed_bht.dtype == torch.bool

    B, H, L, R = q_probs.shape
    _, _, L2, T = k_hard_bhlt.shape
    assert L2 == L

    device = q_probs.device
    M_eff = min(M, T)
    if M_eff > 0:
        ext = _get_soft_hash_ext()
        q_probs_f32 = q_probs.float().unsqueeze(2).contiguous()  # [B,H,1,L,R]
        key_buckets = k_hard_bhlt
        if key_buckets.dtype != torch.int16:
            key_buckets = key_buckets.to(torch.int16)
        key_buckets = key_buckets.contiguous()
        allowed_ext = allowed_bht.unsqueeze(2).contiguous()       # [B,H,1,T]
        v_hist = v_norm_bht.float().unsqueeze(2).contiguous()     # [B,H,1,T]

        scores = ext.soft_hash_collision(
            q_probs_f32,
            key_buckets,
            allowed_ext,
            v_hist,
        ).squeeze(2)  # [B,H,T]

        top = torch.topk(scores, k=M_eff, dim=-1, largest=True)
        heavy_idx = top.indices.to(torch.int32)
    else:
        heavy_idx = torch.empty((B, H, 0), device=device, dtype=torch.int32)

    sink = max(0, min(sink, T))
    window = max(0, min(window, T))

    parts = []
    if sink > 0:
        parts.append(torch.arange(sink, device=device, dtype=torch.int32))
    if window > 0:
        win_start = max(T - window, sink)
        if win_start < T:
            parts.append(torch.arange(win_start, T, device=device, dtype=torch.int32))

    if len(parts) == 0:
        base = torch.tensor([T - 1], device=device, dtype=torch.int32)
    else:
        base = torch.cat(parts, dim=0)

    base = base.view(1, 1, -1).expand(B, H, -1)
    base_ok = torch.gather(allowed_bht, dim=-1, index=base.to(torch.long))
    base = base.masked_fill(~base_ok, -1)

    sparse_list = torch.cat([base, heavy_idx], dim=-1).contiguous()
    sparse_len = torch.full((B, H), sparse_list.shape[-1], device=device, dtype=torch.int32)
    return sparse_list, sparse_len

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
        self.masker_mode = "joint"

        self.bucket_K = getattr(config, "bucket_K", 7)           # P
        self.bucket_L = getattr(config, "bucket_L", 20)           # L
        self.bucket_top_t = getattr(config, "bucket_top_t", 8)
        
        self._planes_cache = {}
        self._protos_cache = {}

        # ---- RNG controls (for deterministic SRP plane init) ----
        # If None: planes are initialized non-deterministically (PyTorch default RNG).
        self._seed = 123456789
        self._rng_cache: dict[torch.device, torch.Generator] = {}

        # ---- Debug controls ----
        self.debug_mask_max_items: int = int(getattr(config, "debug_mask_max_items", 64))


    def _rng(self, device: torch.device) -> Optional[torch.Generator]:
        if self._seed is None:
            return None
        g = self._rng_cache.get(device)
        if g is None:
            g = torch.Generator(device=device)
            # offset so this doesn't collide with other seeds in the model
            g.manual_seed(int(self._seed) + 7777)
            self._rng_cache[device] = g
        return g

    def _debug_log_masks(self, masks: dict[str, torch.Tensor]) -> None:
        """
        Log mask shapes/densities and sample allowed indices for batch 0 / head 0.
        """
        try:
            parts: list[str] = []
            limit = max(1, self.debug_mask_max_items)
            for name, mask in masks.items():
                if mask is None:
                    continue
                bool_mask = mask.to(torch.bool)
                shape = tuple(bool_mask.shape)
                total = bool_mask.sum().item()
                density = float(total) / float(bool_mask.numel())

                head_vec = bool_mask[0]
                if head_vec.dim() > 1:
                    head_vec = head_vec[0]
                if head_vec.dim() > 1:
                    head_vec = head_vec[0]
                head_vec = head_vec.reshape(-1).detach().to("cpu")

                allowed_idx = torch.nonzero(head_vec, as_tuple=False).view(-1)
                sample = allowed_idx[:limit].tolist()

                parts.append(
                    f"{name}: shape={shape}, density={density:.6f}, head0_allowed={allowed_idx.numel()}, head0_sample={sample}"
                )
            if parts:
                print(f"[bucket-debug] layer={self.layer_idx} " + " | ".join(parts), flush=True)
        except Exception as e:
            print(f"[bucket-debug] mask logging failed in layer {self.layer_idx}: {e}", flush=True)


    def get_bucket_states(self, past_key_values, runtime_states):
        """
        Store per-layer state on the HF cache object when available, otherwise use runtime_states.
        """
        if past_key_values is not None:
            states = getattr(past_key_values, "bucket_states", None)
            if states is None:
                states = {}
                setattr(past_key_values, "bucket_states", states)
            return states
        return runtime_states

    def init_bucket_state(self, states, B, H, L, R, cap, device):
        """
        Allocate slots/counts lazily.
        """
        st = states.get(self.layer_idx, None)
        if st is None:
            st = {}
            states[self.layer_idx] = st

        slots = st.get("slots", None)
        counts = st.get("counts", None)
        v_norm = st.get("v_norm", None)

        # slots: [B,H,L,R,cap], initialize -1
        if slots is None or slots.shape[:4] != (B, H, L, R) or slots.shape[-1] != cap or slots.device != device:
            slots = torch.full((B, H, L, R, cap), -1, device=device, dtype=torch.int32)
            counts = torch.zeros((B, H, L, R), device=device, dtype=torch.int32)
            v_norm = None  # will be rebuilt/extended
            st["slots"] = slots
            st["counts"] = counts
            st["v_norm"] = v_norm
        else:
            # ensure counts exists
            if counts is None:
                counts = torch.zeros((B, H, L, R), device=device, dtype=torch.int32)
                st["counts"] = counts

        return st

    def get_hyper_planes(
        self,
        cache,
        D: int,
        L: int,
        P: int,
        device: torch.device,
        dtype: torch.dtype,
        rng: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Independent SRP planes per table:
            planes: [L, P, D]

        Memoized by (D, device, dtype, L, P).
        """
        key = (D, device, dtype, L, P)
        planes = cache.get(key)
        if planes is None:
            base = torch.randn((L, P, D), device=device, dtype=torch.float32, generator=rng)
            planes = base.to(dtype)
            cache[key] = planes
        return planes


    def get_protos_T(
        self,
        cache,
        P: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Hypercube corners: protos_T in {-1,+1}^P, shape [P, R], where R = 2^P.
        Memoized by (P, device, dtype).
        """
        key = (P, device, dtype)
        protos_T = cache.get(key)
        if protos_T is None:
            corners = torch.tensor(
                list(itertools.product([-1.0, +1.0], repeat=P)),
                device=device,
                dtype=torch.float32,
            )  # [R, P]
            protos_T = corners.t().to(dtype)  # [P, R]
            cache[key] = protos_T
        return protos_T


    def pack_bits(self, bits: torch.Tensor) -> torch.Tensor:
        """
        Pack last-dim bits into integer codes (big-endian).
        bits: [..., P] bool
        returns: [...] int16
        """
        P = bits.shape[-1]
        weights = (1 << torch.arange(P - 1, -1, -1, device=bits.device, dtype=torch.int16))  # [P]
        view_shape = (*([1] * (bits.ndim - 1)), P)
        return torch.sum(bits.to(torch.int16) * weights.view(*view_shape), dim=-1)


    def hard_hash(self, tensor: torch.Tensor, planes: torch.Tensor) -> torch.Tensor:
        """
        tensor: [B, H, N, D]
        planes: [L, P, D]
        returns bucket codes per table: [B, H, L, N]
        """
        # proj: [B, H, N, L, P]
        proj = torch.einsum("bhnd,lkd->bhnlk", tensor, planes)
        bits = proj >= 0  # [B, H, N, L, P] bool
        codes = self.pack_bits(bits)  # [B, H, N, L] int16
        return codes.permute(0, 1, 3, 2).contiguous()  # [B, H, L, N]


    def soft_hash(
        self,
        queries: torch.Tensor,
        planes: torch.Tensor,
        protos_T: torch.Tensor,
    ) -> torch.Tensor:
        """
        queries:   [B, H, Q, D]
        planes:    [L, P, D]
        protos_T:  [P, R]
        returns soft bucket probabilities: [B, H, Q, L, R]
        """
        # q_proj: [B, H, Q, L, P]
        q_proj = torch.einsum("bhqd,lkd->bhqlk", queries, planes)

        temp = math.sqrt(queries.size(-1))
        qh = torch.tanh(q_proj) / max(temp, 1e-6)

        # logits: [B, H, Q, L, R]
        logits = torch.einsum("bhqlk,kr->bhqlr", qh, protos_T)
        return F.softmax(logits, dim=-1)


    # -----------------------------------------------------------------------------
    # Your forward with Triton-integrated PREFILL (no Python for pos in range(T_k))
    # -----------------------------------------------------------------------------
    def _effective_size(self, size, N: int) -> int:
        # Match common masker behavior: float => ratio, int => absolute
        if isinstance(size, float):
            return int(size * N)
        return int(size)


    def _build_prev_allowed_like_maskers(
        self,
        B: int,
        H: int,
        T_q: int,
        T_k: int,
        allowed_ext: torch.Tensor,   # [B,H,T_q,T_k] or [B,H,1,T_k]
        sink_size_cfg,
        window_size_cfg,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Build prev_allowed = SinkMasker + LocalMasker, matching the masker implementation.

        Conventions:
        - In your Mask class dense mask uses 1.0 = MASKED (blocked), 0.0 = allowed.
        - Here we return bool allowed=True (opposite), consistent with your LlamaAttention code.
        """
        # effective sizes (float => ratio)
        sink_size = self._effective_size(sink_size_cfg, T_k)
        window_size = self._effective_size(window_size_cfg, T_k)
        sink_size = max(0, min(int(sink_size), T_k))
        window_size = max(0, min(int(window_size), T_k))

        # Start with all-False allowed mask
        prev_allowed = torch.zeros((B, H, T_q, T_k), device=device, dtype=torch.bool)

        # ---- SinkMasker: allow first sink_size keys ----
        if sink_size > 0:
            prev_allowed[..., :sink_size] = True

        if window_size > 0:
            q_pos = torch.arange(T_q, device=device).view(T_q, 1)     # [T_q,1]
            k_pos = torch.arange(T_k, device=device).view(1, T_k)     # [1,T_k]
            diagonal_offset = k_pos - q_pos                           # [T_q,T_k]

            offset1 = T_k - T_q - window_size + 1
            offset2 = T_k - T_q + 1

            # Masked band (blocked by local pattern)
            band_masked = (diagonal_offset >= offset1) & (diagonal_offset < offset2)

            local_allowed = band_masked
            prev_allowed |= local_allowed.view(1, 1, T_q, T_k)

        # ---- Gate with external allowed mask (pad+causal) ----
        # allowed_ext is already [B,H,1,T_k] in your code; handle either shape safely.
        if allowed_ext.dim() == 4 and allowed_ext.shape[2] == T_q:
            prev_allowed &= allowed_ext
        else:
            prev_allowed &= allowed_ext.expand(B, H, T_q, T_k)

        return prev_allowed


    def _get_states_container(
        self,
        past_key_values,
        runtime_states: dict | None,
    ) -> dict:
        """
        Return a mutable dict that persists across decode steps.

        Priority:
        1) If HF past_key_values is provided, attach state to it
        so it survives across generation steps.
        2) Otherwise, fall back to runtime_states (caller-managed).
        """

        # ---- Preferred: attach to HF cache object ----
        if past_key_values is not None:
            # HuggingFace cache objects are mutable; we can safely hang attrs on them
            states = getattr(past_key_values, "bucket_states", None)
            if states is None:
                states = {}
                setattr(past_key_values, "bucket_states", states)
            return states

        # ---- Fallback: external runtime_states dict ----
        if runtime_states is None:
            runtime_states = {}

        return runtime_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values=None,
        cache_position=None,
        cra_states: dict | None = None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        B, T_q, _ = hidden_states.shape
        hidden_shape = (*input_shape, -1, self.head_dim)

        # ---- projections ----
        q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # [B,H,T_q,D]
        k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # [B,H_kv,T_q,D]
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # [B,H_kv,T_q,D]

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        past_len = 0
        if past_key_values is not None:
            past_len = past_key_values.get_seq_length()
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

        k_full = repeat_kv(k, self.num_key_value_groups)  # [B,H,T_k,D]
        v_full = repeat_kv(v, self.num_key_value_groups)  # [B,H,T_k,D]
        B, H, T_k, D = k_full.shape
        device = q.device

        # Only sparse path for true single-token decode (common HF generation)
        is_decode_1tok = (T_q == 1) and (past_len > 0)

        # ============================================================
        # Prefill / non-1tok decode: do dense, but init sparse state
        # ============================================================
        if not is_decode_1tok:
            attn_out = F.scaled_dot_product_attention(
                q, k_full, v_full,
                attn_mask=attention_mask,
                dropout_p=0.0 if not self.training else self.attention_dropout,
                scale=self.scaling,
                is_causal=False,
            )
            attn_out = attn_out.transpose(1, 2).contiguous().reshape(*input_shape, -1)
            out = self.o_proj(attn_out)

            # ---- init/refresh state for later 1-tok decode ----
            states = self._get_states_container(past_key_values, cra_states)

            P = int(getattr(self, "bucket_K", 7))
            L = int(getattr(self, "bucket_L", 20))
            R = 1 << P

            planes = self.get_hyper_planes(
                cache=self._planes_cache,
                D=D, L=L, P=P,
                device=device,
                dtype=k_full.dtype,
                rng=self._rng(device),
            )

            # hard hash all keys once
            key_buckets = self.hard_hash(k_full, planes).to(torch.int16)  # [B,H,L,T_k]

            # cache v norms (float16 is fine)
            max_seq = int(getattr(self.config, "max_position_embeddings", max(2048, T_k)))
            v_mag = torch.empty((B, H, max_seq), device=device, dtype=torch.float16)
            v_mag[..., :T_k] = torch.linalg.vector_norm(v_full.float(), ord=2, dim=-1).to(v_mag.dtype)

            states[self.layer_idx] = {
                "P": P, "L": L, "R": R,
                "planes": planes,                 # cache planes for deterministic reuse
                "key_buckets": key_buckets,       # [B,H,L,T_cached]
                "v_mag": v_mag,                   # [B,H,max_seq]
                "max_seq": max_seq,
                "device": device,
                "B": B, "H": H,
            }
            return out, None, {"bucket_sparse": False}

        # ============================================================
        # 1-token decode: simple expected-collision topM + sink/local
        # ============================================================
        states = self._get_states_container(past_key_values, cra_states)
        st = states.get(self.layer_idx, None)

        P = int(st["P"])
        L = int(st["L"])
        R = int(st["R"])
        planes = st["planes"]
        key_buckets = st["key_buckets"]   # [B,H,L,T_cached]
        v_mag = st["v_mag"]
        max_seq = int(st["max_seq"])

        # ---- keep cached key_buckets in sync by appending ONLY the new token ----
        # cached length might already match (depending on how HF calls forward)
        T_cached = key_buckets.shape[-1]
        if T_cached < T_k:
            # hash only the new keys at the end (assumes monotonic decode)
            k_new = k_full[:, :, T_cached:T_k, :]  # [B,H,delta,D] (usually delta=1)
            bkt_new = self.hard_hash(k_new, planes).to(torch.int16)  # [B,H,L,delta]
            key_buckets = torch.cat([key_buckets, bkt_new], dim=-1)
            st["key_buckets"] = key_buckets

        # ---- update v norm cache for the new token(s) ----
        if T_k <= max_seq:
            v_mag[..., :T_k] = torch.linalg.vector_norm(v_full.float(), ord=2, dim=-1).to(v_mag.dtype)

        # ---------------------------
        # external allowed mask -> allowed_bht
        # ---------------------------
        if attention_mask is not None:
            allowed_prob = attention_mask_to_allowed_prob(attention_mask, T_k)
            allowed_bht = (allowed_prob > 0).expand(B, H, 1, T_k).squeeze(2).contiguous()
        else:
            allowed_bht = torch.ones((B, H, T_k), device=device, dtype=torch.bool)

        # ---------------------------
        # budget + sparse list
        # ---------------------------
        sink = int(getattr(self.config, "sink_size", 20))
        window = int(getattr(self.config, "window_size", 20))
        M_cfg = getattr(self.config, "heavy_const", getattr(self.config, "heavy_size", 0.2))
        M = self._effective_size(M_cfg, T_k)
        M = max(0, min(int(M), T_k))

        protosT = self.get_protos_T(
            cache=self._protos_cache,
            P=P,
            device=device,
            dtype=k_full.dtype,
        )
        q_probs = self.soft_hash(q[:, :, 0:1, :], planes, protosT).squeeze(2)  # [B,H,L,R]

        if T_k <= max_seq:
            v_norm_bht = v_mag[..., :T_k].to(torch.float16)
        else:
            v_norm_bht = torch.linalg.vector_norm(v_full.float(), ord=2, dim=-1).to(torch.float16)

        k_hard_bhlt = key_buckets[..., :T_k].contiguous()

        sparse_list, sparse_len = build_sparse_list_decode(
            q_probs,
            k_hard_bhlt,
            v_norm_bht,
            allowed_bht,
            sink=sink,
            window=window,
            M=M,
            KC=8,
            BLOCK_N=512,
            num_warps=8,
            num_stages=2,
        )
        if sparse_list.dtype != torch.int32:
            sparse_list = sparse_list.to(torch.int32)
        if sparse_len.dtype != torch.int32:
            sparse_len = sparse_len.to(torch.int32)

        q_bhd = q[:, :, 0, :].contiguous()
        out_bhd = sparse_attention_fwd(
            q_bhd,
            k,
            v,
            sparse_list,
            sparse_len,
            block_seq=256,
        )
        out = out_bhd.unsqueeze(2).transpose(1, 2).contiguous().reshape(B, 1, -1)
        out = self.o_proj(out)

        return out, None, {
            "bucket_sparse": True,
            "mode": "cra_sparse",
            "T_k": int(T_k),
            "P": int(P),
            "R": int(R),
            "L": int(L),
            "M": int(M),
        }




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
        cra_states: Optional[dict] = None,
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
            cra_states=cra_states,
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

        runtime_cra_states = {} if past_key_values is None else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states, extra = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                cra_states=runtime_cra_states,
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
