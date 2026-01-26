import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
import os, sys

# Keep this repo root on sys.path so local kernels import cleanly.
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from kernels.sparse import build_sparse_list_decode, sparse_attention_fwd


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


class CUDATimer:
    __slots__ = ("enabled", "start", "end")

    def __init__(self, enabled: bool):
        self.enabled = enabled
        if enabled:
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
        else:
            self.start = None
            self.end = None

    def __enter__(self):
        if self.enabled:
            self.start.record()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.enabled:
            self.end.record()

    def ms(self) -> float:
        if not self.enabled:
            return 0.0
        # Ensure timing events are completed before measuring.
        # This is required with async kernels / non-default streams.
        self.end.synchronize()
        return self.start.elapsed_time(self.end)


def _prof_init(attn_obj):
    attn_obj._prof = {
        "qkv_rope": 0.0,
        "cache_update": 0.0,
        "index_build": 0.0,
        "kv_relayout": 0.0,
        "sparse_kernel": 0.0,
        "wo": 0.0,
        "tokens_decode": 0,
        "tokens_prefill": 0,
    }


def _prof_print(attn_obj, prefix="[tok-sparse prof] ", reset=True):
    p = getattr(attn_obj, "_prof", None)
    if not p:
        print(prefix + "No profiling data.")
        return

    def fmt(name, ms_total, n):
        if n == 0:
            return f"{name:14s}: {ms_total:8.3f} ms total"
        ms_tok = ms_total / n
        tok_s = 1000.0 / ms_tok if ms_tok > 0 else float("inf")
        return f"{name:14s}: {ms_total:8.3f} ms total | {ms_tok:7.4f} ms/tok | {tok_s:8.2f} tok/s"

    ndec = p["tokens_decode"]
    npre = p["tokens_prefill"]

    if npre:
        print(prefix + "=== Prefill breakdown ===")
        print(prefix + fmt("qkv_rope", p["qkv_rope"], npre))
        print(prefix + fmt("cache_update", p["cache_update"], npre))
        print(prefix + fmt("wo", p["wo"], npre))

    if ndec:
        print(prefix + "=== Decode breakdown ===")
        print(prefix + fmt("qkv_rope", p["qkv_rope"], ndec))
        print(prefix + fmt("cache_update", p["cache_update"], ndec))
        print(prefix + fmt("kv_relayout", p["kv_relayout"], ndec))
        print(prefix + fmt("index_build", p["index_build"], ndec))
        print(prefix + fmt("sparse_kernel", p["sparse_kernel"], ndec))
        print(prefix + fmt("wo", p["wo"], ndec))

    if reset:
        _prof_init(attn_obj)


@dataclass
class ModelArgs:
    block_size: int = 300000
    vocab_size: int = 32000
    n_layer: int = 1
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 40000
    norm_eps: float = 1e-5
    L: int = 60
    R: int = 1024
    K: int = 10
    heavy_const: int = 3600  # budget

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.dim // self.n_head

    @classmethod
    def from_name(cls, name: str):
        if name in transformer_configs:
            return cls(**transformer_configs[name])
        config = [c for c in transformer_configs if c in str(name).upper() or c in str(name)]
        assert len(config) == 1, name
        return cls(**transformer_configs[config[0]])


transformer_configs = {
    "CodeLlama-7b-Python-hf": dict(block_size=16384, vocab_size=32000, n_layer=1, dim=4096, rope_base=1000000),
    "Llama-2-7b-chat-hf": dict(block_size=300000, vocab_size=32000, n_layer=1, n_head=32, dim=4096, rope_base=10000),
    "7B": dict(n_layer=1, n_head=32, dim=4096),
    "13B": dict(n_layer=1, n_head=40, dim=5120),
    "30B": dict(n_layer=1, n_head=52, dim=6656),
    "34B": dict(
        n_layer=1,
        n_head=64,
        dim=8192,
        vocab_size=32000,
        n_local_heads=8,
        intermediate_size=22016,
        rope_base=1000000,
    ),
    "70B": dict(n_layer=1, n_head=64, dim=8192, n_local_heads=8, intermediate_size=28672),
}


class KVCache(nn.Module):
    """
    KV is stored as (B, H, T, D).
    """

    def __init__(
        self,
        max_batch_size: int,
        max_seq_length: int,
        n_heads: int,
        head_dim: int,
        L: int,
        R: int,
        dtype=torch.bfloat16,
    ):
        super().__init__()
        B, H, T, D = max_batch_size, n_heads, max_seq_length, head_dim

        self.register_buffer("k_cache", torch.zeros((B, H, T, D), dtype=dtype))
        self.register_buffer("v_cache", torch.zeros((B, H, T, D), dtype=dtype))

        self.L = L
        self.R = R

        self.register_buffer("k_hard", torch.zeros((B, H, T, L), dtype=torch.int16))
        self.register_buffer("v_norm", torch.zeros((B, H, T), dtype=torch.float16))

        self.register_buffer("attn_out", torch.zeros((B, H, D), dtype=dtype))
        self.register_buffer("prefill_len", torch.zeros((), dtype=torch.int32))

    def update(
        self,
        input_pos: Tensor,
        k_val: Tensor,
        v_val: Tensor,
        v_norm: Optional[Tensor] = None,
        k_hard: Optional[Tensor] = None,
    ):
        """
        input_pos: [S] (positions)
        k_val: [B, S, H, D]
        v_val: [B, S, H, D]
        v_norm: [B, S, H] (optional)
        k_hard: [B, S, H, L] (optional)
        """
        if input_pos.dtype != torch.long:
            input_pos = input_pos.long()

        assert input_pos.ndim == 1, f"input_pos must be [S], got {tuple(input_pos.shape)}"
        S = int(input_pos.numel())
        assert k_val.ndim == 4 and v_val.ndim == 4
        assert k_val.shape[1] == S and v_val.shape[1] == S, "k/v S dim must match input_pos length"

        Tcap = self.k_cache.size(2)
        max_pos = int(input_pos.max().item()) if S > 0 else -1
        min_pos = int(input_pos.min().item()) if S > 0 else 0
        if max_pos >= Tcap or min_pos < 0:
            raise RuntimeError(
                f"KVCache.update out-of-bounds: input_pos in [{min_pos},{max_pos}] but cache T={Tcap}. "
                f"Did you call setup_caches(max_seq_length >= max(input_pos)+1)?"
            )

        # [B,S,H,D] -> [B,H,S,D] and write into T
        self.k_cache[:, :, input_pos, :] = k_val.permute(0, 2, 1, 3).contiguous()
        self.v_cache[:, :, input_pos, :] = v_val.permute(0, 2, 1, 3).contiguous()

        if v_norm is not None:
            self.v_norm[:, :, input_pos] = v_norm.permute(0, 2, 1).contiguous()
        if k_hard is not None:
            self.k_hard[:, :, input_pos, :] = k_hard.permute(0, 2, 1, 3).contiguous()

        self.prefill_len.fill_(max(self.prefill_len.item(), max_pos + 1))
        return self.k_cache, self.v_cache


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Buffers that must follow the model device
        self.register_buffer("freqs_cis", None, persistent=False)
        self.register_buffer("causal_mask", None, persistent=False)

        self.max_batch_size = -1
        self.max_seq_length = -1

    def setup_caches(self, max_batch_size, max_seq_length):
        if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
            return

        device = self.tok_embeddings.weight.device
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)

        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size

        for b in self.layers:
            b.attention.kv_cache = KVCache(
                max_batch_size,
                max_seq_length,
                self.config.n_local_heads,
                head_dim,
                L=self.config.L,
                R=self.config.R,
            ).to(device=device)

        self.freqs_cis = precompute_freqs_cis(
            self.config.block_size,
            self.config.dim // self.config.n_head,
            self.config.rope_base,
        ).to(device=device)

        self.causal_mask = torch.tril(
            torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool, device=device)
        )

    def _check_input_pos(self, input_pos: Tensor):
        if input_pos.dtype != torch.long:
            input_pos = input_pos.long()
        max_pos = int(input_pos.max().item())
        min_pos = int(input_pos.min().item())
        if min_pos < 0 or max_pos >= self.max_seq_length:
            raise RuntimeError(
                f"input_pos out of bounds: [{min_pos},{max_pos}] vs max_seq_length={self.max_seq_length}. "
                f"Call setup_caches(max_seq_length >= max(input_pos)+1)."
            )
        return input_pos

    def forward(self, idx: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        assert self.freqs_cis is not None and self.causal_mask is not None, "Caches must be initialized first"
        input_pos = self._check_input_pos(input_pos)

        # Optional extra safety:
        if idx.dtype != torch.long:
            idx = idx.long()
        if idx.numel() > 0:
            mx = int(idx.max().item())
            mn = int(idx.min().item())
            if mn < 0 or mx >= self.config.vocab_size:
                raise RuntimeError(f"Token id out of vocab: [{mn},{mx}] vs vocab_size={self.config.vocab_size}")

        mask = self.causal_mask[None, None, input_pos]  # [1,1,S,T]
        freqs_cis = self.freqs_cis[input_pos]           # [S, ...]
        x = self.tok_embeddings(idx)

        for layer in self.layers:
            x = layer(x, input_pos, freqs_cis, mask)
        x = self.norm(x)
        return self.output(x)

    def sparse_forward(self, idx: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        assert self.freqs_cis is not None and self.causal_mask is not None, "Caches must be initialized first"
        input_pos = self._check_input_pos(input_pos)

        if idx.dtype != torch.long:
            idx = idx.long()

        mask = self.causal_mask[None, None, input_pos]
        freqs_cis = self.freqs_cis[input_pos]
        x = self.tok_embeddings(idx)

        for layer in self.layers:
            x = layer.sparse_forward(x, input_pos, freqs_cis, mask)
        x = self.norm(x)
        # self.layers[0].attention.print_prof(reset=True)
        return self.output(x)

    @classmethod
    def from_name(cls, name: str):
        return cls(ModelArgs.from_name(name))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(self, x: Tensor, input_pos: Tensor, freqs_cis: Tensor, mask: Tensor) -> Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, input_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def sparse_forward(self, x: Tensor, input_pos: Tensor, freqs_cis: Tensor, mask: Tensor) -> Tensor:
        h = x + self.attention.sparse_forward(self.attention_norm(x), freqs_cis, mask, None, input_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self._register_load_state_dict_pre_hook(self.load_hook)

        self.config = config
        self.L = config.L
        self.R = config.R
        self.K = config.K
        self.heavy_const = config.heavy_const

        self.register_buffer("planes", torch.randn(self.L, self.K, self.head_dim) * 0.02)
        self.register_buffer("protos_T", torch.randn(self.K, self.R) * 0.02)

        _prof_init(self)

    def print_prof(self, prefix="[tok-sparse prof] ", reset=True):
        _prof_print(self, prefix=prefix, reset=reset)

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def soft_hash(self, queries_bhd: Tensor) -> Tensor:
        queries = queries_bhd.unsqueeze(2)  # [B,H,1,D]
        q_proj = torch.einsum("bhqd,lkd->bhqlk", queries, self.planes)  # [B,H,1,L,K]
        temp = math.sqrt(queries.size(-1))
        logits = torch.einsum(
            "bhqlk,kr->bhqlr",
            torch.tanh(q_proj) / max(temp, 1e-6),
            self.protos_T,
        )  # [B,H,1,L,R]
        return torch.softmax(logits, dim=-1).squeeze(2)  # [B,H,L,R]

    def pack_bits(self, bits: Tensor) -> Tensor:
        K = bits.shape[-1]
        weights = (1 << torch.arange(K - 1, -1, -1, device=bits.device, dtype=torch.int16))
        view_shape = (1,) * (bits.ndim - 1) + (K,)
        return (bits.to(torch.int16) * weights.view(view_shape)).sum(dim=-1)

    @torch.no_grad()
    def hard_hash_keys(self, keys_bshd: Tensor) -> Tensor:
        proj = torch.einsum("bshd,lkd->bshlk", keys_bshd, self.planes)
        bits = proj >= 0
        return self.pack_bits(bits).to(torch.int16)  # [B,S,Hl,L]

    def sparse_forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask1: torch.Tensor,      # [1,1,S,Tmax] bool
        mask2: torch.Tensor,      # unused
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert input_pos is not None, "sparse_forward expects input_pos"
        if input_pos.dtype != torch.long:
            input_pos = input_pos.long()

        bsz, seqlen, _ = x.shape
        assert self.kv_cache is not None, "Call setup_caches() first so kv_cache exists"

        p = self._prof
        cuda_timing = False

        # QKV + RoPE
        with CUDATimer(cuda_timing) as t_qkv:
            kv_size = self.n_local_heads * self.head_dim
            q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

            q = q.view(bsz, seqlen, self.n_head, self.head_dim)
            k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
            v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

            q = apply_rotary_emb(q, freqs_cis)
            k = apply_rotary_emb(k, freqs_cis)
        p["qkv_rope"] += t_qkv.ms()

        # Cache update (+ metadata)
        with CUDATimer(cuda_timing) as t_cache:
            with torch.no_grad():
                k_hard = self.hard_hash_keys(k)  # [B,S,Hl,L]
                v_norm = torch.linalg.vector_norm(v.float(), ord=2, dim=-1).to(torch.float16)  # [B,S,Hl]
            k_cache, v_cache = self.kv_cache.update(input_pos, k, v, v_norm=v_norm, k_hard=k_hard)  # [B,Hl,T,D]
        p["cache_update"] += t_cache.ms()

        if seqlen == 1:
            p["tokens_decode"] += int(bsz)
        else:
            p["tokens_prefill"] += int(bsz * seqlen)

        assert self.n_head % self.n_local_heads == 0
        rep = self.n_head // self.n_local_heads

        # SDPA expects [B,H,S,D]
        q_sdpa = q.transpose(1, 2)  # [B,H,S,D]

        # Prefill: dense attention
        if seqlen != 1:
            if rep == 1:
                k_sdpa = k_cache
                v_sdpa = v_cache
            else:
                k_sdpa = k_cache.repeat_interleave(rep, dim=1)
                v_sdpa = v_cache.repeat_interleave(rep, dim=1)

            y = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, attn_mask=mask1, dropout_p=0.0)

            with CUDATimer(cuda_timing) as t_wo:
                y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
                out = self.wo(y)
            p["wo"] += t_wo.ms()
            return out

        # Decode
        T = int(self.kv_cache.prefill_len.item())
        assert T > 0


        # allowed_bt = mask1[0, 0, :, :T]     # [B,T]  True=allowed
        # allowed_bht = allowed_bt[:, None, :].expand(bsz, self.n_head, T).contiguous()

        pos = input_pos.view(-1)
        allowed = torch.arange(T, device=pos.device) <= pos.max()
        allowed_bht = allowed.view(1, 1, T).expand(bsz, self.n_head, T).contiguous()

        with CUDATimer(cuda_timing) as t_relayout:
            # k_hard: [B,Hl,T,L] -> [B,H,L,T]
            k_hard_pref = self.kv_cache.k_hard[:, :, :T, :]  # [B,Hl,T,L]
            if rep != 1:
                k_hard_pref = k_hard_pref.repeat_interleave(rep, dim=1)  # [B,H,T,L]
            k_hard_bhlt = k_hard_pref.permute(0, 1, 3, 2).contiguous()   # [B,H,L,T]

            # v_norm: [B,Hl,T] -> [B,H,T]
            v_norm_pref = self.kv_cache.v_norm[:, :, :T]  # [B,Hl,T]
            if rep != 1:
                v_norm_bht = v_norm_pref.repeat_interleave(rep, dim=1)
            else:
                v_norm_bht = v_norm_pref

            q_bhd = q_sdpa[:, :, 0, :].contiguous()  # [B,H,D]
        p["kv_relayout"] += t_relayout.ms()

        sink = int(getattr(self.config, "sink_size", 120))
        window = int(getattr(self.config, "window_size", 120))
        M = int(self.heavy_const)
        sink = max(0, min(sink, T))
        window = max(0, min(window, T))
        M = max(0, min(M, T))

        q_probs = self.soft_hash(q_bhd)  # [B,H,L,R]

        with CUDATimer(cuda_timing) as t_index:
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
        p["index_build"] += t_index.ms()

        # Backend wants [B,Hl,T,D]
        k_backend = k_cache[:, :, :T, :].contiguous()
        v_backend = v_cache[:, :, :T, :].contiguous()

        with CUDATimer(cuda_timing) as t_sparse:
            out_bhd = sparse_attention_fwd(
                q_bhd,            # [B,H,D]
                k_backend,        # [B,Hl,T,D]
                v_backend,        # [B,Hl,T,D]
                sparse_list,      # [B,H,Ktotal]
                sparse_len,       # [B,H]
                block_seq=256,
            )
        p["sparse_kernel"] += t_sparse.ms()

        with CUDATimer(cuda_timing) as t_wo:
            y = out_bhd.unsqueeze(2)  # [B,H,1,D]
            y = y.transpose(1, 2).contiguous().view(bsz, 1, self.dim)
            out = self.wo(y)
        p["wo"] += t_wo.ms()
        return out

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        # print("DENSE")
        bsz, seqlen, _ = x.shape
        if input_pos.dtype != torch.long:
            input_pos = input_pos.long()

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        if self.kv_cache is not None:
            k_cache, v_cache = self.kv_cache.update(input_pos, k, v)  # [B,Hl,T,D]
            k = k_cache
            v = v_cache

        q = q.transpose(1, 2)  # [B,H,S,D]
        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)  # [B,H,T,D]
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)  # [B,H,T,D]

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        return self.wo(y)


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000) -> Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=torch.bfloat16)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)
