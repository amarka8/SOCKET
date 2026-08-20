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

# True only when a LEGACY (pre-optimization) path that consumes the [B,H,maxlen] bool
# `allowed` tensor is selected. Read once at import -> a torch.compile guard, not a
# graph break.
# MUST stay in sync with kernels/sparse.py::_SCORER_IMPL (default "triton"; see the long
# comment there -- the old "cuda" default was only ever competitive because that kernel
# silently did not execute under CUDA-graph replay).
_SCORER_IMPL = os.environ.get(
    "SOCKET_SCORER_IMPL",
    "cuda" if os.environ.get("SOCKET_TRITON_SCORER", "1") == "0" else "triton",
).strip().lower()
_NEEDS_ALLOWED_MASK = (
    _SCORER_IMPL == "cuda"                       # the CUDA scorer takes allowed_ext
    or os.environ.get("SOCKET_FUSED_LIST", "1") != "1"   # eager list assembly gathers it
)


# ---------------------------------------------------------------------------
# KV-cache storage layout policy.
#   dense  -> "bthd"  = [B, T, Hkv, D], flash_attn_with_kvcache's native layout, so dense
#             decode hands FA the cache buffer with ZERO per-step relayout.
#   sparse -> "bhtd"  = [B, Hkv, T, D], the layout the SOCKET gather kernels require. The
#             sparse path is therefore BIT-IDENTICAL to before this change.
# SOCKET_KV_LAYOUT overrides: "bhtd" reproduces the OLD (relayout-every-step) dense path
# exactly, which is what the within-job dense control measurement runs.
# ---------------------------------------------------------------------------
def resolve_kv_layout(decode_type: str = "sparse") -> str:
    env = os.environ.get("SOCKET_KV_LAYOUT", "auto").strip().lower()
    if env in ("bhtd", "bthd"):
        return env
    if env not in ("", "auto"):
        raise ValueError(f"SOCKET_KV_LAYOUT must be one of auto|bhtd|bthd, got {env!r}")
    return "bthd" if decode_type == "dense" else "bhtd"


# ---------------------------------------------------------------------------
# FA2/FA3 dense-attention backend selection (PLUMBING ONLY — does not touch the
# SOCKET sparse path). Copied verbatim from the reference fork so the dense
# baseline can be timed under BOTH FA2 and FA3 in the same env.
# ---------------------------------------------------------------------------
_flash_attn_func = None
_flash_attn_with_kvcache = None       # decode kernel (static cache + cache_seqlens); None if absent
_FLASH_IS_FA3 = False
try:
    # Allow forcing the FA2 backend even when the FA3 package is importable.
    if os.getenv("SOCKET_FORCE_FA2", "0") == "1":
        raise ImportError("SOCKET_FORCE_FA2=1: forcing FA2 dense backend")
    from flash_attn_interface import flash_attn_func as _flash_attn_func
    _FLASH_IS_FA3 = True
    _FLASH_BACKEND_NAME = "FA3(flash_attn_interface)"
    try:
        from flash_attn_interface import flash_attn_with_kvcache as _flash_attn_with_kvcache
    except Exception:
        _flash_attn_with_kvcache = None   # FA3 hopper fork may not expose a python with_kvcache
except Exception:
    try:
        from flash_attn import flash_attn_func as _flash_attn_func
        import flash_attn as _fa2_pkg
        _FLASH_BACKEND_NAME = "FA2(flash_attn==%s)" % getattr(_fa2_pkg, "__version__", "?")
        try:
            from flash_attn import flash_attn_with_kvcache as _flash_attn_with_kvcache
        except Exception:
            _flash_attn_with_kvcache = None
    except Exception:
        _flash_attn_func = None
        _FLASH_BACKEND_NAME = "none"

_FLASH_PREFILL_WARNED = False
_FLASH_BACKEND_LOGGED = False

# ---------------------------------------------------------------------------
# Measurement integrity: prove which attention kernel actually ran, refuse to
# silently mislabel. SOCKET_REQUIRE_BACKEND in {fa2,fa3,flash,sdpa} turns the
# SDPA fallback into a HARD error and asserts the running kernel matches. The
# print/assert is one-shot and is_compiling-guarded so it never enters the
# compiled graph; it fires in EAGER calls (the short non-compiled probe run)
# whose env matches the compiled timed run.
# ---------------------------------------------------------------------------
_REQUIRE_BACKEND = os.getenv("SOCKET_REQUIRE_BACKEND", "").strip().lower()
_BACKEND_ASSERTED = False


def _assert_backend(ran: str):
    global _BACKEND_ASSERTED
    if _BACKEND_ASSERTED:
        return
    _BACKEND_ASSERTED = True
    eff = _FLASH_BACKEND_NAME if ran == "flash" else "SDPA(torch.sdpa)"
    req = _REQUIRE_BACKEND
    print(f"[ATTN-BACKEND] required={req or '(none)'} effective={eff} ran={ran}",
          file=sys.stderr, flush=True)
    if req:
        ok = ((req == "sdpa" and ran == "sdpa")
              or (req == "flash" and ran == "flash")
              or (req == "fa2" and ran == "flash" and _FLASH_BACKEND_NAME.startswith("FA2"))
              or (req == "fa3" and ran == "flash" and _FLASH_BACKEND_NAME.startswith("FA3")))
        if not ok:
            raise RuntimeError(
                f"SOCKET_REQUIRE_BACKEND={req} but ran={ran} effective={eff} "
                f"— refusing to report a mislabeled backend.")


# Fullgraph + CUDA-graph safe FA2/FA3 dense DECODE op: wraps flash_attn_with_kvcache
# (a raw pybind Dynamo cannot trace -> would graph-break under fullgraph=True) as an
# opaque torch custom op. Uses an on-device cache_seqlens (NO .item()/dynamic slice) so
# the cache stays a STATIC shape. Called WITHOUT k=/v= (no in-cache write) -> mutates_args=().
if _flash_attn_with_kvcache is not None:
    @torch.library.custom_op("socket::flash_dense_decode", mutates_args=())
    def _flash_dense_decode_op(q_bshd: Tensor, k_bthd: Tensor, v_bthd: Tensor,
                               cache_seqlens: Tensor) -> Tensor:
        o = _flash_attn_with_kvcache(q_bshd, k_bthd, v_bthd,
                                     cache_seqlens=cache_seqlens, causal=True)
        # All FA2/FA3 with_kvcache variants in this env return a bare Tensor (default
        # return_softmax_lse=False); defensively unwrap (out, lse) like _dense_attention so a
        # tuple-returning build can't crash the `-> Tensor` op. Behavior-identical in target env.
        return o[0] if isinstance(o, tuple) else o

    @_flash_dense_decode_op.register_fake
    def _flash_dense_decode_fake(q_bshd, k_bthd, v_bthd, cache_seqlens):
        return torch.empty_like(q_bshd)


def _dense_attention(
    q_bhsd: Tensor,
    k_bhtd: Tensor,
    v_bhtd: Tensor,
    attn_mask: Optional[Tensor],
    input_pos: Optional[Tensor] = None,
) -> Tensor:
    """Dense attention (prefill + dense decode fallback) with optional FA2/FA3 backend.
    Inputs/outputs are [B, H, S/T, D]."""
    use_flash = (
        os.getenv("USE_FLASHATTN3", "1") == "1"
        and _flash_attn_func is not None
        and q_bhsd.is_cuda
        and q_bhsd.dtype in (torch.float16, torch.bfloat16)
    )

    global _FLASH_BACKEND_LOGGED
    if os.getenv("FLASH_ATTN_DEBUG", "0") == "1" and not _FLASH_BACKEND_LOGGED:
        backend = "flash" if use_flash else "sdpa"
        print(f"[attention] backend={backend} use_flash={use_flash} cuda={q_bhsd.is_cuda} "
              f"dtype={q_bhsd.dtype} flash_func_available={_flash_attn_func is not None}",
              flush=True)
        _FLASH_BACKEND_LOGGED = True

    if use_flash:
        try:
            q_bshd = q_bhsd.transpose(1, 2).contiguous()
            k_bthd = k_bhtd.transpose(1, 2).contiguous()
            v_bthd = v_bhtd.transpose(1, 2).contiguous()

            if input_pos is not None and input_pos.numel() > 0 and not torch.compiler.is_compiling():
                kv_len = int(input_pos.max().item()) + 1
                kv_len = min(kv_len, k_bthd.size(1))
                k_bthd = k_bthd[:, :kv_len]
                v_bthd = v_bthd[:, :kv_len]
            elif (torch.compiler.is_compiling() and q_bshd.size(1) == 1 and k_bthd.size(1) > 1):
                # COMPILE + single-query DECODE through this dense FALLBACK (NOT the with_kvcache
                # op): the eager-only truncation above is skipped (it would host-sync), and FA's
                # causal mask bottom-right-aligns the lone query to position maxlen-1, so it would
                # attend the UNFILLED zero tail -> silently wrong output + a mislabeled "flash"
                # backend. This path is unreachable in the target run (FA2/FA3 both export
                # flash_attn_with_kvcache, so dense decode routes through socket::flash_dense_decode
                # which uses on-device cache_seqlens correctly). Refuse it loudly rather than
                # measure garbage. Prefill (seqlen_q>1) and the with_kvcache decode op are unaffected.
                raise RuntimeError(
                    "dense DECODE under torch.compile reached the _dense_attention flash fallback "
                    "without flash_attn_with_kvcache; FA causal alignment would attend the unfilled "
                    "cache tail. Use a flash build that exposes flash_attn_with_kvcache (target env "
                    "does), or run the SDPA control (USE_FLASHATTN3=0).")

            out_bshd = _flash_attn_func(q_bshd, k_bthd, v_bthd, causal=True)
            if isinstance(out_bshd, tuple):   # FA3 may return (out, softmax_lse)
                out_bshd = out_bshd[0]
            if not torch.compiler.is_compiling():
                _assert_backend("flash")
            return out_bshd.transpose(1, 2).contiguous()
        except Exception as exc:
            global _FLASH_PREFILL_WARNED
            if _REQUIRE_BACKEND in ("flash", "fa2", "fa3"):
                raise
            if not _FLASH_PREFILL_WARNED:
                print(f"[attention] FlashAttention path failed, falling back to SDPA: "
                      f"{type(exc).__name__}: {exc}", flush=True)
                _FLASH_PREFILL_WARNED = True

    if not torch.compiler.is_compiling():
        _assert_backend("sdpa")
    return F.scaled_dot_product_attention(
        q_bhsd, k_bhtd, v_bhtd, attn_mask=attn_mask, dropout_p=0.0,
    )


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
    rope_scaling: Optional[dict] = None
    norm_eps: float = 1e-5
    L: int = 60
    R: int = 256
    K: int = 8
    heavy_const: int = 860  # budget
    tau: float = 0.3  # soft-hash softmax temperature (matches sparse-attention-hub default)
    # sink/window streaming-attention base. These were previously read only via
    # getattr(config, 'sink_size', 120) with NO field, so the benchmark's 33.3x sparsity
    # (budget = sink + window + heavy_const = 120 + 120 + HEAVY) silently depended on the
    # getattr fallback. Made explicit (same 120/120 -> identical selection/sparsity) so the
    # contract is visible. Do NOT change these values: they set the measured sparsity.
    sink_size: int = 120
    window_size: int = 120

    def __post_init__(self):
        # BENCHMARK-ONLY: n_layer is env-overridable so the 1-layer microbench config and the
        # full 32-layer model can be driven from ONE source tree (the previous setup used a
        # hand-edited COPY of GPT-FAST, which silently drifts from any kernel change). Absent
        # SOCKET_N_LAYER the value from transformer_configs is used unchanged -> no behavior
        # change for any existing caller.
        _nl_env = os.environ.get("SOCKET_N_LAYER")
        if _nl_env is not None:
            self.n_layer = int(_nl_env)
        # L and heavy_const are env-overridable so a single build can sweep configs (the
        # original repo hard-codes them; we expose SOCKET_L / SOCKET_HEAVY_CONST to match
        # the reference fork's sweep harness without changing any SOCKET math).
        self.L = int(os.environ.get("SOCKET_L", self.L))
        self.heavy_const = int(os.environ.get("SOCKET_HEAVY_CONST", self.heavy_const))
        # K (SRP bits per table) and R (=2**K hypercube corners) are also env-overridable
        # so a single build can sweep the soft-hash code width. R is structurally pinned to
        # 2**K (protos_T = {-1,+1}^K corners, pack_bits codes in [0, 2**K)); ENFORCE it here
        # so a stray SOCKET_R can never desync from K. Defaults stay K=8 / R=256.
        _k_env = os.environ.get("SOCKET_K")
        _r_env = os.environ.get("SOCKET_R")
        if _k_env is not None:
            self.K = int(_k_env)
        if _r_env is not None:
            self.R = int(_r_env)
        if _k_env is not None and _r_env is None:
            # K given, R not: derive R = 2**K.
            self.R = 1 << self.K
        if _k_env is not None and _r_env is not None:
            # both given: they must agree with the structural constraint.
            assert self.R == (1 << self.K), (
                f"SOCKET_R ({self.R}) must equal 2**SOCKET_K (2**{self.K} = {1 << self.K})"
            )
        if _k_env is None and _r_env is not None:
            # only R given: derive/validate K from R (R must be a power of two).
            assert self.R > 0 and (self.R & (self.R - 1)) == 0, (
                f"SOCKET_R ({self.R}) must be a power of two"
            )
            self.K = self.R.bit_length() - 1
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
    "llama-3.1-8b": dict(
        block_size=300000,
        n_layer=1,
        n_head=32,
        n_local_heads=8,
        dim=4096,
        intermediate_size=14336,
        vocab_size=128256,
        rope_base=500000,
        rope_scaling=dict(
            factor=8.0,
            low_freq_factor=1.0,
            high_freq_factor=4.0,
            original_max_position_embeddings=8192,
        ),
    ),
}


class KVCache(nn.Module):
    """
    KV storage layout is SELECTABLE (`layout`):

      "bhtd" (default, LEGACY): k_cache/v_cache are (B, Hkv, T, D).  This is the layout the
             SOCKET sparse path's gather kernels require, so the sparse path always uses it
             and is bit-identical to before this change.

      "bthd" (FAIR DENSE): k_cache/v_cache are (B, T, Hkv, D) — exactly the layout
             flash_attn_with_kvcache consumes.  The dense decode path then hands the cache
             buffer straight to FA with NO per-step relayout.  Previously dense decode did
             `k.transpose(1,2).contiguous()` on the WHOLE cache every layer every step
             (2 x 587 MB/layer at 140K -> ~9.2 ms/token over 32 layers, i.e. ~60% of dense
             "attention" time was a layout copy, not attention).  This mirrors what the
             SOCKET path already did for its `k_hard` buffer (see the RANK-1 note below).

    Only k_cache/v_cache are affected; k_hard / v_norm / attn_out are SOCKET-only and
    unchanged.
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
        layout: str = "bhtd",
    ):
        super().__init__()
        B, H, T, D = max_batch_size, n_heads, max_seq_length, head_dim

        assert layout in ("bhtd", "bthd"), f"unknown KV layout {layout!r}"
        self.layout = layout
        kv_shape = (B, T, H, D) if layout == "bthd" else (B, H, T, D)
        self.register_buffer("k_cache", torch.zeros(kv_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(kv_shape, dtype=dtype))

        self.L = L
        self.R = R

        # RANK-1: store k_hard NATIVELY transposed as [B, Hkv, L, T] (the scorer's read layout)
        # so the per-decode-token full-buffer permute().contiguous() relayout (~210MB round-trip
        # @128K, the #1 T-scaling DRAM op) is eliminated — KVCache.update writes only the new
        # column along the last (T) axis. Byte-equal to the old [B,Hkv,T,L]+relayout (gate: T7).
        self.register_buffer("k_hard", torch.zeros((B, H, L, T), dtype=torch.int16))
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

        Tcap = self.k_cache.size(1) if self.layout == "bthd" else self.k_cache.size(2)
        # data-dependent bounds check: eager-only (skipped under compile/cudagraph; the caller
        # sizes the cache to T_new so it cannot overflow during the timed run).
        if not torch.compiler.is_compiling():
            max_pos = int(input_pos.max().item()) if S > 0 else -1
            min_pos = int(input_pos.min().item()) if S > 0 else 0
            if max_pos >= Tcap or min_pos < 0:
                raise RuntimeError(
                    f"KVCache.update out-of-bounds: input_pos in [{min_pos},{max_pos}] but cache T={Tcap}. "
                    f"Did you call setup_caches(max_seq_length >= max(input_pos)+1)?"
                )

        if self.layout == "bthd":
            # Cache is ALREADY [B,T,H,D] == the incoming k_val layout: write the S new rows
            # straight into the T axis, no permute, no relayout anywhere downstream.
            self.k_cache[:, input_pos, :, :] = k_val
            self.v_cache[:, input_pos, :, :] = v_val
        else:
            # [B,S,H,D] -> [B,H,S,D] and write into the T axis.
            self.k_cache[:, :, input_pos, :] = k_val.permute(0, 2, 1, 3).contiguous()
            self.v_cache[:, :, input_pos, :] = v_val.permute(0, 2, 1, 3).contiguous()

        if v_norm is not None:
            self.v_norm[:, :, input_pos] = v_norm.permute(0, 2, 1).contiguous()
        if k_hard is not None:
            # k_hard input is [B,S,Hkv,L]; the buffer is [B,Hkv,L,T] (RANK-1 native layout), so
            # write the S new columns along the LAST (T) axis: permute [B,S,Hkv,L] -> [B,Hkv,L,S].
            self.k_hard[:, :, :, input_pos] = k_hard.permute(0, 2, 3, 1).contiguous()

        # prefill_len = max(prefill_len, max(input_pos)+1) computed entirely on-device so there
        # is NO host sync (.item()) inside the compiled region.
        new_len = input_pos.max().to(self.prefill_len.dtype) + 1
        self.prefill_len.copy_(torch.maximum(self.prefill_len, new_len))
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

    def setup_caches(self, max_batch_size, max_seq_length, decode_type: str = "sparse"):
        if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
            return

        device = self.tok_embeddings.weight.device
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)

        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size

        kv_layout = resolve_kv_layout(decode_type)
        for b in self.layers:
            b.attention.kv_cache = KVCache(
                max_batch_size,
                max_seq_length,
                self.config.n_local_heads,
                head_dim,
                L=self.config.L,
                R=self.config.R,
                layout=kv_layout,
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
        # data-dependent (.item()+branch) -> eager-only; Dynamo constant-folds
        # is_compiling()==True and prunes it under fullgraph=True.
        if not torch.compiler.is_compiling():
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
        if not torch.compiler.is_compiling() and idx.numel() > 0:
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
        # heavy_const (the per-query token budget) is env-overridable so a single build
        # can sweep sparsity = N / (heavy_const + sink + window) across context lengths.
        self.heavy_const = int(os.environ.get("SOCKET_HEAVY_CONST", config.heavy_const))
        self.tau = float(config.tau)
        # ABLATION ONLY (bit-exact, T6-proven): SOCKET_FORCE_REPEAT=1 re-enables the OLD
        # pre-optimization relayout that repeat_interleave's k_hard/v_norm 8->32 before the
        # scorer (rep=1 in-kernel). Default 0 = per-kv-head (Hkv=8). Used to attribute the
        # throughput delta of the per-kv-head optimization. Read once here (constant under
        # torch.compile -> a guard, not a graph break).
        self._force_repeat = bool(int(os.environ.get("SOCKET_FORCE_REPEAT", "0")))

        # SRP hyperplanes ~ N(0,1) (paper/hub use std 1.0; the query soft-hash
        # tanh(Wq)/sqrt(d) needs full-scale projections, not the squashed *0.02 ones).
        self.register_buffer("planes", torch.randn(self.L, self.K, self.head_dim))
        # protos_T = hypercube corners {-1,+1}^K, column r == the sign pattern whose
        # big-endian pack_bits code is r (matches hub get_protos_T). This is a FIXED
        # constant, NOT random — random columns destroy the soft-collision signal.
        assert self.R == (1 << self.K), f"R ({self.R}) must equal 2**K ({1 << self.K})"
        import itertools as _it
        _corners = torch.tensor(list(_it.product([-1.0, 1.0], repeat=self.K)))  # [R, K]
        self.register_buffer("protos_T", _corners.t().contiguous())  # [K, R]

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
        return torch.softmax(logits / self.tau, dim=-1).squeeze(2)  # [B,H,L,R]

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
        # SOCKET's gather kernels read [B,Hkv,T,D]; the fair-dense [B,T,Hkv,D] layout must never
        # reach this path (resolve_kv_layout() only hands "bthd" to decode_type="dense").
        assert self.kv_cache.layout == "bhtd", (
            f"sparse decode requires the [B,Hkv,T,D] KV layout, got {self.kv_cache.layout!r}")

        p = self._prof
        # Profiling timers do device .synchronize() per stage per decode token, which would
        # serialize the launch-bound decode path AND inject syncs into the compiled region.
        # Default OFF (no _prof_enabled flag) so throughput is measured fairly and the graph
        # stays cudagraph-capturable. Bit-exact either way.
        cuda_timing = x.is_cuda and bool(getattr(self, "_prof_enabled", False))

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

        # Token counters feed only the (disabled) profiling printout. They mutate Python ints
        # that torch.compile guards on (-> per-step recompile), so skip under compile.
        if not torch.compiler.is_compiling():
            if seqlen == 1:
                p["tokens_decode"] += int(bsz)
            else:
                p["tokens_prefill"] += int(bsz * seqlen)

        assert self.n_head % self.n_local_heads == 0
        rep = self.n_head // self.n_local_heads

        # SDPA expects [B,H,S,D]
        q_sdpa = q.transpose(1, 2)  # [B,H,S,D]

        # Prefill: dense attention (routed through _dense_attention so the dense baseline's
        # FA2/FA3 backend selection is shared; sparse-path prefill is excluded from steady
        # decode timing so the backend used here does not affect the SOCKET decode numbers).
        if seqlen != 1:
            if rep == 1:
                k_sdpa = k_cache
                v_sdpa = v_cache
            else:
                k_sdpa = k_cache.repeat_interleave(rep, dim=1)
                v_sdpa = v_cache.repeat_interleave(rep, dim=1)

            y = _dense_attention(q_sdpa, k_sdpa, v_sdpa, attn_mask=mask1, input_pos=input_pos)

            with CUDATimer(cuda_timing) as t_wo:
                y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
                out = self.wo(y)
            p["wo"] += t_wo.ms()
            return out

        # ----------------------------
        # Decode (STATIC-SHAPE: operate on the full cache [:, :, :maxlen, :] + allowed mask)
        # ----------------------------
        # maxlen = k_cache.shape[2] is the cache's static T dim, fixed at setup_caches() to
        # prompt_len + max_new_tokens -> COMPILE-TIME CONSTANT, so every shape below is static
        # and no host sync (prefill_len.item()) is needed. We score the FULL cache and rely on
        # the `allowed` mask (index > pos.max() -> -inf in the scorer / dropped by the kernel)
        # so the selected token set + attention output are identical to the eager `:T` path.
        maxlen = k_cache.shape[2]

        pos = input_pos.view(-1)
        # seq_len_t = pos.max()+1 = true filled length, kept as an on-device scalar tensor (no
        # .item()) so it drives index ARITHMETIC (window start) without changing any SHAPE.
        seq_len_t = pos.max().to(torch.int32) + 1
        # The `allowed` mask is exactly `t < seq_len` -- the SAME value for every head and
        # every one of the 32 layers -- yet it was materialized ([1,H,maxlen] bool, a 4.6 MB
        # expand().contiguous() at 140K) and consumed once per layer per decode step. The
        # fused scorer / fused list-assembly take the seq_len scalar instead, so skip building
        # it entirely unless a LEGACY path is selected.
        if _NEEDS_ALLOWED_MASK:
            allowed = torch.arange(maxlen, device=pos.device) <= pos.max()
            allowed_bht = allowed.view(1, 1, maxlen).expand(bsz, self.n_head, maxlen).contiguous()
        else:
            allowed_bht = None

        with CUDATimer(cuda_timing) as t_relayout:
            # PER-KV-HEAD scorer: do NOT repeat_interleave the (identical-per-GQA-group) key
            # data. Keep buckets/v_norm at the kv-head count (Hkv) and let the scorer kernel
            # index by kv_head = h // rep (exactly mirroring the backend's
            # cur_kv_head = cur_head // gqa_group_size). q_probs/allowed stay PER-QUERY-HEAD, so
            # the score sum_l q_probs[h,l,bucket_l(j)]*||v_j|| is bit-identical to the repeated
            # path: repeat_interleave(rep,dim=1) maps out-head h -> in-head h//rep (pure copy).
            #
            # RANK-1: k_hard is now STORED natively as [B,Hkv,L,maxlen] (KVCache.update writes one
            # column/step), so the per-token full-buffer permute(0,1,3,2).contiguous() relayout
            # (~210MB round-trip @128K, the #1 T-scaling DRAM op) is DELETED — read it directly.
            k_hard_bhlt = self.kv_cache.k_hard  # [B,Hkv,L,maxlen] (native; no relayout)
            v_norm_bht = self.kv_cache.v_norm   # [B,Hkv,maxlen]
            if self._force_repeat and rep != 1:
                # ABLATION (default OFF): reproduce the pre-opt 4x repeat (kernel sees Hkv==H,
                # rep=1). Bit-identical selection/output (T6), only slower.
                k_hard_bhlt = k_hard_bhlt.repeat_interleave(rep, dim=1)  # [B,H,L,maxlen]
                v_norm_bht = v_norm_bht.repeat_interleave(rep, dim=1)    # [B,H,maxlen]

            q_bhd = q_sdpa[:, :, 0, :].contiguous()  # [B,H,D]
        p["kv_relayout"] += t_relayout.ms()

        # sink/window/M clamped against the STATIC maxlen. In the decode regime the filled length
        # is the (large) prompt length, so maxlen >> sink+window+M and these Python-int counts are
        # IDENTICAL to the old min(., T) clamp -> same static list width, same selected set.
        sink = int(getattr(self.config, "sink_size", 120))
        window = int(getattr(self.config, "window_size", 120))
        M = int(self.heavy_const)
        sink = max(0, min(sink, maxlen))
        window = max(0, min(window, maxlen))
        M = max(0, min(M, maxlen))

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
                seq_len_t=seq_len_t,
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

        # Backend wants [B,Hl,maxlen,D] (full static cache; sparse_list -1 padding + allowed
        # mask exclude the unfilled tail inside the kernel).
        k_backend = k_cache.contiguous()
        v_backend = v_cache.contiguous()

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

        # FAIR-DENSE LAYOUT. With layout="bthd" the cache is stored [B,T,Hkv,D] — FA's native
        # decode layout — so `k_bthd`/`v_bthd` below are the cache buffers THEMSELVES and the
        # per-step full-cache transpose().contiguous() disappears entirely. `k`/`v` keep their
        # historical [B,Hkv,T,D] meaning via a zero-copy transpose VIEW so the prefill /
        # SDPA-control path below is untouched (its own .contiguous() materializes exactly the
        # same bytes it always did). With layout="bhtd" (SOCKET, or SOCKET_KV_LAYOUT=bhtd)
        # nothing changes at all.
        k_bthd_cache = v_bthd_cache = None
        if self.kv_cache is not None:
            k_cache, v_cache = self.kv_cache.update(input_pos, k, v)
            if self.kv_cache.layout == "bthd":
                k_bthd_cache, v_bthd_cache = k_cache, v_cache      # [B,T,Hkv,D] (native)
                k = k_cache.transpose(1, 2)                        # [B,Hkv,T,D] VIEW (no copy)
                v = v_cache.transpose(1, 2)
            else:
                k = k_cache                                        # [B,Hkv,T,D]
                v = v_cache

        q = q.transpose(1, 2)  # [B,Hq,S,D]

        # Dense DECODE (seqlen==1) via the FA2/FA3 with_kvcache custom op: GQA-native
        # (un-repeated KV, Hkv heads), STATIC shape, fullgraph + CUDA-graph safe (no
        # .item()/dynamic slice; cache_seqlens is on-device). Only when a flash backend is
        # selected; the SDPA control (USE_FLASHATTN3=0) and all prefill (seqlen>1) fall
        # through to the repeat_interleave + _dense_attention path below (unchanged).
        _use_flash = (
            os.getenv("USE_FLASHATTN3", "1") == "1"
            and _flash_attn_func is not None
            and _flash_attn_with_kvcache is not None
            and q.is_cuda
            and q.dtype in (torch.float16, torch.bfloat16)
        )
        if seqlen == 1 and _use_flash and input_pos is not None and input_pos.numel() > 0:
            q_bshd = q.transpose(1, 2).contiguous()        # [B,1,Hq,D]
            if k_bthd_cache is not None:
                # ZERO-COPY: the cache is already [B,maxlen,Hkv,D]. This is the whole point of
                # layout="bthd" — no 2 x 587 MB relayout per layer per step at 140K.
                k_bthd = k_bthd_cache
                v_bthd = v_bthd_cache
            else:
                k_bthd = k.transpose(1, 2).contiguous()    # [B,maxlen,Hkv,D] (un-repeated; GQA-native)
                v_bthd = v.transpose(1, 2).contiguous()
            cache_seqlens = (input_pos.max().to(torch.int32) + 1).reshape(1).expand(bsz).contiguous()
            out_bshd = torch.ops.socket.flash_dense_decode(q_bshd, k_bthd, v_bthd, cache_seqlens)
            if not torch.compiler.is_compiling():
                _assert_backend("flash")
            y = out_bshd.transpose(1, 2).contiguous().view(bsz, 1, self.dim)
            return self.wo(y)

        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)  # [B,H,T,D]
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)  # [B,H,T,D]

        y = _dense_attention(q, k, v, attn_mask=mask, input_pos=input_pos)
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
