"""Fused SMALL-TENSOR kernels for the SOCKET decode step.

Motivation (measured, compiled run, 140K / P10L10 / 50x, one layer of the CUDA-graph
replay -- see /scratch/sj157/socket_speed/fair/compiled-budget/results/p10l10_140k_trace.json):

    3.264 us  cutlass bf16 gemm   q @ planes            (soft_hash)
    1.088 us  triton_poi          tanh / div            (soft_hash)
    3.616 us  cutlass bf16 gemm   tanh @ protos_T       (soft_hash)
    2.016 us  triton_per          softmax(logits/tau)   (soft_hash)
    2.944 us  cutlass bf16 gemm   k @ planes            (hard_hash_keys)
    1.120 us  triton_per          pack_bits             (hard_hash_keys)
    1.184 us  triton_per          v_norm + index_put    (v_norm)
    ------
   15.232 us/layer over 7 kernels, moving < 1.6 MB of data.  At 4 TB/s the DATA in these
   ops costs ~0.4 us; the rest is per-kernel launch/tail latency.  Two Triton kernels
   replace all seven.

Everything here is DECODE-ONLY (seqlen == 1).  The prefill path is untouched: the eager
einsum chain is still used for seqlen > 1, so prefill numerics are bit-identical.
"""

import os

import torch
import triton
import triton.language as tl

try:
    from torch.library import triton_op, wrap_triton
    _HAS_TRITON_OP = True
except Exception:  # pragma: no cover
    _HAS_TRITON_OP = False

    def wrap_triton(k):  # type: ignore
        return k

# `tanh` / `exp` must come from the SAME source Inductor uses for the baseline kernels,
# otherwise the fused result is not bit-comparable: Inductor emits libdevice calls.
#
# IMPORTANT: import the FUNCTIONS unqualified, not the module. Inductor rebuilds a
# user-defined Triton kernel's source in a compile subprocess
# (wrapper.py::user_defined_triton_kernel_transitive_closure_source_code) and only
# re-emits an import for a referenced global when `symbol.__module__` starts with
# "triton". A *module* object has no __module__, so `libdevice.tanh(x)` inside the
# kernel fails with NameError('libdevice is not defined') under torch.compile even
# though it works in eager. A function does have __module__, so `_libdev_tanh(x)`
# gets `from triton.language.extra.libdevice import tanh as _libdev_tanh` emitted.
try:
    from triton.language.extra.libdevice import tanh as _libdev_tanh
    from triton.language.extra.libdevice import exp as _libdev_exp
except Exception:  # pragma: no cover - older triton layout
    from triton.language.extra.cuda.libdevice import tanh as _libdev_tanh
    from triton.language.extra.cuda.libdevice import exp as _libdev_exp


# ---------------------------------------------------------------------------
# 1) FUSED soft_hash:  q [B,H,D] -> q_probs [B,H,L,R]
#
# Reproduces, per (b,h,l):
#     proj[k]   = bf16( sum_d q[d] * planes[l,k,d] )          (bf16 gemm, fp32 accum)
#     t[k]      = bf16( bf16(tanh(fp32(proj[k]))) / temp )    (torch.tanh then /temp)
#     logit[r]  = bf16( sum_k t[k] * protos_T[k,r] )          (bf16 gemm, fp32 accum)
#     out[r]    = bf16( softmax_r( fp32(logit[r]) / tau ) )
#
# ROUND_TRIP controls whether the (logit/tau) intermediate is rounded back to bf16 before
# the softmax.  EAGER torch DOES round-trip (`logits / tau` materializes a bf16 tensor);
# INDUCTOR does NOT (it keeps the fp32 intermediate across the fused div+softmax).  That is
# why the eager and compiled baselines are not bitwise equal to each other either.
# ROUND_TRIP=1 (default) reproduces the eager semantics.
# ---------------------------------------------------------------------------
@triton.jit
def _fwd_kernel_soft_hash_qprobs(
    Q,            # [B,H,D]      bf16/fp16/fp32
    Planes,       # [L,K,D]      same dtype as Q
    ProtosT,      # [K,R]        same dtype as Q
    Out,          # [B,H,L,R]    bf16 (written)
    inv_temp,     # fp32 scalar = 1/max(sqrt(D), 1e-6)
    tau,          # fp32 scalar
    H: tl.constexpr, D: tl.constexpr, L: tl.constexpr, K: tl.constexpr, R: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_R: tl.constexpr,
    ROUND_TRIP: tl.constexpr,
):
    l = tl.program_id(0)
    bh = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    offs_r = tl.arange(0, BLOCK_R)
    mask_r = offs_r < R

    q = tl.load(Q + bh * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)

    acc = tl.zeros([BLOCK_R], dtype=tl.float32)
    for k in tl.static_range(K):
        pl = tl.load(Planes + (l * K + k) * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
        pr = tl.sum(q * pl, axis=0)                        # fp32 dot over D
        pr = pr.to(tl.bfloat16).to(tl.float32)             # the bf16 gemm rounds its output
        th = _libdev_tanh(pr).to(tl.bfloat16).to(tl.float32)   # torch.tanh on a bf16 tensor
        th = (th * inv_temp).to(tl.bfloat16).to(tl.float32)
        pt = tl.load(ProtosT + k * R + offs_r, mask=mask_r, other=0.0).to(tl.float32)
        acc += th * pt

    logit = acc.to(tl.bfloat16).to(tl.float32)             # bf16 gemm output rounding
    x = logit / tau
    if ROUND_TRIP:
        x = x.to(tl.bfloat16).to(tl.float32)
    x = tl.where(mask_r, x, float("-inf"))
    m = tl.max(x, axis=0)
    e = _libdev_exp(x - m)
    e = tl.where(mask_r, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s
    tl.store(Out + (bh * L + l) * R + offs_r, out.to(Out.dtype.element_ty), mask=mask_r)


def _soft_hash_qprobs_impl(
    q_bhd: torch.Tensor,      # [B,H,D]
    planes: torch.Tensor,     # [L,K,D]
    protos_T: torch.Tensor,   # [K,R]
    out: torch.Tensor,        # [B,H,L,R]  (written)
    inv_temp: float,
    tau: float,
) -> None:
    B, H, D = q_bhd.shape
    L, K, D2 = planes.shape
    K2, R = protos_T.shape
    assert D2 == D and K2 == K
    BLOCK_D = triton.next_power_of_2(D)
    BLOCK_R = triton.next_power_of_2(R)
    warps = int(os.environ.get("SOCKET_SH_WARPS", "4"))
    grid = (L, B * H)
    wrap_triton(_fwd_kernel_soft_hash_qprobs)[grid](
        q_bhd, planes, protos_T, out, inv_temp, tau,
        H=H, D=D, L=L, K=K, R=R, BLOCK_D=BLOCK_D, BLOCK_R=BLOCK_R,
        # DEFAULT 1: measured (scripts/gate_numerics.py, job 217527) --
        #   ROUND_TRIP=1 -> BITWISE equal to the EAGER reference chain (0-9 of 327680
        #                   elements differ, all by exactly 1 bf16 ulp, from the softmax
        #                   reduction order alone)
        #   ROUND_TRIP=0 -> 425-467/327680 vs the INDUCTOR-compiled chain
        # For calibration, Inductor's own compiled chain differs from eager on
        # 627/327680 elements, so ROUND_TRIP=1 is MORE faithful to model.py's
        # soft_hash() than the baseline compiled kernels are.
        ROUND_TRIP=(os.environ.get("SOCKET_SH_ROUNDTRIP", "1") == "1"),
        num_warps=warps, num_stages=1,
    )


if _HAS_TRITON_OP:
    soft_hash_qprobs_op = triton_op(
        "socket::soft_hash_qprobs", _soft_hash_qprobs_impl, mutates_args={"out"}
    )
else:
    soft_hash_qprobs_op = torch.no_grad()(_soft_hash_qprobs_impl)


@torch.no_grad()
def fused_soft_hash(q_bhd, planes, protos_T, temp, tau, out_dtype):
    """Decode-only replacement for Attention.soft_hash. Returns [B,H,L,R].

    SOCKET_SH_IMPL selects v1 (K chained 1-D reductions), v2 (2 reductions) or
    v2alloc (v2 as a non-mutating custom_op that allocates its own output).
    """
    B, H, _ = q_bhd.shape
    L = planes.shape[0]
    R = protos_T.shape[1]
    it, tt = 1.0 / max(temp, 1e-6), float(tau)
    impl = os.environ.get("SOCKET_SH_IMPL", "v2")
    if impl == "v2alloc":
        return torch.ops.socket.soft_hash_qprobs_alloc(q_bhd, planes, protos_T, it, tt)
    out = torch.empty((B, H, L, R), device=q_bhd.device, dtype=out_dtype)
    if impl == "v2":
        soft_hash_qprobs_v2_op(q_bhd, planes, protos_T, out, it, tt)
    else:
        soft_hash_qprobs_op(q_bhd, planes, protos_T, out, it, tt)
    return out


@torch.no_grad()
def fused_kv_meta(k_bhd, v_bhd, planes, pos, k_hard, v_norm, k_cache, v_cache, write_kv):
    """SOCKET_KVMETA_IMPL selects v1 (8 CTAs, L*K chained reductions) or v2."""
    if os.environ.get("SOCKET_KVMETA_IMPL", "v2") == "v2":
        kv_meta_v2_op(k_bhd, v_bhd, planes, pos, k_hard, v_norm, k_cache, v_cache, write_kv)
    else:
        kv_meta_op(k_bhd, v_bhd, planes, pos, k_hard, v_norm, k_cache, v_cache, write_kv)


# ---------------------------------------------------------------------------
# 2) FUSED per-step KV metadata: one new token's k/v -> the k_hard and v_norm cache
#    columns, written IN PLACE at column `pos`.
#
# Replaces (decode only):
#     proj   = einsum("bshd,lkd->bshlk", k, planes)          # bf16 gemm
#     bits   = proj >= 0
#     buckets= pack_bits(bits)                               # big-endian over K
#     v_n    = vector_norm(v.float(), 2, -1).to(fp16)
#     k_hard[:,:,:,pos] = buckets.permute(...)
#     v_norm[:,:,pos]   = v_n.permute(...)
#
# One program per (b, hkv).  L*K fp32 dots of length D, then a D-long sum of squares.
# ---------------------------------------------------------------------------
@triton.jit
def _fwd_kernel_kv_meta(
    Kv,            # [B,Hkv,D]   bf16 (the new token's key, post-RoPE)
    Vv,            # [B,Hkv,D]   bf16 (the new token's value)
    Planes,        # [L,K,D]     bf16
    PosPtr,        # int32/int64 scalar on device = the cache column to write
    KHard,         # [B,Hkv,L,T] int16 (written at [...,pos])
    VNorm,         # [B,Hkv,T]   fp16  (written at [...,pos])
    KCache,        # [B,Hkv,T,D] bf16  (written at [...,pos,:]) when WRITE_KV
    VCache,        # [B,Hkv,T,D] bf16  (written at [...,pos,:]) when WRITE_KV
    HKV: tl.constexpr, D: tl.constexpr, L: tl.constexpr, K: tl.constexpr, T: tl.constexpr,
    BLOCK_D: tl.constexpr, WRITE_KV: tl.constexpr,
):
    bh = tl.program_id(0)          # b * HKV + hkv
    pos = tl.load(PosPtr).to(tl.int32)

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    kraw = tl.load(Kv + bh * D + offs_d, mask=mask_d, other=0.0)
    vraw = tl.load(Vv + bh * D + offs_d, mask=mask_d, other=0.0)
    kvec = kraw.to(tl.float32)

    if WRITE_KV:
        # Also absorbs KVCache.update's two index_put writes for the new column (measured
        # 1.12 us/layer as one fused Inductor kernel). Pure data movement of 2 x D bf16
        # elements -> byte-identical to the index_put it replaces.
        tl.store(KCache + (bh * T + pos) * D + offs_d, kraw, mask=mask_d)
        tl.store(VCache + (bh * T + pos) * D + offs_d, vraw, mask=mask_d)

    for l in tl.static_range(L):
        bkt = 0     # scalar accumulator (a [1]-shaped block cannot be stored to a scalar ptr)
        for k in tl.static_range(K):
            pl = tl.load(Planes + (l * K + k) * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
            pr = tl.sum(kvec * pl, axis=0).to(tl.bfloat16).to(tl.float32)
            bit = tl.where(pr >= 0.0, 1, 0).to(tl.int32)
            bkt = bkt * 2 + bit                      # big-endian, matches pack_bits
        tl.store(KHard + (bh * L + l) * T + pos, bkt.to(tl.int16))

    vvec = vraw.to(tl.float32)
    vn = tl.sqrt(tl.sum(vvec * vvec, axis=0))
    tl.store(VNorm + bh * T + pos, vn.to(VNorm.dtype.element_ty))


def _kv_meta_impl(
    k_bhd: torch.Tensor,     # [B,Hkv,D]
    v_bhd: torch.Tensor,     # [B,Hkv,D]
    planes: torch.Tensor,    # [L,K,D]
    pos: torch.Tensor,       # int scalar on device
    k_hard: torch.Tensor,    # [B,Hkv,L,T] int16 (mutated)
    v_norm: torch.Tensor,    # [B,Hkv,T] fp16   (mutated)
    k_cache: torch.Tensor,   # [B,Hkv,T,D] bf16 (mutated iff write_kv)
    v_cache: torch.Tensor,   # [B,Hkv,T,D] bf16 (mutated iff write_kv)
    write_kv: bool,
) -> None:
    B, HKV, D = k_bhd.shape
    L, K, _ = planes.shape
    T = k_hard.shape[-1]
    BLOCK_D = triton.next_power_of_2(D)
    wrap_triton(_fwd_kernel_kv_meta)[(B * HKV,)](
        k_bhd, v_bhd, planes, pos, k_hard, v_norm, k_cache, v_cache,
        HKV=HKV, D=D, L=L, K=K, T=T, BLOCK_D=BLOCK_D, WRITE_KV=write_kv,
        num_warps=int(os.environ.get("SOCKET_KVMETA_WARPS", "4")), num_stages=1,
    )


if _HAS_TRITON_OP:
    kv_meta_op = triton_op(
        "socket::kv_meta", _kv_meta_impl,
        mutates_args={"k_hard", "v_norm", "k_cache", "v_cache"},
    )
else:
    kv_meta_op = torch.no_grad()(_kv_meta_impl)


# ===========================================================================
# V2 KERNELS.  V1 (above) was MEASURED in the compiled CUDA-graph replay and both
# kernels LOSE badly:
#     _fwd_kernel_kv_meta          48.86 us/layer  (replaces 5.24 us of ATen work)
#     _fwd_kernel_soft_hash_qprobs  6.43 us/layer  (replaces 9.98 us -- a small win)
# The kv_meta disaster is structural: V1 uses ONE PROGRAM PER (b,kv-head) = 8 CTAs on
# a 132-SM GPU, and inside it a chain of L*K = 100 SEQUENTIALLY DEPENDENT block-wide
# reductions over BLOCK_D=128 (each `tl.sum(kvec*pl, axis=0)` is a shared-memory
# reduction with two barriers).  100 chained barrier round-trips at 8-way parallelism
# is ~0.49 us each.
#
# V2 fixes the structure:
#   * one program per (b, kv-head, l)  -> 80 (L=10) / 400 (L=50) CTAs
#   * the K projections for that l are ONE 2-D reduction over a [BLOCK_K, BLOCK_D]
#     tile (axis=1) instead of K chained 1-D reductions
#   * the bucket pack is one more reduction over BLOCK_K
#   * the v_norm / k-cache / v-cache column writes happen only in the l == 0 program
# Same for soft_hash: 2 reductions per program instead of K+2.
# ===========================================================================


def _sh_warps(K, R):
    """num_warps for the v2 soft_hash kernel.  Measured, CUDA-graph-timed, T=143411
    (scripts/micro.py, job 217588):
        L=10 K=10 R=1024 : warps 1/2/4/8/16 -> 4.46 / 3.51 / 3.10 / 6.54 / 17.49 us
        L=50 K=8  R=256  : warps 1/2/4/8/16 -> 3.17 / 7.07 / 18.50 / 33.80 / 37.42 us
    The optimum tracks the SIZE of the [BLOCK_K, BLOCK_R] tile, not L: a 16x1024 tile
    wants 4 warps, a 8x256 tile wants 1.  (For reference the baseline 4-kernel chain is
    15.94 us at L=10 and 11.99 us at L=50 by the same measurement.)
    """
    bk = triton.next_power_of_2(K)
    br = triton.next_power_of_2(R)
    default = 4 if bk * br >= 8192 else 1
    return int(os.environ.get("SOCKET_SH_WARPS", default))


@triton.jit
def _fwd_kernel_soft_hash_qprobs_v2(
    Q, Planes, ProtosT, Out, inv_temp, tau,
    H: tl.constexpr, D: tl.constexpr, L: tl.constexpr, K: tl.constexpr, R: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_R: tl.constexpr,
    ROUND_TRIP: tl.constexpr,
):
    l = tl.program_id(0)
    bh = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < K
    offs_r = tl.arange(0, BLOCK_R)
    mask_r = offs_r < R

    q = tl.load(Q + bh * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)   # [BD]
    pl = tl.load(Planes + (l * K + offs_k[:, None]) * D + offs_d[None, :],
                 mask=mask_k[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    pr = tl.sum(q[None, :] * pl, axis=1)                     # [BK]  ONE reduction
    pr = pr.to(tl.bfloat16).to(tl.float32)
    th = _libdev_tanh(pr).to(tl.bfloat16).to(tl.float32)
    th = (th * inv_temp).to(tl.bfloat16).to(tl.float32)
    th = tl.where(mask_k, th, 0.0)

    pt = tl.load(ProtosT + offs_k[:, None] * R + offs_r[None, :],
                 mask=mask_k[:, None] & mask_r[None, :], other=0.0).to(tl.float32)
    acc = tl.sum(th[:, None] * pt, axis=0)                   # [BR]  ONE reduction

    logit = acc.to(tl.bfloat16).to(tl.float32)
    x = logit / tau
    if ROUND_TRIP:
        x = x.to(tl.bfloat16).to(tl.float32)
    x = tl.where(mask_r, x, float("-inf"))
    m = tl.max(x, axis=0)
    e = _libdev_exp(x - m)
    e = tl.where(mask_r, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s
    tl.store(Out + (bh * L + l) * R + offs_r, out.to(Out.dtype.element_ty), mask=mask_r)


def _soft_hash_qprobs_v2_impl(
    q_bhd: torch.Tensor, planes: torch.Tensor, protos_T: torch.Tensor,
    out: torch.Tensor, inv_temp: float, tau: float,
) -> None:
    B, H, D = q_bhd.shape
    L, K, _ = planes.shape
    R = protos_T.shape[1]
    wrap_triton(_fwd_kernel_soft_hash_qprobs_v2)[(L, B * H)](
        q_bhd, planes, protos_T, out, inv_temp, tau,
        H=H, D=D, L=L, K=K, R=R,
        BLOCK_D=triton.next_power_of_2(D),
        BLOCK_K=triton.next_power_of_2(K),
        BLOCK_R=triton.next_power_of_2(R),
        ROUND_TRIP=(os.environ.get("SOCKET_SH_ROUNDTRIP", "1") == "1"),
        num_warps=_sh_warps(K, R), num_stages=1,
    )


@triton.jit
def _fwd_kernel_kv_meta_v2(
    Kv, Vv, Planes, PosPtr, KHard, VNorm, KCache, VCache,
    HKV: tl.constexpr, D: tl.constexpr, L: tl.constexpr, K: tl.constexpr, T: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_K: tl.constexpr, WRITE_KV: tl.constexpr,
):
    l = tl.program_id(0)
    bh = tl.program_id(1)
    pos = tl.load(PosPtr).to(tl.int32)

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < K

    kraw = tl.load(Kv + bh * D + offs_d, mask=mask_d, other=0.0)
    pl = tl.load(Planes + (l * K + offs_k[:, None]) * D + offs_d[None, :],
                 mask=mask_k[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    pr = tl.sum(kraw.to(tl.float32)[None, :] * pl, axis=1)        # [BK] ONE reduction
    pr = pr.to(tl.bfloat16).to(tl.float32)
    bit = tl.where(pr >= 0.0, 1, 0).to(tl.int32)
    sh = tl.where(mask_k, K - 1 - offs_k, 0)
    w = (1 << sh).to(tl.int32)                                    # big-endian pack_bits
    bkt = tl.sum(tl.where(mask_k, bit * w, 0), axis=0)            # scalar
    tl.store(KHard + (bh * L + l) * T + pos, bkt.to(tl.int16))

    # v_norm + (optionally) the k/v cache columns are l-independent: do them once.
    if l == 0:
        vraw = tl.load(Vv + bh * D + offs_d, mask=mask_d, other=0.0)
        vv = vraw.to(tl.float32)
        tl.store(VNorm + bh * T + pos,
                 tl.sqrt(tl.sum(vv * vv, axis=0)).to(VNorm.dtype.element_ty))
        if WRITE_KV:
            tl.store(KCache + (bh * T + pos) * D + offs_d, kraw, mask=mask_d)
            tl.store(VCache + (bh * T + pos) * D + offs_d, vraw, mask=mask_d)


def _kv_meta_v2_impl(
    k_bhd: torch.Tensor, v_bhd: torch.Tensor, planes: torch.Tensor, pos: torch.Tensor,
    k_hard: torch.Tensor, v_norm: torch.Tensor,
    k_cache: torch.Tensor, v_cache: torch.Tensor, write_kv: bool,
) -> None:
    B, HKV, D = k_bhd.shape
    L, K, _ = planes.shape
    T = k_hard.shape[-1]
    wrap_triton(_fwd_kernel_kv_meta_v2)[(L, B * HKV)](
        k_bhd, v_bhd, planes, pos, k_hard, v_norm, k_cache, v_cache,
        HKV=HKV, D=D, L=L, K=K, T=T,
        BLOCK_D=triton.next_power_of_2(D), BLOCK_K=triton.next_power_of_2(K),
        WRITE_KV=write_kv,
        # measured 1.71 / 1.77 / 1.75 / 1.80 us at warps 1/2/4/8 (L=10) and
        # 1.75 / 1.86 / 1.98 / 1.90 (L=50) -- flat, so take the cheapest occupancy.
        num_warps=int(os.environ.get("SOCKET_KVMETA_WARPS", "1")), num_stages=1,
    )


if _HAS_TRITON_OP:
    soft_hash_qprobs_v2_op = triton_op(
        "socket::soft_hash_qprobs_v2", _soft_hash_qprobs_v2_impl, mutates_args={"out"})
    kv_meta_v2_op = triton_op(
        "socket::kv_meta_v2", _kv_meta_v2_impl,
        mutates_args={"k_hard", "v_norm", "k_cache", "v_cache"})
else:
    soft_hash_qprobs_v2_op = torch.no_grad()(_soft_hash_qprobs_v2_impl)
    kv_meta_v2_op = torch.no_grad()(_kv_meta_v2_impl)


# ---- NON-MUTATING form of the v2 soft_hash (allocates its own output).  V1's mutating
# triton_op form coincided with a +32 us/layer re-fusion of the BACKBONE ffn/wo GEMVs
# (see the report); this variant exists to test whether the mutation + externally
# allocated buffer is what perturbs Inductor's scheduling.
@torch.library.custom_op("socket::soft_hash_qprobs_alloc", mutates_args=())
def _soft_hash_qprobs_alloc(q_bhd: torch.Tensor, planes: torch.Tensor,
                            protos_T: torch.Tensor, inv_temp: float,
                            tau: float) -> torch.Tensor:
    B, H, D = q_bhd.shape
    L, K, _ = planes.shape
    R = protos_T.shape[1]
    out = torch.empty((B, H, L, R), device=q_bhd.device, dtype=q_bhd.dtype)
    _fwd_kernel_soft_hash_qprobs_v2[(L, B * H)](
        q_bhd, planes, protos_T, out, inv_temp, tau,
        H=H, D=D, L=L, K=K, R=R,
        BLOCK_D=triton.next_power_of_2(D),
        BLOCK_K=triton.next_power_of_2(K),
        BLOCK_R=triton.next_power_of_2(R),
        ROUND_TRIP=(os.environ.get("SOCKET_SH_ROUNDTRIP", "1") == "1"),
        num_warps=_sh_warps(K, R), num_stages=1,
    )
    return out


@_soft_hash_qprobs_alloc.register_fake
def _soft_hash_qprobs_alloc_fake(q_bhd, planes, protos_T, inv_temp, tau):
    B, H, _ = q_bhd.shape
    L = planes.shape[0]
    R = protos_T.shape[1]
    return q_bhd.new_empty((B, H, L, R))


_SH_IMPL = os.environ.get("SOCKET_SH_IMPL", "v2")
_KVMETA_IMPL = os.environ.get("SOCKET_KVMETA_IMPL", "v2")
