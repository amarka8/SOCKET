import math
import os

import torch
import triton
import triton.language as tl

# torch.library.triton_op + wrap_triton make a Triton-launching python fn an OPAQUE
# custom op to Dynamo: it does NOT trace into the launch (so fullgraph=True does not
# graph-break on the kernel), and it declares the mutated outputs so functionalization
# and cudagraph_trees capture the kernel correctly. Available torch>=2.6 / present in
# the 2.8 build used here. We keep a runtime fallback so the module still imports on a
# torch without these symbols (eager path is then used directly, unchanged numerics).
try:
    from torch.library import triton_op, wrap_triton
    _HAS_TRITON_OP = True
except Exception:  # pragma: no cover - very old torch
    _HAS_TRITON_OP = False

    def wrap_triton(k):  # type: ignore
        return k

_SOFT_HASH_EXT = None

def _get_soft_hash_ext():
    global _SOFT_HASH_EXT
    if _SOFT_HASH_EXT is None:
        from kernels.soft_hash_collision_loader import load_soft_hash_collision
        _SOFT_HASH_EXT = load_soft_hash_collision(3)
    return _SOFT_HASH_EXT


# ---------------------------------------------------------------------------
# EXACT TOP-M BY 3-DIGIT RADIX THRESHOLD SELECT  (kernels/radix_select.cu)
#
# Replaces aten::topk on the [B,H,maxlen] score array. The radix select streams the score
# array, materialises no intermediate buffer, and resolves the exact top-M threshold in
# seven kernels over four streaming passes.
#
# EXACTNESS. Scores are fp16; each is promoted exactly to fp32 on load and compared through
# key(f) = (bits(f) & 0x80000000) ? ~bits(f) : (bits(f) | 0x80000000), strictly monotone over
# ALL floats, so the -inf written into unfilled cache columns sorts last exactly as it does
# under topk. d = 0xFFFFFFFF - key is split MSB-first into 11/11/10 bits; digit 3 has UNIT
# resolution, so three digits resolve the threshold EXACTLY for any input distribution -- no
# adaptive shifts, no data-dependent iteration count, no host sync, fixed launch structure
# (hence CUDA-graph capturable). The selected SCORE MULTISET always equals aten::topk's on
# the same fp16 scores. The INDEX set need not: the threshold score can be tied, and which
# tied token a top-k implementation keeps is arbitrary in topk as well.
#
# TUNING. Ordinary runtime arguments to an opaque custom_op, NOT tl.constexpr values, so
# hardcoding them here is a maintenance choice rather than a cache-correctness requirement:
#   NB   = 4 T-chunks per (b,h) -> B*H*4 = 128 blocks, about one per SM. Larger NB only adds
#          private-histogram flush traffic (NB * B*H * 2048 * 4 bytes per level).
#   THR  = 1024 threads. Bytes in flight = blocks*THR*UNROLL*4, so THR governs how much of
#          the streaming passes' latency is hidden. The most sensitive knob here.
#   STHR = 512 for the three tiny scan kernels (2048 bins / 512 threads = 4 bins per thread).
#   HMODE= 1, a plain per-lane shared atomicAdd (MATCH.ANY.U32 warp aggregation is the
#          expensive alternative; shared-atomic conflict replays are cheap on Hopper).
#   STAGES = 127, i.e. all seven real stages and not the read-only bandwidth probe (bit 128).
#   DET  = 0. The emit reserves output slots with atomics, so WHICH tied-at-threshold tokens
#          are kept (and the slot order) depends on arrival order; the selected score
#          multiset is exact either way. DET=1 makes both a pure function of the grid at the
#          cost of one extra streaming pass -- set it when a bit-reproducible token stream is
#          needed (e.g. the eager-vs-compiled regression gate).
# ---------------------------------------------------------------------------
_RS_NB, _RS_THR, _RS_STHR, _RS_HMODE, _RS_STAGES = 4, 1024, 512, 1, 127
# DET is an ordinary runtime argument to the extension (never a tl.constexpr), so an env
# override is cache-safe; the equivalence gates set it to 1 to make two runs comparable.
_RS_DET = 1 if os.environ.get("SOCKET_RS_DET", "0") == "1" else 0

_RADIX_EXT = None


def _get_radix_ext():
    global _RADIX_EXT
    if _RADIX_EXT is None:
        from kernels.radix_select_loader import load_radix_select
        _RADIX_EXT = load_radix_select()
    return _RADIX_EXT


# The workspace is allocated with torch.empty INSIDE the op body. It needs no zero-init (every
# block writes its whole private histogram slice, and the ctrl block is fully written by
# rs_scan1) and it is dead the moment the op returns, so under cudagraph_trees it is an
# ordinary pool intermediate. It is deliberately NOT a cached module-level tensor: a tensor
# cached across calls is allocated inside the CUDA-graph pool on the recording call, and
# reusing it on a later replay raises "accessing tensor output of CUDAGraphs that has been
# overwritten by a subsequent run".
@torch.library.custom_op("socket::radix_topm", mutates_args=())
def _radix_topm(scores: torch.Tensor, M: int) -> torch.Tensor:
    ext = _get_radix_ext()
    B, H, _ = scores.shape
    out = torch.empty((B, H, M), device=scores.device, dtype=torch.int32)
    ws = torch.empty(int(ext.workspace_ints(_RS_NB, B * H)),
                     device=scores.device, dtype=torch.int32)
    ext.radix_select(scores, out, ws, M, _RS_NB, _RS_THR, _RS_STHR, _RS_STAGES,
                     _RS_HMODE, _RS_DET)
    return out


@_radix_topm.register_fake
def _radix_topm_fake(scores: torch.Tensor, M: int) -> torch.Tensor:
    B, H, _ = scores.shape
    return scores.new_empty((B, H, M), dtype=torch.int32)


# ---------------------------------------------------------------------------
# HEAD-PACKED scorer (socket::soft_hash_score_packed, kernels/rowpack_scorer.cu).
# Bit-equal to socket::soft_hash_score -- same operand bits, same fp32 add order, same fp16
# rounding and -inf tail; only the q_probs transport differs: the GQA group's four query
# heads are interleaved in the last dim of a packed [B,Hkv,L,R,4] copy, so one aligned 8-byte
# load serves the whole group where the unpacked kernel issues four independent gathers into
# four separate rows. Used for the large-R geometries, where the unpacked gather's L2-sector
# amplification and L1 overflow dominate the scorer; at small R the rows are cache-resident
# and the shipped Triton kernel is used unchanged.
# ---------------------------------------------------------------------------
_ROWPACK_EXT = None


def _get_rowpack_ext():
    global _ROWPACK_EXT
    if _ROWPACK_EXT is None:
        from kernels.rowpack_scorer_loader import load_rowpack_scorer
        _ROWPACK_EXT = load_rowpack_scorer()
    return _ROWPACK_EXT


@torch.library.custom_op("socket::soft_hash_score_packed", mutates_args={"out"})
def _soft_hash_score_packed(qp_pack: torch.Tensor, key_buckets: torch.Tensor,
                            v_norm: torch.Tensor, seq_len_t: torch.Tensor,
                            out: torch.Tensor) -> None:
    _get_rowpack_ext().soft_hash_score_packed(qp_pack, key_buckets, v_norm, seq_len_t, out)


@_soft_hash_score_packed.register_fake
def _soft_hash_score_packed_fake(qp_pack: torch.Tensor, key_buckets: torch.Tensor,
                                 v_norm: torch.Tensor, seq_len_t: torch.Tensor,
                                 out: torch.Tensor) -> None:
    return None


soft_hash_score_packed_op = _soft_hash_score_packed

# The packed transport pays a small permute per call and wins only where the row working set
# defeats the caches, so it is enabled by R (a static shape-derived int, stable per run).
_SCORER_PACKED_MIN_R = 1024


def soft_hash_score_auto(q_probs, key_buckets, v_norm_bht, seq_len_t, scores):
    """Run the soft-hash scorer, choosing the transport by geometry.

    Writes fp16 scores into `scores` [B,H,T]. Both transports produce bit-equal output;
    the packed CUDA kernel is used for large-R GQA geometries and the Triton kernel
    otherwise. This is the ONE scorer entry point both the GPT-FAST decode path and the
    HF eval path call, so the two pipelines always run the same kernels.
    """
    assert scores.dtype == torch.float16, (
        f"scores must be fp16 (got {scores.dtype}); every scorer arm rounds its fp32 "
        f"accumulator once on store, and a wider buffer would hide that rounding")
    B, H, L, R = q_probs.shape
    Hkv = key_buckets.shape[1]
    rep = H // Hkv
    if (R >= _SCORER_PACKED_MIN_R and rep == 4
            and q_probs.dtype in (torch.float16, torch.bfloat16)):
        qp_pack = (q_probs.contiguous().view(B, Hkv, 4, L, R)
                   .permute(0, 1, 3, 4, 2).contiguous())
        soft_hash_score_packed_op(qp_pack, key_buckets.contiguous(),
                                  v_norm_bht.contiguous(), seq_len_t.reshape(()), scores)
    else:
        soft_hash_score_op(q_probs.contiguous(), key_buckets.contiguous(),
                           v_norm_bht.contiguous(), seq_len_t.reshape(()), scores)
    return scores


# ---------------------------------------------------------------------------
# torch.compile registration for the custom CUDA scorer.
#
# ext.soft_hash_collision is a *pybind11* C++ function (PYBIND11_MODULE in
# soft_hash_collision_loader.py), NOT a registered Torch operator. Under
# torch.compile(fullgraph=True) Dynamo cannot trace into a pybind callable, so
# it would graph-break (a hard error with fullgraph=True). We wrap it in a
# torch.library.custom_op so Dynamo treats it as a single opaque-but-known op.
#
# Semantics of the underlying kernel (soft_hash_collision_kernel_3):
#   - reads q_probs[B,H,1,L,R] f32, key_buckets[B,H,L,T_k] i16,
#     allowed_ext[B,H,1,T_k] bool, v_hist[B,H,1,T_k] f32
#   - writes ONLY a freshly torch::zeros({B,H,1,T_k}, q.options()) output
#   - does NOT write to / alias any input
#   => mutates_args=() ; output is a new f32 tensor [B,H,1,T_k].
# The wrapper body calls the identical pybind kernel, so the runtime path is
# bit-identical to the eager path.
#
# NOT ON THE DECODE PATH ANY MORE: build_sparse_list_decode uses the Triton scorer
# (socket::soft_hash_score) below, which is bit-identical and strictly cheaper. This op is
# kept as the byte-identical REFERENCE implementation the equivalence gates compare against.
# ---------------------------------------------------------------------------
@torch.library.custom_op("socket::soft_hash_collision", mutates_args=())
def _soft_hash_collision_op(
    q_probs: torch.Tensor,      # [B,H,1,L,R] f32, contiguous (PER-QUERY-HEAD H)
    key_buckets: torch.Tensor,  # [B,Hkv,L,T_k] i16, contiguous (PER-KV-HEAD Hkv; rep=H//Hkv recovered in-kernel)
    allowed_ext: torch.Tensor,  # [B,H,1,T_k] bool, contiguous (PER-QUERY-HEAD H)
    v_hist: torch.Tensor,       # [B,Hkv,1,T_k] f32, contiguous (PER-KV-HEAD Hkv)
) -> torch.Tensor:
    ext = _get_soft_hash_ext()
    return ext.soft_hash_collision(q_probs, key_buckets, allowed_ext, v_hist)


@_soft_hash_collision_op.register_fake
def _soft_hash_collision_fake(
    q_probs: torch.Tensor,
    key_buckets: torch.Tensor,
    allowed_ext: torch.Tensor,
    v_hist: torch.Tensor,
) -> torch.Tensor:
    # out = torch::zeros({B, H, 1, T_k}, q_probs.options())
    # Output is PER-QUERY-HEAD: B,H come from q_probs (still H query heads). T_k is the LAST
    # dim of key_buckets (unchanged by the per-kv-head opt; only dim 1 shrank H->Hkv).
    B = q_probs.shape[0]
    H = q_probs.shape[1]
    T_k = key_buckets.shape[3]
    return q_probs.new_empty((B, H, 1, T_k), dtype=torch.float32)


# ---------------------------------------------------------------------------
# TRITON SOFT-HASH SCORER  (socket::soft_hash_score)  -- the scorer the decode path uses.
#
# The per-token score is
#   acc = 0; for l in 0..L-1: acc += float(q_probs[b,h,l, buckets[b,kv,l,t]]); out = fp16(acc*float(v[t]))
# fp32 adds in ascending l (L is a constexpr so the loop is unrolled but never reassociated),
# one fp32 multiply, then ONE round-to-nearest-even conversion to the fp16 output. q_probs
# and v_norm are read in their native dtypes; float(bf16) and float(fp16) are exact, so the
# accumulated value is bit-identical to the fp32 reference kernel's (soft_hash_collision
# above) and the stored score equals that reference rounded once to fp16.
#
# What it removes, with no arithmetic change:
#   * the [B,H,maxlen] bool `allowed` tensor. The mask is exactly `t < seq_len`, a scalar
#     compare; the CUDA scorer needed it materialized (expand().contiguous()) and then read
#     back once per layer per decode step, for a value identical across all heads and layers.
#   * the full-buffer v_norm.float() cast.
#   * the fp32 [B,H,1,L,R] q_probs copy.
# It is also a genuine triton_op, so torch.compile captures it as a CUDA-graph node without
# the stream hazard the raw <<<>>> launch had.
# ---------------------------------------------------------------------------
@triton.jit
def _fwd_kernel_soft_hash_score(
    QProbs,          # [B,H,L,R]        (bf16 / fp16 / fp32)
    KeyBuckets,      # [B,Hkv,L,T]      int16
    VNorm,           # [B,Hkv,T]        (bf16 / fp16 / fp32)
    SeqLenPtr,       # int32 scalar on device
    Out,             # [B,H,T]          fp16 (written)
    H: tl.constexpr, HKV: tl.constexpr, L: tl.constexpr, R: tl.constexpr,
    T: tl.constexpr, BLOCK_T: tl.constexpr,
):
    tile = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)

    rep = H // HKV
    kv = h // rep

    offs_t = tile * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = offs_t < T

    seq_len = tl.load(SeqLenPtr).to(tl.int32)
    keep = offs_t < seq_len

    kb_base = (b * HKV + kv) * L * T
    qp_base = (b * H + h) * L * R

    acc = tl.zeros([BLOCK_T], dtype=tl.float32)
    # BUCKET RANGE IS A CALLER INVARIANT. Both loads below are masked on `mask_t` (t < T), not
    # on `keep` (t < seq_len), so a column past seq_len is still dereferenced -- the -inf write
    # at the end of the kernel discards its value but does not prevent its gather. Every column
    # of KeyBuckets in [0, T) must therefore hold a bucket in [0, R), or the second load reads
    # outside QProbs. Both callers satisfy this by zero-filling at allocation: model.py's
    # KVCache and the HF path's padded prefill buffer.
    for l in tl.static_range(L):
        bkt = tl.load(KeyBuckets + kb_base + l * T + offs_t, mask=mask_t, other=0).to(tl.int32)
        qv = tl.load(QProbs + qp_base + l * R + bkt, mask=mask_t, other=0.0)
        acc += qv.to(tl.float32)

    v = tl.load(VNorm + (b * HKV + kv) * T + offs_t, mask=mask_t, other=0.0).to(tl.float32)
    out = acc * v
    # Unfilled cache columns (t >= seq_len) get -inf: `allowed` IS `t < seq_len` (model.py
    # builds it as arange(maxlen) <= pos.max(), with seq_len_t = pos.max()+1).
    out = tl.where(keep, out, -float("inf"))
    tl.store(Out + (b * H + h) * T + offs_t, out.to(tl.float16), mask=mask_t)


# ---------------------------------------------------------------------------
# GQA-GROUP BUCKET SHARING.
#
# key_buckets is stored PER-KV-HEAD ([B,Hkv,L,T]) but the kernel above runs PER-QUERY-HEAD
# (grid axis 1 = H), so the rep = H // Hkv query heads of one GQA group each stream the SAME
# bucket row: the scorer ISSUES rep times the unique bucket bytes.
#
# This kernel puts ONE block on each (kv head, tile) and serves the whole group from it: the
# bucket vector is loaded ONCE per l and reused by four explicitly-unrolled 1D
# gathers/accumulators. No 2D broadcast, so the address arithmetic per gather is identical to
# the per-query-head kernel (a [rep, BLOCK_T] 2D accumulator would add broadcast address
# arithmetic). Bucket load instructions, and the L1/L2 read requests they generate, drop
# exactly rep-fold.
#
# BIT-IDENTICAL to the per-query-head kernel. The score is
#     out[h,t] = (sum_l q_probs[h,l,bucket[kv(h),l,t]]) * ||v_t||
# accumulated in fp32 in increasing l. Reordering the launch grid changes nothing, and sharing
# one LOAD between several accumulators changes no arithmetic: every add still happens in the
# same order with the same operands.
#
# rep == 4 for Llama-3.1-8B (H=32, Hkv=8). Other reps fall back to the per-query-head kernel,
# which is also what the rep == 1 callers in test_socket_compile_equiv.py get.
# ---------------------------------------------------------------------------
@triton.jit
def _fwd_kernel_soft_hash_score_gqa4(
    QProbs,          # [B,H,L,R]        (bf16 / fp16 / fp32)
    KeyBuckets,      # [B,Hkv,L,T]      int16
    VNorm,           # [B,Hkv,T]        (bf16 / fp16 / fp32)
    SeqLenPtr,       # int32 scalar on device
    Out,             # [B,H,T]          fp16 (written)
    H: tl.constexpr, HKV: tl.constexpr, L: tl.constexpr, R: tl.constexpr,
    T: tl.constexpr, BLOCK_T: tl.constexpr,
):
    tile = tl.program_id(0)
    kv = tl.program_id(1)
    b = tl.program_id(2)

    offs_t = tile * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = offs_t < T
    seq_len = tl.load(SeqLenPtr).to(tl.int32)
    keep = offs_t < seq_len

    kb_base = (b * HKV + kv) * L * T
    qp = QProbs + (b * H + kv * 4) * L * R
    LR: tl.constexpr = L * R

    a0 = tl.zeros([BLOCK_T], dtype=tl.float32)
    a1 = tl.zeros([BLOCK_T], dtype=tl.float32)
    a2 = tl.zeros([BLOCK_T], dtype=tl.float32)
    a3 = tl.zeros([BLOCK_T], dtype=tl.float32)
    # Same caller invariant as the per-query-head kernel: every column of KeyBuckets in [0, T)
    # must hold a bucket in [0, R), because the four gathers below are masked on t < T rather
    # than on t < seq_len. Here it matters four times per l instead of once.
    for l in tl.static_range(L):
        # ONE bucket load per l; the four gathers reuse it from registers.
        bkt = tl.load(KeyBuckets + kb_base + l * T + offs_t, mask=mask_t, other=0).to(tl.int32)
        off = l * R + bkt
        a0 += tl.load(qp + off, mask=mask_t, other=0.0).to(tl.float32)
        a1 += tl.load(qp + LR + off, mask=mask_t, other=0.0).to(tl.float32)
        a2 += tl.load(qp + 2 * LR + off, mask=mask_t, other=0.0).to(tl.float32)
        a3 += tl.load(qp + 3 * LR + off, mask=mask_t, other=0.0).to(tl.float32)

    v = tl.load(VNorm + (b * HKV + kv) * T + offs_t, mask=mask_t, other=0.0).to(tl.float32)
    ob = Out + (b * H + kv * 4) * T + offs_t
    tl.store(ob, tl.where(keep, a0 * v, -float("inf")).to(tl.float16), mask=mask_t)
    tl.store(ob + T, tl.where(keep, a1 * v, -float("inf")).to(tl.float16), mask=mask_t)
    tl.store(ob + 2 * T, tl.where(keep, a2 * v, -float("inf")).to(tl.float16), mask=mask_t)
    tl.store(ob + 3 * T, tl.where(keep, a3 * v, -float("inf")).to(tl.float16), mask=mask_t)


# LAUNCH CONFIG: HARDCODED LITERALS, deliberately not env-readable. These become tl.constexpr
# values inside a torch.library.triton_op body, which Dynamo evaluates at TRACE time; the
# resulting kernel enters inductor's FX-graph cache key only as an INDEX into
# kernel_side_table, so the value itself is NOT part of the key. Two runs that share a
# TORCHINDUCTOR_CACHE_DIR and differ only in such a value silently execute the FIRST one's
# kernel, so an A/B over such a value can silently compare a kernel against itself. Keeping
# every constexpr a literal removes the hazard at the root; benchmark harnesses should STILL
# give each cell a fresh TORCHINDUCTOR_CACHE_DIR and TRITON_CACHE_DIR.
#
# The optimum is a constant ~4 ELEMENTS PER THREAD (BLOCK_T / (32*num_warps) == 4): the L-deep
# gather loop makes anything wider spill registers. Every (BLOCK_T, num_warps) variant is
# bitwise-equal to the CUDA scorer, so the choice is purely a performance one.
# The GQA-shared kernel keeps the same ~4 elements per thread per accumulator, so one block
# covering rep query heads wants rep times the warps.
_SCORER_BLOCK_T_SMALL_L, _SCORER_WARPS_SMALL_L = 128, 1
_SCORER_BLOCK_T_LARGE_L, _SCORER_WARPS_LARGE_L = 256, 2
_SCORER_GQA4_BLOCK_T_SMALL_L, _SCORER_GQA4_WARPS_SMALL_L = 128, 2
_SCORER_GQA4_BLOCK_T_LARGE_L, _SCORER_GQA4_WARPS_LARGE_L = 256, 4


def _scorer_launch_cfg(L: int, gqa4: bool):
    if gqa4:
        if L <= 16:
            return _SCORER_GQA4_BLOCK_T_SMALL_L, _SCORER_GQA4_WARPS_SMALL_L
        return _SCORER_GQA4_BLOCK_T_LARGE_L, _SCORER_GQA4_WARPS_LARGE_L
    if L <= 16:
        return _SCORER_BLOCK_T_SMALL_L, _SCORER_WARPS_SMALL_L
    return _SCORER_BLOCK_T_LARGE_L, _SCORER_WARPS_LARGE_L


def _soft_hash_score_impl(
    q_probs: torch.Tensor,      # [B,H,L,R]
    key_buckets: torch.Tensor,  # [B,Hkv,L,T] int16
    v_norm: torch.Tensor,       # [B,Hkv,T]
    seq_len_t: torch.Tensor,    # int32 scalar on device
    out: torch.Tensor,          # [B,H,T] fp16 (written)
) -> None:
    B, H, L, R = q_probs.shape
    HKV, T = key_buckets.shape[1], key_buckets.shape[3]
    gqa4 = (H // HKV) == 4
    BLOCK_T, warps = _scorer_launch_cfg(L, gqa4)
    if gqa4:
        wrap_triton(_fwd_kernel_soft_hash_score_gqa4)[(triton.cdiv(T, BLOCK_T), HKV, B)](
            q_probs, key_buckets, v_norm, seq_len_t, out,
            H=H, HKV=HKV, L=L, R=R, T=T, BLOCK_T=BLOCK_T,
            num_warps=warps, num_stages=2,
        )
    else:
        wrap_triton(_fwd_kernel_soft_hash_score)[(triton.cdiv(T, BLOCK_T), H, B)](
            q_probs, key_buckets, v_norm, seq_len_t, out,
            H=H, HKV=HKV, L=L, R=R, T=T, BLOCK_T=BLOCK_T,
            num_warps=warps, num_stages=2,
        )


if _HAS_TRITON_OP:
    # mutates_args declares `out` so functionalization and cudagraph_trees handle it
    # correctly; Dynamo does not trace into the body (no graph break under fullgraph=True).
    soft_hash_score_op = triton_op(
        "socket::soft_hash_score", _soft_hash_score_impl, mutates_args={"out"}
    )
else:
    soft_hash_score_op = torch.no_grad()(_soft_hash_score_impl)


# ---------------------------------------------------------------------------
# FUSED INDEX-LIST ASSEMBLY  (socket::build_list)
#
# Everything after the top-M select used to be ~24 separate tiny CUDA kernels on [B,H,M] and
# [W] tensors:
#   valid=(>=0)&(<maxlen); ok=gather(allowed, clamp(heavy)); masked_fill;
#   arange(sink); clamp(seq_len-window); arange(window)+off; clamp; cat; view/expand;
#   gather(allowed, base); masked_fill; win_start=clamp; in_sink; in_window; or; masked_fill;
#   cat([base, heavy]); contiguous
# One kernel emits sparse_list directly, and it needs NO `allowed` tensor: the mask is exactly
# `t < seq_len`, a scalar compare, so materializing a [B,H,maxlen] bool and reading it back
# once per layer per decode step was pure waste. This is the last consumer of `allowed`, so
# model.py stops building it entirely.
#
# WHICH PARAMETERS ARE constexpr, AND WHY. SINK, WINDOW, BLOCK and DEDUP are constants for a
# whole run in both stacks, so they stay constexpr and the untaken DEDUP arm folds away. MAXLEN
# and W are ORDINARY RUNTIME ARGUMENTS: GPT-FAST decodes against a static cache and holds both
# fixed, but the HuggingFace eval path in pipeline/train_quest/modeling grows its list width
# with the heavy budget, and a constexpr would force a fresh Triton compile per decode step
# there. Each is read exactly once -- W bounds the slot range, MAXLEN clamps the window index --
# so two runtime compares replace two folded immediates in a kernel that touches B*H*W int32
# slots. There was a third, `M`, which the body never read at all; it is gone.
#
# The emitted list is BIT-IDENTICAL by construction -- it is pure integer index arithmetic,
# reproducing exactly the semantics of the op chain it replaces:
#   slot in [0, sink)             -> index = slot                              (sink)
#   slot in [sink, sink+window)   -> index = min(win_start + j, maxlen-1)       (window)
#   slot in [sink+window, W)      -> index = heavy[m]                          (top-M)
#   win_start = max(seq_len - window, sink)
#   every index is dropped to -1 unless 0 <= index < seq_len         (allowed gating)
#   a heavy index is ALSO dropped to -1 if it lies in [0,sink) or [win_start,seq_len) (dedup,
#   because the no-dedup online softmax would otherwise weight that token twice)
# Ordering of the two heavy masks is irrelevant: a -1 fails both `>= 0` predicates.
#
# The dedup step is a constexpr switch (DEDUP) rather than unconditional, because the HF
# path in pipeline/train_quest/modeling reuses this kernel and has to be able to reproduce
# BOTH of its own arms -- its torch op chain concatenates the lists without deduplicating
# unless SOCKET_DEDUP=1. GPT-FAST itself always passes True, which folds to exactly the code
# this kernel emitted before the switch existed.
# ---------------------------------------------------------------------------
@triton.jit
def _fwd_kernel_build_list(
    Heavy,            # [B,H,M] int32   (selected indices; order within the row is irrelevant)
    SeqLenPtr,        # int32 scalar on device (= pos.max()+1, the true filled length)
    Out,              # [B,H,W] int32   (written)
    stride_hb, stride_hh,
    stride_ob, stride_oh,
    MAXLEN, W,        # RUNTIME, not constexpr -- see the note above
    SINK: tl.constexpr, WINDOW: tl.constexpr, BLOCK: tl.constexpr,
    DEDUP: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    blk = tl.program_id(2)

    seq_len = tl.load(SeqLenPtr).to(tl.int32)
    win_start = tl.maximum(seq_len - WINDOW, SINK)

    slot = blk * BLOCK + tl.arange(0, BLOCK)
    in_range = slot < W

    is_sink = slot < SINK
    is_win = (slot >= SINK) & (slot < SINK + WINDOW)
    is_heavy = slot >= (SINK + WINDOW)

    m = slot - (SINK + WINDOW)
    hv = tl.load(Heavy + b * stride_hb + h * stride_hh + m,
                 mask=is_heavy & in_range, other=-1).to(tl.int32)

    idx_win = tl.minimum(win_start + (slot - SINK), MAXLEN - 1)
    idx = tl.where(is_sink, slot.to(tl.int32),
                   tl.where(is_win, idx_win.to(tl.int32), hv))

    # allowed gating: 0 <= idx < seq_len
    keep = (idx >= 0) & (idx < seq_len)
    # heavy dedup against sink U window. DEDUP is a constexpr, so the arm not taken is
    # folded away at compile time and the DEDUP=True path emits exactly the code it did
    # before this switch existed.
    if DEDUP:
        dup = is_heavy & (((idx >= 0) & (idx < SINK)) | ((idx >= win_start) & (idx < seq_len)))
        idx = tl.where(keep & (dup == 0), idx, -1)
    else:
        idx = tl.where(keep, idx, -1)

    tl.store(Out + b * stride_ob + h * stride_oh + slot, idx, mask=in_range)


# BLOCK is a literal for the same inductor-cache-key reason as the scorer's launch config.
_BUILD_LIST_BLOCK = 256


def _build_list_impl(
    heavy_idx: torch.Tensor,   # [B,H,M] int32
    seq_len_t: torch.Tensor,   # int32 scalar on device
    out: torch.Tensor,         # [B,H,W] int32 (written)
    sink: int, window: int, maxlen: int, dedup: bool,
) -> None:
    B, H = heavy_idx.shape[0], heavy_idx.shape[1]
    W = out.shape[-1]
    BLOCK = _BUILD_LIST_BLOCK
    grid = (B, H, triton.cdiv(W, BLOCK))
    wrap_triton(_fwd_kernel_build_list)[grid](
        heavy_idx, seq_len_t, out,
        heavy_idx.stride(0), heavy_idx.stride(1),
        out.stride(0), out.stride(1),
        int(maxlen), int(W),
        SINK=sink, WINDOW=window, BLOCK=BLOCK,
        DEDUP=dedup,
        num_warps=4, num_stages=1,
    )


if _HAS_TRITON_OP:
    build_list_op = triton_op("socket::build_list", _build_list_impl, mutates_args={"out"})
else:
    build_list_op = torch.no_grad()(_build_list_impl)


@torch.no_grad()
def build_sparse_list_decode(
    q_probs: torch.Tensor,         # [B,H,L,R] fp16        (PER-QUERY-HEAD H)
    k_hard_bhlt: torch.Tensor,     # [B,Hkv,L,maxlen] int16/int32 (PER-KV-HEAD Hkv)
    v_norm_bht: torch.Tensor,      # [B,Hkv,maxlen] fp16/bf16     (PER-KV-HEAD Hkv)
    allowed_bht,                   # IGNORED, kept for call compatibility (see below)
    sink: int,
    window: int,
    M: int,
    seq_len_t: torch.Tensor = None,  # on-device int32 scalar = true filled length (pos.max()+1)
    KC: int = 8,
    BLOCK_N: int = 512,
    num_warps: int = 8,
    num_stages: int = 2,
):
    """Static-shape decode index builder.

    All COLUMN counts (maxlen, sink, window, M_eff, list width) are Python ints derived from
    the STATIC cache dimension `maxlen = k_hard_bhlt.shape[-1]`, so every tensor shape here is
    a compile-time constant (torch.compile / CUDA-graph capturable, no host sync).

    The true (growing) sequence length is carried ONLY as the on-device scalar `seq_len_t`
    (pos.max()+1) and is used purely for index ARITHMETIC (the sliding-window start), never for
    a shape. Unfilled cache columns (index >= seq_len_t) are masked to -inf by the `allowed`
    mask `t < seq_len_t`, applied inside the scorer (which writes -inf there) and inside the
    index-list kernel, so they can never be selected and never enter attention. The result is
    identical to slicing the cache to the true filled length.

    `allowed_bht` is IGNORED. It used to carry a [B,H,maxlen] bool that was always exactly
    `arange(maxlen) < seq_len_t` -- the same value for every head and every layer -- and both
    consumers now derive it from the seq_len scalar instead. The parameter is kept so existing
    callers (GPT-FAST/test_socket_compile_equiv.py) still work unchanged.
    """
    assert q_probs.is_cuda and k_hard_bhlt.is_cuda and v_norm_bht.is_cuda
    assert q_probs.dtype in (torch.float16, torch.bfloat16, torch.float32)

    B, H, L, R = q_probs.shape
    Bk, Hkv, L2, maxlen = k_hard_bhlt.shape   # maxlen is STATIC (compile-time constant)
    assert L2 == L
    # PER-KV-HEAD: key data carries Hkv <= H kv heads; q_probs carries H query heads.
    # The scorer kernel reads key_buckets/v_hist at kv head h//rep (rep=H//Hkv) and q_probs/
    # allowed/out at query head h -> bit-identical to repeat_interleave(rep,dim=1) of the key
    # data. These are static-shape (Python int) asserts -> constant-folded under torch.compile.
    assert Bk == B, "batch mismatch between q_probs and key_buckets"
    assert H % Hkv == 0, f"n_head ({H}) must be divisible by n_kv_heads ({Hkv})"
    assert v_norm_bht.shape[1] == Hkv, "v_norm must be per-kv-head [B,Hkv,maxlen]"

    device = q_probs.device

    if seq_len_t is None:
        # Eager fallback: assume the cache is fully filled, true length == maxlen.
        seq_len_t = torch.tensor(maxlen, device=device, dtype=torch.int32)
    else:
        seq_len_t = seq_len_t.to(device=device, dtype=torch.int32).reshape(())

    # NOTE: KC/BLOCK_N/num_warps/num_stages kept for API compatibility.
    M_eff = min(M, maxlen)  # static
    if M_eff > 0:
        key_buckets = k_hard_bhlt                                # [B,Hkv,L,maxlen] (per kv head)
        # int16 REQUIRED: R=1024 buckets span 0..1023 (and even R=256 spans 0..255), neither
        # of which fits signed int8.
        if key_buckets.dtype != torch.int16:
            key_buckets = key_buckets.to(torch.int16)

        # Scores are fp16: fp32-accumulated by every scorer arm and rounded once on store.
        # The select is exact over these fp16 values (see the radix comment above).
        scores = torch.empty((B, H, maxlen), device=device, dtype=torch.float16)
        soft_hash_score_auto(q_probs, key_buckets, v_norm_bht, seq_len_t, scores)

        # Exact radix threshold select over the fp16 scores; see the module-level comment
        # for the exactness argument.
        heavy_idx = torch.ops.socket.radix_topm(scores, M_eff)
    else:
        heavy_idx = torch.empty((B, H, 0), device=device, dtype=torch.int32)

    # ---- FUSED LIST ASSEMBLY (socket::build_list) --------------------------------------
    # Replaces the ~24 tiny kernels this function used to launch. See the kernel comment for
    # the semantics, which are reproduced exactly.
    sink = max(0, min(sink, maxlen))
    window = max(0, min(window, maxlen))
    if sink == 0 and window == 0:
        # Degenerate config (never reached by the model: sink_size == window_size == 120).
        # A window of 1 makes the kernel emit exactly the single index the op chain used as
        # its fallback base, clamp(seq_len-1, min=0).
        window = 1
    W = sink + window + M_eff

    sparse_list = torch.empty((B, H, W), device=device, dtype=torch.int32)
    if M_eff <= 0:
        heavy_in = torch.full((B, H, 1), -1, device=device, dtype=torch.int32)
    else:
        heavy_in = heavy_idx
    build_list_op(heavy_in, seq_len_t.reshape(()), sparse_list, sink, window, maxlen, True)
    # sparse_len is W for every (b,h) by construction (the list is rectangular, padded with
    # -1). Left as a per-call torch.full: it is a 128-byte fill that inductor CSEs across the
    # 32 layers of one graph, and caching a tensor across calls is unsafe under
    # cudagraph_trees (a buffer allocated in the graph pool cannot be reused on a later
    # replay).
    sparse_len = torch.full((B, H), W, device=device, dtype=torch.int32)
    return sparse_list, sparse_len


# =========================================================
# BACKEND (Stage 1+2): your flash-decode style sparse attention
# =========================================================

@triton.jit
def _fwd_kernel_sparse_decode_stage1(
    Q, K, V, sm_scale,
    Sparse_List, Sparse_Len,
    Mid_O, Mid_O_LogExpSum,
    stride_sparse_b, stride_sparse_h,
    stride_qbs, stride_qh, stride_qd,
    stride_kbb, stride_kh, stride_ks,
    stride_vbb, stride_vh, stride_vs,
    stride_splen_b, stride_splen_h,
    stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od,
    stride_mid_o_eb, stride_mid_o_eh, stride_mid_o_es,
    gqa_group_size: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NAN_GUARD: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    seq_start_block = tl.program_id(2)
    cur_kv_head = cur_head // gqa_group_size

    offs_d = tl.arange(0, BLOCK_DMODEL)

    cur_seq_len_ptr = Sparse_Len + cur_batch * stride_splen_b + cur_head * stride_splen_h
    cur_seq_len = tl.load(cur_seq_len_ptr)

    cur_block_start = seq_start_block * BLOCK_SEQ
    cur_block_end = tl.minimum(cur_seq_len, cur_block_start + BLOCK_SEQ)

    sparse_ptr_base = Sparse_List + cur_batch * stride_sparse_b + cur_head * stride_sparse_h

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d
    q = tl.load(Q + off_q)

    sum_exp = 0.0
    max_logic = -float("inf")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    block_n_size = (
        tl.where(cur_block_end - cur_block_start <= 0, 0,
                 cur_block_end - cur_block_start + BLOCK_N - 1) // BLOCK_N
    )

    offs_n = cur_block_start + tl.arange(0, BLOCK_N)

    for start_n in range(0, block_n_size, 1):
        offs_n_new = start_n * BLOCK_N + offs_n

        token_idx = tl.load(
            sparse_ptr_base + offs_n_new,
            mask=offs_n_new < cur_seq_len,
            other=-1,
        )
        # Only attend real, in-range tokens. Slots past the list (offs_n_new>=cur_seq_len)
        # or holding a padding index (token_idx<0) must NOT be dereferenced — gather them
        # with mask=False (Triton skips the load) and give them -inf so they drop out of the
        # softmax. Output-preserving: valid tokens are unchanged; invalid slots never count.
        valid_tok = (offs_n_new < cur_seq_len) & (token_idx >= 0)
        safe_idx = tl.where(valid_tok, token_idx, 0)

        base_ptr = cur_batch * stride_kbb + cur_kv_head * stride_kh
        off_k = base_ptr + safe_idx[:, None] * stride_ks + offs_d[None, :]
        k = tl.load(K + off_k, mask=valid_tok[:, None], other=0.0)
        v = tl.load(V + off_k, mask=valid_tok[:, None], other=0.0)

        att_value = tl.sum(q[None, :] * k, 1)
        att_value *= sm_scale
        # Pin fp32 throughout: a Python float("-inf") in tl.where promotes the result to
        # fp64 under Inductor's stricter typing (eager Triton tolerates it), which then makes
        # the loop-carried sum_exp/acc fp64 -> "loop-carried type stays inconsistent" compile
        # error under torch.compile. Casting to fp32 is numerically identical to the eager
        # online-softmax (which already accumulates in fp32).
        att_value = tl.where(valid_tok, att_value, float("-inf")).to(tl.float32)

        cur_max_logic = tl.max(att_value, axis=0)
        new_max_logic = tl.maximum(cur_max_logic, max_logic).to(tl.float32)

        # EMPTY-CHUNK GUARD. If every slot in this BLOCK_N chunk is padding (-1) or past
        # cur_seq_len, att_value is all -inf; with max_logic also still -inf (no valid token
        # seen yet in this BLOCK_SEQ block) new_max_logic is -inf and BOTH exp() calls below
        # evaluate exp(-inf - -inf) = exp(nan) = nan, poisoning acc/sum_exp for the whole
        # block. Stage2 then merges the NaN partial and it reaches the logits.
        # Substituting 0.0 for the max while the running max is still -inf is EXACT: every
        # exp() argument is then -inf - 0 = -inf -> 0.0, so the chunk contributes nothing and
        # max_logic legitimately stays -inf. When new_max_logic is finite the expression is
        # unchanged, so this is bit-identical on every non-degenerate chunk.
        _safe_max = new_max_logic
        if NAN_GUARD:
            _safe_max = tl.where(new_max_logic == float("-inf"), 0.0, new_max_logic)
        _safe_max = _safe_max.to(tl.float32)

        exp_logic = tl.exp(att_value - _safe_max).to(tl.float32)
        logic_scale = tl.exp(max_logic - _safe_max).to(tl.float32)

        acc = (acc * logic_scale + tl.sum(exp_logic[:, None] * v, axis=0)).to(tl.float32)
        sum_exp = (sum_exp * logic_scale + tl.sum(exp_logic, axis=0)).to(tl.float32)
        max_logic = new_max_logic

    need_store = tl.where(block_n_size == 0, 0, 1)
    for _ in range(0, need_store, 1):
        off_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + seq_start_block * stride_mid_os
            + offs_d
        )
        off_mid_o_logexpsum = (
            cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh + seq_start_block
        )
        # A block whose slots are ALL padding yields sum_exp == 0; 0/0 = nan and log(0) =
        # -inf. Emit a zero partial with logexpsum = -inf so stage2's merge discards it (its
        # weight is exp(-inf - m) = 0). Unchanged whenever sum_exp > 0.
        if NAN_GUARD:
            _empty = sum_exp == 0.0
            _safe_sum = tl.where(_empty, 1.0, sum_exp)
        else:
            _empty = sum_exp < 0.0        # never true: reproduces the unguarded store
            _safe_sum = sum_exp
        tl.store(Mid_O + off_mid_o, acc / _safe_sum)
        tl.store(Mid_O_LogExpSum + off_mid_o_logexpsum,
                 tl.where(_empty, float("-inf"), max_logic + tl.log(_safe_sum)))


@triton.jit
def _fwd_kernel_sparse_decode_stage2(
    Sparse_Len,
    Mid_O,
    Mid_O_LogExpSum,
    O,
    stride_splen_b, stride_splen_h,
    stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od,
    stride_mid_o_eb, stride_mid_o_eh, stride_mid_o_es,
    stride_obs, stride_oh, stride_od,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    NAN_GUARD: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_DMODEL)

    cur_seq_len_ptr = Sparse_Len + cur_batch * stride_splen_b + cur_head * stride_splen_h
    cur_seq_len = tl.load(cur_seq_len_ptr)

    block_n_size = (tl.where(cur_seq_len <= 0, 0, cur_seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ)

    sum_exp = 0.0
    max_logic = -float("inf")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh

    for block_seq_n in range(0, block_n_size, 1):
        tv = tl.load(Mid_O + offs_v + block_seq_n * stride_mid_os)
        tlogic = tl.load(Mid_O_LogExpSum + offs_logic + block_seq_n)

        # fp32-pinned (same Inductor type-consistency reason as stage1); numerically identical.
        new_max_logic = tl.maximum(tlogic, max_logic).to(tl.float32)
        # Same empty guard as stage1: a partial from an all-padding block carries
        # logexpsum = -inf, and merging it while the running max is still -inf would
        # evaluate exp(-inf - -inf) = nan. Bit-identical whenever new_max_logic is finite.
        _safe_max2 = new_max_logic
        if NAN_GUARD:
            _safe_max2 = tl.where(new_max_logic == float("-inf"), 0.0, new_max_logic)
        _safe_max2 = _safe_max2.to(tl.float32)
        old_scale = tl.exp(max_logic - _safe_max2).to(tl.float32)
        exp_logic = tl.exp(tlogic - _safe_max2).to(tl.float32)
        acc = (acc * old_scale + exp_logic * tv).to(tl.float32)
        sum_exp = (sum_exp * old_scale + exp_logic).to(tl.float32)
        max_logic = new_max_logic

    off_o = cur_batch * stride_obs + cur_head * stride_oh + offs_d
    # Defensive: sum_exp == 0 only if EVERY slot for this (b,h) was padding, which cannot
    # happen while sink > 0, but 0/0 would silently produce NaN logits if it ever did.
    _den = sum_exp
    if NAN_GUARD:
        _den = tl.where(sum_exp == 0.0, 1.0, sum_exp)
    tl.store(O + off_o, acc / _den)


# STAGE-1 LAUNCH CONFIG. Literals, for the inductor-cache-key reason spelled out at the
# scorer's config above: these are tl.constexpr values consumed inside a triton_op body, so an
# env read here is baked at trace time and is invisible to the FX-graph cache key, so an A/B
# over such a value can silently compare a kernel against itself.
#
# BLOCK_N is the inner gather width. At a narrow width each warp owns only a few gathered
# 256 B rows, per-thread memory-level parallelism is a couple of outstanding loads, and the
# resulting many chunks serialise behind the online-softmax loop dependency. Widening it
# raises both. Block count is not the lever here; vector width is. BLOCK_SEQ stays 256
# because it sets stage2's merge trip count (W / BLOCK_SEQ).
#
# INVARIANT: BLOCK_SEQ % BLOCK_N == 0 whenever the list spans MORE THAN ONE partition,
# asserted below. The inner mask is `offs_n_new < cur_seq_len`, not `< cur_block_end`, so a
# non-dividing BLOCK_N lets a partition read past its own end into the next partition's range
# and double-count those tokens in the online softmax. A single-partition call
# (block_seq >= list width, which GPT-FAST/test_socket_compile_equiv.py's T9 fixture makes)
# has no next partition, so it is harmless and is allowed.
# THE ONE ENVIRONMENT-DERIVED constexpr IN THIS FILE, and it exists only so the regression
# test can show the guard is load-bearing: SOCKET_NAN_GUARD=0 compiles the empty-partition
# guards out and the NaN they suppress comes back. Read once, here, into a plain Python bool.
#
# It carries the hazard described above -- a value baked at trace time is not part of
# inductor's FX cache key -- so it must never be varied inside a benchmark, and never with a
# shared TORCHINDUCTOR_CACHE_DIR. Nothing but GPT-FAST/tests/test_empty_chunk_nan.py sets it.
_NAN_GUARD = os.environ.get("SOCKET_NAN_GUARD", "1") != "0"

_STAGE1_BLOCK_N = 128
# Widening BLOCK_N is bitwise-neutral; num_warps changes the block reduction order, so
# stage1's output differs from the num_warps=4 kernel by reassociation only (same token set,
# same online-softmax semantics).
_STAGE1_NUM_WARPS = 8


def _sparse_decode_stage1_impl(
    q: torch.Tensor,            # [B,H,D]
    k: torch.Tensor,            # [B,Kv,S,D]
    v: torch.Tensor,            # [B,Kv,S,D]
    sparse_list: torch.Tensor,  # [B,H,Ktotal]
    sparse_len: torch.Tensor,   # [B,H]
    mid_out: torch.Tensor,         # [B,H,block_seq_num,D] fp32
    mid_out_logsumexp: torch.Tensor,# [B,H,block_seq_num] fp32
    block_seq: int,
    max_len_in_batch: int,
) -> None:
    BLOCK_N = _STAGE1_BLOCK_N
    assert block_seq % BLOCK_N == 0 or max_len_in_batch <= block_seq, (
        f"BLOCK_SEQ ({block_seq}) must be divisible by BLOCK_N ({BLOCK_N}) when the list "
        f"(width {max_len_in_batch}) spans more than one partition; otherwise stage1 partitions "
        f"overlap and double-count tokens in the online softmax. A single partition is fine.")
    D = q.shape[-1]
    assert D in {16, 32, 64, 128}

    sm_scale = 1.0 / math.sqrt(D)
    B, H = q.shape[0], q.shape[1]
    # grid is built from python ints (B, H static from cache; max_len_in_batch/block_seq
    # are python ints) -> STATIC launch grid, safe for cudagraph capture & fullgraph.
    grid = (B, H, triton.cdiv(max_len_in_batch, block_seq))
    gqa_group_size = H // k.shape[1]

    wrap_triton(_fwd_kernel_sparse_decode_stage1)[grid](
        q, k, v, sm_scale,
        sparse_list, sparse_len,
        mid_out, mid_out_logsumexp,
        sparse_list.stride(0), sparse_list.stride(1),
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        sparse_len.stride(0), sparse_len.stride(1),
        mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3),
        mid_out_logsumexp.stride(0), mid_out_logsumexp.stride(1), mid_out_logsumexp.stride(2),
        gqa_group_size,
        BLOCK_SEQ=block_seq,
        BLOCK_DMODEL=D,
        BLOCK_N=BLOCK_N,
        NAN_GUARD=_NAN_GUARD,
        num_warps=_STAGE1_NUM_WARPS,
        num_stages=2,
    )


def _sparse_decode_stage2_impl(
    mid_out: torch.Tensor,
    mid_out_logsumexp: torch.Tensor,
    sparse_len: torch.Tensor,
    out: torch.Tensor,       # [B,H,D] fp16/bf16
    block_seq: int,
) -> None:
    D = out.shape[-1]
    assert D in {16, 32, 64, 128}

    B, H = out.shape[0], out.shape[1]
    grid = (B, H)

    wrap_triton(_fwd_kernel_sparse_decode_stage2)[grid](
        sparse_len,
        mid_out,
        mid_out_logsumexp,
        out,
        sparse_len.stride(0), sparse_len.stride(1),
        mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3),
        mid_out_logsumexp.stride(0), mid_out_logsumexp.stride(1), mid_out_logsumexp.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_SEQ=block_seq,
        BLOCK_DMODEL=D,
        NAN_GUARD=_NAN_GUARD,
        num_warps=4,
        num_stages=2,
    )


if _HAS_TRITON_OP:
    # Register as opaque custom ops. mutates_args declares the output buffers the kernel
    # writes so Dynamo/functionalization/cudagraph handle them correctly. Dynamo does NOT
    # trace into these (no graph break under fullgraph=True); they run as a single node.
    sparse_decode_stage1 = triton_op(
        "socket::sparse_decode_stage1",
        _sparse_decode_stage1_impl,
        mutates_args={"mid_out", "mid_out_logsumexp"},
    )
    sparse_decode_stage2 = triton_op(
        "socket::sparse_decode_stage2",
        _sparse_decode_stage2_impl,
        mutates_args={"out"},
    )
else:
    sparse_decode_stage1 = torch.no_grad()(_sparse_decode_stage1_impl)
    sparse_decode_stage2 = torch.no_grad()(_sparse_decode_stage2_impl)


@torch.no_grad()
def sparse_attention_fwd(
    query: torch.Tensor,      # [B,H,D]
    key: torch.Tensor,        # [B,Kv,S,D]
    value: torch.Tensor,      # [B,Kv,S,D]
    sparse_list: torch.Tensor,# [B,H,Ktotal]
    sparse_len: torch.Tensor, # [B,H]
    block_seq: int = 256,
) -> torch.Tensor:
    assert query.is_cuda and key.is_cuda and value.is_cuda and sparse_list.is_cuda and sparse_len.is_cuda
    B, H, D = query.shape

    # max_len_in_batch is the longest per-(b,h) sparse list. build_sparse_list_decode
    # produces a RECTANGULAR sparse_list (every row has exactly sparse_list.shape[-1]
    # entries: sink+window base + M_eff heavy slots, padded with -1) and sets
    # sparse_len == sparse_list.shape[-1] for all (b,h). So the longest list == the last
    # tensor dim, a COMPILE-TIME-STATIC python int -> no `sparse_len.max().item()` host
    # sync, and block_seq_num / mid_o / mid_o_log shapes are static. The stage-1 kernel
    # still reads per-(b,h) Sparse_Len from the tensor for its loop bound, so any future
    # ragged sparse_len with entries <= shape[-1] stays correct (extra blocks see
    # cur_block_start >= cur_seq_len -> block_n_size==0 -> skipped, contribute nothing).
    max_len_in_batch = int(sparse_list.shape[-1])

    block_seq_num = (max_len_in_batch + block_seq - 1) // block_seq
    mid_o = torch.empty((B, H, block_seq_num, D), dtype=torch.float32, device=query.device)
    mid_o_log = torch.empty((B, H, block_seq_num), dtype=torch.float32, device=query.device)
    out = torch.empty((B, H, D), dtype=query.dtype, device=query.device)

    sparse_decode_stage1(
        query, key, value, sparse_list, sparse_len,
        mid_o, mid_o_log, block_seq, max_len_in_batch,
    )
    sparse_decode_stage2(mid_o, mid_o_log, sparse_len, out, block_seq)
    return out
