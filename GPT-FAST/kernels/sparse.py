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

# Which scorer implementation to use: "cuda" (the load_inline kernel, DEFAULT),
# "triton" (mutating triton_op) or "tritonalloc" (same kernel, non-mutating custom_op).
# Default is "cuda" because both Triton forms, despite winning the isolated microbenchmark,
# LOSE end-to-end under torch.compile -- see the comment on _soft_hash_score_alloc.
# SOCKET_TRITON_SCORER=1 is still honoured as a shorthand for "triton".
_SCORER_IMPL = os.environ.get(
    "SOCKET_SCORER_IMPL",
    "triton" if os.environ.get("SOCKET_TRITON_SCORER", "0") == "1" else "cuda",
).strip().lower()

_SOFT_HASH_EXT = None

def _get_soft_hash_ext():
    global _SOFT_HASH_EXT
    if _SOFT_HASH_EXT is None:
        from kernels.soft_hash_collision_loader import load_soft_hash_collision
        _SOFT_HASH_EXT = load_soft_hash_collision(3)
    return _SOFT_HASH_EXT


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
# CHANGE 3/4a: TRITON SCORER (replaces the load_inline CUDA soft_hash_collision).
#
# BIT-EXACT vs soft_hash_collision_kernel_3: the per-token score is accumulated as
#   acc = 0; for l in 0..L-1: acc += float(q_probs[b,h,l, buckets[b,kv,l,t]]);  out = acc * float(v[t])
# i.e. the SAME fp32 adds in the SAME l order, then the same single multiply. `L` is a
# constexpr so the loop is unrolled but not reassociated.
#
# What it removes vs the CUDA version (all pure overhead, no arithmetic change):
#   * allowed_ext ([B,H,maxlen] bool) is GONE. The mask is exactly `t < seq_len`; the old
#     path materialized it (expand().contiguous(), 4.6 MB write at 140K) and then read it
#     (4.6 MB) once per layer per decode step, for a value identical across all heads and
#     all 32 layers.
#   * v_norm is read as fp16 in-kernel instead of forcing v_norm.float() on the whole
#     [B,Hkv,maxlen] buffer every layer every step (2.3 MB read + 4.6 MB write, then 2x the
#     kernel-side read).
#   * q_probs is gathered from the native bf16 table and converted per element, instead of
#     materializing an fp32 [B,H,1,L,R] copy. float(bf16) is exact, so the gathered value is
#     identical.
# ---------------------------------------------------------------------------
@triton.jit
def _fwd_kernel_soft_hash_score(
    QProbs,          # [B,H,L,R]        (bf16 or fp32)
    KeyBuckets,      # [B,Hkv,L,T]      int16
    VNorm,           # [B,Hkv,T]        fp16/bf16/fp32
    SeqLenPtr,       # int32 scalar on device
    Out,             # [B,H,T]          fp32 (written)
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
    for l in tl.static_range(L):
        bkt = tl.load(KeyBuckets + kb_base + l * T + offs_t, mask=mask_t, other=0).to(tl.int32)
        qv = tl.load(QProbs + qp_base + l * R + bkt, mask=mask_t, other=0.0)
        acc += qv.to(tl.float32)

    v = tl.load(VNorm + (b * HKV + kv) * T + offs_t, mask=mask_t, other=0.0).to(tl.float32)
    out = acc * v
    out = tl.where(keep, out, -float("inf"))
    tl.store(Out + (b * H + h) * T + offs_t, out, mask=mask_t)


def _soft_hash_score_impl(
    q_probs: torch.Tensor,      # [B,H,L,R]
    key_buckets: torch.Tensor,  # [B,Hkv,L,T] int16
    v_norm: torch.Tensor,       # [B,Hkv,T]
    seq_len_t: torch.Tensor,    # int32 scalar
    out: torch.Tensor,          # [B,H,T] fp32 (written)
) -> None:
    B, H, L, R = q_probs.shape
    HKV, T = key_buckets.shape[1], key_buckets.shape[3]
    # Launch config matters a LOT here and the optimum is a constant ~4 ELEMENTS PER THREAD
    # (BLOCK_T / (32*num_warps) == 4); the L-deep gather loop makes anything wider spill.
    # Measured at T=143411 vs the CUDA scorer + the fp32 casts it forces (105.4 us at L=10,
    # 251.3 us at L=50):
    #   L=10: (128,1) 47.7us = 2.21x   [(1024,4) -- the first default here -- was 51.2us]
    #   L=50: (256,2) 180.3us = 1.39x  [(1024,4) was 298.3us = 0.84x, i.e. a REGRESSION]
    # 40 (BLOCK_T, num_warps) variants were checked and every one is bitwise-equal to the
    # CUDA scorer, so this choice is purely a performance knob.
    if L <= 16:
        BLOCK_T, warps = 128, 1
    else:
        BLOCK_T, warps = 256, 2
    BLOCK_T = int(os.environ.get("SOCKET_SCORER_BLOCK_T", BLOCK_T))
    warps = int(os.environ.get("SOCKET_SCORER_WARPS", warps))
    grid = (triton.cdiv(T, BLOCK_T), H, B)
    wrap_triton(_fwd_kernel_soft_hash_score)[grid](
        q_probs, key_buckets, v_norm, seq_len_t, out,
        H=H, HKV=HKV, L=L, R=R, T=T, BLOCK_T=BLOCK_T,
        num_warps=warps, num_stages=2,
    )


if _HAS_TRITON_OP:
    soft_hash_score_op = triton_op(
        "socket::soft_hash_score", _soft_hash_score_impl, mutates_args={"out"}
    )
else:
    soft_hash_score_op = torch.no_grad()(_soft_hash_score_impl)


# ---------------------------------------------------------------------------
# ALTERNATE REGISTRATION of the SAME Triton kernel, as a NON-MUTATING custom_op that
# allocates its own output -- structurally identical to how the CUDA scorer is registered
# (mutates_args=(), returns a fresh tensor).
#
# Why: the mutating triton_op form above is 2.0-2.2x FASTER than the CUDA scorer in an
# isolated microbenchmark, yet 6-36% SLOWER end-to-end under
# torch.compile(mode="reduce-overhead"). The kernel accounts for only ~23 us/layer of a
# ~174 us/layer end-to-end loss at L=50, so the cost is in how the mutated 18.4 MB output
# buffer interacts with functionalization / cudagraph_trees, not in the arithmetic. This
# variant removes the mutation so that hypothesis can be measured directly.
# ---------------------------------------------------------------------------
@torch.library.custom_op("socket::soft_hash_score_alloc", mutates_args=())
def _soft_hash_score_alloc(
    q_probs: torch.Tensor,
    key_buckets: torch.Tensor,
    v_norm: torch.Tensor,
    seq_len_t: torch.Tensor,
) -> torch.Tensor:
    B, H, L, R = q_probs.shape
    HKV, T = key_buckets.shape[1], key_buckets.shape[3]
    out = torch.empty((B, H, T), device=q_probs.device, dtype=torch.float32)
    if L <= 16:
        BLOCK_T, warps = 128, 1
    else:
        BLOCK_T, warps = 256, 2
    BLOCK_T = int(os.environ.get("SOCKET_SCORER_BLOCK_T", BLOCK_T))
    warps = int(os.environ.get("SOCKET_SCORER_WARPS", warps))
    grid = (triton.cdiv(T, BLOCK_T), H, B)
    # raw launch (NOT wrap_triton): Dynamo never traces into a custom_op body.
    _fwd_kernel_soft_hash_score[grid](
        q_probs, key_buckets, v_norm, seq_len_t, out,
        H=H, HKV=HKV, L=L, R=R, T=T, BLOCK_T=BLOCK_T,
        num_warps=warps, num_stages=2,
    )
    return out


@_soft_hash_score_alloc.register_fake
def _soft_hash_score_alloc_fake(q_probs, key_buckets, v_norm, seq_len_t):
    B, H = q_probs.shape[0], q_probs.shape[1]
    T = key_buckets.shape[3]
    return q_probs.new_empty((B, H, T), dtype=torch.float32)


# ---------------------------------------------------------------------------
# CHANGE 3: FUSED LIST ASSEMBLY.
# Everything after topk used to be ~25 separate tiny CUDA kernels on [B,H,M] / [W] tensors:
#   valid=(>=0)&(<maxlen); ok=gather(allowed, clamp(heavy)); masked_fill;
#   arange(sink); clamp(seq_len-window); arange(window)+off; clamp; cat; view/expand;
#   gather(allowed, base); masked_fill; win_start=clamp; in_sink; in_window; or; masked_fill;
#   cat([base,heavy]); contiguous; full(sparse_len)
# This single kernel emits sparse_list directly and needs NO `allowed` tensor at all -- the
# allowed mask is exactly `t < seq_len`, a scalar compare, so materializing + reading a
# [B,H,maxlen] bool (4.6 MB each way per layer per decode step at 140K) is pure waste.
#
# Semantics reproduced EXACTLY (see the eager code this replaces):
#   slot in [0, sink)                  -> index = slot                       (sink)
#   slot in [sink, sink+window)        -> index = min(win_start + j, maxlen-1)  (window)
#   slot in [sink+window, W)           -> index = heavy[m]                   (top-M)
#   win_start = max(seq_len - window, sink)
#   every index is dropped to -1 unless 0 <= index < seq_len          (allowed gating)
#   a heavy index is ALSO dropped to -1 if it lies in [0,sink) or [win_start,seq_len)  (dedup)
# Ordering of the two heavy masks is irrelevant: a -1 fails both `>=0` predicates.
# ---------------------------------------------------------------------------
@triton.jit
def _fwd_kernel_build_list(
    Heavy,            # [B,H,M] int32   (topk indices; unsorted is fine)
    SeqLenPtr,        # int32 scalar on device (= pos.max()+1, the true filled length)
    Out,              # [B,H,W] int32   (written)
    stride_hb, stride_hh,
    stride_ob, stride_oh,
    SINK: tl.constexpr, WINDOW: tl.constexpr, M: tl.constexpr,
    MAXLEN: tl.constexpr, W: tl.constexpr, BLOCK: tl.constexpr,
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
    # heavy dedup against sink U window
    dup = is_heavy & (((idx >= 0) & (idx < SINK)) | ((idx >= win_start) & (idx < seq_len)))
    idx = tl.where(keep & (dup == 0), idx, -1)

    tl.store(Out + b * stride_ob + h * stride_oh + slot, idx, mask=in_range)


def _build_list_impl(
    heavy_idx: torch.Tensor,   # [B,H,M] int32
    seq_len_t: torch.Tensor,   # int32 scalar on device
    out: torch.Tensor,         # [B,H,W] int32 (written)
    sink: int, window: int, maxlen: int,
) -> None:
    B, H, M = heavy_idx.shape
    W = out.shape[-1]
    BLOCK = 256
    grid = (B, H, triton.cdiv(W, BLOCK))
    wrap_triton(_fwd_kernel_build_list)[grid](
        heavy_idx, seq_len_t, out,
        heavy_idx.stride(0), heavy_idx.stride(1),
        out.stride(0), out.stride(1),
        SINK=sink, WINDOW=window, M=M, MAXLEN=maxlen, W=W, BLOCK=BLOCK,
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
    allowed_bht,                   # [B,H,maxlen] bool or None (LEGACY paths only)
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
    mask both in the scorer (soft_hash_collision writes -inf there) and via the `base_ok` gather
    below, so they can never be selected and never enter attention. The result is identical to
    slicing the cache to the true filled length.
    """
    assert q_probs.is_cuda and k_hard_bhlt.is_cuda and v_norm_bht.is_cuda
    assert q_probs.dtype in (torch.float16, torch.bfloat16, torch.float32)
    # allowed_bht is only needed by the two LEGACY paths (CUDA scorer / eager list assembly).
    # The fused paths derive the mask from the seq_len scalar, so the caller passes None and
    # the [B,H,maxlen] bool tensor is never built at all.
    if allowed_bht is not None:
        assert allowed_bht.is_cuda
        assert allowed_bht.dtype == torch.bool

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
    if allowed_bht is not None:
        assert allowed_bht.shape[1] == H, "allowed must be per-query-head [B,H,maxlen]"

    device = q_probs.device

    if seq_len_t is None:
        # Eager fallback: assume the cache is fully filled, true length == maxlen.
        seq_len_t = torch.tensor(maxlen, device=device, dtype=torch.int32)
    else:
        seq_len_t = seq_len_t.to(device=device, dtype=torch.int32).reshape(())

    # NOTE: KC/BLOCK_N/num_warps/num_stages kept for API compatibility.
    M_eff = min(M, maxlen)  # static
    if M_eff > 0:
        key_buckets = k_hard_bhlt                                  # [B,Hkv,L,maxlen] (per kv head)
        # int16 REQUIRED: R=256 buckets span 0..255 which do NOT fit signed int8 -> NO int8.
        if key_buckets.dtype != torch.int16:
            key_buckets = key_buckets.to(torch.int16)
        key_buckets = key_buckets.contiguous()

        _impl = _SCORER_IMPL
        if _impl == "triton":
            # Mutating triton_op form.
            scores = torch.empty((B, H, maxlen), device=device, dtype=torch.float32)
            soft_hash_score_op(q_probs.contiguous(), key_buckets,
                               v_norm_bht.contiguous(), seq_len_t.reshape(()), scores)
        elif _impl == "tritonalloc":
            # Same kernel, non-mutating custom_op form (allocates its own output).
            scores = torch.ops.socket.soft_hash_score_alloc(
                q_probs.contiguous(), key_buckets,
                v_norm_bht.contiguous(), seq_len_t.reshape(()))
        else:
            # LEGACY CUDA scorer (SOCKET_TRITON_SCORER=0): kept so the gates can compare
            # against byte-identical baseline math in the same build.
            q_probs_f32 = q_probs.float().unsqueeze(2).contiguous()  # [B,H,1,L,R]
            allowed_ext = allowed_bht.unsqueeze(2).contiguous()      # [B,H,1,maxlen]
            v_hist = v_norm_bht.float().unsqueeze(2).contiguous()    # [B,Hkv,1,maxlen]
            # Registered custom op (socket::soft_hash_collision) so torch.compile
            # (fullgraph=True) sees a known op instead of an opaque pybind call.
            scores = torch.ops.socket.soft_hash_collision(
                q_probs_f32, key_buckets, allowed_ext, v_hist,
            ).squeeze(2)  # [B,H,maxlen]  (unfilled/disallowed columns are -inf)

        # sorted=False: the list is consumed as a SET (the stage1 online softmax visits every
        # slot and -1 slots are masked out), so topk's descending SORT of the M selected
        # entries is dead weight. Measured per layer per decode step at T=143411:
        # 0.212 -> 0.115 ms at M=4104 (33x), 0.135 -> 0.114 ms at M=2627 (50x). The selected
        # SET is unchanged; only the order within the list differs, which changes the
        # online-softmax accumulation ORDER (last-bit fp differences, not selection changes).
        # SOCKET_TOPK_SORTED=1 restores sorted=True (LEGACY control for A/B + the gates).
        top = torch.topk(scores, k=M_eff, dim=-1, largest=True,
                         sorted=(os.environ.get("SOCKET_TOPK_SORTED", "0") == "1"))
        heavy_idx = top.indices.to(torch.int32)
    else:
        heavy_idx = torch.empty((B, H, 0), device=device, dtype=torch.int32)

    # ---- FUSED LIST ASSEMBLY (change 3) -------------------------------------
    # Replaces the ~25 tiny kernels this function used to launch (allowed-gather + masked_fill
    # on heavy, arange/clamp/cat/expand/gather/masked_fill for the sink+window base, the
    # in_sink/in_window dedup, the final cat, and the torch.full for sparse_len) with ONE
    # Triton kernel + a cached sparse_len. `allowed_bht` is no longer needed here: the mask
    # is exactly `t < seq_len`, evaluated as a scalar compare inside the kernel.
    # SOCKET_FUSED_LIST=0 restores the original eager op chain (LEGACY control for the gates).
    sink = max(0, min(sink, maxlen))
    window = max(0, min(window, maxlen))
    W = sink + window + M_eff if (sink + window + M_eff) > 0 else 1

    if os.environ.get("SOCKET_FUSED_LIST", "1") == "1":
        sparse_list = torch.empty((B, H, W), device=device, dtype=torch.int32)
        if M_eff <= 0:
            heavy_in = torch.empty((B, H, 1), device=device, dtype=torch.int32)
            heavy_in.fill_(-1)
        else:
            heavy_in = heavy_idx
        build_list_op(heavy_in, seq_len_t.reshape(()), sparse_list, sink, window, maxlen)
        # NOTE: sparse_len must be allocated FRESH here. Caching it across calls (it is a
        # constant W for every (b,h)) breaks under torch.compile(mode="reduce-overhead"):
        # the cached tensor is allocated inside the CUDA-graph memory pool on the first
        # (recording) call, and reusing it on later replays raises "accessing tensor output
        # of CUDAGraphs that has been overwritten by a subsequent run".
        sparse_len = torch.full((B, H), W, device=device, dtype=torch.int32)
        return sparse_list, sparse_len

    # ---- LEGACY eager path (bit-identical reference used by the equivalence gates) --------
    valid = (heavy_idx >= 0) & (heavy_idx < maxlen)
    ok = torch.gather(
        allowed_bht, dim=-1, index=heavy_idx.clamp(0, maxlen - 1).to(torch.long)
    )
    heavy_idx = heavy_idx.masked_fill(~(valid & ok), -1)

    parts = []
    if sink > 0:
        parts.append(torch.arange(sink, device=device, dtype=torch.int32))
    if window > 0:
        win_off = torch.clamp(seq_len_t - window, min=sink)
        win_idx = torch.arange(window, device=device, dtype=torch.int32) + win_off
        win_idx = torch.clamp(win_idx, max=maxlen - 1)
        parts.append(win_idx)

    if len(parts) == 0:
        base = (seq_len_t - 1).clamp(min=0).reshape(1).to(torch.int32)
    else:
        base = torch.cat(parts, dim=0)

    base = base.view(1, 1, -1).expand(B, H, -1)
    base_ok = torch.gather(allowed_bht, dim=-1, index=base.to(torch.long))
    base = base.masked_fill(~base_ok, -1)

    win_start = torch.clamp(seq_len_t - window, min=sink) if window > 0 else seq_len_t
    in_sink = (heavy_idx >= 0) & (heavy_idx < sink)
    in_window = (heavy_idx >= win_start) & (heavy_idx < seq_len_t)
    heavy_idx = heavy_idx.masked_fill(in_sink | in_window, -1)

    sparse_list = torch.cat([base, heavy_idx], dim=-1).contiguous()
    sparse_len = torch.full((B, H), sparse_list.shape[-1], device=device, dtype=torch.int32)
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

        exp_logic = tl.exp(att_value - new_max_logic).to(tl.float32)
        logic_scale = tl.exp(max_logic - new_max_logic).to(tl.float32)

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
        tl.store(Mid_O + off_mid_o, acc / sum_exp)
        tl.store(Mid_O_LogExpSum + off_mid_o_logexpsum, max_logic + tl.log(sum_exp))


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
        old_scale = tl.exp(max_logic - new_max_logic).to(tl.float32)
        exp_logic = tl.exp(tlogic - new_max_logic).to(tl.float32)
        acc = (acc * old_scale + exp_logic * tv).to(tl.float32)
        sum_exp = (sum_exp * old_scale + exp_logic).to(tl.float32)
        max_logic = new_max_logic

    off_o = cur_batch * stride_obs + cur_head * stride_oh + offs_d
    tl.store(O + off_o, acc / sum_exp)


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
    # BLOCK_N is the inner gather width. 16 was leaving the gather badly under-vectorized:
    # measured stage1 at T=143411/width=3107 is 52.5us at BLOCK_N=16 vs 27.9us at BLOCK_N=64
    # (1.9x), and 68.7 -> 36.0us at width=4584. BLOCK_SEQ stays 256 so stage2's partial count
    # is unchanged. INVARIANT: BLOCK_SEQ % BLOCK_N == 0 -- the inner mask is
    # `offs_n_new < cur_seq_len` (not < cur_block_end), so a non-dividing BLOCK_N would let a
    # block read into the NEXT block's range and DOUBLE-COUNT those tokens.
    BLOCK_N = int(os.environ.get("SOCKET_BLOCK_N", "64"))
    assert block_seq % BLOCK_N == 0, (
        f"BLOCK_SEQ ({block_seq}) must be divisible by BLOCK_N ({BLOCK_N}); otherwise stage1 "
        f"blocks overlap and double-count tokens in the online softmax.")
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
        num_warps=4,
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
