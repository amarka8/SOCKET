import triton
import triton.language as tl
import torch 
from typing import Dict, Optional, Tuple
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Triton kernel: insert one pos into per-(B,H,L) ring buffers
# Correctness matches your PyTorch reference (no atomics needed).
# -----------------------------------------------------------------------------
@triton.jit
def bucket_insert_onepos_kernel(
    slots_ptr, counts_ptr, bkt_ptr,
    pos_idx: tl.constexpr,
    stride_bkt_row: tl.constexpr,
    stride_slots_row: tl.constexpr, stride_slots_r: tl.constexpr, stride_slots_cap: tl.constexpr,
    stride_counts_row: tl.constexpr, stride_counts_r: tl.constexpr,
    R: tl.constexpr, CAP: tl.constexpr,
):
    row = tl.program_id(0)  # 0..N-1

    # bucket id for this (b,h,l)
    bkt = tl.load(bkt_ptr + row * stride_bkt_row).to(tl.int32)  # scalar

    # pointers to this row's slots/counts
    slots_row_ptr  = slots_ptr  + row * stride_slots_row
    counts_row_ptr = counts_ptr + row * stride_counts_row

    # c = counts[row, bkt]
    c = tl.load(counts_row_ptr + bkt * stride_counts_r).to(tl.int32)
    wp = c % CAP

    # slots[row, bkt, wp] = pos_idx
    tl.store(slots_row_ptr + bkt * stride_slots_r + wp * stride_slots_cap, pos_idx)

    # counts[row, bkt] = c + 1
    tl.store(counts_row_ptr + bkt * stride_counts_r, c + 1)


def insert_into_buckets_triton(
    slots: torch.Tensor,                 # [B,H,L,R,cap] int32 CUDA contiguous
    counts: torch.Tensor,                # [B,H,L,R] int32 CUDA contiguous
    key_buckets_for_pos: torch.Tensor,   # [B,H,L] int16/int32 CUDA contiguous
    pos_idx: int,
):
    """
    In-place Triton equivalent of your insert_into_buckets() for a single pos_idx.
    Assumes slots/counts/key_buckets_for_pos are contiguous.
    """
    assert slots.is_cuda and counts.is_cuda and key_buckets_for_pos.is_cuda
    assert slots.dtype == torch.int32 and counts.dtype == torch.int32
    assert key_buckets_for_pos.dtype in (torch.int16, torch.int32, torch.int64)
    assert slots.is_contiguous() and counts.is_contiguous() and key_buckets_for_pos.is_contiguous()

    B, H, L, R, cap = slots.shape
    assert counts.shape == (B, H, L, R)
    assert key_buckets_for_pos.shape == (B, H, L)

    # Flatten (B,H,L) -> N rows
    N = B * H * L
    slots_f  = slots.view(N, R, cap)
    counts_f = counts.view(N, R)
    bkt_f    = key_buckets_for_pos.view(N).to(torch.int32)  # [N]

    # Strides in elements (Triton uses element offsets)
    stride_bkt_row = bkt_f.stride(0)

    stride_slots_row = slots_f.stride(0)
    stride_slots_r   = slots_f.stride(1)
    stride_slots_cap = slots_f.stride(2)

    stride_counts_row = counts_f.stride(0)
    stride_counts_r   = counts_f.stride(1)

    grid = (N,)

    bucket_insert_onepos_kernel[grid](
        slots_f, counts_f, bkt_f,
        pos_idx=pos_idx,
        stride_bkt_row=stride_bkt_row,
        stride_slots_row=stride_slots_row, stride_slots_r=stride_slots_r, stride_slots_cap=stride_slots_cap,
        stride_counts_row=stride_counts_row, stride_counts_r=stride_counts_r,
        R=R, CAP=cap,
        num_warps=1,
        num_stages=1,
    )

@triton.jit
def cra_prefill_kernel(
    slots_ptr, counts_ptr, buckets_ptr,
    stride_buckets_row: tl.constexpr, stride_buckets_t: tl.constexpr,
    stride_slots_row: tl.constexpr, stride_slots_r: tl.constexpr, stride_slots_cap: tl.constexpr,
    stride_counts_row: tl.constexpr, stride_counts_r: tl.constexpr,
    T: tl.constexpr, R: tl.constexpr, CAP: tl.constexpr,
    ):
    row = tl.program_id(0)  # 0..N-1

    buckets_row_ptr = buckets_ptr + row * stride_buckets_row
    slots_row_ptr   = slots_ptr   + row * stride_slots_row
    counts_row_ptr  = counts_ptr  + row * stride_counts_row

    # Sequential inserts: match Python semantics exactly.
    # NOTE: T is tl.constexpr -> specialized per T (prefill length).
    for pos in range(0, T):
        bkt = tl.load(buckets_row_ptr + pos * stride_buckets_t).to(tl.int32)  # scalar
        # counts[bkt]
        c = tl.load(counts_row_ptr + bkt * stride_counts_r)                   # scalar int32
        wp = c % CAP
        # slots[bkt, wp] = pos
        tl.store(slots_row_ptr + bkt * stride_slots_r + wp * stride_slots_cap, pos)
        # counts[bkt] += 1
        tl.store(counts_row_ptr + bkt * stride_counts_r, c + 1)


def prefill_with_triton(slots: torch.Tensor, counts: torch.Tensor, key_buckets: torch.Tensor):
    """
    In-place build of ring-buffer tables from scratch (prefill).

    slots:      [B,H,L,R,cap] int32 (CUDA)
    counts:     [B,H,L,R]     int32 (CUDA)
    key_buckets:[B,H,L,T]     int16/int32 (CUDA) bucket ids in [0,R)
    """
    assert slots.is_cuda and counts.is_cuda and key_buckets.is_cuda
    assert slots.dtype == torch.int32 and counts.dtype == torch.int32
    assert key_buckets.dtype in (torch.int16, torch.int32, torch.int64)

    B, H, L, R, cap = slots.shape
    _, _, _, T = key_buckets.shape

    # Reset outside kernel (fast + avoids huge tl.arange init)
    slots.fill_(-1)
    counts.zero_()

    # Flatten (B,H,L) -> N rows
    N = B * H * L
    buckets_flat = key_buckets.reshape(N, T).contiguous()
    slots_flat   = slots.view(N, R, cap)
    counts_flat  = counts.view(N, R)

    # Strides in elements (Triton expects element offsets)
    stride_buckets_row = buckets_flat.stride(0)
    stride_buckets_t   = buckets_flat.stride(1)

    stride_slots_row = slots_flat.stride(0)
    stride_slots_r   = slots_flat.stride(1)
    stride_slots_cap = slots_flat.stride(2)

    stride_counts_row = counts_flat.stride(0)
    stride_counts_r   = counts_flat.stride(1)

    grid = (N,)

    # R/CAP/T are constexpr -> specialized kernel per (T,R,CAP).
    cra_prefill_kernel[grid](
        slots_flat, counts_flat, buckets_flat,
        stride_buckets_row=stride_buckets_row, stride_buckets_t=stride_buckets_t,
        stride_slots_row=stride_slots_row, stride_slots_r=stride_slots_r, stride_slots_cap=stride_slots_cap,
        stride_counts_row=stride_counts_row, stride_counts_r=stride_counts_r,
        T=T, R=R, CAP=cap,
        num_warps=1,
        num_stages=1,
    )

@triton.jit
def collisions_from_ranges_kernel(
    # per-entry metadata (length S)
    b_ptr, h_ptr, q_ptr, row_ptr, start_ptr, len_ptr,   # int32
    # perm_flat: flattened [BHL*N]
    perm_ptr,                                           # int32
    # output flattened collision: [B*H*Q*N]
    out_ptr,                                            # int32
    # runtime sizes
    S: tl.constexpr,                                    # compile-time grid dim 0 upper bound is S, but we still mask
    # constexpr sizes for address math
    H: tl.constexpr, Q: tl.constexpr, N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    s = tl.program_id(0)        # entry id
    tile = tl.program_id(1)     # tile along positions

    # In practice grid[0] == S, but keep a guard anyway
    s_mask = s < S

    # Load metadata for entry s
    b = tl.load(b_ptr + s, mask=s_mask, other=0).to(tl.int64)
    h = tl.load(h_ptr + s, mask=s_mask, other=0).to(tl.int64)
    q = tl.load(q_ptr + s, mask=s_mask, other=0).to(tl.int64)
    row = tl.load(row_ptr + s, mask=s_mask, other=0).to(tl.int64)
    start = tl.load(start_ptr + s, mask=s_mask, other=0).to(tl.int64)
    length = tl.load(len_ptr + s, mask=s_mask, other=0).to(tl.int64)

    offs = tile * BLOCK + tl.arange(0, BLOCK)
    idx_in_row = start + offs

    # Extra safety: idx_in_row < N
    in_bounds = s_mask & (offs < length) & (idx_in_row < N)

    # perm_flat is [BHL, N] row-major => linear = row*N + idx_in_row
    perm_lin = row * N + idx_in_row
    tok = tl.load(perm_ptr + perm_lin, mask=in_bounds, other=0).to(tl.int64)

    out_lin = (((b * H + h) * Q + q) * N + tok)
    tl.atomic_add(out_ptr + out_lin, 1, mask=in_bounds)


def scatter_from_ranges_triton(
    collision_flat_i32: torch.Tensor,   # [B*H*Q*N] int32
    perm_flat: torch.Tensor,            # [BHL, N] int32
    b: torch.Tensor, h: torch.Tensor, q: torch.Tensor,
    row: torch.Tensor, start: torch.Tensor, lens: torch.Tensor,
    H: int, Q: int, N: int,
    block: int = 256,
    num_warps: int = 4,
) -> torch.Tensor:
    # Ensure correct dtypes / contiguity
    perm_flat = perm_flat.contiguous().to(torch.int32)
    collision_flat_i32 = collision_flat_i32.contiguous()

    b = b.contiguous().to(torch.int32)
    h = h.contiguous().to(torch.int32)
    q = q.contiguous().to(torch.int32)
    row = row.contiguous().to(torch.int32)
    start = start.contiguous().to(torch.int32)
    lens = lens.contiguous().to(torch.int32)

    S = lens.numel()
    if S == 0:
        return collision_flat_i32

    max_len = int(lens.max().item())
    if max_len == 0:
        return collision_flat_i32

    tiles = triton.cdiv(max_len, block)
    grid = (S, tiles)

    # Note: perm_ptr expects flattened memory; passing perm_flat works (it’s contiguous)
    collisions_from_ranges_kernel[grid](
        b, h, q, row, start, lens,
        perm_flat,
        collision_flat_i32,
        S=S,          # compile-time constexpr for masking
        H=H, Q=Q, N=N,
        BLOCK=block,
        num_warps=num_warps,
    )
    return collision_flat_i32


def get_collision_counts_indexed(
    perm: torch.Tensor,         # [B,H,L,N] int32
    offsets: torch.Tensor,      # [B,H,L,R+1] int32
    top_buckets: torch.Tensor,  # [B,H,Q,L,top_t] int
    N: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, H, L, _N = perm.shape
    assert _N == N
    _, _, Q, _L, _top_t = top_buckets.shape
    assert _L == L
    device = perm.device

    perm_flat = perm.reshape(B * H * L, N).contiguous()
    offsets_flat = offsets.reshape(B * H * L, -1).contiguous()

    tb = top_buckets.to(torch.int64)
    tb_sorted, _ = torch.sort(tb, dim=-1)

    keep = torch.ones_like(tb_sorted, dtype=torch.bool)
    keep[..., 1:] = tb_sorted[..., 1:] != tb_sorted[..., :-1]

    bhqlt = keep.nonzero(as_tuple=False)
    b = bhqlt[:, 0]
    h = bhqlt[:, 1]
    q = bhqlt[:, 2]
    l = bhqlt[:, 3]
    tpos = bhqlt[:, 4]

    row = (b * H + h) * L + l
    buckets = tb_sorted[b, h, q, l, tpos]

    R = offsets_flat.shape[1] - 1
    buckets = buckets.clamp_(0, R - 1)

    start = offsets_flat[row, buckets]
    end   = offsets_flat[row, buckets + 1]
    lens = (end - start).clamp_min(0)
    row0 = 0
    bucket_counts = offsets_flat[row0, 1:] - offsets_flat[row0, :-1]


    collision_i32 = torch.zeros((B, H, Q, N), device=device, dtype=torch.int32)
    collision_flat_i32 = collision_i32.view(-1)

    collision_flat_i32 = scatter_from_ranges_triton(
        collision_flat_i32=collision_flat_i32,
        perm_flat=perm_flat,
        b=b, h=h, q=q,
        row=row, start=start, lens=lens,
        H=H, Q=Q, N=N,
        block=128,      # 128/256/512 worth sweeping
        num_warps=4,
    )

    collision_i32 = collision_flat_i32.view(B, H, Q, N)
    candidate_mask = collision_i32 > 0
    collision_i16 = collision_i32.clamp_max(torch.iinfo(torch.int16).max).to(torch.int16)
    return candidate_mask, collision_i16


def build_inverted_index_csr(
    key_buckets: torch.Tensor,  # [B,H,L,N] int
    R: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CSR-style inverted index for bucket -> token ids, per (B,H,L).

    Returns:
      perm:    [B,H,L,N] int32   token indices sorted by bucket id
      offsets: [B,H,L,R+1] int32 prefix sums over counts; bucket r is
               perm[..., offsets[...,r] : offsets[...,r+1]]
    """
    B, H, L, N = key_buckets.shape
    device = key_buckets.device

    # Sort token indices by bucket id (per row)
    perm = torch.argsort(key_buckets, dim=-1)  # [B,H,L,N] int64
    sorted_b = torch.gather(key_buckets, dim=-1, index=perm)  # [B,H,L,N]

    # Count bucket sizes per row using scatter_add into [BHL, R]
    flat = sorted_b.reshape(-1, N).to(torch.int64)  # [BHL, N]
    BHL = flat.shape[0]

    counts = torch.zeros((BHL, R), device=device, dtype=torch.int32)
    src = torch.ones_like(flat, dtype=torch.int32)
    counts.scatter_add_(dim=1, index=flat, src=src)  # [BHL, R]

    # Prefix sums -> offsets
    offsets = torch.zeros((BHL, R + 1), device=device, dtype=torch.int32)
    offsets[:, 1:] = torch.cumsum(counts, dim=1)  # [BHL, R]
    offsets = offsets.view(B, H, L, R + 1)

    return perm.to(torch.int32), offsets

def attention_mask_to_allowed_prob(attention_mask: torch.Tensor, K: int) -> torch.Tensor:
    """
    Convert attention_mask to allowed-probabilities in [0,1], shape [B,1,*,K].

    Heuristics:
      - bool masks:         0 => allow (1.0), 1 => forbid (0.0)
      - additive float mask: >=0 => allow (1.0), <0 => forbid (0.0)
    """
    am = attention_mask[..., :K]
    if am.dtype == torch.bool:
        allowed = (am == 0).to(torch.float32)
    else:
        allowed = (am >= 0).to(torch.float32)

    if allowed.dim() == 3:
        allowed = allowed.unsqueeze(1)  # [B, 1, *, K]
    return allowed