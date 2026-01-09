import triton
import triton.language as tl
import torch 


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