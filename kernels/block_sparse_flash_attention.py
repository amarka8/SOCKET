import triton
import triton.language as tl
import torch
import torch.nn.functional as F


# ----------------------------
# Block index builder (robust)
# ----------------------------
@torch.no_grad()
def random_walk_indices(
    random_walk_probs: torch.Tensor,  # (B, T_q, T_k)
    block_m: int,
    block_n: int,
    K: int,
    q_quantile: float = 0.8,
    enforce_causal_blocks: bool = True,
    force_include_diagonal: bool = True,
    window_blocks: int | None = None,
) -> torch.Tensor:
    """
    Returns: (B, num_rows, K) int32
    - NO -1s (always valid block ids)
    - Always causal (if enforce_causal_blocks)
    - Always includes diagonal (if force_include_diagonal)
    - Always filled to length K (repeats block 0 if needed)
    """
    assert random_walk_probs.dim() == 3
    B, T_q, T_k = random_walk_probs.shape
    device = random_walk_probs.device
    assert block_m > 0 and block_n > 0
    assert K > 0

    num_rows = (T_q + block_m - 1) // block_m
    num_cols = (T_k + block_n - 1) // block_n

    # Pad to full blocks for reshape
    pad_q = num_rows * block_m - T_q
    pad_k = num_cols * block_n - T_k
    x = random_walk_probs
    if pad_q or pad_k:
        x = F.pad(x, (0, pad_k, 0, pad_q), value=0.0)

    # Aggregate to block scores: (B, num_rows, num_cols)
    x = x.view(B, num_rows, block_m, num_cols, block_n)
    scores = x.sum(dim=(2, 4))  # float

    # Build causal-valid mask per row (block-level)
    k_blks = torch.arange(num_cols, device=device)  # (num_cols,)
    if enforce_causal_blocks:
        # last query token in each rowblock (clamped for last partial)
        q_last = torch.minimum(
            (torch.arange(num_rows, device=device) + 1) * block_m - 1,
            torch.tensor(T_q - 1, device=device),
        )
        max_kblk = q_last // block_n  # (num_rows,)
        valid = k_blks[None, :] <= max_kblk[:, None]  # (num_rows, num_cols)
    else:
        valid = torch.ones((num_rows, num_cols), device=device, dtype=torch.bool)

    # Forced blocks mask: diagonal + optional window
    forced = torch.zeros((num_rows, num_cols), device=device, dtype=torch.bool)

    if force_include_diagonal:
        diag = (torch.arange(num_rows, device=device) * block_m) // block_n
        diag = diag.clamp(0, num_cols - 1)
        forced[torch.arange(num_rows, device=device), diag] = True

    if window_blocks is not None and window_blocks > 0:
        diag_blk = ((torch.arange(num_rows, device=device) * block_m) // block_n)[:, None]  # (num_rows,1)
        c = torch.arange(num_cols, device=device)[None, :]  # (1,num_cols)
        win_lo = diag_blk - (window_blocks - 1)
        window_mask = (c <= diag_blk) & (c >= win_lo)
        forced |= window_mask

    forced &= valid

    # Tie-breaker to stabilize topk on flat scores (prefer recent blocks)
    tie = (k_blks.to(scores.dtype) / max(1, num_cols))[None, None, :]
    scores = scores + 1e-6 * tie

    # Mask invalid blocks out for selection
    scores_valid = scores.masked_fill(~valid.unsqueeze(0), float("-inf"))

    forced[:, 0] = True
    forced &= valid

    # Candidate pool excludes forced blocks
    scores_fill = scores_valid.masked_fill(forced.unsqueeze(0), float("-inf"))

    # TopK fill (may include -inf if row has few candidates)
    K_eff = min(K, num_cols)
    fill_idx = torch.topk(scores_fill, K_eff, dim=-1).indices  # (B,num_rows,K_eff)
    fill_scores = torch.gather(scores_fill, dim=-1, index=fill_idx)

    # Convert -inf fills to a safe default (block 0)
    fill_idx = torch.where(fill_scores.isneginf(), torch.zeros_like(fill_idx), fill_idx)

    # Build forced list per row (ragged -> padded)
    forced_idx_list = []
    for r in range(num_rows):
        forced_idx_list.append(torch.nonzero(forced[r], as_tuple=False).flatten())
    maxF = max((t.numel() for t in forced_idx_list), default=0)

    if maxF == 0:
        forced_idx_b = torch.empty((B, num_rows, 0), device=device, dtype=torch.long)
    else:
        forced_pad = torch.full((num_rows, maxF), 0, device=device, dtype=torch.long)  # default to 0
        for r, t in enumerate(forced_idx_list):
            forced_pad[r, : t.numel()] = t
        forced_idx_b = forced_pad.unsqueeze(0).expand(B, -1, -1)

    # Concatenate forced + fill, then unique-per-row, then take first K
    idx_cat = torch.cat([forced_idx_b, fill_idx], dim=-1)  # (B,num_rows,maxF+K_eff)
    idx_cat, _ = torch.sort(idx_cat, dim=-1)              # sorted helps dedup

    # Dedup adjacent equals
    dup = idx_cat[..., 1:] == idx_cat[..., :-1]
    idx_cat2 = idx_cat.clone()
    idx_cat2[..., 1:] = torch.where(dup, torch.tensor(0, device=device, dtype=idx_cat2.dtype), idx_cat2[..., 1:])

    idx_cat2, _ = torch.sort(idx_cat2, dim=-1)
    idx_out = idx_cat2[..., :K_eff].contiguous()

    if force_include_diagonal:
        diag = (torch.arange(num_rows, device=device) * block_m) // block_n
        diag = diag.clamp(0, num_cols - 1).view(1, num_rows, 1).expand(B, -1, -1)
        # replace first slot with diag (cheap, deterministic)
        idx_out[..., 0:1] = diag

    # Ensure causal validity (defensive)
    idx_out = torch.clamp(idx_out, 0, num_cols - 1)

    # Final sort (nice for cache + determinism)
    idx_out, _ = torch.sort(idx_out, dim=-1)
    return idx_out.to(torch.int32)


@torch.no_grad()
def expand_block_index_to_heads(block_index_brk: torch.Tensor, H: int) -> torch.Tensor:
    assert block_index_brk.dtype in (torch.int32, torch.int64)
    B, R, K = block_index_brk.shape
    return block_index_brk[:, None, :, :].expand(B, H, R, K).contiguous().to(torch.int32)


@torch.no_grad()
def block_mask_to_indices(
    allow_blocks: torch.Tensor,  # (B, num_rows, num_cols) bool
    block_m: int,
    block_n: int,
    K: int | None = None,
    force_include_diagonal: bool = True,
    enforce_causal_blocks: bool = True,
) -> torch.Tensor:
    assert allow_blocks.dim() == 3
    assert allow_blocks.dtype == torch.bool
    assert block_m > 0 and block_n > 0
    B, num_rows, num_cols = allow_blocks.shape
    device = allow_blocks.device

    mask = allow_blocks
    if enforce_causal_blocks:
        k_blks = torch.arange(num_cols, device=device)
        q_last = torch.minimum(
            (torch.arange(num_rows, device=device) + 1) * block_m - 1,
            torch.tensor(num_rows * block_m - 1, device=device),
        )
        max_kblk = q_last // block_n
        valid = k_blks[None, :] <= max_kblk[:, None]
        mask = mask & valid.unsqueeze(0)
    if force_include_diagonal:
        diag = (torch.arange(num_rows, device=device) * block_m) // block_n
        diag = diag.clamp(0, num_cols - 1)
        mask = mask.clone()
        row_idx = torch.arange(num_rows, device=device)
        mask[:, row_idx, diag] = True

    if K is None or K <= 0:
        K_eff = int(mask.sum(dim=-1).max().item()) if num_cols > 0 else 0
    else:
        K_eff = K
    K_eff = max(1, min(K_eff, num_cols))

    idx = torch.arange(num_cols, device=device).view(1, 1, num_cols).expand(B, num_rows, num_cols)
    masked = idx.masked_fill(~mask, num_cols)
    idx_sorted, _ = torch.sort(masked, dim=-1)
    idx_out = idx_sorted[..., :K_eff].contiguous()
    idx_out = torch.where(idx_out >= num_cols, torch.zeros_like(idx_out), idx_out)
    return idx_out.to(torch.int32)


# ----------------------------
# Triton kernel (safe, no -1)
# ----------------------------
@triton.jit
def random_walk_fwd(
    Q, K, V, seqlens,
    block_index,   # (B, H, NUM_ROWS, KLIST) int32, ALWAYS VALID
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    NUM_ROWS, KLIST,
    sm_scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    dtype: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz  = tl.program_id(1)

    b = off_hz // H
    h = off_hz % H

    seqlen = tl.load(seqlens + b)
    if start_m * BLOCK_M >= seqlen:
        return

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_off = b * stride_qz + h * stride_qh
    k_off = b * stride_kz + h * stride_kh
    v_off = b * stride_vz + h * stride_vh
    o_off = b * stride_oz + h * stride_oh

    q_ptrs = Q   + q_off + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K   + k_off + offs_d[:, None] * stride_kk
    v_ptrs = V   + v_off + offs_d[None, :] * stride_vk
    o_ptrs = Out + o_off + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok

    blocks_ptr = block_index + ((b * H + h) * NUM_ROWS + start_m) * KLIST

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

    qk_scale = sm_scale * 1.44269504
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < seqlen), other=0.0).to(tl.float32)
    q = (q * qk_scale).to(tl.float32)  # keep fp32 for stability

    m_mask = offs_m[:, None] < seqlen

    # KLIST blocks are always valid ids
    for j in range(0, KLIST):
        real_block_idx = tl.load(blocks_ptr + j).to(tl.int32)
        start_n = real_block_idx * BLOCK_N
        cols = start_n + offs_n

        k = tl.load(
            k_ptrs + cols[None, :] * stride_kn,
            mask=(cols[None, :] < seqlen),
            other=0.0
        ).to(tl.float32)

        v = tl.load(
            v_ptrs + cols[:, None] * stride_vn,
            mask=(cols[:, None] < seqlen),
            other=0.0
        ).to(tl.float32)

        qk = tl.dot(q, k)  # fp32
        causal = cols[None, :] <= offs_m[:, None]
        qk = tl.where(m_mask & causal, qk, -float("inf"))

        m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])

        acc *= alpha[:, None]
        acc += tl.dot(p.to(tl.float32), v)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_i_new

    # normalize (safe epsilon)
    acc = acc / (l_i[:, None] + 1e-9)
    tl.store(o_ptrs, acc.to(dtype), mask=(offs_m[:, None] < seqlen))


def random_walk_sparse_attention(
    q: torch.Tensor,          # (B,H,T,D)
    k: torch.Tensor,          # (B,H,T,D)
    v: torch.Tensor,          # (B,H,T,D)
    seqlens: torch.Tensor,    # (B,) int32
    block_index: torch.Tensor,# (B,H,NUM_ROWS,K) int32 (NO -1)
    block_m: int = 64,
    block_n: int = 64,
) -> torch.Tensor:
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.dtype in (torch.float16, torch.bfloat16)
    assert q.shape == k.shape == v.shape
    B, H, T, D = q.shape
    assert D in (16, 32, 64, 128)
    assert block_index.dtype == torch.int32

    # Ensure contiguous for predictable strides
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    # Pad to full rowblocks
    num_rows = (T + block_m - 1) // block_m
    T_pad = num_rows * block_m
    pad_t = T_pad - T
    if pad_t:
        q = F.pad(q, (0, 0, 0, pad_t))
        k = F.pad(k, (0, 0, 0, pad_t))
        v = F.pad(v, (0, 0, 0, pad_t))

    # Allocate Out explicitly (contiguous)
    out = torch.empty((B, H, T_pad, D), device=q.device, dtype=q.dtype)

    grid = (triton.cdiv(T_pad, block_m), B * H, 1)
    dtype = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16
    sm_scale = (D ** -0.5)

    random_walk_fwd[grid](
        q, k, v, seqlens,
        block_index,
        out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, T_pad,
        block_index.shape[2], block_index.shape[3],
        sm_scale=sm_scale,
        BLOCK_M=block_m, BLOCK_N=block_n,
        BLOCK_D=D,
        dtype=dtype,
        num_warps=4, num_stages=2,
    )

    return out[:, :, :T, :]


def build_and_run_rw_sparse_attn(
    q: torch.Tensor,  # (B,H,T,D)
    k: torch.Tensor,  # (B,H,T,D)
    v: torch.Tensor,  # (B,H,T,D)
    random_walk_probs: torch.Tensor,  # (B,T,T)
    block_size: int = 64,
    Kblocks: int = 8,
    q_quantile: float = 0.8,
    window_blocks: int | None = 4,
):
    B, H, T, D = q.shape
    seqlens = torch.full((B,), T, device=q.device, dtype=torch.int32)

    blk_brk = random_walk_indices(
        random_walk_probs,
        block_size, block_size, Kblocks,
        q_quantile=q_quantile,
        enforce_causal_blocks=True,
        force_include_diagonal=True,
        window_blocks=window_blocks,
    )
    blk_bhrk = expand_block_index_to_heads(blk_brk, H)

    return random_walk_sparse_attention(
        q, k, v, seqlens, blk_bhrk,
        block_m=block_size, block_n=block_size
    )
