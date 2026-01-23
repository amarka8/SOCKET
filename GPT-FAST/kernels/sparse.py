import math
import torch
import triton
import triton.language as tl

_SOFT_HASH_EXT = None

def _get_soft_hash_ext():
    global _SOFT_HASH_EXT
    if _SOFT_HASH_EXT is None:
        from kernels.soft_hash_collision_loader import load_soft_hash_collision
        _SOFT_HASH_EXT = load_soft_hash_collision(3)
    return _SOFT_HASH_EXT

@torch.no_grad()
def build_sparse_list_decode(
    q_probs: torch.Tensor,         # [B,H,L,R] fp16
    k_hard_bhlt: torch.Tensor,     # [B,H,L,T] int16/int32
    v_norm_bht: torch.Tensor,      # [B,H,T] fp16/bf16
    allowed_bht: torch.Tensor,     # [B,H,T] bool
    sink: int,
    window: int,
    M: int,
    KC: int = 8,
    BLOCK_N: int = 512,
    num_warps: int = 8,
    num_stages: int = 2,
):
    assert q_probs.is_cuda and k_hard_bhlt.is_cuda and v_norm_bht.is_cuda and allowed_bht.is_cuda
    assert q_probs.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert allowed_bht.dtype == torch.bool

    B, H, L, R = q_probs.shape
    _, _, L2, T = k_hard_bhlt.shape
    assert L2 == L

    device = q_probs.device
    # NOTE: KC/BLOCK_N/num_warps/num_stages kept for API compatibility.
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

    # heavy_idx: [B,H,M_eff]
    valid = (heavy_idx >= 0) & (heavy_idx < T)
    if valid.any():
        ok = torch.gather(allowed_bht, dim=-1, index=heavy_idx.clamp(0, T-1).to(torch.long))
        heavy_idx = heavy_idx.masked_fill(~(valid & ok), -1)
    else:
        heavy_idx = heavy_idx.fill_(-1)
    # 4) sink + window base indices (structured)
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

    # Filter base by allowed mask (keeps shape)
    base_ok = torch.gather(allowed_bht, dim=-1, index=base.to(torch.long))
    base = base.masked_fill(~base_ok, -1)

    # 5) sparse_list / sparse_len
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
            other=0,
        )

        base_ptr = cur_batch * stride_kbb + cur_kv_head * stride_kh
        off_k = base_ptr + token_idx[:, None] * stride_ks + offs_d[None, :]
        k = tl.load(K + off_k, mask=offs_n_new[:, None] < cur_seq_len, other=0.0)
        v = tl.load(V + off_k, mask=offs_n_new[:, None] < cur_seq_len, other=0.0)

        att_value = tl.sum(q[None, :] * k, 1)
        att_value *= sm_scale
        att_value = tl.where(offs_n_new < cur_seq_len, att_value, float("-inf"))

        cur_max_logic = tl.max(att_value, axis=0)
        new_max_logic = tl.maximum(cur_max_logic, max_logic)

        exp_logic = tl.exp(att_value - new_max_logic)
        logic_scale = tl.exp(max_logic - new_max_logic)

        acc *= logic_scale
        acc += tl.sum(exp_logic[:, None] * v, axis=0)
        sum_exp = sum_exp * logic_scale + tl.sum(exp_logic, axis=0)
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

        new_max_logic = tl.maximum(tlogic, max_logic)
        old_scale = tl.exp(max_logic - new_max_logic)
        acc *= old_scale
        exp_logic = tl.exp(tlogic - new_max_logic)
        acc += exp_logic * tv
        sum_exp = sum_exp * old_scale + exp_logic
        max_logic = new_max_logic

    off_o = cur_batch * stride_obs + cur_head * stride_oh + offs_d
    tl.store(O + off_o, acc / sum_exp)


@torch.no_grad()
def sparse_decode_stage1(
    q: torch.Tensor,            # [B,H,D]
    k: torch.Tensor,            # [B,Kv,S,D]
    v: torch.Tensor,            # [B,Kv,S,D]
    sparse_list: torch.Tensor,  # [B,H,Ktotal]
    sparse_len: torch.Tensor,   # [B,H]
    max_len_in_batch: int,
    mid_out: torch.Tensor,         # [B,H,block_seq_num,D] fp32
    mid_out_logsumexp: torch.Tensor,# [B,H,block_seq_num] fp32
    block_seq: int,
):
    BLOCK_N = 16
    D = q.shape[-1]
    assert D in {16, 32, 64, 128}

    sm_scale = 1.0 / math.sqrt(D)
    B, H = q.shape[0], q.shape[1]
    grid = (B, H, triton.cdiv(max_len_in_batch, block_seq))
    gqa_group_size = H // k.shape[1]

    _fwd_kernel_sparse_decode_stage1[grid](
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


@torch.no_grad()
def sparse_decode_stage2(
    mid_out: torch.Tensor,
    mid_out_logsumexp: torch.Tensor,
    sparse_len: torch.Tensor,
    out: torch.Tensor,       # [B,H,D] fp16/bf16
    block_seq: int,
):
    D = out.shape[-1]
    assert D in {16, 32, 64, 128}

    B, H = out.shape[0], out.shape[1]
    grid = (B, H)

    _fwd_kernel_sparse_decode_stage2[grid](
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
    max_len_in_batch = int(sparse_len.max().item())

    block_seq_num = (max_len_in_batch + block_seq - 1) // block_seq
    mid_o = torch.empty((B, H, block_seq_num, D), dtype=torch.float32, device=query.device)
    mid_o_log = torch.empty((B, H, block_seq_num), dtype=torch.float32, device=query.device)
    out = torch.empty((B, H, D), dtype=query.dtype, device=query.device)

    sparse_decode_stage1(query, key, value, sparse_list, sparse_len, max_len_in_batch, mid_o, mid_o_log, block_seq)
    sparse_decode_stage2(mid_o, mid_o_log, sparse_len, out, block_seq)
    return out
