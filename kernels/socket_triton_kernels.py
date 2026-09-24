import triton
import triton.language as tl
import torch 
from typing import Dict, Optional, Tuple
import torch.nn.functional as F

import math

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

        # The sparse list uses -1 as a padding entry, so a slot counts as real only when
        # it lies inside the list AND holds a non-negative index. safe_idx substitutes 0
        # for padded lanes so the address arithmetic below stays in bounds; those lanes
        # are masked off the load and set to -inf for the softmax.
        valid_tok = (offs_n_new < cur_seq_len) & (token_idx >= 0)
        safe_idx = tl.where(valid_tok, token_idx, 0)

        base_ptr = cur_batch * stride_kbb + cur_kv_head * stride_kh
        off_k = base_ptr + safe_idx[:, None] * stride_ks + offs_d[None, :]
        k = tl.load(K + off_k, mask=valid_tok[:, None], other=0.0)
        v = tl.load(V + off_k, mask=valid_tok[:, None], other=0.0)

        att_value = tl.sum(q[None, :] * k, 1)
        att_value *= sm_scale
        att_value = tl.where(valid_tok, att_value, float("-inf"))

        cur_max_logic = tl.max(att_value, axis=0)
        new_max_logic = tl.maximum(cur_max_logic, max_logic)

        # When every slot in a chunk is invalid, att_value and new_max_logic are all -inf
        # and exp(-inf - -inf) is NaN. safe_max substitutes 0.0 for a still--inf running
        # max, which leaves each exponent at -inf - 0 = -inf and so contributes 0. On any
        # chunk holding at least one valid slot new_max_logic is finite and safe_max is
        # exactly new_max_logic.
        safe_max = tl.where(new_max_logic == float("-inf"), 0.0, new_max_logic)

        exp_logic = tl.exp(att_value - safe_max)
        # max_logic is used unsubstituted: exp(-inf - safe_max) is 0 for both a finite and
        # a zeroed safe_max, which is the correct rescale for an accumulator that has not
        # yet taken any contribution.
        logic_scale = tl.exp(max_logic - safe_max)

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
        # A partition whose every slot is padding or out of range takes no contribution, so
        # sum_exp is still 0: acc / sum_exp is 0/0 = NaN and log(0) = -inf, and stage2 merges
        # that partial straight into the output. Store a neutral partial instead -- any value
        # divided by 1, carrying logsumexp -inf, which stage2 weights by exp(-inf - m) = 0.
        # Unchanged whenever sum_exp > 0, i.e. on every partition holding a valid slot.
        _empty = sum_exp == 0.0
        _safe_sum = tl.where(_empty, 1.0, sum_exp)
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
        # Same -inf - -inf hazard as stage1, one level up: a partial carrying logsumexp -inf
        # (an all-padding partition) merged while the running max is also still -inf makes
        # both exp() calls evaluate exp(-inf - -inf) = NaN. Bit-identical whenever
        # new_max_logic is finite.
        _safe_max2 = tl.where(new_max_logic == float("-inf"), 0.0, new_max_logic)
        old_scale = tl.exp(max_logic - _safe_max2)
        acc *= old_scale
        exp_logic = tl.exp(tlogic - _safe_max2)
        acc += exp_logic * tv
        sum_exp = sum_exp * old_scale + exp_logic
        max_logic = new_max_logic

    off_o = cur_batch * stride_obs + cur_head * stride_oh + offs_d
    # Defensive: sum_exp == 0 here means EVERY slot for this (b,h) was padding, which the
    # list builder should never produce (sink alone is always valid). Emit zeros rather than
    # NaN if it ever does, so the failure stays local instead of poisoning the whole layer.
    tl.store(O + off_o, acc / tl.where(sum_exp == 0.0, 1.0, sum_exp))


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
