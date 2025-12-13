import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedScorer(nn.Module):
    """
    Scores each chunk j given per-token query state h_t.
    """
    def __init__(self, q_dim: int, k_dim: int, hidden: int):
        super().__init__()
        self.q_proj = nn.Linear(q_dim, hidden, bias=False)
        self.k_proj = nn.Linear(k_dim, hidden, bias=False)
        self.ln     = nn.LayerNorm(3 * hidden)
        self.fc_u   = nn.Linear(3 * hidden, hidden)  # value stream
        self.fc_v   = nn.Linear(3 * hidden, hidden)  # gate stream
        self.out    = nn.Linear(hidden, 1)

    def forward(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """
        Q: (B, q_len, qdim)  per-token states used as "queries" for routing (e.g., Q-projected or hidden)
        K: (B, k_len, kdim) chunk summaries
        Returns:
          scores s_{t,j}: (B, q_len, k_len)
        """
        B, q_len, qdim = Q.shape
        _, k_len, kdim = K.shape

        q = self.q_proj(Q).unsqueeze(2).expand(B, q_len, k_len, -1)
        k = self.k_proj(K).unsqueeze(1).expand(B, q_len, k_len, -1)
        x = torch.cat([q, k, q * k], dim=-1)                  # (B,q_len,k_len,3hidden)
        x = self.ln(x)
        U = self.fc_u(x)
        V = self.fc_v(x)
        Z = U * F.gelu(V)                                     # GEGLU
        s = self.out(Z).squeeze(-1)                           # (B,q_len,k_len)
        return s


def topk_indices(logits: torch.Tensor, k: int) -> torch.Tensor:
    # logits: (B,q_len,k_len) -> indices (B,q_len,k)
    B, q_len, k_len = logits.shape
    causal_mask = torch.tril(torch.ones(q_len, k_len, device=logits.device)).bool()
    masked_logits = logits.masked_fill(~causal_mask, float('-inf'))
    return torch.topk(masked_logits, k=k, dim=-1).indices

def soft_selection_weights(logits: torch.Tensor, tau: float) -> torch.Tensor:
    # logits: (B,q_len,k_len) -> soft probs (B,q_len,k_len)
    return F.softmax(logits / tau, dim=-1)

def build_block_bias_from_topk(
    topk_idx: torch.Tensor,             # (B, q_len, k_len) selected chunk indices
    L_ctx: int,                         # total # of context tokens
    chunk_len: int,                     # base chunk size k
    dtype,
    soft_weights: Optional[torch.Tensor] = None,  # (B, q_len, k_len) optional soft gates
    soft_alpha: float = 8.0,
) -> torch.Tensor:
    """
    Additive attention bias (B, q_len, L_ctx) with equal-sized chunks except possibly the last:
      chunk j covers [j*chunk_len,  min((j+1)*chunk_len, L_ctx)).
    Selected chunks get bias 0; others get -inf (hard) or a finite negative soft bias.
    """
    B, q_len, K = topk_idx.shape
    device = topk_idx.device

    num_chunks = (L_ctx + chunk_len - 1) // chunk_len  # ceil(L/k)

    # selected_chunks[b,q_len,j] = True if j chosen at (b,q_len)
    if topk_idx.max() > num_chunks:
        import pdb
        import torch.distributed as dist
        dist.barrier()
        if dist.get_rank() == 0:
            import pdb; pdb.set_trace()
        dist.barrier()
    selected_chunks = torch.zeros((B, q_len, num_chunks), device=device, dtype=torch.bool)
    selected_chunks.scatter_(-1, topk_idx, True)

    # Map token positions -> chunk ids: chunk_id[l] = l // chunk_len in [0, num_chunks-1]
    token_ids = torch.arange(L_ctx, device=device)
    token_chunk = torch.div(token_ids, chunk_len, rounding_mode='floor')  # (L,)
    token_chunk = token_chunk.clamp_max(num_chunks - 1).view(1, 1, L_ctx).expand(B, q_len, L_ctx)

    # For each (b,q_len,l), whether its chunk is selected
    selected_tokens = torch.gather(selected_chunks, dim=-1, index=token_chunk)  # (B,q_len,L_ctx), bool

    if soft_weights is None:
        # Hard mask
        bias = torch.full((B, q_len, L_ctx), float("-inf"), device=device, dtype=dtype)
        bias[selected_tokens] = 0.0
    else:
        # Soft warmup: per-chunk gate -> per-token bias via gather
        assert soft_weights.shape == (B, q_len, num_chunks), f"soft_weights must be (B,q_len,J) with J={num_chunks}"
        per_chunk_bias = -soft_alpha * (1.0 - soft_weights.to(dtype))    # (B,q_len,J)
        bias = torch.gather(per_chunk_bias, dim=-1, index=token_chunk)   # (B,q_len,L_ctx)
        bias[selected_tokens] = 0.0

    # Causal mask (positions > q_len are invalid)
    tri = torch.triu(torch.ones((q_len, L_ctx), device=device, dtype=torch.bool), diagonal=1)
    bias.masked_fill_(tri.unsqueeze(0), float("-inf"))  # (1,q_len,L) -> (B,q_len,L)

    return bias


def build_block_bias_from_selected(
    selected_chunks: torch.Tensor,  # (B,T,J) bool
    L_ctx: int,
    chunk_len: int,
    dtype: torch.dtype,
    soft_weights: Optional[torch.Tensor] = None,  # (B,T,J)
    soft_alpha: float = 8.0,
) -> torch.Tensor:
    
    B, T, J = selected_chunks.shape
    device = selected_chunks.device

    token_ids   = torch.arange(L_ctx, device=device)
    token_chunk = torch.div(token_ids, chunk_len, rounding_mode='floor').clamp_max(J-1)
    token_chunk = token_chunk.view(1,1,L_ctx).expand(B,T,L_ctx)
    selected_tokens = torch.gather(selected_chunks, dim=-1, index=token_chunk)  # (B,T,L)

    if soft_weights is None:
        bias = torch.full((B,T,L_ctx), float("-inf"), device=device, dtype=dtype)
        bias[selected_tokens] = 0.0
    else:
        per_chunk_bias = -soft_alpha * (1.0 - soft_weights.to(dtype))
        bias = torch.gather(per_chunk_bias, dim=-1, index=token_chunk)
        bias[selected_tokens] = 0.0

    tri = torch.triu(torch.ones((T, L_ctx), device=device, dtype=torch.bool), diagonal=1)
    bias.masked_fill_(tri.unsqueeze(0), float("-inf"))

    return bias

@torch.no_grad()
def select_tail_variable_k_ranks(logits_tail: torch.Tensor, k_tail: torch.Tensor) -> torch.Tensor:
    """
    logits_tail: (B,T,J_tail); k_tail: (B,T) -> sel_tail: (B,T,J_tail) bool
    """
    B, T, J_tail = logits_tail.shape
    if J_tail == 0:
        return torch.zeros((B,T,0), dtype=torch.bool, device=logits_tail.device)
    
    k_tail = k_tail.clamp(min=0, max=J_tail)
    _, idx_desc = torch.sort(logits_tail, dim=-1, descending=True)         # (B,T,J_tail)
    ranks = torch.empty_like(idx_desc)
    order = torch.arange(J_tail, device=logits_tail.device).view(1,1,J_tail).expand(B,T,J_tail)
    ranks.scatter_(-1, idx_desc, order)                                     # invert argsort

    return ranks < k_tail.unsqueeze(-1)


class TopkMasker(nn.Module):
    """
    Top-k Masker with optional per-token learnable k.
    - If use_learnable_k=False: uses the original deterministic top-k selection.
    - If use_learnable_k=True: learns a per-token gate to scale k adaptively.
    Robustly handles sink_blocks >= num_chunks (J).
    """
    def __init__(
        self,
        q_dim: int,
        k_dim: int,
        hidden: int,
        topk: int,
        gate_hidden: Optional[int] = None,
        bce_pos_weight: Optional[float] = None,
    ):
        super().__init__()
        self.scorer = GatedScorer(q_dim, k_dim, hidden)
        self.topk = topk

        # gate head (always exists; can be disabled at runtime)
        h = gate_hidden or max(32, q_dim // 4)
        self.gate_head = nn.Sequential(
            nn.Linear(q_dim, h),
            nn.GELU(),
            nn.Linear(h, 1)
        )
        if bce_pos_weight is not None:
            self.register_buffer("pos_w", torch.tensor([float(bce_pos_weight)]))
        else:
            self.pos_w = None

        # warmup params
        self.register_buffer("tau", torch.tensor(1.0))
        self.soft_alpha = 8.0
        self.use_soft_warmup = True
        self._tag = getattr(self, "_tag", f"TopkMasker@{id(self)%10000}")

    def set_warmup(self, on=True, tau: float = 1.0, soft_alpha: float = 8.0):
        self.use_soft_warmup = on
        self.tau.fill_(tau)
        self.soft_alpha = soft_alpha

    @torch.no_grad()
    def _handle_all_sink(self, logits, Q, L_ctx, chunk_len):
        """If sink_blocks >= num_chunks (J), everything attendable."""
        B, T, J = logits.shape
        idx = torch.arange(J, device=Q.device).view(1, 1, J).expand(B, T, J)
        w_soft = soft_selection_weights(logits, tau=float(self.tau.item())) if self.use_soft_warmup else None
        attn_bias = build_block_bias_from_topk(
            idx,
            L_ctx,
            chunk_len,
            soft_weights=w_soft,
            soft_alpha=self.soft_alpha,
            dtype=Q.dtype,
        )
        return logits, attn_bias, None
    
    def forward(
        self,
        Q: torch.Tensor,             # (B, q_len, qdim)
        K: torch.Tensor,             # (B, k_len, kdim)
        chunk_len: int,
        L_ctx: int,
        sink_blocks: int = 5,
        use_learnable_k: bool = False,                   # toggle learnable gating
    ):
        B, T, _ = Q.shape
        device = Q.device
        assert T > 0, "invalid T"

        logits = self.scorer(Q, K)                      # (B, T, J)
        B, T, J = logits.shape
        sink_blocks = int(min(max(0, sink_blocks), J))  # clamp sink_blocks ∈ [0, J]
        J_tail = J - sink_blocks

        if J_tail <= 0:
            return self._handle_all_sink(logits, Q, L_ctx, chunk_len)

        if not use_learnable_k:
            topk_main = max(min(J, self.topk) - sink_blocks, 0)

            if topk_main > 0:
                logits_tail = logits[..., sink_blocks:]
                idx_tail = topk_indices(logits_tail, k=topk_main)
                idx_tail = idx_tail + sink_blocks  # shift indices
            else:
                idx_tail = torch.empty(B, T, 0, dtype=torch.long, device=device)

            if sink_blocks > 0:
                sink_idx = torch.arange(sink_blocks, device=device).view(1, 1, sink_blocks)
                sink_idx = sink_idx.expand(B, T, sink_blocks)
                idx = torch.cat([sink_idx, idx_tail], dim=-1)
            else:
                idx = idx_tail

            w_soft = soft_selection_weights(logits, tau=float(self.tau.item())) if self.use_soft_warmup else None
            attn_bias = build_block_bias_from_topk(
                idx,
                L_ctx,
                chunk_len,
                soft_weights=w_soft,
                soft_alpha=self.soft_alpha,
                dtype=Q.dtype,
            )
            return logits, attn_bias, None

        # === learnable per-token k ===
        gate_logit = self.gate_head(Q).squeeze(-1)  # (B,T)
        gate_sig = torch.sigmoid(gate_logit)        # (B,T)

        # define range dynamically (e.g., [topk/2, topk*2])
        k_min = int(self.topk // 2)
        k_max = int(self.topk * 2)
        k_total = torch.round(k_min + gate_sig * (k_max - k_min)).clamp(1, J).to(torch.long)

        logits_tail = logits[..., sink_blocks:]
        sel_tail = select_tail_variable_k_ranks(logits_tail, k_total.clamp(max=J_tail))

        selected = torch.zeros((B, T, J), dtype=torch.bool, device=device)
        if sink_blocks > 0:
            selected[..., :sink_blocks] = True
        selected[..., sink_blocks:] = sel_tail

        w_soft = soft_selection_weights(logits, tau=float(self.tau.item())) if self.use_soft_warmup else None
        attn_bias = build_block_bias_from_selected(
            selected_chunks=selected,
            L_ctx=L_ctx,
            chunk_len=chunk_len,
            dtype=Q.dtype,
            soft_weights=w_soft,
            soft_alpha=self.soft_alpha,
        )

        return logits, attn_bias, gate_logit
