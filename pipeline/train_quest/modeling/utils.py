import torch
from typing import Tuple

def _kl_full_over_chunks(pt_chunks: torch.Tensor, logits_chunks: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    DeepSeek-style KL over the FULL chunk support:
        sum_t KL( pt_chunks[t,:] || softmax(logits_chunks[t,:]) )
    Shapes: pt_chunks, logits_chunks : (B, T_q, J)
    """
    pm_chunks = torch.softmax(logits_chunks, dim=-1)                 # (B,T_q,J)
    pt_chunks = pt_chunks / (pt_chunks.sum(dim=-1, keepdim=True) + eps)
    kl = (pt_chunks * (pt_chunks.add(eps).log() - pm_chunks.add(eps).log())).sum(dim=-1)  # (B,T_q)
    return kl.mean()

def _build_chunk_summaries(
    hidden: torch.Tensor,  # (B, T, D)
    chunk_len: int,
) -> Tuple[torch.Tensor, int]:
    assert chunk_len > 0
    B, T, D = hidden.shape
    J = (T + chunk_len - 1) // chunk_len
    pad_T = J * chunk_len - T

    if pad_T > 0:
        pad = hidden.new_zeros(B, pad_T, D)
        hidden_pad = torch.cat([hidden, pad], dim=1)            # (B, J*chunk_len, D)
        mask = torch.cat(
            [torch.ones(B, T, 1, device=hidden.device, dtype=torch.bool),
             torch.zeros(B, pad_T, 1, device=hidden.device, dtype=torch.bool)],
            dim=1
        )                                                        # (B, J*chunk_len, 1), bool
    else:
        hidden_pad = hidden
        mask = torch.ones(B, T, 1, device=hidden.device, dtype=torch.bool)

    # reshape into chunks
    hidden_chunks = hidden_pad.reshape(B, J, chunk_len, D)       # (B, J, chunk_len, D)
    mask_chunks   = mask.reshape(B, J, chunk_len, 1)             # (B, J, chunk_len, 1)

    # masked mean over chunk_len
    num = (hidden_chunks * mask_chunks.to(hidden.dtype)).sum(dim=2)                       # (B, J, D)
    den = mask_chunks.sum(dim=2, dtype=hidden.dtype).clamp_min(1e-6)       # (B, J, 1)
    K_chunks = num / den                                                                  # (B, J, D)

    return K_chunks, J