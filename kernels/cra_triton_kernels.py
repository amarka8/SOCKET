import triton
import triton.language as tl
import torch 
from typing import Dict, Optional, Tuple
import torch.nn.functional as F

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