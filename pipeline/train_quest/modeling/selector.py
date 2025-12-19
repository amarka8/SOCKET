import torch
import torch.nn as nn
import torch.nn.functional as F


class FullAttentionEstimator(nn.Module):
    """
    Lightweight estimator that predicts per-token attention scores (B, T_q, T_k).
    """

    def __init__(self, q_dim: int, k_dim: int, hidden: int):
        super().__init__()
        self.q_proj = nn.Linear(q_dim, hidden, bias=False)
        self.k_proj = nn.Linear(k_dim, hidden, bias=False)
        self.ln = nn.LayerNorm(3 * hidden)
        self.fc_u = nn.Linear(3 * hidden, hidden)
        self.fc_v = nn.Linear(3 * hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Q: Tensor of shape (B, T_q, q_dim) containing per-token query states.
            K: Tensor of shape (B, T_k, k_dim) containing key states to compare against.
        Returns:
            Tensor of shape (B, T_q, T_k) with unnormalized attention logits.
        """
        B, T_q, _ = Q.shape
        _, T_k, _ = K.shape

        q = self.q_proj(Q).unsqueeze(2)  # (B,T_q,1,H)
        k = self.k_proj(K).unsqueeze(1)  # (B,1,T_k,H)

        # Broadcast to (B, T_q, T_k, H)
        q_exp = q.expand(-1, -1, T_k, -1)
        k_exp = k.expand(-1, T_q, -1, -1)
        x = torch.cat([q_exp, k_exp, q_exp * k_exp], dim=-1)
        x = self.ln(x)
        u = self.fc_u(x)
        v = self.fc_v(x)
        z = u * F.gelu(v)
        return self.out(z).squeeze(-1)
