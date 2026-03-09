"""
Regime-Aware Soft Mixture of Experts (MoE) Head

Routes the MaTCHS output through 3 specialised expert FFNs. A gating network
takes the combined (stock embedding + regime embedding) as input and produces
soft weights via softmax — no hard routing, all experts are always active.

A load-balancing auxiliary loss prevents the model from collapsing to a single
expert. This loss is returned alongside the output and added to the diffusion
training loss in the trainer.

Reference: Shazeer et al. (2017) "Outrageously Large Neural Networks: The
Sparsely-Gated Mixture-of-Experts Layer"
Switch Transformer load-balance: Fedus et al. (2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegimeAwareMoE(nn.Module):
    """
    Soft Mixture of Experts conditioned on regime embedding.

    Each expert is a 2-layer FFN with LayerNorm + GELU:
        Linear(d_model, expert_hidden) → LayerNorm → GELU
        → Linear(expert_hidden, d_model) → LayerNorm

    Gating network:
        concat(stock_emb, regime_emb) → Linear → GELU → Linear → Softmax

    Args:
        d_model:            stock embedding dimension (192)
        n_experts:          number of expert FFNs (3)
        expert_hidden_dim:  hidden dim of each expert (384 = 2 × d_model)
        regime_embed_dim:   dimension of regime embedding (16)
        load_balance_coef:  weight of auxiliary load-balance loss (0.01)
    """

    def __init__(
        self,
        d_model: int = 192,
        n_experts: int = 3,
        expert_hidden_dim: int = 384,
        regime_embed_dim: int = 16,
        load_balance_coef: float = 0.01,
    ):
        super().__init__()
        self.n_experts         = n_experts
        self.load_balance_coef = load_balance_coef

        # Expert FFNs
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, expert_hidden_dim),
                nn.LayerNorm(expert_hidden_dim),
                nn.GELU(),
                nn.Linear(expert_hidden_dim, d_model),
                nn.LayerNorm(d_model),
            )
            for _ in range(n_experts)
        ])

        # Gating network: (stock_emb + regime_emb) → soft weights over experts
        self.gate = nn.Sequential(
            nn.Linear(d_model + regime_embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, n_experts),
        )

    def forward(
        self,
        x: torch.Tensor,
        regime_embedding: torch.Tensor,
    ):
        """
        Args:
            x:                (B, N, d_model)  stock embeddings from MRT
            regime_embedding: (B, regime_embed_dim)  from RegimeEmbedding

        Returns:
            output:            (B, N, d_model)  mixture output
            load_balance_loss: scalar auxiliary loss
        """
        B, N, _ = x.shape

        # Broadcast regime embedding to all stocks
        r = regime_embedding.unsqueeze(1).expand(B, N, -1)   # (B, N, regime_dim)
        gate_input = torch.cat([x, r], dim=-1)                # (B, N, d_model+regime_dim)

        # Soft gate weights
        gate_logits  = self.gate(gate_input)                  # (B, N, n_experts)
        gate_weights = F.softmax(gate_logits, dim=-1)         # (B, N, n_experts)

        # Expert outputs stacked: (B, N, n_experts, d_model)
        expert_outs = torch.stack([e(x) for e in self.experts], dim=2)

        # Weighted mixture
        output = (gate_weights.unsqueeze(-1) * expert_outs).sum(dim=2)  # (B, N, d_model)

        # Load-balance loss — penalise unequal expert utilisation
        lb_loss = self._load_balance_loss(gate_weights)

        return output, lb_loss

    def _load_balance_loss(self, gate_weights: torch.Tensor) -> torch.Tensor:
        """
        Encourage uniform utilisation across experts (Switch Transformer style).

        gate_weights: (B, N, n_experts)
        Returns scalar.
        """
        mean_weights = gate_weights.mean(dim=[0, 1])                        # (n_experts,)
        target       = torch.full_like(mean_weights, 1.0 / self.n_experts)
        return F.mse_loss(mean_weights, target) * self.load_balance_coef
