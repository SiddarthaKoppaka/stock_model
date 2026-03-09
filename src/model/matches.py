"""
MaTCHS: Masked Temporal-Cross-stock Historical Signal Encoder

Combines Att-DiCEm (temporal) + MRT (cross-stock) to produce rich
conditional embeddings for diffusion model.

With regime-aware extensions (all backward-compatible via config flags):
  use_revin  — RevIN per-sample normalisation before Att-DiCEm
  use_regime — RegimeEmbedding concatenated before MRT
  use_moe    — RegimeAwareMoE after MRT instead of identity

Input:  (B, L, N, F) features  +  (N, N) relation mask
        optional: (B, n_states) regime_probs
Output: (B, N, d_model) conditional embeddings
        optional: scalar load_balance_loss (non-zero when use_moe=True)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .att_dicem import AttDiCEm
from .mrt import MaskedRelationalTransformer
from .revin import RevIN
from .regime_embedding import RegimeEmbedding
from .moe_head import RegimeAwareMoE


class MaTCHS(nn.Module):
    """
    MaTCHS: Complete conditional encoder for DiffSTOCK.

    Pipeline:
        1. [optional] RevIN normalise per-sample window
        2. Att-DiCEm: temporal features per stock
        3. [optional] Append regime embedding → project back to d_model
        4. MRT: cross-stock relational attention
        5. [optional] MoE: regime-specialised expert mixture
        6. Output: (B, N, d_model) embeddings
    """

    def __init__(
        self,
        in_features: int,
        d_model: int,
        n_heads_mrt: int = 8,
        n_layers_dicem: int = 4,
        n_layers_mrt: int = 3,
        dropout: float = 0.25,
        # ── ablation flags ────────────────────────────────────────────────────
        use_revin: bool = False,
        use_regime: bool = False,
        use_moe: bool = False,
        # ── regime / MoE hyper-params ─────────────────────────────────────────
        n_regime_states: int = 4,
        regime_embedding_dim: int = 16,
        n_experts: int = 3,
        expert_hidden_dim: int = 384,
        moe_load_balance_coef: float = 0.01,
    ):
        super().__init__()

        self.in_features = in_features
        self.d_model     = d_model
        self.use_revin   = use_revin
        self.use_regime  = use_regime
        self.use_moe     = use_moe

        # ── RevIN ─────────────────────────────────────────────────────────────
        if use_revin:
            self.revin = RevIN(n_features=in_features, affine=True)

        # ── Temporal encoder ──────────────────────────────────────────────────
        self.att_dicem = AttDiCEm(
            in_features=in_features,
            d_model=d_model,
            n_layers=n_layers_dicem,
            dropout=dropout,
        )

        # ── Regime embedding + projection ─────────────────────────────────────
        if use_regime:
            self.regime_embed = RegimeEmbedding(
                n_states=n_regime_states,
                embedding_dim=regime_embedding_dim,
            )
            # Projects (d_model + regime_dim) back to d_model before MRT
            self.regime_proj = nn.Linear(d_model + regime_embedding_dim, d_model)

        # ── Cross-stock relational encoder ────────────────────────────────────
        self.mrt = MaskedRelationalTransformer(
            d_model=d_model,
            n_heads=n_heads_mrt,
            n_layers=n_layers_mrt,
            dropout=dropout,
        )

        # ── MoE head ──────────────────────────────────────────────────────────
        if use_moe:
            self.moe = RegimeAwareMoE(
                d_model=d_model,
                n_experts=n_experts,
                expert_hidden_dim=expert_hidden_dim,
                regime_embed_dim=regime_embedding_dim if use_regime else 1,
                load_balance_coef=moe_load_balance_coef,
            )
            # If regime is disabled, MoE needs its own regime input placeholder
            if not use_regime:
                self.regime_embed = RegimeEmbedding(
                    n_states=n_regime_states,
                    embedding_dim=1,
                )

    def forward(
        self,
        x: torch.Tensor,
        R_mask: torch.Tensor,
        regime_probs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x:            (B, L, N, F) input features (raw if use_revin, else pre-normalised)
            R_mask:       (N, N) relation mask
            regime_probs: (B, n_states) soft HMM regime probabilities (optional)

        Returns:
            h:                (B, N, d_model) conditional embeddings
            load_balance_loss: scalar (0 when use_moe=False)
        """
        B, L, N, F = x.shape
        load_balance_loss = torch.tensor(0.0, device=x.device)

        # Permute to (B, N, L, F) for Att-DiCEm
        x = x.permute(0, 2, 1, 3)  # (B, N, L, F)

        # 1. RevIN normalisation per sample
        if self.use_revin:
            x = self.revin.normalize(x)

        # 2. Temporal encoding: (B, N, L, F) → (B, N, d_model)
        h = self.att_dicem(x)

        # 3. Regime embedding concatenation
        if self.use_regime and regime_probs is not None:
            r_emb = self.regime_embed(regime_probs)           # (B, regime_dim)
            r_exp = r_emb.unsqueeze(1).expand(B, N, -1)       # (B, N, regime_dim)
            h = self.regime_proj(torch.cat([h, r_exp], dim=-1))  # (B, N, d_model)

        # 4. Cross-stock relational attention
        h = self.mrt(h, R_mask)  # (B, N, d_model)

        # 5. MoE routing
        if self.use_moe:
            if regime_probs is not None:
                r_emb_moe = self.regime_embed(regime_probs) if not self.use_regime \
                            else self.regime_embed(regime_probs)
            else:
                # Uniform fallback if regime_probs not provided
                r_emb_moe = torch.zeros(B, self.moe.gate[0].in_features - self.d_model,
                                        device=x.device)
            h, load_balance_loss = self.moe(h, r_emb_moe)

        return h, load_balance_loss

    def count_parameters(self) -> dict:
        att_dicem_params = sum(p.numel() for p in self.att_dicem.parameters())
        mrt_params       = sum(p.numel() for p in self.mrt.parameters())
        extra_params     = 0
        if self.use_revin:
            extra_params += sum(p.numel() for p in self.revin.parameters())
        if self.use_regime:
            extra_params += sum(p.numel() for p in self.regime_embed.parameters())
            extra_params += sum(p.numel() for p in self.regime_proj.parameters())
        if self.use_moe:
            extra_params += sum(p.numel() for p in self.moe.parameters())

        return {
            'att_dicem': att_dicem_params,
            'mrt':       mrt_params,
            'extras':    extra_params,
            'total':     att_dicem_params + mrt_params + extra_params,
        }
