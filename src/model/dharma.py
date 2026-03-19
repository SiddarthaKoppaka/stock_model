"""
DHARMA: Top-level model combining MaTCHS + Adaptive DDPM

Full pipeline:
    1. MaTCHS: Extract temporal and cross-stock features
       (optionally with RevIN, regime embedding, MoE)
    2. AdaptiveDDPM: Generate probabilistic return predictions

Training: Computes diffusion loss + optional MoE load-balance loss
Inference: Samples multiple predictions and returns mean + uncertainty
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict
from .matches import MaTCHS
from .diffusion import AdaptiveDDPM


class DHARMA(nn.Module):
    """
    DHARMA: Diffusion-based Stock Prediction Model
    """

    def __init__(
        self,
        n_stocks: int,
        in_features: int,
        d_model: int = 128,
        n_heads_mrt: int = 8,
        n_layers_dicem: int = 4,
        n_layers_mrt: int = 3,
        diffusion_T: int = 200,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        dropout: float = 0.25,
        # ── ablation flags ────────────────────────────────────────────────────
        use_revin: bool = False,
        use_regime: bool = False,
        use_moe: bool = False,
        # ── regime / MoE hyper-params ─────────────────────────────────────────
        n_regime_states: int = 4,
        regime_embedding_dim: int = 32,
        n_experts: int = 3,
        expert_hidden_dim: int = 384,
        moe_load_balance_coef: float = 0.01,
    ):
        super().__init__()

        self.n_stocks    = n_stocks
        self.in_features = in_features
        self.d_model     = d_model

        # Conditional encoder (MaTCHS)
        self.matches = MaTCHS(
            in_features=in_features,
            d_model=d_model,
            n_heads_mrt=n_heads_mrt,
            n_layers_dicem=n_layers_dicem,
            n_layers_mrt=n_layers_mrt,
            dropout=dropout,
            use_revin=use_revin,
            use_regime=use_regime,
            use_moe=use_moe,
            n_regime_states=n_regime_states,
            regime_embedding_dim=regime_embedding_dim,
            n_experts=n_experts,
            expert_hidden_dim=expert_hidden_dim,
            moe_load_balance_coef=moe_load_balance_coef,
        )

        # Diffusion head
        self.diffusion = AdaptiveDDPM(
            n_stocks=n_stocks,
            d_model=d_model,
            T=diffusion_T,
            beta_start=beta_start,
            beta_end=beta_end,
        )

    def forward(
        self,
        x: torch.Tensor,
        R_mask: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        n_samples: int = 50,
        regime_probs: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x:            (B, L, N, F) input features
            R_mask:       (N, N) relation mask
            y:            (B, N) target returns — training only
            n_samples:    diffusion samples for inference
            regime_probs: (B, n_states) soft HMM regime probabilities (optional)
            valid_mask:   (B, N) bool — True where label is MISSING; gates diffusion loss

        Returns:
            Training (y not None): (loss, None)
            Inference (y is None): (predictions (B,N), uncertainty (B,N))
        """
        condition, lb_loss = self.matches(x, R_mask, regime_probs)

        if self.training and y is not None:
            diff_loss = self.diffusion.compute_loss(y, condition, valid_mask=valid_mask)
            return diff_loss + lb_loss, None

        else:
            samples      = self.diffusion.sample(condition, n_samples=n_samples)
            predictions  = samples.mean(dim=0)   # (B, N)
            uncertainty  = samples.std(dim=0)    # (B, N)
            return predictions, uncertainty

    def count_parameters(self) -> Dict[str, int]:
        matches_params   = self.matches.count_parameters()
        diffusion_params = sum(p.numel() for p in self.diffusion.parameters())
        return {
            'MaTCHS':    matches_params,
            'Diffusion': diffusion_params,
            'Total':     matches_params['total'] + diffusion_params,
        }

    def print_model_summary(self):
        params = self.count_parameters()
        print("=" * 80)
        print("DHARMA Model Parameters")
        print("=" * 80)
        print(f"MaTCHS (Conditional Encoder):")
        print(f"  - Att-DiCEm:  {params['MaTCHS']['att_dicem']:>12,}")
        print(f"  - MRT:        {params['MaTCHS']['mrt']:>12,}")
        print(f"  - Extras:     {params['MaTCHS']['extras']:>12,}")
        print(f"  - Subtotal:   {params['MaTCHS']['total']:>12,}")
        print(f"\nDiffusion Model:            {params['Diffusion']:>12,}")
        print(f"\nTotal Parameters:           {params['Total']:>12,}")
        param_mb = params['Total'] * 4 / 1024 ** 2
        print(f"Estimated memory (fp32):    {param_mb:.1f} MB")
        print("=" * 80)


def create_dharma_model(config: Dict, n_stocks: int) -> DHARMA:
    """
    Factory function — creates DHARMA from config dict.

    Backward compatible: use_revin / use_regime / use_moe default to False
    if not present in config, preserving old checkpoint behaviour.
    """
    mc = config['model']
    moe_cfg = mc.get('moe', {})
    revin_cfg = config.get('revin', {})
    hmm_cfg = config.get('hmm', {})
    return DHARMA(
        n_stocks=n_stocks,
        in_features=config['data']['n_features'],
        d_model=mc['d_model'],
        n_heads_mrt=mc['n_heads_mrt'],
        n_layers_dicem=mc['n_layers_dicem'],
        n_layers_mrt=mc['n_layers_mrt'],
        diffusion_T=mc['diffusion_T'],
        beta_start=mc['beta_start'],
        beta_end=mc['beta_end'],
        dropout=mc['dropout'],
        # ablation flags — now in their own top-level sections
        use_revin=revin_cfg.get('enabled', False),
        use_regime=hmm_cfg.get('enabled', False),
        use_moe=moe_cfg.get('enabled', False),
        # regime params (from hmm section)
        n_regime_states=hmm_cfg.get('n_regimes', 4),
        regime_embedding_dim=hmm_cfg.get('regime_embed_dim', 32),
        # MoE params (from model.moe section)
        n_experts=moe_cfg.get('n_experts', 3),
        expert_hidden_dim=moe_cfg.get('expert_hidden_dim', 256),
        moe_load_balance_coef=moe_cfg.get('load_balance_coef', 0.01),
    )
