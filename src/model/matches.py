"""
MaTCHS: Masked Temporal-Cross-stock Historical Signal Encoder

Combines Att-DiCEm (temporal) + MRT (cross-stock) to produce rich
conditional embeddings for diffusion model.

Input: (B, L, N, F) features and (N, N) relation mask
Output: (B, N, d_model) conditional embeddings
"""

import torch
import torch.nn as nn
from .att_dicem import AttDiCEm
from .mrt import MaskedRelationalTransformer


class MaTCHS(nn.Module):
    """
    MaTCHS: Complete conditional encoder for DiffSTOCK.

    Pipeline:
        1. Att-DiCEm: Extract temporal features per stock
        2. MRT: Model cross-stock relationships
        3. Output: Rich conditional embeddings
    """

    def __init__(
        self,
        in_features: int,
        d_model: int,
        n_heads_mrt: int = 8,
        n_layers_dicem: int = 4,
        n_layers_mrt: int = 3,
        dropout: float = 0.25
    ):
        """
        Args:
            in_features: Number of input features (F)
            d_model: Model dimension
            n_heads_mrt: Number of attention heads in MRT
            n_layers_dicem: Number of dilated conv layers in Att-DiCEm
            n_layers_mrt: Number of transformer layers in MRT
            dropout: Dropout probability
        """
        super().__init__()

        self.in_features = in_features
        self.d_model = d_model

        # Temporal encoder
        self.att_dicem = AttDiCEm(
            in_features=in_features,
            d_model=d_model,
            n_layers=n_layers_dicem,
            dropout=dropout
        )

        # Cross-stock relational encoder
        self.mrt = MaskedRelationalTransformer(
            d_model=d_model,
            n_heads=n_heads_mrt,
            n_layers=n_layers_mrt,
            dropout=dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        R_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, L, N, F) input features
               B = batch size
               L = lookback window length
               N = number of stocks
               F = number of features

            R_mask: (N, N) relation mask

        Returns:
            (B, N, d_model) conditional embeddings
        """
        B, L, N, F = x.shape

        # Permute to (B, N, L, F) for Att-DiCEm
        x = x.permute(0, 2, 1, 3)  # (B, N, L, F)

        # Extract temporal features
        h = self.att_dicem(x)  # (B, N, d_model)

        # Model cross-stock relationships
        h = self.mrt(h, R_mask)  # (B, N, d_model)

        return h

    def count_parameters(self) -> dict:
        """Count parameters in each component."""
        att_dicem_params = sum(p.numel() for p in self.att_dicem.parameters())
        mrt_params = sum(p.numel() for p in self.mrt.parameters())
        total_params = att_dicem_params + mrt_params

        return {
            'att_dicem': att_dicem_params,
            'mrt': mrt_params,
            'total': total_params
        }


if __name__ == "__main__":
    # Test module
    B, L, N, F = 32, 20, 400, 15
    d_model = 128

    model = MaTCHS(
        in_features=F,
        d_model=d_model,
        n_heads_mrt=8,
        n_layers_dicem=4,
        n_layers_mrt=3
    )

    # Random input
    x = torch.randn(B, L, N, F)

    # Random relation mask
    R_mask = (torch.rand(N, N) > 0.5).float()
    R_mask.fill_diagonal_(0)

    # Forward pass
    out = model(x, R_mask)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Expected: ({B}, {N}, {d_model})")

    # Count parameters
    params = model.count_parameters()
    print(f"\nParameter counts:")
    print(f"  Att-DiCEm: {params['att_dicem']:,}")
    print(f"  MRT: {params['mrt']:,}")
    print(f"  Total: {params['total']:,}")

    assert out.shape == (B, N, d_model), f"Shape mismatch! Got {out.shape}"
    print("\nMaTCHS test passed!")
