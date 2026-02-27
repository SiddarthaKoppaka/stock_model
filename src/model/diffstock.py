"""
DiffSTOCK: Top-level model combining MaTCHS + Adaptive DDPM

Full pipeline:
    1. MaTCHS: Extract temporal and cross-stock features
    2. AdaptiveDDPM: Generate probabilistic return predictions

Training: Computes diffusion loss
Inference: Samples multiple predictions and returns mean + uncertainty
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict
from .matches import MaTCHS
from .diffusion import AdaptiveDDPM


class DiffSTOCK(nn.Module):
    """
    DiffSTOCK: Diffusion-based Stock Prediction Model

    Complete architecture for probabilistic stock return forecasting.
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
        dropout: float = 0.25
    ):
        """
        Args:
            n_stocks: Number of stocks (N)
            in_features: Number of input features (F)
            d_model: Model dimension
            n_heads_mrt: Number of attention heads in MRT
            n_layers_dicem: Number of dilated conv layers
            n_layers_mrt: Number of transformer layers
            diffusion_T: Number of diffusion timesteps
            beta_start: Diffusion noise schedule start
            beta_end: Diffusion noise schedule end
            dropout: Dropout probability
        """
        super().__init__()

        self.n_stocks = n_stocks
        self.in_features = in_features
        self.d_model = d_model

        # Conditional encoder (MaTCHS)
        self.matches = MaTCHS(
            in_features=in_features,
            d_model=d_model,
            n_heads_mrt=n_heads_mrt,
            n_layers_dicem=n_layers_dicem,
            n_layers_mrt=n_layers_mrt,
            dropout=dropout
        )

        # Diffusion model
        self.diffusion = AdaptiveDDPM(
            n_stocks=n_stocks,
            d_model=d_model,
            T=diffusion_T,
            beta_start=beta_start,
            beta_end=beta_end
        )

    def forward(
        self,
        x: torch.Tensor,
        R_mask: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        n_samples: int = 50
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: (B, L, N, F) input features
            R_mask: (N, N) relation mask
            y: (B, N) target returns (only for training)
            n_samples: Number of samples for inference

        Returns:
            If training (y is not None):
                - loss: Scalar diffusion loss
            If inference (y is None):
                - predictions: (B, N) mean predictions
                - uncertainty: (B, N) prediction std
        """
        # Extract conditional embeddings
        condition = self.matches(x, R_mask)  # (B, N, d_model)

        if self.training and y is not None:
            # Training mode: compute loss
            loss = self.diffusion.compute_loss(y, condition)
            return loss, None

        else:
            # Inference mode: generate samples
            samples = self.diffusion.sample(condition, n_samples=n_samples)  # (n_samples, B, N)

            # Compute statistics
            predictions = samples.mean(dim=0)  # (B, N)
            uncertainty = samples.std(dim=0)  # (B, N)

            return predictions, uncertainty

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters in each component."""
        matches_params = self.matches.count_parameters()
        diffusion_params = sum(p.numel() for p in self.diffusion.parameters())

        return {
            'MaTCHS': {
                'Att-DiCEm': matches_params['att_dicem'],
                'MRT': matches_params['mrt'],
                'subtotal': matches_params['total']
            },
            'Diffusion': diffusion_params,
            'Total': matches_params['total'] + diffusion_params
        }

    def print_model_summary(self):
        """Print detailed model summary."""
        params = self.count_parameters()

        print("=" * 80)
        print("DiffSTOCK Model Parameters")
        print("=" * 80)
        print(f"MaTCHS (Conditional Encoder):")
        print(f"  - Att-DiCEm:  {params['MaTCHS']['Att-DiCEm']:>12,}")
        print(f"  - MRT:        {params['MaTCHS']['MRT']:>12,}")
        print(f"  - Subtotal:   {params['MaTCHS']['subtotal']:>12,}")
        print(f"\nDiffusion Model:")
        print(f"  - DDPM:       {params['Diffusion']:>12,}")
        print(f"\nTotal Parameters: {params['Total']:>12,}")
        print("=" * 80)

        # Estimate memory usage
        param_bytes = params['Total'] * 4  # Assuming float32
        param_mb = param_bytes / (1024 ** 2)
        print(f"Estimated memory (params only): {param_mb:.1f} MB")
        print("=" * 80)


def create_diffstock_model(config: Dict, n_stocks: int) -> DiffSTOCK:
    """
    Factory function to create DiffSTOCK model from config.

    Args:
        config: Configuration dictionary
        n_stocks: Number of stocks

    Returns:
        Initialized DiffSTOCK model
    """
    model = DiffSTOCK(
        n_stocks=n_stocks,
        in_features=config['data']['n_features'],
        d_model=config['model']['d_model'],
        n_heads_mrt=config['model']['n_heads_mrt'],
        n_layers_dicem=config['model']['n_layers_dicem'],
        n_layers_mrt=config['model']['n_layers_mrt'],
        diffusion_T=config['model']['diffusion_T'],
        beta_start=config['model']['beta_start'],
        beta_end=config['model']['beta_end'],
        dropout=config['model']['dropout']
    )

    return model


if __name__ == "__main__":
    # Test full model
    B, L, N, F = 32, 20, 400, 15
    d_model = 128

    model = DiffSTOCK(
        n_stocks=N,
        in_features=F,
        d_model=d_model,
        n_heads_mrt=8,
        n_layers_dicem=4,
        n_layers_mrt=3,
        diffusion_T=200
    )

    # Print model summary
    model.print_model_summary()

    # Create random data
    x = torch.randn(B, L, N, F)
    y = torch.randn(B, N) * 0.02
    R_mask = (torch.rand(N, N) > 0.5).float()
    R_mask.fill_diagonal_(0)

    # Training mode
    print("\nTraining mode:")
    model.train()
    loss, _ = model(x, R_mask, y)
    print(f"Loss: {loss.item():.4f}")

    # Inference mode
    print("\nInference mode:")
    model.eval()
    predictions, uncertainty = model(x, R_mask, n_samples=10)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Uncertainty shape: {uncertainty.shape}")
    print(f"Mean return: {predictions.mean():.4f}")
    print(f"Mean uncertainty: {uncertainty.mean():.4f}")

    print("\nDiffSTOCK test passed!")
