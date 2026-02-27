"""
Test script to verify model architecture and tensor shapes.

Usage:
    pytest tests/test_model_shapes.py -v
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.att_dicem import AttDiCEm
from src.model.mrt import MaskedRelationalTransformer
from src.model.matches import MaTCHS
from src.model.diffusion import AdaptiveDDPM
from src.model.diffstock import DiffSTOCK


def test_att_dicem():
    """Test Att-DiCEm shape."""
    B, N, L, F = 8, 100, 20, 15
    d_model = 64

    model = AttDiCEm(in_features=F, d_model=d_model, n_layers=4)
    x = torch.randn(B, N, L, F)
    out = model(x)

    assert out.shape == (B, N, d_model), f"Expected {(B, N, d_model)}, got {out.shape}"
    print(f"✓ Att-DiCEm: {x.shape} -> {out.shape}")


def test_mrt():
    """Test MRT shape."""
    B, N, d_model = 8, 100, 64

    model = MaskedRelationalTransformer(d_model=d_model, n_heads=4, n_layers=2)
    x = torch.randn(B, N, d_model)
    R_mask = (torch.rand(N, N) > 0.5).float()

    out = model(x, R_mask)

    assert out.shape == (B, N, d_model), f"Expected {(B, N, d_model)}, got {out.shape}"
    print(f"✓ MRT: {x.shape} -> {out.shape}")


def test_matches():
    """Test MaTCHS shape."""
    B, L, N, F = 8, 20, 100, 15
    d_model = 64

    model = MaTCHS(in_features=F, d_model=d_model, n_heads_mrt=4, n_layers_dicem=3, n_layers_mrt=2)
    x = torch.randn(B, L, N, F)
    R_mask = (torch.rand(N, N) > 0.5).float()

    out = model(x, R_mask)

    assert out.shape == (B, N, d_model), f"Expected {(B, N, d_model)}, got {out.shape}"
    print(f"✓ MaTCHS: {x.shape} -> {out.shape}")


def test_diffusion():
    """Test Adaptive DDPM."""
    B, N, d_model = 8, 100, 64

    model = AdaptiveDDPM(n_stocks=N, d_model=d_model, T=100)

    x_0 = torch.randn(B, N) * 0.02
    condition = torch.randn(B, N, d_model)

    # Training loss
    loss = model.compute_loss(x_0, condition)
    assert loss.item() > 0, "Loss should be positive"
    print(f"✓ DDPM loss: {loss.item():.4f}")

    # Sampling
    samples = model.sample(condition, n_samples=5)
    assert samples.shape == (5, B, N), f"Expected {(5, B, N)}, got {samples.shape}"
    print(f"✓ DDPM sampling: {samples.shape}")


def test_diffstock():
    """Test full DiffSTOCK model."""
    B, L, N, F = 8, 20, 100, 15
    d_model = 64

    model = DiffSTOCK(
        n_stocks=N,
        in_features=F,
        d_model=d_model,
        n_heads_mrt=4,
        n_layers_dicem=3,
        n_layers_mrt=2,
        diffusion_T=100
    )

    x = torch.randn(B, L, N, F)
    y = torch.randn(B, N) * 0.02
    R_mask = (torch.rand(N, N) > 0.5).float()

    # Training mode
    model.train()
    loss, _ = model(x, R_mask, y)
    assert loss.item() > 0, "Loss should be positive"
    print(f"✓ DiffSTOCK training loss: {loss.item():.4f}")

    # Inference mode
    model.eval()
    predictions, uncertainty = model(x, R_mask, n_samples=5)
    assert predictions.shape == (B, N), f"Expected {(B, N)}, got {predictions.shape}"
    assert uncertainty.shape == (B, N), f"Expected {(B, N)}, got {uncertainty.shape}"
    print(f"✓ DiffSTOCK inference: predictions {predictions.shape}, uncertainty {uncertainty.shape}")


def test_parameter_count():
    """Test parameter counting."""
    N, F = 400, 15
    model = DiffSTOCK(n_stocks=N, in_features=F, d_model=128)

    params = model.count_parameters()
    total = params['Total']

    assert total > 1_000_000, f"Model should have >1M params, got {total:,}"
    print(f"✓ Total parameters: {total:,}")


def run_all_tests():
    """Run all shape tests."""
    print("=" * 80)
    print("DiffSTOCK Model Shape Tests")
    print("=" * 80)

    try:
        test_att_dicem()
        test_mrt()
        test_matches()
        test_diffusion()
        test_diffstock()
        test_parameter_count()

        print("=" * 80)
        print("✓ All tests passed!")
        print("=" * 80)

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
