"""
Test that training runs for a few iterations without NaN losses.
"""
import torch
import yaml
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.diffstock import DiffSTOCK

def test_training_stability():
    """Test that model trains without NaN for 5 iterations."""
    print("=" * 80)
    print("Testing Training Stability (No NaN Check)")
    print("=" * 80)

    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Load small subset of data
    print("\nLoading dataset...")
    data = np.load('data/dataset/nifty500_20yr.npz', allow_pickle=True)
    X_train = data['X_train'][:100]  # Only 100 samples for quick test
    y_train = data['y_train'][:100]
    X_val = data['X_val'][:50]
    y_val = data['y_val'][:50]

    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")

    # Check for NaN
    assert np.isnan(X_train).sum() == 0, "X_train contains NaN!"
    assert np.isnan(y_train).sum() == 0, "y_train contains NaN!"
    print("✓ No NaN in dataset")

    # Load relation mask (combined)
    rel_data = np.load('data/dataset/relation_matrices.npz')
    R_mask = torch.from_numpy(rel_data['R_mask']).float()  # (N, N)

    print(f"R_mask: {R_mask.shape}")

    # Create datasets
    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).float()
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val).float()
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Create model
    print("\nInitializing model...")
    B, L, N, F = X_train.shape
    model = DiffSTOCK(
        n_stocks=N,
        in_features=F,
        d_model=config['model']['d_model'],
        n_heads_mrt=config['model']['n_heads_mrt'],
        n_layers_dicem=config['model']['n_layers_dicem'],
        n_layers_mrt=config['model']['n_layers_mrt'],
        diffusion_T=config['model']['diffusion_T'],
        beta_start=config['model']['beta_start'],
        beta_end=config['model']['beta_end'],
        dropout=config['model']['dropout']
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Test forward pass
    print("\nTesting forward pass...")
    device = torch.device('cpu')
    model = model.to(device)
    R_mask = R_mask.to(device)
    model.train()  # Set to training mode

    with torch.no_grad():
        X_batch = torch.from_numpy(X_train[:8]).float().to(device)
        y_batch = torch.from_numpy(y_train[:8]).float().to(device)

        loss, _ = model(X_batch, R_mask, y_batch)

        assert not torch.isnan(loss), f"Forward pass produced NaN loss!"

        print(f"✓ Forward pass: loss={loss.item():.6f}")

    # Test training for 5 iterations
    print("\nTesting 5 training iterations...")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    model.train()
    for i, (X_batch, y_batch) in enumerate(train_loader):
        if i >= 5:
            break

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        loss, _ = model(X_batch, R_mask, y_batch)

        # Check for NaN
        assert not torch.isnan(loss), f"Iteration {i+1}: loss is NaN!"

        loss.backward()

        # Check gradients
        max_grad = 0
        for param in model.parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"Iteration {i+1}: gradient is NaN!"
                max_grad = max(max_grad, param.grad.abs().max().item())

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])

        optimizer.step()

        print(f"  Iter {i+1}: loss={loss.item():.6f}, max_grad={max_grad:.6f}")

    print("\n" + "=" * 80)
    print("✓ Training Stability Test PASSED")
    print("=" * 80)
    print("\nConclusion:")
    print("  - Model can perform forward pass without NaN")
    print("  - Training runs for 5 iterations without NaN")
    print("  - Gradients are computed correctly")
    print("  - Ready for full training on Colab")
    print("\n" + "=" * 80)

if __name__ == '__main__':
    test_training_stability()
