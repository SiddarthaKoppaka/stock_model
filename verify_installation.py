"""
Installation verification script.

Run this after pip install to ensure everything is set up correctly.

Usage:
    python verify_installation.py
"""

import sys
from pathlib import Path

def check_dependencies():
    """Check if all required packages are installed."""
    print("Checking dependencies...")

    required = [
        'torch', 'numpy', 'pandas', 'scipy', 'sklearn',
        'tqdm', 'loguru', 'yaml'
    ]

    optional = [
        'yfinance', 'pandas_ta', 'optuna', 'matplotlib'
    ]

    missing = []
    for pkg in required:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg} - MISSING")
            missing.append(pkg)

    print("\nOptional dependencies:")
    for pkg in optional:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ~ {pkg} - not installed (optional)")

    if missing:
        print(f"\n✗ Missing required packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False

    print("\n✓ All required dependencies installed")
    return True


def check_directory_structure():
    """Check if project directories exist."""
    print("\nChecking directory structure...")

    required_dirs = [
        'config',
        'data/raw',
        'data/processed',
        'data/dataset',
        'src/data',
        'src/model',
        'src/training',
        'src/evaluation',
        'src/utils',
        'scripts',
        'tests'
    ]

    missing = []
    for dir_path in required_dirs:
        full_path = Path(dir_path)
        if full_path.exists():
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path} - MISSING")
            missing.append(dir_path)

    if missing:
        print(f"\n✗ Missing directories: {', '.join(missing)}")
        return False

    print("\n✓ All directories present")
    return True


def check_config_file():
    """Check if config file exists and is valid."""
    print("\nChecking configuration...")

    config_path = Path('config/config.yaml')

    if not config_path.exists():
        print(f"  ✗ config/config.yaml not found")
        return False

    try:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)

        required_keys = ['data', 'model', 'training', 'evaluation', 'paths']
        missing = [k for k in required_keys if k not in config]

        if missing:
            print(f"  ✗ Missing config sections: {', '.join(missing)}")
            return False

        print(f"  ✓ config.yaml valid")
        print(f"    - Data period: {config['data']['start_date']} to {config['data']['end_date']}")
        print(f"    - Model: d_model={config['model']['d_model']}, T={config['model']['diffusion_T']}")
        print(f"    - Training: {config['training']['max_epochs']} epochs, batch_size={config['training']['batch_size']}")

        return True

    except Exception as e:
        print(f"  ✗ Error loading config: {e}")
        return False


def check_model_import():
    """Check if model modules can be imported."""
    print("\nChecking model imports...")

    try:
        from src.model.att_dicem import AttDiCEm
        print("  ✓ AttDiCEm")

        from src.model.mrt import MaskedRelationalTransformer
        print("  ✓ MaskedRelationalTransformer")

        from src.model.matches import MaTCHS
        print("  ✓ MaTCHS")

        from src.model.diffusion import AdaptiveDDPM
        print("  ✓ AdaptiveDDPM")

        from src.model.diffstock import DiffSTOCK
        print("  ✓ DiffSTOCK")

        print("\n✓ All model modules imported successfully")
        return True

    except Exception as e:
        print(f"\n✗ Error importing models: {e}")
        return False


def run_quick_model_test():
    """Run a quick forward pass test."""
    print("\nRunning quick model test...")

    try:
        import torch
        from src.model.diffstock import DiffSTOCK

        # Small model for quick test
        B, L, N, F = 4, 20, 50, 15
        model = DiffSTOCK(n_stocks=N, in_features=F, d_model=32, diffusion_T=50)

        x = torch.randn(B, L, N, F)
        y = torch.randn(B, N) * 0.02
        R_mask = (torch.rand(N, N) > 0.5).float()

        # Training forward pass
        model.train()
        loss, _ = model(x, R_mask, y)

        print(f"  ✓ Training forward pass: loss = {loss.item():.4f}")

        # Inference forward pass
        model.eval()
        with torch.no_grad():
            pred, unc = model(x, R_mask, n_samples=5)

        print(f"  ✓ Inference forward pass: pred shape = {pred.shape}")

        # Count parameters
        params = model.count_parameters()
        print(f"  ✓ Model has {params['Total']:,} parameters")

        print("\n✓ Model test passed")
        return True

    except Exception as e:
        print(f"\n✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification checks."""
    print("=" * 80)
    print("DiffSTOCK India - Installation Verification")
    print("=" * 80)

    checks = [
        ("Dependencies", check_dependencies),
        ("Directory Structure", check_directory_structure),
        ("Configuration", check_config_file),
        ("Model Imports", check_model_import),
        ("Model Test", run_quick_model_test)
    ]

    results = {}
    for name, check_func in checks:
        results[name] = check_func()
        print()

    print("=" * 80)
    print("Verification Summary")
    print("=" * 80)

    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:.<50} {status}")
        all_passed = all_passed and passed

    print("=" * 80)

    if all_passed:
        print("\n🎉 Installation verified successfully!")
        print("\nNext steps:")
        print("  1. Read QUICKSTART.md for usage instructions")
        print("  2. Run: python scripts/run_scrape.py (to download data)")
        print("  3. Or try quick demo with synthetic data")
        print("\nFor detailed documentation, see README.md")
    else:
        print("\n⚠️  Some checks failed. Please review errors above.")
        print("Try: pip install -r requirements.txt")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
