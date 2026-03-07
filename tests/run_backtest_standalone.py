"""
Standalone runner for tests/backtest.py.

Loads pre-computed predictions from the checkpoint run's test_results.npz,
sets up the variables expected by backtest.py, then executes it.

Usage:
    python tests/run_backtest_standalone.py [--run RUN_DIR]
"""

import sys
import argparse
import matplotlib
matplotlib.use('Agg')  # non-interactive backend (no GUI window needed)
import numpy as np
import yaml
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--run',
        type=str,
        default=None,
        help='Run directory under checkpoints/ (auto-detected if omitted)'
    )
    args = parser.parse_args()

    # ── Locate run directory ───────────────────────────────────────────────
    ckpt_root = ROOT / 'checkpoints'
    if args.run:
        run_dir = ckpt_root / args.run
    else:
        # Pick the most-recently modified run that has final_model.pt
        runs = sorted(
            [d for d in ckpt_root.iterdir()
             if d.is_dir() and (d / 'checkpoints' / 'final_model.pt').exists()],
            key=lambda d: d.stat().st_mtime
        )
        if not runs:
            print("ERROR: No run directory with checkpoints/final_model.pt found.")
            sys.exit(1)
        run_dir = runs[-1]

    print(f"Using run: {run_dir}")
    test_results_path = run_dir / 'test_results.npz'
    if not test_results_path.exists():
        print(f"ERROR: {test_results_path} not found. Run inference first.")
        sys.exit(1)

    # ── Load config ────────────────────────────────────────────────────────
    with open(ROOT / 'config' / 'config.yaml') as f:
        config = yaml.safe_load(f)

    # ── Load predictions ───────────────────────────────────────────────────
    res = np.load(test_results_path, allow_pickle=True)
    test_predictions = res['predictions']   # (T, N)
    test_targets     = res['targets']       # (T, N)
    print(f"Predictions: {test_predictions.shape}, Targets: {test_targets.shape}")

    # ── Load dataset (for dates_test) ──────────────────────────────────────
    dataset_path = ROOT / 'data' / 'dataset' / 'nifty500_20yr.npz'
    data = dict(np.load(dataset_path, allow_pickle=True))

    # ── Override RESULTS_DIR to local ─────────────────────────────────────
    results_dir = str(run_dir / 'aligned_backtest')

    # ── Execute backtest.py in this namespace ──────────────────────────────
    backtest_src = (ROOT / 'tests' / 'backtest.py').read_text()
    # Patch the hardcoded Colab path
    backtest_src = backtest_src.replace(
        '"/content/drive/MyDrive/DiffSTOCK_Outputs/aligned_backtest"',
        repr(results_dir)
    )

    ns = {
        'test_predictions': test_predictions,
        'test_targets':     test_targets,
        'data':             data,
        'config':           config,
    }
    exec(compile(backtest_src, 'backtest.py', 'exec'), ns)


if __name__ == '__main__':
    main()
