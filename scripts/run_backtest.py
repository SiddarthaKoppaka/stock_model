"""
Entry point for running backtest on trained model.

Usage:
    python scripts/run_backtest.py [--checkpoint path/to/model.pt]
"""

import yaml
import argparse
import sys
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger
from src.model.diffstock import create_diffstock_model
from src.evaluation.backtester import IndianMarketBacktester


def main():
    parser = argparse.ArgumentParser(description='Backtest DiffSTOCK model')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/best_model.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['val', 'test'],
        help='Which split to backtest on'
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(__file__).parent.parent / args.config

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup logging
    setup_logger(
        log_dir=Path(config['paths']['root']) / config['paths']['logs'],
        log_level='INFO'
    )

    print("=" * 80)
    print("DiffSTOCK Backtest")
    print("=" * 80)

    # Load dataset
    dataset_path = Path(config['paths']['root']) / config['paths']['dataset'] / 'nifty500_10yr.npz'
    data = np.load(dataset_path, allow_pickle=True)

    if args.split == 'test':
        X = data['X_test']
        y = data['y_test']
        dates = data['dates_test']
    else:
        X = data['X_val']
        y = data['y_val']
        dates = data['dates_val']

    stock_symbols = data['stock_symbols'].tolist()

    print(f"Samples: {len(X)}")
    print(f"Stocks: {len(stock_symbols)}")
    print(f"Date range: {dates[0]} to {dates[-1]}")

    # Load relation mask
    relation_path = Path(config['paths']['root']) / config['paths']['dataset'] / 'relation_matrices.npz'
    relations = np.load(relation_path)
    R_mask = torch.FloatTensor(relations['R_mask'])

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_diffstock_model(config, len(stock_symbols))
    model = model.to(device)

    # Load checkpoint
    checkpoint_path = Path(config['paths']['root']) / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    # Generate predictions
    print("\nGenerating predictions...")
    model.eval()
    R_mask = R_mask.to(device)

    predictions = []

    with torch.no_grad():
        for i in range(len(X)):
            x = torch.FloatTensor(X[i:i+1]).to(device)  # (1, L, N, F)
            pred, _ = model(x, R_mask, n_samples=50)
            predictions.append(pred.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)  # (T, N)

    print(f"Predictions shape: {predictions.shape}")

    # Run backtest
    print("\nRunning backtest...")
    backtester = IndianMarketBacktester(
        predictions=predictions,
        actuals=y,
        dates=dates,
        stock_symbols=stock_symbols,
        transaction_costs=config['evaluation']['transaction_costs']
    )

    results = backtester.run_topk_strategy(
        K=config['evaluation']['top_k'],
        rebalance_freq=config['evaluation']['rebalance_freq']
    )

    # Print results
    backtester.print_backtest_summary(results)

    # Save results
    results_dir = Path(config['paths']['root']) / config['paths']['results']
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / f'backtest_{args.split}.npz'
    np.savez(
        results_path,
        portfolio_values=results['portfolio_values'],
        daily_returns=results['daily_returns'],
        predictions=predictions,
        actuals=y,
        dates=dates
    )

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
