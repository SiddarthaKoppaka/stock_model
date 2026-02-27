"""
Entry point for training DiffSTOCK model.

Usage:
    python scripts/run_train.py [--config path/to/config.yaml]
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
from src.utils.seed import set_seed
from src.model.diffstock import create_diffstock_model
from src.training.trainer import DiffSTOCKTrainer


def main():
    parser = argparse.ArgumentParser(description='Train DiffSTOCK model')
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
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

    # Set seed
    set_seed(config['seed'])

    print("=" * 80)
    print("DiffSTOCK Training")
    print("=" * 80)

    # Load dataset
    dataset_path = Path(config['paths']['root']) / config['paths']['dataset'] / 'nifty500_10yr.npz'

    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}")
        print("Please run data pipeline first")
        return

    print(f"Loading dataset from {dataset_path}...")
    data = np.load(dataset_path, allow_pickle=True)

    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    stock_symbols = data['stock_symbols']

    print(f"Train samples: {len(X_train)}")
    print(f"Val samples: {len(X_val)}")
    print(f"Stocks: {len(stock_symbols)}")

    # Load relation mask
    relation_path = Path(config['paths']['root']) / config['paths']['dataset'] / 'relation_matrices.npz'
    relations = np.load(relation_path)
    R_mask = torch.FloatTensor(relations['R_mask'])

    print(f"Relation mask density: {R_mask.mean():.2%}")

    # Create model
    n_stocks = len(stock_symbols)
    model = create_diffstock_model(config, n_stocks)

    # Print model summary
    model.print_model_summary()

    # Create trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = DiffSTOCKTrainer(model, config, R_mask, device)

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"Resumed from checkpoint: {args.resume}")

    # Train
    history = trainer.train(
        train_data=(X_train, y_train),
        val_data=(X_val, y_val)
    )

    print("\nTraining completed!")
    print(f"Best validation IC: {trainer.best_val_ic:.4f}")


if __name__ == "__main__":
    main()
