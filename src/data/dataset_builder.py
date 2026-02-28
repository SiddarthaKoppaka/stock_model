"""
Dataset Builder

Assembles final training dataset by:
1. Running all previous pipeline steps (scrape, clean, validate, feature engineer, relations)
2. Creating sliding window samples
3. Splitting into train/val/test
4. Saving as compressed .npz

Inputs:
    - All previous pipeline outputs

Outputs:
    - data/dataset/nifty500_10yr.npz: Final tensor dataset
"""

import yaml
from pathlib import Path
from typing import Dict, Tuple, List
import pandas as pd
import numpy as np
from tqdm import tqdm
from loguru import logger

from .scraper import scrape_nifty500_data
from .cleaner import clean_nifty500_data
from .validator import validate_nifty500_data
from .feature_engineer import engineer_nifty500_features
from .relation_builder import build_relation_matrices


class DatasetBuilder:
    """
    Builds final sliding window dataset for DiffSTOCK training.
    """

    def __init__(
        self,
        config: Dict,
        lookback_window: int = 20,
        prediction_horizon: int = 1
    ):
        """
        Initialize dataset builder.

        Args:
            config: Configuration dictionary
            lookback_window: Number of days to look back (L)
            prediction_horizon: Days ahead to predict (H)
        """
        self.config = config
        self.lookback_window = lookback_window
        self.prediction_horizon = prediction_horizon

        self.root = Path(config['paths']['root'])
        self.processed_dir = self.root / config['paths']['processed_data']
        self.dataset_dir = self.root / config['paths']['dataset']

        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"DatasetBuilder initialized (L={lookback_window}, H={prediction_horizon})")

    def load_stock_features(self, symbol: str) -> pd.DataFrame:
        """Load feature data for a single stock."""
        features_path = self.processed_dir / f"{symbol}_features.parquet"

        if not features_path.exists():
            return None

        df = pd.read_parquet(features_path)
        df['Date'] = pd.to_datetime(df['Date'])
        return df

    def assemble_feature_tensor(
        self,
        stock_list: List[str],
        feature_cols: List[str]
    ) -> Tuple[np.ndarray, pd.DatetimeIndex]:
        """
        Assemble (T, N, F) feature tensor for all stocks.

        Args:
            stock_list: List of stock symbols
            feature_cols: List of feature column names

        Returns:
            Tuple of (feature_tensor, date_index)
        """
        logger.info(f"Assembling feature tensor for {len(stock_list)} stocks...")

        # Load all stock data
        all_data = {}
        for symbol in tqdm(stock_list, desc="Loading features"):
            df = self.load_stock_features(symbol)
            if df is not None:
                all_data[symbol] = df

        # Get common date range (intersection of all dates)
        all_dates = set(all_data[list(all_data.keys())[0]]['Date'])
        for symbol, df in all_data.items():
            all_dates = all_dates.intersection(set(df['Date']))

        date_index = pd.DatetimeIndex(sorted(all_dates))
        T = len(date_index)
        N = len(stock_list)
        F = len(feature_cols)

        logger.info(f"Feature tensor shape: T={T}, N={N}, F={F}")

        # Initialize tensor
        feature_tensor = np.full((T, N, F), np.nan, dtype=np.float32)
        target_tensor = np.full((T, N), np.nan, dtype=np.float32)

        # Fill tensor
        for stock_idx, symbol in enumerate(tqdm(stock_list, desc="Building tensor")):
            if symbol not in all_data:
                continue

            df = all_data[symbol].set_index('Date')

            for time_idx, date in enumerate(date_index):
                if date in df.index:
                    # Extract features
                    for feat_idx, feat_col in enumerate(feature_cols):
                        if feat_col in df.columns:
                            feature_tensor[time_idx, stock_idx, feat_idx] = df.loc[date, feat_col]

                    # Extract target (close_ret)
                    if 'close_ret' in df.columns:
                        target_tensor[time_idx, stock_idx] = df.loc[date, 'close_ret']

        # Report NaN statistics
        nan_pct_features = np.isnan(feature_tensor).mean()
        nan_pct_targets = np.isnan(target_tensor).mean()

        logger.info(f"NaN percentage - Features: {nan_pct_features:.2%}, Targets: {nan_pct_targets:.2%}")

        return feature_tensor, target_tensor, date_index

    def create_sliding_windows(
        self,
        feature_tensor: np.ndarray,
        target_tensor: np.ndarray,
        date_index: pd.DatetimeIndex,
        split_dates: Dict[str, str]
    ) -> Dict[str, Tuple]:
        """
        Create sliding window samples and split into train/val/test.

        Args:
            feature_tensor: (T, N, F) feature array
            target_tensor: (T, N) target array
            date_index: DatetimeIndex for T
            split_dates: Dict with train_end, val_start, val_end, test_start

        Returns:
            Dict with train/val/test splits
        """
        T, N, F = feature_tensor.shape
        L = self.lookback_window
        H = self.prediction_horizon

        logger.info(f"Creating sliding windows (L={L}, H={H})...")

        # Convert split dates
        train_end = pd.to_datetime(split_dates['train_end'])
        val_start = pd.to_datetime(split_dates['val_start'])
        val_end = pd.to_datetime(split_dates['val_end'])
        test_start = pd.to_datetime(split_dates['test_start'])

        # Create samples
        X_samples = []
        y_samples = []
        date_samples = []
        split_labels = []  # 'train', 'val', or 'test'

        for t in range(L, T - H + 1):
            # Input window: [t-L:t]
            X = feature_tensor[t-L:t, :, :]  # (L, N, F)

            # Target: t (next day's return)
            y = target_tensor[t, :]  # (N,)

            # Skip if too many NaNs in input or target
            if np.isnan(X).mean() > 0.5 or np.isnan(y).mean() > 0.5:
                continue

            current_date = date_index[t]

            # Determine split
            if current_date <= train_end:
                split = 'train'
            elif val_start <= current_date <= val_end:
                split = 'val'
            elif current_date >= test_start:
                split = 'test'
            else:
                continue  # Skip dates not in any split

            X_samples.append(X)
            y_samples.append(y)
            date_samples.append(current_date)
            split_labels.append(split)

        # Convert to arrays
        X_all = np.array(X_samples, dtype=np.float32)  # (n_samples, L, N, F)
        y_all = np.array(y_samples, dtype=np.float32)  # (n_samples, N)
        dates_all = np.array(date_samples)
        splits_all = np.array(split_labels)

        logger.info(f"Total samples created: {len(X_all)}")

        # Split
        train_mask = splits_all == 'train'
        val_mask = splits_all == 'val'
        test_mask = splits_all == 'test'

        dataset = {
            'X_train': X_all[train_mask],
            'y_train': y_all[train_mask],
            'dates_train': dates_all[train_mask],
            'X_val': X_all[val_mask],
            'y_val': y_all[val_mask],
            'dates_val': dates_all[val_mask],
            'X_test': X_all[test_mask],
            'y_test': y_all[test_mask],
            'dates_test': dates_all[test_mask]
        }

        logger.info(f"Train samples: {len(dataset['X_train'])}")
        logger.info(f"Val samples: {len(dataset['X_val'])}")
        logger.info(f"Test samples: {len(dataset['X_test'])}")

        return dataset

    def build_full_dataset(self, run_scraping: bool = False) -> str:
        """
        Run full pipeline and build dataset.

        Args:
            run_scraping: Whether to run data scraping (slow)

        Returns:
            Path to saved dataset
        """
        logger.info("=" * 80)
        logger.info("Starting DiffSTOCK Dataset Building Pipeline")
        logger.info("=" * 80)

        # Step 1: Scrape data (optional)
        if run_scraping:
            logger.info("\nStep 1/6: Scraping data...")
            scrape_nifty500_data(self.config)
        else:
            logger.info("\nStep 1/6: Skipping scraping (using existing data)")

        # Step 2: Clean data
        logger.info("\nStep 2/6: Cleaning data...")
        clean_nifty500_data(self.config)

        # Step 3: Validate data
        logger.info("\nStep 3/6: Validating data...")
        validation_report = validate_nifty500_data(self.config)

        # Get list of passing stocks
        passing_symbols = [
            symbol for symbol, stats in validation_report['per_stock'].items()
            if stats.get('passes_threshold', False)
        ]

        logger.info(f"Stocks passing validation: {len(passing_symbols)}")

        if len(passing_symbols) < 100:
            logger.error(f"Too few stocks passed validation ({len(passing_symbols)}), aborting")
            return None

        # Step 4: Feature engineering
        logger.info("\nStep 4/6: Engineering features...")
        engineer_nifty500_features(self.config, passing_symbols)

        # Step 5: Build relation matrices
        logger.info("\nStep 5/6: Building relation matrices...")
        relation_path = self.dataset_dir / 'relation_matrices.npz'
        build_relation_matrices(self.config, passing_symbols, relation_path)

        # Step 6: Assemble final dataset
        logger.info("\nStep 6/6: Assembling final dataset...")

        # Feature columns (16 normalized features)
        feature_cols = [
            'open_ret_norm', 'high_ret_norm', 'low_ret_norm', 'close_ret_norm', 'log_volume_norm', 'hl_spread_norm',
            'rsi_14_norm', 'rsi_5_norm', 'bb_pct_norm', 'vol_ratio_5_norm', 'vol_ratio_20_norm',
            'macd_signal_norm', 'atr_14_norm', 'mom_5_norm', 'mom_20_norm', 'close_vwap_norm'
        ]

        # Assemble tensor
        feature_tensor, target_tensor, date_index = self.assemble_feature_tensor(
            passing_symbols,
            feature_cols
        )

        # Create sliding windows and split
        split_dates = {
            'train_end': self.config['training']['train_end'],
            'val_start': self.config['training']['val_start'],
            'val_end': self.config['training']['val_end'],
            'test_start': self.config['training']['test_start']
        }

        dataset = self.create_sliding_windows(
            feature_tensor,
            target_tensor,
            date_index,
            split_dates
        )

        # Add metadata
        dataset['stock_symbols'] = np.array(passing_symbols)
        dataset['feature_names'] = np.array(feature_cols)

        # Save
        output_path = self.dataset_dir / 'nifty500_10yr.npz'
        np.savez_compressed(output_path, **dataset)

        logger.info(f"\nDataset saved to: {output_path}")
        logger.info("=" * 80)
        logger.info("Dataset building complete!")
        logger.info("=" * 80)

        return str(output_path)


def build_dataset(config_path: str = None, run_scraping: bool = False) -> str:
    """
    Entry point for dataset building.

    Args:
        config_path: Path to config YAML file
        run_scraping: Whether to run data scraping

    Returns:
        Path to saved dataset
    """
    # Load config
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / 'config' / 'config.yaml'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Build dataset
    builder = DatasetBuilder(
        config=config,
        lookback_window=config['data']['lookback_window'],
        prediction_horizon=1
    )

    dataset_path = builder.build_full_dataset(run_scraping=run_scraping)

    return dataset_path
