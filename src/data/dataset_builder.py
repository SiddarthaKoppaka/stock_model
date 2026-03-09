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
    - data/dataset/nifty500_20yr.npz: Final tensor dataset
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

        # Get full date range (union of all dates — stocks with shorter history get NaN)
        all_dates = set()
        for symbol, df in all_data.items():
            all_dates = all_dates.union(set(df['Date']))

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

        for t in range(L, T - H - 1):          # -1: label window t+1:t+1+H needs t+H in bounds
            X = feature_tensor[t-L:t, :, :]   # (L, N, F) — window ends at t-1

            # Multi-horizon label: H-day compounded forward return
            # Label starts at t+1 (the day AFTER the last feature day) to avoid
            # any same-day overlap between features and the label window.
            # target_tensor[t+1:t+1+H, :] = daily returns for days t+1, ..., t+H
            forward_rets = target_tensor[t+1:t+1+H, :]  # (H, N)
            y = (1 + forward_rets).prod(axis=0) - 1      # (N,) compounded H-day return

            if np.isnan(X).mean() > 0.5 or np.isnan(y).mean() > 0.5:
                continue
            current_date = date_index[t-1] 

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

        # Handle NaN values (replace with 0 for normalized features)
        nan_count_x = np.isnan(X_all).sum()
        nan_count_y = np.isnan(y_all).sum()
        if nan_count_x > 0:
            logger.warning(f"Found {nan_count_x} NaN values in X features, filling with 0")
            X_all = np.nan_to_num(X_all, nan=0.0)
        if nan_count_y > 0:
            logger.warning(f"Found {nan_count_y} NaN values in y targets, filling with 0")
            y_all = np.nan_to_num(y_all, nan=0.0)

        logger.info(f"Total samples created: {len(X_all)}")

        # Split
        train_mask = splits_all == 'train'
        val_mask = splits_all == 'val'
        test_mask = splits_all == 'test'

        X_train = X_all[train_mask]
        X_val = X_all[val_mask]
        X_test = X_all[test_mask]

        # Clip extreme outliers at ±5σ per feature (0.15% of values were beyond this in audit)
        for feat_idx in range(X_train.shape[-1]):
            mu  = X_train[..., feat_idx].mean()
            sig = X_train[..., feat_idx].std()
            clip_lo, clip_hi = mu - 5 * sig, mu + 5 * sig
            X_train[..., feat_idx] = np.clip(X_train[..., feat_idx], clip_lo, clip_hi)
            X_val[...,   feat_idx] = np.clip(X_val[...,   feat_idx], clip_lo, clip_hi)
            X_test[...,  feat_idx] = np.clip(X_test[...,  feat_idx], clip_lo, clip_hi)
        logger.info("Applied ±5σ feature clipping using train-set statistics")

        dataset = {
            'X_train': X_train,
            'y_train': y_all[train_mask],
            'dates_train': dates_all[train_mask],
            'X_val': X_val,
            'y_val': y_all[val_mask],
            'dates_val': dates_all[val_mask],
            'X_test': X_test,
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

        # Raw feature columns (used when use_revin=True — RevIN normalizes at runtime)
        raw_feature_cols = [
            'open_ret', 'high_ret', 'low_ret', 'close_ret', 'log_volume', 'hl_spread',
            'rsi_14', 'rsi_5', 'bb_pct', 'vol_ratio_5', 'vol_ratio_20',
            'macd_signal', 'atr_14', 'mom_5', 'mom_20', 'close_vwap'
        ]
        # Normalized feature columns (legacy rolling z-score, used when use_revin=False)
        norm_feature_cols = [f'{c}_norm' for c in raw_feature_cols]

        use_revin = self.config.get('model', {}).get('use_revin', False)
        feature_cols = raw_feature_cols if use_revin else norm_feature_cols
        logger.info(f"Feature mode: {'raw (RevIN)' if use_revin else 'rolling z-score (legacy)'}")

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

        # ── Regime probabilities (Change 2) ──────────────────────────────────────
        use_regime = self.config.get('model', {}).get('use_regime', False)
        if use_regime:
            regime_probs_path = (
                self.root / self.config['data'].get('regime_probs_path', 'data/regime/daily_regime_probs.parquet')
            )
            if regime_probs_path.exists():
                logger.info("Attaching regime probabilities to dataset...")
                regime_df = pd.read_parquet(regime_probs_path)
                regime_df['date'] = pd.to_datetime(regime_df['date'])
                regime_df = regime_df.set_index('date')
                n_states = self.config.get('model', {}).get('n_regime_states', 4)
                prob_cols = [f'prob_state_{i}' for i in range(n_states)]

                for split in ['train', 'val', 'test']:
                    dates = dataset[f'dates_{split}']
                    probs = np.zeros((len(dates), n_states), dtype=np.float32)
                    for i, d in enumerate(dates):
                        d_ts = pd.Timestamp(d)
                        if d_ts in regime_df.index:
                            probs[i] = regime_df.loc[d_ts, prob_cols].values
                        else:
                            # Nearest-date fallback
                            nearest = regime_df.index[regime_df.index.get_indexer([d_ts], method='nearest')[0]]
                            probs[i] = regime_df.loc[nearest, prob_cols].values
                    dataset[f'regime_probs_{split}'] = probs
                logger.info("Regime probabilities attached ✓")
            else:
                logger.warning(f"Regime probs not found at {regime_probs_path} — skipping")

        # Add metadata
        dataset['stock_symbols'] = np.array(passing_symbols)
        dataset['feature_names'] = np.array(raw_feature_cols)  # always store raw names

        # Save
        output_path = self.dataset_dir / 'nifty500_20yr.npz'
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
        prediction_horizon=config['data'].get('label_horizon', 5)  # default 5-day horizon
    )

    dataset_path = builder.build_full_dataset(run_scraping=run_scraping)

    return dataset_path
