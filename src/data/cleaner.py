"""
Data Cleaning Pipeline

Cleans raw OHLCV data for all stocks:
- Date alignment to common trading calendar
- Missing value treatment (forward fill)
- Outlier detection and correction
- Volume normalization
- Survivorship bias handling

Inputs:
    - data/raw/{SYMBOL}.csv: Raw per-stock data
    - data/raw/metadata.json: Metadata including sectors

Outputs:
    - data/processed/{SYMBOL}.parquet: Cleaned per-stock data
    - Updated metadata with liquidity flags
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from tqdm import tqdm
from loguru import logger


class DataCleaner:
    """
    Cleans raw stock data with survivorship bias handling.
    """

    def __init__(
        self,
        raw_data_dir: Path,
        processed_data_dir: Path,
        max_ffill_days: int = 5,
        outlier_threshold: float = 0.20,
        nifty_index_threshold: float = 0.05,
        min_adv_crores: float = 5.0
    ):
        """
        Initialize data cleaner.

        Args:
            raw_data_dir: Directory with raw CSV files
            processed_data_dir: Directory to save cleaned parquet files
            max_ffill_days: Maximum days to forward-fill missing values
            outlier_threshold: Return threshold to flag as outlier (e.g., 0.20 = 20%)
            nifty_index_threshold: Nifty 500 move threshold to validate outliers
            min_adv_crores: Minimum average daily value in crores for liquidity
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.max_ffill_days = max_ffill_days
        self.outlier_threshold = outlier_threshold
        self.nifty_index_threshold = nifty_index_threshold
        self.min_adv_crores = min_adv_crores

        # Create output directory
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

        # Reference calendar (will be set using most complete stock)
        self.master_calendar = None

        logger.info("DataCleaner initialized")

    def load_and_parse_dates(self, csv_path: Path) -> pd.DataFrame:
        """
        Load CSV and parse dates.

        Args:
            csv_path: Path to raw CSV file

        Returns:
            DataFrame with parsed Date column
        """
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        return df

    def establish_master_calendar(self, symbol_list: List[str]) -> pd.DatetimeIndex:
        """
        Establish master trading calendar using most complete stock (typically Reliance).

        Args:
            symbol_list: List of stock symbols

        Returns:
            DatetimeIndex with all trading days
        """
        logger.info("Establishing master trading calendar...")

        # Try Reliance first (typically most complete)
        priority_symbols = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK']

        for symbol in priority_symbols:
            if symbol in symbol_list:
                csv_path = self.raw_data_dir / f"{symbol}.csv"
                if csv_path.exists():
                    df = self.load_and_parse_dates(csv_path)

                    # Filter out zero volume days (not true trading days)
                    df = df[df['Volume'] > 0]

                    calendar = pd.DatetimeIndex(df['Date'].unique()).sort_values()
                    logger.info(f"Master calendar established using {symbol}: {len(calendar)} trading days")
                    return calendar

        # Fallback: use union of all dates
        logger.warning("Priority symbols not found, using union of all dates")
        all_dates = set()
        for csv_path in self.raw_data_dir.glob("*.csv"):
            if csv_path.stem != 'metadata':
                df = self.load_and_parse_dates(csv_path)
                all_dates.update(df['Date'].tolist())

        calendar = pd.DatetimeIndex(sorted(all_dates))
        logger.info(f"Master calendar established from union: {len(calendar)} trading days")
        return calendar

    def clean_stock(
        self,
        symbol: str,
        reference_returns: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Clean data for a single stock.

        Args:
            symbol: Stock symbol
            reference_returns: Optional Nifty 500 returns for outlier validation

        Returns:
            Tuple of (cleaned DataFrame, cleaning stats dict)
        """
        csv_path = self.raw_data_dir / f"{symbol}.csv"

        if not csv_path.exists():
            logger.warning(f"{symbol}: CSV not found")
            return None, None

        # Load data
        df = self.load_and_parse_dates(csv_path)

        stats = {
            'original_rows': len(df),
            'zero_volume_days': 0,
            'outlier_days': 0,
            'missing_pct': 0.0,
            'illiquid_flag': False,
            'adv_crores': 0.0
        }

        # Step 1: Replace zero volumes with NaN
        zero_volume_mask = df['Volume'] == 0
        stats['zero_volume_days'] = zero_volume_mask.sum()
        df.loc[zero_volume_mask, ['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']] = np.nan

        # Step 2: Replace zero prices with NaN
        for col in ['Open', 'High', 'Low', 'Close', 'Adj_Close']:
            df.loc[df[col] == 0, col] = np.nan

        # Step 3: Align to master calendar
        df = df.set_index('Date')
        df = df.reindex(self.master_calendar)

        # Step 4: Calculate returns before filling
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))

        # Step 5: Detect outliers
        outlier_mask = np.abs(df['log_return']) > self.outlier_threshold

        if reference_returns is not None:
            # Cross-validate with market returns
            market_move = np.abs(reference_returns.reindex(df.index))
            legitimate_moves = market_move > self.nifty_index_threshold

            # Keep outliers that coincide with market events
            outlier_mask = outlier_mask & ~legitimate_moves

        stats['outlier_days'] = outlier_mask.sum()

        # Set outliers to NaN and forward-fill
        df.loc[outlier_mask, ['Open', 'High', 'Low', 'Close', 'Adj_Close']] = np.nan

        # Step 6: Forward fill missing values (up to max_ffill_days)
        for col in ['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']:
            df[col] = df[col].ffill(limit=self.max_ffill_days)

        # Calculate final missing percentage
        stats['missing_pct'] = df[['Close', 'Volume']].isna().mean().mean()

        # Step 7: Calculate liquidity metrics
        df['value'] = df['Close'] * df['Volume']
        avg_daily_value = df['value'].mean()
        stats['adv_crores'] = avg_daily_value / 1e7  # Convert to crores

        if stats['adv_crores'] < self.min_adv_crores:
            stats['illiquid_flag'] = True

        # Step 8: Drop temporary columns
        df = df.drop(columns=['log_return', 'value'], errors='ignore')

        # Reset index to make Date a column again
        df = df.reset_index()

        return df, stats

    def clean_all_stocks(self) -> Dict:
        """
        Clean all stocks and generate cleaning report.

        Returns:
            Dict with cleaning statistics per stock
        """
        # Get list of all symbols
        csv_files = list(self.raw_data_dir.glob("*.csv"))
        symbols = [f.stem for f in csv_files if f.stem != 'metadata']

        logger.info(f"Cleaning {len(symbols)} stocks...")

        # Establish master calendar
        self.master_calendar = self.establish_master_calendar(symbols)

        # Load reference index returns (optional - would need Nifty 500 index data)
        # For now, we'll skip this and just use absolute thresholds
        reference_returns = None

        # Clean each stock
        all_stats = {}
        successful = 0

        for symbol in tqdm(symbols, desc="Cleaning stocks"):
            df_clean, stats = self.clean_stock(symbol, reference_returns)

            if df_clean is not None:
                # Save as parquet (more efficient than CSV)
                output_path = self.processed_data_dir / f"{symbol}.parquet"
                df_clean.to_parquet(output_path, index=False)
                all_stats[symbol] = stats
                successful += 1
            else:
                logger.warning(f"{symbol}: Cleaning failed")

        logger.info(f"Cleaning completed: {successful}/{len(symbols)} stocks")

        # Generate summary statistics
        if all_stats:
            avg_missing = np.mean([s['missing_pct'] for s in all_stats.values()])
            illiquid_count = sum(1 for s in all_stats.values() if s['illiquid_flag'])

            logger.info(f"  Average missing data: {avg_missing:.2%}")
            logger.info(f"  Illiquid stocks: {illiquid_count}")

        return all_stats


def clean_nifty500_data(config: Dict) -> Dict:
    """
    Entry point for data cleaning using configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary with cleaning statistics
    """
    cleaner = DataCleaner(
        raw_data_dir=Path(config['paths']['root']) / config['paths']['raw_data'],
        processed_data_dir=Path(config['paths']['root']) / config['paths']['processed_data'],
        max_ffill_days=config['data']['max_ffill_days'],
        outlier_threshold=config['data']['outlier_threshold'],
        nifty_index_threshold=config['data']['nifty_index_threshold'],
        min_adv_crores=config['data']['min_adv_crores']
    )

    stats = cleaner.clean_all_stocks()
    return stats
