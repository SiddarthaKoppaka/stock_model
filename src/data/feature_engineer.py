"""
Feature Engineering

Computes 16 features per stock including:
- Base return features (6)
- Technical indicators (10)
- Per-stock rolling z-score normalization

Inputs:
    - data/processed/{SYMBOL}.parquet: Cleaned stock data

Outputs:
    - data/processed/{SYMBOL}_features.parquet: Normalized feature matrix
"""

from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm
from loguru import logger

# Try pandas_ta first, fallback to manual implementation
try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    logger.warning("pandas_ta not available, using manual indicator calculation")


class FeatureEngineer:
    """
    Engineers technical features for stock prediction.
    """

    def __init__(
        self,
        processed_data_dir: Path,
        norm_window: int = 252
    ):
        """
        Initialize feature engineer.

        Args:
            processed_data_dir: Directory with processed parquet files
            norm_window: Rolling window for z-score normalization (252 = 1 year)
        """
        self.processed_data_dir = Path(processed_data_dir)
        self.norm_window = norm_window

        logger.info("FeatureEngineer initialized")

    def compute_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute 6 base return features.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with base features added
        """
        # Shift close for previous close
        df['prev_close'] = df['Close'].shift(1)

        # Base features
        df['open_ret'] = (df['Open'] - df['prev_close']) / df['prev_close']
        df['high_ret'] = (df['High'] - df['prev_close']) / df['prev_close']
        df['low_ret'] = (df['Low'] - df['prev_close']) / df['prev_close']
        df['close_ret'] = (df['Close'] - df['prev_close']) / df['prev_close']
        df['log_volume'] = np.log(df['Volume'] + 1)
        df['hl_spread'] = (df['High'] - df['Low']) / df['prev_close']

        return df

    def compute_rsi(self, series: pd.Series, window: int = 14) -> pd.Series:
        """Compute RSI (Relative Strength Index)."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def compute_bollinger_bands(
        self,
        series: pd.Series,
        window: int = 20,
        num_std: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Compute Bollinger Bands."""
        sma = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()

        upper = sma + (num_std * std)
        lower = sma - (num_std * std)

        return upper, sma, lower

    def compute_macd(
        self,
        series: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> pd.Series:
        """Compute MACD signal line."""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return signal_line

    def compute_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 14
    ) -> pd.Series:
        """Compute Average True Range."""
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()

        return atr

    def compute_vwap(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Compute rolling VWAP (Volume Weighted Average Price)."""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        vwap = (typical_price * df['Volume']).rolling(window=window).sum() / \
               df['Volume'].rolling(window=window).sum()
        return vwap

    def compute_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute 10 technical indicator features.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with technical features added
        """
        close = df['Close']
        volume = df['Volume']
        high = df['High']
        low = df['Low']

        # RSI
        df['rsi_14'] = self.compute_rsi(close, window=14)
        df['rsi_5'] = self.compute_rsi(close, window=5)

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.compute_bollinger_bands(close, window=20)
        df['bb_pct'] = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)

        # Volume ratios
        df['vol_ratio_5'] = volume / (volume.rolling(5).mean() + 1e-10)
        df['vol_ratio_20'] = volume / (volume.rolling(20).mean() + 1e-10)

        # MACD
        df['macd_signal'] = self.compute_macd(close, 12, 26, 9)

        # ATR (normalized)
        df['atr_14'] = self.compute_atr(high, low, close, 14) / (close + 1e-10)

        # Momentum
        df['mom_5'] = close / close.shift(5) - 1
        df['mom_20'] = close / close.shift(20) - 1

        # VWAP deviation
        vwap = self.compute_vwap(df, window=20)
        df['close_vwap'] = (close - vwap) / (close + 1e-10)

        return df

    def normalize_features(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """
        Apply rolling z-score normalization to features.

        Args:
            df: DataFrame with features
            feature_cols: List of feature column names to normalize

        Returns:
            DataFrame with normalized features
        """
        for col in feature_cols:
            if col in df.columns:
                rolling_mean = df[col].rolling(window=self.norm_window, min_periods=20).mean()
                rolling_std = df[col].rolling(window=self.norm_window, min_periods=20).std()

                df[f"{col}_norm"] = (df[col] - rolling_mean) / (rolling_std + 1e-8)

        return df

    def engineer_features(self, symbol: str) -> pd.DataFrame:
        """
        Engineer all features for a single stock.

        Args:
            symbol: Stock symbol

        Returns:
            DataFrame with engineered and normalized features
        """
        parquet_path = self.processed_data_dir / f"{symbol}.parquet"

        if not parquet_path.exists():
            logger.warning(f"{symbol}: Processed data not found")
            return None

        df = pd.read_parquet(parquet_path)

        # Compute base features
        df = self.compute_base_features(df)

        # Compute technical features
        df = self.compute_technical_features(df)

        # Feature list (including close_ret as a feature, target will be next day's close_ret)
        feature_cols = [
            'open_ret', 'high_ret', 'low_ret', 'close_ret', 'log_volume', 'hl_spread',
            'rsi_14', 'rsi_5', 'bb_pct', 'vol_ratio_5', 'vol_ratio_20',
            'macd_signal', 'atr_14', 'mom_5', 'mom_20', 'close_vwap'
        ]

        # Normalize features (16 features, including close_ret)
        df = self.normalize_features(df, feature_cols)

        # Keep target (close_ret) unnormalized + all normalized features
        output_cols = ['Date', 'Close', 'close_ret'] + [f"{col}_norm" for col in feature_cols]

        # Select only existing columns
        output_cols = [col for col in output_cols if col in df.columns]

        df_features = df[output_cols].copy()

        return df_features

    def engineer_all_stocks(self, passing_symbols: List[str] = None) -> int:
        """
        Engineer features for all stocks.

        Args:
            passing_symbols: Optional list of symbols that passed validation

        Returns:
            Number of stocks processed
        """
        if passing_symbols is None:
            # Process all stocks
            parquet_files = list(self.processed_data_dir.glob("*.parquet"))
            symbols = [f.stem for f in parquet_files if '_features' not in f.stem]
        else:
            symbols = passing_symbols

        logger.info(f"Engineering features for {len(symbols)} stocks...")

        successful = 0

        for symbol in tqdm(symbols, desc="Feature engineering"):
            df_features = self.engineer_features(symbol)

            if df_features is not None:
                output_path = self.processed_data_dir / f"{symbol}_features.parquet"
                df_features.to_parquet(output_path, index=False)
                successful += 1

        logger.info(f"Feature engineering completed: {successful}/{len(symbols)} stocks")

        return successful


def engineer_nifty500_features(config: Dict, passing_symbols: List[str] = None) -> int:
    """
    Entry point for feature engineering.

    Args:
        config: Configuration dictionary
        passing_symbols: Optional list of symbols that passed validation

    Returns:
        Number of stocks processed
    """
    engineer = FeatureEngineer(
        processed_data_dir=Path(config['paths']['root']) / config['paths']['processed_data'],
        norm_window=252
    )

    count = engineer.engineer_all_stocks(passing_symbols)
    return count
