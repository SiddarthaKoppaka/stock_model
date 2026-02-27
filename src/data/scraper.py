"""
Nifty 500 Stock Data Scraper

Downloads historical OHLCV data for all Nifty 500 stocks from 2015-2026.
Uses yfinance as primary source with jugaad-data as fallback for failed symbols.
Also fetches sector/industry metadata.

Inputs:
    - config: Configuration dict with date ranges and data sources

Outputs:
    - data/raw/{SYMBOL}.csv: Per-stock OHLCV data
    - data/raw/metadata.json: Sector/industry information per stock
    - data/raw/failed_symbols.txt: List of symbols that failed to download
"""

import time
import json
from pathlib import Path
from datetime import date, datetime
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm
from loguru import logger

# Optional imports with fallback
try:
    from jugaad_data.nse import stock_df
    JUGAAD_AVAILABLE = True
except ImportError:
    JUGAAD_AVAILABLE = False
    logger.warning("jugaad-data not available, will skip fallback for failed symbols")

try:
    from nsetools import Nse
    NSETOOLS_AVAILABLE = True
except ImportError:
    NSETOOLS_AVAILABLE = False
    logger.warning("nsetools not available, metadata fallback limited")


class NiftyStockScraper:
    """
    Scraper for Nifty 500 stock data with resume capability and metadata fetching.
    """

    def __init__(
        self,
        start_date: str = "2015-01-01",
        end_date: str = "2026-02-26",
        raw_data_dir: Path = Path("data/raw"),
        nifty500_url: str = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    ):
        """
        Initialize the scraper.

        Args:
            start_date: Start date for historical data (YYYY-MM-DD)
            end_date: End date for historical data (YYYY-MM-DD)
            raw_data_dir: Directory to save raw CSV files
            nifty500_url: URL to Nifty 500 constituent list
        """
        self.start_date = start_date
        self.end_date = end_date
        self.raw_data_dir = Path(raw_data_dir)
        self.nifty500_url = nifty500_url

        # Create directories
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize NSE connection if available
        self.nse = Nse() if NSETOOLS_AVAILABLE else None

        logger.info(f"NiftyStockScraper initialized: {start_date} to {end_date}")

    def fetch_nifty500_list(self) -> pd.DataFrame:
        """
        Fetch current Nifty 500 constituent list from NSE.

        Returns:
            DataFrame with columns: Symbol, Company Name, Industry, Series
        """
        logger.info(f"Fetching Nifty 500 list from {self.nifty500_url}")

        try:
            # Try direct download from NSE
            df = pd.read_csv(self.nifty500_url)
            logger.info(f"Downloaded Nifty 500 list: {len(df)} entries")

            # Filter for EQ series only (exclude BE, BZ, etc.)
            if 'Series' in df.columns:
                df = df[df['Series'] == 'EQ']
                logger.info(f"Filtered to EQ series: {len(df)} stocks")

            # Rename columns for consistency
            column_mapping = {
                'Symbol': 'symbol',
                'Company Name': 'company_name',
                'Industry': 'industry',
                'Series': 'series'
            }
            df = df.rename(columns=column_mapping)

            return df[['symbol', 'company_name', 'industry']]

        except Exception as e:
            logger.error(f"Failed to fetch Nifty 500 list: {e}")
            logger.info("Attempting to use fallback symbol list...")

            # Fallback: use a hardcoded list of major stocks (last resort)
            fallback_symbols = [
                'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR',
                'ITC', 'SBIN', 'BHARTIARTL', 'KOTAKBANK', 'LT', 'AXISBANK',
                'ASIANPAINT', 'MARUTI', 'HCLTECH', 'WIPRO', 'ULTRACEMCO', 'TITAN',
                'NESTLEIND', 'BAJFINANCE', 'SUNPHARMA', 'ONGC', 'NTPC', 'POWERGRID'
            ]
            logger.warning(f"Using fallback list of {len(fallback_symbols)} major stocks")

            return pd.DataFrame({
                'symbol': fallback_symbols,
                'company_name': fallback_symbols,
                'industry': ['Unknown'] * len(fallback_symbols)
            })

    def download_stock_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Download OHLCV data for a single stock using yfinance.

        Args:
            symbol: Stock symbol (e.g., 'RELIANCE')

        Returns:
            DataFrame with OHLCV data or None if failed
        """
        ticker = f"{symbol}.NS"

        try:
            df = yf.download(
                ticker,
                start=self.start_date,
                end=self.end_date,
                auto_adjust=True,
                progress=False
            )

            if df.empty or len(df) < 500:
                logger.debug(f"{symbol}: yfinance returned insufficient data ({len(df)} rows)")
                return None

            # Standardize column names
            df = df.reset_index()
            df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            df['Adj_Close'] = df['Close']  # auto_adjust=True means Close is already adjusted

            return df

        except Exception as e:
            logger.debug(f"{symbol}: yfinance failed - {e}")
            return None

    def download_stock_data_jugaad(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Fallback: Download data using jugaad-data (direct NSE source).

        Args:
            symbol: Stock symbol

        Returns:
            DataFrame with OHLCV data or None if failed
        """
        if not JUGAAD_AVAILABLE:
            return None

        try:
            start = datetime.strptime(self.start_date, "%Y-%m-%d").date()
            end = datetime.strptime(self.end_date, "%Y-%m-%d").date()

            df = stock_df(
                symbol=symbol,
                from_date=start,
                to_date=end,
                series="EQ"
            )

            if df.empty or len(df) < 500:
                return None

            # Standardize columns
            df = df.rename(columns={
                'DATE': 'Date',
                'OPEN': 'Open',
                'HIGH': 'High',
                'LOW': 'Low',
                'CLOSE': 'Close',
                'LTP': 'Close',  # Last Traded Price
                'VOLUME': 'Volume',
                'VALUE': 'Volume'
            })

            # Select required columns
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            df = df[[col for col in required_cols if col in df.columns]]

            if 'Date' not in df.columns:
                df = df.reset_index()

            df['Adj_Close'] = df['Close']

            return df

        except Exception as e:
            logger.debug(f"{symbol}: jugaad-data failed - {e}")
            return None

    def fetch_metadata(self, symbol: str, max_retries: int = 3) -> Dict:
        """
        Fetch sector/industry metadata for a stock with retry logic.

        Args:
            symbol: Stock symbol
            max_retries: Number of retry attempts

        Returns:
            Dict with sector, industry, marketCap, etc.
        """
        metadata = {
            'sector': 'Unknown',
            'industry': 'Unknown',
            'marketCap': None,
            'fullTimeEmployees': None
        }

        ticker = f"{symbol}.NS"

        for attempt in range(max_retries):
            try:
                info = yf.Ticker(ticker).info

                if info:
                    metadata['sector'] = info.get('sector', 'Unknown')
                    metadata['industry'] = info.get('industry', 'Unknown')
                    metadata['marketCap'] = info.get('marketCap', None)
                    metadata['fullTimeEmployees'] = info.get('fullTimeEmployees', None)
                    return metadata

            except Exception as e:
                wait_time = 5 * (2 ** attempt)  # Exponential backoff: 5s, 10s, 20s
                logger.debug(f"{symbol}: metadata fetch failed (attempt {attempt+1}/{max_retries}), retrying in {wait_time}s")
                time.sleep(wait_time)

        # Fallback: try nsetools if available
        if self.nse:
            try:
                quote = self.nse.get_quote(symbol)
                if quote:
                    metadata['industry'] = quote.get('industry', 'Unknown')
            except:
                pass

        return metadata

    def scrape_all(self, batch_size: int = 20, batch_delay: float = 2.0) -> Tuple[int, int]:
        """
        Scrape all Nifty 500 stocks with batching and resume capability.

        Args:
            batch_size: Number of stocks to download in parallel per batch
            batch_delay: Delay in seconds between batches

        Returns:
            Tuple of (successful_count, failed_count)
        """
        # Fetch Nifty 500 list
        nifty_df = self.fetch_nifty500_list()
        all_symbols = nifty_df['symbol'].tolist()

        logger.info(f"Starting scrape for {len(all_symbols)} symbols")

        # Track progress
        successful = 0
        failed_symbols = []
        all_metadata = {}

        # Resume: skip already downloaded symbols
        existing_symbols = {p.stem for p in self.raw_data_dir.glob("*.csv")}
        symbols_to_download = [s for s in all_symbols if s not in existing_symbols]

        logger.info(f"Resuming: {len(existing_symbols)} already downloaded, {len(symbols_to_download)} remaining")

        # Download in batches
        for i in tqdm(range(0, len(symbols_to_download), batch_size), desc="Batches"):
            batch = symbols_to_download[i:i+batch_size]

            for symbol in tqdm(batch, desc=f"Batch {i//batch_size + 1}", leave=False):
                # Try yfinance first
                df = self.download_stock_data(symbol)

                # Fallback to jugaad if yfinance failed
                if df is None:
                    df = self.download_stock_data_jugaad(symbol)

                # Save if successful
                if df is not None and not df.empty:
                    output_path = self.raw_data_dir / f"{symbol}.csv"
                    df.to_csv(output_path, index=False)
                    successful += 1
                else:
                    failed_symbols.append(symbol)
                    logger.warning(f"{symbol}: Failed to download data")

            # Rate limiting between batches
            if i + batch_size < len(symbols_to_download):
                time.sleep(batch_delay)

        # Fetch metadata for all symbols (including previously downloaded)
        logger.info("Fetching metadata for all symbols...")
        for symbol in tqdm(all_symbols, desc="Metadata"):
            all_metadata[symbol] = self.fetch_metadata(symbol)
            time.sleep(0.5)  # Rate limit

        # Save metadata
        metadata_path = self.raw_data_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(all_metadata, f, indent=2)
        logger.info(f"Metadata saved to {metadata_path}")

        # Save failed symbols
        if failed_symbols:
            failed_path = self.raw_data_dir / "failed_symbols.txt"
            with open(failed_path, 'w') as f:
                f.write('\n'.join(failed_symbols))
            logger.warning(f"{len(failed_symbols)} symbols failed, saved to {failed_path}")

        # Generate summary
        logger.info(f"\nScraping Summary:")
        logger.info(f"  Total symbols: {len(all_symbols)}")
        logger.info(f"  Successful: {successful + len(existing_symbols)}")
        logger.info(f"  Failed: {len(failed_symbols)}")

        return successful + len(existing_symbols), len(failed_symbols)


def scrape_nifty500_data(config: Dict) -> None:
    """
    Entry point for data scraping using configuration.

    Args:
        config: Configuration dictionary with data parameters
    """
    scraper = NiftyStockScraper(
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date'],
        raw_data_dir=Path(config['paths']['root']) / config['paths']['raw_data'],
        nifty500_url=config['data']['nifty500_url']
    )

    successful, failed = scraper.scrape_all()

    if failed > 0:
        logger.warning(f"Scraping completed with {failed} failures")
    else:
        logger.info("Scraping completed successfully!")
