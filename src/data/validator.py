"""
Data Validation and Quality Report

Validates cleaned data and generates quality reports.
Determines which stocks pass quality thresholds for inclusion in training.

Inputs:
    - data/processed/{SYMBOL}.parquet: Cleaned stock data
    - data/raw/metadata.json: Metadata

Outputs:
    - data/validation_report.json: Comprehensive validation report
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
from loguru import logger


class DataValidator:
    """
    Validates cleaned stock data for training inclusion.
    """

    def __init__(
        self,
        processed_data_dir: Path,
        metadata_path: Path,
        max_missing_pct: float = 0.15,
        min_trading_days: int = 500,
        max_zero_volume_pct: float = 0.10
    ):
        """
        Initialize validator.

        Args:
            processed_data_dir: Directory with cleaned parquet files
            metadata_path: Path to metadata JSON
            max_missing_pct: Maximum allowed missing data percentage
            min_trading_days: Minimum required trading days
            max_zero_volume_pct: Maximum allowed zero volume days
        """
        self.processed_data_dir = Path(processed_data_dir)
        self.metadata_path = Path(metadata_path)
        self.max_missing_pct = max_missing_pct
        self.min_trading_days = min_trading_days
        self.max_zero_volume_pct = max_zero_volume_pct

        # Load metadata
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)

        logger.info("DataValidator initialized")

    def validate_stock(self, symbol: str) -> Tuple[bool, Dict]:
        """
        Validate a single stock against quality criteria.

        Args:
            symbol: Stock symbol

        Returns:
            Tuple of (passes_validation, stats_dict)
        """
        parquet_path = self.processed_data_dir / f"{symbol}.parquet"

        if not parquet_path.exists():
            return False, {'error': 'File not found'}

        df = pd.read_parquet(parquet_path)

        stats = {
            'available_days': len(df),
            'missing_pct': 0.0,
            'zero_volume_days': 0,
            'outlier_days': 0,
            'listing_date': None,
            'passes_threshold': False
        }

        # Calculate missing data percentage
        stats['missing_pct'] = df[['Close', 'Volume']].isna().mean().mean()

        # Count zero volume days
        stats['zero_volume_days'] = (df['Volume'] == 0).sum()

        # Determine listing date (first non-NaN close)
        first_valid_idx = df['Close'].first_valid_index()
        if first_valid_idx is not None:
            stats['listing_date'] = df.loc[first_valid_idx, 'Date'].strftime('%Y-%m-%d')

        # Count available (non-NaN) trading days
        available_days = df['Close'].notna().sum()

        # Calculate zero volume percentage
        zero_volume_pct = stats['zero_volume_days'] / max(available_days, 1)

        # Validation checks
        passes = True
        reasons = []

        if stats['missing_pct'] > self.max_missing_pct:
            passes = False
            reasons.append(f"missing_pct={stats['missing_pct']:.2%} > {self.max_missing_pct:.2%}")

        if available_days < self.min_trading_days:
            passes = False
            reasons.append(f"available_days={available_days} < {self.min_trading_days}")

        if zero_volume_pct > self.max_zero_volume_pct:
            passes = False
            reasons.append(f"zero_volume_pct={zero_volume_pct:.2%} > {self.max_zero_volume_pct:.2%}")

        stats['passes_threshold'] = passes
        stats['exclusion_reasons'] = reasons if not passes else []

        return passes, stats

    def validate_all_stocks(self) -> Dict:
        """
        Validate all stocks and generate comprehensive report.

        Returns:
            Validation report dictionary
        """
        parquet_files = list(self.processed_data_dir.glob("*.parquet"))
        symbols = [f.stem for f in parquet_files]

        logger.info(f"Validating {len(symbols)} stocks...")

        report = {
            'total_stocks': len(symbols),
            'date_range': {'start': None, 'end': None},
            'total_trading_days': 0,
            'per_stock': {},
            'excluded_stocks': [],
            'exclusion_reasons': {},
            'sector_coverage': {},
            'unknown_sector_count': 0
        }

        passing_count = 0
        all_dates = []

        for symbol in tqdm(symbols, desc="Validating"):
            passes, stats = self.validate_stock(symbol)
            report['per_stock'][symbol] = stats

            if passes:
                passing_count += 1

                # Collect dates for range
                df = pd.read_parquet(self.processed_data_dir / f"{symbol}.parquet")
                all_dates.extend(df['Date'].tolist())
            else:
                report['excluded_stocks'].append(symbol)
                report['exclusion_reasons'][symbol] = ', '.join(stats.get('exclusion_reasons', []))

        # Calculate date range
        if all_dates:
            unique_dates = sorted(set(all_dates))
            report['date_range']['start'] = unique_dates[0].strftime('%Y-%m-%d')
            report['date_range']['end'] = unique_dates[-1].strftime('%Y-%m-%d')
            report['total_trading_days'] = len(unique_dates)

        # Calculate sector coverage
        for symbol in symbols:
            if symbol not in report['excluded_stocks']:
                sector = self.metadata.get(symbol, {}).get('sector', 'Unknown')

                if sector == 'Unknown':
                    report['unknown_sector_count'] += 1
                else:
                    report['sector_coverage'][sector] = report['sector_coverage'].get(sector, 0) + 1

        # Summary
        logger.info(f"\nValidation Summary:")
        logger.info(f"  Total stocks: {len(symbols)}")
        logger.info(f"  Passed: {passing_count}")
        logger.info(f"  Excluded: {len(report['excluded_stocks'])}")
        logger.info(f"  Date range: {report['date_range']['start']} to {report['date_range']['end']}")
        logger.info(f"  Trading days: {report['total_trading_days']}")

        # Warn if too few stocks pass
        if passing_count < 380:
            logger.warning(f"Only {passing_count} stocks passed validation (target: 380+)")
            logger.warning("Consider relaxing thresholds")

        return report

    def save_report(self, report: Dict, output_path: Path) -> None:
        """Save validation report to JSON."""
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Validation report saved to {output_path}")


def validate_nifty500_data(config: Dict) -> Dict:
    """
    Entry point for data validation.

    Args:
        config: Configuration dictionary

    Returns:
        Validation report dictionary
    """
    root = Path(config['paths']['root'])

    validator = DataValidator(
        processed_data_dir=root / config['paths']['processed_data'],
        metadata_path=root / config['paths']['raw_data'] / 'metadata.json',
        max_missing_pct=config['data']['max_missing_pct'],
        min_trading_days=config['data']['min_trading_days'],
        max_zero_volume_pct=config['data']['max_zero_volume_pct']
    )

    report = validator.validate_all_stocks()

    # Save report
    report_path = root / config['paths']['dataset'] / 'validation_report.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    validator.save_report(report, report_path)

    return report
