"""
Relation Matrix Builder

Builds three N×N relation matrices:
1. Sector relation (binary)
2. Industry relation (binary)
3. Price correlation (continuous, thresholded)

CRITICAL: Correlation matrix computed ONLY on training period to avoid lookahead bias.

Inputs:
    - data/processed/{SYMBOL}_features.parquet: Feature data
    - data/raw/metadata.json: Sector/industry info

Outputs:
    - data/dataset/relation_matrices.npz: All relation matrices + stock list
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm
from loguru import logger


class RelationBuilder:
    """
    Builds relation matrices for Masked Relational Transformer.
    """

    def __init__(
        self,
        processed_data_dir: Path,
        metadata_path: Path,
        corr_threshold: float = 0.4,
        train_end_date: str = "2022-12-31"
    ):
        """
        Initialize relation builder.

        Args:
            processed_data_dir: Directory with feature parquet files
            metadata_path: Path to metadata JSON
            corr_threshold: Minimum correlation to create edge
            train_end_date: End date of training period for correlation calculation
        """
        self.processed_data_dir = Path(processed_data_dir)
        self.metadata_path = Path(metadata_path)
        self.corr_threshold = corr_threshold
        self.train_end_date = pd.to_datetime(train_end_date)

        # Load metadata
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)

        logger.info(f"RelationBuilder initialized (corr_threshold={corr_threshold})")

    def build_sector_relation(self, stock_list: List[str]) -> np.ndarray:
        """
        Build binary sector relation matrix.

        Args:
            stock_list: Ordered list of stock symbols

        Returns:
            N×N binary matrix where R[i,j]=1 if same sector
        """
        N = len(stock_list)
        R_sector = np.zeros((N, N), dtype=np.float32)

        # Get sectors for all stocks
        sectors = []
        for symbol in stock_list:
            sector = self.metadata.get(symbol, {}).get('sector', 'Unknown')
            sectors.append(sector)

        # Build relation matrix
        for i in range(N):
            for j in range(N):
                if i != j and sectors[i] != 'Unknown' and sectors[i] == sectors[j]:
                    R_sector[i, j] = 1.0

        connection_count = R_sector.sum()
        logger.info(f"Sector relation matrix: {connection_count:.0f} connections")

        return R_sector

    def build_industry_relation(self, stock_list: List[str]) -> np.ndarray:
        """
        Build binary industry relation matrix.

        Args:
            stock_list: Ordered list of stock symbols

        Returns:
            N×N binary matrix where R[i,j]=1 if same industry
        """
        N = len(stock_list)
        R_industry = np.zeros((N, N), dtype=np.float32)

        # Get industries for all stocks
        industries = []
        for symbol in stock_list:
            industry = self.metadata.get(symbol, {}).get('industry', 'Unknown')
            industries.append(industry)

        # Build relation matrix
        for i in range(N):
            for j in range(N):
                if i != j and industries[i] != 'Unknown' and industries[i] == industries[j]:
                    R_industry[i, j] = 1.0

        connection_count = R_industry.sum()
        logger.info(f"Industry relation matrix: {connection_count:.0f} connections")

        return R_industry

    def build_correlation_relation(self, stock_list: List[str]) -> np.ndarray:
        """
        Build correlation relation matrix using TRAINING PERIOD ONLY.

        Args:
            stock_list: Ordered list of stock symbols

        Returns:
            N×N correlation matrix (thresholded and row-normalized)
        """
        N = len(stock_list)
        logger.info(f"Building correlation matrix for {N} stocks (training period only)...")

        # Load returns for all stocks (training period only)
        returns_dict = {}

        for symbol in tqdm(stock_list, desc="Loading returns"):
            features_path = self.processed_data_dir / f"{symbol}_features.parquet"

            if not features_path.exists():
                logger.warning(f"{symbol}: Features not found")
                returns_dict[symbol] = None
                continue

            df = pd.read_parquet(features_path)
            df['Date'] = pd.to_datetime(df['Date'])

            # Filter to training period ONLY
            df_train = df[df['Date'] <= self.train_end_date].copy()

            if len(df_train) < 100:
                logger.warning(f"{symbol}: Insufficient training data ({len(df_train)} days)")
                returns_dict[symbol] = None
                continue

            returns_dict[symbol] = df_train.set_index('Date')['close_ret']

        # Create returns matrix (aligned dates)
        valid_symbols = [s for s in stock_list if returns_dict[s] is not None]

        if len(valid_symbols) < len(stock_list):
            logger.warning(f"{len(stock_list) - len(valid_symbols)} stocks excluded due to missing data")

        # Align all returns to common date index
        returns_df = pd.DataFrame({s: returns_dict[s] for s in valid_symbols})
        returns_df = returns_df.dropna(how='all')  # Drop dates with all NaN

        logger.info(f"Computing correlation on {len(returns_df)} trading days")

        # Compute correlation matrix
        corr_matrix = returns_df.corr().values

        # Assert symmetry and no test leakage
        assert np.allclose(corr_matrix, corr_matrix.T), "Correlation matrix not symmetric"
        logger.info("Verified: Correlation matrix is symmetric")

        # Threshold by absolute correlation
        R_corr = np.abs(corr_matrix)
        R_corr[R_corr < self.corr_threshold] = 0.0

        # Set diagonal to 0 (no self-connections)
        np.fill_diagonal(R_corr, 0.0)

        # Row-stochastic normalization
        row_sums = R_corr.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # Avoid division by zero
        R_corr = R_corr / row_sums

        connection_count = (R_corr > 0).sum()
        logger.info(f"Correlation relation matrix: {connection_count:.0f} connections")

        # Map back to original stock_list with NaNs filled
        R_corr_full = np.zeros((N, N), dtype=np.float32)

        symbol_to_idx = {s: i for i, s in enumerate(valid_symbols)}
        for i, s1 in enumerate(stock_list):
            for j, s2 in enumerate(stock_list):
                if s1 in symbol_to_idx and s2 in symbol_to_idx:
                    idx1 = symbol_to_idx[s1]
                    idx2 = symbol_to_idx[s2]
                    R_corr_full[i, j] = R_corr[idx1, idx2]

        return R_corr_full

    def build_combined_mask(
        self,
        R_sector: np.ndarray,
        R_industry: np.ndarray,
        R_corr: np.ndarray
    ) -> np.ndarray:
        """
        Build combined attention mask (union of all relations).

        Args:
            R_sector: Sector relation matrix
            R_industry: Industry relation matrix
            R_corr: Correlation relation matrix

        Returns:
            N×N boolean mask
        """
        # Union: any connection from any matrix
        R_mask = (R_sector > 0) | (R_industry > 0) | (R_corr > 0)

        # Handle isolated nodes (no connections)
        isolated = R_mask.sum(axis=1) == 0

        if isolated.any():
            logger.warning(f"{isolated.sum()} stocks have no connections, allowing full attention")
            R_mask[isolated, :] = True

        connection_pct = R_mask.sum() / (R_mask.shape[0] ** 2)
        logger.info(f"Combined mask: {connection_pct:.2%} of possible connections")

        return R_mask.astype(np.float32)

    def build_all_relations(self, stock_list: List[str]) -> Dict[str, np.ndarray]:
        """
        Build all relation matrices.

        Args:
            stock_list: Ordered list of stock symbols

        Returns:
            Dict with all matrices
        """
        logger.info(f"Building relation matrices for {len(stock_list)} stocks...")

        # Build individual matrices
        R_sector = self.build_sector_relation(stock_list)
        R_industry = self.build_industry_relation(stock_list)
        R_corr = self.build_correlation_relation(stock_list)

        # Build combined mask
        R_mask = self.build_combined_mask(R_sector, R_industry, R_corr)

        return {
            'R_sector': R_sector,
            'R_industry': R_industry,
            'R_corr': R_corr,
            'R_mask': R_mask,
            'stock_symbols': np.array(stock_list)
        }


def build_relation_matrices(
    config: Dict,
    stock_list: List[str],
    output_path: Path
) -> Dict[str, np.ndarray]:
    """
    Entry point for relation matrix building.

    Args:
        config: Configuration dictionary
        stock_list: List of stock symbols
        output_path: Path to save relation matrices

    Returns:
        Dict with all relation matrices
    """
    root = Path(config['paths']['root'])

    builder = RelationBuilder(
        processed_data_dir=root / config['paths']['processed_data'],
        metadata_path=root / config['paths']['raw_data'] / 'metadata.json',
        corr_threshold=config['data']['corr_threshold'],
        train_end_date=config['training']['train_end']
    )

    matrices = builder.build_all_relations(stock_list)

    # Save
    np.savez_compressed(output_path, **matrices)
    logger.info(f"Relation matrices saved to {output_path}")

    return matrices
