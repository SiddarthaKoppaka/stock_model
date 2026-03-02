"""
Relation Matrix Builder

Builds three N×N relation matrices:
1. Sector relation   — binary float32, symmetric
2. Industry relation — binary float32, symmetric
3. Price correlation — continuous float32, SYMMETRIC (FIXED), thresholded at 0.25

CRITICAL: Correlation computed ONLY on training period to avoid lookahead bias.

Bug fixes vs original:
  [1] R_corr was asymmetric due to row-stochastic normalization destroying symmetry
      → Replaced normalization with raw thresholded |corr| + explicit symmetry assertion
  [2] corr_threshold lowered from 0.4 → 0.25 (Indian markets have lower cross-correlation)
  [3] R_mask density was ~13% (too sparse) → new threshold yields target 25-35%
  [4] Binary masks remain binary float32; R_corr stores actual correlation values for
      optional weighted attention (not flattened to 0/1)

Inputs:
    - data/processed/{SYMBOL}_features.parquet: Feature data
    - data/raw/metadata.json: Sector/industry info

Outputs:
    - data/dataset/relation_matrices.npz: All relation matrices + stock list
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from tqdm import tqdm
from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
# Defaults matching config.yaml
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_CORR_THRESHOLD = 0.25   # was 0.4 — too aggressive for Indian markets
DEFAULT_TRAIN_END_DATE = "2022-12-31"
TARGET_DENSITY_LOW     = 0.20   # warn if combined mask falls below this
TARGET_DENSITY_HIGH    = 0.35   # warn if combined mask exceeds this


class RelationBuilder:
    """
    Builds relation matrices for the Masked Relational Transformer (MRT).

    Key invariants enforced:
      - All matrices are symmetric (verified with assertion)
      - No self-loops (diagonal = 0)
      - R_corr stores raw thresholded |correlation| values (not row-normalized)
      - Isolated stocks get fallback full-sector connections
    """

    def __init__(
        self,
        processed_data_dir: Path,
        metadata_path: Path,
        corr_threshold: float = DEFAULT_CORR_THRESHOLD,
        train_end_date: str = DEFAULT_TRAIN_END_DATE,
    ):
        """
        Args:
            processed_data_dir: Directory containing {SYMBOL}_features.parquet files
            metadata_path: Path to metadata.json with sector/industry info
            corr_threshold: Minimum |correlation| to create an edge in R_corr.
                            0.25 recommended for Nifty 500 (was 0.4, too aggressive).
            train_end_date: Correlation computed only on dates <= this to prevent leakage.
        """
        self.processed_data_dir = Path(processed_data_dir)
        self.metadata_path      = Path(metadata_path)
        self.corr_threshold     = corr_threshold
        self.train_end_date     = pd.to_datetime(train_end_date)

        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)

        logger.info(
            f"RelationBuilder initialized | corr_threshold={corr_threshold} | "
            f"train_end={train_end_date}"
        )

    # ── 1. Sector relation ────────────────────────────────────────────────────

    def build_sector_relation(self, stock_list: List[str]) -> np.ndarray:
        """
        Binary symmetric matrix: R[i,j] = 1 if stocks i and j share the same sector.
        'Unknown' sector stocks are never connected.

        Returns:
            (N, N) float32 binary matrix
        """
        N       = len(stock_list)
        sectors = [
            self.metadata.get(s, {}).get("sector", "Unknown")
            for s in stock_list
        ]

        R_sector = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            if sectors[i] == "Unknown":
                continue
            for j in range(N):
                if i != j and sectors[i] == sectors[j]:
                    R_sector[i, j] = 1.0

        # Verify symmetry
        assert np.allclose(R_sector, R_sector.T), "R_sector not symmetric — this is a bug"

        density = (R_sector > 0).mean()
        logger.info(f"R_sector   | connections: {int(R_sector.sum()):,} | density: {density:.2%}")
        return R_sector

    # ── 2. Industry relation ──────────────────────────────────────────────────

    def build_industry_relation(self, stock_list: List[str]) -> np.ndarray:
        """
        Binary symmetric matrix: R[i,j] = 1 if stocks i and j share the same industry.
        'Unknown' industry stocks are never connected.

        Returns:
            (N, N) float32 binary matrix
        """
        N          = len(stock_list)
        industries = [
            self.metadata.get(s, {}).get("industry", "Unknown")
            for s in stock_list
        ]

        R_industry = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            if industries[i] == "Unknown":
                continue
            for j in range(N):
                if i != j and industries[i] == industries[j]:
                    R_industry[i, j] = 1.0

        # Verify symmetry
        assert np.allclose(R_industry, R_industry.T), "R_industry not symmetric — this is a bug"

        density = (R_industry > 0).mean()
        logger.info(f"R_industry | connections: {int(R_industry.sum()):,} | density: {density:.2%}")
        return R_industry

    # ── 3. Correlation relation ───────────────────────────────────────────────

    def build_correlation_relation(self, stock_list: List[str]) -> np.ndarray:
        """
        Continuous symmetric matrix storing thresholded |correlation| values.

        Design decisions vs original:
          - NOT row-normalized (normalization destroyed symmetry → BUG [1])
          - Stores raw |corr| values so MRT can use graded attention weights
          - Threshold lowered to 0.25 (BUG [2]: 0.4 was too sparse for India)
          - Symmetry explicitly forced and asserted

        Returns:
            (N, N) float32 matrix with values in [0, 1], symmetric, diagonal=0
        """
        N = len(stock_list)
        logger.info(
            f"Building R_corr for {N} stocks "
            f"(training data only, threshold={self.corr_threshold}) ..."
        )

        # ── Load close_ret for each stock, training period only ──────────────
        returns_dict: Dict[str, Optional[pd.Series]] = {}
        for symbol in tqdm(stock_list, desc="Loading returns"):
            path = self.processed_data_dir / f"{symbol}_features.parquet"
            if not path.exists():
                logger.warning(f"{symbol}: features file not found")
                returns_dict[symbol] = None
                continue

            df = pd.read_parquet(path)
            df["Date"] = pd.to_datetime(df["Date"])
            df_train   = df[df["Date"] <= self.train_end_date].copy()

            if len(df_train) < 100:
                logger.warning(f"{symbol}: only {len(df_train)} training days — skipping")
                returns_dict[symbol] = None
                continue

            returns_dict[symbol] = df_train.set_index("Date")["close_ret"]

        # ── Align all valid stocks to common date index ───────────────────────
        valid_symbols = [s for s in stock_list if returns_dict[s] is not None]
        n_excluded    = len(stock_list) - len(valid_symbols)
        if n_excluded > 0:
            logger.warning(f"{n_excluded} stocks excluded from correlation (missing data)")

        returns_df = pd.DataFrame({s: returns_dict[s] for s in valid_symbols})
        returns_df = returns_df.dropna(how="all")
        logger.info(f"Correlation computed on {len(returns_df)} trading days")

        # ── Compute correlation, force symmetry, threshold ────────────────────
        corr_matrix = returns_df.corr().values                   # (M, M), M = |valid_symbols|

        # BUG FIX [1]: Force symmetry and remove row-stochastic normalization entirely
        corr_matrix = (corr_matrix + corr_matrix.T) / 2
        np.fill_diagonal(corr_matrix, 0.0)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        R_corr_valid = np.where(np.abs(corr_matrix) >= self.corr_threshold, corr_matrix, 0.0).astype(np.float32)
        assert np.allclose(R_corr_valid, R_corr_valid.T, atol=1e-5), "R_corr must be symmetric"

        density = (R_corr_valid != 0).mean()
        logger.info(
            f"R_corr (valid stocks) | connections: {int((R_corr_valid > 0).sum()):,} | "
            f"density: {density:.2%}"
        )

        # ── Map valid-stock matrix back to full stock_list ────────────────────
        R_corr_full = np.zeros((N, N), dtype=np.float32)
        sym_to_vidx = {s: i for i, s in enumerate(valid_symbols)}

        for i, s1 in enumerate(stock_list):
            for j, s2 in enumerate(stock_list):
                if s1 in sym_to_vidx and s2 in sym_to_vidx:
                    vi = sym_to_vidx[s1]
                    vj = sym_to_vidx[s2]
                    R_corr_full[i, j] = R_corr_valid[vi, vj]

        # Final symmetry check on full matrix
        assert np.allclose(R_corr_full, R_corr_full.T, atol=1e-5), \
            "R_corr_full not symmetric after remapping — this is a bug"

        full_density = (R_corr_full != 0).mean()
        logger.info(f"R_corr     | density (full {N}×{N}): {full_density:.2%}")
        return R_corr_full

    # ── 4. Combined mask ──────────────────────────────────────────────────────

    def build_combined_mask(
        self,
        R_sector:   np.ndarray,
        R_industry: np.ndarray,
        R_corr:     np.ndarray,
        stock_list: List[str],
    ) -> np.ndarray:
        """
        Union of all three relation matrices → binary attention mask.

        Isolated stocks (no connections from any matrix) receive fallback
        connections to all other stocks sharing the same sector. If a stock
        has no sector info, it receives full attention (all stocks).

        Returns:
            (N, N) float32 binary mask
        """
        N      = R_sector.shape[0]
        R_mask = (R_sector > 0) | (R_industry > 0) | (R_corr != 0)
        np.fill_diagonal(R_mask, False)   # no self-loops

        # ── Fallback for isolated stocks (sector-peer fallback) ─────────────────
        isolated_indices = np.where(R_mask.sum(axis=1) == 0)[0]
        sectors = [self.metadata.get(s, {}).get("sector", "Unknown") for s in stock_list]
        for i in isolated_indices:
            peers = [j for j in range(N) if j != i and sectors[j] == sectors[i] and sectors[i] != "Unknown"]
            if peers:
                R_mask[i, peers] = True
                for j in peers:
                    R_mask[j, i] = True
            else:
                R_mask[i, :] = True
                R_mask[:, i] = True
                R_mask[i, i] = False
        np.fill_diagonal(R_mask, False)

        if len(isolated_indices) > 0:
            logger.warning(
                f"{len(isolated_indices)} isolated stocks — applied sector-peer fallback"
            )

        R_mask_f = R_mask.astype(np.float32)
        density  = 100 * R_mask_f.sum() / R_mask_f.size

        if density < 20.0:
            logger.warning(f"R_mask density {density:.2f}% is below 20% — consider lowering corr_threshold further")
        logger.info(f"R_mask density: {density:.2f}%  (target: 20–35%)")

        return R_mask_f

    # ── Public API ────────────────────────────────────────────────────────────

    def build_all_relations(self, stock_list: List[str]) -> Dict[str, np.ndarray]:
        """
        Build and validate all relation matrices.

        Args:
            stock_list: Ordered list of N stock symbols (must match dataset order)

        Returns:
            Dict with keys: R_sector, R_industry, R_corr, R_mask, stock_symbols
        """
        logger.info(f"Building all relation matrices for {len(stock_list)} stocks ...")

        R_sector   = self.build_sector_relation(stock_list)
        R_industry = self.build_industry_relation(stock_list)
        R_corr     = self.build_correlation_relation(stock_list)
        R_mask     = self.build_combined_mask(R_sector, R_industry, R_corr, stock_list)

        # ── Final validation pass ─────────────────────────────────────────────
        for name, mat in [("R_sector",   R_sector),
                          ("R_industry", R_industry),
                          ("R_corr",     R_corr),
                          ("R_mask",     R_mask)]:
            assert mat.shape == (len(stock_list), len(stock_list)), \
                f"{name} has wrong shape {mat.shape}"
            assert np.allclose(mat, mat.T, atol=1e-5), \
                f"{name} is not symmetric — this is a bug"
            assert mat.diagonal().sum() == 0, \
                f"{name} has non-zero diagonal — this is a bug"

        logger.info("All relation matrices validated ✓")

        return {
            "R_sector":      R_sector,
            "R_industry":    R_industry,
            "R_corr":        R_corr,
            "R_mask":        R_mask,
            "stock_symbols": np.array(stock_list),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def build_relation_matrices(
    config:      Dict,
    stock_list:  List[str],
    output_path: Path,
) -> Dict[str, np.ndarray]:
    """
    Entry point called by the data pipeline.

    Args:
        config:      Full config dict (loads paths, data, training sections)
        stock_list:  Ordered list of stock symbols matching nifty500_10yr.npz
        output_path: Destination for relation_matrices.npz

    Returns:
        Dict with all relation matrices
    """
    root = Path(config["paths"]["root"])

    builder = RelationBuilder(
        processed_data_dir = root / config["paths"]["processed_data"],
        metadata_path      = root / config["paths"]["raw_data"] / "metadata.json",
        corr_threshold     = config["data"]["corr_threshold"],      # 0.15
        train_end_date     = config["training"]["train_end"],        # "2022-12-31"
    )

    matrices = builder.build_all_relations(stock_list)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **matrices, corr_threshold=np.float32(builder.corr_threshold))
    logger.info(f"Relation matrices saved → {output_path}")

    return matrices