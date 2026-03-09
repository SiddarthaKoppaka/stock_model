"""
Tool 1 — DiffSTOCK Model Predictions

Wraps the DiffSTOCK simulator to expose model predictions as a
callable tool that the Claude advisor can invoke.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List
from loguru import logger

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Sector mapping for major NIFTY500 stocks (partial — extends gracefully)
_SECTOR_MAP = {
    'RELIANCE': 'Energy', 'TCS': 'IT', 'HDFCBANK': 'Banking', 'INFY': 'IT',
    'ICICIBANK': 'Banking', 'HINDUNILVR': 'FMCG', 'ITC': 'FMCG',
    'SBIN': 'Banking', 'BHARTIARTL': 'Telecom', 'KOTAKBANK': 'Banking',
    'LT': 'Infrastructure', 'AXISBANK': 'Banking', 'WIPRO': 'IT',
    'HCLTECH': 'IT', 'ASIANPAINT': 'Consumer', 'MARUTI': 'Auto',
    'SUNPHARMA': 'Pharma', 'ULTRACEMCO': 'Cement', 'TITAN': 'Consumer',
    'BAJFINANCE': 'Finance', 'BAJAJFINSV': 'Finance', 'NESTLEIND': 'FMCG',
    'TECHM': 'IT', 'POWERGRID': 'Power', 'NTPC': 'Power',
    'TATAMOTORS': 'Auto', 'TATASTEEL': 'Metals', 'ONGC': 'Energy',
    'JSWSTEEL': 'Metals', 'M&M': 'Auto', 'ADANIENT': 'Conglomerate',
    'ADANIPORTS': 'Infrastructure', 'COALINDIA': 'Mining',
    'DRREDDY': 'Pharma', 'CIPLA': 'Pharma', 'DIVISLAB': 'Pharma',
    'BRITANNIA': 'FMCG', 'GRASIM': 'Cement', 'HEROMOTOCO': 'Auto',
    'EICHERMOT': 'Auto', 'APOLLOHOSP': 'Healthcare', 'INDUSINDBK': 'Banking',
    'SBILIFE': 'Insurance', 'HDFCLIFE': 'Insurance', 'BAJAJ-AUTO': 'Auto',
    'TATACONSUM': 'FMCG', 'DABUR': 'FMCG', 'PIDILITIND': 'Chemicals',
}


class ModelPredictionsTool:
    """
    Wraps DiffSTOCK simulator for generating model predictions.
    Designed to be used as a tool by the AI advisor.
    """

    def __init__(self, simulator):
        """
        Args:
            simulator: Initialised DiffSTOCKSimulator instance (model already loaded)
        """
        self.sim = simulator
        self._prev_predictions = None  # cache for rank-change computation

    def get_predictions(
        self,
        as_of_date: str,
        top_k: int = 20,
        include_uncertainty: bool = True,
    ) -> Dict:
        """
        Return DiffSTOCK's current top-K predictions.

        Args:
            as_of_date: date string e.g. "2024-10-15"
            top_k: how many stocks to return
            include_uncertainty: whether to include uncertainty scores

        Returns:
            Dict with predictions, model metadata
        """
        # Find closest date in test set
        target = pd.Timestamp(as_of_date)
        diffs = np.abs((self.sim.dates_test - target).days)
        day_idx = int(np.argmin(diffs))
        actual_date = self.sim.dates_test[day_idx]

        logger.info(f"Model predictions for {actual_date.date()} (requested {as_of_date})")

        # Run inference
        X_window = self.sim.X_test[day_idx:day_idx + 1]
        preds, unc = self.sim._predict(X_window, n_samples=50)

        # Rank stocks
        valid = ~np.isnan(preds)
        ranks = np.full(len(preds), len(preds))
        ranks[valid] = len(preds) - np.argsort(np.argsort(preds[valid]))

        # Compute rank changes from previous call
        if self._prev_predictions is not None:
            prev_ranks = np.full(len(self._prev_predictions), len(self._prev_predictions))
            prev_valid = ~np.isnan(self._prev_predictions)
            prev_ranks[prev_valid] = len(self._prev_predictions) - np.argsort(
                np.argsort(self._prev_predictions[prev_valid])
            )
        else:
            prev_ranks = ranks.copy()

        # Get top-K
        top_indices = np.argsort(preds)[-top_k:][::-1]

        predictions_list = []
        for idx in top_indices:
            if np.isnan(preds[idx]):
                continue
            symbol = self.sim.stock_symbols[idx]
            entry = {
                'symbol': symbol,
                'predicted_5d_return': round(float(preds[idx]), 6),
                'rank': int(ranks[idx]),
                'prev_rank': int(prev_ranks[idx]),
                'rank_change': int(prev_ranks[idx] - ranks[idx]),
                'sector': _SECTOR_MAP.get(symbol, 'Other'),
            }
            if include_uncertainty:
                entry['uncertainty'] = round(float(unc[idx]), 6)
                # Confidence = predicted return / uncertainty (signal-to-noise)
                entry['confidence'] = round(
                    float(abs(preds[idx]) / max(unc[idx], 1e-8)), 2
                )
            predictions_list.append(entry)

        self._prev_predictions = preds.copy()

        return {
            'as_of_date': str(actual_date.date()),
            'predictions': predictions_list,
            'model_metadata': {
                'val_ic': 0.304,  # from training run
                'checkpoint': str(self.sim.checkpoint_path.name),
                'n_stocks_universe': int(self.sim.n_stocks),
                'diffusion_samples': 50,
            },
        }


def get_model_predictions(
    simulator,
    as_of_date: str,
    top_k: int = 20,
    include_uncertainty: bool = True,
) -> Dict:
    """
    Functional interface for the model predictions tool.

    Args:
        simulator: DiffSTOCKSimulator instance
        as_of_date: date string
        top_k: number of stocks
        include_uncertainty: include uncertainty scores

    Returns:
        predictions dict
    """
    tool = ModelPredictionsTool(simulator)
    return tool.get_predictions(as_of_date, top_k, include_uncertainty)


# ── Tool schema for Claude ──────────────────────────────────────────────────

MODEL_TOOL_SCHEMA = {
    "name": "get_model_predictions",
    "description": (
        "Get DiffSTOCK model's predicted 5-day forward returns for top-K "
        "NIFTY500 stocks. Returns predicted return, uncertainty, confidence, "
        "rank, and rank change for each stock."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "as_of_date": {
                "type": "string",
                "description": "Date to generate predictions for, format YYYY-MM-DD",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of top stocks to return (default 20)",
                "default": 20,
            },
            "include_uncertainty": {
                "type": "boolean",
                "description": "Whether to include model uncertainty scores",
                "default": True,
            },
        },
        "required": ["as_of_date"],
    },
}


if __name__ == '__main__':
    print("Model Tool — standalone test")
    print("This tool requires a DiffSTOCKSimulator instance.")
    print("Use: from src.advisor.tools.model_tool import get_model_predictions")
