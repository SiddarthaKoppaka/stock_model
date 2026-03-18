"""Evaluation modules for DHARMA."""

from .metrics import (
    information_coefficient,
    ic_information_ratio,
    rank_ic,
    portfolio_sharpe,
    max_drawdown,
    binary_accuracy,
    mcc
)
from .backtester import IndianMarketBacktester

__all__ = [
    'information_coefficient',
    'ic_information_ratio',
    'rank_ic',
    'portfolio_sharpe',
    'max_drawdown',
    'binary_accuracy',
    'mcc',
    'IndianMarketBacktester'
]
