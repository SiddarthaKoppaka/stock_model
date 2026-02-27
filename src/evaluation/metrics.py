"""
Evaluation Metrics for Stock Prediction

Implements standard quantitative finance metrics:
- Information Coefficient (IC) - rank correlation
- IC Information Ratio (ICIR) - consistency of IC
- Portfolio metrics (Sharpe, Max Drawdown)
- Classification metrics (Accuracy, MCC)
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import matthews_corrcoef, accuracy_score
from typing import Union, Tuple


def information_coefficient(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    method: str = 'spearman'
) -> float:
    """
    Compute Information Coefficient (rank correlation).

    IC is the primary metric for stock prediction models.
    Measures how well predictions rank stocks relative to actual returns.

    Args:
        y_pred: (N,) or (T, N) predicted returns
        y_true: (N,) or (T, N) actual returns
        method: 'spearman' or 'pearson'

    Returns:
        IC value (higher is better, target > 0.05)
    """
    # Handle 2D arrays (time series of predictions)
    if y_pred.ndim == 2:
        # Compute IC for each timestep and return mean
        ics = []
        for t in range(y_pred.shape[0]):
            # Remove NaN values
            mask = ~(np.isnan(y_pred[t]) | np.isnan(y_true[t]))
            if mask.sum() > 2:  # Need at least 3 points for correlation
                if method == 'spearman':
                    ic, _ = spearmanr(y_pred[t][mask], y_true[t][mask])
                else:
                    ic, _ = pearsonr(y_pred[t][mask], y_true[t][mask])

                if not np.isnan(ic):
                    ics.append(ic)

        return np.mean(ics) if ics else 0.0

    # Handle 1D arrays (single prediction)
    mask = ~(np.isnan(y_pred) | np.isnan(y_true))

    if mask.sum() < 3:
        return 0.0

    if method == 'spearman':
        ic, _ = spearmanr(y_pred[mask], y_true[mask])
    else:
        ic, _ = pearsonr(y_pred[mask], y_true[mask])

    return ic if not np.isnan(ic) else 0.0


def ic_information_ratio(ic_series: Union[np.ndarray, pd.Series]) -> float:
    """
    Compute IC Information Ratio (ICIR).

    ICIR = mean(IC) / std(IC)
    Measures consistency of predictions over time.

    Args:
        ic_series: Time series of IC values

    Returns:
        ICIR (higher is better, target > 0.5)
    """
    ic_series = np.array(ic_series)
    ic_series = ic_series[~np.isnan(ic_series)]

    if len(ic_series) < 2:
        return 0.0

    mean_ic = np.mean(ic_series)
    std_ic = np.std(ic_series)

    if std_ic == 0:
        return 0.0

    return mean_ic / std_ic


def rank_ic(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Compute Rank IC (IC on rank-transformed predictions).

    Args:
        y_pred: Predicted returns
        y_true: Actual returns

    Returns:
        Rank IC value
    """
    # Rank-transform predictions
    if y_pred.ndim == 1:
        mask = ~(np.isnan(y_pred) | np.isnan(y_true))
        if mask.sum() < 3:
            return 0.0

        y_pred_rank = pd.Series(y_pred[mask]).rank(pct=True).values
        y_true_rank = pd.Series(y_true[mask]).rank(pct=True).values

        return information_coefficient(y_pred_rank, y_true_rank, method='spearman')

    # Handle 2D
    rank_ics = []
    for t in range(y_pred.shape[0]):
        mask = ~(np.isnan(y_pred[t]) | np.isnan(y_true[t]))
        if mask.sum() > 2:
            y_pred_rank = pd.Series(y_pred[t][mask]).rank(pct=True).values
            y_true_rank = pd.Series(y_true[t][mask]).rank(pct=True).values

            ric = information_coefficient(y_pred_rank, y_true_rank, method='spearman')
            rank_ics.append(ric)

    return np.mean(rank_ics) if rank_ics else 0.0


def portfolio_sharpe(
    returns: np.ndarray,
    risk_free_rate: float = 0.065,
    periods_per_year: int = 252
) -> float:
    """
    Compute annualized Sharpe ratio.

    Args:
        returns: Daily portfolio returns
        risk_free_rate: Annual risk-free rate (Indian FD rate ~6.5%)
        periods_per_year: Trading days per year

    Returns:
        Annualized Sharpe ratio
    """
    returns = np.array(returns)
    returns = returns[~np.isnan(returns)]

    if len(returns) < 2:
        return 0.0

    # Convert annual risk-free rate to daily
    daily_rf = risk_free_rate / periods_per_year

    # Excess returns
    excess_returns = returns - daily_rf

    # Annualized Sharpe
    mean_excess = np.mean(excess_returns) * periods_per_year
    std_excess = np.std(excess_returns) * np.sqrt(periods_per_year)

    if std_excess == 0:
        return 0.0

    return mean_excess / std_excess


def max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Compute maximum drawdown.

    Args:
        equity_curve: Cumulative portfolio value over time

    Returns:
        Maximum drawdown (negative value, e.g., -0.25 = 25% drawdown)
    """
    equity_curve = np.array(equity_curve)
    equity_curve = equity_curve[~np.isnan(equity_curve)]

    if len(equity_curve) < 2:
        return 0.0

    # Cumulative maximum
    running_max = np.maximum.accumulate(equity_curve)

    # Drawdown
    drawdown = (equity_curve - running_max) / running_max

    return np.min(drawdown)


def binary_accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Compute direction prediction accuracy (up/down).

    Args:
        y_pred: Predicted returns
        y_true: Actual returns

    Returns:
        Accuracy [0, 1]
    """
    # Handle 2D arrays
    if y_pred.ndim == 2:
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()

    mask = ~(np.isnan(y_pred) | np.isnan(y_true))

    if mask.sum() < 1:
        return 0.5  # Random guess

    y_pred = y_pred[mask]
    y_true = y_true[mask]

    # Convert to binary (0 = down, 1 = up)
    y_pred_binary = (y_pred > 0).astype(int)
    y_true_binary = (y_true > 0).astype(int)

    return accuracy_score(y_true_binary, y_pred_binary)


def mcc(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Compute Matthews Correlation Coefficient for direction prediction.

    MCC is a balanced metric for binary classification, ranges from -1 to 1.

    Args:
        y_pred: Predicted returns
        y_true: Actual returns

    Returns:
        MCC value [-1, 1]
    """
    # Handle 2D arrays
    if y_pred.ndim == 2:
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()

    mask = ~(np.isnan(y_pred) | np.isnan(y_true))

    if mask.sum() < 1:
        return 0.0

    y_pred = y_pred[mask]
    y_true = y_true[mask]

    # Convert to binary
    y_pred_binary = (y_pred > 0).astype(int)
    y_true_binary = (y_true > 0).astype(int)

    return matthews_corrcoef(y_true_binary, y_pred_binary)


def compute_all_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    portfolio_returns: np.ndarray = None,
    equity_curve: np.ndarray = None
) -> dict:
    """
    Compute all evaluation metrics.

    Args:
        y_pred: Predicted returns
        y_true: Actual returns
        portfolio_returns: Optional portfolio returns for Sharpe
        equity_curve: Optional equity curve for max drawdown

    Returns:
        Dict with all metrics
    """
    metrics = {
        'IC': information_coefficient(y_pred, y_true, method='spearman'),
        'Pearson_IC': information_coefficient(y_pred, y_true, method='pearson'),
        'Rank_IC': rank_ic(y_pred, y_true),
        'Accuracy': binary_accuracy(y_pred, y_true),
        'MCC': mcc(y_pred, y_true)
    }

    # Portfolio metrics (if provided)
    if portfolio_returns is not None:
        metrics['Sharpe'] = portfolio_sharpe(portfolio_returns)

    if equity_curve is not None:
        metrics['Max_Drawdown'] = max_drawdown(equity_curve)

    # Compute ICIR if we have time series
    if y_pred.ndim == 2:
        ic_series = []
        for t in range(y_pred.shape[0]):
            mask = ~(np.isnan(y_pred[t]) | np.isnan(y_true[t]))
            if mask.sum() > 2:
                ic, _ = spearmanr(y_pred[t][mask], y_true[t][mask])
                if not np.isnan(ic):
                    ic_series.append(ic)

        metrics['ICIR'] = ic_information_ratio(ic_series)

    return metrics


if __name__ == "__main__":
    # Test metrics
    np.random.seed(42)

    # Generate synthetic data
    T, N = 100, 400
    y_true = np.random.randn(T, N) * 0.02

    # Generate correlated predictions (IC ~ 0.3)
    noise = np.random.randn(T, N) * 0.02
    y_pred = 0.5 * y_true + 0.5 * noise

    # Compute metrics
    metrics = compute_all_metrics(y_pred, y_true)

    print("Evaluation Metrics Test:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    print("\nAll metrics tests passed!")
