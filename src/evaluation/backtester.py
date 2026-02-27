"""
Indian Market Backtester

Walk-forward backtest with realistic Indian market transaction costs.
Implements long-only top-K strategy with weekly rebalancing.

Transaction costs include:
- Brokerage, STT, exchange charges, GST, stamp duty, slippage
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from loguru import logger


class IndianMarketBacktester:
    """
    Backtester for Indian equity markets with realistic costs.
    """

    # Default transaction costs (can be overridden)
    DEFAULT_COSTS = {
        'brokerage': 0.0003,     # 0.03%
        'stt_buy': 0.001,        # 0.1%
        'stt_sell': 0.001,       # 0.1%
        'exchange': 0.0000335,   # NSE charges
        'sebi': 0.000001,        # SEBI fee
        'gst': 0.18,             # 18% on brokerage + exchange
        'stamp': 0.00015,        # 0.015% on buy
        'slippage': 0.002        # 0.2% average slippage
    }

    def __init__(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        dates: np.ndarray,
        stock_symbols: List[str],
        transaction_costs: Dict = None,
        initial_capital: float = 1000000.0  # 10 lakhs
    ):
        """
        Initialize backtester.

        Args:
            predictions: (T, N) predicted returns
            actuals: (T, N) actual returns
            dates: (T,) timestamps
            stock_symbols: List of stock symbols
            transaction_costs: Dict of transaction cost parameters
            initial_capital: Starting portfolio value (in INR)
        """
        self.predictions = predictions
        self.actuals = actuals
        self.dates = pd.to_datetime(dates)
        self.stock_symbols = stock_symbols
        self.initial_capital = initial_capital

        # Set transaction costs
        if transaction_costs is None:
            self.costs = self.DEFAULT_COSTS
        else:
            self.costs = {**self.DEFAULT_COSTS, **transaction_costs}

        # Calculate total round-trip cost
        self.total_cost_pct = self._calculate_total_cost()

        logger.info(f"IndianMarketBacktester initialized with {len(dates)} days, {len(stock_symbols)} stocks")
        logger.info(f"Total round-trip cost: {self.total_cost_pct:.4%}")

    def _calculate_total_cost(self) -> float:
        """Calculate total round-trip transaction cost percentage."""
        # Buy costs
        buy_base = self.costs['brokerage'] + self.costs['stamp']
        buy_gst_on = self.costs['brokerage'] + self.costs['exchange']
        buy_cost = buy_base + self.costs['exchange'] + self.costs['sebi'] + \
                   (buy_gst_on * self.costs['gst']) + self.costs['stt_buy']

        # Sell costs
        sell_base = self.costs['brokerage']
        sell_gst_on = self.costs['brokerage'] + self.costs['exchange']
        sell_cost = sell_base + self.costs['exchange'] + self.costs['sebi'] + \
                    (sell_gst_on * self.costs['gst']) + self.costs['stt_sell']

        # Slippage (both ways)
        slippage_cost = 2 * self.costs['slippage']

        # Total round-trip
        total = buy_cost + sell_cost + slippage_cost

        return total

    def run_topk_strategy(
        self,
        K: int = 20,
        rebalance_freq: str = 'weekly',
        return_daily_positions: bool = False
    ) -> Dict:
        """
        Run long-only Top-K strategy.

        Each rebalance period:
        1. Select top K stocks by predicted return
        2. Equally weight portfolio
        3. Compute returns after transaction costs

        Args:
            K: Number of stocks to hold
            rebalance_freq: 'daily' or 'weekly'
            return_daily_positions: Whether to return daily position data

        Returns:
            Dict with backtest results
        """
        T, N = self.predictions.shape

        # Initialize tracking
        portfolio_value = np.zeros(T)
        portfolio_value[0] = self.initial_capital

        daily_returns = np.zeros(T)
        turnover = np.zeros(T)

        current_positions = np.zeros(N)  # Number of shares held

        # Rebalance schedule
        if rebalance_freq == 'weekly':
            # Rebalance every Monday (or first trading day of week)
            rebalance_days = self._get_weekly_rebalance_days()
        else:
            rebalance_days = set(range(T))  # Rebalance every day

        daily_position_data = [] if return_daily_positions else None

        for t in range(1, T):
            # Check if rebalance day
            if t in rebalance_days:
                # Get top K stocks
                top_k_indices = self._get_top_k_stocks(t - 1, K)

                # Calculate target positions (equal weight)
                target_positions = np.zeros(N)
                target_value_per_stock = portfolio_value[t - 1] / K

                for idx in top_k_indices:
                    # Assume stock price = 1 (we're working in returns space)
                    # In practice, would use actual prices
                    target_positions[idx] = target_value_per_stock

                # Calculate turnover
                turnover_value = np.sum(np.abs(target_positions - current_positions))
                turnover[t] = turnover_value / portfolio_value[t - 1]

                # Apply transaction costs
                transaction_cost = turnover_value * self.total_cost_pct
                portfolio_value[t] = portfolio_value[t - 1] - transaction_cost

                # Update positions
                current_positions = target_positions
            else:
                # No rebalance, carry forward value minus costs
                portfolio_value[t] = portfolio_value[t - 1]

            # Apply returns for held stocks
            if np.sum(current_positions > 0) > 0:
                # Get actual returns for held stocks
                held_stocks = current_positions > 0
                held_returns = self.actuals[t, held_stocks]

                # Remove NaN returns (replace with 0)
                held_returns = np.nan_to_num(held_returns, nan=0.0)

                # Weighted return (equal weight)
                portfolio_return = np.mean(held_returns)

                # Apply return
                portfolio_value[t] = portfolio_value[t] * (1 + portfolio_return)
                daily_returns[t] = portfolio_return
            else:
                daily_returns[t] = 0.0

            # Store position data
            if return_daily_positions:
                daily_position_data.append({
                    'date': self.dates[t],
                    'value': portfolio_value[t],
                    'positions': current_positions.copy(),
                    'return': daily_returns[t]
                })

        # Calculate metrics
        results = self._calculate_backtest_metrics(
            portfolio_value,
            daily_returns,
            turnover
        )

        if return_daily_positions:
            results['daily_positions'] = daily_position_data

        return results

    def _get_weekly_rebalance_days(self) -> set:
        """Get indices of weekly rebalance days (Mondays)."""
        df = pd.DataFrame({'date': self.dates})
        df['week'] = df['date'].dt.isocalendar().week
        df['year'] = df['date'].dt.year

        # First day of each week
        rebalance_indices = df.groupby(['year', 'week']).head(1).index.tolist()

        return set(rebalance_indices)

    def _get_top_k_stocks(self, t: int, K: int) -> List[int]:
        """Get indices of top K stocks by predicted return."""
        preds = self.predictions[t]

        # Remove NaN predictions
        valid_mask = ~np.isnan(preds)
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) < K:
            return valid_indices.tolist()

        # Get top K
        valid_preds = preds[valid_mask]
        top_k_relative = np.argsort(valid_preds)[-K:]

        # Map back to original indices
        top_k_absolute = valid_indices[top_k_relative]

        return top_k_absolute.tolist()

    def _calculate_backtest_metrics(
        self,
        portfolio_value: np.ndarray,
        daily_returns: np.ndarray,
        turnover: np.ndarray
    ) -> Dict:
        """Calculate comprehensive backtest metrics."""
        # Total return
        total_return = (portfolio_value[-1] - self.initial_capital) / self.initial_capital

        # Annualized return
        n_years = len(portfolio_value) / 252
        annualized_return = (1 + total_return) ** (1 / n_years) - 1

        # Sharpe ratio
        from .metrics import portfolio_sharpe, max_drawdown

        sharpe = portfolio_sharpe(daily_returns[1:])
        max_dd = max_drawdown(portfolio_value)

        # Win rate
        positive_days = (daily_returns > 0).sum()
        win_rate = positive_days / len(daily_returns)

        # Average turnover
        avg_turnover = turnover[turnover > 0].mean()

        results = {
            'final_value': portfolio_value[-1],
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'avg_turnover': avg_turnover,
            'portfolio_values': portfolio_value,
            'daily_returns': daily_returns
        }

        return results

    def print_backtest_summary(self, results: Dict):
        """Print formatted backtest summary."""
        print("=" * 80)
        print("Backtest Results Summary")
        print("=" * 80)
        print(f"Initial Capital:       ₹{self.initial_capital:,.0f}")
        print(f"Final Value:           ₹{results['final_value']:,.0f}")
        print(f"Total Return:          {results['total_return']:.2%}")
        print(f"Annualized Return:     {results['annualized_return']:.2%}")
        print(f"Sharpe Ratio:          {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown:          {results['max_drawdown']:.2%}")
        print(f"Win Rate:              {results['win_rate']:.2%}")
        print(f"Avg Turnover:          {results['avg_turnover']:.2%}")
        print(f"Transaction Cost:      {self.total_cost_pct:.4%} (round-trip)")
        print("=" * 80)


if __name__ == "__main__":
    # Test backtester
    np.random.seed(42)

    T, N = 252, 400  # 1 year, 400 stocks
    dates = pd.date_range('2023-01-01', periods=T, freq='B')

    # Generate synthetic predictions and actuals
    actuals = np.random.randn(T, N) * 0.02  # 2% daily vol
    predictions = actuals + np.random.randn(T, N) * 0.015  # Add noise

    stock_symbols = [f"STOCK{i}" for i in range(N)]

    # Run backtest
    backtester = IndianMarketBacktester(
        predictions=predictions,
        actuals=actuals,
        dates=dates,
        stock_symbols=stock_symbols
    )

    results = backtester.run_topk_strategy(K=20, rebalance_freq='weekly')

    backtester.print_backtest_summary(results)

    print("\nBacktester test passed!")
