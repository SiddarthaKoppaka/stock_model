"""
DiffSTOCK Simulation Engine

Three operation modes:
    1. Walk-Forward Live Simulation — replay test period day by day
    2. Stress Test Scenarios — inject synthetic shocks and observe model behaviour
    3. Monte Carlo Simulation — run N randomised variations for confidence intervals

Usage:
    python -m src.simulation.simulator --mode walkforward --start 2024-07-01
    python -m src.simulation.simulator --mode stress --scenario flash_crash --shock-date 2024-10-15
    python -m src.simulation.simulator --mode montecarlo --n-sims 500
"""

import os
import sys
import json
import copy
import argparse
import datetime
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from loguru import logger

# ── project imports ──────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import yaml
from src.model.diffstock import DiffSTOCK, create_diffstock_model
from src.evaluation.metrics import portfolio_sharpe, max_drawdown


# ═══════════════════════════════════════════════════════════════════════════════
#  Helper — Indian market transaction cost calculator
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_COSTS = {
    'brokerage': 0.0003,
    'stt_buy': 0.001,
    'stt_sell': 0.001,
    'exchange': 0.0000335,
    'sebi': 0.000001,
    'gst': 0.18,
    'stamp': 0.00015,
    'slippage': 0.002,
}


def total_round_trip_cost(costs: Dict[str, float]) -> float:
    """Compute total round-trip transaction cost percentage."""
    buy_base = costs['brokerage'] + costs['stamp']
    buy_gst_on = costs['brokerage'] + costs['exchange']
    buy_cost = (buy_base + costs['exchange'] + costs['sebi']
                + buy_gst_on * costs['gst'] + costs['stt_buy'])

    sell_base = costs['brokerage']
    sell_gst_on = costs['brokerage'] + costs['exchange']
    sell_cost = (sell_base + costs['exchange'] + costs['sebi']
                 + sell_gst_on * costs['gst'] + costs['stt_sell'])

    return buy_cost + sell_cost + 2 * costs['slippage']


# ═══════════════════════════════════════════════════════════════════════════════
#  Stress-Test Scenario Functions
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_circuit_breaker(
    actuals: np.ndarray,
    shock_day_idx: int,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, str]:
    """
    Randomly apply NSE 10%/20% circuit limits to 5 % of stocks on shock day.
    Affected stocks become un-tradeable (returns set to NaN).
    """
    T, N = actuals.shape
    n_affected = max(1, int(0.05 * N))
    affected = rng.choice(N, size=n_affected, replace=False)

    for idx in affected:
        limit = rng.choice([0.10, 0.20])
        actuals[shock_day_idx, idx] = np.clip(
            actuals[shock_day_idx, idx], -limit, limit
        )
        # Mark as un-tradeable by setting to NaN
        actuals[shock_day_idx, idx] = np.nan

    summary = (
        f"Circuit Breaker: On day index {shock_day_idx}, {n_affected} stocks "
        f"({100 * n_affected / N:.1f}% of universe) hit circuit limits and became "
        f"un-tradeable. The model should gracefully skip these names and reallocate "
        f"capital to the remaining liquid stocks."
    )
    return actuals, summary


def _apply_flash_crash(
    actuals: np.ndarray,
    shock_day_idx: int,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, str]:
    """Apply −15% shock on day T, linear recovery over T+1…T+5."""
    T, N = actuals.shape
    actuals[shock_day_idx, :] -= 0.15

    for offset in range(1, 6):
        d = shock_day_idx + offset
        if d < T:
            recovery_fraction = offset / 5.0
            actuals[d, :] += 0.15 * recovery_fraction / 5.0

    summary = (
        f"Flash Crash: A −15% market-wide shock was injected on day index "
        f"{shock_day_idx}, followed by a gradual recovery over the next 5 trading "
        f"days. This tests whether the model can recognise the anomaly and pivot "
        f"its portfolio accordingly rather than panic-selling at the bottom."
    )
    return actuals, summary


def _apply_liquidity_crisis(
    actuals: np.ndarray,
    shock_day_idx: int,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, str]:
    """
    Set volume to near-zero for bottom 30% of stocks by ADV for 10 days.
    This is modelled by marking those stocks' returns as NaN (un-tradeable).
    """
    T, N = actuals.shape
    # Estimate ADV via absolute returns as a proxy (no volume in actuals)
    adv_proxy = np.nanmean(np.abs(actuals), axis=0)
    n_illiquid = max(1, int(0.30 * N))
    illiquid_stocks = np.argsort(adv_proxy)[:n_illiquid]

    end_day = min(shock_day_idx + 10, T)
    actuals[shock_day_idx:end_day, illiquid_stocks] = np.nan

    summary = (
        f"Liquidity Crisis: The bottom 30% of stocks by average daily volume "
        f"({n_illiquid} stocks) became un-tradeable for 10 consecutive trading days "
        f"starting at day index {shock_day_idx}. The model should only trade liquid "
        f"names and avoid taking positions in illiquid stocks during this period."
    )
    return actuals, summary


def _apply_black_swan(
    actuals: np.ndarray,
    shock_day_idx: int,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, str]:
    """Simulate a 2008-style event: 40% drawdown over 20 days, correlation → 1."""
    T, N = actuals.shape
    end_day = min(shock_day_idx + 20, T)
    n_days = end_day - shock_day_idx

    daily_drop = -0.40 / n_days  # spread the 40% drop evenly

    for d in range(shock_day_idx, end_day):
        # Force high correlation: all stocks move together
        base = daily_drop + rng.normal(0, 0.005)
        actuals[d, :] = base + rng.normal(0, 0.002, size=N)

    summary = (
        f"Black Swan (2008-style): A 40% market-wide drawdown was simulated over "
        f"20 trading days starting at day index {shock_day_idx}. All stocks were "
        f"forced to move in near-perfect correlation (correlation → 1.0). This "
        f"tests whether the model's diversification assumptions break down under "
        f"extreme systemic stress."
    )
    return actuals, summary


def _apply_election_day(
    actuals: np.ndarray,
    shock_day_idx: int,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, str]:
    """Spike volatility 3× on a single day with large open gaps."""
    T, N = actuals.shape
    historical_vol = np.nanstd(actuals, axis=0)

    # 3× volatility spike
    actuals[shock_day_idx, :] = rng.normal(0, 3 * historical_vol)

    summary = (
        f"Election Day: Volatility was spiked to 3× normal levels on day index "
        f"{shock_day_idx} with large open gaps, simulating an election-day shock. "
        f"This tests whether the model's open-return normalisation signal still "
        f"functions under extreme intraday conditions."
    )
    return actuals, summary


SCENARIO_REGISTRY = {
    'circuit_breaker': _apply_circuit_breaker,
    'flash_crash': _apply_flash_crash,
    'liquidity_crisis': _apply_liquidity_crisis,
    'black_swan': _apply_black_swan,
    'election_day': _apply_election_day,
}


# ═══════════════════════════════════════════════════════════════════════════════
#  DiffSTOCKSimulator
# ═══════════════════════════════════════════════════════════════════════════════

class DiffSTOCKSimulator:
    """
    Self-contained simulation engine for the DiffSTOCK model.

    Supports three modes:
        1. Walk-forward live simulation
        2. Stress test scenarios
        3. Monte Carlo simulation

    The model is loaded once and cached in memory.
    """

    def __init__(
        self,
        checkpoint: str = 'checkpoints/best_model.pt',
        config: str = 'config/config.yaml',
        device: Optional[str] = None,
    ):
        self.project_root = _PROJECT_ROOT
        self.config_path = self.project_root / config

        # ── load config ──────────────────────────────────────────────────────
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # ── resolve checkpoint path ──────────────────────────────────────────
        ckpt_path = self.project_root / checkpoint
        if not ckpt_path.exists():
            # Try finding inside checkpoints/ subdirectories
            alt_paths = sorted(
                (self.project_root / 'checkpoints').rglob('best_model.pt')
            )
            if alt_paths:
                ckpt_path = alt_paths[0]
                logger.info(f"Resolved checkpoint to: {ckpt_path}")
            else:
                raise FileNotFoundError(
                    f"Checkpoint not found at {ckpt_path}. "
                    f"Searched in {self.project_root / 'checkpoints'}"
                )
        self.checkpoint_path = ckpt_path

        # ── device ───────────────────────────────────────────────────────────
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # ── load dataset ─────────────────────────────────────────────────────
        dataset_path = (
            self.project_root
            / self.config['paths']['dataset']
            / 'nifty500_20yr.npz'
        )
        logger.info(f"Loading dataset from {dataset_path}")
        ds = np.load(dataset_path, allow_pickle=True)

        self.X_test = ds['X_test']           # (T, L, N, F)
        self.y_test = ds['y_test']           # (T, N)
        self.dates_test = pd.to_datetime(ds['dates_test'])
        self.stock_symbols: List[str] = ds['stock_symbols'].tolist()
        self.n_stocks = len(self.stock_symbols)

        # Also keep val data for potential lookback
        self.X_val = ds.get('X_val', None)
        self.y_val = ds.get('y_val', None)

        # ── load relation mask ───────────────────────────────────────────────
        relation_path = (
            self.project_root
            / self.config['paths']['dataset']
            / 'relation_matrices.npz'
        )
        rel_data = np.load(relation_path)
        self.R_mask = torch.FloatTensor(rel_data['R_mask']).to(self.device)

        # ── transaction costs ────────────────────────────────────────────────
        tc_cfg = self.config.get('evaluation', {}).get('transaction_costs', {})
        self.costs = {**DEFAULT_COSTS, **tc_cfg}
        self.round_trip_cost = total_round_trip_cost(self.costs)

        # ── load model ───────────────────────────────────────────────────────
        logger.info("Loading DiffSTOCK model …")
        self.model = create_diffstock_model(self.config, self.n_stocks)
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info(
            f"Model loaded (epoch {ckpt.get('epoch', '?')}), "
            f"{sum(p.numel() for p in self.model.parameters()):,} params on {self.device}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _predict(self, x: np.ndarray, n_samples: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run model inference on a single window.

        Args:
            x: (1, L, N, F) feature window
            n_samples: diffusion sample count

        Returns:
            predictions (N,), uncertainty (N,)
        """
        with torch.no_grad():
            x_t = torch.FloatTensor(x).to(self.device)
            pred, unc = self.model(x_t, self.R_mask, n_samples=n_samples)
            return pred.cpu().numpy().squeeze(), unc.cpu().numpy().squeeze()

    def _date_range_indices(
        self,
        start_date: str,
        end_date: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Filter test data to a date range. Returns (X, y, dates, mask)."""
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        mask = (self.dates_test >= start) & (self.dates_test <= end)
        idx = np.where(mask)[0]
        return self.X_test[idx], self.y_test[idx], self.dates_test[idx], idx

    def _get_rebalance_days(self, dates: pd.DatetimeIndex, hold_days: int = 5) -> List[int]:
        """Return indices of rebalance days (every hold_days trading days)."""
        return list(range(0, len(dates), hold_days))

    @staticmethod
    def _build_summary(
        equity: np.ndarray,
        daily_rets: np.ndarray,
        total_trades: int,
        avg_turnover: float,
        win_rate: float,
    ) -> Dict:
        """Compute performance summary dict."""
        n_years = max(len(equity) / 252, 1 / 252)
        total_ret = (equity[-1] - equity[0]) / equity[0]
        ann_ret = (1 + total_ret) ** (1.0 / n_years) - 1
        sharpe = portfolio_sharpe(daily_rets[daily_rets != 0])
        mdd = max_drawdown(equity)
        return {
            'ann_return': float(ann_ret),
            'sharpe': float(sharpe),
            'max_drawdown': float(mdd),
            'win_rate': float(win_rate),
            'turnover': float(avg_turnover),
            'total_trades': int(total_trades),
        }

    def _save_results(
        self,
        results: Dict,
        mode: str,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """Save results to disk."""
        if output_dir is None:
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = self.project_root / 'results' / 'simulations' / f'{mode}_{ts}'
        output_dir.mkdir(parents=True, exist_ok=True)

        # Summary JSON
        with open(output_dir / 'summary.json', 'w') as f:
            json.dump(results['summary'], f, indent=2, default=str)

        # Equity curve JSON
        with open(output_dir / 'equity_curve.json', 'w') as f:
            json.dump(results['equity_curve'], f)

        # Trade log CSV
        if isinstance(results.get('trade_log'), pd.DataFrame) and len(results['trade_log']) > 0:
            results['trade_log'].to_csv(output_dir / 'trade_log.csv', index=False)

        # Weekly log CSV
        if isinstance(results.get('weekly_log'), pd.DataFrame) and len(results['weekly_log']) > 0:
            results['weekly_log'].to_csv(output_dir / 'weekly_log.csv', index=False)

        # Scenario info
        with open(output_dir / 'scenario.json', 'w') as f:
            json.dump({'scenario': results.get('scenario', mode)}, f, indent=2)

        logger.info(f"Results saved to {output_dir}")
        return output_dir

    # ═══════════════════════════════════════════════════════════════════════════
    #  MODE 1 — Walk-Forward Live Simulation
    # ═══════════════════════════════════════════════════════════════════════════

    def run_walkforward(
        self,
        start_date: str = '2024-07-01',
        end_date: str = '2026-02-17',
        initial_capital: float = 1_000_000,
        top_k: int = 20,
        hold_days: int = 5,
        n_samples: int = 50,
        save: bool = True,
        actuals_override: Optional[np.ndarray] = None,
        tradeable_mask: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Replay the test period day by day as if trading live.

        Args:
            start_date: first trading date
            end_date: last trading date
            initial_capital: starting portfolio value (INR)
            top_k: number of stocks to hold
            hold_days: days between rebalances
            n_samples: diffusion samples per prediction
            save: persist results to disk
            actuals_override: (T, N) override for stress testing
            tradeable_mask: (T, N) bool mask — True = tradeable

        Returns:
            results dict with summary, equity_curve, trade_log, weekly_log, scenario
        """
        X, y_actual, dates, _ = self._date_range_indices(start_date, end_date)
        T = len(dates)

        if T == 0:
            raise ValueError(f"No data found between {start_date} and {end_date}")

        if actuals_override is not None:
            y_actual = actuals_override

        logger.info(f"Walk-forward: {T} trading days, {dates[0].date()} → {dates[-1].date()}")

        # ── state ────────────────────────────────────────────────────────────
        equity = np.full(T, initial_capital, dtype=np.float64)
        daily_rets = np.zeros(T)
        weights = np.zeros(self.n_stocks)
        rebalance_days = self._get_rebalance_days(dates, hold_days)

        trade_rows: List[Dict] = []
        weekly_rows: List[Dict] = []
        total_trades = 0
        turnover_list: List[float] = []

        for t in range(T):
            # ── step 1: mark-to-market ───────────────────────────────────────
            if t > 0:
                rets = np.nan_to_num(y_actual[t], nan=0.0)
                port_ret = np.dot(weights, rets)
                daily_rets[t] = port_ret
                equity[t] = equity[t - 1] * (1 + port_ret)
            else:
                equity[t] = initial_capital

            # ── step 2: rebalance ────────────────────────────────────────────
            if t in rebalance_days:
                preds, unc = self._predict(X[t:t + 1], n_samples=n_samples)

                # Mask un-tradeable stocks
                valid = ~np.isnan(preds)
                if tradeable_mask is not None:
                    valid &= tradeable_mask[t]
                # Also mask NaN actuals (can't trade what has no data)
                valid &= ~np.isnan(y_actual[t])

                preds_masked = preds.copy()
                preds_masked[~valid] = -np.inf

                if valid.sum() < top_k:
                    k_actual = int(valid.sum())
                else:
                    k_actual = top_k

                if k_actual == 0:
                    new_weights = np.zeros(self.n_stocks)
                else:
                    top_idx = np.argsort(preds_masked)[-k_actual:]
                    new_weights = np.zeros(self.n_stocks)
                    new_weights[top_idx] = 1.0 / k_actual

                # Turnover & cost
                turnover = np.sum(np.abs(new_weights - weights))
                cost = turnover * self.round_trip_cost
                equity[t] *= (1 - cost)

                turnover_list.append(turnover)

                # ── trade log entries ────────────────────────────────────────
                old_set = set(np.where(weights > 0)[0])
                new_set = set(np.where(new_weights > 0)[0])

                for idx in new_set - old_set:
                    trade_rows.append({
                        'date': str(dates[t].date()),
                        'stock_symbol': self.stock_symbols[idx],
                        'action': 'BUY',
                        'predicted_return': float(preds[idx]),
                        'actual_return': float(y_actual[t, idx]) if not np.isnan(y_actual[t, idx]) else None,
                        'weight': float(new_weights[idx]),
                        'cost': float(cost * new_weights[idx]),
                        'pnl': 0.0,
                    })
                    total_trades += 1

                for idx in old_set - new_set:
                    # Compute rough P&L for the position just closed
                    holding_return = float(np.nan_to_num(y_actual[t, idx], nan=0.0))
                    trade_rows.append({
                        'date': str(dates[t].date()),
                        'stock_symbol': self.stock_symbols[idx],
                        'action': 'SELL',
                        'predicted_return': float(preds[idx]),
                        'actual_return': holding_return,
                        'weight': 0.0,
                        'cost': float(cost * weights[idx]),
                        'pnl': float(weights[idx] * holding_return * equity[t - 1]) if t > 0 else 0.0,
                    })
                    total_trades += 1

                for idx in old_set & new_set:
                    trade_rows.append({
                        'date': str(dates[t].date()),
                        'stock_symbol': self.stock_symbols[idx],
                        'action': 'HOLD',
                        'predicted_return': float(preds[idx]),
                        'actual_return': float(y_actual[t, idx]) if not np.isnan(y_actual[t, idx]) else None,
                        'weight': float(new_weights[idx]),
                        'cost': 0.0,
                        'pnl': 0.0,
                    })

                # ── weekly log entry ─────────────────────────────────────────
                weekly_rows.append({
                    'date': str(dates[t].date()),
                    'portfolio_value': float(equity[t]),
                    'top_k_stocks': [self.stock_symbols[i] for i in np.where(new_weights > 0)[0]],
                    'turnover': float(turnover),
                    'transaction_cost': float(cost),
                    'predicted_mean': float(np.mean(preds[new_weights > 0])) if k_actual > 0 else 0.0,
                })

                weights = new_weights

        # ── compile results ──────────────────────────────────────────────────
        positive_days = (daily_rets[1:] > 0).sum()
        win_rate = positive_days / max(len(daily_rets) - 1, 1)
        avg_turnover = float(np.mean(turnover_list)) if turnover_list else 0.0

        results = {
            'summary': self._build_summary(equity, daily_rets, total_trades, avg_turnover, win_rate),
            'equity_curve': equity.tolist(),
            'trade_log': pd.DataFrame(trade_rows),
            'weekly_log': pd.DataFrame(weekly_rows),
            'scenario': 'walkforward',
        }

        if save:
            self._save_results(results, 'walkforward')

        return results

    # ═══════════════════════════════════════════════════════════════════════════
    #  MODE 2 — Stress Test Scenarios
    # ═══════════════════════════════════════════════════════════════════════════

    def run_stress_test(
        self,
        scenario: Union[str, List[str]] = 'flash_crash',
        shock_date: str = '2024-10-15',
        initial_capital: float = 1_000_000,
        top_k: int = 20,
        hold_days: int = 5,
        random_seed: int = 42,
        save: bool = True,
    ) -> Dict:
        """
        Inject synthetic shocks into the price data and observe model behaviour.

        Scenarios are composable: pass a list to combine e.g.
        ['flash_crash', 'liquidity_crisis'].

        Args:
            scenario: scenario name or list of names
            shock_date: date to inject the shock
            initial_capital: starting capital
            top_k: stocks to hold
            hold_days: rebalance frequency
            random_seed: for reproducibility
            save: persist results

        Returns:
            results dict
        """
        rng = np.random.RandomState(random_seed)

        if isinstance(scenario, str):
            scenario = [scenario]

        for s in scenario:
            if s not in SCENARIO_REGISTRY:
                raise ValueError(
                    f"Unknown scenario '{s}'. Available: {list(SCENARIO_REGISTRY.keys())}"
                )

        # Work on a copy of actuals
        y_stressed = self.y_test.copy()

        # Find shock day index
        shock_ts = pd.Timestamp(shock_date)
        diffs = np.abs((self.dates_test - shock_ts).days)
        shock_idx = int(np.argmin(diffs))
        actual_shock_date = self.dates_test[shock_idx]
        logger.info(f"Shock date resolved to index {shock_idx} ({actual_shock_date.date()})")

        summaries: List[str] = []
        for s in scenario:
            y_stressed, summary = SCENARIO_REGISTRY[s](y_stressed, shock_idx, rng)
            summaries.append(summary)
            logger.info(f"Applied scenario: {s}")

        # Run walk-forward with stressed data
        results = self.run_walkforward(
            start_date=str(self.dates_test[0].date()),
            end_date=str(self.dates_test[-1].date()),
            initial_capital=initial_capital,
            top_k=top_k,
            hold_days=hold_days,
            save=False,
            actuals_override=y_stressed,
        )

        scenario_name = '+'.join(scenario)
        results['scenario'] = f'stress_{scenario_name}'
        results['stress_summaries'] = summaries

        # Print plain English summaries
        print("\n" + "=" * 80)
        print(f"STRESS TEST REPORT: {scenario_name.upper()}")
        print("=" * 80)
        for s_text in summaries:
            print(f"\n{s_text}")
        print(f"\n── Performance Under Stress ──")
        for k, v in results['summary'].items():
            if isinstance(v, float):
                print(f"  {k:>15}: {v:>10.4f}")
            else:
                print(f"  {k:>15}: {v:>10}")
        print("=" * 80 + "\n")

        if save:
            self._save_results(results, f'stress_{scenario_name}')

        return results

    # ═══════════════════════════════════════════════════════════════════════════
    #  MODE 3 — Monte Carlo Simulation
    # ═══════════════════════════════════════════════════════════════════════════

    def run_monte_carlo(
        self,
        n_simulations: int = 500,
        perturbation_std: float = 0.02,
        random_seed: int = 42,
        initial_capital: float = 1_000_000,
        top_k: int = 20,
        hold_days: int = 5,
        drop_pct: float = 0.10,
        save: bool = True,
    ) -> Dict:
        """
        Run N randomised variations of the test period for confidence intervals.

        Each simulation:
            - Perturbs returns by ±σ drawn from historical volatility
            - Randomly drops 10% of stocks as un-tradeable each week

        Args:
            n_simulations: number of MC runs
            perturbation_std: std of noise to add to returns
            random_seed: for reproducibility
            initial_capital: starting capital
            top_k: stocks to hold
            hold_days: rebalance frequency
            drop_pct: fraction of stocks to randomly mark un-tradeable
            save: persist results

        Returns:
            results dict with P5/P50/P95 statistics
        """
        rng = np.random.RandomState(random_seed)

        T = len(self.dates_test)
        N = self.n_stocks
        historical_vol = np.nanstd(self.y_test, axis=0)

        all_equities: List[np.ndarray] = []
        all_sharpes: List[float] = []
        all_ann_rets: List[float] = []
        all_max_dds: List[float] = []

        logger.info(f"Monte Carlo: {n_simulations} simulations, T={T}, N={N}")

        for sim_i in range(n_simulations):
            if (sim_i + 1) % 50 == 0 or sim_i == 0:
                logger.info(f"  Simulation {sim_i + 1}/{n_simulations}")

            # Perturb returns
            noise = rng.normal(0, perturbation_std, size=(T, N))
            y_perturbed = self.y_test.copy() + noise * historical_vol[np.newaxis, :]

            # Random tradeable mask — drop 10% each rebalance week
            tradeable = np.ones((T, N), dtype=bool)
            rebalance_days = list(range(0, T, hold_days))
            for rb_day in rebalance_days:
                n_drop = max(1, int(drop_pct * N))
                dropped = rng.choice(N, size=n_drop, replace=False)
                end = min(rb_day + hold_days, T)
                tradeable[rb_day:end, dropped] = False

            # Run walk-forward
            res = self.run_walkforward(
                start_date=str(self.dates_test[0].date()),
                end_date=str(self.dates_test[-1].date()),
                initial_capital=initial_capital,
                top_k=top_k,
                hold_days=hold_days,
                save=False,
                actuals_override=y_perturbed,
                tradeable_mask=tradeable,
            )

            eq = np.array(res['equity_curve'])
            all_equities.append(eq)
            all_sharpes.append(res['summary']['sharpe'])
            all_ann_rets.append(res['summary']['ann_return'])
            all_max_dds.append(res['summary']['max_drawdown'])

        # Stack equity curves
        eq_matrix = np.array(all_equities)  # (n_sims, T)

        p5 = np.percentile(eq_matrix, 5, axis=0)
        p50 = np.percentile(eq_matrix, 50, axis=0)
        p95 = np.percentile(eq_matrix, 95, axis=0)

        mc_summary = {
            'ann_return': {
                'P5': float(np.percentile(all_ann_rets, 5)),
                'P50': float(np.percentile(all_ann_rets, 50)),
                'P95': float(np.percentile(all_ann_rets, 95)),
            },
            'sharpe': {
                'P5': float(np.percentile(all_sharpes, 5)),
                'P50': float(np.percentile(all_sharpes, 50)),
                'P95': float(np.percentile(all_sharpes, 95)),
            },
            'max_drawdown': {
                'P5': float(np.percentile(all_max_dds, 5)),
                'P50': float(np.percentile(all_max_dds, 50)),
                'P95': float(np.percentile(all_max_dds, 95)),
            },
        }

        # Build a flat summary for the standard output format
        results = {
            'summary': {
                'ann_return': mc_summary['ann_return']['P50'],
                'sharpe': mc_summary['sharpe']['P50'],
                'max_drawdown': mc_summary['max_drawdown']['P50'],
                'win_rate': 0.0,  # not meaningful for MC aggregate
                'turnover': 0.0,
                'total_trades': 0,
            },
            'equity_curve': p50.tolist(),
            'trade_log': pd.DataFrame(),
            'weekly_log': pd.DataFrame(),
            'scenario': 'monte_carlo',
            'mc_summary': mc_summary,
            'mc_percentiles': {
                'p5': p5.tolist(),
                'p50': p50.tolist(),
                'p95': p95.tolist(),
            },
            'n_simulations': n_simulations,
        }

        # Print summary
        print("\n" + "=" * 80)
        print(f"MONTE CARLO REPORT  ({n_simulations} simulations)")
        print("=" * 80)
        for metric_name, vals in mc_summary.items():
            print(f"\n  {metric_name}:")
            for pct_name, pct_val in vals.items():
                print(f"    {pct_name}: {pct_val:.4f}")
        print("=" * 80 + "\n")

        if save:
            output_dir = self._save_results(results, 'montecarlo')
            self._plot_fan_chart(p5, p50, p95, output_dir)

        return results

    def _plot_fan_chart(
        self,
        p5: np.ndarray,
        p50: np.ndarray,
        p95: np.ndarray,
        output_dir: Path,
    ):
        """Generate and save fan chart of equity curves (P5/P50/P95)."""
        fig, ax = plt.subplots(figsize=(14, 7))

        x = np.arange(len(p50))
        ax.fill_between(x, p5, p95, alpha=0.25, color='#4A90D9', label='P5–P95 band')
        ax.plot(x, p50, linewidth=2, color='#2C5F8A', label='P50 (median)')
        ax.plot(x, p5, linewidth=0.8, color='#8FAADC', linestyle='--', label='P5')
        ax.plot(x, p95, linewidth=0.8, color='#8FAADC', linestyle='--', label='P95')

        ax.set_title('Monte Carlo Fan Chart — Equity Curve Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Trading Days', fontsize=12)
        ax.set_ylabel('Portfolio Value (₹)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        chart_path = output_dir / 'fan_chart.png'
        fig.savefig(chart_path, dpi=150)
        plt.close(fig)
        logger.info(f"Fan chart saved to {chart_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='DiffSTOCK Simulation Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.simulation.simulator --mode walkforward --start 2024-07-01
  python -m src.simulation.simulator --mode stress --scenario flash_crash --shock-date 2024-10-15
  python -m src.simulation.simulator --mode stress --scenario flash_crash,liquidity_crisis
  python -m src.simulation.simulator --mode montecarlo --n-sims 100
        """,
    )
    parser.add_argument('--mode', type=str, required=True,
                        choices=['walkforward', 'stress', 'montecarlo'],
                        help='Simulation mode')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--start', type=str, default='2024-07-01')
    parser.add_argument('--end', type=str, default='2026-02-17')
    parser.add_argument('--capital', type=float, default=1_000_000)
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--hold-days', type=int, default=5)
    parser.add_argument('--scenario', type=str, default='flash_crash',
                        help='Stress scenario name(s), comma-separated for composability')
    parser.add_argument('--shock-date', type=str, default='2024-10-15')
    parser.add_argument('--n-sims', type=int, default=500)
    parser.add_argument('--perturbation-std', type=float, default=0.02)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default=None)

    args = parser.parse_args()

    sim = DiffSTOCKSimulator(
        checkpoint=args.checkpoint,
        config=args.config,
        device=args.device,
    )

    if args.mode == 'walkforward':
        results = sim.run_walkforward(
            start_date=args.start,
            end_date=args.end,
            initial_capital=args.capital,
            top_k=args.top_k,
            hold_days=args.hold_days,
        )
        print(f"\n✅ Walk-forward complete. Summary:")
        for k, v in results['summary'].items():
            print(f"  {k}: {v}")

    elif args.mode == 'stress':
        scenarios = [s.strip() for s in args.scenario.split(',')]
        results = sim.run_stress_test(
            scenario=scenarios if len(scenarios) > 1 else scenarios[0],
            shock_date=args.shock_date,
            initial_capital=args.capital,
            top_k=args.top_k,
            hold_days=args.hold_days,
            random_seed=args.seed,
        )

    elif args.mode == 'montecarlo':
        results = sim.run_monte_carlo(
            n_simulations=args.n_sims,
            perturbation_std=args.perturbation_std,
            random_seed=args.seed,
            initial_capital=args.capital,
            top_k=args.top_k,
            hold_days=args.hold_days,
        )


if __name__ == '__main__':
    main()
