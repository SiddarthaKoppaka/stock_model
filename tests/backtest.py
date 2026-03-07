"""
DiffSTOCK — Aligned Backtest
==============================
The signal predicts 5-day forward returns.
So we must:
  1. Predict on day T
  2. Enter position on day T
  3. EXIT on day T+5 (hold for the full prediction window)
  4. Only then predict again and rebalance

This is the correct backtest for a 5-day label model.
The previous backtest was checking predictions daily against a weekly portfolio
— causing the -0.005 rank autocorrelation (the model IS right, just for 5 days).

Run AFTER your inference cell.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import os, json
import warnings
warnings.filterwarnings('ignore')

# ── INPUTS ────────────────────────────────────────────────────────────────────
preds  = test_predictions    # (T, N) — predicted 5-day return from each day t
actual = test_targets         # (T, N) — actual 5-day compound return from day t
dates  = pd.to_datetime(data['dates_test'])
T, N   = preds.shape
K      = config['evaluation']['top_k']   # 20

INITIAL_CAP     = 1_000_000
ROUND_TRIP_COST = 0.006939   # full round trip
COST_BUY        = 0.003469
COST_SELL       = 0.003470
RESULTS_DIR     = "/content/drive/MyDrive/DiffSTOCK_Outputs/aligned_backtest"
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Test period: {dates[0].date()} → {dates[-1].date()}")
print(f"Days: {T}, Stocks: {N}, K: {K}")

# ── ALIGNED 5-DAY HOLD BACKTEST ───────────────────────────────────────────────
def run_aligned_backtest(preds, actual, K, initial_cap, hold_days=5,
                          top_pct=None, buy_pct=None, sell_pct=None,
                          name="Aligned"):
    """
    Correct backtest for 5-day label model.

    On each rebalance day t:
      - Use preds[t] to select top-K stocks
      - Hold those stocks for exactly `hold_days`
      - On day t+hold_days, collect 5-day actual return, rebalance

    buy_pct / sell_pct: optional banding on top of aligned holding
    """
    T, N = preds.shape
    n_buy  = K if buy_pct is None else max(K, int(N * buy_pct))
    n_hold = K if sell_pct is None else int(N * sell_pct)

    portfolio_value  = initial_cap
    equity_curve     = [initial_cap]
    rebal_returns    = []
    overlaps         = []
    trade_counts     = []

    prev_holdings = None
    t = 0

    while t + hold_days <= T:
        # ── Predict on day t ─────────────────────────────────────────────────
        pred_t = preds[t]
        valid  = ~np.isnan(pred_t)

        ranked     = np.where(valid)[0][np.argsort(-pred_t[valid])]
        top_stocks = set(ranked[:n_buy].tolist())

        # Banding: if we have prev holdings, only exit stocks below sell zone
        if prev_holdings is not None and sell_pct is not None:
            hold_zone  = set(ranked[:n_hold].tolist())
            forced_out = prev_holdings - hold_zone
            forced_in  = top_stocks - prev_holdings
            holdings   = (prev_holdings - forced_out) | forced_in
            # Cap at K
            if len(holdings) > K:
                holdings = set(list(holdings)[:K])
        else:
            holdings = set(ranked[:K].tolist())

        n_trades = len(holdings.symmetric_difference(prev_holdings)) \
                   if prev_holdings else len(holdings)

        # ── Collect actual 5-day return ───────────────────────────────────────
        # actual[t] = compound return from day t to t+5
        actual_t = actual[t]
        held_list = list(holdings)
        if len(held_list) > 0:
            gross_ret = float(np.nanmean(actual_t[held_list]))
        else:
            gross_ret = 0.0

        # Cost: round-trip on traded stocks only
        pct_traded = n_trades / max(len(holdings), 1)
        cost       = pct_traded * ROUND_TRIP_COST
        net_ret    = gross_ret - cost

        portfolio_value *= (1 + net_ret)
        equity_curve.append(portfolio_value)
        rebal_returns.append(net_ret)
        trade_counts.append(n_trades)

        if prev_holdings:
            overlaps.append(len(holdings & prev_holdings))

        prev_holdings = holdings
        t += hold_days   # move forward by hold_days

    # ── Metrics ───────────────────────────────────────────────────────────────
    rebal_returns = np.array(rebal_returns)
    equity_curve  = np.array(equity_curve)

    n_periods  = len(rebal_returns)
    n_years    = n_periods * hold_days / 252
    total_ret  = (equity_curve[-1] - initial_cap) / initial_cap
    ann_ret    = (1 + total_ret) ** (1 / n_years) - 1

    # Sharpe (per-period, annualised to 52 weeks)
    periods_per_year = 252 / hold_days
    sharpe = (rebal_returns.mean() / (rebal_returns.std() + 1e-8)) * np.sqrt(periods_per_year)

    running_max = np.maximum.accumulate(equity_curve)
    max_dd      = float(((equity_curve - running_max) / (running_max + 1e-8)).min())
    win_rate    = float((rebal_returns > 0).mean())

    avg_trades  = float(np.mean(trade_counts))
    ann_turnover = avg_trades / K * periods_per_year * 100   # annualised %
    avg_overlap  = float(np.mean(overlaps)) if overlaps else 0.0

    return {
        "name"         : name,
        "ann_ret"      : ann_ret,
        "sharpe"       : sharpe,
        "max_dd"       : max_dd,
        "win_rate"     : win_rate,
        "ann_turnover" : ann_turnover,
        "avg_overlap"  : avg_overlap,
        "n_periods"    : n_periods,
        "equity_curve" : equity_curve,
        "rebal_returns": rebal_returns,
        "avg_trades"   : avg_trades,
    }


# ── VARIANTS ──────────────────────────────────────────────────────────────────
variants = [
    # Hold days variants — same top-K strict selection
    dict(name="5-day hold, strict top-20",    hold_days=5,  buy_pct=None, sell_pct=None),
    dict(name="10-day hold, strict top-20",   hold_days=10, buy_pct=None, sell_pct=None),
    dict(name="5-day hold, band 6%/25%",      hold_days=5,  buy_pct=0.061,sell_pct=0.25),
    dict(name="5-day hold, band 6%/40%",      hold_days=5,  buy_pct=0.061,sell_pct=0.40),
    dict(name="5-day hold, band 10%/40%",     hold_days=5,  buy_pct=0.10, sell_pct=0.40),
    dict(name="10-day hold, band 6%/40%",     hold_days=10, buy_pct=0.061,sell_pct=0.40),
]

print(f"\n{'Variant':<35} {'Ann Ret':>9} {'Sharpe':>8} {'MaxDD':>8} "
      f"{'WinRate':>8} {'Turnover':>10} {'Overlap':>9} {'Periods':>8}")
print("-" * 105)

all_results = []
best, best_name = None, None

for v in variants:
    r = run_aligned_backtest(
        preds     = preds,
        actual    = actual,
        K         = K,
        initial_cap = INITIAL_CAP,
        hold_days = v['hold_days'],
        buy_pct   = v.get('buy_pct'),
        sell_pct  = v.get('sell_pct'),
        name      = v['name'],
    )
    all_results.append(r)

    print(f"{r['name']:<35} {r['ann_ret']:>+9.2%} {r['sharpe']:>8.2f} "
          f"{r['max_dd']:>8.2%} {r['win_rate']:>7.2f}% "
          f"{r['ann_turnover']:>9.1f}% "
          f"{r['avg_overlap']:>6.1f}/{K} {r['n_periods']:>8}")

    if best is None or r['sharpe'] > best['sharpe']:
        best, best_name = r, r['name']

print("=" * 105)
print(f"\n✓ Best variant : {best_name}")
print(f"  Sharpe       : {best['sharpe']:.4f}")
print(f"  Ann Return   : {best['ann_ret']:+.2%}")
print(f"  Turnover     : {best['ann_turnover']:.1f}%")
print(f"  Overlap      : {best['avg_overlap']:.1f}/{K}")
print(f"  Win Rate     : {best['win_rate']:.2%}")
print(f"  Max Drawdown : {best['max_dd']:.2%}")

# ── ALSO: THEORETICAL MAX (no costs) ─────────────────────────────────────────
print("\n[Theoretical max — zero transaction costs]")
r_nocost = run_aligned_backtest(preds, actual, K, INITIAL_CAP,
                                 hold_days=5, name="Zero cost")
# Override cost
eq = [INITIAL_CAP]
t  = 0
while t + 5 <= T:
    pred_t   = preds[t]
    valid    = ~np.isnan(pred_t)
    ranked   = np.where(valid)[0][np.argsort(-pred_t[valid])]
    holdings = ranked[:K]
    gross    = float(np.nanmean(actual[t][holdings]))
    eq.append(eq[-1] * (1 + gross))
    t += 5
eq = np.array(eq)
n_p   = len(eq) - 1
n_yrs = n_p * 5 / 252
rets  = np.diff(eq) / eq[:-1]
print(f"  Theoretical Ann Return (0 cost): {((eq[-1]/INITIAL_CAP)**(1/n_yrs)-1):+.2%}")
print(f"  Theoretical Sharpe (0 cost):     {rets.mean()/(rets.std()+1e-8)*np.sqrt(52):.2f}")

# ── PLOT ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
for r in all_results:
    ax.plot(r['equity_curve'] / INITIAL_CAP, label=r['name'], linewidth=1.8)
ax.plot(eq / INITIAL_CAP, label='Zero cost (theoretical)', linewidth=2,
        linestyle='--', color='black')
ax.axhline(1.0, color='gray', linestyle=':', linewidth=0.8)
ax.set_title('Equity Curves — Aligned 5-Day Hold', fontsize=13)
ax.set_xlabel('Rebalance Periods')
ax.set_ylabel('Portfolio Value (normalised)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[1]
for r in all_results:
    ax.scatter(r['ann_turnover'], r['sharpe'], s=100, zorder=5)
    ax.annotate(r['name'], (r['ann_turnover'], r['sharpe']),
                textcoords='offset points', xytext=(4, 3), fontsize=7)
ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax.set_title('Sharpe vs Annualised Turnover', fontsize=13)
ax.set_xlabel('Annualised Turnover (%)')
ax.set_ylabel('Sharpe Ratio')
ax.grid(True, alpha=0.3)

plt.suptitle('DiffSTOCK — Aligned Backtest (5-day hold = 5-day labels)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plot_path = f"{RESULTS_DIR}/aligned_backtest.png"
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"\nPlot saved: {plot_path}")

# ── SAVE ──────────────────────────────────────────────────────────────────────
summary = {r['name']: {k: float(v) for k, v in r.items()
                        if not isinstance(v, np.ndarray) and not isinstance(v, str)}
           for r in all_results}
with open(f"{RESULTS_DIR}/aligned_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)
print(f"Summary saved.")