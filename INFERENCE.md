# DiffSTOCK Inference Guide

Everything an integration or inference application needs to load and run the trained model.

---

## Checkpoint

| Field | Value |
|---|---|
| Path | `checkpoints/run_20260302_235104/checkpoints/best_model.pt` |
| Best val IC | 0.304 (epoch 84 of 124) |
| Test IC (Spearman) | **0.349** |
| Test Pearson IC | 0.365 |
| Test ICIR | 3.99 |
| Test accuracy (direction) | 61.9% |
| Total parameters | 1,784,897 (~7 MB fp32) |

> **Note:** This checkpoint was trained on the 2015–2026 dataset (329 stocks). If you retrain on the new 20-year dataset (266 stocks), the `n_stocks` dimension changes and you must use the new `stock_symbols` and `relation_matrices.npz`.

---

## Files needed for inference

```
diffstock_india/
├── checkpoints/run_20260302_235104/checkpoints/
│   └── best_model.pt          # model weights + EMA shadow + config
├── data/dataset/
│   ├── nifty500_10yr.npz      # stock_symbols, feature_names (for this checkpoint)
│   └── relation_matrices.npz  # R_mask (N×N attention mask)
├── src/
│   └── model/
│       ├── diffstock.py       # DiffSTOCK, create_diffstock_model
│       ├── matches.py         # MaTCHS encoder
│       ├── mrt.py             # Masked Relational Transformer
│       ├── att_dicem.py       # Att-DiCEm temporal encoder
│       └── diffusion.py       # AdaptiveDDPM
└── config/config.yaml
```

---

## Architecture

```
Input (B, L=20, N=329, F=16)
        │
        ▼
   MaTCHS encoder
   ├── Att-DiCEm  (dilated causal conv, temporal encoding per stock)
   └── MRT        (masked relational transformer, cross-stock attention)
        │ (B, N, d_model=192)
        ▼
   AdaptiveDDPM   (reverse diffusion, T=150 steps)
        │ sample n_samples=50 trajectories
        ▼
   mean → predictions  (B, N)   ← 5-day compounded forward return rank scores
   std  → uncertainty  (B, N)
```

**Key hyperparameters baked into the checkpoint:**

| Param | Value |
|---|---|
| `d_model` | 192 |
| `n_heads_mrt` | 12 |
| `n_layers_dicem` | 5 |
| `n_layers_mrt` | 3 |
| `diffusion_T` | 150 |
| `beta_start / beta_end` | 0.0001 / 0.02 |

---

## Input specification

### Feature tensor `X`
Shape: `(B, L, N, F)` = `(batch, 20 days, 329 stocks, 16 features)`
dtype: `float32`

The 16 features in order (index 0–15):

| # | Name | Description |
|---|---|---|
| 0 | `open_ret_norm` | Open-to-prev-close return, rolling 252-day z-score |
| 1 | `high_ret_norm` | High-to-prev-close return, z-score |
| 2 | `low_ret_norm` | Low-to-prev-close return, z-score |
| 3 | `close_ret_norm` | Close-to-prev-close return, z-score |
| 4 | `log_volume_norm` | log(volume), z-score |
| 5 | `hl_spread_norm` | (High-Low)/Close, z-score |
| 6 | `rsi_14_norm` | RSI-14, z-score |
| 7 | `rsi_5_norm` | RSI-5, z-score |
| 8 | `bb_pct_norm` | Bollinger %B (20d), z-score |
| 9 | `vol_ratio_5_norm` | Volume / 5-day avg volume, z-score |
| 10 | `vol_ratio_20_norm` | Volume / 20-day avg volume, z-score |
| 11 | `macd_signal_norm` | MACD signal line, z-score |
| 12 | `atr_14_norm` | ATR-14 / Close, z-score |
| 13 | `mom_5_norm` | 5-day momentum (close/close[5]-1), z-score |
| 14 | `mom_20_norm` | 20-day momentum, z-score |
| 15 | `close_vwap_norm` | Close / VWAP deviation, z-score |

> All features are normalized with a **causal rolling 252-day z-score** per stock — no look-ahead. Missing values (stock not yet listed) are filled with `0`.

### Relation mask `R_mask`
Shape: `(N, N)` = `(329, 329)`
dtype: `float32`, binary (0 or 1), diagonal = 0
Source: `data/dataset/relation_matrices.npz['R_mask']`

Union of sector, industry, and price-correlation edges. Density ~27%.

---

## Stock universe

The 329 stocks and their order are fixed. Load from the dataset:

```python
import numpy as np
data = np.load('data/dataset/nifty500_10yr.npz', allow_pickle=True)
stock_symbols = data['stock_symbols'].tolist()   # list of 329 NSE symbols
# e.g. ['SUNPHARMA', 'AUROPHARMA', 'INFY', ...]
```

The position of each stock in the `N` dimension of `X` and predictions must match this order exactly.

---

## Loading the model

```python
import torch
import numpy as np
from src.model.diffstock import create_diffstock_model

CHECKPOINT = "checkpoints/run_20260302_235104/checkpoints/best_model.pt"
DATASET    = "data/dataset/nifty500_10yr.npz"
RELATIONS  = "data/dataset/relation_matrices.npz"

# ── Load checkpoint ───────────────────────────────────────────────────────────
ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
config   = ckpt["config"]
n_stocks = 329                          # locked to training universe

# ── Build model and load weights ──────────────────────────────────────────────
model = create_diffstock_model(config, n_stocks)
model.load_state_dict(ckpt["model_state_dict"])   # EMA weights
model.eval()

# ── Load relation mask ────────────────────────────────────────────────────────
R_mask = torch.FloatTensor(np.load(RELATIONS)["R_mask"])  # (329, 329)

# ── Load stock symbols ────────────────────────────────────────────────────────
stock_symbols = np.load(DATASET, allow_pickle=True)["stock_symbols"].tolist()
```

> The checkpoint stores **EMA-smoothed weights** under `model_state_dict` (decay=0.999). These are the weights that produced the reported metrics — load them directly, no separate EMA application needed.

---

## Running inference

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = model.to(device)
R_mask = R_mask.to(device)

# X: (B, 20, 329, 16) float32 — your pre-built feature windows
X = torch.FloatTensor(X_batch).to(device)

with torch.no_grad():
    predictions, uncertainty = model(X, R_mask, n_samples=50)
    # predictions: (B, 329) — predicted 5-day compounded return (rank signal)
    # uncertainty: (B, 329) — std across 50 diffusion samples
```

`n_samples=50` matches training. You can lower to `n_samples=10` for faster inference with minimal accuracy loss.

---

## Interpreting outputs

`predictions[t, i]` is the model's estimated **5-day compounded forward return** for stock `i` on day `t`. The **absolute value is not calibrated** — use it as a **cross-sectional rank signal**:

```python
# Rank stocks on each day (highest score = best expected return)
import numpy as np

ranks = np.argsort(-predictions.cpu().numpy(), axis=-1)  # (B, 329), best first

# Top-K portfolio (K=20 matches the backtest)
K = 20
top_k_indices  = ranks[:, :K]          # (B, 20) stock indices
top_k_symbols  = [[stock_symbols[i] for i in row] for row in top_k_indices]
top_k_conf     = uncertainty.cpu().numpy()  # lower = more confident
```

### Uncertainty-weighted selection
```python
# Score = prediction / uncertainty  (Sharpe-like, penalise uncertain picks)
score = predictions / (uncertainty + 1e-6)
top_k = torch.argsort(score, dim=-1, descending=True)[:, :K]
```

---

## Backtest performance (test set: Jul 2024 – Feb 2026)

| Strategy | Ann. Return | Sharpe | Max DD | Win Rate |
|---|---|---|---|---|
| 5-day hold, top-20 strict | 209.6% | 5.27 | -6.2% | 80.2% |
| 10-day hold, top-20 strict | 80.0% | 4.28 | -7.2% | 87.5% |
| 5-day hold, band 6%/25% | 191.4% | 5.00 | -6.9% | 79.0% |
| 5-day hold, band 6%/40% | 157.6% | 4.57 | -7.2% | 79.0% |

> Transaction costs included: brokerage 0.03% + STT 0.1% buy/sell + GST + exchange + slippage 0.2% (round-trip ~0.69%).

---

## Transaction costs (Indian markets)

| Component | Rate |
|---|---|
| Brokerage | 0.03% |
| STT (buy + sell) | 0.2% total |
| Exchange charges | 0.00335% |
| SEBI turnover fee | 0.0001% |
| GST on brokerage | 18% of brokerage |
| Stamp duty | 0.015% |
| Slippage | 0.20% |
| **Round-trip total** | **~0.694%** |

---

## Integration checklist

- [ ] Feature windows built with **rolling 252-day z-score per stock** (causal, no look-ahead)
- [ ] Input shape is `(B, 20, 329, 16)` with features in the exact order above
- [ ] Stock ordering in `X[:, :, i, :]` matches `stock_symbols[i]`
- [ ] `R_mask` loaded from `relation_matrices.npz` (do **not** recompute at inference time)
- [ ] Model in `eval()` mode and `torch.no_grad()` context
- [ ] Predictions used as **rank signal**, not absolute return forecasts
- [ ] Rebalance every **5 trading days** (weekly) to match the prediction horizon
