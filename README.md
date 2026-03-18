# DHARMA: Diffusion HMM-Adaptive Regime Mixture Architecture

Regime-aware probabilistic stock return prediction for the **Nifty 500** universe, built on 20 years of Indian equity data (2005–2026).

## Architecture (v3.0)

```
Input (B, L=20, N, F=16)
        │
   Layer 0 — RevIN          per-window robust normalization (median/IQR)
        │
   Layer 1 — HMM Embed      4-state regime → 16-dim embedding appended to features
        │
   Layer 2 — MaTCHS          Att-DiCEm (temporal) + MRT (cross-stock relational)
        │                   d_model=192, 12 heads, 5+3 layers
   Layer 3 — Soft MoE        3 regime-specialized expert FFNs, soft routing
        │
   Layer 4 — Adaptive DDPM   150-step cosine-schedule diffusion
        │
   Output: predictions (B, N) + uncertainty (B, N)
```

| Layer | Module | Purpose |
|-------|--------|---------|
| 0 | **RevIN** | Reversible Instance Normalization — each 20-day window uses its own median/IQR so 2008 crash and 2022 calm are equally learnable |
| 1 | **HMM** | 4-state Gaussian HMM on NIFTY50 returns → soft regime probabilities → 16-dim learned embedding |
| 2 | **MaTCHS** | Att-DiCEm (dilated causal conv, temporal) + MRT (masked relational transformer, cross-stock attention) |
| 3 | **Soft MoE** | 3 expert FFNs gated by regime embedding; load-balance loss prevents collapse |
| 4 | **DDPM** | Denoising diffusion with cosine beta schedule for probabilistic 5-day return prediction |

## Project Structure

```
diffstock_india/
├── config/config.yaml               # Single source of truth for all hyperparams
├── data/
│   ├── raw/                         # Downloaded CSVs (~400 stocks)
│   ├── processed/                   # Cleaned parquet files
│   ├── dataset/                     # Final tensors + relation matrices
│   └── regime/                      # HMM model + daily_regime_probs.parquet
├── src/
│   ├── data/
│   │   ├── scraper.py               # Download from yfinance/jugaad-data
│   │   ├── cleaner.py               # Clean, align, handle survivorship bias
│   │   ├── validator.py             # Quality gates
│   │   ├── feature_engineer.py      # 16 technical features
│   │   ├── relation_builder.py      # Sector/industry/correlation matrices
│   │   ├── regime_detector.py       # HMM regime fitting
│   │   └── dataset_builder.py       # Sliding window assembly
│   ├── model/
│   │   ├── revin.py                 # Reversible Instance Normalization
│   │   ├── att_dicem.py             # Temporal encoder (dilated causal conv)
│   │   ├── mrt.py                   # Masked Relational Transformer
│   │   ├── matches.py               # Combined MaTCHS encoder
│   │   ├── regime_embedding.py      # HMM regime → learned embedding
│   │   ├── moe_head.py              # Soft Mixture of Experts
│   │   ├── diffusion.py             # Adaptive DDPM
│   │   └── dharma.py                # Top-level model (DHARMA)
│   ├── training/trainer.py          # Training loop (EMA, AMP, crisis upsampling)
│   ├── evaluation/
│   │   ├── metrics.py               # IC, ICIR, Sharpe, MCC
│   │   └── backtester.py            # Indian market backtest with realistic costs
│   ├── simulation/simulator.py      # End-to-end simulation engine
│   ├── advisor/                     # AI investment advisor (LLM + model)
│   └── utils/
├── scripts/
│   ├── run_scrape.py                # Download data
│   ├── run_full_scrape_and_build.py # Full pipeline (scrape → dataset)
│   ├── run_train.py                 # Train model
│   └── run_backtest.py              # Evaluate model
├── notebooks/
│   └── DHARMA_Training_Colab.ipynb  # Google Colab training notebook
├── tests/                           # Model shape + training stability tests
├── checkpoints/                     # Saved models
├── logs/                            # Training logs
└── requirements.txt
```

## Quick Start

```bash
cd diffstock_india
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Verify installation
python verify_installation.py
```

### 1. Data Pipeline

```bash
# Scrape 20 years of Nifty 500 data (30-60 min)
python scripts/run_scrape.py

# Build full dataset (clean → features → relations → regime → sliding windows)
python scripts/run_full_scrape_and_build.py
```

Output: `data/dataset/nifty500_20yr.npz` + `data/regime/daily_regime_probs.parquet`

### 2. Train

```bash
python scripts/run_train.py
```

Key training settings (from `config/config.yaml`):

| Param | Value | Notes |
|-------|-------|-------|
| Batch size | 16 | |
| Learning rate | 0.0001 | AdamW + cosine w/ restarts |
| Max epochs | 300 | patience=50 |
| EMA decay | 0.999 | |
| Warmup | 4000 steps | |
| Crisis upsampling | 3× | GFC, COVID, Demonetization |

**Abort gate**: If IC < 0.08 by epoch 30, stop — something is structurally wrong.

### 3. Backtest

```bash
python scripts/run_backtest.py --split test   # Jul 2024 – Feb 2026
python scripts/run_backtest.py --split val    # Jan 2023 – Jun 2024
```

Strategy: Long-only Top-20, weekly rebalance. Realistic Indian costs (~0.6-0.8% round trip).

## Inference

```python
import torch, numpy as np
from src.model.dharma import create_dharma_model

# Load checkpoint
ckpt = torch.load("checkpoints/<run>/checkpoints/best_model.pt", map_location="cpu", weights_only=False)
config = ckpt["config"]
n_stocks = len(np.load("data/dataset/nifty500_20yr.npz", allow_pickle=True)["stock_symbols"])

# Build model + load EMA weights
model = create_dharma_model(config, n_stocks)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Relation mask
R_mask = torch.FloatTensor(np.load("data/dataset/relation_matrices.npz")["R_mask"])

# Predict
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, R_mask = model.to(device), R_mask.to(device)

with torch.no_grad():
    # X: (B, 20, N, 16) float32 — 20-day lookback windows
    predictions, uncertainty = model(X.to(device), R_mask, n_samples=50)
    # predictions: (B, N) — 5-day forward return signal (use as rank)
    # uncertainty: (B, N) — std across 50 diffusion samples
```

### Input Features (16)

| # | Feature | Description |
|---|---------|-------------|
| 0–3 | `open/high/low/close_ret` | OHLC returns vs prev close |
| 4 | `log_volume` | log(volume) |
| 5 | `hl_spread` | (High-Low)/Close |
| 6–7 | `rsi_14`, `rsi_5` | RSI indicators |
| 8 | `bb_pct` | Bollinger %B (20d) |
| 9–10 | `vol_ratio_5`, `vol_ratio_20` | Volume ratios |
| 11 | `macd_signal` | MACD signal line |
| 12 | `atr_14` | ATR-14 / Close |
| 13–14 | `mom_5`, `mom_20` | 5d and 20d momentum |
| 15 | `close_vwap` | Close/VWAP deviation |

With `revin.enabled: true`, raw features are passed in — RevIN normalizes per-window at runtime.

## IC Targets by Epoch

| Checkpoint | IC Target | What it validates |
|------------|-----------|-------------------|
| Epoch 30 | ≥ 0.08 | Minimum viable signal (abort if below) |
| Epoch 50 | ≥ 0.12 | RevIN working correctly |
| Epoch 100 | ≥ 0.18 | HMM regime embedding contributing |
| Epoch 150 | ≥ 0.22 | MoE routing stabilized |
| Final | ≥ 0.25 | Deployment consideration |

## Transaction Costs (Indian Market)

| Component | Rate |
|-----------|------|
| Brokerage | 0.03% |
| STT (buy + sell) | 0.1% + 0.1% |
| Exchange charges | 0.00335% |
| SEBI turnover | 0.0001% |
| GST | 18% on brokerage + exchange |
| Stamp duty | 0.015% (buy) |
| Slippage | 0.2% |
| **Total round-trip** | **~0.6-0.8%** |

## Evaluation Metrics

- **IC** (Information Coefficient) — Spearman rank correlation between predicted and actual returns
- **ICIR** — IC mean / IC std (consistency)
- **Sharpe Ratio** — Risk-adjusted annualized returns
- **Max Drawdown** — Peak-to-trough decline
- **Win Rate** — % of positive return rebalance periods
- **MCC** — Matthews Correlation Coefficient (directional accuracy)

## Config Reference

All hyperparameters live in `config/config.yaml`. Key sections:

| Section | Controls |
|---------|----------|
| `data` | Date range, features, quality thresholds, crisis periods |
| `revin` | Reversible instance normalization (enabled, robust, affine) |
| `hmm` | HMM regime detector (n_regimes, features, smoothing) |
| `model` | MaTCHS dimensions, dropout, diffusion params |
| `model.moe` | Mixture of Experts (n_experts, gating, load balance) |
| `training` | LR, scheduler, epochs, patience, crisis upsampling |
| `evaluation` | IC targets, transaction costs, rebalance freq |
| `paths` | All directory paths |

## References

- [Stock Top Papers](https://github.com/marcuswang6/stock-top-papers) — curated collection of stock prediction papers
- **DiffSTOCK** (ICASSP 2024) — base diffusion architecture for stock prediction
- **MASTER** (AAAI 2024) — market-guided stock transformer
- **HIST** (Wentao Xu, 2021) — graph-based stock trend forecasting
- **MERA** — mixture of experts with retrieval augmentation (inspiration for regime-gated MoE)
- **RevIN** (Kim et al. 2021) — reversible instance normalization for time series

## Regulatory

- SEBI-compliant algorithmic trading guidelines
- Built-in position limits + drawdown controls
- Full audit trail (all predictions logged)
- Paper trade before live deployment

## Troubleshomenshooting

### Data Download Issues
```bash
# If yfinance fails, manually install jugaad-data
pip install jugaad-data

# If NSE website blocks, use VPN or wait and retry
```

### Memory Issues
```bash
# Reduce batch size in config.yaml
batch_size: 16  # Instead of 32

# Or reduce number of stocks
# Edit validator.py thresholds to keep only most liquid stocks
```

### GPU Out of Memory
```python
# In config.yaml, reduce model size:
d_model: 64
n_layers_dicem: 3
n_layers_mrt: 2
```

## License

MIT License - See LICENSE file for details.

## Disclaimer

This model is for educational and research purposes only. Stock trading involves substantial risk of loss. Past performance does not guarantee future results. Always conduct thorough due diligence and consult with financial advisors before making investment decisions.

## Contact

For questions or collaboration:
- Issues: Open a GitHub issue
- Email: [Your contact]

---

**Built with**: PyTorch, pandas, yfinance, NumPy, scipy

**Target Market**: NSE India (Nifty 500 universe)

**Status**: Research/Development ✅ | Production: Pending compliance review
