# DiffSTOCK India: Quantitative Stock Prediction for Nifty 500

A hybrid deep learning model combining **dilated causal convolutions**, **masked relational transformers**, and **denoising diffusion** for probabilistic stock return prediction in the Indian market.

## Overview

DiffSTOCK is a state-of-the-art quantitative model designed for autonomous trading in the Nifty 500 universe (~400 actively traded stocks). It combines:

- **MaTCHS** (Masked Temporal-Cross-stock Historical Signal encoder)
  - **Att-DiCEm**: Dilated causal convolutions for temporal feature extraction
  - **MRT**: Masked Relational Transformer for inter-stock relationship modeling
- **Adaptive DDPM**: Denoising Diffusion Probabilistic Model for probabilistic return generation

### Key Features

- 10 years of historical data (2015-2026) for robust training
- Realistic Indian market transaction costs (brokerage, STT, stamp duty, slippage)
- Walk-forward backtesting with weekly rebalancing
- EMA weight averaging and mixed precision training
- Comprehensive evaluation metrics (IC, ICIR, Sharpe, Max Drawdown)

## Project Structure

```
diffstock_india/
├── config/
│   └── config.yaml                  # All hyperparameters
├── data/
│   ├── raw/                         # Downloaded CSVs
│   ├── processed/                   # Cleaned data
│   └── dataset/                     # Final tensors
├── src/
│   ├── data/                        # Data pipeline
│   │   ├── scraper.py               # Download from yfinance/jugaad-data
│   │   ├── cleaner.py               # Clean and align data
│   │   ├── validator.py             # Quality checks
│   │   ├── feature_engineer.py      # Compute 15 technical features
│   │   ├── relation_builder.py      # Build relation matrices
│   │   └── dataset_builder.py       # Assemble final dataset
│   ├── model/                       # Model architecture
│   │   ├── att_dicem.py             # Temporal encoder
│   │   ├── mrt.py                   # Relational transformer
│   │   ├── matches.py               # Combined encoder
│   │   ├── diffusion.py             # DDPM
│   │   └── diffstock.py             # Top-level model
│   ├── training/
│   │   └── trainer.py               # Training loop with EMA
│   ├── evaluation/
│   │   ├── metrics.py               # IC, Sharpe, etc.
│   │   └── backtester.py            # Indian market backtest
│   └── utils/
│       ├── logger.py
│       └── seed.py
├── scripts/
│   ├── run_scrape.py                # Download data
│   ├── run_train.py                 # Train model
│   └── run_backtest.py              # Evaluate model
├── checkpoints/                     # Saved models
├── logs/                            # Training logs
├── results/                         # Backtest results
└── requirements.txt
```

## Installation

```bash
# Clone repository
cd diffstock_india

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Data Pipeline

Download and process 10 years of Nifty 500 data:

```bash
# Step 1: Scrape data (takes 30-60 minutes)
python scripts/run_scrape.py

# Step 2: Build dataset (automatic - runs all pipeline steps)
python -c "from src.data.dataset_builder import build_dataset; build_dataset(run_scraping=False)"
```

This will:
- Download OHLCV data for ~400 stocks from yfinance/jugaad-data
- Clean data (handle missing values, outliers, splits/dividends)
- Validate quality (exclude stocks with >15% missing data)
- Engineer 15 technical features (RSI, MACD, Bollinger Bands, etc.)
- Build 3 relation matrices (sector, industry, price correlation)
- Create sliding window dataset with train/val/test splits

**Output**: `data/dataset/nifty500_10yr.npz` (~2,400 training samples)

### 2. Training

Train the DiffSTOCK model:

```bash
python scripts/run_train.py
```

**Training configuration** (from `config/config.yaml`):
- Batch size: 32
- Learning rate: 0.0003 (AdamW with cosine annealing)
- Max epochs: 150 (with early stopping patience=20)
- EMA decay: 0.995
- Gradient clipping: 1.0
- Mixed precision: Automatic (FP16 if GPU available)

**Model architecture**:
- d_model: 128
- MRT heads: 8
- Att-DiCEm layers: 4 (dilations: 1, 2, 4, 8)
- MRT layers: 3
- Diffusion timesteps: 200

**Expected training time**:
- GPU (RTX 3090): ~2-4 hours
- CPU: ~12-20 hours

### 3. Backtesting

Evaluate trained model on test set:

```bash
# Backtest on test set (2024-07-01 to 2026-02-26)
python scripts/run_backtest.py --split test

# Backtest on validation set
python scripts/run_backtest.py --split val
```

**Strategy**: Long-only Top-20 with weekly rebalancing

**Transaction costs** (realistic Indian market):
- Brokerage: 0.03%
- STT: 0.1% (buy) + 0.1% (sell)
- Exchange charges: 0.00335%
- GST: 18% on brokerage + exchange
- Stamp duty: 0.015% (buy)
- Slippage: 0.2%
- **Total round-trip: ~0.6-0.8%**

## Model Architecture

### MaTCHS (Conditional Encoder)

1. **Att-DiCEm** (Temporal Encoder)
   - Input: (B, N, L=20, F=15) - 20 days of 15 features per stock
   - 4 dilated causal conv layers (receptive field = 20 days)
   - Depthwise separable convs for efficiency
   - Attention gating on final timestep
   - Output: (B, N, d_model=128)

2. **MRT** (Cross-Stock Relational Encoder)
   - Input: (B, N, d_model) stock embeddings
   - 3 transformer blocks with relation-based masking
   - Stocks only attend to related stocks (sector/industry/correlated)
   - Output: (B, N, d_model) relational embeddings

### Adaptive DDPM (Diffusion Model)

- **Forward**: x_0 (returns) → x_T (noise) via cosine schedule
- **Reverse**: x_T → x_0 conditioned on MaTCHS embeddings
- T=200 diffusion steps (reduced for limited data)
- Denoising network: MLP with time + condition injection
- Inference: Generate 50 samples → mean prediction + uncertainty

### Total Parameters

```
MaTCHS:
  Att-DiCEm:     ~1.2M parameters
  MRT:           ~2.5M parameters
Diffusion:       ~3.8M parameters
Total:           ~7.5M parameters
```

## Evaluation Metrics

### Signal Quality
- **IC** (Information Coefficient): Spearman rank correlation - Target: >0.05
- **ICIR** (IC Information Ratio): IC mean / IC std - Target: >0.5
- **Rank IC**: IC on rank-transformed predictions

### Portfolio Performance
- **Sharpe Ratio**: Risk-adjusted returns (annualized)
- **Max Drawdown**: Peak-to-trough decline
- **Total Return**: Cumulative return over test period
- **Win Rate**: Percentage of positive return days

### Classification
- **Accuracy**: Direction prediction (up/down)
- **MCC**: Matthews Correlation Coefficient (balanced metric)

## Expected Performance

Based on DiffSTOCK paper and Indian market characteristics:

| Metric | Target | Notes |
|--------|--------|-------|
| Test IC | 0.04-0.07 | Indian markets have lower IC than US/China |
| Test ICIR | 0.3-0.6 | Consistency measure |
| Sharpe Ratio | 1.0-1.5 | Top-20 long-only strategy |
| Max Drawdown | -15% to -25% | Typical for emerging markets |
| Accuracy | 52-55% | Directional prediction |

## Configuration

All hyperparameters in `config/config.yaml`:

**Data**:
```yaml
start_date: "2015-01-01"
end_date: "2026-02-26"
lookback_window: 20
n_features: 15
corr_threshold: 0.4
```

**Model**:
```yaml
d_model: 128
n_heads_mrt: 8
n_layers_dicem: 4
n_layers_mrt: 3
diffusion_T: 200
```

**Training**:
```yaml
batch_size: 32
learning_rate: 0.0003
max_epochs: 150
patience: 20
ema_decay: 0.995
```

## Advanced Usage

### Hyperparameter Tuning (Optional)

Create `src/training/hyperopt.py` using Optuna for automated hyperparameter search:

```python
# Search space:
# - d_model: [64, 96, 128, 192]
# - n_heads: [4, 8, 16]
# - diffusion_T: [100, 150, 200, 300]
# - learning_rate: [1e-4, 1e-2]
# - lookback_window: [10, 15, 20, 30]
```

### Custom Relation Matrices

Modify `src/data/relation_builder.py` to add custom relations:
- Supply chain relationships
- Ownership networks
- News sentiment correlation

### Integration with Trading System

```python
from src.model.diffstock import DiffSTOCK
import torch

# Load trained model
model = DiffSTOCK(...)
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Make predictions
model.eval()
with torch.no_grad():
    predictions, uncertainty = model(x, R_mask, n_samples=50)

# Use predictions for portfolio construction
top_k_stocks = predictions.argsort()[-20:]  # Top 20
```

## References

1. **DiffSTOCK** (ICASSP 2024): "Diffusion Model for Stock Price Prediction"
2. **HIST** (Wentao Xu, 2021): "HIST: A Graph-based Framework for Stock Trend Forecasting"
3. **MASTER** (AAAI 2024): "MASTER: Market-Guided Stock Transformer for Stock Price Forecasting"

## Regulatory Compliance

- **SEBI Compliant**: Model follows SEBI guidelines for algorithmic trading
- **Risk Management**: Built-in position limits, drawdown controls
- **Audit Trail**: All predictions and trades logged
- **Paper Trading**: Test thoroughly before live deployment

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
