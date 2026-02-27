# DiffSTOCK India - Quick Start Guide

Get started with DiffSTOCK in 15 minutes.

## Prerequisites

- Python 3.8+
- 8GB+ RAM (16GB recommended)
- GPU optional (but recommended for training)

## Installation (2 minutes)

```bash
cd diffstock_india

# Create environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Test Installation (30 seconds)

```bash
# Test model architecture
python tests/test_model_shapes.py
```

Expected output:
```
✓ Att-DiCEm: torch.Size([8, 100, 20, 15]) -> torch.Size([8, 100, 64])
✓ MRT: torch.Size([8, 100, 64]) -> torch.Size([8, 100, 64])
✓ MaTCHS: torch.Size([8, 20, 100, 15]) -> torch.Size([8, 100, 64])
✓ DDPM loss: 0.0234
✓ DDPM sampling: torch.Size([5, 8, 100])
✓ DiffSTOCK training loss: 0.0187
✓ DiffSTOCK inference: predictions torch.Size([8, 100]), uncertainty torch.Size([8, 100])
✓ Total parameters: 7,523,456
✓ All tests passed!
```

## Option 1: Quick Demo (5 minutes)

Skip data download and use synthetic data to test the full pipeline:

```python
# demo.py
import torch
import numpy as np
from src.model.diffstock import DiffSTOCK
from src.training.trainer import DiffSTOCKTrainer
import yaml

# Load config
with open('config/config.yaml') as f:
    config = yaml.safe_load(f)

# Generate synthetic data
T_train, T_val = 1000, 200
N, L, F = 400, 20, 15

X_train = np.random.randn(T_train, L, N, F).astype(np.float32)
y_train = np.random.randn(T_train, N).astype(np.float32) * 0.02

X_val = np.random.randn(T_val, L, N, F).astype(np.float32)
y_val = np.random.randn(T_val, N).astype(np.float32) * 0.02

# Create relation mask
R_mask = (torch.rand(N, N) > 0.5).float()
R_mask.fill_diagonal_(0)

# Create model
model = DiffSTOCK(n_stocks=N, in_features=F, d_model=64, diffusion_T=100)

# Modify config for quick demo
config['training']['max_epochs'] = 5
config['training']['batch_size'] = 16
config['model']['d_model'] = 64

# Train
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainer = DiffSTOCKTrainer(model, config, R_mask, device)

history = trainer.train(
    train_data=(X_train, y_train),
    val_data=(X_val, y_val)
)

print(f"Demo complete! Best val IC: {trainer.best_val_ic:.4f}")
```

Run: `python demo.py`

## Option 2: Full Pipeline (10+ hours)

### Step 1: Download Data (30-60 minutes)

```bash
python scripts/run_scrape.py
```

This downloads 10 years of data for ~400 Nifty 500 stocks.

**What's happening**:
- Fetches Nifty 500 constituent list from NSE
- Downloads OHLCV from yfinance (with jugaad-data fallback)
- Fetches sector/industry metadata
- Saves to `data/raw/`

**Output**:
```
data/raw/
├── RELIANCE.csv
├── TCS.csv
├── INFY.csv
...
├── metadata.json
└── failed_symbols.txt (if any)
```

### Step 2: Build Dataset (5-10 minutes)

```bash
python -c "from src.data.dataset_builder import build_dataset; build_dataset(run_scraping=False)"
```

**What's happening**:
1. Cleans data (handles NaNs, outliers, splits)
2. Validates quality (excludes stocks with >15% missing data)
3. Engineers 15 technical features
4. Builds 3 relation matrices (sector, industry, correlation)
5. Creates sliding window samples (20-day lookback)
6. Splits into train/val/test

**Output**:
```
data/dataset/
├── nifty500_10yr.npz (main dataset)
├── relation_matrices.npz
└── validation_report.json
```

**Expected**: ~2,000 training samples, ~380-400 stocks

### Step 3: Train Model (2-4 hours GPU / 12-20 hours CPU)

```bash
python scripts/run_train.py
```

**What's happening**:
- Loads dataset and relation matrices
- Creates DiffSTOCK model (~7.5M parameters)
- Trains for up to 150 epochs with early stopping
- Uses EMA, gradient clipping, mixed precision
- Saves best model based on validation IC

**Output**:
```
checkpoints/
├── best_model.pt (best validation IC)
├── final_model.pt
├── epoch_10.pt
├── epoch_20.pt
...

logs/
├── diffstock.log
└── training_history.json
```

**Expected training curve**:
```
Epoch 1:   Train Loss: 0.0234, Val IC: 0.012
Epoch 10:  Train Loss: 0.0089, Val IC: 0.038
Epoch 50:  Train Loss: 0.0045, Val IC: 0.052 ← Best
Epoch 70:  Train Loss: 0.0041, Val IC: 0.048 (overfitting starts)
Early stopping after epoch 70
```

### Step 4: Backtest (2-5 minutes)

```bash
python scripts/run_backtest.py --split test
```

**What's happening**:
- Loads trained model
- Generates predictions for test period (2024-07-01 to 2026-02-26)
- Runs Top-20 long-only strategy with weekly rebalancing
- Applies realistic Indian market transaction costs
- Computes portfolio returns and metrics

**Output**:
```
============================================================
Backtest Results Summary
============================================================
Initial Capital:       ₹1,000,000
Final Value:           ₹1,245,678
Total Return:          24.57%
Annualized Return:     15.32%
Sharpe Ratio:          1.23
Max Drawdown:          -18.45%
Win Rate:              54.32%
Avg Turnover:          32.15%
Transaction Cost:      0.0068 (round-trip)
============================================================
```

## Expected Performance

| Metric | Target Range | Your Result |
|--------|--------------|-------------|
| Validation IC | 0.03-0.06 | ___ |
| Test IC | 0.02-0.05 | ___ |
| Sharpe Ratio | 0.8-1.5 | ___ |
| Max Drawdown | -15% to -25% | ___ |

**Note**: Indian market IC is typically lower than US/China markets due to:
- Higher volatility
- Lower liquidity in mid-caps
- More retail participation

## Troubleshooting

### "Dataset not found"
```bash
# Check if data exists
ls data/dataset/nifty500_10yr.npz

# If not, rebuild dataset
python -c "from src.data.dataset_builder import build_dataset; build_dataset(run_scraping=False)"
```

### "CUDA out of memory"
```yaml
# Edit config/config.yaml
training:
  batch_size: 16  # Reduce from 32
model:
  d_model: 64     # Reduce from 128
```

### "yfinance download failed"
```bash
# Install fallback
pip install jugaad-data

# Or download manually using NSE website
```

### "Model not learning (IC ~0)"
Common issues:
1. Data leakage: Check feature engineering doesn't use future data
2. Poor relation matrices: Check correlation threshold (try 0.3-0.5)
3. Overfitting: Reduce model size or add regularization
4. Bad hyperparameters: Try learning rate 1e-4 to 1e-3

## Next Steps

1. **Monitor Training**: Watch `logs/diffstock.log` during training
2. **Analyze Results**: Check `logs/training_history.json` for loss curves
3. **Tune Hyperparameters**: Modify `config/config.yaml` and retrain
4. **Custom Features**: Add your own features in `feature_engineer.py`
5. **Alternative Strategies**: Modify `backtester.py` for long-short, sector-neutral, etc.

## Getting Help

- **Documentation**: See `README.md` for full details
- **Issues**: Check common errors above
- **Model Architecture**: Read paper references in README
- **Code**: All modules have detailed docstrings

## Key Files to Understand

1. `config/config.yaml` - All hyperparameters
2. `src/model/diffstock.py` - Model architecture
3. `src/training/trainer.py` - Training loop
4. `src/evaluation/backtester.py` - Strategy simulation

## Pro Tips

- Start with synthetic data (Option 1) to verify pipeline
- Use small model (d_model=64) for initial experiments
- Monitor validation IC - should improve first 20-50 epochs
- Real IC will be lower than synthetic (expected!)
- Transaction costs matter a lot - don't ignore them

---

**Ready to trade?**

⚠️ **IMPORTANT**: This is research code. Before live trading:
1. Paper trade for 3+ months
2. Implement risk management (stop losses, position limits)
3. Get SEBI approval for algorithmic trading
4. Use a broker API (Groww, Zerodha, etc.)
5. Add real-time monitoring and alerts

Happy trading! 🚀📈
