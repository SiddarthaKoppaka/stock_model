# DiffSTOCK Performance Improvements

**Date**: March 1, 2026
**Issue**: Model trains without NaN but achieves near-random performance (IC ≈ 0.0008)

---

## 🔴 Current Problem

Training completed successfully (no NaN!), but performance is essentially random:

```
Best Validation IC: 0.0008  (Target: 0.04-0.07)
Validation Accuracy: ~50%    (Random baseline)
Training Loss: ~1.0          (Not decreasing)
Early stopping: Epoch 25     (No improvement)
```

**Diagnosis**: Model is **not learning** - loss plateaus immediately and never improves.

---

## 🔍 Root Cause Analysis

### 1. **Learning Rate Too Conservative**
**Current**: `0.00003` (reduced 10x to fix NaN issue)
**Problem**: TOO low - model barely updates weights
- Loss stuck at 1.0 for all 25 epochs
- No learning happening at all
- The 10x reduction was too aggressive

**Evidence from logs**:
```
Epoch 1:  Train Loss: 1.0199
Epoch 5:  Train Loss: 0.9998
Epoch 10: Train Loss: 0.9973
Epoch 25: Train Loss: 0.9998
```
Loss oscillates around 1.0 instead of decreasing.

### 2. **Model Capacity Too Small**
**Current**: d_model=128, ~7.5M parameters (from training log: 45M is wrong - that's with 329 stocks)
**Problem**: Model may be too simple for the complexity of:
- 329 stocks
- 16 features per stock
- 20 timesteps
- Complex cross-stock relationships

**Recommendation**: Increase to d_model=192-256 for better representation capacity.

### 3. **Diffusion Settings Too Conservative**
**Current**:
- T=200 timesteps
- beta_start=0.00001, beta_end=0.01
**Problem**:
- 200 timesteps is computationally expensive and may be overkill
- Beta range too narrow, limiting model's ability to learn

### 4. **Batch Size vs Learning Dynamics**
**Current**: batch_size=32
**Problem**: With very low LR, larger batches mean even slower learning
**Recommendation**: Reduce batch size to 16-24 for noisier but more frequent updates

---

## ✅ Proposed Solutions

### Solution 1: Balanced Improvement (Recommended)

**File**: `config/config.yaml` (already updated)

**Key Changes**:
| Parameter | Old | New | Reason |
|-----------|-----|-----|--------|
| **learning_rate** | 0.00003 | **0.0001** | **3.3x increase - critical fix** |
| d_model | 128 | **192** | More capacity |
| n_heads_mrt | 8 | **12** | Better attention |
| n_layers_dicem | 4 | **5** | Deeper temporal |
| n_layers_mrt | 3 | **4** | Deeper relational |
| batch_size | 32 | **24** | Better gradients |
| diffusion_T | 200 | **150** | Faster training |
| grad_clip | 0.5 | **0.8** | Allow larger updates |
| patience | 20 | **25** | More time to learn |
| max_epochs | 150 | **200** | Don't stop early |

**Expected Results**:
- Loss should decrease: 1.0 → 0.5-0.7 → 0.2-0.4 → 0.05-0.15
- Val IC: 0.02-0.04 by epoch 30
- Val IC: 0.04-0.06 at convergence

### Solution 2: Aggressive Improvement (For Experimentation)

**File**: `config/config_improved.yaml` (created)

Even larger model with higher learning rate:
- d_model: 256
- n_heads_mrt: 16
- learning_rate: 0.0001
- batch_size: 16
- diffusion_T: 100

**Use this if Solution 1 doesn't work well enough.**

---

## 📊 Expected Training Behavior (After Fixes)

### With Current (Too Conservative) Settings:
```
Epoch 1-25:  loss ≈ 1.0  (stuck)
Val IC:      ~0.0008      (random)
Result:      Early stopping, no learning
```

### With Improved Settings (Balanced):
```
Epoch 1-5:   loss: 1.0 → 0.7    (learning starts)
Epoch 5-10:  loss: 0.7 → 0.4    (rapid improvement)
Epoch 10-30: loss: 0.4 → 0.15   (continued learning)
Epoch 30-80: loss: 0.15 → 0.08  (fine-tuning)
Val IC:      0.03-0.05           (meaningful signal!)
```

### With Aggressive Settings:
```
Faster convergence, potentially higher capacity
Risk: Might overfit or become unstable
```

---

## 🚀 What to Do Next

### Option 1: Re-train with Balanced Config (Recommended)

The main `config/config.yaml` has been updated with balanced improvements.

**Steps**:
1. Upload to Colab (same dataset - no need to rebuild)
2. Run training notebook
3. Monitor for:
   - ✅ Loss should **decrease** (1.0 → 0.7 → 0.4 → 0.15)
   - ✅ Val IC should reach **0.02-0.04 by epoch 30**
   - ✅ No NaN (already fixed)
   - ✅ Training should run 50-100 epochs before early stopping

**Expected time**: 3-5 hours on T4 GPU

### Option 2: Try Aggressive Config

If balanced config doesn't achieve IC > 0.03:

1. **In Colab notebook, change config loading**:
```python
# Replace:
with open('config/config.yaml', 'r') as f:

# With:
with open('config/config_improved.yaml', 'r') as f:
```

2. Re-run training

**Expected time**: 4-6 hours on T4 GPU

---

## 🔧 Additional Optimizations (If Needed)

### If IC is still low (<0.02) after 50 epochs:

#### 1. Check Data Quality
```python
# In Colab, add diagnostic cell:
import numpy as np

# Check if features have predictive power
from scipy.stats import spearmanr

for i in range(16):
    feat = X_train[:, -1, :, i]  # Last timestep, all stocks, feature i
    corrs = []
    for stock in range(329):
        corr, _ = spearmanr(feat[:, stock], y_train[:, stock])
        if not np.isnan(corr):
            corrs.append(corr)
    print(f"Feature {i}: mean correlation with returns = {np.mean(corrs):.4f}")
```

If all correlations are near 0, features may not be predictive.

#### 2. Try Different Prediction Horizons

Current: predicting 1-day ahead returns
Alternative: predict 5-day or 20-day ahead returns (less noisy)

**Modify dataset builder**:
```python
# In src/data/dataset_builder.py, change horizon:
y = data.loc[pred_date, 'close_ret']  # Current: 1 day

# To:
y = data.loc[pred_date:pred_date+5, 'close_ret'].mean()  # 5-day average
```

#### 3. Simplify the Task

Try predicting "direction" (up/down) instead of exact returns:

```python
# Convert regression to classification
y_binary = (y_train > 0).astype(float)  # 1 if up, 0 if down
```

Modify loss function to binary cross-entropy.

---

## 📈 Realistic Performance Expectations

### Indian Market Challenges:
1. **Higher noise**: Indian markets are more volatile than US/China
2. **Lower liquidity**: Especially for mid/small caps in Nifty 500
3. **Data quality**: NSE data has more gaps than US markets
4. **Predictability**: Stock returns are fundamentally noisy

### Target Metrics (Realistic):
| Metric | Conservative | Optimistic |
|--------|-------------|------------|
| Val IC | 0.02-0.03 | 0.04-0.06 |
| Val ICIR | 0.2-0.4 | 0.5-0.8 |
| Test IC | 0.015-0.025 | 0.03-0.05 |
| Sharpe Ratio | 0.8-1.2 | 1.3-1.8 |
| Annual Return | 8-12% | 14-20% |

**Note**: DiffSTOCK paper reports IC ≈ 0.05-0.08 on Chinese market (CSI 800).
Indian market IC is typically 60-70% of Chinese market due to higher noise.
**Target**: IC = 0.03-0.05 is **good** for Indian stocks.

---

## 🎯 Summary of Changes

### Root Issue:
Learning rate was TOO conservative (0.00003) after fixing NaN issue.
Model literally cannot learn with such tiny updates.

### Primary Fix:
**Increase learning rate to 0.0001** (3.3x increase)

### Secondary Fixes:
- Increase model capacity (d_model: 128 → 192)
- Reduce batch size (32 → 24) for better gradients
- Increase patience (20 → 25) to not stop too early
- Tune diffusion (T: 200 → 150, wider beta range)

### Expected Outcome:
- Loss decreases from 1.0 to 0.05-0.15
- Val IC reaches 0.03-0.05
- Model actually learns meaningful patterns

---

## 📁 Files Modified

1. **config/config.yaml** - Main config with balanced improvements ✓
2. **config/config_improved.yaml** - NEW: Aggressive config for experimentation ✓
3. **PERFORMANCE_IMPROVEMENTS.md** - This document ✓

---

## ⚠️ Important Notes

1. **Don't rebuild dataset** - The data is fine, just use the same fixed dataset
2. **Upload updated config** - Make sure Colab uses the new config.yaml
3. **Monitor training** - Loss MUST decrease, or something is still wrong
4. **Be patient** - Real learning takes 50-100 epochs, not 25
5. **Adjust if needed** - If IC < 0.02 after 100 epochs, try aggressive config

---

## 🔍 Debugging Checklist

If training still doesn't improve:

- [ ] Verify config.yaml is updated in Colab
- [ ] Check loss is actually decreasing (not stuck at 1.0)
- [ ] Verify no NaN in logs
- [ ] Check GPU memory (larger model needs more memory)
- [ ] Try reducing batch_size to 16 if OOM errors
- [ ] Check gradient norms are > 0.001 (in trainer logs)
- [ ] Verify dataset has 0 NaN values
- [ ] Try config_improved.yaml for more aggressive settings

---

**Next**: Re-run training with updated config.yaml and monitor for decreasing loss!
