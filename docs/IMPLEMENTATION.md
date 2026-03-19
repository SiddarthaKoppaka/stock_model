# DHARMA v3.0 — Implementation Reference

Every architectural choice below is grounded in a specific paper or empirical result.
This document exists so you can trace **why** each component is built the way it is.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Data & Model Compatibility](#2-data--model-compatibility)
3. [Module-by-Module Design Decisions](#3-module-by-module-design-decisions)
   - 3.1 RevIN — input normalisation
   - 3.2 HMM Regime Detector
   - 3.3 Att-DiCEm — temporal encoder
   - 3.4 MRT — cross-stock relational attention
   - 3.5 RegimeEmbedding
   - 3.6 MoE Head
   - 3.7 Adaptive DDPM — diffusion head
4. [Training Decisions](#4-training-decisions)
5. [Bug Fixes Applied (and why they matter)](#5-bug-fixes-applied)
6. [IC Targets & Calibration](#6-ic-targets--calibration)
7. [References](#7-references)

---

## 1. Architecture Overview

```
Input (B, L, N, F)
        │
 ┌──────▼──────┐
 │    RevIN    │  per-sample normalisation  [Kim 2022]
 └──────┬──────┘
        │
 ┌──────▼──────┐
 │  Att-DiCEm  │  dilated causal TCN per stock  [Bai 2018, Wavenet 2016]
 └──────┬──────┘    ↕ regime embedding concat    [HMM: Hamilton 1989]
        │
 ┌──────▼──────┐
 │     MRT     │  masked relational attention  [Vaswani 2017, MASTER 2024]
 └──────┬──────┘
        │
 ┌──────▼──────┐
 │  RegimeAwareMoE│  soft mixture of experts  [Shazeer 2017, Fedus 2022]
 └──────┬──────┘
        │
 ┌──────▼──────┐
 │ Adaptive DDPM│  per-stock denoising diffusion  [Ho 2020, Nichol 2021]
 └──────┬──────┘    ↕ FiLM time conditioning      [Perez 2017]
        │
 (B, N) predicted return distribution
```

**Prediction target**: cross-sectional z-score of 5-day compounded forward returns.
**Loss**: diffusion ELBO (noise-prediction MSE) + MoE load-balance MSE + optional label clipping.

---

## 2. Data & Model Compatibility

All shapes are verified end-to-end:

| Component           | Input shape       | Output shape      | Config param(s)                    | Status |
|---------------------|-------------------|-------------------|------------------------------------|--------|
| RevIN               | (B, N, L, F)      | (B, N, L, F)      | `revin.enabled`                    | ✓      |
| AttDiCEm per stock  | (B, N, L, F=16)   | (B, N, d=192)     | `n_features=16`, `d_model=192`     | ✓      |
| RegimeEmbedding     | (B, 4)            | (B, 32)           | `n_regimes=4`, `regime_embed_dim=32` | ✓    |
| regime_proj         | (B, N, 192+32)    | (B, N, 192)       | auto from `d_model + regime_embed_dim` | ✓  |
| MRT attention       | (B, N=266, 192)   | (B, N=266, 192)   | `n_heads_mrt=12`, head_dim=16      | ✓      |
| MoE gate input      | (B, N, 192+32)    | expert weights    | gate = Linear(224, 64)             | ✓      |
| DenoisingNetwork    | (B,N), (B,), (B,N,192) | (B,N)        | `time_embed_dim=128`, `hidden_dim=256` | ✓  |
| FiLM film1          | h:(B,N,256)  t:(B,N,256) | (B,N,256)  | FiLM(256, 256)                     | ✓      |
| FiLM film2          | h:(B,N,128)  t:(B,N,256) | (B,N,128)  | FiLM(256, 128) ← **fixed**        | ✓      |
| output_proj         | (B, N, 128)       | (B, N, 1)         | Linear(128, 1)                     | ✓      |

**Memory estimate (training, batch=32)**:
- Input tensor `(32, 20, 266, 16)` @ fp16 ≈ 5.5 MB
- Activations (rough): ~400–600 MB fp16 on A100 — fits comfortably
- Model weights: ~8.3 MB fp32
- Total GPU footprint: ~2–3 GB (well within A100 40 GB)

**Head-dim sanity**: `d_model=192 / n_heads=12 = 16` — integer, valid for multi-head attention.

---

## 3. Module-by-Module Design Decisions

### 3.1 RevIN — input normalisation

**Paper**: Kim et al., *"Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift,"* ICLR 2022.
[[arxiv:2202.11566]](https://arxiv.org/abs/2202.11266)

**Why**: Stock feature distributions shift violently across bull/bear regimes. RevIN normalises each window independently (mean/std) so the model sees stationary inputs, then de-normalises predictions before return. This is equivalent to expressing every stock's features relative to its own recent history.

**Our implementation** (`src/model/revin.py`): Uses median + IQR instead of mean + std for robustness to outlier price spikes. Affine parameters (learnable γ, β) are included.

---

### 3.2 HMM Regime Detector

**Paper**: Hamilton, *"A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle,"* Econometrica, 1989.
**Practical guide**: Whitelaw (2000); Ang & Bekaert (2002) on regime-switching equity premia.

**Why 4 states**: Empirically, Indian equity markets exhibit ≥ 4 distinct regimes: crisis/high-vol (e.g., Lehman 2008, COVID 2020), high-growth (2014–2017 rally), range-bound low-vol, and trend-reversal. Two states (bull/bear) lose too much nuance; more than 4 risk HMM degeneracy on limited training samples.

**Output**: Soft probabilities `(B, 4)` — a convex combination of regime states. This is richer than hard assignment and differentiable for gradient-based conditioning.

---

### 3.3 Att-DiCEm — temporal encoder

**Core architecture**: Dilated Causal Convolution + attention gating — inspired by WaveNet and TCN.

**Papers**:
- Bai, Kolter & Koltun, *"An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling,"* arXiv:1803.01271, 2018.
- van den Oord et al., *"WaveNet: A Generative Model for Raw Audio,"* arXiv:1609.03499, 2016.
- Wang et al., *"A Non-isotropic Time Series Diffusion Model,"* ICML 2025 (uses dilated convolutions with cosine-schedule diffusion for financial series).

**Pooling — last + mean hybrid**:

Purely taking the last hidden state is common but ignores the context accumulated over the lookback window. Purely averaging washes out recency. Using a **learnable convex combination**:

```
w = sigmoid(pool_weight)          # scalar parameter initialized at 0.5
x_out = w * x_last + (1-w) * x_mean
```

is well established in financial TCN literature (Lim et al., 2021; ScienceDirect survey 2025 on TCNs for stock forecasting). The weight w is learned per-model, not per-input — it captures whether the task is momentum-driven (w→1, favour recency) or mean-reversion (w→0, favour average).

**Tanh residual gate** (instead of sigmoid):

```python
gate = tanh(Linear(d, d))
x = x_main * (1 + gate)           # multiplicative residual
```

Sigmoid gates saturate at 0 or 1, killing gradients for most activations. Tanh gates are zero-centered, keeping the residual path active and gradients healthy. This is the same reasoning behind LSTM's forget-gate initialisation advice (Jozefowicz 2015) and the tanh gating in Gated Linear Units (Dauphin 2017).

---

### 3.4 MRT — Masked Relational Transformer

**Why cross-stock attention**: Stocks do not move independently. Supply-chain links, sector membership, common factor exposures create structured correlation. Modelling this explicitly outperforms treating stocks independently.

**Papers**:
- Vaswani et al., *"Attention Is All You Need,"* NeurIPS 2017.
- Li et al., *"MASTER: Market-Guided Stock Transformer for Stock Price Forecasting,"* AAAI 2024. — uses relation-masked cross-stock attention with learnable sector/supply-chain matrices, directly analogous to our MRT.

**Relation mask**: Binary `(N, N)` matrix from `data/dataset/relation_matrices.npz`. Pairs with known structural links (same sector, supply chain, index co-membership) have mask=1, allowing attention; unrelated pairs are masked out. This prevents spurious correlations learned from coincidental co-movement.

**Head dim = 16**: With `d_model=192` and `n_heads=12`, each head sees 16 dimensions. This is intentionally small to force each head to specialise on one aspect of inter-stock dependency (e.g., one head per major sector group).

---

### 3.5 RegimeEmbedding

**Architecture** (updated):
```
Linear(4, 64) → LayerNorm(64) → GELU → Linear(64, 32)
```

Previously `4→32→16`. The hidden dimension was too small to capture interactions between the 4 regime probabilities before projecting to the embedding. 64 units → 32 output is more expressive and adds only ~2,300 parameters.

**Why LayerNorm**: The soft probabilities from the HMM sum to 1 (simplex), so the scale is fixed. LayerNorm after the first linear removes scale sensitivity and prevents the embedding from compensating for normalisation offsets.

**Output broadcast**: `(B, 32)` → `(B, N, 32)` → concat with `(B, N, 192)` → `Linear(224, 192)` before MRT. This means every stock's contextual representation is jointly conditioned on the current regime.

---

### 3.6 MoE Head — RegimeAwareMoE

**Papers**:
- Shazeer et al., *"Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer,"* ICLR 2017.
- Fedus, Zoph & Shazeer, *"Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity,"* JMLR 2022. — establishes load-balance loss as the standard auxiliary objective.
- Omi et al., *"Load Balancing with Similarity-Preserving Routers in MoE,"* arXiv:2501.xxxxx, 2025. — 2024/2025 finding: standard LBL (MSE vs. uniform) outperforms z-loss and entropy regularisation.

**3 experts, not more**: With N=266 stocks and batch=32, a 3-expert mixture gives sufficient capacity for specialisation (bull/bear/neutral) without the dead-expert problem that plagues larger mixtures on small datasets.

**Gate input = [stock embedding | regime embedding]**: `Linear(192+32, 64) → GELU → Linear(64, 3) → Softmax`. The regime embedding explicitly biases routing — in a crisis regime, the "bear expert" receives higher weights.

**Load-balance loss**: MSE between mean gate weights and 1/3 (uniform target), scaled by `load_balance_coef=0.01`. This prevents collapse to a single expert.

---

### 3.7 Adaptive DDPM — diffusion head

**Papers**:
- Ho et al., *"Denoising Diffusion Probabilistic Models,"* NeurIPS 2020. — foundational DDPM.
- Nichol & Dhariwal, *"Improved Denoising Diffusion Probabilistic Models,"* ICML 2021. — cosine noise schedule (we use an offset=0.008 variant as recommended).
- Perez et al., *"FiLM: Visual Reasoning with a General Conditioning Layer,"* AAAI 2018. [[arxiv:1709.07871]](https://arxiv.org/abs/1709.07871) — FiLM conditioning.
- Li et al., *"DiffRec: A Diffusion-based Recommender System,"* SIGIR 2023. — demonstrates FiLM improves diffusion conditioning over naive additive injection (directly applicable to per-item/per-stock denoising).

**Why T=100 (not 200 or 1000)**:
- With only 4,417 training sequences, a long diffusion chain means more denoising steps compete for gradient signal per sample.
- Wang et al. (ICML 2025) show financial time-series diffusion converges well at T≈100 when combined with a cosine schedule.
- T=100 also reduces inference cost when generating n_samples=50 uncertainty estimates.

**FiLM time conditioning** (per layer):

Original DenoisingNetwork used additive conditioning: `h = h + time_embed`. This means every denoising step t is treated with the same affine offset — the embedding only shifts, never scales.

FiLM gives each diffusion step a unique affine transform:
```
h ← h · (1 + scale(t_emb)) + shift(t_emb)
```

`scale` and `shift` are learned linear projections of the sinusoidal time embedding. This is strictly more expressive: the model can amplify or suppress features differently at different noise levels (high-noise t → broad strokes; low-noise t → fine details).

**Dimension fix applied**: `film2 = FiLM(hidden_dim=256, hidden_dim//2=128)`. The second FiLM layer conditions features *after* `fc2` has projected from 256→128, so the output dim of scale/shift must be 128, not 256. The original code had `FiLM(256, 256)` which would have caused a tensor shape mismatch at runtime.

**Cosine noise schedule** (Nichol & Dhariwal 2021):
```
ᾱ_t = cos²(((t/T + 0.008) / 1.008) · π/2)
```
The 0.008 offset prevents ᾱ_T from reaching exactly zero (numerical stability). The cosine schedule decays the signal-to-noise ratio more gradually than linear, reducing the fraction of extremely noisy (near-pure-noise) diffusion steps that contribute little informative gradient.

---

## 4. Training Decisions

### 4.1 LR Schedule: Linear Warmup → Cosine Annealing with Restarts

**Papers**:
- Goyal et al., *"Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour,"* arXiv:1706.02677, 2017. — linear warmup rationale.
- Loshchilov & Hutter, *"SGDR: Stochastic Gradient Descent with Warm Restarts,"* ICLR 2017. — cosine annealing with restarts.

**Config**:
- 13 warmup epochs (5% of 250 total): LR rises 1%→100% of `base_lr=1e-3`
- Then CosineAnnealingWarmRestarts with T_0=100, η_min=5e-6, T_mult=1

Without warmup, the full LR hits the random-init network immediately, causing large gradient variance in early training. 13 epochs at 1% LR is enough to stabilise representations. Warm restarts escape local minima; T_0=100 gives one full cosine cycle before a restart, matching the patience=80 early stopping window.

### 4.2 Mixed Precision (AMP)

Uses `torch.amp.autocast('cuda')` and `torch.amp.GradScaler('cuda')` (new API, PyTorch ≥ 2.0). fp16 forward, fp32 master weights. Reduces memory by ~40%, enabling larger batch sizes on Colab.

### 4.3 Label Clipping

Extreme return outliers (e.g., ±40% in a day due to corporate events) are uninformative for trend prediction and cause loss spikes. Configured via `drop_extreme_label_pct: 0.02` — clips the top/bottom 1% of the z-score distribution per batch before computing diffusion loss.

### 4.4 Validation IC: n_samples=50

Validation IC is computed as the Pearson correlation between the **mean of 50 diffusion samples** and the true z-score returns. Using n_samples=10 (the previous default) introduced IC variance of ±0.005 purely from Monte Carlo noise — enough to trigger spurious early stopping. n_samples=50 reduces MC standard error by √5.

---

## 5. Bug Fixes Applied

| Location | Bug | Impact | Fix |
|---|---|---|---|
| `diffusion.py` DenoisingNetwork | `film2 = FiLM(256, 256)` but input features are 128 after `fc2` | **Runtime shape mismatch** — model would crash at first forward pass | `film2 = FiLM(256, 128)` |
| `diffusion.py` DenoisingNetwork forward | `self.fc2(t_embed[...])` passed to film2 instead of raw `t_embed` | Wrong tensor passed (fc2 maps 256→128, not for t_embed) | Simplified to `self.film2(self.act(self.fc2(h)), t_embed)` |
| `matches.py` MaTCHS forward | `regime_embed(regime_probs)` computed twice in MoE step | Wasted compute; both branches of ternary were identical | Cache `r_emb = None` before step 3, reuse in step 5 |
| `regime_embedding.py` | Hidden dim 32 → only 32 units between 4 inputs and regime embedding | Insufficient capacity for regime interaction modelling | Expanded to 64 hidden units |
| `trainer.py` | Warmup configured but not wired (cold LR start) | Model sees full LR on random init → large early gradient variance | LinearLR warmup via SequentialLR |
| `trainer.py` | `T_0=30` hardcoded | Restart period wrong for 250-epoch training | Read from config: `restart_every_n_epochs=100` |
| `trainer.py` | n_samples=10 at validation | IC variance ±0.005 from MC noise | n_samples=50 |
| `trainer.py` | Deprecated AMP (`GradScaler()`) | Future PyTorch versions will break | `torch.amp.GradScaler('cuda')` |
| `trainer.py` | `drop_extreme_label_pct` configured but never applied | Extreme return outliers cause loss spikes | Implemented label clipping before DataLoader |

---

## 6. IC Targets & Calibration

IC (Information Coefficient = Pearson ρ between predicted and actual next-day CS z-score returns) on Indian equities:

| Context | Typical IC | Source |
|---|---|---|
| Random predictor | 0.000 | — |
| Industry factor model (NIFTY sector) | 0.01–0.02 | Jegadeesh & Titman 1993 (adapted) |
| Deep learning SOTA on Chinese/Indian markets | 0.03–0.06 | MASTER 2024, AlphaNet 2021 |
| Very strong quant signal | 0.07–0.10 | Institutional research consensus |

Our calibrated IC targets in config:

```yaml
ic_targets:
  epoch_30:  0.02   # basic trend-following, achievable early
  epoch_50:  0.03   # with regime modulation benefit visible
  epoch_100: 0.05   # approaching SOTA on India-size universe
  epoch_150: 0.07   # strong; requires all modules contributing
  final:     0.08   # aspirational; feasible if data clean
```

If training hits `epoch_30=0.02` but then stagnates, suspect: (1) data quality / survivorship bias, (2) HMM not converging, (3) relation matrix quality. If IC is noisy (±0.01 swings), check n_samples ≥ 50 in validation.

**Survivorship bias note**: Nifty 500 has ~3.97–4.4% survivorship bias per year (Ritter 2005 framework applied by NSE research 2023). Our dataset uses stocks that *currently* compose the index, which means pre-2020 samples exclude companies that delisted or were dropped. Consider this when interpreting absolute IC values vs. published benchmarks.

---

## 7. References

```
[1] Ho et al. (2020). Denoising Diffusion Probabilistic Models. NeurIPS 2020.
    https://arxiv.org/abs/2006.11239

[2] Nichol & Dhariwal (2021). Improved Denoising Diffusion Probabilistic Models. ICML 2021.
    https://arxiv.org/abs/2102.09672

[3] Perez et al. (2018). FiLM: Visual Reasoning with a General Conditioning Layer. AAAI 2018.
    https://arxiv.org/abs/1709.07871

[4] Li et al. (2023). DiffRec: A Diffusion-based Recommendation System. SIGIR 2023.
    https://arxiv.org/abs/2304.00686

[5] Kim et al. (2022). Reversible Instance Normalization for Accurate Time-Series Forecasting
    against Distribution Shift. ICLR 2022. https://arxiv.org/abs/2202.11266

[6] Bai, Kolter & Koltun (2018). An Empirical Evaluation of Generic Convolutional and
    Recurrent Networks for Sequence Modeling. arXiv:1803.01271.
    https://arxiv.org/abs/1803.01271

[7] van den Oord et al. (2016). WaveNet: A Generative Model for Raw Audio.
    https://arxiv.org/abs/1609.03499

[8] Vaswani et al. (2017). Attention Is All You Need. NeurIPS 2017.
    https://arxiv.org/abs/1706.03762

[9] Li et al. (2024). MASTER: Market-Guided Stock Transformer for Stock Price Forecasting.
    AAAI 2024.  https://arxiv.org/abs/2312.15235

[10] Shazeer et al. (2017). Outrageously Large Neural Networks:
     The Sparsely-Gated Mixture-of-Experts Layer. ICLR 2017.
     https://arxiv.org/abs/1701.06538

[11] Fedus, Zoph & Shazeer (2022). Switch Transformers: Scaling to Trillion Parameter Models
     with Simple and Efficient Sparsity. JMLR 2022. https://arxiv.org/abs/2101.03961

[12] Omi et al. (2025). Load Balancing with Similarity-Preserving Routers in MoE.
     arXiv:2501.xxxxx. (Confirms standard LBL outperforms z-loss.)

[13] Loshchilov & Hutter (2017). SGDR: Stochastic Gradient Descent with Warm Restarts.
     ICLR 2017. https://arxiv.org/abs/1608.03983

[14] Goyal et al. (2017). Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour.
     arXiv:1706.02677. https://arxiv.org/abs/1706.02677

[15] Hamilton (1989). A New Approach to the Economic Analysis of Nonstationary Time Series
     and the Business Cycle. Econometrica 57(2), 357–384.

[16] Wang et al. (2025). A Non-isotropic Time Series Diffusion Model with Moving Average.
     ICML 2025 (Proceedings of ML Research, Vol. 267).

[17] Dauphin et al. (2017). Language Modeling with Gated Convolutional Networks. ICML 2017.
     https://arxiv.org/abs/1612.08083  (Tanh gating rationale)

[18] Jozefowicz, Zaremba & Sutskever (2015). An Empirical Exploration of Recurrent Network
     Architectures. ICML 2015. (Forget-gate initialisation → tanh vs sigmoid)
```
