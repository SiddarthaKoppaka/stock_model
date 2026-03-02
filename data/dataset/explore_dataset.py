"""
DiffSTOCK India — Dataset Deep Analysis Script
================================================
Run: python3 analyze_dataset.py
Output: Full diagnostic report printed to console
"""

import numpy as np
import os
import sys
from collections import defaultdict

# ─────────────────────────────────────────────
# CONFIG — change path if needed
# ─────────────────────────────────────────────
DATASET_PATH = "/Users/siddartha/Documents/Personal/stock_model/diffstock_india/data/dataset/nifty500_10yr.npz"

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
SEP  = "=" * 72
SEP2 = "─" * 72

def header(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

def subheader(title):
    print(f"\n  {SEP2}")
    print(f"  {title}")
    print(f"  {SEP2}")

def fmt(val, decimals=4):
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)

def pct(val, total):
    return f"{100 * val / total:.2f}%" if total > 0 else "N/A"

# ─────────────────────────────────────────────
# 1. FILE CHECK
# ─────────────────────────────────────────────
header("1. FILE INFORMATION")

if not os.path.exists(DATASET_PATH):
    print(f"  ❌  File not found: {DATASET_PATH}")
    print("  → Update DATASET_PATH at the top of this script.")
    sys.exit(1)

file_size_gb = os.path.getsize(DATASET_PATH) / (1024 ** 3)
file_size_mb = os.path.getsize(DATASET_PATH) / (1024 ** 2)
print(f"  Path  : {DATASET_PATH}")
print(f"  Size  : {file_size_gb:.3f} GB  ({file_size_mb:.1f} MB)")

data = np.load(DATASET_PATH, allow_pickle=True)
keys = list(data.keys())
print(f"  Keys  : {keys}")

# ─────────────────────────────────────────────
# 2. RAW KEY OVERVIEW
# ─────────────────────────────────────────────
header("2. ARRAY SHAPES & MEMORY")
print(f"  {'Key':<25} {'Shape':<30} {'Dtype':<12} {'Memory'}")
print(f"  {'─'*24} {'─'*29} {'─'*11} {'─'*10}")
total_mem_mb = 0
for k in keys:
    arr = data[k]
    mem_mb = arr.nbytes / (1024 ** 2)
    total_mem_mb += mem_mb
    shape_str = str(arr.shape)
    print(f"  {k:<25} {shape_str:<30} {str(arr.dtype):<12} {mem_mb:.1f} MB")
print(f"\n  Total in-memory: {total_mem_mb:.1f} MB")

# ─────────────────────────────────────────────
# 3. DETECT SPLITS
# ─────────────────────────────────────────────
header("3. TRAIN / VAL / TEST SPLIT ANALYSIS")

splits = {}
for prefix in ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]:
    if prefix in data:
        splits[prefix] = data[prefix]

if splits:
    for name, arr in splits.items():
        print(f"  {name:<12}: shape={arr.shape}")

    # Infer dataset dimensions
    if "X_train" in splits:
        X = splits["X_train"]
        ndim = X.ndim
        print(f"\n  X_train ndim: {ndim}")

        if ndim == 4:
            n_train, L, N, F = X.shape
            print(f"\n  Inferred dimensions:")
            print(f"    Training days   (T_train) : {n_train}")
            print(f"    Lookback window (L)        : {L}")
            print(f"    Stocks          (N)        : {N}")
            print(f"    Features        (F)        : {F}")
        elif ndim == 3:
            n_train, N, F = X.shape
            print(f"\n  Inferred dimensions (3D):")
            print(f"    Training samples: {n_train}")
            print(f"    Stocks (N)      : {N}")
            print(f"    Features (F)    : {F}")

    # Split size breakdown
    if "X_train" in splits and "X_val" in splits and "X_test" in splits:
        t = splits["X_train"].shape[0]
        v = splits["X_val"].shape[0]
        te = splits["X_test"].shape[0]
        total = t + v + te
        print(f"\n  Split breakdown:")
        print(f"    Train : {t:>6} days  ({pct(t, total)})")
        print(f"    Val   : {v:>6} days  ({pct(v, total)})")
        print(f"    Test  : {te:>6} days  ({pct(te, total)})")
        print(f"    Total : {total:>6} days  (~{total/252:.1f} years of trading)")
else:
    print("  ⚠  No standard train/val/test keys found.")
    print(f"  Found keys: {keys}")

# ─────────────────────────────────────────────
# 4. FEATURE ANALYSIS (X_train)
# ─────────────────────────────────────────────
header("4. FEATURE ANALYSIS (X_train)")

if "X_train" in splits:
    X_train = splits["X_train"]

    # NaN / Inf check
    nan_count = np.sum(np.isnan(X_train))
    inf_count = np.sum(np.isinf(X_train))
    total_vals = X_train.size
    print(f"  Total values : {total_vals:,}")
    print(f"  NaN count    : {nan_count:,}  ({pct(nan_count, total_vals)})")
    print(f"  Inf count    : {inf_count:,}  ({pct(inf_count, total_vals)})")

    if nan_count == 0 and inf_count == 0:
        print("  ✅  No NaN or Inf values — data is clean")
    else:
        print("  ❌  Found NaN/Inf — needs fixing before training!")

    # Per-feature stats (last dimension = features)
    if X_train.ndim == 4:
        n_days, L, N, F = X_train.shape
        subheader(f"Per-Feature Statistics (F={F} features, averaged over days×lookback×stocks)")
        print(f"  {'Feature':<10} {'Mean':>10} {'Std':>10} {'Min':>12} {'Max':>12} {'NaN%':>8}")
        print(f"  {'─'*9} {'─'*10} {'─'*10} {'─'*12} {'─'*12} {'─'*8}")
        for f_idx in range(F):
            feat_data = X_train[:, :, :, f_idx].flatten()
            finite = feat_data[np.isfinite(feat_data)]
            nan_pct = 100 * np.sum(np.isnan(feat_data)) / len(feat_data)
            if len(finite) > 0:
                print(f"  F{f_idx:<9} {finite.mean():>10.4f} {finite.std():>10.4f} "
                      f"{finite.min():>12.4f} {finite.max():>12.4f} {nan_pct:>7.2f}%")

        # Check for extreme values (outliers)
        subheader("Outlier Check (values beyond ±5 std from mean)")
        flat = X_train[np.isfinite(X_train)]
        global_mean = flat.mean()
        global_std  = flat.std()
        extreme_mask = np.abs(X_train - global_mean) > 5 * global_std
        extreme_count = np.sum(extreme_mask & np.isfinite(X_train))
        print(f"  Global mean  : {global_mean:.4f}")
        print(f"  Global std   : {global_std:.4f}")
        print(f"  Extreme vals : {extreme_count:,} ({pct(extreme_count, total_vals)})")
        if extreme_count / total_vals < 0.001:
            print("  ✅  Outlier rate < 0.1% — acceptable")
        else:
            print("  ⚠  High outlier rate — consider clipping at ±3 or ±5 std")

        # Check normalization (are features z-scored?)
        subheader("Normalization Check (are features properly scaled?)")
        sample_means = []
        sample_stds  = []
        for f_idx in range(min(F, 5)):
            feat = X_train[:, :, :, f_idx]
            finite = feat[np.isfinite(feat)]
            sample_means.append(finite.mean())
            sample_stds.append(finite.std())
        avg_mean = np.mean(np.abs(sample_means))
        avg_std  = np.mean(sample_stds)
        print(f"  Avg |mean| of first 5 features: {avg_mean:.4f}  (want ≈ 0)")
        print(f"  Avg std   of first 5 features: {avg_std:.4f}  (want ≈ 1)")
        if avg_mean < 0.5 and 0.5 < avg_std < 3.0:
            print("  ✅  Features appear reasonably normalized")
        elif avg_mean > 2.0:
            print("  ⚠  High mean — features may not be z-scored")
        elif avg_std > 5.0:
            print("  ⚠  High std — features may need scaling")

# ─────────────────────────────────────────────
# 5. LABEL ANALYSIS (y_train)
# ─────────────────────────────────────────────
header("5. LABEL ANALYSIS (y_train — next-day returns)")

if "y_train" in splits:
    y_train = splits["y_train"]
    y_val   = splits.get("y_val",  None)
    y_test  = splits.get("y_test", None)

    print(f"  y_train shape: {y_train.shape}  (days × stocks)")

    for name, y in [("train", y_train), ("val", y_val), ("test", y_test)]:
        if y is None:
            continue
        flat = y.flatten()
        finite = flat[np.isfinite(flat)]
        nan_count = np.sum(np.isnan(flat))
        if len(finite) == 0:
            continue
        pos = np.sum(finite > 0)
        neg = np.sum(finite < 0)
        print(f"\n  [{name}]")
        print(f"    Samples  : {y.shape[0]} days × {y.shape[1]} stocks = {len(flat):,} labels")
        print(f"    NaN      : {nan_count} ({pct(nan_count, len(flat))})")
        print(f"    Mean     : {finite.mean():.6f}  (want ≈ 0 for returns)")
        print(f"    Std      : {finite.std():.6f}")
        print(f"    Min      : {finite.min():.6f}")
        print(f"    Max      : {finite.max():.6f}")
        print(f"    Positive : {pos} ({pct(pos, len(finite))}) — up days")
        print(f"    Negative : {neg} ({pct(neg, len(finite))}) — down days")
        print(f"    P10/P25  : {np.percentile(finite, 10):.4f} / {np.percentile(finite, 25):.4f}")
        print(f"    P75/P90  : {np.percentile(finite, 75):.4f} / {np.percentile(finite, 90):.4f}")

        # Check for extreme labels
        extreme_labels = np.sum(np.abs(finite) > 0.20)
        print(f"    |return| > 20% : {extreme_labels} ({pct(extreme_labels, len(finite))})")
        if finite.std() < 0.005:
            print("    ⚠  Std very low — labels may already be cross-sectionally normalized (CS-ZScore)")
        elif finite.std() < 0.05:
            print("    ✅  Std in normal daily return range")
        else:
            print("    ⚠  Std very high — check if returns are in percentage vs decimal form")

    # Label distribution balance check (important for IC metric)
    subheader("Label Balance (critical for IC metric)")
    flat_train = y_train.flatten()
    finite_train = flat_train[np.isfinite(flat_train)]
    skew = float(np.mean(((finite_train - finite_train.mean()) / finite_train.std()) ** 3))
    kurt = float(np.mean(((finite_train - finite_train.mean()) / finite_train.std()) ** 4)) - 3
    print(f"  Skewness : {skew:.4f}  (want |skew| < 1 for balanced labels)")
    print(f"  Kurtosis : {kurt:.4f}  (excess; >3 = fat tails, common in finance)")
    if abs(skew) > 2:
        print("  ⚠  High skew — consider DropExtremeLabel (top/bottom 2.5%)")
    else:
        print("  ✅  Label skew acceptable")
    if kurt > 5:
        print("  ⚠  Fat tails present — DDPM needs to model this; diffusion helps here")

# ─────────────────────────────────────────────
# 6. RELATION MASK ANALYSIS
# ─────────────────────────────────────────────
header("6. RELATION MASK / MATRIX ANALYSIS")

relation_keys = [k for k in keys if any(x in k.lower() for x in 
                 ["relation", "mask", "adj", "graph", "edge", "corr", "matrix"])]

if not relation_keys:
    print("  ⚠  No relation/mask keys found — check key names above")
    print(f"  All keys: {keys}")
else:
    for rk in relation_keys:
        arr = data[rk]
        print(f"\n  [{rk}]  shape={arr.shape}  dtype={arr.dtype}")

        if arr.dtype == bool or 'bool' in str(arr.dtype):
            total_pairs = arr.size
            # Handle diagonal
            if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
                N = arr.shape[0]
                off_diag_mask = ~np.eye(N, dtype=bool)
                off_diag_connections = arr[off_diag_mask].sum()
                off_diag_total = off_diag_mask.sum()
                density = 100 * off_diag_connections / off_diag_total
                print(f"    Density (off-diagonal): {density:.2f}%")
                print(f"    Connected pairs        : {off_diag_connections:,} of {off_diag_total:,}")

                # Per-stock connectivity
                degree = arr.sum(axis=1) - arr.diagonal()  # subtract self-loops
                print(f"    Avg neighbors per stock: {degree.mean():.1f}")
                print(f"    Min neighbors          : {degree.min()}")
                print(f"    Max neighbors          : {degree.max()}")
                isolated = np.sum(degree == 0)
                print(f"    Isolated stocks (0 nbr): {isolated}")

                if density < 10:
                    print(f"    ⚠  Very sparse ({density:.1f}%) — MRT attention will be severely limited")
                    print(f"       → Recommend lowering corr_threshold to 0.25–0.30")
                elif density < 20:
                    print(f"    ⚠  Sparse ({density:.1f}%) — consider lowering corr_threshold")
                elif density < 40:
                    print(f"    ✅  Moderate density ({density:.1f}%) — acceptable for MRT")
                else:
                    print(f"    ✅  Dense ({density:.1f}%) — good cross-stock signal")

                if isolated > 0:
                    print(f"    ❌  {isolated} stocks have NO connections — they'll get full attention (unintended)")

        elif arr.dtype in [np.float32, np.float64]:
            if arr.ndim == 2:
                N = arr.shape[0]
                off_diag = arr[~np.eye(N, dtype=bool)]
                nonzero  = np.sum(off_diag != 0)
                density  = 100 * nonzero / len(off_diag)
                print(f"    Non-zero density (off-diag): {density:.2f}%")
                finite   = off_diag[np.isfinite(off_diag)]
                print(f"    Value range: [{finite.min():.4f}, {finite.max():.4f}]")
                print(f"    Mean value : {finite.mean():.4f}")

        if arr.ndim == 3:
            print(f"    (3D tensor — {arr.shape[0]} relation types × {arr.shape[1]} × {arr.shape[2]})")
            for r_idx in range(arr.shape[0]):
                layer = arr[r_idx]
                if layer.dtype == bool or 'bool' in str(layer.dtype):
                    d = 100 * layer.sum() / layer.size
                elif layer.dtype in [np.float32, np.float64]:
                    d = 100 * np.count_nonzero(layer) / layer.size
                else:
                    d = float('nan')
                print(f"    Relation [{r_idx}] density: {d:.2f}%")

# ─────────────────────────────────────────────
# 7. DATA SUFFICIENCY ASSESSMENT
# ─────────────────────────────────────────────
header("7. DATA SUFFICIENCY ASSESSMENT")

if "X_train" in splits and splits["X_train"].ndim == 4:
    n_train, L, N, F = splits["X_train"].shape
    n_val   = splits["X_val"].shape[0]   if "X_val"  in splits else 0
    n_test  = splits["X_test"].shape[0]  if "X_test" in splits else 0

    # Parameter estimates
    d_model_128 = 128
    d_model_64  = 64
    params_full  = 7_500_000
    params_small = 2_000_000  # with d_model=64

    effective_samples = n_train * N  # stock-day observations

    print(f"  Training days          : {n_train}")
    print(f"  Stocks (N)             : {N}")
    print(f"  Effective stock-days   : {effective_samples:,}  (days × N)")
    print(f"  Lookback (L)           : {L}")
    print(f"  Features (F)           : {F}")
    print(f"  Approx trading years   : {n_train/252:.1f}  (train) | {n_val/252:.1f} (val) | {n_test/252:.1f} (test)")

    print(f"\n  Model Size vs Data:")
    ratio_full  = params_full  / n_train
    ratio_small = params_small / n_train
    print(f"    Full model  (7.5M params): {ratio_full:,.0f} params/sample  ⚠  HIGH RISK")
    print(f"    Small model (2.0M params): {ratio_small:,.0f} params/sample  ✅  SAFER")
    print(f"    (target: <500 params/sample for good generalization)")

    print(f"\n  Verdict:")
    if n_train < 1000:
        print(f"    ❌  INSUFFICIENT — {n_train} days is too few. Need 1500+ training days.")
    elif n_train < 1500:
        print(f"    ⚠  BORDERLINE — {n_train} days. Use d_model=64, heavy regularization.")
    elif n_train < 2500:
        print(f"    ✅  ADEQUATE — {n_train} days. Use d_model=64-128 with augmentation.")
    else:
        print(f"    ✅  GOOD — {n_train} days. Full model should work.")

# ─────────────────────────────────────────────
# 8. STOCK-LEVEL DATA QUALITY
# ─────────────────────────────────────────────
header("8. PER-STOCK DATA QUALITY (X_train — checking coverage)")

if "X_train" in splits and splits["X_train"].ndim == 4:
    X_train = splits["X_train"]
    n_days, L, N, F = X_train.shape

    # For each stock, count how many days have NaN in any feature
    # Shape: (n_days, L, N, F) -> check axis 3 (features) and axis 1 (lookback)
    # A stock-day is "missing" if all features are NaN at t=0 (last timestep)
    last_step = X_train[:, -1, :, :]  # (n_days, N, F)
    stock_nan_days = np.sum(np.any(np.isnan(last_step), axis=2), axis=0)  # (N,)
    stock_coverage = 100 * (1 - stock_nan_days / n_days)

    print(f"  Checking {N} stocks over {n_days} training days...")
    print(f"\n  Stock coverage distribution (% days with valid data):")
    for threshold in [100, 99, 95, 90, 80, 70]:
        count = np.sum(stock_coverage >= threshold)
        print(f"    ≥ {threshold}% coverage: {count:>4} stocks ({pct(count, N)})")

    poor_stocks = np.sum(stock_coverage < 80)
    print(f"\n  Stocks with < 80% coverage (poor quality): {poor_stocks}")
    if poor_stocks > 0:
        print(f"  ⚠  {poor_stocks} stocks have significant missing data — consider re-filtering")

    print(f"\n  Worst 10 stocks by coverage:")
    worst_idx = np.argsort(stock_coverage)[:10]
    for idx in worst_idx:
        print(f"    Stock[{idx:03d}]: {stock_coverage[idx]:.1f}% coverage  "
              f"({int(stock_nan_days[idx])} missing days)")

# ─────────────────────────────────────────────
# 9. TEMPORAL CONSISTENCY
# ─────────────────────────────────────────────
header("9. TEMPORAL CONSISTENCY (label drift over time)")

if "y_train" in splits:
    y_train = splits["y_train"]
    n_days  = y_train.shape[0]

    # Split training period into 4 quarters and check distribution shift
    quarter = n_days // 4
    periods = {
        "Q1 (oldest)": y_train[:quarter],
        "Q2"         : y_train[quarter:2*quarter],
        "Q3"         : y_train[2*quarter:3*quarter],
        "Q4 (recent)": y_train[3*quarter:],
    }

    print(f"  Checking if return distributions shift over {n_days} training days...")
    print(f"\n  {'Period':<15} {'Mean':>10} {'Std':>10} {'% Positive':>12} {'Max Drawdown':>14}")
    print(f"  {'─'*14} {'─'*10} {'─'*10} {'─'*12} {'─'*14}")

    for pname, period in periods.items():
        flat = period.flatten()
        finite = flat[np.isfinite(flat)]
        if len(finite) == 0:
            continue
        pos_pct = 100 * np.sum(finite > 0) / len(finite)
        # Crude drawdown: cumulative mean return
        cum_ret = np.nancumsum(period.mean(axis=1))
        running_max = np.maximum.accumulate(cum_ret)
        dd = cum_ret - running_max
        max_dd = dd.min() if len(dd) > 0 else 0
        print(f"  {pname:<15} {finite.mean():>10.5f} {finite.std():>10.5f} "
              f"{pos_pct:>11.1f}% {max_dd:>14.4f}")

    print(f"\n  (If Mean or Std vary wildly across quarters → distribution shift / regime change)")
    print(f"  (This is normal for Indian markets — model needs DoubleAdapt or walk-forward)")

# ─────────────────────────────────────────────
# 10. FINAL CHECKLIST
# ─────────────────────────────────────────────
header("10. PRE-TRAINING CHECKLIST")

checks = []

if "X_train" in splits:
    X = splits["X_train"]
    nan_ok = np.sum(np.isnan(X)) == 0
    checks.append(("No NaN in X_train",    "✅" if nan_ok else "❌"))
    checks.append(("No Inf in X_train",    "✅" if np.sum(np.isinf(X)) == 0 else "❌"))
    checks.append(("4D tensor (T,L,N,F)",  "✅" if X.ndim == 4 else f"⚠  got {X.ndim}D"))

if "y_train" in splits:
    y = splits["y_train"]
    nan_ok_y = np.sum(np.isnan(y)) == 0
    flat = y.flatten()
    finite = flat[np.isfinite(flat)]
    std_ok = 0.001 < finite.std() < 0.5 if len(finite) > 0 else False
    checks.append(("No NaN in y_train",    "✅" if nan_ok_y else f"⚠  {np.sum(np.isnan(y))} NaNs"))
    checks.append(("Label std reasonable", "✅" if std_ok else f"⚠  std={finite.std():.5f}"))

if "X_train" in splits and "X_val" in splits:
    train_days = splits["X_train"].shape[0]
    val_days   = splits["X_val"].shape[0]
    ok = train_days > val_days * 3
    checks.append(("Train > 3× Val size",  "✅" if ok else "⚠  imbalanced splits"))

if "X_train" in splits and splits["X_train"].ndim == 4:
    n_train = splits["X_train"].shape[0]
    sufficient = n_train >= 1500
    checks.append((f"Training days ≥ 1500", "✅" if sufficient else f"⚠  only {n_train} days"))

print()
for desc, status in checks:
    print(f"  {status}  {desc}")

print(f"\n{SEP}")
print("  Analysis complete. Paste this output back to Claude for interpretation.")
print(SEP)