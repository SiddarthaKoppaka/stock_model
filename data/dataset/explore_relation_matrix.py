"""
DiffSTOCK India — Relation Matrix Deep Analysis
================================================
Run: python3 analyze_relations.py
"""

import numpy as np
import os
import sys

# ─────────────────────────────────────────────
# CONFIG — update path if needed
# ─────────────────────────────────────────────
# Try common locations automatically
CANDIDATE_PATHS = [
    "/Users/siddartha/Documents/Personal/stock_model/diffstock_india/data/dataset/relation_matrices.npz",
    "/Users/siddartha/Documents/Personal/stock_model/diffstock_india/data/processed/relation_matrices.npz",
    "/Users/siddartha/Documents/Personal/stock_model/diffstock_india/data/relation_matrices.npz",
]

SEP  = "=" * 72
SEP2 = "─" * 72

def header(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

def subheader(title):
    print(f"\n  {SEP2}\n  {title}\n  {SEP2}")

# ─────────────────────────────────────────────
# 1. FIND FILE
# ─────────────────────────────────────────────
header("1. LOCATING FILE")

path = None
for p in CANDIDATE_PATHS:
    if os.path.exists(p):
        path = p
        break

if path is None:
    print("  ❌  File not found at any default location.")
    print("  Tried:")
    for p in CANDIDATE_PATHS:
        print(f"    {p}")
    print("\n  → Set RELATION_PATH manually below and rerun.")
    # Uncomment and set manually:
    # path = "/your/actual/path/relation_matrices.npz"
    sys.exit(1)

size_mb = os.path.getsize(path) / (1024 ** 2)
print(f"  ✅  Found: {path}")
print(f"  Size: {size_mb:.2f} MB")

data = np.load(path, allow_pickle=True)
keys = list(data.keys())
print(f"  Keys: {keys}")

# ─────────────────────────────────────────────
# 2. RAW SHAPE OVERVIEW
# ─────────────────────────────────────────────
header("2. ARRAY SHAPES & DTYPES")
print(f"  {'Key':<30} {'Shape':<25} {'Dtype':<12} {'Memory'}")
print(f"  {'─'*29} {'─'*24} {'─'*11} {'─'*10}")
for k in keys:
    arr = data[k]
    mem = arr.nbytes / (1024 ** 2)
    print(f"  {k:<30} {str(arr.shape):<25} {str(arr.dtype):<12} {mem:.2f} MB")

# ─────────────────────────────────────────────
# 3. PER-MATRIX DEEP ANALYSIS
# ─────────────────────────────────────────────
header("3. PER-MATRIX ANALYSIS")

N_expected = 329  # from dataset audit

for k in keys:
    arr = data[k]
    print(f"\n  ▶ [{k}]")
    print(f"    Shape : {arr.shape}   Dtype: {arr.dtype}")

    # ── 2D matrix ──────────────────────────────
    if arr.ndim == 2:
        N, M = arr.shape
        if N != M:
            print(f"    ⚠  Non-square matrix ({N}×{M}) — unusual for relation")

        is_bool   = arr.dtype == bool or 'bool' in str(arr.dtype)
        is_float  = arr.dtype in [np.float32, np.float64]

        diag_vals = np.diag(arr)
        off_diag  = arr[~np.eye(N, dtype=bool)]

        if is_bool:
            density   = 100 * off_diag.sum() / len(off_diag)
            self_loop = diag_vals.sum()
            print(f"    Type  : Binary (bool)")
            print(f"    Density (off-diag)  : {density:.2f}%")
            print(f"    Connected pairs     : {int(off_diag.sum()):,} of {len(off_diag):,}")
            print(f"    Self-loops (diag=1) : {int(self_loop)} of {N}")

            # Symmetry check
            is_sym = np.array_equal(arr, arr.T)
            print(f"    Symmetric           : {'✅ Yes' if is_sym else '❌ No — asymmetric!'}")

            # Per-stock degree
            degree = arr.sum(axis=1).astype(int) - diag_vals.astype(int)
            print(f"\n    Per-stock neighbor count:")
            print(f"      Mean : {degree.mean():.1f}")
            print(f"      Std  : {degree.std():.1f}")
            print(f"      Min  : {degree.min()}  ← {'❌ isolated stocks!' if degree.min()==0 else '✅'}")
            print(f"      Max  : {degree.max()}")

            isolated = np.sum(degree == 0)
            if isolated > 0:
                print(f"\n    ❌  {isolated} stocks have ZERO neighbors!")
                print(f"       These stocks will use full attention in MRT (unmasked)")
                print(f"       → Lower corr_threshold or add sector fallback connection")

            # Density verdict
            print(f"\n    Density verdict:")
            if density < 10:
                print(f"      ❌  Very sparse ({density:.1f}%) — MRT severely limited")
                print(f"         → Lower corr_threshold from 0.4 → 0.25")
            elif density < 20:
                print(f"      ⚠  Sparse ({density:.1f}%) — consider lowering threshold")
                print(f"         → Try corr_threshold = 0.30")
            elif density < 40:
                print(f"      ✅  Moderate ({density:.1f}%) — good for MRT")
            else:
                print(f"      ✅  Dense ({density:.1f}%) — rich cross-stock signal")

        elif is_float:
            finite    = off_diag[np.isfinite(off_diag)]
            nonzero   = np.count_nonzero(np.abs(off_diag) > 1e-6)
            density   = 100 * nonzero / len(off_diag)
            is_sym    = np.allclose(arr, arr.T, atol=1e-5)

            print(f"    Type  : Float (continuous weights)")
            print(f"    Non-zero density    : {density:.2f}%")
            print(f"    Value range         : [{finite.min():.4f}, {finite.max():.4f}]")
            print(f"    Mean (off-diag)     : {finite.mean():.4f}")
            print(f"    Std  (off-diag)     : {finite.std():.4f}")
            print(f"    Symmetric           : {'✅ Yes' if is_sym else '❌ No'}")

            # Distribution of weights
            buckets = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            print(f"\n    Weight distribution (correlation strength):")
            pos_vals = finite[finite > 0]
            neg_vals = finite[finite < 0]
            print(f"      Positive weights: {len(pos_vals):,} ({100*len(pos_vals)/len(finite):.1f}%)")
            print(f"      Negative weights: {len(neg_vals):,} ({100*len(neg_vals)/len(finite):.1f}%)")
            if len(pos_vals) > 0:
                print(f"      Pos mean        : {pos_vals.mean():.4f}")
            if len(neg_vals) > 0:
                print(f"      Neg mean        : {neg_vals.mean():.4f}")

    # ── 3D tensor ──────────────────────────────
    elif arr.ndim == 3:
        R, N, M = arr.shape
        print(f"    Type  : 3D tensor ({R} relation types × {N} × {M})")
        print(f"    ── Per-relation-type breakdown ──")

        for r_idx in range(R):
            layer = arr[r_idx]
            off   = layer[~np.eye(N, dtype=bool)]

            if layer.dtype == bool or 'bool' in str(layer.dtype):
                d = 100 * off.sum() / len(off)
                deg = layer.sum(axis=1) - np.diag(layer)
                print(f"\n    Relation[{r_idx}]  density={d:.2f}%  "
                      f"avg_neighbors={deg.mean():.1f}  "
                      f"isolated={int((deg==0).sum())}")
            elif layer.dtype in [np.float32, np.float64]:
                finite = off[np.isfinite(off)]
                nz = 100 * np.count_nonzero(np.abs(off) > 1e-6) / len(off)
                print(f"\n    Relation[{r_idx}]  non-zero={nz:.2f}%  "
                      f"range=[{finite.min():.3f}, {finite.max():.3f}]  "
                      f"mean={finite.mean():.4f}")

    # ── 1D array ───────────────────────────────
    elif arr.ndim == 1:
        print(f"    1D array — likely stock symbols or metadata")
        if arr.dtype.kind in ['U', 'S', 'O']:
            print(f"    First 10: {list(arr[:10])}")
            print(f"    Last  10: {list(arr[-10:])}")
        else:
            finite = arr[np.isfinite(arr.astype(float))]
            print(f"    Range: [{finite.min():.4f}, {finite.max():.4f}]")
            print(f"    Mean : {finite.mean():.4f}")

# ─────────────────────────────────────────────
# 4. COMBINED MASK ANALYSIS
# ─────────────────────────────────────────────
header("4. COMBINED MASK ANALYSIS (union of all relations)")

bool_matrices = []
float_matrices = []

for k in keys:
    arr = data[k]
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        if arr.dtype == bool or 'bool' in str(arr.dtype):
            bool_matrices.append((k, arr))
        elif arr.dtype in [np.float32, np.float64]:
            float_matrices.append((k, arr))
    elif arr.ndim == 3:
        R, N, M = arr.shape
        if N == M:
            for r in range(R):
                layer = arr[r]
                if layer.dtype == bool or 'bool' in str(layer.dtype):
                    bool_matrices.append((f"{k}[{r}]", layer))
                elif layer.dtype in [np.float32, np.float64]:
                    float_matrices.append((f"{k}[{r}]", layer))

if bool_matrices:
    N = bool_matrices[0][1].shape[0]
    combined = np.zeros((N, N), dtype=bool)
    for name, mat in bool_matrices:
        combined |= mat
    off_diag = combined[~np.eye(N, dtype=bool)]
    density = 100 * off_diag.sum() / len(off_diag)
    degree  = combined.sum(axis=1) - np.diag(combined).astype(int)
    isolated = int((degree == 0).sum())

    print(f"  Union of {len(bool_matrices)} bool matrices:")
    print(f"    Combined density   : {density:.2f}%")
    print(f"    Avg neighbors/stock: {degree.mean():.1f}")
    print(f"    Isolated stocks    : {isolated}")

    if density < 15:
        print(f"\n  ❌  Combined mask still too sparse ({density:.1f}%)")
        print(f"     MRT attention will be severely restricted")
        print(f"     → Rebuild with lower corr_threshold (0.25–0.30)")
    elif density < 30:
        print(f"\n  ⚠  Borderline density ({density:.1f}%)")
        print(f"     → Consider soft mask (float weights) instead of binary")
    else:
        print(f"\n  ✅  Good combined density ({density:.1f}%) — MRT will work well")

elif float_matrices:
    print("  Only float matrices found — no binary union computed")
    print("  Ensure your MRT masking code converts to attention bias correctly")
else:
    print("  ⚠  No 2D or 3D square matrices found for union analysis")

# ─────────────────────────────────────────────
# 5. COMPATIBILITY CHECK WITH DATASET
# ─────────────────────────────────────────────
header("5. COMPATIBILITY WITH MAIN DATASET (N=329 stocks)")

print(f"  Expected N from dataset: {N_expected}")
issues = []

for k in keys:
    arr = data[k]
    if arr.ndim == 2:
        if arr.shape[0] != N_expected or arr.shape[1] != N_expected:
            issues.append(f"  ❌  {k}: shape {arr.shape} ≠ ({N_expected},{N_expected})")
        else:
            print(f"  ✅  {k}: shape matches ({N_expected}×{N_expected})")
    elif arr.ndim == 3:
        if arr.shape[1] != N_expected or arr.shape[2] != N_expected:
            issues.append(f"  ❌  {k}: shape {arr.shape} — stock dims ≠ {N_expected}")
        else:
            print(f"  ✅  {k}: shape matches ({arr.shape[0]}×{N_expected}×{N_expected})")

if issues:
    for issue in issues:
        print(issue)
    print("\n  → Stock count mismatch will cause runtime error in MRT forward pass!")
else:
    print("\n  ✅  All matrices compatible with 329-stock dataset")

# ─────────────────────────────────────────────
# 6. LOADING CODE FOR TRAINER
# ─────────────────────────────────────────────
header("6. HOW TO LOAD IN YOUR TRAINER")

print("""
  Add this to your trainer.py or run_train.py:

    import numpy as np
    import torch

    rel = np.load('data/dataset/relation_matrices.npz')
    print("Relation keys:", list(rel.keys()))

    # If binary masks — combine into one union mask
    # (replace 'sector_mask', 'corr_mask' with your actual key names)
    R_mask = None
    for k in rel.keys():
        arr = rel[k]
        if arr.ndim == 2 and (arr.dtype == bool or 'bool' in str(arr.dtype)):
            mask = torch.tensor(arr, dtype=torch.bool)
            R_mask = mask if R_mask is None else R_mask | mask

    # If already a combined 3D tensor
    # R_mask = torch.tensor(rel['relation_mask'], dtype=torch.bool)  # (R, N, N)

    # Convert to attention bias for MRT:
    # True  = allowed to attend
    # False = masked out (set to -inf before softmax)
    attn_bias = torch.zeros(N, N)
    attn_bias[~R_mask] = float('-inf')

    print(f"R_mask shape : {R_mask.shape}")
    print(f"Mask density : {R_mask.float().mean().item()*100:.2f}%")
""")

print(f"\n{SEP}")
print("  Paste full output back to Claude for next steps.")
print(SEP)