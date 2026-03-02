import numpy as np

rel = np.load('relation_matrices.npz')
R_corr_raw = rel['R_corr']  # already thresholded, but we need raw corr

# We can't recover raw corr from saved file — need to check what threshold gives what density
# Simulate by checking value distribution of current R_corr (thresholded at 0.15)
vals = R_corr_raw[R_corr_raw > 0].flatten()
print(f"Current pos values: {len(vals):,}")
print(f"Distribution of saved correlation values:")
for t in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    count = (vals >= t).sum()
    density = 100 * count / (329 * 329)
    print(f"  threshold={t:.2f}  →  R_corr density={density:.2f}%")