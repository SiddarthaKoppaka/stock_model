import numpy as np
import pandas as pd
from scipy.stats import spearmanr

data = np.load('dataset/nifty500_20yr.npz', allow_pickle=True)
X_train = data['X_train']
y_train = data['y_train']
dates   = data['dates_train']

symbol = str(data['stock_symbols'][0])
df = pd.read_parquet(f"processed/{symbol}_features.parquet")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

target_date = pd.to_datetime(dates[0])
idx = df[df['Date'] == target_date].index[0]

print("=== LEAK CHECK ===")
print(f"dates_train[0] = {dates[0]}  ← should be last INPUT day")
print(f"y_train[0, 0]  = {y_train[0, 0]:.6f}")
print(f"close_ret on dates[0]    = {df.loc[idx,   'close_ret']:.6f}  (should NOT match)")
print(f"close_ret on dates[0]+1  = {df.loc[idx+1, 'close_ret']:.6f}  (should match)")
print(f"Leak fixed: {abs(y_train[0,0] - df.loc[idx+1,'close_ret']) < 0.001}")

print("\n=== FEATURE CORRELATION CHECK ===")
labels = y_train.flatten()
for f in range(X_train.shape[-1]):
    feat = X_train[:, -1, :, f].flatten()
    corr, _ = spearmanr(feat, labels)
    flag = " ⚠ LEAK" if abs(corr) > 0.15 else ""
    print(f"  F{f} ({data['feature_names'][f]}): {corr:.4f}{flag}")