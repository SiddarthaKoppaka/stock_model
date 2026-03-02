import numpy as np
import pandas as pd
from scipy.stats import spearmanr

data = np.load('dataset/nifty500_10yr.npz', allow_pickle=True)
X_train = data['X_train']
y_train = data['y_train']
dates   = data['dates_train']

symbol = str(data['stock_symbols'][0])
df = pd.read_parquet(f"processed/{symbol}_features.parquet")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Check multiple samples, not just t=0
print("Checking alignment for first 5 training samples:")
print(f"{'t':>4} {'dates_train[t]':>20} {'next_trading_day':>20} {'y[t,0]':>10} {'close_ret_next':>15} {'match':>6}")
print("-" * 80)

for t in range(5):
    last_input = pd.to_datetime(dates[t])
    idx = df[df['Date'] == last_input].index
    if len(idx) == 0:
        print(f"{t:>4} {str(last_input.date()):>20} {'NOT FOUND':>20}")
        continue
    idx = idx[0]
    next_close = df.loc[idx+1, 'close_ret'] if idx+1 < len(df) else None
    next_date = df.loc[idx+1, 'Date'].date() if idx+1 < len(df) else None
    match = abs(y_train[t,0] - next_close) < 0.0001 if next_close else False
    print(f"{t:>4} {str(last_input.date()):>20} {str(next_date):>20} {y_train[t,0]:>10.6f} {next_close:>15.6f} {str(match):>6}")

# Now check: what is X_train[t, -1] compared to parquet?
# Specifically: does last timestep open_ret_norm match label day or last input day?
print(f"\nFor t=0, X_train[0, -1, 0, 0] (open_ret_norm last step) = {X_train[0, -1, 0, 0]:.6f}")

last_input_date = pd.to_datetime(dates[0])
label_date = df[df['Date'] > last_input_date]['Date'].iloc[0]
idx_input = df[df['Date'] == last_input_date].index[0]
idx_label = df[df['Date'] == label_date].index[0]

print(f"open_ret_norm on last input day ({last_input_date.date()}): {df.loc[idx_input, 'open_ret_norm']:.6f}")
print(f"open_ret_norm on label day     ({label_date.date()}):       {df.loc[idx_label, 'open_ret_norm']:.6f}")
print(f"\nX matches input day: {abs(X_train[0,-1,0,0] - df.loc[idx_input,'open_ret_norm']) < 0.001}")
print(f"X matches label day: {abs(X_train[0,-1,0,0] - df.loc[idx_label,'open_ret_norm']) < 0.001}")