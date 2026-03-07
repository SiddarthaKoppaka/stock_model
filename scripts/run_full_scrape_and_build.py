"""
Full 20-year scrape + dataset rebuild script.

Steps:
  1. Back up existing raw CSVs (data/raw → data/raw_backup_2015)
  2. Clear CSV files from data/raw (keeps metadata.json + failed_symbols.txt)
  3. Run scraper with start_date=2005-01-01 from config
  4. Rebuild nifty500_20yr.npz
  5. Print leakage verification

Usage:
    cd diffstock_india
    .venv/bin/python scripts/run_full_scrape_and_build.py 2>&1 | tee logs/scrape_build_20yr.log
"""

import sys
import shutil
from pathlib import Path

# ── resolve project root ───────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.logger import setup_logger
setup_logger(log_dir=ROOT / "logs", log_level="INFO")

from loguru import logger

# ── Step 0: imports ────────────────────────────────────────────────────────────
import yaml, numpy as np, pandas as pd
from scipy.stats import spearmanr

# ── Load config ────────────────────────────────────────────────────────────────
config_path = ROOT / "config" / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

raw_dir     = ROOT / config["paths"]["raw_data"]
dataset_dir = ROOT / config["paths"]["dataset"]

# ── Step 1: Back up existing CSVs ─────────────────────────────────────────────
backup_dir = raw_dir.parent / "raw_backup_2015"
if not backup_dir.exists():
    logger.info(f"Backing up existing raw CSVs → {backup_dir}")
    backup_dir.mkdir(parents=True)
    existing_csvs = list(raw_dir.glob("*.csv"))
    for csv in existing_csvs:
        shutil.copy2(csv, backup_dir / csv.name)
    logger.info(f"Backed up {len(existing_csvs)} CSV files to {backup_dir}")
else:
    logger.info(f"Backup already exists at {backup_dir}, skipping backup step")

# ── Step 2: Clear existing CSV files from raw_dir ─────────────────────────────
existing_csvs = list(raw_dir.glob("*.csv"))
logger.info(f"Removing {len(existing_csvs)} existing CSVs from {raw_dir} ...")
for csv in existing_csvs:
    csv.unlink()
logger.info("Cleared existing CSVs — ready for fresh 2005–2026 download")

# ── Step 3: Run full pipeline with scraping ────────────────────────────────────
from src.data.dataset_builder import DatasetBuilder

builder = DatasetBuilder(
    config=config,
    lookback_window=config["data"]["lookback_window"],
    prediction_horizon=config["data"].get("label_horizon", 5),
)

output_path = builder.build_full_dataset(run_scraping=True)
logger.info(f"Dataset built → {output_path}")

# ── Step 4: Leakage verification ───────────────────────────────────────────────
logger.info("Running leakage verification ...")

dataset_file = dataset_dir / "nifty500_20yr.npz"
data = np.load(dataset_file, allow_pickle=True)

print("\n" + "=" * 60)
print("LEAKAGE VERIFICATION — nifty500_20yr.npz")
print("=" * 60)

for split in ["train", "val", "test"]:
    X = data[f"X_{split}"]   # (T, L, N, F)
    y = data[f"y_{split}"]   # (T, N)
    if X.shape[0] == 0:
        print(f"[{split.upper()}] No samples — skipping")
        continue
    last_day_feats = X[:, -1, :, 0].flatten()   # open_ret_norm
    labels         = y.flatten()
    r, _  = spearmanr(last_day_feats, labels)
    fixed = "✓ YES" if abs(r) < 0.05 else "✗ NO (LEAKAGE!)"
    n     = X.shape[0]
    N     = X.shape[2]
    print(f"[{split.upper():5s}] Lag-0 IC (open_ret_norm vs y): r={r:+.4f}  →  Leakage fixed: {fixed}  (T={n}, N={N})")

print(f"\nDataset file exists: {dataset_file.exists()}")
print("=" * 60)
