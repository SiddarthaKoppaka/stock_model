"""
Fix NaN values in the dataset by replacing with 0.
"""
import numpy as np
from pathlib import Path

def fix_dataset_nans():
    """Replace NaN values in dataset with 0."""
    dataset_path = Path('data/dataset/nifty500_20yr.npz')

    print(f"Loading dataset from {dataset_path}...")
    data = np.load(dataset_path, allow_pickle=True)

    # Load all arrays
    arrays = {}
    for key in data.keys():
        arrays[key] = data[key]

    # Fix NaNs in numeric arrays
    for split in ['train', 'val', 'test']:
        X_key = f'X_{split}'
        y_key = f'y_{split}'

        if X_key in arrays:
            X = arrays[X_key]
            nan_count = np.isnan(X).sum()
            if nan_count > 0:
                print(f"\n{X_key}:")
                print(f"  Shape: {X.shape}")
                print(f"  NaN count: {nan_count:,} ({100*nan_count/X.size:.2f}%)")
                arrays[X_key] = np.nan_to_num(X, nan=0.0)
                print(f"  Fixed: All NaNs replaced with 0")

        if y_key in arrays:
            y = arrays[y_key]
            nan_count = np.isnan(y).sum()
            if nan_count > 0:
                print(f"\n{y_key}:")
                print(f"  Shape: {y.shape}")
                print(f"  NaN count: {nan_count:,} ({100*nan_count/y.size:.2f}%)")
                arrays[y_key] = np.nan_to_num(y, nan=0.0)
                print(f"  Fixed: All NaNs replaced with 0")

    # Save fixed dataset
    backup_path = dataset_path.parent / f"{dataset_path.stem}_backup.npz"
    print(f"\nBacking up original to {backup_path}...")
    import shutil
    shutil.copy(dataset_path, backup_path)

    print(f"\nSaving fixed dataset to {dataset_path}...")
    np.savez(dataset_path, **arrays)

    print("\n✓ Dataset fixed successfully!")
    print(f"\nVerifying fix...")

    # Verify
    data = np.load(dataset_path, allow_pickle=True)
    for split in ['train', 'val', 'test']:
        X_key = f'X_{split}'
        y_key = f'y_{split}'

        if X_key in data:
            X = data[X_key]
            nan_count = np.isnan(X).sum()
            print(f"{X_key}: {nan_count} NaNs (expected 0)")

        if y_key in data:
            y = data[y_key]
            nan_count = np.isnan(y).sum()
            print(f"{y_key}: {nan_count} NaNs (expected 0)")

    print("\n✓ All done!")

if __name__ == '__main__':
    fix_dataset_nans()
