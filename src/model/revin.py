"""
RevIN: Reversible Instance Normalization

Normalizes each training sample using robust statistics (median + IQR)
computed over its own 20-day lookback window. This makes features regime-
agnostic: a 2008 crisis window and a 2022 calm window both appear as unit-
scale deviations within their own context.

Reference: Kim et al. (2022) "Reversible Instance Normalization for Accurate
Time-Series Forecasting"
"""

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """
    Reversible Instance Normalization using robust statistics.

    Normalizes x over the time (lookback) dimension per (batch, stock, feature).
    Uses median + normalized IQR (IQR/1.3489 ≈ std for a Gaussian) to prevent
    a single crash day from distorting the entire window's normalization.

    Optionally learns per-feature affine parameters (gamma, beta) so the model
    can re-scale features after normalization.

    Args:
        n_features: number of input features (F)
        eps: numerical stability constant
        affine: whether to learn per-feature gamma and beta
    """

    def __init__(self, n_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.n_features = n_features
        self.eps = eps
        self.affine = affine

        if affine:
            self.gamma = nn.Parameter(torch.ones(n_features))
            self.beta  = nn.Parameter(torch.zeros(n_features))

        # Cached stats for potential denorm (not used when labels are CS z-scores)
        self._median   = None
        self._iqr_norm = None

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize x using per-sample robust window statistics.

        Args:
            x: (B, N, L, F)  — batch, stocks, lookback, features

        Returns:
            (B, N, L, F)  — normalized, with affine if enabled
        """
        # Robust stats over L dimension (20-day window)
        # median: (B, N, F)
        median = x.median(dim=2).values

        # IQR via sorting — avoid torch.quantile which is slow on older PyTorch
        x_sorted, _ = x.sort(dim=2)          # (B, N, L, F)
        L = x.shape[2]
        q25_idx = max(0, int(0.25 * L) - 1)
        q75_idx = min(L - 1, int(0.75 * L))
        q25 = x_sorted[:, :, q25_idx, :]     # (B, N, F)
        q75 = x_sorted[:, :, q75_idx, :]     # (B, N, F)
        iqr_norm = (q75 - q25) / 1.3489      # (B, N, F), ~std for normal dist

        self._median   = median
        self._iqr_norm = iqr_norm

        x_norm = (x - median.unsqueeze(2)) / (iqr_norm.unsqueeze(2) + self.eps)

        if self.affine:
            # gamma/beta broadcast over (B, N, L, F)
            x_norm = x_norm * self.gamma + self.beta

        return x_norm

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reverse the normalization on model output.

        Args:
            x: (B, N)  — predictions in normalized space

        Returns:
            (B, N)  — predictions in original feature scale

        Note: Only meaningful when labels are in the same raw scale as inputs.
        When labels are cross-sectional z-scores this is a no-op and can be
        skipped; it is provided for completeness and ablation experiments.
        """
        if self._iqr_norm is None:
            return x

        if self.affine:
            x = (x - self.beta[3]) / (self.gamma[3] + self.eps)  # feature 3 = close_ret

        # Use close_ret (feature index 3) stats for return denormalization
        iqr_norm_close = self._iqr_norm[:, :, 3]   # (B, N)
        median_close   = self._median[:, :, 3]      # (B, N)

        return x * (iqr_norm_close + self.eps) + median_close
