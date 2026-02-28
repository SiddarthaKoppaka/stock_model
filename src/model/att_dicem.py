"""
Att-DiCEm: Attention-gated Dilated Causal Encoder for temporal modeling.

Processes each stock's time series independently using dilated causal convolutions
with attention gating mechanism.

Input shape: (B, N, L, F) - batch, stocks, timesteps, features
Output shape: (B, N, d_model) - temporal embeddings per stock
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class CausalConv1d(nn.Module):
    """
    Causal 1D convolution with dilation.
    Ensures no information from future timesteps leaks into past.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        groups: int = 1
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=self.padding,
            groups=groups
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, L) tensor

        Returns:
            (B, C, L) tensor with causal masking
        """
        x = self.conv(x)
        # Remove future timesteps added by padding
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x


class DepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise separable convolution (reduces parameters by ~8x).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1
    ):
        super().__init__()
        # Depthwise: each input channel convolved separately
        self.depthwise = CausalConv1d(
            in_channels,
            in_channels,
            kernel_size,
            dilation=dilation,
            groups=in_channels
        )
        # Pointwise: 1x1 conv to mix channels
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class AttDiCEm(nn.Module):
    """
    Attention-gated Dilated Causal Encoder Module.

    Architecture:
        - 4 dilated conv layers with dilation [1, 2, 4, 8]
        - Depthwise separable convolutions for efficiency
        - LayerNorm + GELU activation
        - Attention gate on final output
    """

    def __init__(
        self,
        in_features: int,
        d_model: int,
        n_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.25
    ):
        """
        Args:
            in_features: Number of input features (F)
            d_model: Model dimension
            n_layers: Number of dilated conv layers
            kernel_size: Convolution kernel size
            dropout: Dropout probability
        """
        super().__init__()

        self.in_features = in_features
        self.d_model = d_model
        self.n_layers = n_layers

        # Input projection
        self.input_proj = nn.Linear(in_features, d_model)

        # Dilated causal conv layers
        self.conv_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for i in range(n_layers):
            dilation = 2 ** i  # [1, 2, 4, 8, ...]

            self.conv_layers.append(
                DepthwiseSeparableConv1d(
                    d_model,
                    d_model,
                    kernel_size=kernel_size,
                    dilation=dilation
                )
            )

            self.layer_norms.append(nn.LayerNorm(d_model))

        # Attention gate
        self.attention_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

        # Final projection
        self.output_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, N, L, F)
               B = batch size
               N = number of stocks
               L = sequence length (lookback window)
               F = number of features

        Returns:
            Temporal embeddings of shape (B, N, d_model)
        """
        B, N, L, n_feats = x.shape
        assert n_feats == self.in_features, f"Expected {self.in_features} features, got {n_feats}"

        # Reshape: treat each stock independently (use reshape instead of view for non-contiguous tensors)
        x = x.reshape(B * N, L, n_feats)  # (B*N, L, n_feats)

        # Input projection
        x = self.input_proj(x)  # (B*N, L, d_model)

        # Transpose for Conv1d: (B*N, d_model, L)
        x = x.transpose(1, 2)

        # Apply dilated causal convolutions
        for i, (conv, norm) in enumerate(zip(self.conv_layers, self.layer_norms)):
            residual = x

            # Convolution
            x = conv(x)  # (B*N, d_model, L)

            # Transpose for LayerNorm
            x = x.transpose(1, 2)  # (B*N, L, d_model)

            # LayerNorm + GELU
            x = norm(x)
            x = F.gelu(x)
            x = self.dropout(x)

            # Transpose back
            x = x.transpose(1, 2)  # (B*N, d_model, L)

            # Residual connection (if dimensions match)
            if residual.shape == x.shape:
                x = x + residual

        # Take final timestep (L-1)
        x = x[:, :, -1]  # (B*N, d_model)

        # Apply attention gate
        gate = self.attention_gate(x)  # (B*N, d_model)
        x = x * gate

        # Final projection
        x = self.output_proj(x)  # (B*N, d_model)

        # Reshape back to (B, N, d_model)
        x = x.view(B, N, self.d_model)

        return x


if __name__ == "__main__":
    # Test module
    B, N, L, F = 32, 400, 20, 15
    d_model = 128

    model = AttDiCEm(in_features=F, d_model=d_model, n_layers=4)

    x = torch.randn(B, N, L, F)
    out = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Expected shape: ({B}, {N}, {d_model})")

    assert out.shape == (B, N, d_model), f"Shape mismatch! Got {out.shape}"
    print("Att-DiCEm test passed!")
