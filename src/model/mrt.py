"""
Masked Relational Transformer (MRT)

Multi-head self-attention across stocks with relation-based masking.
Each stock attends only to stocks connected via relation matrices.

Input shape: (B, N, d_model) - stock embeddings from Att-DiCEm
Mask shape: (N, N) - boolean attention mask
Output shape: (B, N, d_model) - relational embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadRelationalAttention(nn.Module):
    """
    Multi-head attention with relation-based masking.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Query, Key, Value projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, d_model) tensor
            mask: (N, N) boolean mask - True where attention is allowed

        Returns:
            (B, N, d_model) attended features
        """
        B, N, d_model = x.shape

        # Linear projections and split into heads
        Q = self.W_q(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)  # (B, h, N, d_k)
        K = self.W_k(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)  # (B, h, N, d_k)
        V = self.W_v(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)  # (B, h, N, d_k)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, h, N, N)

        # Apply mask (set masked positions to large negative value)
        if mask is not None:
            # Expand mask for batch and heads: (N, N) -> (B, h, N, N)
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
            mask = mask.expand(B, self.n_heads, N, N)

            # Apply mask: use torch.finfo to get appropriate min value for the dtype
            # This automatically handles FP16, FP32, etc. without overflow
            mask_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~mask, mask_value)

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)  # (B, h, N, N)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, V)  # (B, h, N, d_k)

        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(B, N, d_model)  # (B, N, d_model)

        # Final projection
        out = self.W_o(out)

        return out


class TransformerBlock(nn.Module):
    """
    Transformer block with Pre-LN architecture and masked attention.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_ff: int = None,
        dropout: float = 0.25
    ):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension (default: 4 * d_model)
            dropout: Dropout probability
        """
        super().__init__()

        if d_ff is None:
            d_ff = 4 * d_model

        # Multi-head attention
        self.attention = MultiHeadRelationalAttention(d_model, n_heads, dropout)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        # Layer norms (Pre-LN)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, d_model) tensor
            mask: (N, N) attention mask

        Returns:
            (B, N, d_model) transformed tensor
        """
        # Pre-LN: Layer norm before attention
        residual = x
        x = self.norm1(x)
        x = self.attention(x, mask)
        x = self.dropout(x)
        x = x + residual

        # Pre-LN: Layer norm before FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + residual

        return x


class MaskedRelationalTransformer(nn.Module):
    """
    Masked Relational Transformer for stock-to-stock attention.

    Uses relation matrices to mask attention:
    - Stocks only attend to related stocks (sector/industry/correlated)
    - Multiple transformer blocks for deep relational reasoning
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.25
    ):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer blocks
            dropout: Dropout probability
        """
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        R_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with masked attention.

        Args:
            x: (B, N, d_model) stock embeddings
            R_mask: (N, N) relation mask - 1.0 where attention allowed, 0.0 otherwise

        Returns:
            (B, N, d_model) relational embeddings
        """
        B, N, d_model = x.shape
        assert d_model == self.d_model, f"Expected d_model={self.d_model}, got {d_model}"

        # Convert R_mask to boolean (for masking)
        mask = R_mask.bool()  # (N, N)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        # Final layer norm
        x = self.final_norm(x)

        return x


if __name__ == "__main__":
    # Test module
    B, N, d_model = 32, 400, 128
    n_heads = 8
    n_layers = 3

    model = MaskedRelationalTransformer(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers
    )

    # Create random input
    x = torch.randn(B, N, d_model)

    # Create random relation mask (50% density)
    R_mask = (torch.rand(N, N) > 0.5).float()
    # Ensure diagonal is 0 (no self-attention via relations)
    R_mask.fill_diagonal_(0)

    # Forward pass
    out = model(x, R_mask)

    print(f"Input shape: {x.shape}")
    print(f"Mask shape: {R_mask.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Mask density: {R_mask.mean():.2%}")

    assert out.shape == (B, N, d_model), f"Shape mismatch! Got {out.shape}"
    print("MRT test passed!")
