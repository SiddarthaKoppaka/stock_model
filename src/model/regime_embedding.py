"""
Regime Embedding

Converts soft HMM regime probability vectors into a dense embedding that is
broadcast to all stocks and concatenated with the MaTCHS stock representations
before they enter the MRT cross-stock attention layer.

This lets the model condition its relational reasoning on the current market
regime: the same normalised RSI reading means something different during a
crash state vs. a calm bull state.
"""

import torch
import torch.nn as nn


class RegimeEmbedding(nn.Module):
    """
    Maps (batch, n_states) soft regime probs → (batch, embedding_dim) embedding.

    Architecture:
        Linear(n_states, 32) → LayerNorm → GELU → Linear(32, embedding_dim)

    The output is broadcast to (batch, n_stocks, embedding_dim) and concatenated
    with the per-stock Att-DiCEm output before the MRT. A projection layer
    then maps back to d_model.

    Args:
        n_states:      number of HMM states (default 4)
        embedding_dim: output embedding dimension (default 16)
    """

    def __init__(self, n_states: int = 4, embedding_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, embedding_dim),
        )

    def forward(self, regime_probs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            regime_probs: (B, n_states) soft probabilities

        Returns:
            (B, embedding_dim) regime embedding
        """
        return self.net(regime_probs)
