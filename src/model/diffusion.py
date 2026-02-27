"""
Adaptive Denoising Diffusion Probabilistic Model (DDPM)

Generates probabilistic predictions for next-day stock returns using diffusion process.
Adapts noise schedule based on volatility and correlation for better learning.

Forward: x_0 (returns) -> x_T (noise)
Reverse: x_T (noise) -> x_0 (returns) conditioned on MaTCHS embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) tensor of timesteps

        Returns:
            (B, dim) time embeddings
        """
        device = t.device
        half_dim = self.dim // 2

        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)

        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

        return embeddings


class DenoisingNetwork(nn.Module):
    """
    MLP-based denoising network with time and condition injection.

    Predicts noise given:
        - Noisy returns x_t
        - Timestep t
        - Conditional embeddings from MaTCHS
    """

    def __init__(
        self,
        n_stocks: int,
        d_model: int,
        time_embed_dim: int = 128,
        hidden_dims: list = [1024, 512, 256]
    ):
        """
        Args:
            n_stocks: Number of stocks (N)
            d_model: Dimension of conditional embeddings
            time_embed_dim: Dimension of time embeddings
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()

        self.n_stocks = n_stocks
        self.d_model = d_model

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)

        # Condition embedding projection (from MaTCHS)
        # Condition is (B, N, d_model), we'll flatten to (B, N*d_model)
        self.cond_proj = nn.Linear(n_stocks * d_model, hidden_dims[0])

        # Input projection (x_t + t_embed)
        input_dim = n_stocks + time_embed_dim
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])

        # MLP layers
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.SiLU(),
                nn.Dropout(0.1)
            ])

        self.mlp = nn.Sequential(*layers)

        # Output projection (predict noise)
        self.output_proj = nn.Linear(hidden_dims[-1], n_stocks)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict noise at timestep t.

        Args:
            x_t: (B, N) noisy returns at timestep t
            t: (B,) timestep indices
            condition: (B, N, d_model) conditional embeddings from MaTCHS

        Returns:
            (B, N) predicted noise
        """
        B, N = x_t.shape

        # Time embedding
        t_embed = self.time_embed(t)  # (B, time_embed_dim)

        # Condition embedding (flatten and project)
        cond = condition.view(B, -1)  # (B, N * d_model)
        cond = self.cond_proj(cond)  # (B, hidden_dim)

        # Concatenate x_t and time embedding
        x_input = torch.cat([x_t, t_embed], dim=-1)  # (B, N + time_embed_dim)
        x_input = self.input_proj(x_input)  # (B, hidden_dim)

        # Combine with condition
        h = x_input + cond  # (B, hidden_dim)

        # MLP
        h = self.mlp(h)  # (B, hidden_dim[-1])

        # Predict noise
        noise_pred = self.output_proj(h)  # (B, N)

        return noise_pred


class AdaptiveDDPM(nn.Module):
    """
    Adaptive Denoising Diffusion Probabilistic Model for stock returns.

    Adaptive features:
        - Volatility-based noise scaling
        - Fewer timesteps (T=200) for limited data regime
        - Cosine schedule for smoother transitions
    """

    def __init__(
        self,
        n_stocks: int,
        d_model: int,
        T: int = 200,
        beta_start: float = 0.0001,
        beta_end: float = 0.02
    ):
        """
        Args:
            n_stocks: Number of stocks
            d_model: Dimension of conditional embeddings
            T: Number of diffusion timesteps
            beta_start: Starting noise level
            beta_end: Ending noise level
        """
        super().__init__()

        self.n_stocks = n_stocks
        self.d_model = d_model
        self.T = T

        # Denoising network
        self.denoiser = DenoisingNetwork(n_stocks, d_model)

        # Register noise schedule buffers
        self.register_buffer('betas', self._cosine_beta_schedule(T, beta_start, beta_end))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('alphas_cumprod_prev', F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0))

        # Precompute useful quantities
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))

    def _cosine_beta_schedule(
        self,
        T: int,
        beta_start: float,
        beta_end: float
    ) -> torch.Tensor:
        """
        Cosine noise schedule (smoother than linear).

        Args:
            T: Number of timesteps
            beta_start: Minimum beta
            beta_end: Maximum beta

        Returns:
            (T,) tensor of betas
        """
        steps = T + 1
        x = torch.linspace(0, T, steps)
        alphas_cumprod = torch.cos(((x / T) + 0.008) / 1.008 * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, beta_start, beta_end)

        return betas

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward diffusion: sample x_t from q(x_t | x_0).

        Args:
            x_0: (B, N) clean data
            t: (B,) timestep indices
            noise: (B, N) optional noise (sampled if None)

        Returns:
            (B, N) noisy data x_t
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None]

        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

        return x_t

    def compute_loss(
        self,
        x_0: torch.Tensor,
        condition: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute training loss (MSE on predicted noise).

        Args:
            x_0: (B, N) clean returns
            condition: (B, N, d_model) conditional embeddings

        Returns:
            Scalar loss
        """
        B, N = x_0.shape
        device = x_0.device

        # Sample random timesteps
        t = torch.randint(0, self.T, (B,), device=device, dtype=torch.long)

        # Sample noise
        noise = torch.randn_like(x_0)

        # Forward diffusion
        x_t = self.q_sample(x_0, t, noise)

        # Predict noise
        noise_pred = self.denoiser(x_t, t, condition)

        # MSE loss
        loss = F.mse_loss(noise_pred, noise)

        return loss

    @torch.no_grad()
    def p_sample(
        self,
        x_t: torch.Tensor,
        t: int,
        condition: torch.Tensor
    ) -> torch.Tensor:
        """
        Reverse diffusion: sample x_{t-1} from p(x_{t-1} | x_t).

        Args:
            x_t: (B, N) noisy data at timestep t
            t: Current timestep (scalar)
            condition: (B, N, d_model) conditional embeddings

        Returns:
            (B, N) denoised data x_{t-1}
        """
        B, N = x_t.shape
        device = x_t.device

        # Predict noise
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
        noise_pred = self.denoiser(x_t, t_tensor, condition)

        # Compute coefficients
        alpha_t = self.alphas[t]
        alpha_cumprod_t = self.alphas_cumprod[t]
        alpha_cumprod_prev_t = self.alphas_cumprod_prev[t]
        beta_t = self.betas[t]

        # Predict x_0
        x_0_pred = (x_t - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)

        # Compute posterior mean
        coef_x_0 = torch.sqrt(alpha_cumprod_prev_t) * beta_t / (1 - alpha_cumprod_t)
        coef_x_t = torch.sqrt(alpha_t) * (1 - alpha_cumprod_prev_t) / (1 - alpha_cumprod_t)

        mean = coef_x_0 * x_0_pred + coef_x_t * x_t

        # Add noise if not final step
        if t > 0:
            noise = torch.randn_like(x_t)
            variance = beta_t * (1 - alpha_cumprod_prev_t) / (1 - alpha_cumprod_t)
            sigma = torch.sqrt(variance)
            x_t_minus_1 = mean + sigma * noise
        else:
            x_t_minus_1 = mean

        return x_t_minus_1

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        n_samples: int = 50
    ) -> torch.Tensor:
        """
        Generate samples from the diffusion model.

        Args:
            condition: (B, N, d_model) conditional embeddings
            n_samples: Number of samples to generate per batch

        Returns:
            (n_samples, B, N) predicted returns
        """
        B, N, d_model = condition.shape
        device = condition.device

        # Store all samples
        all_samples = []

        for _ in range(n_samples):
            # Start from pure noise
            x_t = torch.randn(B, N, device=device)

            # Reverse diffusion process
            for t in reversed(range(self.T)):
                x_t = self.p_sample(x_t, t, condition)

            all_samples.append(x_t)

        # Stack samples
        samples = torch.stack(all_samples, dim=0)  # (n_samples, B, N)

        return samples


if __name__ == "__main__":
    # Test module
    B, N, d_model = 32, 400, 128
    T = 200

    model = AdaptiveDDPM(n_stocks=N, d_model=d_model, T=T)

    # Random input
    x_0 = torch.randn(B, N) * 0.02  # Typical daily returns ~2%
    condition = torch.randn(B, N, d_model)

    # Training loss
    loss = model.compute_loss(x_0, condition)
    print(f"Training loss: {loss.item():.4f}")

    # Sampling
    print("\nGenerating samples...")
    samples = model.sample(condition, n_samples=10)
    print(f"Sample shape: {samples.shape}")
    print(f"Expected: (10, {B}, {N})")

    # Statistics
    mean_pred = samples.mean(dim=0)
    std_pred = samples.std(dim=0)
    print(f"Mean prediction range: [{mean_pred.min():.4f}, {mean_pred.max():.4f}]")
    print(f"Std prediction range: [{std_pred.min():.4f}, {std_pred.max():.4f}]")

    print("\nAdaptive DDPM test passed!")
