"""
DiffSTOCK Trainer

Comprehensive training loop with:
- EMA (Exponential Moving Average) of weights
- Gradient clipping
- Cosine annealing with warm restarts
- Noise augmentation
- Mixed precision training
- Checkpointing and early stopping
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
import json
import shutil
import os
from tqdm import tqdm
from loguru import logger

from ..model.diffstock import DiffSTOCK
from ..evaluation.metrics import compute_all_metrics


class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update EMA weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + \
                                   (1 - self.decay) * param.data

    def apply_shadow(self):
        """Apply EMA weights to model (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class DiffSTOCKTrainer:
    """
    Trainer for DiffSTOCK model.
    """

    def __init__(
        self,
        model: DiffSTOCK,
        config: Dict,
        R_mask: torch.Tensor,
        device: torch.device = None
    ):
        """
        Initialize trainer.

        Args:
            model: DiffSTOCK model
            config: Configuration dict
            R_mask: Relation mask tensor (N, N)
            device: Device to train on
        """
        self.model = model
        self.config = config
        self.R_mask = R_mask.to(device) if device else R_mask

        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )

        # LR Scheduler (Cosine annealing with warm restarts)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=30,
            T_mult=2
        )

        # EMA
        self.ema = EMA(self.model, decay=config['training']['ema_decay'])

        # Training state
        self.current_epoch = 0
        self.best_val_ic = -float('inf')
        self.epochs_without_improvement = 0

        # Paths
        root = Path(config['paths']['root'])
        self.checkpoint_dir = root / config['paths']['checkpoints']
        self.log_dir = root / config['paths']['logs']

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Mixed precision
        self.use_amp = torch.cuda.is_available()
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Mixed precision: {self.use_amp}")

    def _build_crisis_weights(self, dates: np.ndarray) -> torch.Tensor:
        """Build per-sample weights for crisis period upsampling."""
        factor = self.config['training'].get('crisis_upsample_factor', 1.0)
        crisis_periods = self.config['training'].get('crisis_periods', [])

        weights = np.ones(len(dates), dtype=np.float32)
        dates_ts = pd.to_datetime(dates)

        for period in crisis_periods:
            start = pd.Timestamp(period['start'])
            end = pd.Timestamp(period['end'])
            mask = (dates_ts >= start) & (dates_ts <= end)
            weights[mask] = factor

        return torch.FloatTensor(weights)

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")

        for batch in pbar:
            # Batch may be (X, y) or (X, y, regime_probs)
            if len(batch) == 3:
                X, y, regime_probs = batch
                regime_probs = regime_probs.to(self.device)
            else:
                X, y = batch
                regime_probs = None

            X = X.to(self.device)  # (B, L, N, F)
            y = y.to(self.device)  # (B, N)

            # Add noise augmentation
            noise_level = self.config['training']['noise_augmentation']
            X = X + torch.randn_like(X) * noise_level

            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    loss, _ = self.model(X, self.R_mask, y, regime_probs=regime_probs)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['grad_clip']
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss, _ = self.model(X, self.R_mask, y, regime_probs=regime_probs)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['grad_clip']
                )
                self.optimizer.step()

            # Update EMA
            self.ema.update()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({'loss': total_loss / n_batches})

        return total_loss / n_batches

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict:
        """Evaluate on validation set using EMA weights."""
        self.model.eval()
        self.ema.apply_shadow()

        all_predictions = []
        all_targets = []

        for batch in tqdm(val_loader, desc="Validation"):
            if len(batch) == 3:
                X, y, regime_probs = batch
                regime_probs = regime_probs.to(self.device)
            else:
                X, y = batch
                regime_probs = None

            X = X.to(self.device)
            y = y.to(self.device)

            predictions, _ = self.model(X, self.R_mask, n_samples=10, regime_probs=regime_probs)
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(y.cpu().numpy())

        self.ema.restore()

        y_pred = np.concatenate(all_predictions, axis=0)
        y_true = np.concatenate(all_targets, axis=0)
        return compute_all_metrics(y_pred, y_true)

    def train(
        self,
        train_data: Tuple,
        val_data: Tuple,
    ) -> Dict:
        """
        Full training loop.

        Args:
            train_data: (X, y) | (X, y, dates) | (X, y, dates, regime_probs)
            val_data:   (X, y) | (X, y, regime_probs)

        Returns:
            Training history
        """
        # Unpack train data
        if len(train_data) == 4:
            X_train, y_train, dates_train, regime_probs_train = train_data
        elif len(train_data) == 3:
            X_train, y_train, dates_train = train_data
            regime_probs_train = None
        else:
            X_train, y_train = train_data
            dates_train = None
            regime_probs_train = None

        # Unpack val data
        if len(val_data) == 3:
            X_val, y_val, regime_probs_val = val_data
        else:
            X_val, y_val = val_data
            regime_probs_val = None

        # Build crisis-period upsampling weights
        use_crisis = (
            dates_train is not None
            and self.config['training'].get('crisis_upsample_factor', 1.0) > 1.0
            and len(self.config['training'].get('crisis_periods', [])) > 0
        )
        if use_crisis:
            sample_weights = self._build_crisis_weights(dates_train)
            n_crisis = int((sample_weights > 1).sum().item())
            logger.info(f"Crisis upsampling: {n_crisis} crisis samples at "
                        f"{self.config['training']['crisis_upsample_factor']}x weight")
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            train_shuffle = False
        else:
            sampler = None
            train_shuffle = True

        # Build train dataset (include regime_probs as 3rd tensor if available)
        train_tensors = [torch.FloatTensor(X_train), torch.FloatTensor(y_train)]
        if regime_probs_train is not None:
            train_tensors.append(torch.FloatTensor(regime_probs_train))
        train_dataset = TensorDataset(*train_tensors)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=train_shuffle,
            sampler=sampler,
            num_workers=0,
        )

        # Build val dataset
        val_tensors = [torch.FloatTensor(X_val), torch.FloatTensor(y_val)]
        if regime_probs_val is not None:
            val_tensors.append(torch.FloatTensor(regime_probs_val))
        val_dataset = TensorDataset(*val_tensors)

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=0,
        )

        # Training history
        history = {
            'train_loss': [],
            'val_metrics': [],
            'learning_rates': []
        }

        max_epochs = self.config['training']['max_epochs']
        patience = self.config['training']['patience']
        report_every = self.config['evaluation']['report_every']

        logger.info(f"Starting training for {max_epochs} epochs")
        logger.info(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")

        for epoch in range(max_epochs):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch(train_loader)
            history['train_loss'].append(train_loss)
            history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])

            # Step scheduler
            self.scheduler.step()

            # Evaluate
            if (epoch + 1) % report_every == 0 or epoch == 0:
                val_metrics = self.evaluate(val_loader)
                history['val_metrics'].append(val_metrics)

                logger.info(f"\nEpoch {epoch + 1}/{max_epochs}")
                logger.info(f"  Train Loss: {train_loss:.4f}")
                logger.info(f"  Val IC: {val_metrics['IC']:.4f}")
                logger.info(f"  Val ICIR: {val_metrics.get('ICIR', 0):.4f}")
                logger.info(f"  Val Accuracy: {val_metrics['Accuracy']:.4f}")

                # Check for improvement
                if val_metrics['IC'] > self.best_val_ic:
                    self.best_val_ic = val_metrics['IC']
                    self.epochs_without_improvement = 0

                    # Save best model locally
                    self.save_checkpoint('best_model.pt')
                    logger.info(f"  New best model saved! IC: {self.best_val_ic:.4f}")

                    # Copy to Google Drive (for Colab persistence)
                    drive_ckpt = '/content/drive/MyDrive/DiffSTOCK_Outputs/checkpoints'
                    if os.path.exists('/content/drive'):
                        os.makedirs(drive_ckpt, exist_ok=True)
                        local_ckpt = str(self.checkpoint_dir / 'best_model.pt')
                        shutil.copy(local_ckpt, f'{drive_ckpt}/best_model_epoch{epoch}_ic{val_metrics["IC"]:.4f}.pt')
                        shutil.copy(local_ckpt, f'{drive_ckpt}/best_model.pt')
                        logger.info(f"  Checkpoint copied to Drive")
                else:
                    self.epochs_without_improvement += report_every

            # Early stopping
            if self.epochs_without_improvement >= patience:
                logger.info(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"epoch_{epoch + 1}.pt")

        # Save final model
        self.save_checkpoint('final_model.pt')

        # Save history
        history_path = self.log_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            history_json = {
                'train_loss': [float(x) for x in history['train_loss']],
                'val_metrics': [
                    {k: float(v) for k, v in m.items()}
                    for m in history['val_metrics']
                ],
                'learning_rates': [float(x) for x in history['learning_rates']]
            }
            json.dump(history_json, f, indent=2)

        logger.info(f"\nTraining complete! Best val IC: {self.best_val_ic:.4f}")

        return history

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'ema_shadow': self.ema.shadow,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_ic': self.best_val_ic,
            'config': self.config
        }

        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.ema.shadow = checkpoint['ema_shadow']
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_ic = checkpoint['best_val_ic']

        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
