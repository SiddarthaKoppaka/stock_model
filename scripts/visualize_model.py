"""
DHARMA Model Visualizer

Generates three complementary views of the model:

  1. TensorBoard  — computation graph + per-layer parameter histograms
                    Launch: tensorboard --logdir runs/viz --port 6006
                    Then open: http://localhost:6006

  2. torchinfo    — layer-by-layer table (shapes, parameters, MACs)
                    Printed to stdout + saved to runs/viz/model_summary.txt

  3. Parameter report — breakdown by sub-module with trainable counts
                    Saved to runs/viz/parameter_report.txt

Usage:
    cd diffstock_india
    .venv/bin/python scripts/visualize_model.py
    tensorboard --logdir runs/viz --port 6006
"""

import sys
import yaml
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary as torchinfo_summary

with open(ROOT / "config" / "config.yaml") as f:
    config = yaml.safe_load(f)

from src.model.dharma import create_dharma_model

# ── Config ─────────────────────────────────────────────────────────────────────
N_STOCKS   = 266    # matches rebuilt dataset
L          = config["data"]["lookback_window"]      # 20
F          = config["data"]["n_features"]           # 16
N_STATES   = config["hmm"]["n_regimes"]             # 4
BATCH      = 4
DEVICE     = torch.device("cpu")                    # CPU for visualization; no GPU needed

VIZ_DIR = ROOT / "runs" / "viz"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# ── Build model ─────────────────────────────────────────────────────────────────
print("Building DHARMA model from config...")
model = create_dharma_model(config, n_stocks=N_STOCKS).to(DEVICE)
model.eval()

# Dummy inputs — exact shapes the model expects
x          = torch.randn(BATCH, L, N_STOCKS, F)          # (B, L, N, F)
R_mask     = torch.ones(N_STOCKS, N_STOCKS)              # fully-connected for viz
regime_prob= torch.softmax(torch.randn(BATCH, N_STATES), dim=-1)  # (B, K)

# ── Tracing wrapper: TorchScript requires all inputs to be Tensors ────────────
# None + int args break torch.jit.trace used by add_graph, so we wrap the
# inference path (y=None → prediction mode) with a pure-tensor interface.
class _TraceWrapper(nn.Module):
    """Thin wrapper exposing only Tensor inputs for JIT tracing."""
    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor, R_mask: torch.Tensor, regime_probs: torch.Tensor):
        # Call inference branch (y=None) with fixed n_samples=10 for tracing speed
        preds, unc = self.inner(x, R_mask, y=None, n_samples=10, regime_probs=regime_probs)
        return preds, unc

wrapped = _TraceWrapper(model)

# ── 1. TensorBoard graph + weight histograms ───────────────────────────────────
print(f"\n[1/3] Writing TensorBoard graph to {VIZ_DIR} ...")
writer = SummaryWriter(log_dir=str(VIZ_DIR))

with torch.no_grad():
    writer.add_graph(wrapped, input_to_model=(x, R_mask, regime_prob))

# Log initial weight distribution for every named parameter
for name, param in model.named_parameters():
    if param.requires_grad:
        writer.add_histogram(f"params/{name}", param.data, global_step=0)

# Log a breakdown table as text
lines = []
total_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
lines.append(f"Total trainable parameters: {total_p:,}\n")
lines.append(f"{'Module':<55} {'Shape':<30} {'Params':>10}")
lines.append("-" * 100)
for name, param in model.named_parameters():
    if param.requires_grad:
        lines.append(f"{name:<55} {str(list(param.shape)):<30} {param.numel():>10,}")
writer.add_text("architecture/parameter_table", "\n".join(lines))

writer.flush()
writer.close()
print("  TensorBoard writer closed.")

# ── 2. torchinfo summary ───────────────────────────────────────────────────────
print("\n[2/3] Generating torchinfo layer summary ...")
info = torchinfo_summary(
    wrapped,
    input_data=(x, R_mask, regime_prob),
    col_names=["input_size", "output_size", "num_params", "mult_adds"],
    col_width=25,
    depth=6,
    verbose=0,
)
summary_text = str(info)
summary_path = VIZ_DIR / "model_summary.txt"
summary_path.write_text(summary_text)
print(f"  Saved to {summary_path}")
print(summary_text)

# ── 3. Parameter count report ──────────────────────────────────────────────────
print("\n[3/3] Generating parameter breakdown ...")

def param_report(model: nn.Module) -> str:
    rows = []
    rows.append(f"\n{'Sub-module':<45} {'Trainable':>12} {'Non-trainable':>15}")
    rows.append("=" * 74)
    for name, module in model.named_modules():
        if not name:
            continue
        own_params    = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
        own_frozen    = sum(p.numel() for p in module.parameters(recurse=False) if not p.requires_grad)
        if own_params + own_frozen > 0:
            rows.append(f"  {name:<43} {own_params:>12,} {own_frozen:>15,}")
    total_train   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_frozen  = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    rows.append("=" * 74)
    rows.append(f"  {'TOTAL':<43} {total_train:>12,} {total_frozen:>15,}")
    rows.append(f"\nModel size (fp32): {total_train * 4 / 1024**2:.1f} MB")
    return "\n".join(rows)

report = param_report(model)
report_path = VIZ_DIR / "parameter_report.txt"
report_path.write_text(report)
print(report)
print(f"  Saved to {report_path}")

# ── Final instructions ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("DONE. To explore in TensorBoard:")
print(f"  tensorboard --logdir {VIZ_DIR} --port 6006")
print("  Then open: http://localhost:6006")
print()
print("  GRAPHS tab  → full computation graph (zoom into any sub-module)")
print("  HISTOGRAMS  → weight distributions per layer")
print("  TEXT        → architecture/parameter_table")
print("=" * 70)
