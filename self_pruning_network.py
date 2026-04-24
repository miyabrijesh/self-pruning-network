"""
Self-Pruning Neural Network — Tredence AI Engineering Intern Case Study
=======================================================================
A feed-forward network that learns to prune its own weights *during training*
via learnable sigmoid gates penalised by an L1 sparsity regulariser.

Run:
    python self_pruning_network.py          # trains with 3 lambda values
    python self_pruning_network.py --help   # see all CLI flags

Requirements:
    pip install torch torchvision matplotlib
"""

import argparse
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ──────────────────────────────────────────────────────────────────────────────
# Part 1: PrunableLinear Layer
# ──────────────────────────────────────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that multiplies every weight by a
    learnable gate in [0, 1].  During training the L1 penalty on the gate
    values drives many of them toward exactly 0, effectively removing the
    corresponding weights from the network.

    Forward pass (per neuron connection):
        gate        = sigmoid(gate_score)          # ∈ (0, 1)
        pruned_w    = weight * gate                # element-wise
        out         = x @ pruned_w.T + bias        # standard linear op

    Gradients flow through *both* weight and gate_score via autograd
    because all operations (sigmoid, multiply, matmul) are differentiable.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # --- Standard weight + bias (same init as nn.Linear) ----------------
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # --- Gate scores: same shape as weight, initialised near 0 ----------
        # Initialising near 0 means sigmoid(0) ≈ 0.5 → gates start half-open,
        # giving the optimiser room to push them toward 0 or 1.
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        self._reset_parameters()

    def _reset_parameters(self):
        # Kaiming uniform (matches PyTorch's nn.Linear default)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1 — convert scores → gates ∈ (0, 1)
        gates = torch.sigmoid(self.gate_scores)

        # Step 2 — mask the weights
        pruned_weights = self.weight * gates

        # Step 3 — standard linear transformation
        return F.linear(x, pruned_weights, self.bias)

    @torch.no_grad()
    def sparsity(self, threshold: float = 1e-2) -> float:
        """Fraction of connections whose gate is below `threshold`."""
        gates = torch.sigmoid(self.gate_scores)
        return (gates < threshold).float().mean().item()

    @torch.no_grad()
    def gate_values(self) -> torch.Tensor:
        """Return a flat tensor of all gate values (for plotting)."""
        return torch.sigmoid(self.gate_scores).flatten().cpu()

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.bias is not None}")


# ──────────────────────────────────────────────────────────────────────────────
# Network definition
# ──────────────────────────────────────────────────────────────────────────────

class SelfPruningNet(nn.Module):
    """
    A three-hidden-layer feed-forward network for CIFAR-10 classification.
    All linear layers are replaced with PrunableLinear so every weight has
    an associated learnable gate.

    Architecture:
        Input (3072)  →  512  →  256  →  128  →  10 classes
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            PrunableLinear(3 * 32 * 32, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            PrunableLinear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            PrunableLinear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            PrunableLinear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))   # flatten spatial dims

    def prunable_layers(self) -> list[PrunableLinear]:
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]

    def sparsity_loss(self) -> torch.Tensor:
        """
        L1 norm of ALL gate values across every PrunableLinear layer.

        Why L1?  The L1 norm is known to induce *exact* sparsity (zeros)
        rather than merely small values.  Its sub-gradient at 0 is the
        interval [−1, +1], which lets the optimiser push gate values all
        the way to 0 and keep them there.  The L2 norm (squared values)
        only pushes toward 0 asymptotically and never achieves exact zeros.

        Because our gates are sigmoid outputs → always positive, the L1
        norm simplifies to the plain sum: Σ sigmoid(gate_score_i).
        """
        return sum(
            torch.sigmoid(layer.gate_scores).sum()
            for layer in self.prunable_layers()
        )

    @torch.no_grad()
    def overall_sparsity(self, threshold: float = 1e-2) -> float:
        """Network-wide sparsity: fraction of pruned connections."""
        dead = sum(
            (torch.sigmoid(l.gate_scores) < threshold).sum().item()
            for l in self.prunable_layers()
        )
        total = sum(
            l.gate_scores.numel() for l in self.prunable_layers()
        )
        return dead / total if total > 0 else 0.0

    @torch.no_grad()
    def all_gate_values(self) -> torch.Tensor:
        """Concatenate gate values from all layers for histogram plotting."""
        return torch.cat([l.gate_values() for l in self.prunable_layers()])


# ──────────────────────────────────────────────────────────────────────────────
# Part 3: Training & Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def get_cifar10_loaders(batch_size: int = 128, data_dir: str = "./data"):
    """Return train and test DataLoaders for CIFAR-10."""
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.CIFAR10(data_dir, train=True,  download=True, transform=train_tf)
    test_ds  = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=256,        shuffle=False,
                              num_workers=2, pin_memory=True)
    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, device, lam: float) -> tuple[float, float]:
    """Train for one epoch; return (avg_total_loss, avg_cls_loss)."""
    model.train()
    total_loss_sum = cls_loss_sum = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        logits = model(x)

        # ── Part 2: Total Loss = ClassificationLoss + λ * SparsityLoss ──────
        cls_loss      = F.cross_entropy(logits, y)
        sparsity_loss = model.sparsity_loss()
        total_loss    = cls_loss + lam * sparsity_loss

        total_loss.backward()
        optimizer.step()

        total_loss_sum += total_loss.item()
        cls_loss_sum   += cls_loss.item()

    n = len(loader)
    return total_loss_sum / n, cls_loss_sum / n


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    """Return top-1 accuracy on the given DataLoader."""
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total   += y.size(0)
    return correct / total


def train_experiment(
    lam:        float,
    epochs:     int,
    lr:         float,
    batch_size: int,
    device:     torch.device,
    train_loader,
    test_loader,
    verbose:    bool = True,
) -> dict:
    """
    Train a SelfPruningNet with sparsity coefficient `lam`.
    Returns a dict with results suitable for the final table.
    """
    model = SelfPruningNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if verbose:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n{'='*60}")
        print(f"  λ = {lam}  |  {total_params:,} parameters  |  {epochs} epochs")
        print(f"{'='*60}")

    best_acc = 0.0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        total_loss, cls_loss = train_one_epoch(
            model, train_loader, optimizer, device, lam
        )
        scheduler.step()
        acc = evaluate(model, test_loader, device)
        sparsity = model.overall_sparsity()
        best_acc = max(best_acc, acc)

        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == 1):
            elapsed = time.time() - t0
            print(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"loss {total_loss:.4f} (cls {cls_loss:.4f}) | "
                f"acc {acc*100:.2f}% | sparsity {sparsity*100:.1f}% | "
                f"{elapsed:.0f}s"
            )

    final_acc      = evaluate(model, test_loader, device)
    final_sparsity = model.overall_sparsity()
    gate_vals      = model.all_gate_values()

    if verbose:
        print(f"\n  ✓ Final  →  acc {final_acc*100:.2f}%  |  "
              f"sparsity {final_sparsity*100:.1f}%")

    return {
        "lambda":   lam,
        "accuracy": final_acc,
        "sparsity": final_sparsity,
        "gate_vals": gate_vals,
        "model":    model,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Plotting utilities
# ──────────────────────────────────────────────────────────────────────────────

def plot_gate_distribution(results: list[dict], save_path: str = "gate_distribution.png"):
    """
    Plot histogram of final gate values for each λ.
    A successful run shows a sharp spike at 0 and a cluster near 1.
    """
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 4), sharey=False)
    if len(results) == 1:
        axes = [axes]

    for ax, r in zip(axes, results):
        gates = r["gate_vals"].numpy()
        ax.hist(gates, bins=80, range=(0, 1), color="#4C72B0", edgecolor="none", alpha=0.85)
        ax.axvline(x=0.01, color="red", linestyle="--", linewidth=1.2,
                   label="prune threshold (0.01)")
        ax.set_title(
            f"λ = {r['lambda']}\n"
            f"Acc: {r['accuracy']*100:.2f}%  |  "
            f"Sparsity: {r['sparsity']*100:.1f}%",
            fontsize=11,
        )
        ax.set_xlabel("Gate value (sigmoid output)", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle("Distribution of Learnable Gate Values After Training", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Gate distribution plot saved to: {save_path}")
    return fig


def print_results_table(results: list[dict]):
    """Pretty-print the λ comparison table."""
    print("\n" + "="*55)
    print(f"  {'Lambda':>10}  {'Test Acc (%)':>14}  {'Sparsity (%)':>14}")
    print("="*55)
    for r in results:
        print(
            f"  {r['lambda']:>10.4f}  "
            f"{r['accuracy']*100:>14.2f}  "
            f"{r['sparsity']*100:>14.1f}"
        )
    print("="*55)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Self-Pruning Neural Network — Tredence Case Study")
    parser.add_argument("--epochs",     type=int,   default=30,
                        help="Training epochs per lambda (default: 30)")
    parser.add_argument("--lr",         type=float, default=1e-3,
                        help="Initial learning rate (default: 1e-3)")
    parser.add_argument("--batch-size", type=int,   default=128,
                        help="Training batch size (default: 128)")
    parser.add_argument("--lambdas",    type=float, nargs="+",
                        default=[1e-4, 5e-4, 2e-3],
                        help="Sparsity regularisation coefficients (default: 1e-4 5e-4 2e-3)")
    parser.add_argument("--data-dir",   type=str,   default="./data",
                        help="Directory for CIFAR-10 download (default: ./data)")
    parser.add_argument("--plot-path",  type=str,   default="gate_distribution.png",
                        help="Output path for gate histogram plot")
    parser.add_argument("--no-cuda",    action="store_true",
                        help="Disable CUDA even if available")
    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    print(f"\nDevice: {device}")

    train_loader, test_loader = get_cifar10_loaders(args.batch_size, args.data_dir)

    results = []
    for lam in args.lambdas:
        r = train_experiment(
            lam=lam,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=device,
            train_loader=train_loader,
            test_loader=test_loader,
        )
        results.append(r)

    print_results_table(results)
    plot_gate_distribution(results, save_path=args.plot_path)

    print("\nDone! See gate_distribution.png and the table above for analysis.\n")


if __name__ == "__main__":
    main()
