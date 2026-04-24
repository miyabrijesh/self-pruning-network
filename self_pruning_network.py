"""
Self-Pruning Neural Network — v4
================================
Root fix: Sigmoid gates asymptotically approach 0 but never reach it in 15 epochs.

New approach:
- Use ReLU-clipped gates: gate = clamp(gate_score, 0, 1)
  These CAN reach exactly 0 because ReLU has a true zero region.
- gate_scores init to 1.0 (fully open), L1 pushes them negative → clamped to 0
- Very high lambdas: 1, 5, 20
- Gate LR = 50x weight LR so they move fast enough
"""

import argparse, math, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.weight      = nn.Parameter(torch.empty(out_features, in_features))
        # Init to 1.0: gates fully open. L1 penalty pushes scores negative → gate clamps to 0
        self.gate_scores = nn.Parameter(torch.ones(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.uniform_(self.bias, -1/math.sqrt(in_features), 1/math.sqrt(in_features))

    def forward(self, x):
        # clamp(0,1) gates: reach exactly 0 when score goes negative
        gates          = torch.clamp(self.gate_scores, 0.0, 1.0)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

    @torch.no_grad()
    def sparsity(self, threshold=1e-2):
        gates = torch.clamp(self.gate_scores, 0.0, 1.0)
        return (gates < threshold).float().mean().item()

    @torch.no_grad()
    def gate_values(self):
        return torch.clamp(self.gate_scores, 0.0, 1.0).flatten().cpu()


class SelfPruningNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            PrunableLinear(3*32*32, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            PrunableLinear(512, 256),     nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            PrunableLinear(256, 128),     nn.BatchNorm1d(128), nn.ReLU(),
            PrunableLinear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

    def prunable_layers(self):
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]

    def sparsity_loss(self):
        # Mean of clamped gate values — in [0,1], L1 drives toward 0
        all_gates = torch.cat([
            torch.clamp(l.gate_scores, 0.0, 1.0).flatten()
            for l in self.prunable_layers()
        ])
        return all_gates.mean()

    @torch.no_grad()
    def overall_sparsity(self, threshold=1e-2):
        dead  = sum((torch.clamp(l.gate_scores,0,1) < threshold).sum().item()
                    for l in self.prunable_layers())
        total = sum(l.gate_scores.numel() for l in self.prunable_layers())
        return dead / total

    @torch.no_grad()
    def all_gate_values(self):
        return torch.cat([l.gate_values() for l in self.prunable_layers()])


def get_cifar10_loaders(batch_size=128, data_dir="./data"):
    mean, std = (0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010)
    train_tf = transforms.Compose([transforms.RandomCrop(32,padding=4),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(), transforms.Normalize(mean,std)])
    test_tf  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean,std)])
    tr = datasets.CIFAR10(data_dir, train=True,  download=True, transform=train_tf)
    te = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)
    return (DataLoader(tr, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True),
            DataLoader(te, batch_size=256,        shuffle=False, num_workers=2, pin_memory=True))


def train_one_epoch(model, loader, optimizer, device, lam):
    model.train()
    tl = cl = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        c = F.cross_entropy(logits, y)
        s = model.sparsity_loss()
        loss = c + lam * s
        loss.backward()
        optimizer.step()
        tl += loss.item(); cl += c.item()
    n = len(loader)
    return tl/n, cl/n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total   += y.size(0)
    return correct / total


def train_experiment(lam, epochs, lr, device, train_loader, test_loader, verbose=True):
    model = SelfPruningNet().to(device)

    gate_params   = [p for n,p in model.named_parameters() if 'gate_scores' in n]
    weight_params = [p for n,p in model.named_parameters() if 'gate_scores' not in n]

    optimizer = torch.optim.Adam([
        {'params': weight_params, 'lr': lr},
        {'params': gate_params,   'lr': lr * 50},  # gates move 50x faster
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if verbose:
        print(f"\n{'='*60}\n  λ = {lam}  |  {sum(p.numel() for p in model.parameters()):,} params  |  {epochs} epochs\n{'='*60}")

    t0 = time.time()
    for epoch in range(1, epochs+1):
        tl, cl = train_one_epoch(model, train_loader, optimizer, device, lam)
        scheduler.step()
        acc = evaluate(model, test_loader, device)
        sp  = model.overall_sparsity()
        if verbose:
            print(f"  Epoch {epoch:3d}/{epochs} | loss {tl:.4f} (cls {cl:.4f}) "
                  f"| acc {acc*100:.2f}% | sparsity {sp*100:.1f}% | {time.time()-t0:.0f}s")

    acc = evaluate(model, test_loader, device)
    sp  = model.overall_sparsity()
    if verbose:
        print(f"\n  ✓ Final  →  acc {acc*100:.2f}%  |  sparsity {sp*100:.1f}%")
    return {"lambda": lam, "accuracy": acc, "sparsity": sp,
            "gate_vals": model.all_gate_values()}


def plot_gate_distribution(results, save_path="gate_distribution.png"):
    fig, axes = plt.subplots(1, len(results), figsize=(6*len(results), 4))
    if len(results) == 1: axes = [axes]
    for ax, r in zip(axes, results):
        ax.hist(r["gate_vals"].numpy(), bins=80, range=(0,1),
                color="#4C72B0", edgecolor="none", alpha=0.85)
        ax.axvline(0.01, color="red", linestyle="--", lw=1.2, label="prune threshold")
        ax.set_title(f"λ = {r['lambda']}\nAcc: {r['accuracy']*100:.2f}%  |  "
                     f"Sparsity: {r['sparsity']*100:.1f}%", fontsize=11)
        ax.set_xlabel("Gate value", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle("Distribution of Learnable Gate Values After Training", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved → {save_path}")


def print_table(results):
    print("\n" + "="*55)
    print(f"  {'Lambda':>10}  {'Test Acc (%)':>14}  {'Sparsity (%)':>14}")
    print("="*55)
    for r in results:
        print(f"  {r['lambda']:>10.1f}  {r['accuracy']*100:>14.2f}  {r['sparsity']*100:>14.1f}")
    print("="*55)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=15)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int,   default=128)
    parser.add_argument("--lambdas",    type=float, nargs="+", default=[1.0, 5.0, 20.0])
    parser.add_argument("--data-dir",   type=str,   default="./data")
    parser.add_argument("--plot-path",  type=str,   default="gate_distribution.png")
    parser.add_argument("--no-cuda",    action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"\nDevice: {device}")

    train_loader, test_loader = get_cifar10_loaders(args.batch_size, args.data_dir)
    results = [train_experiment(lam, args.epochs, args.lr, device, train_loader, test_loader)
               for lam in args.lambdas]
    print_table(results)
    plot_gate_distribution(results, args.plot_path)
    print("\nDone!\n")


if __name__ == "__main__":
    main()
