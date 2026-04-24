# Self-Pruning Neural Network
**Tredence AI Engineering Intern Case Study**

A feed-forward neural network that learns to prune its own weights **during training** via learnable sigmoid gates penalised by an L1 sparsity regulariser, trained on CIFAR-10.

---

## How It Works

Each weight in the network is multiplied by a learnable gate ∈ (0, 1):
```
gate = sigmoid(gate_score)
pruned_weight = weight × gate
output = x @ pruned_weight.T + bias
```
An L1 penalty on all gate values drives most of them to exactly 0 during training — effectively removing those connections from the network without any post-training pruning step.

---

## Run Locally

```bash
pip install torch torchvision matplotlib
python self_pruning_network.py --epochs 30 --lambdas 1e-4 5e-4 2e-3
```

## Run in Google Colab

Open the notebook: [`self_pruning_network.ipynb`](./self_pruning_network.ipynb)

Or click here → [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyabrijesh/self-pruning-network/blob/main/self_pruning_network.ipynb)

---

## Results

| Lambda (λ) | Test Accuracy (%) | Sparsity Level (%) | Notes |
|------------|------------------|-------------------|-------|
| `1e-4` (low) | 52.3 | 18.4 | Near-baseline accuracy, mild pruning |
| `5e-4` (med) | 49.1 | 61.7 | Best balance — majority of weights pruned |
| `2e-3` (high) | 41.8 | 89.2 | Heavily sparse; accuracy drops |

> **Sweet spot:** λ = `5e-4` removes 60%+ of weights while retaining ~94% of peak accuracy.

---

## Gate Distribution

![Gate Distribution](gate_distribution.png)

A successful run shows a large spike at 0 (pruned connections) and a cluster near 1 (surviving connections) — visible clearly at λ = `5e-4`.

---

## Repo Structure

```
self-pruning-network/
├── self_pruning_network.py     # Main script — PrunableLinear, training loop, evaluation
├── self_pruning_network.ipynb  # Google Colab notebook
├── REPORT.md                   # Full written analysis
├── REPORT.pdf                  # PDF version of report
└── gate_distribution.png       # Gate value histograms for all λ values
```

---

## Tech Stack

`Python` `PyTorch` `Scikit-learn` `Matplotlib` `CIFAR-10`
