# Self-Pruning Neural Network — Case Study Report
**Tredence AI Engineering Internship | CIFAR-10 Classification**

---

## 1. Why Does an L1 Penalty on Gates Encourage Sparsity?

### The Gate Mechanism
Each weight `w_ij` in every `PrunableLinear` layer is multiplied by a scalar gate:

```
gate_ij       = clamp(gate_score_ij, 0, 1)   ∈ [0, 1]
pruned_weight = w_ij × gate_ij
output        = x @ pruned_weight.T + bias
```

The `gate_score` is a learnable parameter updated by the optimiser. Unlike sigmoid, `clamp` can reach **exactly 0** — once a gate_score goes negative, the gate is hard-zeroed immediately.

### Why L1 and Not L2?

The total loss is:

```
Total Loss = CrossEntropyLoss(logits, labels) + λ × mean(gate_ij)
```

The sparsity term `mean(gate_ij)` is the **L1 norm** of all gates (normalised by count). Two properties make L1 the right choice:

**Property 1 — Constant gradient.**
L1 applies a constant push of magnitude λ toward 0 for every gate, regardless of its current value. Even gates already near 0 keep getting pushed — unlike L2 whose gradient `2·gate_ij → 0` as `gate_ij → 0`, meaning it exponentially slows down and never achieves exact zeros.

**Property 2 — Exact zeros via clamp.**
Once the L1 gradient pushes a `gate_score` below 0, `clamp` locks it at exactly 0. Sigmoid can never do this — it asymptotically approaches 0 but never reaches it in finite training steps. This is why previous sigmoid-based versions showed 0% sparsity even with large λ values.

**Intuition:** The network faces a tug-of-war. The classification loss wants each gate open (=1) to maximise capacity. The L1 penalty wants every gate shut (=0) to minimise regularisation cost. The optimiser finds a sparse equilibrium where only the connections that genuinely reduce classification loss justify staying open.

---

## 2. Results — λ Trade-off Table

Training configuration: CIFAR-10, **15 epochs**, Adam (lr=1e-3 weights, lr=5e-2 gates), CosineAnnealingLR, batch size 128.
Network: `3072 → 512 → 256 → 128 → 10` (all PrunableLinear, with BatchNorm + Dropout).
Pruning threshold: gate < 0.01.

| Lambda (λ) | Test Accuracy (%) | Sparsity Level (%) | Notes |
|------------|------------------|-------------------|-------|
| `1.0` (low) | **53.83** | 41.9 | Mild pruning, best accuracy |
| `5.0` (med) | 53.67 | 52.5 | Good balance — majority of weights pruned |
| `20.0` (high) | 53.24 | **68.1** | Heavily sparse, only 0.59% accuracy drop |

> **Key finding:** Even at 68.1% sparsity (λ=20), accuracy only drops by 0.59% compared to the lowest-sparsity run. The network successfully identifies and retains only its most important connections while discarding the majority as redundant.

---

## 3. Gate Distribution Plot

The histogram below shows gate values after training for each λ.

![Gate Distribution](gate_distribution-2.png)

**Reading the plots:**
- A **large spike at 0** means many connections have been pruned — the gate is hard-zeroed.
- A **cluster at 1** means the surviving connections are fully active — the gate is fully open.
- Almost nothing sits in the middle region — indicating clean binary-like pruning.

For `λ = 1.0`: ~42% of gates hard-zeroed, remaining cluster tightly at 1.0.
For `λ = 5.0`: spike at 0 grows larger, active cluster at 1 shrinks proportionally.
For `λ = 20.0`: dominant spike at 0, small surviving cluster at 1 — accuracy barely affected.

The perfectly bimodal pattern across all three runs is the hallmark of **successful self-pruning**: the network has learned which connections to keep and eliminated the rest.

Note: Clamp is used instead of sigmoid to allow exact zero pruning.
---

## 4. Code Architecture Summary

| Component | Description |
|---|---|
| `PrunableLinear` | Custom linear layer with `gate_scores` parameter; forward: `clamp(gate_scores,0,1)` → mask → `F.linear` |
| `SelfPruningNet` | 4-layer MLP using `PrunableLinear`; exposes `sparsity_loss()` and `overall_sparsity()` |
| `train_one_epoch` | Computes `CrossEntropy + λ·mean(gates)`, backprops through both `weight` and `gate_scores` |
| `train_experiment` | End-to-end loop with separate Adam param groups — gate LR = 50× weight LR |
| `plot_gate_distribution` | Saves matplotlib histogram of final gate values per λ |

### Running the Code

```bash
# Default (λ = 1.0, 5.0, 20.0 — 15 epochs)
python self_pruning_network_v4.py

# Custom lambdas and epochs
python self_pruning_network_v4.py --lambdas 0.5 1.0 5.0 20.0 --epochs 30

# All options
python self_pruning_network_v4.py --help
```

---

## 5. Key Design Decisions

**Clamp gates over sigmoid:**
Sigmoid asymptotically approaches 0 but never reaches it — `clamp` gates hard-zero the moment a score goes negative, producing true sparsity measurable at a threshold of 0.01.

**Separate LR for gate_scores (50× higher):**
Gate scores need to move fast enough to cross zero within 15 epochs. Weights need a conservative LR to learn stably. A single shared LR cannot satisfy both simultaneously — separate Adam parameter groups solve this cleanly.

**Normalised sparsity loss (mean, not sum):**
Using `mean(gates)` instead of `sum(gates)` makes λ scale-independent of network size. The same λ values produce comparable sparsity regardless of how many parameters the network has.

**Why BatchNorm?**
`PrunableLinear` can produce very different activation scales as gates are pruned mid-training. BatchNorm stabilises activations, preventing training collapse when many weights are suddenly zeroed out.

**Gradient flow verification:**
Both `weight` and `gate_scores` are `nn.Parameter` objects. `clamp()` is differentiable almost everywhere and tracked by autograd. Running `loss.backward()` populates `.grad` for both tensors — verifiable with:

```python
layer = PrunableLinear(4, 4)
x = torch.randn(2, 4)
loss = layer(x).sum()
loss.backward()
assert layer.weight.grad is not None      # ✓
assert layer.gate_scores.grad is not None # ✓
```
## Key Idea

Each weight is multiplied by a learnable gate:
gate = clamp(gate_score, 0, 1)

**Note:** Clamp gating was used instead of sigmoid to enable exact zero pruning and measurable sparsity.
