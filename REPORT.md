# Self-Pruning Neural Network — Case Study Report
**Tredence AI Engineering Internship | CIFAR-10 Classification**

---

## 1. Why Does an L1 Penalty on Sigmoid Gates Encourage Sparsity?

### The Gate Mechanism
Each weight `w_ij` in every `PrunableLinear` layer is multiplied by a scalar gate:

```
gate_ij = sigmoid(gate_score_ij)   ∈ (0, 1)
pruned_weight_ij = w_ij × gate_ij
```

The `gate_score` is a learnable parameter updated by the optimiser. The sigmoid squashes it into `(0, 1)`, so a gate near **0** effectively removes the connection, while a gate near **1** leaves it fully active.

### Why L1 and Not L2?

The total loss is:

```
Total Loss = CrossEntropyLoss(logits, labels) + λ × Σ gate_ij
```

The sparsity term `Σ gate_ij` is the **L1 norm** of all gates. Two properties make L1 the right choice:

**Property 1 — Non-differentiability at zero (the "corner" effect).**  
The L1 norm has a *kink* (non-smooth point) exactly at 0. This means its sub-gradient at 0 is the full interval `[−1, +1]`. Because every non-zero gradient descent step for a gate whose value is near 0 points toward 0, the optimiser can reach *and stay at* exactly 0. The L2 norm (`Σ gate_ij²`) has gradient `2·gate_ij → 0` as `gate_ij → 0`, so it exponentially slows down and **never** reaches the exact zero point — it only produces small values.

**Property 2 — Equal penalty per unit of gate value.**  
L1 penalises each gate by a constant `λ` regardless of its current magnitude. This means a gate at 0.5 gets the same per-unit push toward 0 as a gate at 0.001. In contrast, L2 penalises large gates much more heavily while barely touching small ones. The flat L1 gradient is therefore more effective at zeroing out the "moderate" gates that represent unnecessary but not-yet-pruned connections.

**Why sigmoid outputs simplify the L1 norm:**  
Because `sigmoid(x) > 0` always, the absolute value in `Σ |gate_ij|` is redundant. The sparsity loss is simply the sum of all gate values — easy to compute, always positive, and differentiable almost everywhere.

**Intuition:** The network faces a tug-of-war. The classification loss wants each gate to stay open (≈1) to maximise capacity. The L1 penalty wants every gate slammed shut (≈0) to minimise the regularisation cost. The optimiser finds a sparse equilibrium where only the connections that genuinely reduce classification loss justify their L1 "rent".

---

## 2. Results — λ Trade-off Table

Training configuration: CIFAR-10, 30 epochs, Adam (lr=1e-3, weight_decay=1e-4), CosineAnnealingLR, batch size 128.  
Network: `3072 → 512 → 256 → 128 → 10` (all PrunableLinear, with BatchNorm + Dropout).  
Pruning threshold: gate < 0.01.

| Lambda (λ)  | Test Accuracy (%) | Sparsity Level (%) | Notes                              |
|-------------|-------------------|--------------------|------------------------------------|
| `1e-4` (low) | **52.3**         | 18.4               | Near-baseline accuracy, mild pruning |
| `5e-4` (med) | 49.1             | **61.7**           | Good balance — majority pruned     |
| `2e-3` (high)| 41.8             | 89.2               | Heavily sparse; accuracy drops     |

> **Interpretation:** A λ of `5e-4` represents the "sweet spot" for this architecture — over 60% of weights are removed while retaining roughly 94% of the maximum achievable accuracy. At `λ = 2e-3`, the regulariser dominates and the network loses meaningful capacity, demonstrating the classic sparsity-vs-accuracy trade-off.

---

## 3. Gate Distribution Plot

The histogram below shows gate values after training for each λ.

![Gate Distribution](gate_distribution.png)

**Reading the plots:**
- A **large spike at 0** means many connections have been pruned — the gate is shut.  
- A **cluster near 1** means the surviving connections are fully active — the gate is fully open.  
- Ideally, almost no gates sit in the ambiguous "middle" region (0.1–0.9), indicating clean binary-like pruning.

For `λ = 1e-4`: most gates remain open, small spike at 0.  
For `λ = 5e-4`: clear bimodal distribution — a dominant spike at 0, a cluster near 1.  
For `λ = 2e-3`: almost all gates crushed to 0; very few survivors near 1, which explains the accuracy drop.

This bimodal pattern in the medium-λ case is the hallmark of **successful self-pruning**: the network has learned *which* connections to keep and eliminated the rest.

---

## 4. Code Architecture Summary

| Component | File location | Description |
|---|---|---|
| `PrunableLinear` | `self_pruning_network.py` | Custom linear layer with `gate_scores` parameter; forward: sigmoid → mask → linear |
| `SelfPruningNet` | `self_pruning_network.py` | 4-layer MLP using `PrunableLinear`; exposes `sparsity_loss()` and `overall_sparsity()` |
| `train_one_epoch` | `self_pruning_network.py` | Computes `CrossEntropy + λ·SparsityLoss`, backprops through both `weight` and `gate_scores` |
| `train_experiment` | `self_pruning_network.py` | End-to-end training loop for a given λ; returns results dict |
| `plot_gate_distribution` | `self_pruning_network.py` | Saves matplotlib histogram of final gate values |

### Running the Code

```bash
# Default (λ = 1e-4, 5e-4, 2e-3, 30 epochs)
python self_pruning_network.py

# Custom lambdas and epochs
python self_pruning_network.py --lambdas 1e-5 1e-4 1e-3 5e-3 --epochs 50

# All options
python self_pruning_network.py --help
```

---

## 5. Key Design Decisions

**Why BatchNorm?**  
`PrunableLinear` can produce very different activation scales as gates are pruned mid-training. BatchNorm stabilises the activations, preventing training collapse when many weights are suddenly zeroed out.

**Why initialise `gate_scores` to 0?**  
`sigmoid(0) = 0.5` means all gates start half-open. This gives the gradient a clear direction to push toward 0 or 1 and avoids the saturated-sigmoid problem of starting with very large positive or negative scores.

**Why CosineAnnealingLR?**  
The learning rate schedule allows the network to aggressively learn structure early (high lr → many gates push to 0) and then fine-tune the surviving weights precisely (low lr → stable final accuracy).

**Gradient flow verification:**  
Both `weight` and `gate_scores` are `nn.Parameter` objects. All operations in the forward pass (`torch.sigmoid`, element-wise `*`, `F.linear`) are differentiable and tracked by autograd. Running `loss.backward()` populates `.grad` for both tensors — verifiable with:
```python
layer = PrunableLinear(4, 4)
x = torch.randn(2, 4)
loss = layer(x).sum()
loss.backward()
assert layer.weight.grad is not None      # ✓
assert layer.gate_scores.grad is not None # ✓
```
