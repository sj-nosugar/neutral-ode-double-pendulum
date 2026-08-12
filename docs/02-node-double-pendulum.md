---
created: 2026-08-12
topic: NODE 学习混沌动力学（双摆）
---

# 02. Learning chaotic dynamics with NODE

> **The one-liner**: a neural ODE learns the double pendulum's vector field from trajectory data — and the fight against chaos's exponential error blow-up is won with curriculum learning, not architecture.

## Setup

**Physical system**: double pendulum, state $s = [\theta_1, \theta_2, \omega_1, \omega_2]$, dynamics $M(\theta)\alpha = F(\theta,\omega) + \tau$ (see `experiments/dynamics.py`).

**Neural ODE**: network learns the vector field $f_\theta(s, \tau) = ds/dt$ (6→128→128→128→4 MLP). Prediction = **integrate** from an initial state; loss = trajectory matching over H steps:

$$
\mathcal{L} = \frac{1}{H}\sum_{k=1}^{H} \| \hat{s}(t_k) - s(t_k) \|^2
$$

## The core problem: chaos amplifies error

A double pendulum has positive Lyapunov exponents — small prediction errors grow **exponentially** as we integrate. Longer horizons → exploding loss:

| Horizon H | Train loss | Val loss |
|:---------:|:----------:|:--------:|
| 5 | 0.0092 | 0.0172 |
| 20 | 0.1040 | 0.1524 |
| 50 | 0.2310 | 0.3251 |

Each doubling of H worsens loss ~10×. Diagnosis: the bottleneck is **error amplification**, not network capacity or data.

## The fix: curriculum learning

Train short-horizon first, then progressively lengthen:

```
H=5 → 10 → 20 → 50  (each stage resumes from the previous weights)
```

Why it works: the network first learns *local* dynamics well (short horizons), then extends — instead of fighting exploding gradients from the start.

## Honest baseline: vs a linear model

The whole point of a baseline is to find where your method *isn't* valuable:

| Horizon | NODE (curriculum) | Linear | NODE advantage |
|:-------:|:-----------------:|:------:|:--------------:|
| H=5 | 0.0683 | 0.0703 | none |
| H=20 | 0.1030 | 0.1367 | **25%** |
| H=50 | 0.1278 | 0.1540 | **17%** |

**Honest conclusion**: short-horizon NODE is *not* better than a linear model (and 200× slower). Its value appears at long horizons, where the chaotic nonlinearity becomes non-negligible — exactly where a linear approximation breaks down.

## Takeaway

- NODE learns nonlinear dynamics, but only when trained at the right horizon scale (curriculum)
- Value is contextual: long-horizon nonlinearity, not raw speed or short-term fit

---

*TODO: add loss-curve figure (H=5/20/50); add prediction-vs-truth trajectory plot; add Lyapunov exponent estimate.*
