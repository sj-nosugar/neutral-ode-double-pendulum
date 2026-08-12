---
created: 2026-08-12
topic: Neural ODE 双摆项目首页
---

# Neural ODE on a Double Pendulum

> **Thesis**: Can a neural network *discover* the dynamics of a chaotic system purely from trajectory data — and can the learned model then *control* it? We answer both on the double pendulum, the canonical chaotic system, with honest baselines.

A complete pipeline: **physics → data → learned dynamics (Neural ODE) → closed-loop control (MPC)** — and a hard look at where the method is actually worth it.

## What's here

1. **[From FEM to neural dynamics — a migration story](01-migration-story.md)** — why a physicist trained on finite elements sees learned dynamics as the same problem with a different parameterization.
2. **[Learning chaotic dynamics with NODE](02-node-double-pendulum.md)** — neural ODEs on the double pendulum: curriculum learning against error blow-up, and an honest comparison with a linear baseline.
3. **[Model predictive control with a learned model](03-mpc.md)** — closing the loop: CEM-based MPC with a pluggable rollout model (oracle vs learned).

### Why not PINN? (a one-paragraph footnote)

Physics-informed neural networks (PINNs) constrain a network to satisfy the equation directly — elegant, but **known to fail on chaotic systems**: the positive Lyapunov exponents make any residual error explode, and the optimizer finds cheap "cheats" (e.g. shifting the initial condition) instead of the true solution (Steger et al. 2022; Wang et al. 2022). We verified this before choosing the data-driven route — see the [migration story](01-migration-story.md).

## Code

All experiments live in [`experiments/`](https://github.com/varick/neural-ode-double-pendulum/tree/main/experiments) — a complete pipeline from physical equations to data, learned dynamics, and closed-loop control:

```
experiments/
├── dynamics.py            # true physics: M·α = F
├── generate_data.py       # numerical integration → trajectories
├── node/                  # Neural ODE training (model/train/diagnose)
├── mpc/                   # CEM-based model predictive control
└── baselines/             # linear model baseline
```

Reproduce everything with:

```bash
conda activate ml_project
cd experiments
# NODE curriculum training
python -m node.train --mode controlled --seq_len 5  --epochs 300
python -m node.train --mode controlled --seq_len 10 --epochs 300 --resume node_controlled.pt
# ... and so on, see article 02
```
