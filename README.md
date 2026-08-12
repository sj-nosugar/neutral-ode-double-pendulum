# 🧠 Neural ODE on a Double Pendulum

**Can a neural network discover the dynamics of a chaotic system purely from trajectory data — and can the learned model then control it?**

A complete, reproducible pipeline on the canonical chaotic system: **physics → data → Neural ODE → closed-loop MPC**, with honest baselines that say where the method is *not* worth it.

## Blog

Full write-up: **[https://varick.github.io/neural-ode-double-pendulum/](https://varick.github.io/neural-ode-double-pendulum/)**

1. From FEM to neural dynamics — a migration story (incl. why we falsified PINN before committing to data-driven)
2. Learning chaotic dynamics with NODE — curriculum learning + honest linear baseline
3. MPC with a learned model — CEM, pluggable rollout (oracle vs NODE)

## Reproduce

```bash
conda activate ml_project
cd experiments

# generate data (50 trajectories, solve_ivp)
python generate_data.py

# NODE curriculum training (chaotic error blow-up mitigation)
python -m node.train --mode controlled --seq_len 5  --epochs 300
python -m node.train --mode controlled --seq_len 10 --epochs 300 --resume node_controlled.pt
python -m node.train --mode controlled --seq_len 20 --epochs 300 --resume node_controlled.pt
python -m node.train --mode controlled --seq_len 50 --epochs 200 --resume node_controlled.pt

# evaluate + honest baseline (NODE vs linear)
python evaluate.py --steps 100 --horizon 5
python compare_linear_vs_node.py --horizon 5
python compare_linear_vs_node.py --horizon 20 --no-oracle
python compare_linear_vs_node.py --horizon 50 --no-oracle
```

## Key results (honest version)

- Short-horizon NODE ≈ linear model (and 200× slower) → **no value there**
- Long-horizon (H≥20) NODE beats linear by 17-25% → value is nonlinearity at scale
- Closed-loop MPC works with the learned model (oracle gap measured)

## Structure

```
├── docs/          # blog (MkDocs Material)
├── experiments/   # full pipeline: physics → data → NODE → MPC → baselines
└── mkdocs.yml
```

---

*Built while preparing PhD applications in AI for Science. From FEM intuition to learned dynamics — the migration story is in the blog.*
