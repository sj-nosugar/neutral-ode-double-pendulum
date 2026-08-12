---
created: 2026-08-12
topic: 迁移故事 — FEM 视角进入神经动力学
---

# 01. From FEM to neural dynamics — a migration story

> **The one-liner**: learning dynamics with neural networks is the same mathematical problem I was trained to solve as a physicist — modelling how a physical field evolves — with a different parameterization.

## Why this story matters

Research directions are built on analogies that survive contact with equations. My route into scientific machine learning runs through **finite elements**, not through a CS curriculum.

## The analogy: FEM ≈ learning a solution with parameters

| | FEM (what I know) | Learned dynamics (what I'm doing) |
|---|---|---|
| Problem | Solve a PDE on a domain | Model how a physical system evolves |
| Solution space | Piecewise polynomial basis | Neural network parameters |
| "Constraint" enforcement | Weak form / variational principle | Trajectory-matching loss |
| Discretization | Mesh | None (mesh-free) |
| Output | Nodal values | Continuous function / vector field |

**The bridge**: both answer "how do I parameterize a physical quantity and fit it to reality?" — FEM picks piecewise polynomials, neural ODEs pick a network. The physics intuition is the same; the parameterization changed.

## Physics intuition as an advantage

- Knowing which quantities are conserved → what the model *should* respect
- Knowing which scales dominate → how to structure training (e.g. short-horizon first on a chaotic system)
- The "no-fabrication" instinct: learned dynamics must be physically plausible, not just curve-fit

## Evidence I can point to

- 3 research projects, all on the same chain: physical system → mathematical model → simulation (LTspice / COMSOL)
- Quantum Zeno effect simulated as an electronic circuit (2022-2023) — a migration across domains, same structure
- This blog: FEM mindset applied to learning chaotic dynamics with neural ODEs

## A detour: why not PINN?

My first instinct was **physics-informed neural networks** (PINN) — constrain the network to satisfy the equation directly. It felt natural: the network *is* the solution, the residual loss *is* the physics.

But a literature check changed the plan. PINNs have **documented failure modes on chaotic systems**:

- Positive Lyapunov exponents amplify any residual error exponentially
- The optimizer finds cheap wins — **shifting the initial condition** to lower the residual without finding the true solution (Steger et al. 2022, demonstrated on the double pendulum itself)
- Fundamental imbalance between data and residual loss terms (Wang, Yu & Perdikaris 2022, NTK analysis)

So the physics-constraint route was **falsified for our problem before we spent weeks on it** — this is itself a finding: knowing where a method fails is as valuable as knowing where it works. We pivoted to the data-driven route: neural ODEs with curriculum training, and honest baselines to show where *that* works (article 02).

## Honest counterpoint

Analogies motivate; experiments justify. The rest of this blog is the experiment.

---

*TODO: add FEM→NODE analogy figure; add Zeno→circuit migration figure.*
