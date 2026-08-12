---
created: 2026-08-12
topic: 用学到的模型做 MPC 控制
---

# 03. Model predictive control with a learned model

> **The one-liner**: a learned dynamics model becomes useful when it closes the loop — MPC with a pluggable rollout model (oracle vs NODE) on the double pendulum.

## Why control matters here

Prediction accuracy is necessary but not sufficient. The real test of a learned model: can it *do* something? MPC turns "the model predicts well" into "the model enables control."

## MPC in one paragraph

At each time step $t$:

1. **Optimize** (CEM): sample $K$ candidate torque sequences, roll each out, keep the best, refine the sampling distribution
2. **Execute**: apply only the *first* torque of the best sequence
3. **Roll**: advance one step, re-measure state, repeat

Cost to minimize:

$$
J(\tau) = \sum_h \left( w_1 \|\hat{s}_h - s^{\text{ref}}_h\|^2 + w_2 \|\tau_h\|^2 \right)
$$

(w₁ tracks the reference; w₂ penalizes control effort.)

## Why CEM, not gradients

- The rollout model (a trained NN) is a black box
- Torque-sequence optimization is non-convex
- CEM is a zero-order method: sample → evaluate → elite → update distribution. No gradients needed, robust to non-smoothness.

## The pluggable rollout — the clean design

`rollout_fn` is a single interface. Two implementations:

| Rollout model | Meaning | Notes |
|---------------|---------|-------|
| `rollout_rk45` (oracle) | True equations, RK45 | "God's view" — upper bound |
| NODE model | Learned dynamics | "Mortal's view" — what we can actually use |

Same MPC code, both models → the gap between them is the cost of learning.

## Results

*TODO: add tracking RMSE and control energy comparison (oracle vs NODE vs random).*

## Connection to robotics

This is the sim-to-real spine: learned dynamics → closed-loop control. The same loop transfers to a robotic arm (and to the 9-month collaboration project).

---

*TODO: add MPC trajectory-following figure; add control-effort comparison.*
