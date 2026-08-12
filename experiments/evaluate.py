"""
evaluate.py
===========
Day 4: 评估三种 MPC 方法。

| 方法        | 预测模型          | 说明                        |
|------------|------------------|-----------------------------|
| Oracle MPC | rollout_rk45()   | 已知真实 ODE，理论最优        |
| NODE MPC   | rollout_node()   | 学到的 NODE                  |
| Random     | 随机 τ           | 最低基准线                   |

指标:
  - 轨迹跟踪 RMSE: sqrt(1/T · Σ ||s_t − s_ref_t||²)
  - 控制能量: Σ ||τ_t||²
  - 每步 MPC 耗时

用法:
  python evaluate.py [--steps 200] [--horizon 10]
"""

import argparse
import time

import numpy as np
import torch

import config as cfg
from dynamics import derivatives
from mpc.cem import CEMOptimizer
from mpc.controller import run_mpc
from mpc.rollout import rollout_node, rollout_rk45
from node.model import NodeDynamics
from node.utils import load_trajectories, split_trajectories, compute_stats


def make_node_rollout(model, stats, device):
    """返回 rollout_node 的闭包（内置标准化/去标准化）。"""
    ms, ss = stats["mean_state"], stats["std_state"]
    mt, st = stats["mean_tau"], stats["std_tau"]

    def normalize(s, tau):
        return (s - ms) / ss, (tau - mt) / st

    def denormalize(s_norm):
        return s_norm * ss + ms

    return lambda s0, tau_seq: rollout_node(
        s0, tau_seq, cfg.DT, model, normalize, denormalize)


def evaluate_method(name, s0, s_ref, rollout_fn, cem, steps, tau_max):
    t0 = time.time()
    s_hist, tau_hist = run_mpc(
        s0, s_ref, cfg.DT, cem.H, rollout_fn, cem, steps,
        cfg.M1, cfg.M2, cfg.L1, cfg.L2, cfg.G, tau_max)
    elapsed = time.time() - t0

    # 跟踪 RMSE（只算角度，位置量纲 rad）
    ang_err = s_hist[1:] - s_ref[:steps]
    rmse = np.sqrt(np.mean(ang_err[:, :2] ** 2))
    energy = np.sum(tau_hist ** 2)
    per_step = elapsed / steps

    print(f"\n=== {name} ===")
    print(f"  RMSE (rad):    {rmse:.4f}")
    print(f"  控制能量:      {energy:.2f}")
    print(f"  每步耗时:      {per_step*1000:.1f} ms")
    return {"rmse": rmse, "energy": energy, "per_step_ms": per_step * 1000,
            "s_hist": s_hist, "tau_hist": tau_hist}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200, help="MPC 执行步数")
    parser.add_argument("--horizon", type=int, default=5, help="预测时域")
    parser.add_argument("--samples", type=int, default=50, help="CEM 采样数")
    parser.add_argument("--elite", type=int, default=8)
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 参考轨迹: 用一条受控测试轨迹的真实状态作为"要跟踪的目标" ──
    trajs = load_trajectories()
    _, _, test_trajs = split_trajectories(trajs)
    ref_traj = test_trajs[0]["state"][:args.steps + 1]   # (steps+1, 4)
    s0 = ref_traj[0].copy()

    # ── 载入 NODE 模型 ──
    stats = compute_stats(load_trajectories()[:18])
    model = NodeDynamics(input_dim=6).to(device)
    model.load_state_dict(torch.load("node_controlled.pt", map_location=device))
    node_rollout = make_node_rollout(model, stats, device)

    # ── CEM 参数 (w2=0.001 是调参甜点: 精度几乎无损, 能量省 71%) ──
    cem_kwargs = dict(horizon=args.horizon, num_samples=args.samples,
                      num_elite=args.elite, num_iters=args.iters,
                      tau_max=cfg.TAU_MAX, w1=1.0, w2=0.001, seed=args.seed)

    # ── 三种方法 ──
    # Oracle: 用真实 ODE 预测
    cem_oracle = CEMOptimizer(**cem_kwargs)
    oracle = evaluate_method("Oracle MPC (真实ODE)",
                             s0, ref_traj,
                             lambda s, tau: rollout_rk45(s, tau, cfg.DT),
                             cem_oracle, args.steps, cfg.TAU_MAX)

    # NODE: 用学到的模型预测
    cem_node = CEMOptimizer(**cem_kwargs)
    node = evaluate_method("NODE MPC",
                           s0, ref_traj, node_rollout,
                           cem_node, args.steps, cfg.TAU_MAX)

    # Random: 随机力矩基线
    rng = np.random.RandomState(args.seed)
    s = s0.copy()
    s_hist = [s0.copy()]
    tau_hist = []
    for t in range(args.steps):
        tau = rng.uniform(-cfg.TAU_MAX, cfg.TAU_MAX, 2)
        traj_step = rollout_rk45(s, tau.reshape(1, 2), cfg.DT)
        s = traj_step[-1]
        s_hist.append(s.copy())
        tau_hist.append(tau.copy())
    s_hist = np.array(s_hist)
    ang_err = s_hist[1:] - ref_traj[:args.steps]
    rmse = np.sqrt(np.mean(ang_err[:, :2] ** 2))
    energy = np.sum(np.array(tau_hist) ** 2)
    print(f"\n=== Random (随机力矩) ===")
    print(f"  RMSE (rad):    {rmse:.4f}")
    print(f"  控制能量:      {energy:.2f}")
    print(f"  (无优化，仅基准)")

    # ── 汇总 ──
    print("\n" + "=" * 40)
    print(f"汇总 (steps={args.steps}, H={args.horizon}):")
    print(f"  Oracle RMSE: {oracle['rmse']:.4f} rad")
    print(f"  NODE   RMSE: {node['rmse']:.4f} rad")
    print(f"  Random RMSE: {rmse:.4f} rad")

    # ── 可视化 ──
    plot_results(ref_traj, oracle, node, s_hist, np.array(tau_hist), args.steps)


def plot_results(ref_traj, oracle, node, random_hist, random_tau, steps):
    """画出三种方法 vs 参考轨迹的跟踪对比 + 力矩。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(steps + 1) * cfg.DT
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # θ1 跟踪
    ax = axes[0, 0]
    ax.plot(t, ref_traj[:steps + 1, 0], "k--", lw=2, label="reference")
    ax.plot(t, oracle["s_hist"][:steps + 1, 0], "b-", lw=1.2, label="Oracle")
    ax.plot(t, node["s_hist"][:steps + 1, 0], "r-", lw=1.2, label="NODE")
    ax.plot(t, random_hist[:steps + 1, 0], "g-", lw=0.8, alpha=0.7, label="Random")
    ax.set_title("θ₁ tracking (rad)"); ax.set_xlabel("t (s)"); ax.legend()

    # θ2 跟踪
    ax = axes[0, 1]
    ax.plot(t, ref_traj[:steps + 1, 1], "k--", lw=2, label="reference")
    ax.plot(t, oracle["s_hist"][:steps + 1, 1], "b-", lw=1.2, label="Oracle")
    ax.plot(t, node["s_hist"][:steps + 1, 1], "r-", lw=1.2, label="NODE")
    ax.plot(t, random_hist[:steps + 1, 1], "g-", lw=0.8, alpha=0.7, label="Random")
    ax.set_title("θ₂ tracking (rad)"); ax.set_xlabel("t (s)"); ax.legend()

    # 角速度 ω1
    ax = axes[1, 0]
    ax.plot(t, ref_traj[:steps + 1, 2], "k--", lw=2, label="reference")
    ax.plot(t, oracle["s_hist"][:steps + 1, 2], "b-", lw=1.2, label="Oracle")
    ax.plot(t, node["s_hist"][:steps + 1, 2], "r-", lw=1.2, label="NODE")
    ax.set_title("ω₁ (rad/s)"); ax.set_xlabel("t (s)"); ax.legend()

    # 力矩 τ1 (NODE vs Oracle)
    ax = axes[1, 1]
    ax.plot(t[1:], oracle["tau_hist"][:, 0], "b-", lw=1, label="Oracle τ₁")
    ax.plot(t[1:], node["tau_hist"][:, 0], "r-", lw=1, label="NODE τ₁")
    ax.set_title("τ₁ control effort (N·m)"); ax.set_xlabel("t (s)"); ax.legend()

    plt.tight_layout()
    out = "mpc_results.png"
    plt.savefig(out, dpi=120)
    print(f"\n图已保存: {out}")


if __name__ == "__main__":
    main()
