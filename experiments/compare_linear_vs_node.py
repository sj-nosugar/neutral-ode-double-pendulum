"""
compare_linear_vs_node.py
=========================
反驳实验 A: 线性基线 vs NODE vs Oracle MPC。

关键变量: --horizon (预测时域 H)
  H=5  : 短时域, 混沌非线性未显现 → 预期线性≈NODE (已证实)
  H=20 : 长时域, 非线性显现 → 预期线性崩溃, NODE 撑住 = NODE 价值
  H=50 : 更长, 验证 NODE 是否真学到非线性动力学

与 evaluate.py 相同的评估协议 (同轨迹/同真实世界RK45)。
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

import config as cfg
from baselines.linear_model import fit_linear_model, make_linear_rollout
from mpc.cem import CEMOptimizer
from mpc.controller import run_mpc
from mpc.rollout import rollout_node, rollout_rk45
from node.model import NodeDynamics
from node.utils import load_trajectories, split_trajectories, compute_stats


def evaluate_traj(s0, ref, rollout_fn, steps, tau_max, cem_kwargs):
    """跑一次 MPC 评估, 返回 (rmse, energy, elapsed)。"""
    cem = CEMOptimizer(**cem_kwargs)
    t0 = time.time()
    s_hist, tau_hist = run_mpc(
        s0, ref, cfg.DT, cem.H, rollout_fn, cem, steps,
        cfg.M1, cfg.M2, cfg.L1, cfg.L2, cfg.G, tau_max)
    elapsed = time.time() - t0
    ang_err = s_hist[1:] - ref[:steps]
    rmse = np.sqrt(np.mean(ang_err[:, :2] ** 2))
    energy = np.sum(tau_hist ** 2)
    return rmse, energy, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=5, help="预测时域 H")
    parser.add_argument("--steps", type=int, default=50, help="MPC 执行步数")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--elite", type=int, default=8)
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--n_traj", type=int, default=6, help="评估轨迹数")
    parser.add_argument("--no-oracle", action="store_true", help="跳过 Oracle (省时间)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"H={args.horizon}, steps={args.steps}, samples={args.samples}, "
          f"elite={args.elite}, iters={args.iters}, n_traj={args.n_traj}")

    # ── 数据与统计量 (与训练完全一致) ──
    trajs = load_trajectories()
    train_trajs, _, test_trajs = split_trajectories(trajs)
    stats = compute_stats(train_trajs)

    # ── 三个预测器 ──
    model = NodeDynamics(input_dim=6).to(device)
    model.load_state_dict(torch.load("node_controlled.pt", map_location=device))
    ms, ss = stats["mean_state"], stats["std_state"]
    mt, st = stats["mean_tau"], stats["std_tau"]
    node_roll = lambda s, tau: rollout_node(
        s, tau, cfg.DT, model,
        lambda a, b: ((a - ms) / ss, (b - mt) / st),
        lambda x: x * ss + ms)

    A, B = fit_linear_model(train_trajs, stats)
    linear_roll = make_linear_rollout(A, B, stats)
    oracle_roll = lambda s, tau: rollout_rk45(s, tau, cfg.DT)

    base = dict(horizon=args.horizon, num_samples=args.samples,
                num_elite=args.elite, num_iters=args.iters,
                tau_max=cfg.TAU_MAX, w1=1.0, w2=0.001, seed=42)
    # Oracle 有精确模型, 少采样就够 (控制总耗时)
    oracle_kw = dict(base, num_samples=max(10, args.samples // 5),
                     num_elite=max(3, args.elite // 3),
                     num_iters=max(2, args.iters // 2))

    rollouts = {"NODE": (node_roll, base),
                "Linear": (linear_roll, base)}
    if not args.no_oracle:
        rollouts = {"Oracle": (oracle_roll, oracle_kw), **rollouts}

    print(f"{'方法':<8} {'轨迹':<4} {'RMSE(rad)':<10} {'能量':<10} {'耗时(s/步)':<10}")
    results = {k: [] for k in rollouts}
    for ti in range(args.n_traj):
        trj = test_trajs[ti]
        ref = trj["state"][:args.steps + 1]
        s0 = ref[0].copy()
        for name, (roll, kw) in rollouts.items():
            rmse, energy, elapsed = evaluate_traj(s0, ref, roll, args.steps,
                                                  cfg.TAU_MAX, kw)
            results[name].append(rmse)
            print(f"{name:<8} {ti:<4} {rmse:<10.4f} {energy:<10.1f} "
                  f"{elapsed/args.steps:<10.3f}")

    print("\n" + "=" * 60)
    print(f"汇总 (H={args.horizon}, {args.n_traj}条测试轨迹, steps={args.steps}):")
    for name in rollouts:
        r = np.array(results[name])
        print(f"  {name:<8}: RMSE {r.mean():.4f} ± {r.std():.4f}")


if __name__ == "__main__":
    main()
