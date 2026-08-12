"""
baselines/linear_model.py
=========================
线性基线: ds/dt ≈ A·s + B·τ (最小二乘拟合)

与 NODE 完全对等:
  - 同一份训练数据
  - 同样的标准化
  - 同样的 RK4 积分方式
  - 唯一区别: 动力学是线性函数而非神经网络

用途:
  回答"H=5 短时窗口内, 线性模型是否已经足够?"
  若 linear ≈ NODE → NODE 没学到非线性, 工作定位需改变
  若 NODE 明显优于 linear → NODE 学到非线性的证据
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 项目根目录

import numpy as np
import torch

import config as cfg
from node.utils import load_trajectories, split_trajectories, compute_stats


def fit_linear_model(trajs, stats, dt=cfg.DT, ridge=1e-6):
    """拟合 ds/dt ≈ A·s + B·τ (标准化空间)。

    数据准备:
      对每条轨迹的相邻点对 (s_k, s_{k+1}) 用有限差分估计 ds/dt:
        ds/dt_k ≈ (s_{k+1} - s_k) / dt
      标准化后组装 X = [s_norm, tau_norm], y = ds/dt_norm

    参数:
        trajs: 训练轨迹列表
        stats: compute_stats 输出
        dt   : 采样步长
        ridge: 正则化系数 (防止奇异)

    返回:
        A: (4,4), B: (4,2) — 标准化空间的线性动力学
    """
    ms, ss = stats["mean_state"], stats["std_state"]
    mt, st = stats["mean_tau"], stats["std_tau"]

    X_parts, y_parts = [], []
    for tr in trajs:
        s, tau = tr["state"], tr["tau"]          # (T,4), (T,2)
        ds = np.diff(s, axis=0) / dt             # (T-1,4) 有限差分
        # 标准化
        s_norm = (s[:-1] - ms) / ss
        tau_norm = (tau[:-1] - mt) / st
        ds_norm = ds / ss                        # 导数同尺度标准化
        X_parts.append(np.hstack([s_norm, tau_norm]))  # (T-1, 6)
        y_parts.append(ds_norm)                        # (T-1, 4)

    X = np.vstack(X_parts)
    y = np.vstack(y_parts)

    # 最小二乘: y ≈ X @ W, 加 ridge 稳定
    # W = (X^T X + λI)^-1 X^T y
    XtX = X.T @ X + ridge * np.eye(X.shape[1])
    W = np.linalg.solve(XtX, X.T @ y)            # (6, 4)
    A, B = W[:4].T, W[4:].T                       # (4,4), (4,2) — 注意转置
    return A, B


def rollout_linear(s0, tau_seq, dt, A, B, normalize_fn=None,
                   denormalize_fn=None):
    """用线性模型 RK4 步进, 与 rollout_node 完全同构。

    参数:
        s0     : (4,) 物理空间
        tau_seq: (H,2) 物理空间
        A, B   : 标准化空间线性系数
        normalize_fn / denormalize_fn: 同 rollout_node

    返回:
        traj: (H+1, 4) 物理空间
    """
    if normalize_fn is not None:
        s0, tau_seq = normalize_fn(s0, tau_seq)

    x = s0.astype(np.float64)
    traj = [denormalize_fn(x) if denormalize_fn else x.copy()]

    def f(xx, tau_k):
        return A @ xx + B @ tau_k

    for k in range(len(tau_seq)):
        tau_k = tau_seq[k]
        k1 = f(x, tau_k)
        k2 = f(x + 0.5 * dt * k1, tau_k)
        k3 = f(x + 0.5 * dt * k2, tau_k)
        k4 = f(x + dt * k3, tau_k)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        traj.append(denormalize_fn(x) if denormalize_fn else x.copy())

    return np.array(traj)


def make_linear_rollout(A, B, stats):
    """返回 rollout_linear 的闭包 (内置标准化/去标准化), 与 make_node_rollout 同构。"""
    ms, ss = stats["mean_state"], stats["std_state"]
    mt, st = stats["mean_tau"], stats["std_tau"]

    def normalize(s, tau):
        return (s - ms) / ss, (tau - mt) / st

    def denormalize(s_norm):
        return s_norm * ss + ms

    return lambda s0, tau_seq: rollout_linear(
        s0, tau_seq, cfg.DT, A, B, normalize, denormalize)


if __name__ == "__main__":
    # 冒烟测试: 拟合 + 单次 rollout
    trajs = load_trajectories()
    train_trajs, _, _ = split_trajectories(trajs)
    stats = compute_stats(train_trajs)
    A, B = fit_linear_model(train_trajs, stats)
    print(f"拟合完成: A shape={A.shape}, B shape={B.shape}")

    roll = make_linear_rollout(A, B, stats)
    s0 = train_trajs[0]["state"][0]
    tau_seq = train_trajs[0]["tau"][:5]
    traj = roll(s0, tau_seq)
    print(f"rollout: {traj.shape}, 首行={traj[0][:2]}")
