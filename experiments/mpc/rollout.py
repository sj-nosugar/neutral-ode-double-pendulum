"""
mpc/rollout.py
==============
轨迹预测：Oracle (RK45) vs NODE。

两种预测模型:
  rollout_rk45  — 用真实 ODE (dynamics.py) 积分 → Oracle 基准
  rollout_node  — 用训练好的 NODE 积分 → 你的方法
"""

import numpy as np
import torch
from scipy.integrate import solve_ivp

from dynamics import controlled_derivatives
import config as cfg


def rollout_rk45(s0, tau_seq, dt):
    """用真实 ODE (RK45) 从 s0 出发，施加 tau_seq 的力矩序列。

    参数:
        s0     : (4,) 初始状态 [θ1, θ2, ω1, ω2]
        tau_seq: (H, 2) 力矩序列
        dt     : 时间步长

    返回:
        traj: (H+1, 4) 轨迹（含 s0）
    """
    T = len(tau_seq)
    t_eval = np.arange(T + 1) * dt

    # 零阶保持力矩函数（tau 在每步内恒定）
    def tau_func(t):
        t = np.asarray(t, dtype=float)
        scalar = t.ndim == 0
        t = np.atleast_1d(t)
        idx = np.clip(np.searchsorted(t_eval, t, side="right") - 1, 0, T - 1)
        result = tau_seq[idx]
        return result[0] if scalar else result

    sol = solve_ivp(
        lambda t, s: controlled_derivatives(t, s, tau_func, cfg.M1, cfg.M2,
                                            cfg.L1, cfg.L2, cfg.G),
        [0, T * dt], s0, t_eval=t_eval, method="RK45",
        rtol=cfg.RTOL, atol=cfg.ATOL,
    )
    return sol.y.T  # (H+1, 4)


def rollout_node(s0, tau_seq, dt, node_model, normalize_fn=None, denormalize_fn=None):
    """用训练好的 NODE 从 s0 出发，施加 tau_seq 的力矩序列 (RK4 步进)。

    参数:
        s0      : (4,) 初始状态（物理空间）
        tau_seq : (H, 2) 力矩序列（物理空间）
        dt      : 时间步长
        node_model: NodeDynamics 网络
        normalize_fn: 可选的标准化函数 (state, tau) -> (s_norm, tau_norm)
        denormalize_fn: 可选的去标准化函数 s_norm -> s_phys

    返回:
        traj: (H+1, 4) 轨迹（物理空间）
    """
    device = next(node_model.parameters()).device
    node_model.eval()

    H = len(tau_seq)
    if normalize_fn is not None:
        s0, tau_seq = normalize_fn(s0, tau_seq)

    x = torch.tensor(s0, dtype=torch.float32, device=device).unsqueeze(0)  # (1,4)
    traj = [x.detach().cpu().numpy()[0]]

    with torch.no_grad():
        for k in range(H):
            tau_k = torch.tensor(tau_seq[k], dtype=torch.float32,
                                 device=device).unsqueeze(0)  # (1,2)

            def f(xx):
                return node_model(None, torch.cat([xx, tau_k], dim=-1))

            k1 = f(x)
            k2 = f(x + 0.5 * dt * k1)
            k3 = f(x + 0.5 * dt * k2)
            k4 = f(x + dt * k3)
            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            traj.append(x.detach().cpu().numpy()[0])

    traj = np.array(traj)  # (H+1, 4)
    if denormalize_fn is not None:
        traj = denormalize_fn(traj)
    return traj
