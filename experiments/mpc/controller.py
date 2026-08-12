"""
mpc/controller.py
=================
MPC 控制器 — 滚动时域控制。

数学:
  for t in range(steps):
      τ_opt = cem.optimize(s_t, s_ref[t:t+H+1], rollout_fn)
      s_{t+1} = true_dynamics(s_t, τ_opt[0])   # 用 RK45 模拟"真实世界"
"""

import numpy as np

from mpc.rollout import rollout_rk45


def run_mpc(s0, s_ref, dt, horizon, rollout_fn, cem_optimizer, steps,
            m1, m2, l1, l2, g, tau_max):
    """滚动时域 MPC。

    参数:
        s0          : (4,) 初始状态
        s_ref       : (T, 4) 完整参考轨迹
        dt          : 时间步长
        horizon     : 预测时域 H（步数）
        rollout_fn  : 预测模型 callable(s0, tau_seq) -> (H+1, 4)
        cem_optimizer: CEM 优化器实例
        steps       : MPC 执行步数
        m1..g       : 真实世界物理参数（用 RK45 模拟）
        tau_max     : 力矩限幅

    返回:
        s_hist  : (steps+1, 4) 实际轨迹（含 s0）
        tau_hist: (steps, 2) 施加的力矩
    """
    s = s0.copy()
    s_hist = [s0.copy()]
    tau_hist = []

    for t in range(steps):
        # 参考轨迹片段: 当前到 t+H (clip 到末尾)
        end = min(t + horizon + 1, len(s_ref))
        s_ref_seg = s_ref[t:end]
        # 如果片段不足 H+1，用最后状态补齐
        if len(s_ref_seg) < horizon + 1:
            pad = np.tile(s_ref_seg[-1], (horizon + 1 - len(s_ref_seg), 1))
            s_ref_seg = np.vstack([s_ref_seg, pad])

        # 1. 求解当前最优力矩
        tau_opt = cem_optimizer.optimize(s, s_ref_seg, rollout_fn)  # (H,2)

        # 2. 只施加第一步，用"真实世界"(RK45) 推进
        tau_first = tau_opt[0].clip(-tau_max, tau_max)
        traj_step = rollout_rk45(s, tau_first.reshape(1, 2), dt)
        s = traj_step[-1]

        s_hist.append(s.copy())
        tau_hist.append(tau_first.copy())

    return np.array(s_hist), np.array(tau_hist)
