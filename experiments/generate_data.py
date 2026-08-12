"""
generate_data.py
================
双摆轨迹数据生成器。

功能:
  1. simulate():   单条轨迹的数值积分（使用 scipy.integrate.solve_ivp）
  2. generate_dataset(): 多条轨迹的批量生成（供 NODE 训练使用）

数据流:
  config.py  →  dynamics.derivatives()  →  solve_ivp  →  .npy 文件

输出数据格式:
  每条轨迹保存为一个 .npy 文件:
    shape: (T, 5)  — 时间步 × [t, θ₁, θ₂, ω₁, ω₂]
    其中 T = (t_end - t_start) / dt + 1
"""

import numpy as np
from scipy.integrate import solve_ivp
from pathlib import Path

from dynamics import derivatives, compute_energy
import config as cfg


def simulate(
    t_span=cfg.T_SPAN,
    dt=cfg.DT,
    initial_state=None,
    method=cfg.METHOD,
    rtol=cfg.RTOL,
    atol=cfg.ATOL,
    m1=cfg.M1, m2=cfg.M2,
    l1=cfg.L1, l2=cfg.L2,
    g=cfg.G,
    return_energy=False,
):
    """
    对双摆进行一次完整的数值积分。

    参数:
        initial_state : [θ₁₀, θ₂₀, ω₁₀, ω₂₀] (默认: config.py 中的默认值)
        return_energy : 是否同时返回能量序列（用于检查能量守恒）
        其余参数全部从 config.py 取默认值，也可在这里覆盖。

    返回:
        trajectory : ndarray, shape=(T, 4)
                     列为 [θ₁, θ₂, ω₁, ω₂]
        t          : ndarray, shape=(T,)
        (可选) energy : ndarray, shape=(T,)，总能量序列

    积分器说明:
      solve_ivp 是 scipy 的自适应步长积分器。
      - 它内部使用变步长（不是固定 dt），确保满足 rtol/atol 精度要求
      - t_eval 参数指定在哪些时间点输出结果（我们的 dt 控制采样率）
      - 实际积分步长由 rtol/atol 控制，比 dt 精细得多
    """
    if initial_state is None:
        initial_state = [cfg.THETA1_0, cfg.THETA2_0,
                         cfg.OMEGA1_0, cfg.OMEGA2_0]

    # 生成等间距的时间点作为输出
    # 用 linspace 避免 np.arange 的浮点舍入误差
    n_points = int((t_span[1] - t_span[0]) / dt) + 1
    t_eval = np.linspace(t_span[0], t_span[1], n_points)

    # ── solve_ivp 调用 ──
    # 参数解释:
    #   derivatives : ODE 右端函数 → 来自 dynamics.py
    #   args        : 传给 derivatives 的额外参数 (m1, m2, l1, l2, g)
    sol = solve_ivp(
        derivatives,
        t_span,
        initial_state,
        method=method,
        t_eval=t_eval,
        args=(m1, m2, l1, l2, g),
        rtol=rtol,
        atol=atol,
        dense_output=True,  # 允许在任何时间点插值
    )

    if not sol.success:
        raise RuntimeError(f"积分失败: {sol.message}")

    # 结果矩阵: (T, 4)
    trajectory = sol.y.T  # sol.y 是 (4, T), 转置为 (T, 4)

    if return_energy:
        # 用 compute_energy 计算每个时间点的总能量
        energy = np.array([
            compute_energy(s, m1, m2, l1, l2, g)[0]
            for s in trajectory
        ])
        return trajectory, sol.t, energy

    return trajectory, sol.t


def generate_dataset(
    n_trajectories=cfg.N_TRAJECTORIES,
    theta_range=cfg.THETA_RANGE,
    omega_range=cfg.OMEGA_RANGE,
    seed=42,
    save_dir="data",
    **sim_kwargs,
):
    """
    生成多条不同初始条件的轨迹并保存为 .npy 文件。

    用途:
      为 NODE 训练准备数据 — NODE 需要多条轨迹来学习动力学。

    采样策略:
      初始角度 θ₁, θ₂ 在 theta_range 内均匀采样
      初始角速度 ω₁, ω₂ 在 omega_range 内均匀采样

    参数:
        n_trajectories : 生成的轨迹数量
        theta_range    : [θ_min, θ_max] 均匀采样范围
        omega_range    : [ω_min, ω_max] 均匀采样范围
        seed           : 随机种子（确保可复现）
        save_dir       : 保存路径（相对/绝对）
        **sim_kwargs   : 传给 simulate 的其他参数

    输出:
        在 save_dir 下生成文件:
          trajectories_info.csv  — 每条轨迹的初始条件记录
          trajectory_00000.npy   — 第一条轨迹 (T, 5): [t, θ₁, θ₂, ω₁, ω₂]
          trajectory_00001.npy   — 第二条轨迹
          ...
    """
    rng = np.random.default_rng(seed)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # 记录文件
    info_lines = ["idx,theta1_0,theta2_0,omega1_0,omega2_0"]

    for i in range(n_trajectories):
        # 随机采样初始条件
        t1_0 = rng.uniform(*theta_range)
        t2_0 = rng.uniform(*theta_range)
        w1_0 = rng.uniform(*omega_range)
        w2_0 = rng.uniform(*omega_range)

        init_state = [t1_0, t2_0, w1_0, w2_0]

        # 数值积分
        traj, t = simulate(initial_state=init_state, **sim_kwargs)

        # 拼接时间列: (T, 5) = [t, θ₁, θ₂, ω₁, ω₂]
        data = np.column_stack([t, traj])

        # 保存
        filename = f"trajectory_{i:05d}.npy"
        np.save(save_path / filename, data)

        info_lines.append(f"{i},{t1_0:.6f},{t2_0:.6f},{w1_0:.6f},{w2_0:.6f}")

    # 写入记录文件
    with open(save_path / "trajectories_info.csv", "w") as f:
        f.write("\n".join(info_lines))

    print(f"已生成 {n_trajectories} 条轨迹到 {save_path.resolve()}")
    return


# ─── 命令行入口 ───────────────────────────────────────────
# python generate_data.py  → 使用 config 默认参数生成数据

if __name__ == "__main__":
    # 单条轨迹测试
    print("测试单条轨迹模拟...")
    traj, t = simulate()
    print(f"  时间点: {len(t)}")
    print(f"  轨迹 shape: {traj.shape}")
    print(f"  θ₁ 范围: [{traj[:,0].min():.3f}, {traj[:,0].max():.3f}]")
    print(f"  θ₂ 范围: [{traj[:,1].min():.3f}, {traj[:,1].max():.3f}]")
    print()

    # 检查能量守恒
    _, _, energy = simulate(return_energy=True)
    energy_drift = energy[-1] - energy[0]
    print(f"能量漂移: {energy_drift:.6e}  (应接近 0)")
    if abs(energy_drift) > cfg.ENERGY_TOLERANCE:
        print(f"  ⚠️ 警告: 能量漂移超过容限 {cfg.ENERGY_TOLERANCE:.0e}")
    else:
        print(f"  ✅ 能量守恒验证通过")
    print()

    # 批量生成
    print(f"批量生成 {cfg.N_TRAJECTORIES} 条轨迹...")
    generate_dataset()
