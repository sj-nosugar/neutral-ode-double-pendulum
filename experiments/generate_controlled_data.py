"""
generate_controlled_data.py
============================
受控双摆轨迹数据生成器。

功能:
  1. make_tau_func():      随机力矩序列 + 零阶保持插值函数
  2. simulate_controlled(): 单条受控轨迹的数值积分
  3. generate_controlled_dataset(): 批量生成受控轨迹

输出格式:
  每条轨迹保存为 data/controlled/trajectory_*.npz:
    t     : ndarray (T,)       — 时间戳
    state : ndarray (T, 4)    — [θ₁, θ₂, ω₁, ω₂]
    tau   : ndarray (T, 2)    — [τ₁, τ₂] 对应每个时间步的力矩

与 generate_data.py 的区别:
  - 使用 controlled_derivatives 而非 derivatives
  - 在每个时间步施加随机力矩 τ(t) ~ U(-TAU_MAX, TAU_MAX)
  - 输出 .npz 而非 .npy（多了 tau 通道）
"""

import numpy as np
from scipy.integrate import solve_ivp
from pathlib import Path

from dynamics import controlled_derivatives
import config as cfg


def make_tau_func(t_eval, tau_max):
    """
    生成随机力矩序列及其零阶保持插值函数。

    策略:
      在 t_eval 的每个输出点上独立采样 τ₁, τ₂ ~ U(-tau_max, tau_max)。
      零阶保持插值确保在 [t_eval[i], t_eval[i+1]) 区间内 tau 恒定不变，
      这样 solve_ivp 的自适应步长在任何中间点求值都能得到正确的力矩。

    参数:
        t_eval  : 输出时间点数组 (T,)
        tau_max : 力矩范围 [-tau_max, tau_max]

    返回:
        tau_seq : ndarray (T, 2) — 每个时间点的力矩
        tau_func: callable tau(t) → ndarray (2,) — 零阶保持插值函数
    """
    T = len(t_eval)
    tau_seq = np.random.uniform(-tau_max, tau_max, size=(T, 2))

    def tau_func(t):
        """零阶保持插值：返回 t 所在区间左端点的力矩值。"""
        # 对于批量和标量 t 都支持
        t = np.asarray(t, dtype=float)
        scalar_input = t.ndim == 0
        t = np.atleast_1d(t)

        # 找到每个 t 对应的区间索引
        # searchsorted 返回插入位置，-1 后 clip 到合法范围
        idx = np.searchsorted(t_eval, t, side='right') - 1
        idx = np.clip(idx, 0, T - 1)

        result = tau_seq[idx]  # (..., 2)

        if scalar_input:
            return result[0]
        return result

    return tau_seq, tau_func


def simulate_controlled(
    t_span=cfg.T_SPAN,
    dt=cfg.DT,
    tau_max=cfg.TAU_MAX,
    initial_state=None,
    method=cfg.METHOD,
    rtol=cfg.RTOL,
    atol=cfg.ATOL,
    m1=cfg.M1, m2=cfg.M2,
    l1=cfg.L1, l2=cfg.L2,
    g=cfg.G,
):
    """
    对受控双摆进行一次完整的数值积分。

    流程:
      1. 生成输出时间点 t_eval
      2. 随机采样力矩序列 τ(t) ~ U(-tau_max, tau_max)
      3. 用 solve_ivp 积分受控 ODE

    参数:
        initial_state : [θ₁₀, θ₂₀, ω₁₀, ω₂₀]
        其余参数从 config.py 取默认值

    返回:
        state : ndarray (T, 4) — [θ₁, θ₂, ω₁, ω₂]
        t     : ndarray (T,)   — 时间戳
        tau   : ndarray (T, 2) — [τ₁, τ₂]
    """
    if initial_state is None:
        initial_state = [cfg.THETA1_0, cfg.THETA2_0,
                         cfg.OMEGA1_0, cfg.OMEGA2_0]

    # 输出时间点
    n_points = int((t_span[1] - t_span[0]) / dt) + 1
    t_eval = np.linspace(t_span[0], t_span[1], n_points)

    # 生成力矩序列和插值函数
    tau_seq, tau_func = make_tau_func(t_eval, tau_max)

    # ── solve_ivp 调用 ──
    # args 传入 tau_func（可调用对象，接收时间 t 返回当前力矩）
    sol = solve_ivp(
        controlled_derivatives,
        t_span,
        initial_state,
        method=method,
        t_eval=t_eval,
        args=(tau_func, m1, m2, l1, l2, g),
        rtol=rtol,
        atol=atol,
        dense_output=True,
    )

    if not sol.success:
        raise RuntimeError(f"受控轨迹积分失败: {sol.message}")

    # 结果矩阵: (T, 4)
    state = sol.y.T

    return state, sol.t, tau_seq


def generate_controlled_dataset(
    n_trajectories=cfg.CONTROLLED_N_TRAJECTORIES,
    tau_max=cfg.TAU_MAX,
    theta_range=cfg.THETA_RANGE,
    omega_range=cfg.OMEGA_RANGE,
    seq_len=cfg.CONTROLLED_SEQ_LEN,
    seed=42,
    save_dir=cfg.CONTROLLED_DATA_DIR,
    **sim_kwargs,
):
    """
    生成多条不同初始条件的受控轨迹并保存为 .npz 文件。

    参数:
        n_trajectories : 轨迹数量（默认 30）
        tau_max        : 力矩范围（默认 10.0）
        theta_range    : [θ_min, θ_max] 均匀采样范围
        omega_range    : [ω_min, ω_max] 均匀采样范围
        seq_len        : 每条轨迹的步数（默认 2000）
        seed           : 随机种子（确保可复现）
        save_dir       : 保存路径
        **sim_kwargs   : 传给 simulate_controlled 的其他参数

    输出:
        在 save_dir 下生成:
          info.csv              — 每条轨迹的初始条件记录
          trajectory_00000.npz  — 第一条受控轨迹
          trajectory_00001.npz  — 第二条受控轨迹
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

        # 受控数值积分
        state, t, tau = simulate_controlled(
            initial_state=init_state,
            tau_max=tau_max,
            **sim_kwargs,
        )

        # 确保步数一致
        assert len(t) == seq_len, f"轨迹 {i}: 期望 {seq_len} 步，实际 {len(t)} 步"
        assert state.shape == (seq_len, 4), f"轨迹 {i}: state shape 异常 {state.shape}"
        assert tau.shape == (seq_len, 2), f"轨迹 {i}: tau shape 异常 {tau.shape}"

        # 保存为 .npz
        filename = f"trajectory_{i:05d}.npz"
        np.savez(save_path / filename, t=t, state=state, tau=tau)

        info_lines.append(f"{i},{t1_0:.6f},{t2_0:.6f},{w1_0:.6f},{w2_0:.6f}")

    # 写入记录文件
    with open(save_path / "info.csv", "w") as f:
        f.write("\n".join(info_lines))

    print(f"已生成 {n_trajectories} 条受控轨迹到 {save_path.resolve()}")
    return


# ─── 命令行入口 ───────────────────────────────────────────
# python generate_controlled_data.py  → 生成受控轨迹

if __name__ == "__main__":
    # 单条轨迹测试
    print("测试单条受控轨迹模拟...")
    state, t, tau = simulate_controlled()
    print(f"  时间点: {len(t)}")
    print(f"  状态 shape: {state.shape}")
    print(f"  力矩 shape: {tau.shape}")
    print(f"  θ₁ 范围: [{state[:,0].min():.3f}, {state[:,0].max():.3f}]")
    print(f"  θ₂ 范围: [{state[:,1].min():.3f}, {state[:,1].max():.3f}]")
    print(f"  τ₁ 范围: [{tau[:,0].min():.3f}, {tau[:,0].max():.3f}]")
    print(f"  τ₂ 范围: [{tau[:,1].min():.3f}, {tau[:,1].max():.3f}]")
    print()

    # 批量生成受控轨迹
    print(f"批量生成 {cfg.CONTROLLED_N_TRAJECTORIES} 条受控轨迹...")
    generate_controlled_dataset()
