"""
visualize.py
============
双摆结果可视化模块。

提供了三个视角来审视模拟结果:
  1. plot_trajectory()     — θ(t) 曲线：看振荡模式是否合理
  2. plot_energy()        — E(t) 曲线：验证能量守恒
  3. animate_pendulum()   — 双摆动画：直观感受运动是否物理

每个函数输出一个 matplotlib 图形或动画。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

from dynamics import compute_energy, compute_positions


def plot_trajectory(t, trajectory, save_path=None, title=None):
    """
    绘制 θ₁(t) 和 θ₂(t) 随时间的变化曲线。

    从这张图能看出:
      - 振荡是否平滑（数值积分是否稳定）
      - 是否呈现混沌行为（后期是否变得不规则）
      - θ₁ 和 θ₂ 的耦合关系

    参数:
        t          : 时间数组 (T,)
        trajectory : 状态数组 (T, 4) = [θ₁, θ₂, ω₁, ω₂]
        save_path  : 图片保存路径（None 则不保存）
        title      : 图标题（可选）
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # 上子图: 角度
    ax = axes[0]
    ax.plot(t, trajectory[:, 0], label=r"$\theta_1$", color="#3498db", linewidth=1.2)
    ax.plot(t, trajectory[:, 1], label=r"$\theta_2$", color="#e74c3c", linewidth=1.2)
    ax.set_ylabel("Angle [rad]")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title("Angles vs Time" if title is None else title)

    # 下子图: 角速度
    ax = axes[1]
    ax.plot(t, trajectory[:, 2], label=r"$\omega_1$", color="#2ecc71", linewidth=1.2)
    ax.plot(t, trajectory[:, 3], label=r"$\omega_2$", color="#f39c12", linewidth=1.2)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Angular velocity [rad/s]")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"已保存: {save_path}")

    plt.show()


def plot_energy(t, trajectory, m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81,
                save_path=None):
    """
    绘制总能量 E(t) 及其组分 T(t), V(t) 随时间的变化。

    对于保守系统（无摩擦），总能量应严格守恒。
    能量漂移是数值误差的体现，也是评判积分精度的核心指标。

    参数:
        t          : 时间数组 (T,)
        trajectory : 状态数组 (T, 4)
        物理参数默认使用标准值
    """
    # 逐点计算能量
    E_arr = np.zeros(len(t))
    T_arr = np.zeros(len(t))
    V_arr = np.zeros(len(t))

    for i, state in enumerate(trajectory):
        E_arr[i], T_arr[i], V_arr[i] = compute_energy(state, m1, m2, l1, l2, g)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # 上子图: 能量组分
    ax = axes[0]
    ax.plot(t, T_arr, label="Kinetic $T$", color="#2ecc71", linewidth=1.2)
    ax.plot(t, V_arr, label="Potential $V$", color="#e74c3c", linewidth=1.2)
    ax.plot(t, E_arr, label="Total $E = T+V$", color="#2c3e50", linewidth=2)
    ax.set_ylabel("Energy [J]")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title("Energy vs Time")

    # 下子图: 能量漂移（相对于 t=0 的偏差）
    ax = axes[1]
    delta_E = E_arr - E_arr[0]
    ax.plot(t, delta_E, color="#8e44ad", linewidth=1.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("$\\Delta E$ [J]")
    ax.set_title(f"Energy Drift: max $|\\Delta E|$ = {np.max(np.abs(delta_E)):.2e}")
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"已保存: {save_path}")

    plt.show()


def animate_pendulum(t, trajectory, l1=1.0, l2=1.0,
                     interval=20, save_path=None):
    """
    双摆运动的动画可视化。

    绘制两个质点的运动轨迹，直观展示双摆的摆动是否合理。
    这是最直接的"物理合理性检查"。

    参数:
        t          : 时间数组 (T,)
        trajectory : 状态数组 (T, 4)
        l1, l2     : 杆长
        interval   : 帧间隔 [ms]
        save_path  : 动画保存路径（.gif 或 .mp4，None 则不保存）

    返回:
        anim : matplotlib.animation.FuncAnimation 对象
    """
    # 预先计算所有帧的质点位置
    n_frames = len(t)
    positions = np.zeros((n_frames, 4))
    for i, state in enumerate(trajectory):
        (x1, y1), (x2, y2) = compute_positions(state, l1, l2)
        positions[i] = [x1, y1, x2, y2]

    # 确定坐标范围（留点余量）
    margin = 0.3
    all_x = positions[:, [0, 2]].flatten()
    all_y = positions[:, [1, 3]].flatten()
    x_min, x_max = all_x.min() - margin, all_x.max() + margin
    y_min, y_max = all_y.min() - margin, all_y.max() + margin
    # 保证 x/y 比例一致（正方形）
    half_range = max(x_max - x_min, y_max - y_min) / 2
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(x_center - half_range, x_center + half_range)
    ax.set_ylim(y_center - half_range, y_center + half_range)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title("Double Pendulum")

    # 图形元素: 杆和质点
    line, = ax.plot([], [], "o-", lw=2, markersize=8,
                    markerfacecolor="red", markeredgecolor="darkred")
    # 轨迹尾迹（质点2的历史轨迹）
    trail_len = min(200, n_frames)
    trail, = ax.plot([], [], "-", color="orange", alpha=0.4, lw=1)

    def init():
        line.set_data([], [])
        trail.set_data([], [])
        return line, trail

    def update(frame):
        # 当前帧的位置
        x1, y1, x2, y2 = positions[frame]
        line.set_data([0, x1, x2], [0, y1, y2])

        # 尾迹: 最近 trail_len 帧的 m₂ 轨迹
        start = max(0, frame - trail_len)
        trail.set_data(positions[start:frame+1, 2],
                       positions[start:frame+1, 3])
        return line, trail

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, init_func=init,
        interval=interval, blit=True
    )

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        if save_path.endswith(".gif"):
            anim.save(save_path, writer="pillow", fps=1000 // interval)
        else:
            anim.save(save_path, writer="ffmpeg", fps=1000 // interval)
        print(f"已保存: {save_path}")

    plt.show()
    return anim
