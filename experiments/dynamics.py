"""
dynamics.py
===========
双摆动力学核心模块。

这个文件包含所有物理公式的代码实现，是物理世界和代码之间的桥梁。

核心函数:
  derivatives(t, state, *args)  →  双摆 ODE 的右端函数 f(s)
  compute_energy(state, *args)  →  计算总能量 E = T + V
  compute_positions(state, *args)  →  计算质点坐标 (x, y) 用于可视化

公式来源：
  见 VC_obsidian/PINN-NODE-DoublePendulum-Physics.md

物理公式 → 代码的对应关系:
  ODE 右端:  ds/dt = [ω₁, ω₂, α₁, α₂]
  角加速度:  M·α = F  →  α = M⁻¹·F
  其中 M 是质量矩阵，依赖 θ₁, θ₂
       F 是力向量，依赖 θ₁, θ₂, ω₁, ω₂
"""

import numpy as np


def derivatives(t, state, m1, m2, l1, l2, g):
    """
    双摆 ODE 右端函数 f(s)。

    物理公式:
        dθ₁/dt = ω₁
        dθ₂/dt = ω₂
        M(θ₁,θ₂) · [α₁, α₂]ᵀ = F(θ₁,θ₂,ω₁,ω₂)
        α = M⁻¹ · F

    参数:
        t     : 时间 (float) — solve_ivp 自动传入，但这里的 ODE 是自治系统（不显含 t）
        state : 当前状态 [θ₁, θ₂, ω₁, ω₂] (ndarray, shape=(4,))
        m1, m2, l1, l2, g : 系统物理参数

    返回:
        dstate_dt : [ω₁, ω₂, α₁, α₂] (ndarray, shape=(4,))
    """
    θ1, θ2, ω1, ω2 = state

    # ── 角度差 ──
    # cos(θ₁-θ₂) 和 sin(θ₁-θ₂) 在耦合项中反复出现，提前计算一次
    Δθ = θ1 - θ2
    cΔ = np.cos(Δθ)
    sΔ = np.sin(Δθ)

    # ── 质量矩阵 M (2×2) ──
    # M₁₁ = (m₁+m₂)l₁         ← m₁ 和 m₂ 一起转动的惯性
    # M₁₂ = m₂l₂·cos(θ₁-θ₂)   ← 耦合项：m₂ 的摆动对 m₁ 的影响
    # M₂₁ = m₂l₁·cos(θ₁-θ₂)   ← (对称)
    # M₂₂ = m₂l₂              ← m₂ 自身的转动惯性
    M11 = (m1 + m2) * l1
    M12 = m2 * l2 * cΔ
    M21 = m2 * l1 * cΔ
    M22 = m2 * l2

    # ── 右手边向量 F (2×1) ──
    # F₁ = -m₂l₂·ω₂²·sin(θ₁-θ₂) - (m₁+m₂)g·sinθ₁
    #   ↑ 离心力项                ↑ 重力回复力
    # F₂ =  m₂l₁·ω₁²·sin(θ₁-θ₂) - m₂g·sinθ₂
    #   ↑ 离心力反作用            ↑ 重力回复力
    F1 = -m2 * l2 * ω2**2 * sΔ - (m1 + m2) * g * np.sin(θ1)
    F2 =  m2 * l1 * ω1**2 * sΔ - m2 * g * np.sin(θ2)

    # ── 求解 α = M⁻¹·F ──
    # 2×2 矩阵逆的解析形式:
    #   M⁻¹ = 1/det · [[M₂₂, -M₁₂], [-M₂₁, M₁₁]]
    # 其中 det = M₁₁·M₂₂ - M₁₂·M₂₁
    det = M11 * M22 - M12 * M21

    # 避免除零（det=0 意味着耦合奇异——两根杆瞬时共线导致信息丢失）
    # 实际积分中很少发生，如果发生就报错让 solve_ivp 处理
    if abs(det) < 1e-12:
        raise ValueError(f"质量矩阵奇异! det={det:.2e}, θ₁={θ1:.3f}, θ₂={θ2:.3f}")

    α1 = (M22 * F1 - M12 * F2) / det
    α2 = (M11 * F2 - M21 * F1) / det

    return np.array([ω1, ω2, α1, α2], dtype=float)


def controlled_derivatives(t, state, tau_func, m1, m2, l1, l2, g):
    """
    受控双摆 ODE 右端函数 f(s, τ)。

    数学公式:
        自由版本: M(θ)α = F(θ, ω)
        受控版本: M(θ)α = F(θ, ω) + τ(t)
        即原始的 F₁, F₂ 分别加上 τ₁(t), τ₂(t)

    参数:
        t        : 时间 (float) — solve_ivp 自动传入
        state    : 当前状态 [θ₁, θ₂, ω₁, ω₂] (ndarray, shape=(4,))
        tau_func : 可调用 tau_func(t) → [τ₁, τ₂]，零阶保持插值
        m1, m2, l1, l2, g : 系统物理参数

    返回:
        dstate_dt : [ω₁, ω₂, α₁, α₂] (ndarray, shape=(4,))
    """
    # 获取当前时刻的力矩
    tau = tau_func(t)
    θ1, θ2, ω1, ω2 = state

    # ── 角度差 ──
    Δθ = θ1 - θ2
    cΔ = np.cos(Δθ)
    sΔ = np.sin(Δθ)

    # ── 质量矩阵 M (2×2) ──
    M11 = (m1 + m2) * l1
    M12 = m2 * l2 * cΔ
    M21 = m2 * l1 * cΔ
    M22 = m2 * l2

    # ── 右手边向量 F (2×1) —— 加上控制力矩 τ ──
    # 与 derivatives() 的唯一区别：F1 += tau[0], F2 += tau[1]
    F1 = -m2 * l2 * ω2**2 * sΔ - (m1 + m2) * g * np.sin(θ1) + tau[0]
    F2 =  m2 * l1 * ω1**2 * sΔ - m2 * g * np.sin(θ2) + tau[1]

    # ── 求解 α = M⁻¹·F ──
    det = M11 * M22 - M12 * M21
    if abs(det) < 1e-12:
        raise ValueError(f"质量矩阵奇异! det={det:.2e}, θ₁={θ1:.3f}, θ₂={θ2:.3f}")

    α1 = (M22 * F1 - M12 * F2) / det
    α2 = (M11 * F2 - M21 * F1) / det

    return np.array([ω1, ω2, α1, α2], dtype=float)


def compute_energy(state, m1, m2, l1, l2, g):
    """
    计算系统的总能量 E = T + V。

    对于无摩擦的双摆，总能量应该是守恒的。
    这是验证数值积分精度的关键指标。

    物理公式:
        T = ½(m₁+m₂)l₁²·ω₁² + ½m₂l₂²·ω₂² + m₂l₁l₂·ω₁ω₂·cos(θ₁-θ₂)
        V = -(m₁+m₂)gl₁·cosθ₁ - m₂gl₂·cosθ₂
        E = T + V

    参数:
        state : [θ₁, θ₂, ω₁, ω₂]

    返回:
        E, T, V : 总能量, 动能, 势能 (float)
    """
    θ1, θ2, ω1, ω2 = state
    Δθ = θ1 - θ2

    # 动能 T
    # 三项: m₁+m₂ 整体转动 + m₂ 自身转动 + 交叉耦合项
    T = (0.5 * (m1 + m2) * l1**2 * ω1**2
       + 0.5 * m2 * l2**2 * ω2**2
       + m2 * l1 * l2 * ω1 * ω2 * np.cos(Δθ))

    # 势能 V
    # 重力势能，取 θ=0（垂直向下）为零势能点
    V = (-(m1 + m2) * g * l1 * np.cos(θ1)
         - m2 * g * l2 * np.cos(θ2))

    return T + V, T, V


def compute_positions(state, l1, l2):
    """
    从广义坐标 θ₁, θ₂ 计算质点的 (x, y) 坐标。

    用于:
      - 动画可视化
      - 检查运动轨迹的物理合理性

    物理公式:
        x₁ = l₁·sinθ₁,  y₁ = -l₁·cosθ₁
        x₂ = x₁ + l₂·sinθ₂
        y₂ = y₁ - l₂·cosθ₂

    参数:
        state : [θ₁, θ₂, ω₁, ω₂]
        l1, l2 : 杆长

    返回:
        (x₁, y₁), (x₂, y₂) : 两个质点的坐标
    """
    θ1, θ2 = state[0], state[1]

    x1 = l1 * np.sin(θ1)
    y1 = -l1 * np.cos(θ1)

    x2 = x1 + l2 * np.sin(θ2)
    y2 = y1 - l2 * np.cos(θ2)

    return (x1, y1), (x2, y2)
