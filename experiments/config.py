"""
config.py
=========
双摆模拟的参数配置。

物理参数对应公式中的 m₁, m₂, l₁, l₂, g。
初始条件对应 t=0 时系统的状态 [θ₁₀, θ₂₀, ω₁₀, ω₂₀]。
数值参数控制积分精度和采样密度。
"""

import numpy as np

# ─── 物理参数 ─────────────────────────────────────────────
# m₁, m₂ : 两个质点的质量 [kg]
# l₁, l₂ : 两段无质量硬杆的长度 [m]
# g       : 重力加速度 [m/s²]

M1 = 1.0
M2 = 1.0
L1 = 1.0
L2 = 1.0
G  = 9.81

# ─── 默认初始条件 ─────────────────────────────────────────
# 选取 θ₁=π/2, θ₂=π/2, 从水平释放（这是典型的混沌测试初始条件）
# 可用于多轨迹数据生成的 seed 范围

THETA1_0 = np.pi / 2.0   # 初始角度 θ₁ [rad]
THETA2_0 = np.pi / 2.0   # 初始角度 θ₂ [rad]
OMEGA1_0 = 0.0            # 初始角速度 ω₁ [rad/s]
OMEGA2_0 = 0.0            # 初始角速度 ω₂ [rad/s]

# ─── 数值积分参数 ─────────────────────────────────────────
# t_span  : 积分时间区间 [t_start, t_end] [s]
# dt      : 输出采样步长 [s]（不影响积分精度，只控制输出点的密度）
# method  : scipy.integrate.solve_ivp 使用的积分方法
#           RK45  : 4(5)阶 Runge-Kutta（默认，精度/速度平衡好）
#           DOP853 : 8阶方法（更高精度，更慢）
#           Radau  : 隐式方法（适合刚性方程，双摆不需要）
# rtol, atol : 相对/绝对误差容限（越小越精确，越慢）

T_SPAN = [0.0, 20.0]     # 模拟 20 秒
DT     = 0.01             # 100 Hz 采样率 → 20s × 100 = 2000 个时间点
METHOD = "RK45"
RTOL   = 1e-9
ATOL   = 1e-12

# ─── 数据生成参数 ─────────────────────────────────────────
# N_TRAJECTORIES : 生成多少条不同初始条件的轨迹（供 NODE 训练用）
# THETA_RANGE    : 初始角度的随机范围 [rad]
# OMEGA_RANGE    : 初始角速度的随机范围 [rad/s]

N_TRAJECTORIES = 50
THETA_RANGE  = [-np.pi, np.pi]    # 全范围角度
OMEGA_RANGE  = [-3.0, 3.0]        # 合理的角速度范围

# ─── 物理常数 ─────────────────────────────────────────────
# 用于能量检查的收敛阈值

ENERGY_TOLERANCE = 1e-6  # 能量漂移超过此值则警告

# ─── 受控动力学参数 ───────────────────────────────────────
# TAU_MAX                  : 控制力矩采样范围 [-TAU_MAX, TAU_MAX] [N·m]
# CONTROLLED_N_TRAJECTORIES: 受控轨迹数量
# CONTROLLED_SEQ_LEN       : 每条轨迹的步数
# CONTROLLED_DATA_DIR      : 受控数据保存目录

TAU_MAX = 10.0
CONTROLLED_N_TRAJECTORIES = 30
CONTROLLED_SEQ_LEN = 2001
CONTROLLED_DATA_DIR = "data/controlled"
