"""
mpc/cem.py
==========
CEM (Cross-Entropy Method) 优化器 — 用于求解最优控制序列。

数学:
  for iter in range(N_iter):
      1. 采样 K 个候选 τ 序列: τ ~ N(μ, σ²)
      2. 对每个候选 rollout → 计算 cost J
      3. 选择 cost 最小的 P 个 elite
      4. μ = mean(elite), σ = std(elite)
  返回最优 τ 序列的第一项

Cost 函数:
  J(τ) = Σ_h ( w1·‖ŝ_h − s_ref_h‖² + w2·‖τ_h‖² )
"""

import numpy as np


class CEMOptimizer:
    def __init__(self, horizon, num_samples, num_elite, num_iters,
                 tau_max, w1=1.0, w2=0.01, seed=42):
        self.H = horizon            # 预测时域步数
        self.K = num_samples        # 每轮采样的候选数
        self.P = num_elite          # elite 数量
        self.N_iter = num_iters     # CEM 迭代轮数
        self.tau_max = tau_max
        self.w1 = w1                # 跟踪权重
        self.w2 = w2                # 控制能量惩罚
        self.rng = np.random.RandomState(seed)

    def optimize(self, s0, s_ref, rollout_fn):
        """求解最优力矩序列。

        参数:
            s0       : (4,) 当前状态
            s_ref    : (H+1, 4) 参考轨迹（含当前点）
            rollout_fn: callable(s0, tau_seq) -> (H+1, 4) 预测轨迹

        返回:
            tau_opt: (H, 2) 最优力矩序列
        """
        H = self.H
        mu = np.zeros((H, 2))
        sigma = np.full((H, 2), self.tau_max)

        for _ in range(self.N_iter):
            # 1. 采样候选: (K, H, 2)
            candidates = self.rng.normal(mu, sigma, size=(self.K, self.H, 2))
            candidates = candidates.clip(-self.tau_max, self.tau_max)

            # 2. 计算 cost
            costs = np.zeros(self.K)
            for i in range(self.K):
                traj = rollout_fn(s0, candidates[i])          # (H+1, 4)
                track_err = np.mean((traj[1:] - s_ref[1:]) ** 2)   # 跟踪误差
                energy = np.mean(candidates[i] ** 2)               # 控制能量
                costs[i] = self.w1 * track_err + self.w2 * energy

            # 3. 选 elite
            elite_idx = np.argsort(costs)[:self.P]
            elite = candidates[elite_idx]

            # 4. 更新分布
            mu = elite.mean(axis=0)
            sigma = elite.std(axis=0) + 1e-6

        return mu
