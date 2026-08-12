"""
node/model.py
=============
NodeDynamics 网络 — 学习受控双摆的 vector field。

学的是 ds/dt = f_θ([s, τ])，其中:
  s  = [θ₁, θ₂, ω₁, ω₂]     (4 维)
  τ  = [τ₁, τ₂]              (2 维)
  输出 = [dθ₁, dθ₂, dω₁, dω₂] (4 维)

forward 签名必须接收 t (torchdiffeq 要求)，即使不用。
"""

import torch
import torch.nn as nn


class NodeDynamics(nn.Module):
    """6→128→128→128→4 的 MLP，学习 ds/dt = f_θ(s, τ)。"""

    def __init__(self, input_dim=6, hidden_dim=128, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, t, state_and_tau):
        """返回 ds/dt。

        参数:
            t            : 时间 (float 或 tensor) — torchdiffeq 传入，本模型不用
            state_and_tau: 拼接后的 [s, τ]，shape (batch, 6) 或 (6,)

        返回:
            ds/dt, shape (batch, 4) 或 (4,)
        """
        return self.net(state_and_tau)
