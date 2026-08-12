"""
node/train.py
=============
NODE 训练循环。

两种模式:
  free       — 无控制输入，模型 4D→4D，odeint 积分 (方案里的热身)
  controlled — 带力矩输入，模型 6D→4D，已知 tau 序列逐段步进

核心: 从 s(t₀) 积分 H 步 → 预测 ŝ(t₁..t_H) → 与真实轨迹 MSE。

用法:
  conda activate ml_project
  python -m node.train --mode free --epochs 200
  python -m node.train --mode controlled --epochs 300
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint

from node.model import NodeDynamics
from node.utils import (
    load_trajectories, split_trajectories, compute_stats, sample_batch,
)

HERE = Path(__file__).resolve().parent.parent


def integrate_free(model, s0, times):
    """自由模式: 用 odeint 在时间网格上积分。

    s0: (B, 4), times: (H+1,) → (H+1, B, 4)
    """
    return odeint(model, s0, times, method="rk4")


def integrate_controlled(model, s0, tau_seq, dt):
    """受控模式: RK4 逐段步进，每步用已知的 tau_k。

    为什么不用 odeint: 输入 τ(t) 是时变的，odeint 内部自适应步长无法
    在每个求值点正确取到 τ。数据里 τ 每步随机（零阶保持），RK4 中间点
    用同一个 τ_k 是精确的（因为 τ 在 [t_k, t_{k+1}) 内恒定）。

    s0: (B,4), tau_seq: (B,H,2), dt: float → (H+1, B, 4)
    """
    states = [s0]
    x = s0
    for k in range(tau_seq.shape[1]):
        tau_k = tau_seq[:, k]                      # (B,2)

        def f(xx):
            x_in = torch.cat([xx, tau_k], dim=-1)  # (B,6)
            return model(None, x_in)

        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        states.append(x)
    return torch.stack(states)  # (H+1, B, 4)


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    trajs = load_trajectories()
    train_trajs, val_trajs, test_trajs = split_trajectories(trajs)
    stats = compute_stats(train_trajs)
    print(f"trajectories: {len(train_trajs)}/{len(val_trajs)}/{len(test_trajs)} "
          f"(train/val/test)")
    print(f"state mean: {stats['mean_state']}, std: {stats['std_state']}")

    input_dim = 4 if args.mode == "free" else 6
    model = NodeDynamics(input_dim=input_dim, hidden_dim=args.hidden,
                         output_dim=4).to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"已加载权重: {args.resume}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    dt = 0.01  # 与数据采样率一致

    rng = np.random.RandomState(args.seed)
    best_val = float("inf")

    for epoch in range(args.epochs):
        model.train()
        s0, tau_seq, targets = sample_batch(train_trajs, stats, args.batch_size,
                                           args.seq_len, rng)
        s0, tau_seq, targets = s0.to(device), tau_seq.to(device), targets.to(device)

        optimizer.zero_grad()
        if args.mode == "free":
            # free 模式: 只用 state，tau 序列忽略
            times = torch.linspace(0.0, args.seq_len * dt, args.seq_len + 1,
                                   device=device)
            pred = integrate_free(model, s0, times)           # (H+1,B,4)
            pred = pred[1:]                                    # 去掉 t0
        else:
            pred = integrate_controlled(model, s0, tau_seq, dt)
            pred = pred[1:]

        loss = loss_fn(pred, targets.transpose(0, 1))
        loss.backward()
        optimizer.step()

        if (epoch + 1) % args.log_every == 0:
            val_loss = evaluate(model, val_trajs, stats, args, device)
            print(f"epoch {epoch+1:4d}  train={loss.item():.6f}  "
                  f"val={val_loss:.6f}")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), HERE / f"node_{args.mode}.pt")
                print(f"  → saved best model (val {best_val:.6f})")


def evaluate(model, val_trajs, stats, args, device):
    """在验证集上跑一个 batch 的 MSE。"""
    model.eval()
    rng = np.random.RandomState(0)
    s0, tau_seq, targets = sample_batch(val_trajs, stats, args.batch_size,
                                        args.seq_len, rng)
    s0, tau_seq, targets = s0.to(device), tau_seq.to(device), targets.to(device)
    dt = 0.01
    with torch.no_grad():
        if args.mode == "free":
            times = torch.linspace(0.0, args.seq_len * dt, args.seq_len + 1,
                                   device=device)
            pred = integrate_free(model, s0, times)[1:]
        else:
            pred = integrate_controlled(model, s0, tau_seq, dt)[1:]
        return nn.MSELoss()(pred, targets.transpose(0, 1)).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["free", "controlled"], required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--resume", type=str, default=None,
                        help="从已有 checkpoint 继续训练 (如 node_controlled.pt)")
    args = parser.parse_args()

    t0 = time.time()
    train(args)
    print(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
