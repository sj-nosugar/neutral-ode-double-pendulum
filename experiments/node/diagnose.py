"""
node/diagnose.py
================
Day 2 诊断脚本：找出 loss 不降的原因。

实验设计:
  1. 对比不同 H (5 / 20 / 50) 的 train/val loss 曲线
     → H=5 明显更好 ⇒ 混沌放大是主因；H=5 也一样卡 ⇒ 主因不在序列长度
  2. 记录每 epoch 梯度范数 → 检测网络是否死亡（Tanh 饱和 / 梯度消失）
  3. 输出 loss 曲线到 node/diag_*.png + 打印汇总

用法:
  python -m node.diagnose
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint

from node.model import NodeDynamics
from node.utils import (
    load_trajectories, split_trajectories, compute_stats, sample_batch,
)

dt = 0.01


def integrate_controlled(model, s0, tau_seq):
    """RK4 步进，tau 零阶保持。返回 (H+1, B, 4)。"""
    states = [s0]
    x = s0
    for k in range(tau_seq.shape[1]):
        tau_k = tau_seq[:, k]

        def f(xx):
            return model(None, torch.cat([xx, tau_k], dim=-1))

        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        states.append(x)
    return torch.stack(states)


def run_experiment(seq_len, epochs, batch_size, lr, seed, device):
    """跑一个 H 配置，返回 {train_losses, val_losses, grad_norms}。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    trajs = load_trajectories()
    train_trajs, val_trajs, _ = split_trajectories(trajs)
    stats = compute_stats(train_trajs)

    model = NodeDynamics(input_dim=6).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_losses, val_losses, grad_norms = [], [], []

    for epoch in range(epochs):
        model.train()
        s0, tau_seq, targets = sample_batch(train_trajs, stats, batch_size,
                                            seq_len, rng)
        s0, tau_seq, targets = (s0.to(device), tau_seq.to(device),
                                targets.to(device))

        optimizer.zero_grad()
        pred = integrate_controlled(model, s0, tau_seq)[1:]
        loss = loss_fn(pred, targets.transpose(0, 1))
        loss.backward()
        grad_norm = torch.cat([p.grad.flatten() for p in model.parameters()
                               if p.grad is not None]).norm().item()
        optimizer.step()

        train_losses.append(loss.item())
        grad_norms.append(grad_norm)

        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                s0v, tau_v, targetv = sample_batch(val_trajs, stats,
                                                   batch_size, seq_len,
                                                   np.random.RandomState(0))
                s0v, tau_v, targetv = (s0v.to(device), tau_v.to(device),
                                       targetv.to(device))
                predv = integrate_controlled(model, s0v, tau_v)[1:]
                val_losses.append(loss_fn(predv, targetv.transpose(0, 1)).item())
            model.train()

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "grad_norms": grad_norms,
        "final_train": train_losses[-1],
        "final_val": val_losses[-1] if val_losses else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_lens", type=int, nargs="+", default=[5, 20, 50])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}\n")

    results = {}
    for sl in args.seq_lens:
        print(f"── H={sl} ──")
        r = run_experiment(sl, args.epochs, args.batch_size, args.lr,
                           args.seed, device)
        results[sl] = r
        print(f"  final train={r['final_train']:.4f}  val={r['final_val']:.4f}")
        print(f"  grad norm: 首={r['grad_norms'][0]:.2f} "
              f"末={r['grad_norms'][-1]:.2f} "
              f"min={min(r['grad_norms']):.4f}")

    # ── 画图 ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for sl, r in results.items():
        x = np.arange(1, len(r["train_losses"]) + 1)
        axes[0].plot(x, r["train_losses"], label=f"H={sl}")
        axes[1].plot(range(20, args.epochs + 1, 20), r["val_losses"],
                     label=f"H={sl}", marker="o")
        axes[2].plot(x, r["grad_norms"], label=f"H={sl}")

    axes[0].set_title("Train Loss"); axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("MSE"); axes[0].legend()
    axes[1].set_title("Val Loss (every 20 epochs)")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("MSE"); axes[1].legend()
    axes[2].set_title("Gradient Norm"); axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("‖∇θ‖"); axes[2].legend()
    axes[0].set_yscale("log"); axes[1].set_yscale("log")
    axes[2].set_yscale("log")

    plt.tight_layout()
    out = "node/diag_loss_curves.png"
    plt.savefig(out, dpi=120)
    print(f"\n图已保存: {out}")


if __name__ == "__main__":
    main()
