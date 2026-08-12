"""
node/utils.py
=============
数据加载、切分、标准化、batch 采样。

数据来源: data/controlled/trajectory_*.npz
  每条轨迹: {t: (T,), state: (T, 4), tau: (T, 2)}
"""

from pathlib import Path

import numpy as np
import torch

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "controlled"


def load_trajectories(data_dir=DATA_DIR):
    """加载全部受控轨迹。

    返回:
        trajs: list of dict {t, state, tau}
    """
    trajs = []
    for f in sorted(data_dir.glob("trajectory_*.npz")):
        d = np.load(f)
        trajs.append({"t": d["t"], "state": d["state"], "tau": d["tau"]})
    return trajs


def split_trajectories(trajs, train_frac=0.6, val_frac=0.2):
    """按轨迹切分 train/val/test (不打乱轨迹内部时间顺序)。

    30 条 → 18/6/6 (默认 0.6/0.2/0.2)。
    """
    n = len(trajs)
    idx = np.random.RandomState(42).permutation(n)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return (
        [trajs[i] for i in idx[:n_train]],
        [trajs[i] for i in idx[n_train:n_train + n_val]],
        [trajs[i] for i in idx[n_train + n_val:]],
    )


def compute_stats(trajs):
    """计算 state 与 tau 的均值和标准差 (用于标准化)。

    返回:
        stats: dict {mean_state, std_state, mean_tau, std_tau} (numpy arrays)
    """
    all_state = np.concatenate([tr["state"] for tr in trajs], axis=0)
    all_tau = np.concatenate([tr["tau"] for tr in trajs], axis=0)
    return {
        "mean_state": all_state.mean(axis=0),
        "std_state": all_state.std(axis=0) + 1e-8,
        "mean_tau": all_tau.mean(axis=0),
        "std_tau": all_tau.std(axis=0) + 1e-8,
    }


def normalize_traj(traj, stats):
    """标准化一条轨迹的 state 和 tau。

    注意: 时间 t 不标准化。
    """
    return {
        "t": traj["t"],
        "state": (traj["state"] - stats["mean_state"]) / stats["std_state"],
        "tau": (traj["tau"] - stats["mean_tau"]) / stats["std_tau"],
    }


def sample_batch(trajs, stats, batch_size, seq_len, rng):
    """采样一个 batch: 随机轨迹 + 随机起始点 + 长度 seq_len 的片段。

    每条样本:
        s0_state  : (4,)  — [θ₁, θ₂, ω₁, ω₂] 标准化后 (t₀ 时刻)
        tau_seq   : (seq_len, 2) — 从 t₀ 起每个时间步的力矩 (标准化后)
        s_target  : (seq_len, 4) — 后续 seq_len 步的真实状态 (标准化后)

    返回:
        s0_list, tau_list, target_list (list of tensors)
    """
    s0_list, tau_list, target_list = [], [], []
    for _ in range(batch_size):
        traj = trajs[rng.randint(len(trajs))]
        T = len(traj["t"])
        # 起始点必须留出 seq_len 步
        start = rng.randint(0, T - seq_len)
        s_norm = (traj["state"] - stats["mean_state"]) / stats["std_state"]
        tau_norm = (traj["tau"] - stats["mean_tau"]) / stats["std_tau"]

        s0_state = s_norm[start]                                # (4,)
        tau_seq = tau_norm[start:start + seq_len]               # (seq_len, 2)
        target = s_norm[start + 1:start + seq_len + 1]          # (seq_len, 4)

        s0_list.append(torch.tensor(s0_state, dtype=torch.float32))
        tau_list.append(torch.tensor(tau_seq, dtype=torch.float32))
        target_list.append(torch.tensor(target, dtype=torch.float32))

    return (
        torch.stack(s0_list),      # (B,4)
        torch.stack(tau_list),     # (B,H,2)
        torch.stack(target_list),  # (B,H,4)
    )
