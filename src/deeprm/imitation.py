"""
Imitation pre-training for DeepRM_Plus.

Strategy:
    1. Run the SJF baseline against episodes to collect (state, action) pairs.
    2. Train the CNN policy with cross-entropy to imitate SJF.
This warm-starts the policy and sharply reduces the number of PPO
episodes required, exactly as described in the DeepRM_Plus paper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .baselines import sjf_action
from .env import ClusterEnv
from .network import CNNPolicy


@dataclass
class ImitationConfig:
    n_episodes: int = 80
    max_steps: int = 800
    epochs: int = 6
    batch_size: int = 128
    lr: float = 3e-4


def collect_expert_trajectories(
    env_factory, cfg: ImitationConfig, seed: int = 0, verbose: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    states, actions = [], []
    for ep in range(cfg.n_episodes):
        env: ClusterEnv = env_factory(seed + ep)
        s = env.reset()
        for _ in range(cfg.max_steps):
            a = sjf_action(env)
            states.append(s.copy())
            actions.append(a)
            s, _, done, _ = env.step(a)
            if done:
                break
        if verbose and (ep + 1) % 10 == 0:
            print(f"[imitation] collected {ep + 1}/{cfg.n_episodes} episodes  buf={len(states)}")
    return np.stack(states), np.array(actions, dtype=np.int64)


def train_imitation(
    policy: CNNPolicy,
    states: np.ndarray,
    actions: np.ndarray,
    cfg: ImitationConfig,
    device: str = "cpu",
    verbose: bool = True,
) -> CNNPolicy:
    policy.to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)

    ds = TensorDataset(
        torch.from_numpy(states).float(),
        torch.from_numpy(actions).long(),
    )
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    for epoch in range(cfg.epochs):
        total_loss = 0.0
        n = 0
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits, _ = policy(X)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(y)
            n += len(y)
        if verbose:
            print(f"[imitation] epoch {epoch + 1}/{cfg.epochs}  loss={total_loss / n:.4f}")
    return policy
