"""
Lightweight PPO trainer for DeepRM_Plus.

Implements a clipped-objective PPO with GAE, learning-rate decay,
entropy annealing, and KL early-stop to protect the imitation checkpoint
from catastrophic forgetting during fine-tuning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .network import CNNPolicy


@dataclass
class PPOConfig:
    # Scale: 400 updates × 256 steps × 4 envs ≈ 4.1e5 transitions (≈27× old
    # 15-update budget). Raise further via CLI for 1e6-scale runs.
    total_updates: int = 400
    rollout_steps: int = 256
    n_envs: int = 4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.05          # tighter clip preserves imitation prior
    epochs: int = 3
    minibatch_size: int = 128
    lr: float = 5e-5                  # lower LR for fine-tuning from imitation
    lr_end: float = 5e-6              # cosine-decay floor
    ent_coef: float = 0.005           # start modest; annealed further
    ent_coef_end: float = 0.001
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.015          # abort epoch if approx KL exceeds this
    lr_schedule: str = "cosine"       # "cosine" | "linear" | "constant"


class _RolloutBuf:
    def __init__(self, T: int, n_envs: int, state_shape, device):
        self.T, self.n = T, n_envs
        self.s = torch.zeros((T, n_envs, *state_shape), device=device)
        self.a = torch.zeros((T, n_envs), dtype=torch.long, device=device)
        self.logp = torch.zeros((T, n_envs), device=device)
        self.r = torch.zeros((T, n_envs), device=device)
        self.v = torch.zeros((T, n_envs), device=device)
        self.done = torch.zeros((T, n_envs), device=device)


def _select_action(policy, s, deterministic=False):
    logits, value = policy(s)
    if deterministic:
        a = torch.argmax(logits, dim=-1)
    else:
        a = torch.distributions.Categorical(logits=logits).sample()
    logp = F.log_softmax(logits, dim=-1).gather(-1, a.unsqueeze(-1)).squeeze(-1)
    return a, logp, value


def _annealed(start: float, end: float, progress: float, schedule: str) -> float:
    """progress in [0, 1]."""
    p = float(np.clip(progress, 0.0, 1.0))
    if schedule == "constant":
        return start
    if schedule == "linear":
        return start + (end - start) * p
    # cosine
    return end + 0.5 * (start - end) * (1.0 + np.cos(np.pi * p))


def train_ppo(
    policy: CNNPolicy,
    env_factory,
    cfg: PPOConfig | None = None,
    device: str = "cpu",
    verbose: bool = True,
    seed: int = 0,
) -> dict:
    cfg = cfg or PPOConfig()
    policy.to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)

    envs = [env_factory(seed + i) for i in range(cfg.n_envs)]
    obs = np.stack([e.reset() for e in envs])  # (n_envs, C, H, W)
    state_shape = obs.shape[1:]

    history: dict[str, list[float]] = {
        "update": [],
        "mean_reward": [],
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "approx_kl": [],
        "lr": [],
        "clip_range": [],
        "ent_coef": [],
        "transitions": [],
    }
    transitions_seen = 0

    for update in range(cfg.total_updates):
        progress = update / max(cfg.total_updates - 1, 1)
        lr_now = _annealed(cfg.lr, cfg.lr_end, progress, cfg.lr_schedule)
        ent_now = _annealed(cfg.ent_coef, cfg.ent_coef_end, progress, cfg.lr_schedule)
        for g in opt.param_groups:
            g["lr"] = lr_now

        buf = _RolloutBuf(cfg.rollout_steps, cfg.n_envs, state_shape, device)
        ep_rewards = [0.0] * cfg.n_envs
        finished_rewards: list[float] = []

        for t in range(cfg.rollout_steps):
            s = torch.from_numpy(obs).float().to(device)
            with torch.no_grad():
                a, logp, v = _select_action(policy, s)

            new_obs = []
            rewards = []
            dones = []
            for i, env in enumerate(envs):
                ns, r, d, _ = env.step(int(a[i].item()))
                ep_rewards[i] += r
                if d:
                    finished_rewards.append(ep_rewards[i])
                    ep_rewards[i] = 0.0
                    ns = env.reset(seed=seed + update * cfg.n_envs + i + 1000)
                new_obs.append(ns)
                rewards.append(r)
                dones.append(float(d))

            buf.s[t] = s
            buf.a[t] = a
            buf.logp[t] = logp
            buf.v[t] = v
            buf.r[t] = torch.tensor(rewards, device=device, dtype=torch.float32)
            buf.done[t] = torch.tensor(dones, device=device, dtype=torch.float32)
            obs = np.stack(new_obs)

        transitions_seen += cfg.rollout_steps * cfg.n_envs

        # bootstrap value
        with torch.no_grad():
            _, _, last_v = _select_action(policy, torch.from_numpy(obs).float().to(device))
        # GAE
        adv = torch.zeros_like(buf.r)
        last_gae = torch.zeros(cfg.n_envs, device=device)
        for t in reversed(range(cfg.rollout_steps)):
            next_v = last_v if t == cfg.rollout_steps - 1 else buf.v[t + 1]
            next_nonterm = 1.0 - buf.done[t]
            delta = buf.r[t] + cfg.gamma * next_v * next_nonterm - buf.v[t]
            last_gae = delta + cfg.gamma * cfg.gae_lambda * next_nonterm * last_gae
            adv[t] = last_gae
        returns = adv + buf.v
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # flatten
        b_s = buf.s.reshape(-1, *state_shape)
        b_a = buf.a.reshape(-1)
        b_lp = buf.logp.reshape(-1)
        b_adv = adv.reshape(-1)
        b_ret = returns.reshape(-1)

        N = b_s.size(0)
        idx = np.arange(N)
        last_pl, last_vl, last_ent, last_kl = 0.0, 0.0, 0.0, 0.0
        early_stopped = False

        for _ in range(cfg.epochs):
            if early_stopped:
                break
            np.random.shuffle(idx)
            for start in range(0, N, cfg.minibatch_size):
                mb = idx[start : start + cfg.minibatch_size]
                logits, vals = policy(b_s[mb])
                dist = torch.distributions.Categorical(logits=logits)
                new_lp = dist.log_prob(b_a[mb])
                ratio = torch.exp(new_lp - b_lp[mb])
                pg1 = ratio * b_adv[mb]
                pg2 = torch.clamp(ratio, 1 - cfg.clip_range, 1 + cfg.clip_range) * b_adv[mb]
                pg_loss = -torch.min(pg1, pg2).mean()
                v_loss = F.mse_loss(vals, b_ret[mb])
                ent = dist.entropy().mean()
                loss = pg_loss + cfg.vf_coef * v_loss - ent_now * ent

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                opt.step()

                with torch.no_grad():
                    approx_kl = (b_lp[mb] - new_lp).mean().item()
                last_pl, last_vl = pg_loss.item(), v_loss.item()
                last_ent, last_kl = ent.item(), approx_kl
                if approx_kl > cfg.target_kl:
                    early_stopped = True
                    break

        mean_r = (
            float(np.mean(finished_rewards))
            if finished_rewards
            else float(buf.r.sum().item() / cfg.n_envs)
        )
        history["update"].append(update)
        history["mean_reward"].append(mean_r)
        history["policy_loss"].append(last_pl)
        history["value_loss"].append(last_vl)
        history["entropy"].append(last_ent)
        history["approx_kl"].append(last_kl)
        history["lr"].append(lr_now)
        history["clip_range"].append(cfg.clip_range)
        history["ent_coef"].append(ent_now)
        history["transitions"].append(transitions_seen)
        if verbose:
            print(
                f"[ppo] upd={update + 1:04d}/{cfg.total_updates}  "
                f"R={mean_r:8.2f}  pl={last_pl:7.4f}  vl={last_vl:7.4f}  "
                f"ent={last_ent:6.3f}  kl={last_kl:6.4f}  lr={lr_now:.2e}  "
                f"N={transitions_seen}"
            )
    return history
