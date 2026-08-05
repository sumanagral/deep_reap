"""
Minimal DeepRM_Plus / DeepREAP facade (CMPE-294 aligned).

- Vanilla DeepRM_Plus: reap_channels=0
- DeepREAP: reap_channels=2 with live REAP / proxy / oracle callback
- Imitation warm-start from SJF, then short PPO
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.deeprm.env import ClusterConfig, ClusterEnv
from src.deeprm.forecasts import (
    attach_forecast_callback,
    build_offline_utilization,
    diurnal_proxy_forecast,
    inject_forecast_noise,
    oracle_forecast_at,
)
from src.deeprm.imitation import ImitationConfig, collect_expert_trajectories, train_imitation
from src.deeprm.network import CNNPolicy
from src.deeprm.ppo import PPOConfig, train_ppo
from src.reap.ensemble import REAPModel


def _reap_predict_fn(cpu_model: REAPModel, mem_model: REAPModel | None, cfg: ClusterConfig):
    """Map env time → (C,R,T) forecast using fitted REAP models (diurnal roll)."""
    import datetime as dt
    from src.integration.deepreap import ForecastInput, build_reap_forecast

    # Scale observed targets roughly into REAP's training range.
    def _fn(t: int) -> np.ndarray:
        base = dt.datetime(2024, 1, 1) + dt.timedelta(hours=int(t) % (24 * 60))
        inp = ForecastInput(
            timestamp=base,
            service_type="Web",
            active_users=200,
            previous_hour_cpu=30.0,
            previous_hour_memory=40.0,
            network_utilization=10.0,
        )
        return build_reap_forecast(
            cpu_model, mem_model, inp,
            n_resources=cfg.n_resources,
            time_horizon=cfg.time_horizon,
        )

    return _fn


def make_env(
    job_trace: pd.DataFrame | None,
    channels: int,
    mode: str,
    seed: int,
    episode_max_steps: int = 1500,
    util_timeline: np.ndarray | None = None,
    reap_fn=None,
    noise_pct: float = 0.0,
    noise_mode: str = "gaussian",
) -> ClusterEnv:
    cfg = ClusterConfig(
        reap_channels=channels,
        episode_max_steps=episode_max_steps,
        reward_throughput_coef=2.0,
        reward_backlog_coef=0.2,
        reward_wait_coef=0.05,
    )
    env = ClusterEnv(cfg=cfg, job_trace=job_trace, seed=seed)
    if channels > 0:
        attach_forecast_callback(
            env,
            mode=mode if mode != "reap" else ("reap" if reap_fn else "proxy"),
            util_timeline=util_timeline,
            noise_pct=noise_pct,
            seed=seed,
            reap_predict_fn=reap_fn,
            noise_mode=noise_mode,
        )
    return env


def train_policy(
    job_trace_path: str | Path,
    out_dir: str | Path,
    channels: int = 0,
    forecast_mode: str = "proxy",
    cpu_model: REAPModel | None = None,
    mem_model: REAPModel | None = None,
    imitation_episodes: int = 40,
    ppo_updates: int = 120,
    max_jobs: int = 5000,
    seed: int = 42,
    device: str = "cpu",
    tag: str | None = None,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trace = pd.read_csv(job_trace_path)
    if len(trace) > max_jobs:
        trace = trace.head(max_jobs).reset_index(drop=True)
        clip = out / "_train_jobs.csv"
        trace.to_csv(clip, index=False)
        job_trace_path = clip

    cfg = ClusterConfig(
        reap_channels=channels,
        episode_max_steps=1500,
        reward_throughput_coef=2.0,
        reward_backlog_coef=0.2,
        reward_wait_coef=0.05,
    )
    util = None
    reap_fn = None
    if channels > 0 and forecast_mode == "oracle":
        util = build_offline_utilization(pd.read_csv(job_trace_path), cfg)
    if channels > 0 and forecast_mode == "reap" and cpu_model is not None:
        reap_fn = _reap_predict_fn(cpu_model, mem_model, cfg)

    def factory(s: int) -> ClusterEnv:
        return make_env(
            pd.read_csv(job_trace_path),
            channels=channels,
            mode=forecast_mode,
            seed=s,
            util_timeline=util,
            reap_fn=reap_fn,
        )

    sample = factory(seed)
    policy = CNNPolicy(in_channels=sample.state_shape[0], action_dim=sample.action_dim)
    name = tag or ("deepreap" if channels > 0 else "deeprm_plus")

    imi = ImitationConfig(n_episodes=imitation_episodes)
    s, a = collect_expert_trajectories(factory, imi, seed=seed)
    train_imitation(policy, s, a, imi, device=device)
    torch.save(
        {"policy": policy.state_dict(), "cfg": cfg.__dict__,
         "in_channels": sample.state_shape[0], "action_dim": sample.action_dim},
        out / f"{name}_imitation.pt",
    )
    bc = copy.deepcopy(policy)
    hist = train_ppo(
        policy, factory,
        PPOConfig(
            total_updates=ppo_updates,
            clip_range=0.05,
            lr=5e-5,
            lr_end=5e-6,
            ent_coef=0.02,
            ent_coef_end=0.005,
            bc_coef=0.15,
            target_kl=0.02,
        ),
        device=device,
        seed=seed,
        bc_policy=bc,
    )
    ckpt = out / f"{name}.pt"
    torch.save(
        {"policy": policy.state_dict(), "cfg": cfg.__dict__,
         "in_channels": sample.state_shape[0], "action_dim": sample.action_dim},
        ckpt,
    )
    (out / f"{name}_history.json").write_text(json.dumps(hist, indent=2))
    return ckpt


def run_policy_metrics(
    ckpt: Path,
    job_trace: pd.DataFrame,
    channels: int,
    mode: str,
    seed: int,
    max_steps: int = 3000,
    episode_max_steps: int = 1500,
    util_timeline: np.ndarray | None = None,
    reap_fn=None,
    noise_pct: float = 0.0,
    noise_mode: str = "gaussian",
) -> dict:
    from src.evaluation.benchmark import _load_policy, run_policy

    policy, _ = _load_policy(ckpt)
    env = make_env(
        job_trace, channels, mode, seed,
        episode_max_steps=episode_max_steps,
        util_timeline=util_timeline,
        reap_fn=reap_fn,
        noise_pct=noise_pct,
        noise_mode=noise_mode,
    )
    return run_policy(env, policy, max_steps=max_steps)


def run_sjf(job_trace: pd.DataFrame, seed: int, max_steps: int = 3000) -> dict:
    from src.deeprm.baselines import run_baseline

    env = make_env(job_trace, channels=0, mode="zero", seed=seed)
    return run_baseline(env, "sjf", max_steps=max_steps)
