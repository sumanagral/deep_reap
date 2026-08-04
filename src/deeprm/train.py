"""
DeepRM_Plus training entrypoint.

  1. Imitation pre-training from SJF expert.
  2. PPO refinement (longer horizon, LR decay, tight clip, entropy anneal).
  3. Save the trained policy.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pandas as pd
import torch

from .env import ClusterConfig, ClusterEnv
from .forecasts import attach_forecast_callback, build_offline_utilization
from .imitation import ImitationConfig, collect_expert_trajectories, train_imitation
from .network import CNNPolicy
from .ppo import PPOConfig, train_ppo


def make_env_factory(
    job_trace_path: str | None,
    cfg: ClusterConfig,
    forecast_mode: str = "proxy",
):
    """
    forecast_mode (used when cfg.reap_channels > 0):
      proxy  — diurnal stand-in (fast, always available)
      oracle — offline packed utilization from the job trace
      zero   — ablating the forecast channels
    """
    trace = pd.read_csv(job_trace_path) if job_trace_path else None
    util_timeline = None
    if cfg.reap_channels > 0 and forecast_mode == "oracle" and trace is not None:
        util_timeline = build_offline_utilization(trace, cfg)

    def _factory(seed: int) -> ClusterEnv:
        env = ClusterEnv(cfg=cfg, job_trace=trace, seed=seed)
        if cfg.reap_channels > 0:
            attach_forecast_callback(
                env,
                mode=forecast_mode,
                util_timeline=util_timeline,
                seed=seed,
            )
        return env

    return _factory


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-trace", default="data/job_trace.csv",
                    help="path to job trace CSV; pass '' for synthetic-on-the-fly")
    ap.add_argument("--out", default="models/deeprm")
    ap.add_argument("--reap-channels", type=int, default=0,
                    help="extra channels for REAP forecast (0 = vanilla DeepRM_Plus)")
    ap.add_argument(
        "--forecast-mode",
        default="proxy",
        choices=["proxy", "oracle", "zero"],
        help="How to fill REAP channels during training when --reap-channels>0",
    )
    ap.add_argument("--imitation-episodes", type=int, default=60)
    # Default ≈ 400 updates × 256 × 4 ≈ 4e5 transitions (~27× old 15-update budget).
    ap.add_argument("--ppo-updates", type=int, default=400)
    ap.add_argument("--ppo-lr", type=float, default=5e-5)
    ap.add_argument("--ppo-lr-end", type=float, default=5e-6)
    ap.add_argument("--ppo-clip", type=float, default=0.05)
    ap.add_argument("--ppo-ent", type=float, default=0.02)
    ap.add_argument("--ppo-ent-end", type=float, default=0.005)
    ap.add_argument("--ppo-target-kl", type=float, default=0.02)
    ap.add_argument("--ppo-bc-coef", type=float, default=0.1,
                    help="Behavioral-cloning pull toward the imitation checkpoint")
    ap.add_argument("--ppo-lr-schedule", default="cosine",
                    choices=["cosine", "linear", "constant"])
    ap.add_argument("--rollout-steps", type=int, default=256)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--n-resources", type=int, default=2)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--n-visible", type=int, default=5)
    ap.add_argument("--episode-max-steps", type=int, default=2000)
    ap.add_argument("--reward-throughput", type=float, default=2.0)
    ap.add_argument("--reward-backlog", type=float, default=0.2)
    ap.add_argument("--reward-wait", type=float, default=0.05)
    ap.add_argument("--max-train-jobs", type=int, default=5000,
                    help="Cap jobs loaded from --job-trace for faster, denser training")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = ClusterConfig(
        n_resources=args.n_resources,
        time_horizon=args.horizon,
        n_visible=args.n_visible,
        reap_channels=args.reap_channels,
        episode_max_steps=args.episode_max_steps,
        reward_throughput_coef=args.reward_throughput,
        reward_backlog_coef=args.reward_backlog,
        reward_wait_coef=args.reward_wait,
    )
    # Optionally truncate the real dump so training sees a denser arrival window.
    job_trace_path = args.job_trace or None
    if job_trace_path and args.max_train_jobs > 0:
        full = pd.read_csv(job_trace_path)
        if len(full) > args.max_train_jobs:
            clipped = Path(args.out) / "_train_jobs_clip.csv"
            Path(args.out).mkdir(parents=True, exist_ok=True)
            full.head(args.max_train_jobs).to_csv(clipped, index=False)
            print(f"[train_deeprm] clipped {len(full)} → {args.max_train_jobs} jobs ({clipped})")
            job_trace_path = str(clipped)

    env_factory = make_env_factory(
        job_trace_path, cfg, forecast_mode=args.forecast_mode,
    )

    sample = env_factory(args.seed)
    in_channels = sample.state_shape[0]
    print(f"[train_deeprm] state_shape={sample.state_shape}  action_dim={sample.action_dim}")

    policy = CNNPolicy(in_channels=in_channels, action_dim=sample.action_dim)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = "deepreap" if args.reap_channels > 0 else "deeprm_plus"

    def _save(name: str) -> None:
        torch.save(
            {"policy": policy.state_dict(), "cfg": cfg.__dict__,
             "in_channels": in_channels, "action_dim": sample.action_dim},
            out / f"{name}.pt",
        )
        print(f"[train_deeprm] saved {out / (name + '.pt')}")

    # ---------- imitation -----------------------------------------
    imi_cfg = ImitationConfig(n_episodes=args.imitation_episodes)
    print("[train_deeprm] collecting SJF expert trajectories...")
    s, a = collect_expert_trajectories(env_factory, imi_cfg, seed=args.seed)
    print(f"[train_deeprm] expert buffer: states={s.shape}  actions={a.shape}")
    train_imitation(policy, s, a, imi_cfg, device=args.device)
    # checkpoint after imitation -- a solid SJF-like baseline before any RL
    _save(f"{tag}_imitation")
    bc_policy = copy.deepcopy(policy)

    # ---------- PPO -----------------------------------------------
    ppo_cfg = PPOConfig(
        total_updates=args.ppo_updates,
        rollout_steps=args.rollout_steps,
        n_envs=args.n_envs,
        lr=args.ppo_lr,
        lr_end=args.ppo_lr_end,
        clip_range=args.ppo_clip,
        ent_coef=args.ppo_ent,
        ent_coef_end=args.ppo_ent_end,
        target_kl=args.ppo_target_kl,
        lr_schedule=args.ppo_lr_schedule,
        bc_coef=args.ppo_bc_coef,
    )
    n_trans = args.ppo_updates * args.rollout_steps * args.n_envs
    print(
        f"[train_deeprm] PPO: updates={args.ppo_updates}  "
        f"approx_transitions={n_trans}  clip={args.ppo_clip}  "
        f"lr={args.ppo_lr}->{args.ppo_lr_end} ({args.ppo_lr_schedule})  "
        f"bc_coef={args.ppo_bc_coef}"
    )
    history = train_ppo(
        policy, env_factory, ppo_cfg, device=args.device, seed=args.seed,
        bc_policy=bc_policy,
    )

    # ---------- save final ---------------------------------------
    _save(tag)
    with open(out / f"{tag}_history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
