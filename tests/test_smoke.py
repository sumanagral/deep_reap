"""
Fast smoke tests. They exercise the wiring of every module on tiny inputs;
they are NOT a substitute for the full benchmark.

Run with:
    pytest -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from data.synthetic_generator import generate_job_trace, generate_resource_usage
from src.deeprm.baselines import run_baseline
from src.deeprm.env import ClusterConfig, ClusterEnv
from src.deeprm.imitation import ImitationConfig, collect_expert_trajectories, train_imitation
from src.deeprm.network import CNNPolicy
from src.deeprm.ppo import PPOConfig, train_ppo
from src.reap.feature_selection import GAConfig, select_features
from src.reap.ensemble import EnsembleConfig, compute_weights, train_reap
from src.evaluation.benchmark import summarize_results


def test_generators_shapes():
    df = generate_resource_usage(n_hours=48)
    assert len(df) == 48 * 4
    assert {"cpu_load", "memory_usage", "network_utilization"} <= set(df.columns)

    jobs = generate_job_trace(n_jobs=200, horizon=200)
    assert {"job_id", "arrival_time", "duration", "res_0", "res_1"} <= set(jobs.columns)


def test_ga_feature_selection_runs():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 8))
    # only features 1 and 4 matter
    y = 2.0 * X[:, 1] - 1.0 * X[:, 4] + rng.normal(scale=0.1, size=200)
    cfg = GAConfig(pop_size=10, n_generations=5, seed=0)
    mask, selected, fit = select_features(X, y, cfg=cfg)
    assert len(mask) == 8
    assert mask.sum() >= 2
    assert fit > -1e9


def test_softmax_weights_sharper_than_inv_mse():
    # Similar MSEs → inv_mse nearly uniform; softmax(τ=0.5) concentrates on best
    mses = np.array([24.5, 25.0, 25.1, 30.0, 40.0])
    inv = compute_weights(mses, scheme="inv_mse")
    soft = compute_weights(mses, scheme="softmax", temperature=0.5)
    topk = compute_weights(mses, scheme="topk", temperature=1.0, top_k=2)
    assert np.isclose(inv.sum(), 1.0)
    assert np.isclose(soft.sum(), 1.0)
    assert np.isclose(topk.sum(), 1.0)
    # Softmax should put more mass on the best model than inv_mse
    assert soft[0] > inv[0]
    # Top-K zeros out models beyond K
    assert (topk > 0).sum() == 2
    assert topk[0] > 0 and topk[1] > 0


def test_reap_train_softmax_and_stacking():
    df = generate_resource_usage(n_hours=24 * 7)
    feats = ["hour_of_day", "day_of_week", "is_weekend", "active_users",
             "previous_hour_cpu", "network_utilization"]
    X = df[feats].to_numpy(dtype=float)
    y = df["cpu_load"].to_numpy(dtype=float)
    for scheme in ("softmax", "topk", "stacking"):
        model = train_reap(
            X, y, feats, "cpu_load",
            ga_cfg=GAConfig(pop_size=8, n_generations=3, seed=0),
            ensemble_cfg=EnsembleConfig(scheme=scheme, temperature=1.5, top_k=3),
            verbose=False,
        )
        pred = model.predict(X[:10])
        assert pred.shape == (10,)
        assert "ENSEMBLE" in model.metrics
        assert model.scheme == scheme


def test_reap_online_reweight():
    df = generate_resource_usage(n_hours=24 * 5)
    feats = ["hour_of_day", "day_of_week", "active_users", "previous_hour_cpu"]
    X = df[feats].to_numpy(dtype=float)
    y = df["cpu_load"].to_numpy(dtype=float)
    model = train_reap(
        X, y, feats, "cpu_load",
        ga_cfg=GAConfig(pop_size=6, n_generations=2, seed=1),
        ensemble_cfg=EnsembleConfig(scheme="softmax", temperature=1.0, online_lr=0.5),
        verbose=False,
    )
    w0 = model.weights.copy()
    # Feed several feedback points
    for i in range(5):
        model.update_online_weights(y_true=float(y[i]), X_raw=X[i])
    assert model.weights.shape == w0.shape
    assert np.isclose(model.weights.sum(), 1.0)


def test_env_multiobjective_reward_and_baselines():
    cfg = ClusterConfig(
        time_horizon=10, n_visible=3, episode_max_steps=80,
        reward_throughput_coef=0.5, reward_backlog_coef=0.05, reward_wait_coef=0.01,
    )
    jobs = generate_job_trace(n_jobs=80, horizon=80, seed=1)
    env = ClusterEnv(cfg=cfg, job_trace=jobs, seed=1)
    obs = env.reset()
    assert obs.shape == env.state_shape
    # Advance a few steps and ensure reward is finite
    rewards = []
    for _ in range(20):
        _, r, done, _ = env.step(env.action_dim - 1)  # no-op advance
        rewards.append(r)
        if done:
            break
    assert all(np.isfinite(rewards))
    m = env.metrics()
    assert "throughput" in m

    for name in ("fifo", "sjf", "packer"):
        env2 = ClusterEnv(cfg=cfg, job_trace=jobs, seed=2)
        mb = run_baseline(env2, name, max_steps=200)
        assert mb["steps"] > 0


def test_cnn_policy_forward():
    cfg = ClusterConfig(time_horizon=10, n_visible=3, reap_channels=2)
    env = ClusterEnv(cfg=cfg, seed=0)
    s = env.reset()
    pol = CNNPolicy(in_channels=s.shape[0], action_dim=env.action_dim)
    x = torch.from_numpy(s[None]).float()
    logits, v = pol(x)
    assert logits.shape == (1, env.action_dim)
    assert v.shape == (1,)


def test_imitation_pipeline_tiny():
    cfg = ClusterConfig(time_horizon=8, n_visible=3, episode_max_steps=30)

    def factory(seed):
        return ClusterEnv(cfg=cfg, seed=seed)

    pol = CNNPolicy(in_channels=1, action_dim=cfg.n_visible + 1)
    icfg = ImitationConfig(n_episodes=3, max_steps=30, epochs=1, batch_size=32)
    s, a = collect_expert_trajectories(factory, icfg, verbose=False)
    assert s.shape[0] == a.shape[0] > 0
    train_imitation(pol, s, a, icfg, verbose=False)


def test_ppo_tiny_with_decay_and_kl():
    cfg = ClusterConfig(time_horizon=8, n_visible=3, episode_max_steps=40)

    def factory(seed):
        return ClusterEnv(cfg=cfg, seed=seed)

    pol = CNNPolicy(in_channels=1, action_dim=cfg.n_visible + 1)
    pcfg = PPOConfig(
        total_updates=2,
        rollout_steps=16,
        n_envs=2,
        epochs=2,
        minibatch_size=16,
        clip_range=0.05,
        lr=1e-4,
        lr_end=1e-5,
        ent_coef=0.01,
        ent_coef_end=0.001,
        target_kl=0.05,
        lr_schedule="cosine",
    )
    hist = train_ppo(pol, factory, pcfg, verbose=False, seed=0)
    assert len(hist["update"]) == 2
    assert hist["transitions"][-1] == 2 * 16 * 2
    assert hist["lr"][0] >= hist["lr"][-1]  # cosine decay


def test_trace_loaders_canonical_and_google():
    from data.trace_loaders import load_job_trace, load_google_cluster
    import tempfile
    from pathlib import Path

    # synthetic via loader
    jobs = load_job_trace(source="synthetic", n_jobs=50, horizon=100, seed=3)
    assert len(jobs) > 0
    assert set(["job_id", "arrival_time", "duration", "res_0", "res_1"]) <= set(jobs.columns)

    # google-like raw dump
    raw = pd.DataFrame({
        "time": np.arange(20) * 1000,
        "finish_time": np.arange(20) * 1000 + 5000,
        "job_id": np.arange(20),
        "cpu_request": np.linspace(0.1, 0.9, 20),
        "memory_request": np.linspace(0.05, 0.8, 20),
    })
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "google.csv"
        raw.to_csv(p, index=False)
        g = load_google_cluster(p, max_jobs=15)
        assert len(g) == 15
        assert g["duration"].min() >= 1
        assert g["res_0"].between(1, 10).all()


def test_multiseed_summary_ci_and_pvalue():
    rng = np.random.default_rng(0)
    per_seed = {
        "sjf": [{"avg_slowdown": float(x), "avg_completion": 10.0, "n_done": 50,
                 "throughput": 0.2, "total_reward": -100.0}
                for x in rng.normal(20, 1, size=10)],
        "deepreap": [{"avg_slowdown": float(x), "avg_completion": 8.0, "n_done": 45,
                      "throughput": 0.18, "total_reward": -90.0}
                     for x in rng.normal(12, 1, size=10)],
    }
    summary = summarize_results(per_seed, baseline_for_test="sjf", metric="avg_slowdown")
    assert "deepreap" in summary["metrics"]
    assert summary["metrics"]["deepreap"]["avg_slowdown"]["ci95_halfwidth"] > 0
    assert "deepreap" in summary["significance"]
    assert 0.0 <= summary["significance"]["deepreap"]["p_value"] <= 1.0
    assert "wilcoxon" in summary["significance"]["deepreap"]
    assert "ttest_rel" in summary["significance"]["deepreap"]


def test_drf_and_ilp_baselines():
    cfg = ClusterConfig(time_horizon=10, n_visible=3, episode_max_steps=60)
    jobs = generate_job_trace(n_jobs=60, horizon=60, seed=5)
    for name in ("drf", "ilp"):
        env = ClusterEnv(cfg=cfg, job_trace=jobs, seed=5)
        m = run_baseline(env, name, max_steps=150)
        assert m["steps"] > 0
        assert "p95_slowdown" in m
        assert "fragmentation" in m
        assert "sla_breach_rate" in m


def test_expanded_env_metrics():
    cfg = ClusterConfig(time_horizon=10, n_visible=3, episode_max_steps=80, sla_max_wait=5)
    jobs = generate_job_trace(n_jobs=80, horizon=80, seed=3)
    env = ClusterEnv(cfg=cfg, job_trace=jobs, seed=3)
    m = run_baseline(env, "sjf", max_steps=200)
    for key in ("p95_slowdown", "p99_slowdown", "p95_wait", "p99_wait",
                "fragmentation", "sla_breach_rate", "throughput"):
        assert key in m


def test_oracle_forecast_and_noise():
    from src.deeprm.forecasts import (
        build_offline_utilization,
        inject_forecast_noise,
        oracle_forecast_at,
        attach_forecast_callback,
    )
    jobs = generate_job_trace(n_jobs=40, horizon=40, seed=2)
    cfg = ClusterConfig(time_horizon=8, n_visible=3, reap_channels=2, episode_max_steps=40)
    util = build_offline_utilization(jobs, cfg)
    assert util.shape[0] == cfg.n_resources
    fc = oracle_forecast_at(util, t=0, cfg=cfg, channels=2)
    assert fc.shape == (2, cfg.n_resources, cfg.time_horizon)
    noisy = inject_forecast_noise(fc, 0.2, rng=np.random.default_rng(0))
    assert noisy.shape == fc.shape
    assert not np.allclose(noisy, fc)

    env = ClusterEnv(cfg=cfg, job_trace=jobs, seed=0)
    attach_forecast_callback(env, mode="oracle", util_timeline=util, noise_pct=0.1, seed=0)
    env.reset()
    assert env.reap_forecast.shape == (2, cfg.n_resources, cfg.time_horizon)


def test_production_traces_and_latency():
    from data.production_traces import (
        generate_alibaba_like_trace,
        generate_google_like_trace,
    )
    from src.evaluation.ablation import latency_microbenchmark

    g = generate_google_like_trace(n_jobs=100, horizon=200, seed=1)
    a = generate_alibaba_like_trace(n_jobs=100, horizon=200, seed=2)
    assert len(g) > 0 and len(a) > 0
    assert {"job_id", "arrival_time", "duration", "res_0", "res_1"} <= set(g.columns)

    lat = latency_microbenchmark(
        __import__("pathlib").Path("models/deeprm/deepreap.pt"),
        n_iters=30,
        warmup=5,
    )
    assert "total" in lat
    assert lat["total"]["mean_ms"] >= 0.0


def test_reap_timeseries_split_no_shuffle():
    df = generate_resource_usage(n_hours=24 * 5)
    feats = ["hour_of_day", "day_of_week", "active_users", "previous_hour_cpu"]
    X = df[feats].to_numpy(dtype=float)
    y = df["cpu_load"].to_numpy(dtype=float)
    model = train_reap(
        X, y, feats, "cpu_load",
        ga_cfg=GAConfig(pop_size=6, n_generations=2, seed=0),
        ensemble_cfg=EnsembleConfig(
            scheme="softmax", temperature=1.5, use_timeseries_cv=True, n_splits=3
        ),
        verbose=False,
    )
    assert model.metrics["ENSEMBLE"].get("timeseries_cv") is True
    assert "cv_mse" in model.metrics[model.models[0].name]
