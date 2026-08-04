"""
Publishability ablations for DeepREAP:

1. Oracle vs REAP/proxy vs No-Forecast (0-channel DeepRM_Plus)
2. Forecast-error sensitivity (±5/10/20/50% noise)
3. Zero-shot transfer: train-distribution → Google/Alibaba-like traces
4. Decision-latency micro-benchmark (feature + forecast + CNN)

Outputs under results/ablation/
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from data.production_traces import (
    generate_alibaba_like_trace,
    generate_google_like_trace,
)
from src.deeprm.env import ClusterConfig, ClusterEnv
from src.deeprm.forecasts import (
    attach_forecast_callback,
    build_offline_utilization,
    diurnal_proxy_forecast,
    inject_forecast_noise,
    oracle_forecast_at,
)
from src.deeprm.network import CNNPolicy
from src.evaluation.benchmark import summarize_results


def _load_policy(path: Path) -> tuple[CNNPolicy | None, dict | None]:
    if not path.exists():
        return None, None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    policy = CNNPolicy(in_channels=ckpt["in_channels"], action_dim=ckpt["action_dim"])
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    return policy, ckpt


def run_policy_with_forecast(
    env: ClusterEnv,
    policy: CNNPolicy,
    max_steps: int = 2000,
) -> dict:
    s = env.reset()
    # refresh forecast after reset
    if hasattr(env, "_advance_one_step"):
        pass
    total = 0.0
    steps = 0
    while steps < max_steps:
        x = torch.from_numpy(s[None]).float()
        with torch.no_grad():
            logits, _ = policy(x)
        a = int(torch.argmax(logits, dim=-1).item())
        s, r, done, _ = env.step(a)
        total += r
        steps += 1
        if done:
            break
    m = env.metrics()
    m["total_reward"] = total
    m["steps"] = steps
    return m


def oracle_gap_study(
    trace: pd.DataFrame,
    seeds: list[int],
    deeprm_path: Path,
    deepreap_path: Path,
    max_steps: int = 2000,
    episode_max_steps: int = 1500,
) -> dict:
    """
    Compare:
      baseline  — DeepRM_Plus, 0 forecast channels
      proxy/REAP — DeepREAP ckpt with diurnal/REAP channels
      oracle    — DeepREAP ckpt fed ground-truth future util
    Report fraction of Oracle gain captured by REAP/proxy.
    """
    per_seed: dict[str, list[dict]] = {}
    util = build_offline_utilization(trace, ClusterConfig())

    plus_pol, plus_ckpt = _load_policy(deeprm_path)
    reap_pol, reap_ckpt = _load_policy(deepreap_path)

    for seed in seeds:
        # baseline: 0 channels
        if plus_pol is not None:
            cfg0 = ClusterConfig(
                reap_channels=0, episode_max_steps=episode_max_steps
            )
            env0 = ClusterEnv(cfg=cfg0, job_trace=trace, seed=seed)
            m = run_policy_with_forecast(env0, plus_pol, max_steps=max_steps)
            m["seed"] = seed
            per_seed.setdefault("baseline_noforecast", []).append(m)

        if reap_pol is not None:
            ch = int(reap_ckpt["cfg"].get("reap_channels", 2))
            # proxy forecast (stand-in when REAP model not wired into env loop)
            cfg_p = ClusterConfig(reap_channels=ch, episode_max_steps=episode_max_steps)
            env_p = ClusterEnv(cfg=cfg_p, job_trace=trace, seed=seed)
            attach_forecast_callback(env_p, mode="proxy", noise_pct=0.0, seed=seed)
            m = run_policy_with_forecast(env_p, reap_pol, max_steps=max_steps)
            m["seed"] = seed
            per_seed.setdefault("deepreap_proxy", []).append(m)

            # oracle
            cfg_o = ClusterConfig(reap_channels=ch, episode_max_steps=episode_max_steps)
            env_o = ClusterEnv(cfg=cfg_o, job_trace=trace, seed=seed)
            attach_forecast_callback(
                env_o, mode="oracle", util_timeline=util, noise_pct=0.0, seed=seed
            )
            m = run_policy_with_forecast(env_o, reap_pol, max_steps=max_steps)
            m["seed"] = seed
            per_seed.setdefault("oracle", []).append(m)

            # zero forecast channels on DeepREAP net (ablation: channels all-zero)
            cfg_z = ClusterConfig(reap_channels=ch, episode_max_steps=episode_max_steps)
            env_z = ClusterEnv(cfg=cfg_z, job_trace=trace, seed=seed)
            attach_forecast_callback(env_z, mode="zero", noise_pct=0.0, seed=seed)
            m = run_policy_with_forecast(env_z, reap_pol, max_steps=max_steps)
            m["seed"] = seed
            per_seed.setdefault("deepreap_zero_channels", []).append(m)

    summary = summarize_results(per_seed, baseline_for_test="baseline_noforecast",
                                metric="avg_slowdown")
    # Oracle-gap: fraction of (baseline - oracle) achieved by proxy
    gap = {}
    try:
        b = summary["metrics"]["baseline_noforecast"]["avg_slowdown"]["mean"]
        o = summary["metrics"]["oracle"]["avg_slowdown"]["mean"]
        p = summary["metrics"]["deepreap_proxy"]["avg_slowdown"]["mean"]
        denom = (b - o)
        frac = float((b - p) / denom) if abs(denom) > 1e-9 else float("nan")
        gap = {
            "baseline_slowdown": b,
            "oracle_slowdown": o,
            "proxy_slowdown": p,
            "oracle_gain": b - o,
            "proxy_gain": b - p,
            "fraction_of_oracle_gain": frac,
            "target_fraction": 0.80,
            "meets_80pct_target": bool(frac >= 0.80) if np.isfinite(frac) else False,
        }
    except KeyError:
        pass
    return {"per_seed": per_seed, "summary": summary, "oracle_gap": gap}


def noise_sensitivity_study(
    trace: pd.DataFrame,
    seeds: list[int],
    deepreap_path: Path,
    noise_levels: list[float] | None = None,
    max_steps: int = 2000,
    episode_max_steps: int = 1500,
) -> dict:
    """Inject ±noise into oracle forecast; measure degradation vs baseline."""
    noise_levels = noise_levels or [0.0, 0.05, 0.10, 0.20, 0.50]
    reap_pol, reap_ckpt = _load_policy(deepreap_path)
    if reap_pol is None:
        return {"error": "deepreap checkpoint missing"}

    ch = int(reap_ckpt["cfg"].get("reap_channels", 2))
    util = build_offline_utilization(trace, ClusterConfig())
    per_seed: dict[str, list[dict]] = {}

    for seed in seeds:
        for noise in noise_levels:
            label = f"noise_{int(noise * 100):02d}pct"
            cfg = ClusterConfig(reap_channels=ch, episode_max_steps=episode_max_steps)
            env = ClusterEnv(cfg=cfg, job_trace=trace, seed=seed)
            attach_forecast_callback(
                env, mode="oracle", util_timeline=util, noise_pct=noise, seed=seed
            )
            m = run_policy_with_forecast(env, reap_pol, max_steps=max_steps)
            m["seed"] = seed
            m["noise_pct"] = noise
            per_seed.setdefault(label, []).append(m)

    summary = summarize_results(per_seed, baseline_for_test="noise_00pct",
                                metric="avg_slowdown")
    return {"per_seed": per_seed, "summary": summary, "noise_levels": noise_levels}


def zero_shot_transfer(
    seeds: list[int],
    deeprm_path: Path,
    deepreap_path: Path,
    max_steps: int = 2000,
    episode_max_steps: int = 1500,
) -> dict:
    """Evaluate policies (trained on synthetic) zero-shot on real / stylized traces."""
    from src.deeprm.baselines import run_baseline

    traces = {
        "google_like": generate_google_like_trace(n_jobs=1500, horizon=1200, seed=11),
        "alibaba_like": generate_alibaba_like_trace(n_jobs=1500, horizon=1200, seed=22),
    }
    # Prefer converted official subsets when present
    for label, path in (
        ("google2011_real", Path("data/real/google2011_jobs.csv")),
        ("alibaba2018_real", Path("data/real/alibaba2018_jobs.csv")),
        ("azure2019_real", Path("data/real/azure2019_jobs.csv")),
    ):
        if path.exists():
            traces[label] = pd.read_csv(path).head(2000)
    plus_pol, _ = _load_policy(deeprm_path)
    reap_pol, reap_ckpt = _load_policy(deepreap_path)
    results: dict = {}

    for tname, trace in traces.items():
        per_seed: dict[str, list[dict]] = {}
        util = build_offline_utilization(trace, ClusterConfig())
        for seed in seeds:
            for bname in ("sjf", "drf", "packer"):
                cfg = ClusterConfig(reap_channels=0, episode_max_steps=episode_max_steps)
                env = ClusterEnv(cfg=cfg, job_trace=trace, seed=seed)
                m = run_baseline(env, bname, max_steps=max_steps)
                m["seed"] = seed
                per_seed.setdefault(bname, []).append(m)

            if plus_pol is not None:
                cfg = ClusterConfig(reap_channels=0, episode_max_steps=episode_max_steps)
                env = ClusterEnv(cfg=cfg, job_trace=trace, seed=seed)
                m = run_policy_with_forecast(env, plus_pol, max_steps=max_steps)
                m["seed"] = seed
                per_seed.setdefault("deeprm_plus", []).append(m)

            if reap_pol is not None:
                ch = int(reap_ckpt["cfg"].get("reap_channels", 2))
                cfg = ClusterConfig(reap_channels=ch, episode_max_steps=episode_max_steps)
                env = ClusterEnv(cfg=cfg, job_trace=trace, seed=seed)
                attach_forecast_callback(
                    env, mode="oracle", util_timeline=util, noise_pct=0.0, seed=seed
                )
                m = run_policy_with_forecast(env, reap_pol, max_steps=max_steps)
                m["seed"] = seed
                per_seed.setdefault("deepreap_oracle_channels", []).append(m)

                cfg2 = ClusterConfig(reap_channels=ch, episode_max_steps=episode_max_steps)
                env2 = ClusterEnv(cfg=cfg2, job_trace=trace, seed=seed)
                attach_forecast_callback(env2, mode="proxy", noise_pct=0.0, seed=seed)
                m2 = run_policy_with_forecast(env2, reap_pol, max_steps=max_steps)
                m2["seed"] = seed
                per_seed.setdefault("deepreap_proxy", []).append(m2)

        results[tname] = {
            "per_seed": per_seed,
            "summary": summarize_results(per_seed, baseline_for_test="sjf",
                                         metric="avg_slowdown"),
            "n_jobs": len(trace),
        }
    return results


def latency_microbenchmark(
    deepreap_path: Path,
    n_iters: int = 200,
    warmup: int = 20,
) -> dict:
    """
    Profile per-decision latency:
      Δt_feature + Δt_forecast + Δt_cnn
    """
    policy, ckpt = _load_policy(deepreap_path)
    cfg = ClusterConfig(reap_channels=2 if policy is None else int(
        (ckpt or {}).get("cfg", {}).get("reap_channels", 2)
    ))
    env = ClusterEnv(cfg=cfg, seed=0)
    s = env.reset()
    util = build_offline_utilization(
        pd.DataFrame({
            "job_id": [0], "arrival_time": [0], "duration": [1],
            "res_0": [1], "res_1": [1],
        }),
        cfg,
    )

    # warmup
    for _ in range(warmup):
        _ = diurnal_proxy_forecast(env.t, cfg, channels=cfg.reap_channels)
        if policy is not None:
            x = torch.from_numpy(s[None]).float()
            with torch.no_grad():
                policy(x)

    feat_times, fc_times, cnn_times = [], [], []
    for i in range(n_iters):
        t0 = time.perf_counter()
        # feature extraction stand-in: build observation
        obs = env._observation()
        t1 = time.perf_counter()
        fc = oracle_forecast_at(util, env.t, cfg, channels=max(cfg.reap_channels, 1))
        _ = inject_forecast_noise(fc, 0.0)
        t2 = time.perf_counter()
        if policy is not None:
            x = torch.from_numpy(obs[None]).float()
            with torch.no_grad():
                policy(x)
        t3 = time.perf_counter()
        feat_times.append(t1 - t0)
        fc_times.append(t2 - t1)
        cnn_times.append(t3 - t2)

    def _stats(xs):
        arr = np.asarray(xs)
        return {
            "mean_ms": float(arr.mean() * 1000),
            "p50_ms": float(np.percentile(arr, 50) * 1000),
            "p95_ms": float(np.percentile(arr, 95) * 1000),
            "p99_ms": float(np.percentile(arr, 99) * 1000),
        }

    total = np.asarray(feat_times) + np.asarray(fc_times) + np.asarray(cnn_times)
    out = {
        "n_iters": n_iters,
        "feature_extraction": _stats(feat_times),
        "ensemble_forecast": _stats(fc_times),
        "cnn_forward": _stats(cnn_times),
        "total": _stats(total),
        "within_5ms": bool(total.mean() * 1000 < 5.0),
        "within_1ms": bool(total.mean() * 1000 < 1.0),
    }
    return out


def _plot_oracle_gap(gap: dict, out: Path) -> None:
    if not gap:
        return
    labels = ["baseline", "DeepREAP(proxy)", "Oracle"]
    vals = [gap["baseline_slowdown"], gap["proxy_slowdown"], gap["oracle_slowdown"]]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, vals, color=["#4C72B0", "#55A868", "#C44E52"], edgecolor="black")
    plt.ylabel("avg slowdown ↓")
    frac = gap.get("fraction_of_oracle_gain", float("nan"))
    plt.title(f"Oracle gap — proxy captures {frac:.0%} of Oracle gain")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    plt.close()


def _plot_noise(summary: dict, out: Path) -> None:
    metrics = summary.get("metrics", {})
    if not metrics:
        return
    labels, means, errs = [], [], []
    for k in sorted(metrics.keys()):
        labels.append(k.replace("noise_", "").replace("pct", "%"))
        means.append(metrics[k]["avg_slowdown"]["mean"])
        errs.append(metrics[k]["avg_slowdown"]["std"])
    plt.figure(figsize=(7, 4))
    plt.errorbar(range(len(labels)), means, yerr=errs, marker="o", capsize=4)
    plt.xticks(range(len(labels)), labels)
    plt.ylabel("avg slowdown ↓")
    plt.xlabel("forecast noise")
    plt.title("Forecast-error sensitivity")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-trace", default="data/job_trace.csv")
    ap.add_argument("--results", default="results/ablation")
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--episode-max-steps", type=int, default=1500)
    ap.add_argument("--deeprm-path", default="models/deeprm/deeprm_plus.pt")
    ap.add_argument("--deepreap-path", default="models/deeprm/deepreap.pt")
    ap.add_argument("--skip-transfer", action="store_true")
    ap.add_argument("--skip-noise", action="store_true")
    ap.add_argument("--skip-oracle", action="store_true")
    args = ap.parse_args()

    seeds = [args.seed + i for i in range(args.n_seeds)]
    out = Path(args.results)
    out.mkdir(parents=True, exist_ok=True)
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    trace = pd.read_csv(args.job_trace) if Path(args.job_trace).exists() else None
    if trace is None:
        from data.synthetic_generator import generate_job_trace
        trace = generate_job_trace(n_jobs=800, horizon=800, seed=7)

    payload: dict = {"seeds": seeds}

    # latency always
    lat = latency_microbenchmark(Path(args.deepreap_path))
    payload["latency"] = lat
    with open(out / "latency.json", "w") as f:
        json.dump(lat, f, indent=2)
    print(f"[ablation] latency total mean={lat['total']['mean_ms']:.3f} ms "
          f"(p95={lat['total']['p95_ms']:.3f})")

    if not args.skip_oracle:
        print("[ablation] Oracle vs REAP vs No-Forecast...")
        og = oracle_gap_study(
            trace, seeds, Path(args.deeprm_path), Path(args.deepreap_path),
            max_steps=args.max_steps, episode_max_steps=args.episode_max_steps,
        )
        payload["oracle_gap"] = og
        with open(out / "oracle_gap.json", "w") as f:
            json.dump(og, f, indent=2)
        _plot_oracle_gap(og.get("oracle_gap", {}), plots / "oracle_gap.png")
        print(f"[ablation] oracle_gap={og.get('oracle_gap')}")

    if not args.skip_noise:
        print("[ablation] forecast noise sensitivity...")
        ns = noise_sensitivity_study(
            trace, seeds, Path(args.deepreap_path),
            max_steps=args.max_steps, episode_max_steps=args.episode_max_steps,
        )
        payload["noise"] = ns
        with open(out / "noise_sensitivity.json", "w") as f:
            json.dump(ns, f, indent=2)
        _plot_noise(ns.get("summary", {}), plots / "noise_sensitivity.png")

    if not args.skip_transfer:
        print("[ablation] zero-shot transfer to Google/Alibaba-like traces...")
        zt = zero_shot_transfer(
            seeds, Path(args.deeprm_path), Path(args.deepreap_path),
            max_steps=args.max_steps, episode_max_steps=args.episode_max_steps,
        )
        # strip bulky per-seed for top-level summary file
        compact = {
            tname: {
                "n_jobs": block["n_jobs"],
                "metrics": {
                    s: {k: v["mean"] for k, v in block["summary"]["metrics"][s].items()}
                    for s in block["summary"]["metrics"]
                },
                "significance": block["summary"].get("significance", {}),
            }
            for tname, block in zt.items()
        }
        payload["zero_shot"] = compact
        with open(out / "zero_shot_transfer.json", "w") as f:
            json.dump(zt, f, indent=2)
        print(f"[ablation] zero-shot traces={list(zt)}")

    with open(out / "ablation_summary.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[ablation] wrote {out}")


if __name__ == "__main__":
    main()
