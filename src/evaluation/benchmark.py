"""
Benchmark DeepREAP against heuristic baselines and the vanilla DeepRM_Plus.

Supports multi-seed evaluation with mean ± CI and paired significance tests
against a non-predictive baseline. Evaluation horizon defaults to a long
budget so deferred jobs are forced toward completion.

Outputs
-------
results/benchmark.json         per-seed + aggregate metrics
results/benchmark_summary.json mean / std / CI / p-values
results/plots/*.png            bar charts and learning-curve plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

from src.deeprm.baselines import run_baseline
from src.deeprm.env import ClusterConfig, ClusterEnv
from src.deeprm.forecasts import attach_forecast_callback, build_offline_utilization
from src.deeprm.network import CNNPolicy


# ---------- scheduler runners ---------------------------------------
def run_policy(
    env: ClusterEnv,
    policy: CNNPolicy,
    max_steps: int = 5000,
    profile_latency: bool = False,
) -> dict:
    import time

    s = env.reset()
    total = 0.0
    steps = 0
    latencies: list[float] = []
    policy.eval()
    while steps < max_steps:
        x = torch.from_numpy(s[None]).float()
        t0 = time.perf_counter()
        with torch.no_grad():
            logits, _ = policy(x)
        if profile_latency:
            latencies.append(time.perf_counter() - t0)
        a = int(torch.argmax(logits, dim=-1).item())
        s, r, done, _ = env.step(a)
        total += r
        steps += 1
        if done:
            break
    m = env.metrics()
    m["total_reward"] = total
    m["steps"] = steps
    if latencies:
        arr = np.asarray(latencies)
        m["cnn_latency_mean_ms"] = float(arr.mean() * 1000)
        m["cnn_latency_p95_ms"] = float(np.percentile(arr, 95) * 1000)
        m["cnn_latency_p99_ms"] = float(np.percentile(arr, 99) * 1000)
    return m


def _make_env(
    trace: pd.DataFrame | None,
    reap_channels: int,
    seed: int,
    episode_max_steps: int = 2000,
    forecast_mode: str = "proxy",
    util_timeline: np.ndarray | None = None,
) -> ClusterEnv:
    cfg = ClusterConfig(reap_channels=reap_channels, episode_max_steps=episode_max_steps)
    env = ClusterEnv(cfg=cfg, job_trace=trace, seed=seed)
    if reap_channels > 0:
        attach_forecast_callback(
            env,
            mode=forecast_mode,
            util_timeline=util_timeline,
            seed=seed,
        )
    return env


def _load_policy(path: Path) -> tuple[CNNPolicy, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    policy = CNNPolicy(in_channels=ckpt["in_channels"], action_dim=ckpt["action_dim"])
    policy.load_state_dict(ckpt["policy"])
    return policy, ckpt


def _load_trace(job_trace: str | None, source: str, seed: int) -> pd.DataFrame | None:
    if not job_trace:
        return None
    path = Path(job_trace)
    if not path.exists():
        return None
    if source in ("canonical", "synthetic") or path.suffix.lower() == ".csv":
        # Prefer the dedicated loader when a non-canonical source is requested
        if source not in ("canonical", "synthetic"):
            from data.trace_loaders import load_job_trace
            return load_job_trace(path, source=source, seed=seed)
        return pd.read_csv(path)
    from data.trace_loaders import load_job_trace
    return load_job_trace(path, source=source, seed=seed)


# ---------- stats ---------------------------------------------------
def _ci(values: list[float], alpha: float = 0.05) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean = float(arr.mean()) if n else 0.0
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    if n < 2:
        return mean, std, 0.0
    se = std / np.sqrt(n)
    tcrit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    return mean, std, tcrit * se


METRIC_KEYS = [
    "avg_slowdown", "avg_completion", "p95_slowdown", "p99_slowdown",
    "avg_wait", "p95_wait", "p99_wait",
    "n_done", "throughput", "fragmentation", "avg_utilization",
    "sla_breach_rate", "total_reward",
    "cnn_latency_mean_ms", "cnn_latency_p95_ms",
]


def summarize_results(
    per_seed: dict[str, list[dict]],
    baseline_for_test: str = "sjf",
    metric: str = "avg_slowdown",
) -> dict:
    """
    Aggregate multi-seed runs into mean ± std / 95% CI and paired
    Wilcoxon + t-test p-values vs a non-predictive baseline (default SJF).
    """
    summary: dict = {"n_seeds": {}, "metrics": {}, "significance": {}}
    for sched, runs in per_seed.items():
        summary["n_seeds"][sched] = len(runs)
        summary["metrics"][sched] = {}
        # discover keys present in runs
        keys = list(METRIC_KEYS)
        for r in runs:
            for k in r:
                if k not in keys and isinstance(r[k], (int, float)):
                    keys.append(k)
        for key in keys:
            if key == "seed":
                continue
            vals = [float(r[key]) for r in runs if key in r]
            if not vals:
                continue
            mean, std, half = _ci(vals)
            summary["metrics"][sched][key] = {
                "mean": mean,
                "std": std,
                "mean_pm_std": f"{mean:.4f} ± {std:.4f}",
                "ci95_halfwidth": half,
                "ci95": [mean - half, mean + half],
                "values": vals,
            }

    if baseline_for_test in per_seed:
        base_vals = [float(r.get(metric, 0.0)) for r in per_seed[baseline_for_test]]
        for sched, runs in per_seed.items():
            if sched == baseline_for_test:
                continue
            vals = [float(r.get(metric, 0.0)) for r in runs]
            n = min(len(vals), len(base_vals))
            if n < 2:
                continue
            entry: dict = {
                "vs": baseline_for_test,
                "metric": metric,
                "n": n,
            }
            a = np.asarray(vals[:n], dtype=float)
            b = np.asarray(base_vals[:n], dtype=float)
            if np.allclose(a, b):
                entry["wilcoxon"] = {"statistic": 0.0, "p_value": 1.0}
                entry["ttest_rel"] = {"statistic": 0.0, "p_value": 1.0}
            else:
                try:
                    w_stat, w_p = stats.wilcoxon(a, b, alternative="two-sided")
                    entry["wilcoxon"] = {
                        "statistic": float(w_stat),
                        "p_value": float(w_p) if np.isfinite(w_p) else None,
                    }
                except ValueError:
                    entry["wilcoxon"] = {"statistic": None, "p_value": None}
                with np.errstate(divide="ignore", invalid="ignore"):
                    t_stat, t_p = stats.ttest_rel(a, b)
                entry["ttest_rel"] = {
                    "statistic": float(t_stat) if np.isfinite(t_stat) else None,
                    "p_value": float(t_p) if np.isfinite(t_p) else None,
                }
            # primary p-value: Wilcoxon if available else t-test
            primary = entry["wilcoxon"]["p_value"]
            if primary is None:
                primary = entry["ttest_rel"]["p_value"]
            entry["p_value"] = primary
            entry["significant_at_0.01"] = bool(primary is not None and primary < 0.01)
            entry["significant_at_0.05"] = bool(primary is not None and primary < 0.05)
            summary["significance"][sched] = entry
    return summary


# ---------- plots ---------------------------------------------------
def plot_bars(summary: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_block = summary.get("metrics", {})
    if not metrics_block:
        return
    schedulers = list(metrics_block.keys())
    for metric in [
        "avg_slowdown", "p95_slowdown", "p99_slowdown",
        "avg_completion", "n_done", "throughput",
        "fragmentation", "sla_breach_rate",
    ]:
        if not all(metric in metrics_block[s] for s in schedulers):
            continue
        means = [metrics_block[s][metric]["mean"] for s in schedulers]
        errs = [metrics_block[s][metric]["ci95_halfwidth"] for s in schedulers]
        plt.figure(figsize=(9, 4.5))
        bars = plt.bar(
            schedulers, means, yerr=errs, capsize=4,
            color="steelblue", edgecolor="black", ecolor="black",
        )
        plt.ylabel(metric)
        plt.title(f"{metric} across schedulers (mean ± 95% CI)")
        plt.xticks(rotation=20)
        for b, v in zip(bars, means):
            plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                     ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        out_path = out_dir / f"{metric}.png"
        plt.savefig(out_path, dpi=120)
        plt.close()
        print(f"[plot] wrote {out_path}")


def plot_learning_curve(history_path: Path, out_dir: Path, name: str) -> None:
    if not history_path.exists():
        return
    with open(history_path) as f:
        h = json.load(f)
    if not h.get("update"):
        return
    plt.figure(figsize=(8, 4))
    x = h.get("transitions") or h["update"]
    plt.plot(x, h["mean_reward"], marker="o", linewidth=1.2, markersize=3)
    plt.xlabel("environment transitions" if "transitions" in h else "PPO update")
    plt.ylabel("mean episode reward")
    plt.title(f"PPO learning curve — {name}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = out_dir / f"learning_curve_{name}.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[plot] wrote {out}")


def plot_reap_quality(reap_metrics_path: Path, out_dir: Path) -> None:
    if not reap_metrics_path.exists():
        return
    with open(reap_metrics_path) as f:
        d = json.load(f)
    metrics = d["metrics"]
    names = list(metrics.keys())
    mses = [metrics[n]["mse"] for n in names]
    maes = [metrics[n]["mae"] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(names, mses, color="indianred", edgecolor="black")
    axes[0].set_title(f"REAP MSE — {d['target']}")
    axes[0].set_ylabel("MSE")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(names, maes, color="seagreen", edgecolor="black")
    axes[1].set_title(f"REAP MAE — {d['target']}")
    axes[1].set_ylabel("MAE")
    axes[1].tick_params(axis="x", rotation=20)
    plt.tight_layout()
    out = out_dir / f"reap_quality_{d['target']}.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[plot] wrote {out}")


def plot_reap_weights(reap_metrics_path: Path, out_dir: Path) -> None:
    if not reap_metrics_path.exists():
        return
    with open(reap_metrics_path) as f:
        d = json.load(f)
    w = d["weights"]
    plt.figure(figsize=(7, 4))
    plt.bar(list(w.keys()), list(w.values()), color="slateblue", edgecolor="black")
    scheme = d.get("scheme", "inv_mse")
    tau = d.get("temperature", None)
    title = f"REAP ensemble weights — {d['target']} ({scheme}"
    if tau is not None and scheme in ("softmax", "topk"):
        title += f", τ={tau}"
    title += ")"
    plt.title(title)
    plt.ylabel("weight")
    plt.xticks(rotation=20)
    plt.tight_layout()
    out = out_dir / f"reap_weights_{d['target']}.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[plot] wrote {out}")


# ---------- main ----------------------------------------------------
def run_benchmark(
    job_trace: str | None,
    seeds: list[int],
    max_steps: int,
    episode_max_steps: int,
    deeprm_path: str,
    deepreap_path: str,
    trace_source: str = "canonical",
    include_ilp: bool = True,
    forecast_mode: str = "proxy",
    deeprm_imi_path: str = "models/deeprm/deeprm_plus_imitation.pt",
    deepreap_imi_path: str = "models/deeprm/deepreap_imitation.pt",
) -> tuple[dict, dict]:
    """Run all schedulers across seeds; return (per_seed_raw, summary)."""
    per_seed: dict[str, list[dict]] = {}
    heuristics = ["fifo", "sjf", "packer", "drf"]
    if include_ilp:
        heuristics.append("ilp")

    for seed in seeds:
        trace = _load_trace(job_trace, trace_source, seed=seed)
        # Per-seed windowing on large real dumps → non-degenerate CIs.
        if trace is not None and len(trace) > 4000:
            rng = np.random.default_rng(seed)
            start = int(rng.integers(0, max(1, len(trace) - 4000)))
            trace = trace.iloc[start : start + 4000].reset_index(drop=True)
            # Re-base arrivals so the window starts at t=0.
            trace = trace.copy()
            trace["arrival_time"] = trace["arrival_time"] - int(trace["arrival_time"].min())
        print(f"[bench] seed={seed}  trace_rows={0 if trace is None else len(trace)}")
        util_timeline = None
        if forecast_mode == "oracle" and trace is not None and len(trace) > 0:
            # Cap oracle build cost on large real dumps.
            util_timeline = build_offline_utilization(trace.head(min(len(trace), 4000)))

        for name in heuristics:
            env = _make_env(trace, reap_channels=0, seed=seed, episode_max_steps=episode_max_steps)
            m = run_baseline(env, name, max_steps=max_steps)
            m["seed"] = seed
            per_seed.setdefault(name, []).append(m)
            print(
                f"[bench]   {name:12s}  slow={m.get('avg_slowdown', 0):.2f}  "
                f"p95={m.get('p95_slowdown', 0):.2f}  n_done={m.get('n_done', 0)}  "
                f"thru={m.get('throughput', 0):.3f}  "
                f"frag={m.get('fragmentation', 0):.3f}  sla={m.get('sla_breach_rate', 0):.3f}"
            )

        for label, ckpt_path, reap_ch in [
            ("deeprm_plus_imitation", deeprm_imi_path, 0),
            ("deeprm_plus", deeprm_path, 0),
            ("deepreap_imitation", deepreap_imi_path, 2),
            ("deepreap", deepreap_path, 2),
        ]:
            p = Path(ckpt_path)
            if not p.exists():
                continue
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            reap_ch_actual = ckpt["cfg"].get("reap_channels", reap_ch)
            env = _make_env(
                trace, reap_channels=reap_ch_actual, seed=seed,
                episode_max_steps=episode_max_steps,
                forecast_mode=forecast_mode,
                util_timeline=util_timeline,
            )
            policy, _ = _load_policy(p)
            m = run_policy(env, policy, max_steps=max_steps, profile_latency=True)
            m["seed"] = seed
            per_seed.setdefault(label, []).append(m)
            print(
                f"[bench]   {label:24s}  slow={m.get('avg_slowdown', 0):.2f}  "
                f"p95={m.get('p95_slowdown', 0):.2f}  n_done={m.get('n_done', 0)}  "
                f"thru={m.get('throughput', 0):.3f}  "
                f"lat_ms={m.get('cnn_latency_mean_ms', 0):.3f}"
            )

    summary = summarize_results(per_seed, baseline_for_test="sjf", metric="avg_slowdown")
    return per_seed, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-trace", default="data/job_trace.csv")
    ap.add_argument(
        "--trace-source",
        default="canonical",
        choices=["canonical", "synthetic", "google_cluster", "alibaba", "azure_vm"],
        help="How to interpret --job-trace (use google_cluster/alibaba/azure_vm for raw dumps)",
    )
    ap.add_argument("--results", default="results")
    ap.add_argument("--seed", type=int, default=123, help="base seed (used if --n-seeds=1)")
    ap.add_argument(
        "--n-seeds", type=int, default=5,
        help="number of evaluation seeds (n>=5; use 10+ for camera-ready)",
    )
    ap.add_argument(
        "--seeds", type=str, default="",
        help="comma-separated explicit seeds (overrides --n-seeds)",
    )
    ap.add_argument("--max-steps", type=int, default=8000,
                    help="evaluation step budget (longer = fairer throughput)")
    ap.add_argument("--episode-max-steps", type=int, default=2000)
    ap.add_argument("--no-ilp", action="store_true", help="skip short-horizon ILP baseline")
    ap.add_argument("--deeprm-path", default="models/deeprm/deeprm_plus.pt")
    ap.add_argument("--deepreap-path", default="models/deeprm/deepreap.pt")
    ap.add_argument("--deeprm-imi-path", default="models/deeprm/deeprm_plus_imitation.pt")
    ap.add_argument("--deepreap-imi-path", default="models/deeprm/deepreap_imitation.pt")
    ap.add_argument(
        "--forecast-mode", default="proxy",
        choices=["proxy", "oracle", "zero"],
        help="Forecast channels for DeepREAP policies at eval time",
    )
    ap.add_argument("--reap-cpu-metrics", default="models/reap/reap_cpu_load_metrics.json")
    ap.add_argument("--reap-mem-metrics", default="models/reap/reap_memory_usage_metrics.json")
    args = ap.parse_args()

    if args.seeds.strip():
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = [args.seed + i for i in range(args.n_seeds)]

    out = Path(args.results)
    plots = out / "plots"
    out.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)

    per_seed, summary = run_benchmark(
        job_trace=args.job_trace or None,
        seeds=seeds,
        max_steps=args.max_steps,
        episode_max_steps=args.episode_max_steps,
        deeprm_path=args.deeprm_path,
        deepreap_path=args.deepreap_path,
        trace_source=args.trace_source,
        include_ilp=not args.no_ilp,
        forecast_mode=args.forecast_mode,
        deeprm_imi_path=args.deeprm_imi_path,
        deepreap_imi_path=args.deepreap_imi_path,
    )

    # serialize raw + summary
    # keep a compact single-seed view for backward compat (first seed means)
    compact = {
        sched: {k: v["mean"] for k, v in summary["metrics"][sched].items()}
        for sched in summary["metrics"]
    }
    with open(out / "benchmark.json", "w") as f:
        json.dump({"aggregate": compact, "per_seed": per_seed, "seeds": seeds}, f, indent=2)
    with open(out / "benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[bench] wrote {out / 'benchmark.json'}")
    print(f"[bench] wrote {out / 'benchmark_summary.json'}")
    for sched, sig in summary.get("significance", {}).items():
        p = sig.get("p_value")
        p_str = f"{p:.4g}" if p is not None else "n/a"
        print(
            f"[bench] {sched} vs {sig['vs']} on {sig['metric']}: "
            f"p={p_str} (wilcoxon/ttest, n={sig['n']}, "
            f"sig@0.01={sig.get('significant_at_0.01')})"
        )

    # plots
    plot_bars(summary, plots)
    plot_learning_curve(Path("models/deeprm/deeprm_plus_history.json"), plots, "deeprm_plus")
    plot_learning_curve(Path("models/deeprm/deepreap_history.json"), plots, "deepreap")
    plot_reap_quality(Path(args.reap_cpu_metrics), plots)
    plot_reap_quality(Path(args.reap_mem_metrics), plots)
    plot_reap_weights(Path(args.reap_cpu_metrics), plots)
    plot_reap_weights(Path(args.reap_mem_metrics), plots)


if __name__ == "__main__":
    main()
