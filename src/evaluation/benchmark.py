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
from src.deeprm.network import CNNPolicy


# ---------- scheduler runners ---------------------------------------
def run_policy(env: ClusterEnv, policy: CNNPolicy, max_steps: int = 5000) -> dict:
    s = env.reset()
    total = 0.0
    steps = 0
    policy.eval()
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


def _make_env(
    trace: pd.DataFrame | None,
    reap_channels: int,
    seed: int,
    episode_max_steps: int = 2000,
) -> ClusterEnv:
    cfg = ClusterConfig(reap_channels=reap_channels, episode_max_steps=episode_max_steps)
    return ClusterEnv(cfg=cfg, job_trace=trace, seed=seed)


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


def summarize_results(
    per_seed: dict[str, list[dict]],
    baseline_for_test: str = "sjf",
    metric: str = "avg_slowdown",
) -> dict:
    """
    Aggregate multi-seed runs into mean/std/CI and Wilcoxon p-values
    vs a non-predictive baseline (default SJF).
    """
    summary: dict = {"n_seeds": {}, "metrics": {}, "significance": {}}
    # collect metric arrays
    metric_keys = ["avg_slowdown", "avg_completion", "n_done", "throughput", "total_reward"]
    for sched, runs in per_seed.items():
        summary["n_seeds"][sched] = len(runs)
        summary["metrics"][sched] = {}
        for key in metric_keys:
            vals = [float(r.get(key, 0.0)) for r in runs]
            mean, std, half = _ci(vals)
            summary["metrics"][sched][key] = {
                "mean": mean,
                "std": std,
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
            # Wilcoxon signed-rank (paired across seeds); fall back to t-test
            try:
                stat, p = stats.wilcoxon(vals[:n], base_vals[:n], alternative="two-sided")
                test = "wilcoxon"
            except ValueError:
                stat, p = stats.ttest_rel(vals[:n], base_vals[:n])
                test = "ttest_rel"
            summary["significance"][sched] = {
                "vs": baseline_for_test,
                "metric": metric,
                "test": test,
                "statistic": float(stat),
                "p_value": float(p),
                "n": n,
            }
    return summary


# ---------- plots ---------------------------------------------------
def plot_bars(summary: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_block = summary.get("metrics", {})
    if not metrics_block:
        return
    schedulers = list(metrics_block.keys())
    for metric in ["avg_slowdown", "avg_completion", "n_done", "throughput"]:
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
) -> tuple[dict, dict]:
    """Run all schedulers across seeds; return (per_seed_raw, summary)."""
    per_seed: dict[str, list[dict]] = {}

    for seed in seeds:
        trace = _load_trace(job_trace, trace_source, seed=seed)
        print(f"[bench] seed={seed}  trace_rows={0 if trace is None else len(trace)}")

        for name in ["fifo", "sjf", "packer"]:
            env = _make_env(trace, reap_channels=0, seed=seed, episode_max_steps=episode_max_steps)
            m = run_baseline(env, name, max_steps=max_steps)
            m["seed"] = seed
            per_seed.setdefault(name, []).append(m)
            print(f"[bench]   {name:12s}  {m}")

        for label, ckpt_path, reap_ch in [
            ("deeprm_plus_imitation", "models/deeprm/deeprm_plus_imitation.pt", 0),
            ("deeprm_plus", deeprm_path, 0),
            ("deepreap_imitation", "models/deeprm/deepreap_imitation.pt", 2),
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
            )
            policy, _ = _load_policy(p)
            m = run_policy(env, policy, max_steps=max_steps)
            m["seed"] = seed
            per_seed.setdefault(label, []).append(m)
            print(f"[bench]   {label:24s}  {m}")

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
        "--n-seeds", type=int, default=10,
        help="number of evaluation seeds (n≥10 for statistical confidence)",
    )
    ap.add_argument(
        "--seeds", type=str, default="",
        help="comma-separated explicit seeds (overrides --n-seeds)",
    )
    ap.add_argument("--max-steps", type=int, default=8000,
                    help="evaluation step budget (longer = fairer throughput)")
    ap.add_argument("--episode-max-steps", type=int, default=2000)
    ap.add_argument("--deeprm-path", default="models/deeprm/deeprm_plus.pt")
    ap.add_argument("--deepreap-path", default="models/deeprm/deepreap.pt")
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
        print(
            f"[bench] {sched} vs {sig['vs']} on {sig['metric']}: "
            f"p={sig['p_value']:.4g} ({sig['test']}, n={sig['n']})"
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
