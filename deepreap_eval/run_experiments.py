#!/usr/bin/env python3
"""
CMPE-294 streamlined evaluation protocol (Phases A / B / C).

Produces the two claim tables from the original paper methodology:

  1. Demand Prediction Performance  (Phase A)
  2. Overall Scheduling Efficiency  (Phase B)
  plus a Phase-C noise-robustness sweep.

Usage
-----
  PYTHONPATH=. python3 -m deepreap_eval.run_experiments
  PYTHONPATH=. python3 -m deepreap_eval.run_experiments --n-episodes 30 --ppo-updates 100
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from deepreap_eval import deeprm_plus, reap
from src.deeprm.env import ClusterConfig
from src.deeprm.forecasts import build_offline_utilization
from src.evaluation.benchmark import summarize_results
from src.reap.ensemble import REAPModel

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- helpers
def _mean_std(vals: list[float]) -> dict:
    arr = np.asarray(vals, dtype=float)
    if len(arr) == 0:
        return {"mean": float("nan"), "std": float("nan"), "values": []}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "values": vals,
    }


def _scale_trace_load(trace: pd.DataFrame, load: str, seed: int) -> pd.DataFrame:
    """
    Create low / medium / high-burst arrival intensity by rescaling arrival
    times (and injecting periodic spikes for high load). Identical job
    sizes; only the arrival process changes.
    """
    df = trace.copy().reset_index(drop=True)
    arr = df["arrival_time"].to_numpy(dtype=float)
    arr = arr - arr.min()
    # denser arrivals → higher load
    scale = {"low": 2.2, "medium": 1.0, "high": 0.55}[load]
    arr = arr * scale
    if load == "high":
        # Periodic bursts: compress every 5th window of 200 jobs into a spike
        rng = np.random.default_rng(seed)
        n = len(arr)
        for start in range(0, n, 200):
            end = min(n, start + 200)
            if (start // 200) % 5 == 0:
                base = arr[start]
                arr[start:end] = base + rng.uniform(0, 8, size=end - start)
        arr = np.sort(arr)
    df["arrival_time"] = arr.astype(int)
    return df


def _window_trace(full: pd.DataFrame, seed: int, n_jobs: int = 2500) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if len(full) <= n_jobs:
        out = full.copy()
    else:
        start = int(rng.integers(0, max(1, len(full) - n_jobs)))
        out = full.iloc[start : start + n_jobs].reset_index(drop=True).copy()
    out["arrival_time"] = out["arrival_time"] - int(out["arrival_time"].min())
    return out


def _fmt(x: float | None, pct: bool = False) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    if pct:
        return f"{100.0 * x:.2f}"
    return f"{x:.4f}"


# --------------------------------------------------------------------------- Phase A
def phase_a(args: argparse.Namespace, out: Path) -> dict:
    print("\n========== PHASE A: REAP prediction accuracy ==========", flush=True)
    usage = Path(args.usage_csv)
    model_dir = Path(args.model_root) / "reap"
    t0 = time.time()
    cpu = reap.train_reap_inv_mse(
        usage, model_dir, target="cpu_load", seed=args.seed,
        ga_pop=args.ga_pop, ga_gen=args.ga_gen,
    )
    mem = reap.train_reap_inv_mse(
        usage, model_dir, target="memory_usage", seed=args.seed,
        ga_pop=args.ga_pop, ga_gen=args.ga_gen,
    )
    cpu_m = json.loads((model_dir / "reap_cpu_load_metrics.json").read_text())
    mem_m = json.loads((model_dir / "reap_memory_usage_metrics.json").read_text())
    rows = reap.prediction_table(cpu_m, mem_m)
    # Prefer paper table order
    order = [
        "LinearRegression", "SVR", "RandomForest",
        "BayesianRidge", "DecisionTree", "REAP Ensemble",
    ]
    rows = sorted(rows, key=lambda r: order.index(r["model"]) if r["model"] in order else 99)

    claim = {
        "phase": "A",
        "elapsed_s": time.time() - t0,
        "usage_csv": str(usage),
        "table": rows,
        "ensemble_best_cpu_mae": (
            rows[-1]["cpu_mae"] <= min(
                r["cpu_mae"] for r in rows if r["model"] != "REAP Ensemble" and r["cpu_mae"] is not None
            )
            if any(r["model"] == "REAP Ensemble" for r in rows) else False
        ),
    }
    # Honest check: ensemble MAE vs best single model
    ens = next(r for r in rows if r["model"] == "REAP Ensemble")
    singles = [r for r in rows if r["model"] != "REAP Ensemble"]
    best_cpu = min(singles, key=lambda r: r["cpu_mae"] if r["cpu_mae"] is not None else 1e9)
    best_mem = min(singles, key=lambda r: r["memory_mae"] if r["memory_mae"] is not None else 1e9)
    claim["vs_best_single"] = {
        "best_cpu_model": best_cpu["model"],
        "best_cpu_mae": best_cpu["cpu_mae"],
        "ensemble_cpu_mae": ens["cpu_mae"],
        "cpu_ensemble_wins": ens["cpu_mae"] <= best_cpu["cpu_mae"],
        "best_memory_model": best_mem["model"],
        "best_memory_mae": best_mem["memory_mae"],
        "ensemble_memory_mae": ens["memory_mae"],
        "memory_ensemble_wins": ens["memory_mae"] <= best_mem["memory_mae"],
    }
    (out / "phase_a.json").write_text(json.dumps(claim, indent=2))
    print(json.dumps(claim["vs_best_single"], indent=2), flush=True)
    return {"cpu": cpu, "mem": mem, "report": claim}


# --------------------------------------------------------------------------- Phase B
def phase_b(
    args: argparse.Namespace,
    out: Path,
    cpu_model: REAPModel,
    mem_model: REAPModel,
) -> dict:
    print("\n========== PHASE B: scheduling efficiency ==========", flush=True)
    model_root = Path(args.model_root)
    job_path = Path(args.job_trace)
    full = pd.read_csv(job_path)

    # --- train policies (identical budget; only forecast channels differ) ---
    policies = {
        "vanilla": {
            "channels": 0, "mode": "zero", "tag": "vanilla",
            "out": model_root / "vanilla",
        },
        "deepreap": {
            "channels": 2, "mode": "reap", "tag": "deepreap",
            "out": model_root / "deepreap",
        },
        "oracle": {
            "channels": 2, "mode": "oracle", "tag": "oracle",
            "out": model_root / "oracle",
        },
    }
    ckpts: dict[str, Path] = {}
    if not args.skip_train:
        for name, spec in policies.items():
            print(f"[B] training {name} …", flush=True)
            ckpts[name] = deeprm_plus.train_policy(
                job_path,
                spec["out"],
                channels=spec["channels"],
                forecast_mode=spec["mode"],
                cpu_model=cpu_model if name == "deepreap" else None,
                mem_model=mem_model if name == "deepreap" else None,
                imitation_episodes=args.imitation_episodes,
                ppo_updates=args.ppo_updates,
                max_jobs=args.max_train_jobs,
                seed=args.seed,
                device=args.device,
                tag=spec["tag"],
            )
    else:
        for name, spec in policies.items():
            cand = list(spec["out"].glob("*.pt"))
            cand = [p for p in cand if "imitation" not in p.name]
            if not cand:
                raise FileNotFoundError(f"missing checkpoint under {spec['out']}")
            ckpts[name] = cand[0]

    loads = ["low", "medium", "high"]
    seeds = list(range(args.seed, args.seed + args.n_episodes))
    per_load: dict[str, dict] = {}
    aggregate_runs: dict[str, list[dict]] = {
        "sjf": [], "vanilla": [], "deepreap": [], "oracle": [],
    }

    for load in loads:
        print(f"[B] evaluating load={load} over {len(seeds)} episodes …", flush=True)
        load_runs: dict[str, list[dict]] = {k: [] for k in aggregate_runs}
        for i, seed in enumerate(seeds):
            base = _window_trace(full, seed=seed, n_jobs=args.eval_jobs)
            trace = _scale_trace_load(base, load=load, seed=seed)
            cfg = ClusterConfig(reap_channels=2, episode_max_steps=args.episode_max_steps)
            util = build_offline_utilization(trace, cfg)
            reap_fn = deeprm_plus._reap_predict_fn(cpu_model, mem_model, cfg)

            # SJF
            m = deeprm_plus.run_sjf(trace, seed=seed, max_steps=args.max_steps)
            m["seed"] = seed
            m["load"] = load
            load_runs["sjf"].append(m)

            for name, spec in policies.items():
                mm = deeprm_plus.run_policy_metrics(
                    ckpts[name],
                    trace,
                    channels=spec["channels"],
                    mode=spec["mode"],
                    seed=seed,
                    max_steps=args.max_steps,
                    episode_max_steps=args.episode_max_steps,
                    util_timeline=util if spec["mode"] == "oracle" else None,
                    reap_fn=reap_fn if name == "deepreap" else None,
                    noise_pct=0.0,
                )
                mm["seed"] = seed
                mm["load"] = load
                load_runs[name].append(mm)

            if (i + 1) % 5 == 0 or i == 0:
                print(
                    f"  [{load}] ep {i+1}/{len(seeds)}  "
                    f"sjf_tat={load_runs['sjf'][-1].get('avg_completion', 0):.2f}  "
                    f"van={load_runs['vanilla'][-1].get('avg_completion', 0):.2f}  "
                    f"reap={load_runs['deepreap'][-1].get('avg_completion', 0):.2f}  "
                    f"orc={load_runs['oracle'][-1].get('avg_completion', 0):.2f}",
                    flush=True,
                )

        summary = summarize_results(load_runs, baseline_for_test="sjf", metric="avg_completion")
        per_load[load] = {"summary": summary, "n": len(seeds)}
        for k, runs in load_runs.items():
            aggregate_runs[k].extend(runs)

    # Overall table across loads (paper Table 2)
    overall = summarize_results(aggregate_runs, baseline_for_test="sjf", metric="avg_completion")
    table = []
    for name, label in [
        ("sjf", "Shortest Job First (SJF)"),
        ("vanilla", "Vanilla DeepRM_Plus"),
        ("deepreap", "DeepREAP (Proposed)"),
        ("oracle", "Oracle DRL"),
    ]:
        m = overall["metrics"].get(name, {})
        table.append({
            "policy": label,
            "avg_turnaround_s": m.get("avg_completion", {}).get("mean"),
            "avg_turnaround_std": m.get("avg_completion", {}).get("std"),
            "avg_wait_s": m.get("avg_wait", {}).get("mean"),
            "avg_wait_std": m.get("avg_wait", {}).get("std"),
            "resource_utilization_pct": (
                100.0 * m["avg_utilization"]["mean"]
                if "avg_utilization" in m else None
            ),
            "avg_slowdown": m.get("avg_slowdown", {}).get("mean"),
            "throughput": m.get("throughput", {}).get("mean"),
        })

    # Pre-registered claim checks (CMPE-294 core)
    def _vals(arm: str, key: str) -> list[float]:
        return [float(r.get(key, np.nan)) for r in aggregate_runs[arm]]

    def _paired(a: str, b: str, key: str, alternative: str = "less") -> dict:
        va, vb = _vals(a, key), _vals(b, key)
        n = min(len(va), len(vb))
        aa, bb = np.asarray(va[:n]), np.asarray(vb[:n])
        if n < 2 or np.allclose(aa, bb):
            return {"p_value": 1.0, "n": n, "mean_a": float(aa.mean()) if n else None,
                    "mean_b": float(bb.mean()) if n else None}
        try:
            # Wilcoxon: alternative='less' means a < b
            if alternative == "less":
                stat, p = stats.wilcoxon(aa, bb, alternative="less")
            elif alternative == "greater":
                stat, p = stats.wilcoxon(aa, bb, alternative="greater")
            else:
                stat, p = stats.wilcoxon(aa, bb, alternative="two-sided")
            return {
                "statistic": float(stat), "p_value": float(p), "n": n,
                "mean_a": float(aa.mean()), "mean_b": float(bb.mean()),
            }
        except ValueError:
            return {"p_value": None, "n": n}

    claims = {
        "oracle_beats_vanilla_turnaround": _paired("oracle", "vanilla", "avg_completion", "less"),
        "deepreap_beats_vanilla_turnaround": _paired("deepreap", "vanilla", "avg_completion", "less"),
        "deepreap_beats_sjf_turnaround": _paired("deepreap", "sjf", "avg_completion", "less"),
        "deepreap_beats_vanilla_wait": _paired("deepreap", "vanilla", "avg_wait", "less"),
        "deepreap_util_vs_vanilla": _paired("deepreap", "vanilla", "avg_utilization", "greater"),
    }
    # Pass if oracle (upper bound) and DeepREAP both beat vanilla on turnaround
    claims["scheduling_claim_supported"] = bool(
        (claims["oracle_beats_vanilla_turnaround"].get("p_value") or 1) < 0.05
        and (claims["deepreap_beats_vanilla_turnaround"].get("p_value") or 1) < 0.05
    )

    report = {
        "phase": "B",
        "n_episodes_per_load": args.n_episodes,
        "loads": loads,
        "ckpts": {k: str(v) for k, v in ckpts.items()},
        "table": table,
        "per_load": {
            load: {
                k: {
                    "avg_completion": v["metrics"].get(k, {}).get("avg_completion", {}).get("mean"),
                    "avg_wait": v["metrics"].get(k, {}).get("avg_wait", {}).get("mean"),
                    "avg_utilization": v["metrics"].get(k, {}).get("avg_utilization", {}).get("mean"),
                }
                for k in ("sjf", "vanilla", "deepreap", "oracle")
            }
            for load, v in (
                (ld, per_load[ld]["summary"]) for ld in loads
            )
        },
        "claims": claims,
        "overall_summary": overall,
    }
    (out / "phase_b.json").write_text(json.dumps(report, indent=2, default=str))
    print("[B] scheduling claim supported:", claims["scheduling_claim_supported"], flush=True)
    for row in table:
        print(
            f"  {row['policy']:28s}  TAT={_fmt(row['avg_turnaround_s'])}  "
            f"wait={_fmt(row['avg_wait_s'])}  util%={_fmt(row['resource_utilization_pct'])}",
            flush=True,
        )
    return report


# --------------------------------------------------------------------------- Phase C
def phase_c(
    args: argparse.Namespace,
    out: Path,
    cpu_model: REAPModel,
    mem_model: REAPModel,
    phase_b_report: dict,
) -> dict:
    print("\n========== PHASE C: noise robustness sweep ==========", flush=True)
    ckpt = Path(phase_b_report["ckpts"]["deepreap"])
    vanilla_ckpt = Path(phase_b_report["ckpts"]["vanilla"])
    full = pd.read_csv(args.job_trace)
    seeds = list(range(args.seed, args.seed + max(10, args.n_episodes // 2)))
    sigmas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    sweep = []

    # Vanilla reference (no forecast, noise irrelevant)
    van_tats = []
    for seed in seeds:
        trace = _scale_trace_load(
            _window_trace(full, seed=seed, n_jobs=args.eval_jobs),
            load="medium", seed=seed,
        )
        m = deeprm_plus.run_policy_metrics(
            vanilla_ckpt, trace, channels=0, mode="zero", seed=seed,
            max_steps=args.max_steps, episode_max_steps=args.episode_max_steps,
        )
        van_tats.append(float(m.get("avg_completion", np.nan)))
    vanilla_ref = _mean_std(van_tats)

    for sigma in sigmas:
        tats, waits, utils = [], [], []
        for seed in seeds:
            trace = _scale_trace_load(
                _window_trace(full, seed=seed, n_jobs=args.eval_jobs),
                load="medium", seed=seed,
            )
            cfg = ClusterConfig(reap_channels=2, episode_max_steps=args.episode_max_steps)
            reap_fn = deeprm_plus._reap_predict_fn(cpu_model, mem_model, cfg)
            m = deeprm_plus.run_policy_metrics(
                ckpt, trace, channels=2, mode="reap", seed=seed,
                max_steps=args.max_steps, episode_max_steps=args.episode_max_steps,
                reap_fn=reap_fn, noise_pct=sigma, noise_mode="gaussian",
            )
            tats.append(float(m.get("avg_completion", np.nan)))
            waits.append(float(m.get("avg_wait", np.nan)))
            utils.append(float(m.get("avg_utilization", np.nan)))
        entry = {
            "sigma": sigma,
            "sigma_pct": int(round(100 * sigma)),
            "avg_turnaround": _mean_std(tats),
            "avg_wait": _mean_std(waits),
            "avg_utilization": _mean_std(utils),
            "beats_vanilla_mean": float(np.nanmean(tats)) < vanilla_ref["mean"],
        }
        # paired test vs vanilla
        n = min(len(tats), len(van_tats))
        if n >= 2 and not np.allclose(tats[:n], van_tats[:n]):
            try:
                _, p = stats.wilcoxon(
                    np.asarray(tats[:n]), np.asarray(van_tats[:n]), alternative="less"
                )
                entry["wilcoxon_p_vs_vanilla"] = float(p)
            except ValueError:
                entry["wilcoxon_p_vs_vanilla"] = None
        else:
            entry["wilcoxon_p_vs_vanilla"] = 1.0
        sweep.append(entry)
        print(
            f"  σ={entry['sigma_pct']:2d}%  TAT={entry['avg_turnaround']['mean']:.3f}  "
            f"vs_vanilla={vanilla_ref['mean']:.3f}  "
            f"beats_mean={entry['beats_vanilla_mean']}  "
            f"p={entry.get('wilcoxon_p_vs_vanilla')}",
            flush=True,
        )

    graceful = all(
        e["avg_turnaround"]["mean"] <= sweep[0]["avg_turnaround"]["mean"] * 1.35
        for e in sweep
    )
    still_better_at_30 = bool(sweep[-1]["beats_vanilla_mean"])
    report = {
        "phase": "C",
        "vanilla_turnaround_ref": vanilla_ref,
        "sweep": sweep,
        "graceful_degradation": graceful,
        "beats_vanilla_at_30pct_noise": still_better_at_30,
        "robustness_claim_supported": graceful and still_better_at_30,
    }
    (out / "phase_c.json").write_text(json.dumps(report, indent=2))
    print("[C] robustness claim supported:", report["robustness_claim_supported"], flush=True)
    return report


# --------------------------------------------------------------------------- tables
def write_claim_tables(out: Path, a: dict, b: dict, c: dict) -> Path:
    lines = [
        "# CMPE-294 DeepREAP Claim Tables",
        "",
        "## 1. Demand Prediction Performance",
        "",
        "| Model | CPU MAE | Memory MAE | Overall MSE |",
        "| --- | --- | --- | --- |",
    ]
    for row in a["report"]["table"]:
        name = row["model"]
        if name == "REAP Ensemble":
            name = f"**{name}**"
        lines.append(
            f"| {name} | {_fmt(row['cpu_mae'])} | {_fmt(row['memory_mae'])} | "
            f"{_fmt(row['overall_mse'])} |"
        )
    lines += [
        "",
        f"- Ensemble wins CPU MAE vs best single: "
        f"**{a['report']['vs_best_single']['cpu_ensemble_wins']}**",
        f"- Ensemble wins Memory MAE vs best single: "
        f"**{a['report']['vs_best_single']['memory_ensemble_wins']}**",
        "",
        "## 2. Overall Scheduling Efficiency",
        "",
        "| Policy / System | Avg Turnaround Time (s) | Avg Wait Time (s) | Resource Utilization (%) |",
        "| --- | --- | --- | --- |",
    ]
    for row in b["table"]:
        name = row["policy"]
        if "DeepREAP" in name:
            name = f"**{name}**"
        lines.append(
            f"| {name} | {_fmt(row['avg_turnaround_s'])} | {_fmt(row['avg_wait_s'])} | "
            f"{_fmt(row['resource_utilization_pct'])} |"
        )
    lines += [
        "",
        f"- Scheduling claim supported (DeepREAP & Oracle beat Vanilla on TAT, p<0.05): "
        f"**{b['claims']['scheduling_claim_supported']}**",
        f"- Oracle vs Vanilla TAT p="
        f"{b['claims']['oracle_beats_vanilla_turnaround'].get('p_value')}",
        f"- DeepREAP vs Vanilla TAT p="
        f"{b['claims']['deepreap_beats_vanilla_turnaround'].get('p_value')}",
        f"- DeepREAP vs SJF TAT p="
        f"{b['claims']['deepreap_beats_sjf_turnaround'].get('p_value')}",
        "",
        "## 3. Phase C — Noise Robustness (Gaussian σ on REAP channels)",
        "",
        "| σ (%) | Avg Turnaround | Beats Vanilla (mean) | Wilcoxon p (less) |",
        "| --- | --- | --- | --- |",
    ]
    for e in c["sweep"]:
        lines.append(
            f"| {e['sigma_pct']} | {_fmt(e['avg_turnaround']['mean'])} | "
            f"{e['beats_vanilla_mean']} | {_fmt(e.get('wilcoxon_p_vs_vanilla'))} |"
        )
    lines += [
        "",
        f"- Robustness claim supported: **{c['robustness_claim_supported']}**",
        "",
    ]
    md = out / "CLAIM_TABLES.md"
    md.write_text("\n".join(lines))
    print(f"\nWrote {md}", flush=True)
    return md


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="CMPE-294 DeepREAP Phases A/B/C")
    ap.add_argument("--usage-csv", default="data/real/resource_usage_google.csv")
    ap.add_argument("--job-trace", default="data/real/google2011_jobs.csv")
    ap.add_argument("--out", default="results/cmpe294_eval")
    ap.add_argument("--model-root", default="models/cmpe294_eval")
    ap.add_argument("--n-episodes", type=int, default=30,
                    help="Episodes per load condition (Phase B). Paper suggests 100.")
    ap.add_argument("--ppo-updates", type=int, default=100)
    ap.add_argument("--imitation-episodes", type=int, default=30)
    ap.add_argument("--max-train-jobs", type=int, default=4000)
    ap.add_argument("--eval-jobs", type=int, default=2500)
    ap.add_argument("--episode-max-steps", type=int, default=1500)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--ga-pop", type=int, default=20)
    ap.add_argument("--ga-gen", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-a", action="store_true")
    ap.add_argument("--skip-b", action="store_true")
    ap.add_argument("--skip-c", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    # Symlink / copy pointers into deepreap_eval/data for the paper layout
    data_dir = ROOT / "deepreap_eval" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in [
        (args.usage_csv, "resource_usage_google.csv"),
        (args.job_trace, "google2011_jobs.csv"),
    ]:
        src = Path(src_name)
        if not src.is_absolute():
            src = ROOT / src
        dst = data_dir / dst_name
        if src.exists() and not dst.exists():
            try:
                dst.symlink_to(src.resolve())
            except OSError:
                dst.write_bytes(src.read_bytes())

    t0 = time.time()
    if args.skip_a:
        model_dir = Path(args.model_root) / "reap"
        cpu = REAPModel.load(model_dir / "reap_cpu_load.joblib")
        mem = REAPModel.load(model_dir / "reap_memory_usage.joblib")
        a = {
            "cpu": cpu, "mem": mem,
            "report": json.loads((out / "phase_a.json").read_text()),
        }
    else:
        a = phase_a(args, out)

    if args.skip_b:
        b = json.loads((out / "phase_b.json").read_text())
    else:
        b = phase_b(args, out, a["cpu"], a["mem"])

    if args.skip_c:
        c = json.loads((out / "phase_c.json").read_text())
    else:
        c = phase_c(args, out, a["cpu"], a["mem"], b)

    write_claim_tables(out, a, b, c)
    final = {
        "elapsed_s": time.time() - t0,
        "prediction_claim": a["report"]["vs_best_single"],
        "scheduling_claim_supported": b["claims"]["scheduling_claim_supported"],
        "robustness_claim_supported": c["robustness_claim_supported"],
        "tables": str(out / "CLAIM_TABLES.md"),
    }
    (out / "summary.json").write_text(json.dumps(final, indent=2, default=str))
    print("\n========== SUMMARY ==========", flush=True)
    print(json.dumps(final, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
