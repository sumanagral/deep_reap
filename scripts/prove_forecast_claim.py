"""
Decisive experiment for the paper claim:

  "Demand-forecast channels in a DeepRM-style state improve average job
   slowdown versus an otherwise-identical no-forecast DRL baseline, at
   matched throughput."

Design (everything identical except forecast channels)
------------------------------------------------------
  A) noforecast  — reap_channels=0
  B) proxy       — reap_channels=2, diurnal proxy (cheap REAP stand-in)
  C) oracle      — reap_channels=2, offline packed utilization (upper bound)

Train A/B/C with the same reward, seeds, imitation+PPO budget, and Google
job window. Evaluate each checkpoint on n seeds with paired Wilcoxon tests
of B/C against A on slowdown AND throughput.

Pass criteria (pre-registered)
-----------------------------
  1. slowdown(C) < slowdown(A) with paired Wilcoxon p < 0.05
     (oracle must beat no-forecast — otherwise fusion cannot help)
  2. slowdown(B) < slowdown(A) with paired Wilcoxon p < 0.05
     (the deployable forecast must also beat no-forecast)
  3. |throughput(B) - throughput(A)| / throughput(A) < 5%
     (matched throughput; not a deferral trick)

Usage
-----
  PYTHONPATH=. python scripts/prove_forecast_claim.py
  PYTHONPATH=. python scripts/prove_forecast_claim.py --n-seeds 30 --ppo-updates 150
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def train_one(
    out: Path,
    channels: int,
    forecast_mode: str,
    job_trace: str,
    args: argparse.Namespace,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "src.deeprm.train",
        "--job-trace", job_trace,
        "--out", str(out),
        "--reap-channels", str(channels),
        "--forecast-mode", forecast_mode,
        "--imitation-episodes", str(args.imitation_episodes),
        "--ppo-updates", str(args.ppo_updates),
        "--max-train-jobs", str(args.max_train_jobs),
        "--episode-max-steps", str(args.episode_max_steps),
        "--reward-throughput", "2.0",
        "--reward-backlog", "0.2",
        "--reward-wait", "0.05",
        "--ppo-ent", "0.02",
        "--ppo-ent-end", "0.005",
        "--ppo-bc-coef", "0.15",
        "--ppo-clip", "0.05",
        "--seed", str(args.seed),
        "--device", args.device,
    ]
    run(cmd)


def eval_ckpt(
    ckpt: Path,
    channels: int,
    forecast_mode: str,
    job_trace: str,
    seeds: list[int],
    max_steps: int,
    episode_max_steps: int,
) -> list[dict]:
    from src.deeprm.env import ClusterConfig, ClusterEnv
    from src.deeprm.forecasts import attach_forecast_callback, build_offline_utilization
    from src.evaluation.benchmark import _load_policy, run_policy

    trace_full = pd.read_csv(job_trace)
    policy, _ = _load_policy(ckpt)
    rows = []
    util_cache: dict[int, np.ndarray] = {}

    for seed in seeds:
        rng = np.random.default_rng(seed)
        if len(trace_full) > 4000:
            start = int(rng.integers(0, max(1, len(trace_full) - 4000)))
            trace = trace_full.iloc[start : start + 4000].reset_index(drop=True).copy()
            trace["arrival_time"] = trace["arrival_time"] - int(trace["arrival_time"].min())
        else:
            trace = trace_full.copy()

        cfg = ClusterConfig(
            reap_channels=channels,
            episode_max_steps=episode_max_steps,
            reward_throughput_coef=2.0,
            reward_backlog_coef=0.2,
            reward_wait_coef=0.05,
        )
        env = ClusterEnv(cfg=cfg, job_trace=trace, seed=seed)
        if channels > 0:
            util = None
            if forecast_mode == "oracle":
                # cache by window start for speed
                key = int(trace["arrival_time"].iloc[0]) + seed
                if key not in util_cache:
                    util_cache[key] = build_offline_utilization(trace, cfg)
                util = util_cache[key]
            attach_forecast_callback(
                env, mode=forecast_mode, util_timeline=util, seed=seed,
            )
        m = run_policy(env, policy, max_steps=max_steps)
        m["seed"] = seed
        rows.append(m)
        print(
            f"[prove] seed={seed} ch={channels} mode={forecast_mode} "
            f"slow={m['avg_slowdown']:.3f} thru={m['throughput']:.3f} "
            f"done={m['n_done']}",
            flush=True,
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    out = {}
    for key in ["avg_slowdown", "throughput", "n_done", "avg_completion", "p95_slowdown"]:
        vals = np.array([r[key] for r in rows], dtype=float)
        out[key] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "values": vals.tolist(),
        }
    return out


def paired(a: list[float], b: list[float]) -> dict:
    a_arr, b_arr = np.asarray(a, float), np.asarray(b, float)
    diff = a_arr - b_arr
    res = {"mean_diff": float(diff.mean()), "n": int(len(diff))}
    if len(diff) >= 6 and np.any(diff != 0):
        w = stats.wilcoxon(a_arr, b_arr, alternative="less")  # a < b ?
        res["wilcoxon_less_p"] = float(w.pvalue)
        res["wilcoxon_stat"] = float(w.statistic)
        t = stats.ttest_rel(a_arr, b_arr, alternative="less")
        res["ttest_less_p"] = float(t.pvalue)
    else:
        res["wilcoxon_less_p"] = None
        res["ttest_less_p"] = None
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-trace", default="data/real/google2011_jobs.csv")
    ap.add_argument("--out", default="results/prove_forecast_claim")
    ap.add_argument("--model-root", default="models/prove_forecast")
    ap.add_argument("--n-seeds", type=int, default=30)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--ppo-updates", type=int, default=150)
    ap.add_argument("--imitation-episodes", type=int, default=40)
    ap.add_argument("--max-train-jobs", type=int, default=5000)
    ap.add_argument("--episode-max-steps", type=int, default=1500)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-oracle-train", action="store_true",
                    help="reuse proxy weights for oracle eval only (not a fair train)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model_root = Path(args.model_root)
    seeds = [args.seed + i for i in range(args.n_seeds)]

    configs = {
        "noforecast": {"channels": 0, "mode": "zero", "tag": "deeprm_plus"},
        "proxy": {"channels": 2, "mode": "proxy", "tag": "deepreap"},
        "oracle": {"channels": 2, "mode": "oracle", "tag": "deepreap"},
    }

    if not args.skip_train:
        # A: no forecast
        train_one(model_root / "noforecast", 0, "proxy", args.job_trace, args)
        # B: proxy forecast
        train_one(model_root / "proxy", 2, "proxy", args.job_trace, args)
        # C: oracle forecast (upper bound on the value of perfect channels)
        if not args.skip_oracle_train:
            train_one(model_root / "oracle", 2, "oracle", args.job_trace, args)

    results = {}
    for name, cfg in configs.items():
        mdir = model_root / ("proxy" if name == "oracle" and args.skip_oracle_train else name)
        # prefer PPO ckpt; fall back to imitation
        ckpt = mdir / f"{cfg['tag']}.pt"
        if not ckpt.exists():
            ckpt = mdir / f"{cfg['tag']}_imitation.pt"
        if not ckpt.exists():
            print(f"[prove] missing {ckpt}, skip {name}")
            continue
        rows = eval_ckpt(
            ckpt, cfg["channels"], cfg["mode"], args.job_trace,
            seeds, args.max_steps, args.episode_max_steps,
        )
        results[name] = {"per_seed": rows, "summary": summarize(rows), "ckpt": str(ckpt)}

    # Paired tests vs noforecast
    verdict = {"pass_oracle_slows": False, "pass_proxy_slows": False, "pass_throughput": False}
    tests = {}
    if "noforecast" in results:
        base_slow = results["noforecast"]["summary"]["avg_slowdown"]["values"]
        base_thru = results["noforecast"]["summary"]["throughput"]["values"]
        for name in ["proxy", "oracle"]:
            if name not in results:
                continue
            slow = results[name]["summary"]["avg_slowdown"]["values"]
            thru = results[name]["summary"]["throughput"]["values"]
            tests[name] = {
                "slowdown_vs_noforecast": paired(slow, base_slow),  # want slow < base
                "throughput_rel_diff_mean": float(
                    (np.mean(thru) - np.mean(base_thru)) / max(np.mean(base_thru), 1e-9)
                ),
            }
        if "oracle" in tests and tests["oracle"]["slowdown_vs_noforecast"].get("wilcoxon_less_p") is not None:
            verdict["pass_oracle_slows"] = tests["oracle"]["slowdown_vs_noforecast"]["wilcoxon_less_p"] < 0.05
        if "proxy" in tests and tests["proxy"]["slowdown_vs_noforecast"].get("wilcoxon_less_p") is not None:
            verdict["pass_proxy_slows"] = tests["proxy"]["slowdown_vs_noforecast"]["wilcoxon_less_p"] < 0.05
            verdict["pass_throughput"] = abs(tests["proxy"]["throughput_rel_diff_mean"]) < 0.05

    claim_supported = bool(
        verdict["pass_oracle_slows"] and verdict["pass_proxy_slows"] and verdict["pass_throughput"]
    )

    report = {
        "claim": (
            "Demand-forecast channels improve avg slowdown vs identical "
            "no-forecast DRL baseline at matched throughput."
        ),
        "config": vars(args),
        "seeds": seeds,
        "results": {
            k: {"summary": v["summary"], "ckpt": v["ckpt"]} for k, v in results.items()
        },
        "tests": tests,
        "verdict": verdict,
        "claim_supported": claim_supported,
    }
    (out / "proof_report.json").write_text(json.dumps(report, indent=2))

    print("\n========== PROOF REPORT ==========")
    for name, v in results.items():
        s = v["summary"]
        print(
            f"{name:12s}  slow={s['avg_slowdown']['mean']:.3f}±{s['avg_slowdown']['std']:.3f}  "
            f"thru={s['throughput']['mean']:.3f}±{s['throughput']['std']:.3f}  "
            f"done={s['n_done']['mean']:.1f}"
        )
    for name, t in tests.items():
        p = t["slowdown_vs_noforecast"].get("wilcoxon_less_p")
        print(
            f"vs noforecast [{name}]: Δslow={t['slowdown_vs_noforecast']['mean_diff']:+.3f}  "
            f"p(slow<base)={p}  thru_rel={t['throughput_rel_diff_mean']:+.3%}"
        )
    print("verdict:", verdict)
    print("CLAIM SUPPORTED:" if claim_supported else "CLAIM NOT YET SUPPORTED:", claim_supported)
    print(f"wrote {out / 'proof_report.json'}")


if __name__ == "__main__":
    main()
