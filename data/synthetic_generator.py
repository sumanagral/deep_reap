"""
Synthetic workload generator for DeepREAP.

Produces two artifacts:
  1. resource_usage.csv  -- time-series of cloud resource utilization
                            features used by REAP to predict future demand.
  2. job_trace.csv       -- discrete job arrivals for DeepRM_Plus, with
                            duration and per-resource demand vectors.

Patterns are designed to mimic real cloud workloads:
  - diurnal cycle (peaks during business hours)
  - weekly cycle (lower traffic on weekends)
  - per-service-type baselines (Web / Database / Media / Compute)
  - autoregressive component on previous CPU
  - Gaussian noise
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

SERVICE_TYPES = ["Web", "Database", "Media", "Compute"]
SERVICE_BASELINE = {"Web": 25.0, "Database": 35.0, "Media": 30.0, "Compute": 40.0}
SERVICE_NETWORK = {"Web": 0.6, "Database": 0.3, "Media": 0.9, "Compute": 0.4}


def _diurnal(hour: int) -> float:
    # cosine bump centered at 13:00, normalized 0..1
    return 0.5 * (1.0 - np.cos(2 * np.pi * (hour - 1) / 24.0))


def _weekly(dow: int) -> float:
    # Mon..Fri ~1.0, Sat..Sun ~0.6
    return 1.0 if dow < 5 else 0.6


def generate_resource_usage(
    n_hours: int = 24 * 30,
    seed: int = 42,
    start: str = "2024-01-01",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start=start, periods=n_hours, freq="h")

    rows = []
    prev_cpu = {s: SERVICE_BASELINE[s] for s in SERVICE_TYPES}

    for ts in timestamps:
        for service in SERVICE_TYPES:
            hour = ts.hour
            dow = ts.dayofweek
            base = SERVICE_BASELINE[service]
            d = _diurnal(hour)
            w = _weekly(dow)

            # active users scale with diurnal/weekly + service factor
            users_mu = 200 + 800 * d * w * (1.5 if service == "Web" else 1.0)
            active_users = max(10, int(rng.normal(users_mu, 80)))

            # CPU follows base + diurnal + AR(1) + noise
            target_cpu = (
                base * (0.4 + 0.9 * d * w)
                + 0.25 * (prev_cpu[service] - base)
                + rng.normal(0, 3.0)
            )
            target_cpu = float(np.clip(target_cpu, 1.0, 99.0))

            memory = float(np.clip(target_cpu * 0.85 + rng.normal(0, 4.0), 1, 99))
            net_util = float(
                np.clip(
                    SERVICE_NETWORK[service] * 100 * d * w
                    + rng.normal(0, 5.0),
                    1,
                    99,
                )
            )

            rows.append(
                {
                    "timestamp": ts,
                    "hour_of_day": hour,
                    "day_of_week": dow,
                    "is_weekend": int(dow >= 5),
                    "service_type": service,
                    "active_users": active_users,
                    "previous_hour_cpu": prev_cpu[service],
                    "network_utilization": net_util,
                    "memory_usage": memory,
                    "cpu_load": target_cpu,
                }
            )
            prev_cpu[service] = target_cpu

    df = pd.DataFrame(rows)
    # one-hot service for downstream consumers
    df = pd.concat(
        [df, pd.get_dummies(df["service_type"], prefix="svc").astype(int)],
        axis=1,
    )
    return df


def generate_job_trace(
    n_jobs: int = 2000,
    horizon: int = 1000,
    n_resources: int = 2,
    max_duration: int = 15,
    max_resource: int = 10,
    seed: int = 7,
) -> pd.DataFrame:
    """
    Generate jobs with bimodal duration (mostly short, some long) and
    bimodal resource demand (most light, some heavy) -- the same regime
    used in the original DeepRM paper.
    """
    rng = np.random.default_rng(seed)

    # arrival times: Poisson process compressed into [0, horizon]
    inter = rng.exponential(scale=horizon / n_jobs, size=n_jobs)
    arrivals = np.cumsum(inter).astype(int)
    arrivals = arrivals[arrivals < horizon]
    n = len(arrivals)

    # 80% short jobs (1..3), 20% long jobs (10..15)
    is_long = rng.random(n) < 0.20
    duration = np.where(
        is_long,
        rng.integers(10, max_duration + 1, size=n),
        rng.integers(1, 4, size=n),
    )

    # dominant resource per job; the other is light
    rows = []
    for i in range(n):
        dom = rng.integers(0, n_resources)
        demands = []
        for r in range(n_resources):
            if r == dom:
                demands.append(int(rng.integers(int(0.25 * max_resource) + 1, max_resource + 1)))
            else:
                demands.append(int(rng.integers(1, int(0.25 * max_resource) + 2)))
        rows.append(
            {
                "job_id": i,
                "arrival_time": int(arrivals[i]),
                "duration": int(duration[i]),
                **{f"res_{r}": demands[r] for r in range(n_resources)},
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data", help="output directory")
    ap.add_argument("--hours", type=int, default=24 * 60)
    ap.add_argument("--jobs", type=int, default=3000)
    ap.add_argument("--horizon", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--n-seeds", type=int, default=1,
        help="If >1, also emit job_trace_seed{k}.csv for multi-seed eval",
    )
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[gen] resource_usage: {args.hours} hours x {len(SERVICE_TYPES)} services")
    usage = generate_resource_usage(n_hours=args.hours, seed=args.seed)
    usage_path = out / "resource_usage.csv"
    usage.to_csv(usage_path, index=False)
    print(f"[gen] wrote {usage_path}  rows={len(usage)}")

    print(f"[gen] job_trace: target n_jobs={args.jobs} horizon={args.horizon}")
    jobs = generate_job_trace(
        n_jobs=args.jobs, horizon=args.horizon, seed=args.seed + 1
    )
    jobs_path = out / "job_trace.csv"
    jobs.to_csv(jobs_path, index=False)
    print(f"[gen] wrote {jobs_path}  rows={len(jobs)}")

    if args.n_seeds > 1:
        for k in range(args.n_seeds):
            js = generate_job_trace(
                n_jobs=args.jobs, horizon=args.horizon, seed=args.seed + 1 + k
            )
            p = out / f"job_trace_seed{k}.csv"
            js.to_csv(p, index=False)
            print(f"[gen] wrote {p}  rows={len(js)}")


if __name__ == "__main__":
    main()
