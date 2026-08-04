"""
Generate production-style cluster job traces that mimic published
Google Borg and Alibaba Cluster Trace statistics, and emit them in the
canonical DeepREAP schema (and optional "raw" loader-friendly CSVs).

Full multi-GB public dumps are not vendored in-repo; these generators
reproduce the published *distributions* (arrival burstiness, bimodal
durations, heavy-tailed resource requests) so evaluation and
zero-shot transfer experiments are reproducible without external
downloads. Real dumps can still be plugged in via
`data.trace_loaders.load_job_trace(..., source=...)`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .trace_loaders import _ensure_canonical


def generate_google_like_trace(
    n_jobs: int = 4000,
    horizon: int = 2000,
    n_resources: int = 2,
    max_resource: int = 10,
    seed: int = 11,
) -> pd.DataFrame:
    """
    Google Borg–like task mix (Reiss et al. / Tirmazi et al. stylized):
      - highly bursty arrivals (log-normal interarrivals)
      - ~70% short tasks, long tail of multi-hour tasks
      - CPU/mem requests often < 0.5 machine; rare large tasks
    """
    rng = np.random.default_rng(seed)
    # log-normal interarrivals → bursty
    inter = rng.lognormal(mean=np.log(horizon / n_jobs), sigma=0.9, size=n_jobs)
    arrivals = np.cumsum(inter).astype(int)
    arrivals = arrivals[arrivals < horizon]
    n = len(arrivals)

    # duration: mostly short, heavy tail
    u = rng.random(n)
    duration = np.where(
        u < 0.70,
        rng.integers(1, 4, size=n),
        np.where(
            u < 0.95,
            rng.integers(4, 12, size=n),
            rng.integers(12, 16, size=n),
        ),
    )

    # resource requests: Beta-like concentration at low usage
    cpu_frac = rng.beta(1.5, 6.0, size=n)
    mem_frac = rng.beta(1.8, 5.5, size=n)
    res_0 = np.clip(np.round(cpu_frac * max_resource), 1, max_resource).astype(int)
    res_1 = np.clip(np.round(mem_frac * max_resource), 1, max_resource).astype(int)

    raw = pd.DataFrame({
        "time": arrivals * 1_000_000,          # us-like timestamps for loader
        "finish_time": (arrivals + duration) * 1_000_000,
        "job_id": np.arange(n),
        "cpu_request": cpu_frac,
        "memory_request": mem_frac,
    })
    # also return canonical via ensure
    canon = pd.DataFrame({
        "job_id": np.arange(n),
        "arrival_time": arrivals,
        "duration": duration,
        "res_0": res_0,
        "res_1": res_1,
    })
    canon.attrs["raw"] = raw
    return canon


def generate_alibaba_like_trace(
    n_jobs: int = 4000,
    horizon: int = 2000,
    n_resources: int = 2,
    max_resource: int = 10,
    seed: int = 22,
) -> pd.DataFrame:
    """
    Alibaba batch-task–like mix (stylized from published cluster traces):
      - diurnal-modulated Poisson arrivals
      - longer batch jobs than Google interactive mix
      - plan_cpu / plan_mem correlated
    """
    rng = np.random.default_rng(seed)
    t = 0
    arrivals = []
    while len(arrivals) < n_jobs and t < horizon:
        # diurnal rate: higher mid-horizon "day"
        phase = 0.5 + 0.5 * np.sin(2 * np.pi * t / max(horizon / 3, 1))
        rate = (n_jobs / horizon) * (0.5 + phase)
        gap = int(rng.exponential(1.0 / max(rate, 1e-6)))
        t = t + max(gap, 1)
        if t < horizon:
            arrivals.append(t)
    arrivals = np.asarray(arrivals[:n_jobs], dtype=int)
    n = len(arrivals)

    duration = np.clip(
        rng.choice([2, 3, 5, 8, 12, 15], size=n, p=[0.25, 0.25, 0.2, 0.15, 0.1, 0.05]),
        1,
        15,
    )
    plan_cpu = rng.gamma(2.0, 8.0, size=n)       # percent-like
    plan_mem = plan_cpu * rng.uniform(0.6, 1.3, size=n) + rng.normal(0, 2, size=n)
    plan_cpu = np.clip(plan_cpu, 1, 100)
    plan_mem = np.clip(plan_mem, 1, 100)

    raw = pd.DataFrame({
        "start_time": arrivals,
        "end_time": arrivals + duration,
        "job_name": [f"j{i}" for i in range(n)],
        "plan_cpu": plan_cpu,
        "plan_mem": plan_mem,
    })
    res_0 = np.clip(np.round(plan_cpu / 100.0 * max_resource), 1, max_resource).astype(int)
    res_1 = np.clip(np.round(plan_mem / 100.0 * max_resource), 1, max_resource).astype(int)
    canon = pd.DataFrame({
        "job_id": np.arange(n),
        "arrival_time": arrivals,
        "duration": duration.astype(int),
        "res_0": res_0,
        "res_1": res_1,
    })
    canon.attrs["raw"] = raw
    return canon


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--jobs", type=int, default=4000)
    ap.add_argument("--horizon", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    g = generate_google_like_trace(n_jobs=args.jobs, horizon=args.horizon, seed=args.seed)
    g_path = out / "google_like_trace.csv"
    g.to_csv(g_path, index=False)
    raw_g = g.attrs.get("raw")
    if raw_g is not None:
        raw_g.to_csv(out / "google_like_raw.csv", index=False)
    print(f"[prod] wrote {g_path} rows={len(g)}")

    a = generate_alibaba_like_trace(n_jobs=args.jobs, horizon=args.horizon, seed=args.seed + 11)
    a_path = out / "alibaba_like_trace.csv"
    a.to_csv(a_path, index=False)
    raw_a = a.attrs.get("raw")
    if raw_a is not None:
        raw_a.to_csv(out / "alibaba_like_raw.csv", index=False)
    print(f"[prod] wrote {a_path} rows={len(a)}")

    # sanity: loader round-trip on raw dumps
    from .trace_loaders import load_google_cluster, load_alibaba_cluster
    g2 = load_google_cluster(out / "google_like_raw.csv", max_jobs=None)
    a2 = load_alibaba_cluster(out / "alibaba_like_raw.csv", max_jobs=None)
    _ensure_canonical(g2).to_csv(out / "google_like_canonical.csv", index=False)
    _ensure_canonical(a2).to_csv(out / "alibaba_like_canonical.csv", index=False)
    print(f"[prod] loader round-trip google={len(g2)} alibaba={len(a2)}")


if __name__ == "__main__":
    main()
