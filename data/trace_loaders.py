"""
Loaders that map industry cluster traces onto the canonical DeepREAP job schema:

    job_id, arrival_time, duration, res_0, res_1, ...

Supported sources
-----------------
* google_cluster  — Google Cluster Data (Borg) task events (CSV subset)
* alibaba         — Alibaba Cluster Trace batch_task / batch_instance CSV
* azure_vm        — Azure Public Dataset VM allocations (CSV subset)
* synthetic       — fall through to generate_job_trace

The loaders are intentionally tolerant: they accept either a full raw dump
or a pre-normalized CSV already in the canonical schema. Missing optional
columns are filled with sensible defaults so evaluation can still proceed
when only a slim public sample is available.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .synthetic_generator import generate_job_trace

CANONICAL_COLS = ["job_id", "arrival_time", "duration", "res_0", "res_1"]


def _series(df: pd.DataFrame, col: str) -> pd.Series:
    """Return a 1-d Series even if duplicate column labels exist."""
    obj = df[col]
    if isinstance(obj, pd.DataFrame):
        return obj.iloc[:, 0]
    return obj


def _ensure_canonical(df: pd.DataFrame, n_resources: int = 2) -> pd.DataFrame:
    """Normalize column names / dtypes into the scheduler schema."""
    out = df.copy()
    # common aliases — only rename if the target is not already present
    rename = {
        "job": "job_id",
        "JobId": "job_id",
        "task_id": "job_id",
        "submit_time": "arrival_time",
        "arrival": "arrival_time",
        "start_time": "arrival_time",
        "Timestamp": "arrival_time",
        "runtime": "duration",
        "sched_class": "duration",  # placeholder if duration missing
        "cpu": "res_0",
        "mem": "res_1",
        "memory": "res_1",
        "resource_request.cpus": "res_0",
        "resource_request.memory": "res_1",
        "cpu_avg": "res_0",
        "mem_avg": "res_1",
    }
    for src, dst in rename.items():
        if src in out.columns and dst not in out.columns:
            out = out.rename(columns={src: dst})
        elif src in out.columns and dst in out.columns and src != dst:
            out = out.drop(columns=[src])

    # collapse any accidental duplicate labels
    out = out.loc[:, ~out.columns.duplicated()]

    if "job_id" not in out.columns:
        out["job_id"] = np.arange(len(out))
    if "arrival_time" not in out.columns:
        out["arrival_time"] = np.arange(len(out))
    if "duration" not in out.columns:
        out["duration"] = 1

    # coerce numeric
    out["job_id"] = _series(out, "job_id").astype(int)
    out["arrival_time"] = (
        pd.to_numeric(_series(out, "arrival_time"), errors="coerce").fillna(0).astype(int)
    )
    out["duration"] = (
        pd.to_numeric(_series(out, "duration"), errors="coerce")
        .fillna(1)
        .clip(lower=1)
        .astype(int)
    )

    for r in range(n_resources):
        col = f"res_{r}"
        if col not in out.columns:
            # default light demand if resource columns absent
            out[col] = 1 if r > 0 else 2
        out[col] = (
            pd.to_numeric(out[col], errors="coerce")
            .fillna(1)
            .clip(lower=1)
            .astype(int)
        )

    # shift arrivals to start at 0 and clamp duration to env-friendly range
    out["arrival_time"] = out["arrival_time"] - int(out["arrival_time"].min())
    out["duration"] = out["duration"].clip(1, 15)

    # If a converter collapsed every arrival to the same slot (common when a
    # trace shard starts at timestamp 0), spread jobs uniformly over a
    # DeepRM-compatible horizon so the scheduler sees a real arrival process.
    if len(out) > 1 and int(out["arrival_time"].nunique()) <= 1:
        horizon = min(2000, max(100, len(out) // 2))
        out["arrival_time"] = np.linspace(0, horizon, num=len(out), dtype=int)

    cols = ["job_id", "arrival_time", "duration"] + [f"res_{r}" for r in range(n_resources)]
    return out[cols].reset_index(drop=True)


def _quantize_resources(series: pd.Series, max_resource: int = 10) -> pd.Series:
    """Map continuous CPU/mem fractions into integer demand 1..max_resource."""
    s = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    if s.max() <= 1.0 + 1e-6:
        # already a fraction of machine
        scaled = (s * max_resource).clip(1, max_resource)
    elif s.max() <= 100.0:
        scaled = (s / 100.0 * max_resource).clip(1, max_resource)
    else:
        # absolute cores / MB — percentile rank into bins
        ranks = s.rank(pct=True)
        scaled = (ranks * max_resource).clip(1, max_resource)
    return scaled.round().astype(int)


def load_google_cluster(
    path: str | Path,
    n_resources: int = 2,
    max_resource: int = 10,
    max_jobs: int | None = 5000,
) -> pd.DataFrame:
    """
    Load a Google Borg task-events CSV subset.

    Expected (flexible) columns include: time / submit_time, job_id,
    resource_request.cpus, resource_request.memory, and optionally
    finish_time for duration.
    """
    path = Path(path)
    df = pd.read_csv(path)
    # duration from finish - start when available
    if "duration" not in df.columns:
        start_col = next((c for c in ("time", "timestamp", "schedule_time", "start_time") if c in df.columns), None)
        end_col = next((c for c in ("finish_time", "end_time") if c in df.columns), None)
        if start_col and end_col:
            df["duration"] = (
                pd.to_numeric(df[end_col], errors="coerce")
                - pd.to_numeric(df[start_col], errors="coerce")
            ).clip(lower=1)
            df["arrival_time"] = pd.to_numeric(df[start_col], errors="coerce")
        elif start_col:
            df["arrival_time"] = pd.to_numeric(df[start_col], errors="coerce")

    for src, dst in (
        ("resource_request.cpus", "res_0"),
        ("cpu_request", "res_0"),
        ("cpus", "res_0"),
        ("resource_request.memory", "res_1"),
        ("memory_request", "res_1"),
        ("memory", "res_1"),
    ):
        if src in df.columns and dst not in df.columns:
            df[dst] = _quantize_resources(df[src], max_resource=max_resource)

    out = _ensure_canonical(df, n_resources=n_resources)
    if max_jobs is not None and len(out) > max_jobs:
        out = out.iloc[:max_jobs].copy()
    return out


def load_alibaba_cluster(
    path: str | Path,
    n_resources: int = 2,
    max_resource: int = 10,
    max_jobs: int | None = 5000,
) -> pd.DataFrame:
    """
    Load an Alibaba Cluster Trace batch_task CSV subset.

    Typical columns: job_name, task_name, start_time, end_time,
    plan_cpu, plan_mem.
    """
    path = Path(path)
    df = pd.read_csv(path)
    if "start_time" in df.columns:
        df["arrival_time"] = pd.to_numeric(df["start_time"], errors="coerce")
    if "end_time" in df.columns and "start_time" in df.columns:
        df["duration"] = (
            pd.to_numeric(df["end_time"], errors="coerce")
            - pd.to_numeric(df["start_time"], errors="coerce")
        ).clip(lower=1)
    if "plan_cpu" in df.columns:
        df["res_0"] = _quantize_resources(df["plan_cpu"], max_resource=max_resource)
    if "plan_mem" in df.columns:
        df["res_1"] = _quantize_resources(df["plan_mem"], max_resource=max_resource)
    out = _ensure_canonical(df, n_resources=n_resources)
    if max_jobs is not None and len(out) > max_jobs:
        out = out.iloc[:max_jobs].copy()
    return out


def load_azure_vm(
    path: str | Path,
    n_resources: int = 2,
    max_resource: int = 10,
    max_jobs: int | None = 5000,
) -> pd.DataFrame:
    """
    Load an Azure Public Dataset VM allocation CSV subset.

    Typical columns: vmId, timestamp, vmTypeId, coreCount, memory, ...
    """
    path = Path(path)
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["arrival_time"] = pd.to_numeric(df["timestamp"], errors="coerce")
    elif "starttime" in df.columns:
        df["arrival_time"] = pd.to_numeric(df["starttime"], errors="coerce")
    if "vmId" in df.columns:
        df["job_id"] = pd.factorize(df["vmId"])[0]
    if "coreCount" in df.columns:
        df["res_0"] = _quantize_resources(df["coreCount"], max_resource=max_resource)
    if "memory" in df.columns:
        df["res_1"] = _quantize_resources(df["memory"], max_resource=max_resource)
    # Azure often lacks per-row duration; derive from consecutive timestamps per vm
    if "duration" not in df.columns and "vmId" in df.columns and "arrival_time" in df.columns:
        df = df.sort_values(["vmId", "arrival_time"])
        df["duration"] = (
            df.groupby("vmId")["arrival_time"].diff().shift(-1).fillna(1).clip(lower=1)
        )
    out = _ensure_canonical(df, n_resources=n_resources)
    if max_jobs is not None and len(out) > max_jobs:
        out = out.iloc[:max_jobs].copy()
    return out


def load_job_trace(
    path: str | Path | None = None,
    source: str = "synthetic",
    n_resources: int = 2,
    max_jobs: int | None = None,
    seed: int = 42,
    **synthetic_kwargs,
) -> pd.DataFrame:
    """
    Unified entrypoint.

    Parameters
    ----------
    path : file to load (required unless source == 'synthetic')
    source : one of synthetic | google_cluster | alibaba | azure_vm | canonical
    """
    source = source.lower()
    if source == "synthetic" or path is None:
        return generate_job_trace(n_resources=n_resources, seed=seed, **synthetic_kwargs)

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Trace file not found: {path}")

    # Already-canonical CSV shortcut
    if source == "canonical":
        df = pd.read_csv(path)
        out = _ensure_canonical(df, n_resources=n_resources)
        if max_jobs is not None and len(out) > max_jobs:
            out = out.iloc[:max_jobs].copy()
        return out

    loaders = {
        "google_cluster": load_google_cluster,
        "google": load_google_cluster,
        "alibaba": load_alibaba_cluster,
        "azure_vm": load_azure_vm,
        "azure": load_azure_vm,
    }
    if source not in loaders:
        raise ValueError(f"Unknown trace source '{source}'. Choose from {list(loaders) + ['synthetic', 'canonical']}")
    return loaders[source](path, n_resources=n_resources, max_jobs=max_jobs)
