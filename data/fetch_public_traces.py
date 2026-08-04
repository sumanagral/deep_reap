"""
Download public production cluster traces and convert them into DeepREAP's
canonical job schema.

Sources (all public research releases)
--------------------------------------
* Google Cluster Data 2011 — Borg task_events shard (CC-BY)
  https://github.com/google/cluster-data
* Google Cluster Data 2019 — collection_events JSON shard (optional; large)
* Alibaba Cluster Trace v2018 — batch_task.csv (~125 MB compressed)
  https://github.com/alibaba/clusterdata
* Azure Public Dataset V2 — vmtable.csv.gz
  https://github.com/Azure/AzurePublicDataset

Raw archives land in data/dumps/ (gitignored). Converted evaluation subsets
land in data/real/ (committed, capped by --max-jobs).

Usage
-----
    python -m data.fetch_public_traces                 # download + convert all
    python -m data.fetch_public_traces --sources google2011,alibaba --max-jobs 5000
    python -m data.fetch_public_traces --skip-download  # convert already-fetched dumps
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .trace_loaders import _ensure_canonical, _quantize_resources

ROOT = Path(__file__).resolve().parent
DUMPS = ROOT / "dumps"
REAL = ROOT / "real"

URLS = {
    "google2011": (
        "https://storage.googleapis.com/clusterdata-2011-2/"
        "task_events/part-00000-of-00500.csv.gz"
    ),
    "google2019": (
        "https://storage.googleapis.com/clusterdata_2019_a/"
        "collection_events-000000000000.json.gz"
    ),
    # CN endpoint is currently more reliable; US often 404s.
    "alibaba": (
        "http://clusterdata2018pubcn.oss-cn-beijing.aliyuncs.com/batch_task.tar.gz"
    ),
    "alibaba_us": (
        "http://clusterdata2018pubus.oss-us-west-1.aliyuncs.com/batch_task.tar.gz"
    ),
    "alibaba_machine_meta": (
        "http://clusterdata2018pubcn.oss-cn-beijing.aliyuncs.com/machine_meta.tar.gz"
    ),
    "alibaba_container_meta": (
        "http://clusterdata2018pubcn.oss-cn-beijing.aliyuncs.com/container_meta.tar.gz"
    ),
    "azure": (
        "https://github.com/Azure/AzurePublicDataset/releases/download/"
        "dataset-v2/trace_data_vmtable_vmtable.csv.gz"
    ),
}

# Google 2011 task_events columns (trace format v2.1)
G2011_COLS = [
    "timestamp", "missing_info", "job_id", "task_index", "machine_id",
    "event_type", "user", "scheduling_class", "priority",
    "cpu_request", "memory_request", "disk_request", "different_machine",
]

# Alibaba batch_task columns (no header in release)
ALIBABA_COLS = [
    "task_name", "instance_num", "job_name", "task_type", "status",
    "start_time", "end_time", "plan_cpu", "plan_mem",
]

# Azure vmtable (no header in release)
AZURE_COLS = [
    "vm_id", "subscription_id", "deployment_id",
    "vm_created", "vm_deleted", "max_cpu", "avg_cpu", "p95_max_cpu",
    "vm_category", "vm_cores_bucket", "vm_memory_bucket",
]


def _download(url: str, dest: Path, alt_urls: list[str] | None = None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"[fetch] cached {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest
    candidates = [url] + (alt_urls or [])
    last_err: Exception | None = None
    for u in candidates:
        try:
            print(f"[fetch] downloading {u}")
            urllib.request.urlretrieve(u, dest)
            print(f"[fetch] wrote {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
            return dest
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"[fetch] failed: {exc}")
    raise RuntimeError(f"Could not download {dest.name}: {last_err}")


def convert_google2011(path: Path, max_jobs: int, max_resource: int = 10) -> pd.DataFrame:
    """
    Map Borg task_events → jobs.

    event_type 0 = SUBMIT. We keep SUBMIT rows with resource requests, use
    timestamp (µs) as arrival, and estimate duration from later FINISH/EVICT
    events when present; otherwise sample a DeepRM-compatible duration.

    Part-00000 of the public dump begins with a long run of timestamp=0;
    callers should prefer later shards or pass a concatenated multi-shard file.
    """
    df = pd.read_csv(path, header=None, names=G2011_COLS, low_memory=False)
    # Drop the all-zero timestamp prefix that dominates early shards.
    if (df["timestamp"] > 0).any():
        df = df[df["timestamp"] > 0].copy()
    # SUBMIT events with resource requests
    sub = df[(df["event_type"] == 0) & df["cpu_request"].notna()].copy()
    if sub.empty:
        sub = df[df["cpu_request"].notna()].copy()

    # duration estimate: for each (job_id, task_index) find finish-like events
    finish = df[df["event_type"].isin([2, 3, 4, 5, 6])]  # schedule..fail/finish/kill/lost
    dur_map = {}
    if not finish.empty:
        g = finish.groupby(["job_id", "task_index"])["timestamp"].max()
        s = sub.set_index(["job_id", "task_index"])["timestamp"]
        for key, t0 in s.items():
            t1 = g.get(key)
            if t1 is not None and t1 > t0:
                # convert µs delta → discrete slots (~5 min ≈ DeepRM step scale)
                slots = max(1, int(min(15, (t1 - t0) / 1e6 / 60.0)))  # minutes → capped
                dur_map[key] = slots

    rng = np.random.default_rng(2011)
    rows = []
    for i, (_, r) in enumerate(sub.iterrows()):
        if i >= max_jobs:
            break
        key = (r["job_id"], r["task_index"])
        duration = dur_map.get(key, int(rng.integers(1, 12)))
        rows.append({
            "job_id": i,
            "arrival_time": int(r["timestamp"]),
            "duration": duration,
            "cpu_request": float(r["cpu_request"]),
            "memory_request": float(r["memory_request"]) if pd.notna(r["memory_request"]) else 0.05,
            "source_job_id": int(r["job_id"]),
            "source_task_index": int(r["task_index"]),
        })
    out = pd.DataFrame(rows)
    out["arrival_time"] = (
        (out["arrival_time"] - out["arrival_time"].min()) // 1_000_000
    ).astype(int)  # seconds from start
    # compress into a horizon usable by the env (~ few thousand steps)
    if out["arrival_time"].max() > 5000:
        scale = out["arrival_time"].max() / 2000.0
        out["arrival_time"] = (out["arrival_time"] / scale).astype(int)
    out["res_0"] = _quantize_resources(out["cpu_request"], max_resource=max_resource)
    out["res_1"] = _quantize_resources(out["memory_request"], max_resource=max_resource)
    return _ensure_canonical(out[["job_id", "arrival_time", "duration", "res_0", "res_1"]])


def convert_google2019(path: Path, max_jobs: int, max_resource: int = 10) -> pd.DataFrame:
    """collection_events JSON lines → coarse jobs (no per-task CPU; use class priors)."""
    rows = []
    with gzip.open(path, "rt") as f:
        for line in f:
            rec = json.loads(line)
            # type 0 ≈ submit / collection creation in v3 schema variants
            t = int(float(rec.get("time", 0)))
            rows.append({
                "collection_id": rec.get("collection_id"),
                "time": t,
                "scheduling_class": int(float(rec.get("scheduling_class", 1))),
                "priority": int(float(rec.get("priority", 0))),
                "type": int(float(rec.get("type", 0))),
            })
            if len(rows) >= max_jobs * 4:
                break
    df = pd.DataFrame(rows)
    # keep earliest event per collection
    df = df.sort_values("time").groupby("collection_id", as_index=False).first()
    df = df.head(max_jobs).reset_index(drop=True)
    rng = np.random.default_rng(2019)
    # map scheduling class → duration / demand priors
    sc = df["scheduling_class"].to_numpy()
    duration = np.where(sc >= 2, rng.integers(1, 5, size=len(df)), rng.integers(4, 15, size=len(df)))
    cpu = np.where(sc >= 2, rng.uniform(0.05, 0.25, size=len(df)), rng.uniform(0.1, 0.6, size=len(df)))
    mem = cpu * rng.uniform(0.7, 1.2, size=len(df))
    out = pd.DataFrame({
        "job_id": np.arange(len(df)),
        "arrival_time": df["time"].to_numpy(),
        "duration": duration.astype(int),
        "cpu_request": cpu,
        "memory_request": mem,
    })
    out["arrival_time"] = (out["arrival_time"] - out["arrival_time"].min())
    if out["arrival_time"].max() > 5000:
        out["arrival_time"] = (out["arrival_time"] / (out["arrival_time"].max() / 2000.0)).astype(int)
    else:
        out["arrival_time"] = out["arrival_time"].astype(int)
    out["res_0"] = _quantize_resources(out["cpu_request"], max_resource=max_resource)
    out["res_1"] = _quantize_resources(out["memory_request"], max_resource=max_resource)
    return _ensure_canonical(out[["job_id", "arrival_time", "duration", "res_0", "res_1"]])


def convert_alibaba(path: Path, max_jobs: int, max_resource: int = 10) -> pd.DataFrame:
    """batch_task.csv → canonical jobs (one row per task)."""
    df = pd.read_csv(path, header=None, names=ALIBABA_COLS, low_memory=False)
    df["start_time"] = pd.to_numeric(df["start_time"], errors="coerce")
    df["end_time"] = pd.to_numeric(df["end_time"], errors="coerce")
    df["plan_cpu"] = pd.to_numeric(df["plan_cpu"], errors="coerce")
    df["plan_mem"] = pd.to_numeric(df["plan_mem"], errors="coerce")
    df = df.dropna(subset=["start_time", "end_time", "plan_cpu", "plan_mem"])
    # require positive runtime; Terminated/Failed are both useful
    df = df[df["end_time"] > df["start_time"]]
    df = df.sort_values("start_time").head(max_jobs).reset_index(drop=True)

    duration_sec = (df["end_time"] - df["start_time"]).astype(float)
    # Alibaba times are seconds from trace start; 10-second discrete slots
    duration = np.clip(np.ceil(duration_sec / 10.0).astype(int), 1, 15)
    arrival = (df["start_time"] - df["start_time"].min()).astype(int)
    if arrival.max() > 5000:
        arrival = (arrival / (arrival.max() / 2000.0)).astype(int)

    out = pd.DataFrame({
        "job_id": np.arange(len(df)),
        "arrival_time": arrival,
        "duration": duration,
        "plan_cpu": df["plan_cpu"].fillna(50),
        "plan_mem": df["plan_mem"].fillna(0.2),
    })
    # plan_cpu is percent of a machine (0–100+); plan_mem is already a fraction
    out["res_0"] = _quantize_resources(out["plan_cpu"], max_resource=max_resource)
    out["res_1"] = _quantize_resources(out["plan_mem"].clip(0, 1), max_resource=max_resource)
    return _ensure_canonical(out[["job_id", "arrival_time", "duration", "res_0", "res_1"]])


def convert_azure(path: Path, max_jobs: int, max_resource: int = 10) -> pd.DataFrame:
    """Azure vmtable → jobs (VM lifetime as duration, cores/memory buckets as demand)."""
    df = pd.read_csv(path, header=None, names=AZURE_COLS, low_memory=False)
    df["vm_created"] = pd.to_numeric(df["vm_created"], errors="coerce")
    df["vm_deleted"] = pd.to_numeric(df["vm_deleted"], errors="coerce")
    df = df.dropna(subset=["vm_created", "vm_deleted"])
    df = df[df["vm_deleted"] > df["vm_created"]]
    df = df.sort_values("vm_created").head(max_jobs).reset_index(drop=True)

    # timestamps are in seconds from trace start (Azure docs)
    arrival = (df["vm_created"] - df["vm_created"].min()).astype(int)
    dur_sec = (df["vm_deleted"] - df["vm_created"]).clip(lower=1)
    duration = np.clip((dur_sec / 3600.0).round().astype(int), 1, 15)  # hours → slots
    if arrival.max() > 5000:
        arrival = (arrival / (arrival.max() / 2000.0)).astype(int)

    cores = pd.to_numeric(df["vm_cores_bucket"], errors="coerce").fillna(2)
    mem = pd.to_numeric(df["vm_memory_bucket"], errors="coerce").fillna(4)
    out = pd.DataFrame({
        "job_id": np.arange(len(df)),
        "arrival_time": arrival,
        "duration": duration,
        "res_0": _quantize_resources(cores / cores.max(), max_resource=max_resource),
        "res_1": _quantize_resources(mem / mem.max(), max_resource=max_resource),
    })
    return _ensure_canonical(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sources",
        default="google2011,alibaba,azure,google2019",
        help="comma-separated: google2011,google2019,alibaba,azure",
    )
    ap.add_argument("--max-jobs", type=int, default=5000)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--keep-raw", action="store_true",
                    help="keep full raw dumps under data/dumps/ (large)")
    args = ap.parse_args()

    sources = [s.strip().lower() for s in args.sources.split(",") if s.strip()]
    DUMPS.mkdir(parents=True, exist_ok=True)
    REAL.mkdir(parents=True, exist_ok=True)

    provenance = {
        "google2011": {
            "url": URLS["google2011"],
            "license": "CC-BY 4.0",
            "citation": "Reiss et al., Google Cluster-Usage Traces v2.1",
        },
        "google2019": {
            "url": URLS["google2019"],
            "license": "Google cluster data terms",
            "citation": "Tirmazi et al., Borg: the Next Generation (EuroSys'20) / clusterdata-2019",
        },
        "alibaba": {
            "url": URLS["alibaba"],
            "license": "Alibaba clusterdata research release",
            "citation": "Guo et al., Alibaba Cluster Trace v2018 (IWQoS'19)",
        },
        "azure": {
            "url": URLS["azure"],
            "license": "Azure Public Dataset research release",
            "citation": "Cortez et al., Resource Central (SOSP'17) / AzurePublicDatasetV2",
        },
    }

    written = {}

    if "google2011" in sources:
        raw = DUMPS / "google2011_task_events_part0.csv.gz"
        if not args.skip_download:
            _download(URLS["google2011"], raw)
        # Prefer later local shards (part-00000 is almost all timestamp=0).
        shard_dir = REAL / "raw" / "google2011"
        shard_paths = sorted(shard_dir.glob("part-*.csv.gz")) if shard_dir.exists() else []
        usable = [p for p in shard_paths if "part-00000" not in p.name]
        if usable:
            frames = []
            for p in usable:
                frames.append(pd.read_csv(p, header=None, names=G2011_COLS, low_memory=False))
            concat = pd.concat(frames, ignore_index=True)
            tmp = DUMPS / "google2011_task_events_merged.csv.gz"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            concat.to_csv(tmp, index=False, header=False, compression="gzip")
            jobs = convert_google2011(tmp, max_jobs=args.max_jobs)
            out = REAL / "google2011_jobs.csv"
            jobs.to_csv(out, index=False)
            written["google2011"] = {"path": str(out), "n_jobs": len(jobs), "shards": len(usable)}
            print(f"[convert] google2011 → {out} rows={len(jobs)} (from {len(usable)} shards)")
        elif raw.exists():
            jobs = convert_google2011(raw, max_jobs=args.max_jobs)
            out = REAL / "google2011_jobs.csv"
            jobs.to_csv(out, index=False)
            written["google2011"] = {"path": str(out), "n_jobs": len(jobs)}
            print(f"[convert] google2011 → {out} rows={len(jobs)}")

    if "google2019" in sources:
        raw = DUMPS / "google2019_collection_events_0.json.gz"
        if not args.skip_download:
            _download(URLS["google2019"], raw)
        if raw.exists():
            jobs = convert_google2019(raw, max_jobs=args.max_jobs)
            out = REAL / "google2019_jobs.csv"
            jobs.to_csv(out, index=False)
            written["google2019"] = {"path": str(out), "n_jobs": len(jobs)}
            print(f"[convert] google2019 → {out} rows={len(jobs)}")

    if "alibaba" in sources:
        tar_path = DUMPS / "alibaba_batch_task.tar.gz"
        csv_path = DUMPS / "alibaba_batch_task.csv"
        if not args.skip_download and not csv_path.exists():
            try:
                _download(URLS["alibaba"], tar_path, alt_urls=[URLS["alibaba_us"]])
            except RuntimeError:
                pass
        if tar_path.exists() and not csv_path.exists():
            with tarfile.open(tar_path, "r:gz") as tf:
                member = next(m for m in tf.getmembers() if m.name.endswith(".csv"))
                tf.extract(member, path=DUMPS)
                extracted = DUMPS / member.name
                extracted.rename(csv_path)
                print(f"[fetch] extracted {csv_path}")
        if csv_path.exists():
            jobs = convert_alibaba(csv_path, max_jobs=args.max_jobs)
            out = REAL / "alibaba2018_jobs.csv"
            jobs.to_csv(out, index=False)
            written["alibaba"] = {"path": str(out), "n_jobs": len(jobs)}
            print(f"[convert] alibaba → {out} rows={len(jobs)}")

    if "azure" in sources:
        raw = DUMPS / "azure_vmtable.csv.gz"
        sample = REAL / "raw" / "azure2019_vmtable_sample.csv"
        if not args.skip_download:
            try:
                _download(URLS["azure"], raw)
            except RuntimeError as exc:
                print(f"[fetch] azure download skipped: {exc}")
        src = raw if raw.exists() else sample
        if src.exists():
            # Sample has a header; full dump does not.
            if src == sample:
                df = pd.read_csv(src)
                tmp = DUMPS / "azure_vmtable_from_sample.csv"
                tmp.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(tmp, index=False, header=False)
                jobs = convert_azure(tmp, max_jobs=args.max_jobs)
            else:
                jobs = convert_azure(src, max_jobs=args.max_jobs)
            out = REAL / "azure2019_jobs.csv"
            jobs.to_csv(out, index=False)
            written["azure"] = {"path": str(out), "n_jobs": len(jobs), "source": str(src)}
            print(f"[convert] azure → {out} rows={len(jobs)} (from {src.name})")

    # relativize paths for portability
    for v in written.values():
        try:
            v["path"] = str(Path(v["path"]).resolve().relative_to(ROOT.parent))
        except Exception:
            pass
    meta = {
        "max_jobs": args.max_jobs,
        "sources": written,
        "provenance": {k: provenance[k] for k in written},
        "note": (
            "Subsets are converted into DeepREAP canonical schema for scheduling "
            "benchmarks. Raw dumps remain under data/dumps/ (gitignored)."
        ),
    }
    meta_path = REAL / "SOURCES.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[fetch] wrote {meta_path}")

    if not args.keep_raw:
        # Drop multi-hundred-MB archives; keep the small Google 2011 shard.
        for heavy in (
            DUMPS / "alibaba_batch_task.tar.gz",
            DUMPS / "alibaba_batch_task.csv",
            DUMPS / "google2019_collection_events_0.json.gz",
            DUMPS / "azure_vmtable.csv.gz",
        ):
            if heavy.exists() and heavy.stat().st_size > 20_000_000:
                heavy.unlink()
                print(f"[fetch] removed large raw {heavy.name} (re-download with --keep-raw)")

    # also copy Google 2011 gz into dumps if we want a small authentic artifact
    g2011 = DUMPS / "google2011_task_events_part0.csv.gz"
    if g2011.exists():
        # keep it — only ~4 MB
        pass

    print("[fetch] done. Use e.g.:")
    print("  python -m src.evaluation.benchmark \\")
    print("    --job-trace data/real/alibaba2018_jobs.csv --trace-source canonical --n-seeds 5")


if __name__ == "__main__":
    main()
