"""
End-to-end campaign addressing the DeepREAP critique on real traces:

1) Rebuild real job traces + Google-derived REAP usage
2) Train REAP with softmax / topk / stacking
3) Train DeepRM_Plus + DeepREAP with long PPO + multi-obj reward + forecast channels
4) Multi-seed (n=10) long-horizon benchmarks on Alibaba / Google2019 / Azure / Google2011
5) Emit a compact results JSON for the LaTeX update

Usage:
    PYTHONPATH=. python scripts/run_real_campaign.py
    PYTHONPATH=. python scripts/run_real_campaign.py --ppo-updates 200 --n-seeds 10
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], **kw) -> None:
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT), **kw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppo-updates", type=int, default=200,
                    help="≈ updates×256×4 transitions (200 → ~2e5)")
    ap.add_argument("--imitation-episodes", type=int, default=40)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--episode-max-steps", type=int, default=2000)
    ap.add_argument("--train-trace", default="data/real/alibaba2018_jobs.csv")
    ap.add_argument("--skip-reap", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-bench", action="store_true")
    ap.add_argument("--skip-convert", action="store_true")
    args = ap.parse_args()

    out_root = ROOT / "results" / "real_campaign"
    out_root.mkdir(parents=True, exist_ok=True)
    model_dir = ROOT / "models" / "deeprm_real"
    reap_dir = ROOT / "models" / "reap_real"

    if not args.skip_convert:
        run([sys.executable, "-m", "data.fetch_public_traces",
             "--skip-download", "--sources", "google2011,alibaba,azure",
             "--max-jobs", "20000"])
        # google2019 already good if present; don't re-download
        run([sys.executable, "scripts/build_real_resource_usage.py",
             "--out", "data/real/resource_usage_google.csv"])

    if not args.skip_reap:
        reap_dir.mkdir(parents=True, exist_ok=True)
        for scheme in ["softmax", "topk", "stacking", "inv_mse"]:
            for target in ["cpu_load", "memory_usage"]:
                run([
                    sys.executable, "-m", "src.reap.train",
                    "--data", "data/real/resource_usage_google.csv",
                    "--out", str(reap_dir / scheme),
                    "--target", target,
                    "--ensemble-scheme", scheme,
                    "--temperature", "2.0",
                    "--top-k", "3",
                    "--ga-pop", "20",
                    "--ga-gen", "15",
                    "--seed", "42",
                ])

    if not args.skip_train:
        model_dir.mkdir(parents=True, exist_ok=True)
        common = [
            "--job-trace", args.train_trace,
            "--out", str(model_dir),
            "--imitation-episodes", str(args.imitation_episodes),
            "--ppo-updates", str(args.ppo_updates),
            "--ppo-lr", "5e-5",
            "--ppo-lr-end", "5e-6",
            "--ppo-clip", "0.05",
            "--ppo-ent", "0.005",
            "--ppo-ent-end", "0.001",
            "--ppo-target-kl", "0.015",
            "--ppo-lr-schedule", "cosine",
            "--episode-max-steps", str(args.episode_max_steps),
            "--reward-throughput", "0.5",
            "--reward-backlog", "0.05",
            "--reward-wait", "0.01",
            "--seed", "42",
            "--device", "cpu",
        ]
        run([sys.executable, "-m", "src.deeprm.train",
             "--reap-channels", "0", *common])
        run([sys.executable, "-m", "src.deeprm.train",
             "--reap-channels", "2", "--forecast-mode", "proxy", *common])

    if not args.skip_bench:
        traces = {
            "alibaba2018": "data/real/alibaba2018_jobs.csv",
            "google2019": "data/real/google2019_jobs.csv",
            "google2011": "data/real/google2011_jobs.csv",
            "azure2019": "data/real/azure2019_jobs.csv",
        }
        for name, path in traces.items():
            if not (ROOT / path).exists():
                print(f"[campaign] skip missing {path}")
                continue
            dest = out_root / name
            dest.mkdir(parents=True, exist_ok=True)
            run([
                sys.executable, "-m", "src.evaluation.benchmark",
                "--job-trace", path,
                "--trace-source", "canonical",
                "--results", str(dest),
                "--n-seeds", str(args.n_seeds),
                "--seed", "123",
                "--max-steps", str(args.max_steps),
                "--episode-max-steps", str(args.episode_max_steps),
                "--forecast-mode", "proxy",
                "--no-ilp",
                "--deeprm-path", str(model_dir / "deeprm_plus.pt"),
                "--deepreap-path", str(model_dir / "deepreap.pt"),
                "--deeprm-imi-path", str(model_dir / "deeprm_plus_imitation.pt"),
                "--deepreap-imi-path", str(model_dir / "deepreap_imitation.pt"),
                "--reap-cpu-metrics", str(reap_dir / "softmax" / "reap_cpu_load_metrics.json"),
                "--reap-mem-metrics", str(reap_dir / "softmax" / "reap_memory_usage_metrics.json"),
            ])

    # Compact summary for LaTeX
    summary = {"reap": {}, "benchmarks": {}, "config": vars(args)}
    for scheme in ["softmax", "topk", "stacking", "inv_mse"]:
        for target in ["cpu_load", "memory_usage"]:
            p = reap_dir / scheme / f"reap_{target}_metrics.json"
            if p.exists():
                summary["reap"].setdefault(scheme, {})[target] = json.loads(p.read_text())
    for name in ["alibaba2018", "google2019", "google2011", "azure2019"]:
        p = out_root / name / "benchmark_summary.json"
        if p.exists():
            summary["benchmarks"][name] = json.loads(p.read_text())
    (out_root / "campaign_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[campaign] wrote {out_root / 'campaign_summary.json'}")


if __name__ == "__main__":
    main()
