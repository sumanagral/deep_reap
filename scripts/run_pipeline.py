"""
End-to-end pipeline runner.

Equivalent to running:
    python -m data.synthetic_generator
    python -m src.reap.train --target cpu_load
    python -m src.reap.train --target memory_usage
    python -m src.deeprm.train --reap-channels 0 ...      # vanilla DeepRM_Plus
    python -m src.deeprm.train --reap-channels 2 ...      # DeepREAP
    python -m src.evaluation.benchmark
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def _run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd), flush=True)
    res = subprocess.run(cmd)
    if res.returncode != 0:
        sys.exit(res.returncode)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-data", action="store_true")
    ap.add_argument("--skip-reap", action="store_true")
    ap.add_argument("--skip-deeprm", action="store_true")
    ap.add_argument("--skip-deepreap", action="store_true")
    ap.add_argument("--skip-bench", action="store_true")
    ap.add_argument("--imitation-episodes", default="40")
    # Default PPO budget ≈ 4e5 transitions (400 × 256 × 4). Use --quick for CI.
    ap.add_argument("--ppo-updates", default="400")
    ap.add_argument("--ppo-clip", default="0.05")
    ap.add_argument("--ppo-lr", default="5e-5")
    ap.add_argument("--ensemble-scheme", default="softmax")
    ap.add_argument("--temperature", default="2.0")
    ap.add_argument("--n-seeds", default="10")
    ap.add_argument("--max-steps", default="8000")
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Tiny budget for smoke/CI (overrides ppo-updates / n-seeds)",
    )
    args = ap.parse_args()

    if args.quick:
        args.ppo_updates = "5"
        args.imitation_episodes = "5"
        args.n_seeds = "2"
        args.max_steps = "200"

    py = sys.executable

    if not args.skip_data:
        _run([py, "-m", "data.synthetic_generator"])

    if not args.skip_reap:
        for target in ("cpu_load", "memory_usage"):
            _run([
                py, "-m", "src.reap.train",
                "--target", target,
                "--ensemble-scheme", args.ensemble_scheme,
                "--temperature", args.temperature,
            ])

    common_ppo = [
        "--imitation-episodes", args.imitation_episodes,
        "--ppo-updates", args.ppo_updates,
        "--ppo-clip", args.ppo_clip,
        "--ppo-lr", args.ppo_lr,
    ]

    if not args.skip_deeprm:
        _run([py, "-m", "src.deeprm.train",
              "--reap-channels", "0", *common_ppo])

    if not args.skip_deepreap:
        _run([py, "-m", "src.deeprm.train",
              "--reap-channels", "2", *common_ppo])

    if not args.skip_bench:
        _run([
            py, "-m", "src.evaluation.benchmark",
            "--n-seeds", args.n_seeds,
            "--max-steps", args.max_steps,
        ])

    print("\n[pipeline] done. See results/ and models/ for artifacts.")


if __name__ == "__main__":
    main()
