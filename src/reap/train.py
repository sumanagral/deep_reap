"""
Training entrypoint for REAP.

Usage:
    python -m src.reap.train --data data/resource_usage.csv --out models/reap
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .ensemble import EnsembleConfig, metrics_to_json, train_reap
from .feature_selection import GAConfig


# Columns that REAP is allowed to choose from (everything else is target / label).
CANDIDATE_FEATURES: list[str] = [
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "active_users",
    "previous_hour_cpu",
    "network_utilization",
    "memory_usage",
    "svc_Compute",
    "svc_Database",
    "svc_Media",
    "svc_Web",
]


def load_dataset(csv_path: str | Path, target: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(csv_path)
    # if memory_usage is the target, exclude it from features and vice versa.
    feats = [c for c in CANDIDATE_FEATURES if c != target and c in df.columns]
    X = df[feats].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)
    return X, y, feats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/resource_usage.csv")
    ap.add_argument("--out", default="models/reap")
    ap.add_argument("--target", default="cpu_load",
                    choices=["cpu_load", "memory_usage"])
    ap.add_argument("--ga-pop", type=int, default=24)
    ap.add_argument("--ga-gen", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--ensemble-scheme",
        default="softmax",
        choices=["softmax", "inv_mse", "topk", "stacking"],
        help="How to combine base regressors (default: softmax over -MSE)",
    )
    ap.add_argument("--temperature", type=float, default=2.0,
                    help="Softmax temperature τ (lower = sharper weights)")
    ap.add_argument("--top-k", type=int, default=3,
                    help="Keep K best models when scheme=topk")
    ap.add_argument("--online-lr", type=float, default=0.2,
                    help="EMA blend factor for online re-weighting")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    X, y, feats = load_dataset(args.data, target=args.target)
    print(f"[train_reap] X={X.shape}  y={y.shape}  features={feats}")
    print(
        f"[train_reap] ensemble={args.ensemble_scheme}  "
        f"τ={args.temperature}  top_k={args.top_k}"
    )

    cfg = GAConfig(pop_size=args.ga_pop, n_generations=args.ga_gen, seed=args.seed)
    ens_cfg = EnsembleConfig(
        scheme=args.ensemble_scheme,
        temperature=args.temperature,
        top_k=args.top_k,
        online_lr=args.online_lr,
    )
    model = train_reap(
        X, y,
        feature_names=feats,
        target_name=args.target,
        ga_cfg=cfg,
        seed=args.seed,
        ensemble_cfg=ens_cfg,
    )

    model_path = out / f"reap_{args.target}.joblib"
    metrics_path = out / f"reap_{args.target}_metrics.json"
    model.save(model_path)
    metrics_to_json(model, metrics_path)
    print(f"[train_reap] saved {model_path}")
    print(f"[train_reap] saved {metrics_path}")


if __name__ == "__main__":
    main()
