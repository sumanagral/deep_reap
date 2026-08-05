"""
Minimal REAP facade (CMPE-294 aligned).

Accuracy-weighted ensemble of LinearRegression, SVR, RandomForest
(+ BayesianRidge, DecisionTree for parity with the full REAP family).
Default combiner = inverse-MSE (as in the original paper).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.reap.ensemble import EnsembleConfig, REAPModel, metrics_to_json, train_reap
from src.reap.feature_selection import GAConfig
from src.reap.train import load_dataset


def train_reap_inv_mse(
    data_csv: str | Path,
    out_dir: str | Path,
    target: str = "cpu_load",
    seed: int = 42,
    ga_pop: int = 20,
    ga_gen: int = 15,
) -> REAPModel:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    X, y, feats = load_dataset(data_csv, target=target)
    model = train_reap(
        X, y,
        feature_names=feats,
        target_name=target,
        seed=seed,
        ga_cfg=GAConfig(pop_size=ga_pop, n_generations=ga_gen, seed=seed),
        ensemble_cfg=EnsembleConfig(scheme="inv_mse", use_timeseries_cv=True),
    )
    model.save(out / f"reap_{target}.joblib")
    metrics_to_json(model, out / f"reap_{target}_metrics.json")
    return model


def load_metrics(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def prediction_table(cpu_metrics: dict, mem_metrics: dict) -> list[dict]:
    """Rows for the Demand Prediction Performance table."""
    rows = []
    models = [
        "LinearRegression", "SVR", "RandomForest", "BayesianRidge",
        "DecisionTree", "ENSEMBLE",
    ]
    for name in models:
        if name not in cpu_metrics["metrics"] and name != "ENSEMBLE":
            continue
        c = cpu_metrics["metrics"].get(name, {})
        m = mem_metrics["metrics"].get(name, {})
        rows.append({
            "model": "REAP Ensemble" if name == "ENSEMBLE" else name,
            "cpu_mae": c.get("mae"),
            "cpu_mse": c.get("mse"),
            "memory_mae": m.get("mae"),
            "memory_mse": m.get("mse"),
            "overall_mse": float(np.mean([c.get("mse", np.nan), m.get("mse", np.nan)])),
        })
    return rows
