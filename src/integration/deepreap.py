"""
DeepREAP: glue between REAP (demand prediction) and DeepRM_Plus (RL scheduling).

The forecast tensor produced by `build_reap_forecast` has shape
    (reap_channels, n_resources, time_horizon)
and is plugged directly into the env via `ClusterEnv.set_reap_forecast`.

Channel layout (default, reap_channels=2):
    ch 0 : predicted CPU demand normalized to [0, 1]
    ch 1 : predicted memory demand normalized to [0, 1]

Both predictions are broadcast across the resource axis since REAP
forecasts cluster-level (aggregate) demand, while the cluster image is
per-physical-resource. The agent learns whether to weight this signal
during the PPO refinement stage.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.reap.ensemble import REAPModel


@dataclass
class ForecastInput:
    """Minimal feature payload required to drive the REAP models forward."""
    timestamp: dt.datetime
    service_type: str          # one of {Web, Database, Media, Compute}
    active_users: int
    previous_hour_cpu: float
    previous_hour_memory: float
    network_utilization: float


def features_from_input(inp: ForecastInput, feature_names: list[str]) -> np.ndarray:
    """Map a ForecastInput onto the same feature vector REAP was trained on."""
    is_weekend = int(inp.timestamp.weekday() >= 5)
    raw = {
        "hour_of_day": inp.timestamp.hour,
        "day_of_week": inp.timestamp.weekday(),
        "is_weekend": is_weekend,
        "active_users": inp.active_users,
        "previous_hour_cpu": inp.previous_hour_cpu,
        "network_utilization": inp.network_utilization,
        "memory_usage": inp.previous_hour_memory,
        "svc_Web": int(inp.service_type == "Web"),
        "svc_Database": int(inp.service_type == "Database"),
        "svc_Media": int(inp.service_type == "Media"),
        "svc_Compute": int(inp.service_type == "Compute"),
    }
    vec = np.array([raw[name] for name in feature_names], dtype=float)
    return vec


def reap_predict_one(model: REAPModel, inp: ForecastInput) -> float:
    X = features_from_input(inp, model.feature_names).reshape(1, -1)
    return float(model.predict(X)[0])


def build_reap_forecast(
    cpu_model: REAPModel,
    mem_model: REAPModel | None,
    inp: ForecastInput,
    n_resources: int,
    time_horizon: int,
) -> np.ndarray:
    """
    Produce a (channels, n_resources, time_horizon) forecast tensor by
    rolling REAP forward `time_horizon` steps in 1-hour increments.

    For each future step the inputs are advanced (timestamp +1h, the
    previous-hour readings shift) so the prediction reflects diurnal
    patterns the model has learned.
    """
    channels = 1 + (1 if mem_model is not None else 0)
    out = np.zeros((channels, n_resources, time_horizon), dtype=np.float32)

    cur = inp
    prev_cpu = inp.previous_hour_cpu
    prev_mem = inp.previous_hour_memory
    for t in range(time_horizon):
        cpu_pred = reap_predict_one(
            cpu_model,
            ForecastInput(
                timestamp=cur.timestamp + dt.timedelta(hours=t),
                service_type=cur.service_type,
                active_users=cur.active_users,
                previous_hour_cpu=prev_cpu,
                previous_hour_memory=prev_mem,
                network_utilization=cur.network_utilization,
            ),
        )
        cpu_pred = float(np.clip(cpu_pred, 0.0, 100.0)) / 100.0
        out[0, :, t] = cpu_pred
        if mem_model is not None:
            mem_pred = reap_predict_one(
                mem_model,
                ForecastInput(
                    timestamp=cur.timestamp + dt.timedelta(hours=t),
                    service_type=cur.service_type,
                    active_users=cur.active_users,
                    previous_hour_cpu=prev_cpu,
                    previous_hour_memory=prev_mem,
                    network_utilization=cur.network_utilization,
                ),
            )
            mem_pred = float(np.clip(mem_pred, 0.0, 100.0)) / 100.0
            out[1, :, t] = mem_pred
            prev_mem = mem_pred * 100.0
        prev_cpu = cpu_pred * 100.0
    return out
