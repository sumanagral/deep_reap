"""
FastAPI service exposing the DeepREAP system.

Endpoints
---------
GET  /health        -> liveness
POST /predict       -> single REAP demand prediction
POST /forecast      -> tensor forecast over the env time horizon
POST /schedule      -> scheduling decision given a current ClusterEnv state
POST /feedback      -> report observed value to update REAP error gauges
GET  /metrics       -> Prometheus exposition

The service loads model artifacts from disk on startup (paths configurable
via env vars). If a model is missing the relevant endpoint returns 503.
"""

from __future__ import annotations

import datetime as dt
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from src.deeprm.env import ClusterConfig, ClusterEnv
from src.deeprm.network import CNNPolicy
from src.integration.deepreap import (
    ForecastInput,
    build_reap_forecast,
    reap_predict_one,
)
from src.monitoring import metrics as M
from src.reap.ensemble import REAPModel

VERSION = "0.1.0"
MODELS_DIR = Path(os.environ.get("DEEPREAP_MODELS_DIR", "models"))

state: dict = {
    "reap_cpu": None,
    "reap_mem": None,
    "policy": None,
    "policy_meta": None,
}


# ---------- Pydantic schemas -----------------------------------------
class PredictRequest(BaseModel):
    timestamp: dt.datetime
    service_type: str = Field(pattern="^(Web|Database|Media|Compute)$")
    active_users: int = Field(ge=0)
    previous_hour_cpu: float = Field(ge=0, le=100)
    previous_hour_memory: float = Field(ge=0, le=100)
    network_utilization: float = Field(ge=0, le=100)
    target: str = Field(default="cpu_load", pattern="^(cpu_load|memory_usage)$")


class PredictResponse(BaseModel):
    target: str
    prediction: float


class ForecastRequest(PredictRequest):
    n_resources: int = 2
    time_horizon: int = 20


class ForecastResponse(BaseModel):
    shape: list[int]
    forecast: list[list[list[float]]]


class ScheduleRequest(BaseModel):
    state_image: list[list[list[list[float]]]]  # (B=1, C, H, W)
    deterministic: bool = True


class ScheduleResponse(BaseModel):
    action: int
    is_noop: bool
    logits: list[float]


class FeedbackRequest(BaseModel):
    target: str = Field(pattern="^(cpu_load|memory_usage)$")
    predicted: float
    observed: float
    # Optional feature vector matching the REAP model's feature_names order.
    # When provided, online ensemble re-weighting is applied using per-model
    # residuals on this observation (closes the monitoring → prediction loop).
    features: list[float] | None = None
    feature_names: list[str] | None = None


# ---------- Lifecycle -------------------------------------------------
def _load_reap(name: str) -> REAPModel | None:
    path = MODELS_DIR / "reap" / f"reap_{name}.joblib"
    if path.exists():
        return REAPModel.load(path)
    return None


def _load_policy() -> tuple[CNNPolicy | None, dict | None]:
    candidates = [
        MODELS_DIR / "deeprm" / "deepreap.pt",
        MODELS_DIR / "deeprm" / "deeprm_plus.pt",
    ]
    for p in candidates:
        if p.exists():
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            policy = CNNPolicy(in_channels=ckpt["in_channels"], action_dim=ckpt["action_dim"])
            policy.load_state_dict(ckpt["policy"])
            policy.eval()
            return policy, ckpt
    return None, None


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["reap_cpu"] = _load_reap("cpu_load")
    state["reap_mem"] = _load_reap("memory_usage")
    state["policy"], state["policy_meta"] = _load_policy()
    M.SERVICE_INFO.labels(version=VERSION).set(1)
    yield


app = FastAPI(title="DeepREAP", version=VERSION, lifespan=lifespan)


# ---------- Endpoints -------------------------------------------------
@app.get("/health")
def health() -> dict:
    M.HTTP_REQUESTS.labels(endpoint="/health", status="200").inc()
    return {
        "status": "ok",
        "version": VERSION,
        "reap_cpu_loaded": state["reap_cpu"] is not None,
        "reap_mem_loaded": state["reap_mem"] is not None,
        "policy_loaded": state["policy"] is not None,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    model = state["reap_cpu"] if req.target == "cpu_load" else state["reap_mem"]
    if model is None:
        M.HTTP_REQUESTS.labels(endpoint="/predict", status="503").inc()
        raise HTTPException(status_code=503, detail=f"REAP model for {req.target} not loaded")
    inp = ForecastInput(
        timestamp=req.timestamp,
        service_type=req.service_type,
        active_users=req.active_users,
        previous_hour_cpu=req.previous_hour_cpu,
        previous_hour_memory=req.previous_hour_memory,
        network_utilization=req.network_utilization,
    )
    t0 = time.perf_counter()
    pred = reap_predict_one(model, inp)
    M.REAP_LATENCY.labels(target=req.target).observe(time.perf_counter() - t0)
    M.REAP_PREDICTIONS.labels(target=req.target).inc()
    M.REAP_PREDICTED_VALUE.labels(target=req.target).set(pred)
    M.HTTP_REQUESTS.labels(endpoint="/predict", status="200").inc()
    return PredictResponse(target=req.target, prediction=pred)


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest) -> ForecastResponse:
    cpu = state["reap_cpu"]
    if cpu is None:
        M.HTTP_REQUESTS.labels(endpoint="/forecast", status="503").inc()
        raise HTTPException(status_code=503, detail="REAP CPU model not loaded")
    inp = ForecastInput(
        timestamp=req.timestamp,
        service_type=req.service_type,
        active_users=req.active_users,
        previous_hour_cpu=req.previous_hour_cpu,
        previous_hour_memory=req.previous_hour_memory,
        network_utilization=req.network_utilization,
    )
    arr = build_reap_forecast(
        cpu_model=cpu,
        mem_model=state["reap_mem"],
        inp=inp,
        n_resources=req.n_resources,
        time_horizon=req.time_horizon,
    )
    M.HTTP_REQUESTS.labels(endpoint="/forecast", status="200").inc()
    return ForecastResponse(shape=list(arr.shape), forecast=arr.tolist())


@app.post("/schedule", response_model=ScheduleResponse)
def schedule(req: ScheduleRequest) -> ScheduleResponse:
    policy = state["policy"]
    if policy is None:
        M.HTTP_REQUESTS.labels(endpoint="/schedule", status="503").inc()
        raise HTTPException(status_code=503, detail="Policy not loaded")
    x = torch.tensor(req.state_image, dtype=torch.float32)
    if x.dim() != 4:
        M.HTTP_REQUESTS.labels(endpoint="/schedule", status="400").inc()
        raise HTTPException(status_code=400, detail="state_image must be 4-D (B,C,H,W)")
    t0 = time.perf_counter()
    with torch.no_grad():
        logits, _ = policy(x)
    if req.deterministic:
        action = int(torch.argmax(logits, dim=-1).item())
    else:
        action = int(torch.distributions.Categorical(logits=logits).sample().item())
    M.SCHED_LATENCY.observe(time.perf_counter() - t0)
    meta = state["policy_meta"]
    is_noop = action == meta["action_dim"] - 1
    M.SCHED_DECISIONS.labels(action_type="noop" if is_noop else "schedule").inc()
    M.HTTP_REQUESTS.labels(endpoint="/schedule", status="200").inc()
    return ScheduleResponse(action=action, is_noop=is_noop, logits=logits[0].tolist())


@app.post("/feedback")
def feedback(req: FeedbackRequest) -> dict:
    err = abs(req.predicted - req.observed)
    M.REAP_OBSERVED_ERROR.labels(target=req.target).set(err)

    model = state["reap_cpu"] if req.target == "cpu_load" else state["reap_mem"]
    weights_updated = False
    new_weights: dict[str, float] | None = None
    if model is not None and req.features is not None:
        try:
            import numpy as np

            feats = np.asarray(req.features, dtype=float).reshape(1, -1)
            # If caller supplies a subset / different order, align to model.feature_names
            if req.feature_names is not None and list(req.feature_names) != list(model.feature_names):
                name_to_val = dict(zip(req.feature_names, req.features))
                feats = np.array(
                    [[float(name_to_val.get(n, 0.0)) for n in model.feature_names]],
                    dtype=float,
                )
            if feats.shape[1] == len(model.feature_names):
                w = model.update_online_weights(y_true=req.observed, X_raw=feats)
                weights_updated = True
                new_weights = {
                    m.name: float(wi) for m, wi in zip(model.models, w)
                }
                M.REAP_ONLINE_UPDATES.labels(target=req.target).inc()
        except Exception as exc:  # keep /feedback resilient
            M.HTTP_REQUESTS.labels(endpoint="/feedback", status="200").inc()
            return {
                "recorded": True,
                "abs_error": err,
                "weights_updated": False,
                "error": str(exc),
            }

    M.HTTP_REQUESTS.labels(endpoint="/feedback", status="200").inc()
    return {
        "recorded": True,
        "abs_error": err,
        "weights_updated": weights_updated,
        "weights": new_weights,
    }


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
