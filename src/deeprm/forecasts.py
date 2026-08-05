"""
Forecast helpers for ablation studies:

* Oracle — ground-truth future demand from an offline utilization timeline
* Noise injection — multiplicative / additive corruption of a forecast tensor
* Simple REAP stand-in — diurnal proxy when a trained REAP model is absent
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .env import ClusterConfig, ClusterEnv


def build_offline_utilization(
    job_trace: pd.DataFrame,
    cfg: ClusterConfig | None = None,
) -> np.ndarray:
    """
    Greedy earliest-fit packing of the full trace → utilization timeline
    of shape (n_resources, T_end). Used as the Oracle's ground truth.
    """
    cfg = cfg or ClusterConfig()
    if job_trace is None or len(job_trace) == 0:
        return np.zeros((cfg.n_resources, 1), dtype=np.float32)

    jobs = job_trace.sort_values("arrival_time").reset_index(drop=True)
    t_end = int(jobs["arrival_time"].max() + jobs["duration"].max() + cfg.time_horizon + 1)
    load = np.zeros((cfg.n_resources, t_end), dtype=np.float32)

    for _, row in jobs.iterrows():
        arrival = int(row["arrival_time"])
        dur = int(row["duration"])
        demand = np.array(
            [float(row[f"res_{r}"]) for r in range(cfg.n_resources)],
            dtype=np.float32,
        )
        placed = False
        for t0 in range(arrival, max(arrival + 1, t_end - dur)):
            if t0 + dur > t_end:
                break
            window = load[:, t0 : t0 + dur]
            if np.all(cfg.res_capacity - window >= demand[:, None]):
                load[:, t0 : t0 + dur] += demand[:, None]
                placed = True
                break
        if not placed:
            # best-effort: clamp onto the end
            t0 = max(0, t_end - dur)
            load[:, t0 : t0 + dur] += demand[:, None]
    return load


def oracle_forecast_at(
    util_timeline: np.ndarray,
    t: int,
    cfg: ClusterConfig,
    channels: int = 2,
) -> np.ndarray:
    """
    Slice the offline utilization timeline into a
    (channels, n_resources, time_horizon) forecast tensor, normalized to [0, 1].
    Channel 0 = CPU (res 0) demand; channel 1 = memory (res 1) if present.
    """
    T = cfg.time_horizon
    R = cfg.n_resources
    out = np.zeros((channels, R, T), dtype=np.float32)
    for k in range(T):
        idx = t + k
        if 0 <= idx < util_timeline.shape[1]:
            col = util_timeline[:, idx] / max(cfg.res_capacity, 1)
        else:
            col = np.zeros(R, dtype=np.float32)
        # broadcast each resource's own future util onto channel layout
        out[0, :, k] = float(col[0]) if R > 0 else 0.0
        if channels > 1:
            out[1, :, k] = float(col[1]) if R > 1 else float(col[0])
    return np.clip(out, 0.0, 1.0)


def inject_forecast_noise(
    forecast: np.ndarray,
    noise_pct: float,
    rng: np.random.Generator | None = None,
    mode: str = "multiplicative",
) -> np.ndarray:
    """
    Corrupt a forecast tensor by ±noise_pct (e.g. 0.1 = ±10%).

    mode='multiplicative' → forecast * (1 + U(-p, p))
    mode='additive'       → forecast + U(-p, p)  (clipped to [0, 1])
    mode='gaussian'       → forecast + N(0, p²)   (CMPE-294 Phase C)
    """
    rng = rng or np.random.default_rng(0)
    p = float(abs(noise_pct))
    out = forecast.astype(np.float32).copy()
    if p <= 0:
        return out
    if mode == "additive":
        out = out + rng.uniform(-p, p, size=out.shape).astype(np.float32)
    elif mode == "gaussian":
        out = out + rng.normal(0.0, p, size=out.shape).astype(np.float32)
    else:
        out = out * (1.0 + rng.uniform(-p, p, size=out.shape).astype(np.float32))
    return np.clip(out, 0.0, 1.0)


def diurnal_proxy_forecast(
    t: int,
    cfg: ClusterConfig,
    channels: int = 2,
) -> np.ndarray:
    """Cheap stand-in when no REAP model is loaded: smooth diurnal bump."""
    T = cfg.time_horizon
    R = cfg.n_resources
    out = np.zeros((channels, R, T), dtype=np.float32)
    for k in range(T):
        phase = 0.5 + 0.4 * np.sin(2 * np.pi * (t + k) / 24.0)
        out[:, :, k] = float(np.clip(phase, 0.0, 1.0))
    return out


def attach_forecast_callback(
    env: ClusterEnv,
    mode: str = "oracle",
    util_timeline: np.ndarray | None = None,
    noise_pct: float = 0.0,
    seed: int = 0,
    reap_predict_fn=None,
    noise_mode: str = "multiplicative",
):
    """
    Monkey-patch env._advance_one_step to refresh the REAP/oracle channels
    each time the cluster clock ticks. `mode` ∈ {oracle, reap, proxy, zero}.
    """
    rng = np.random.default_rng(seed)
    cfg = env.cfg
    channels = cfg.reap_channels
    if channels <= 0:
        return env

    if mode == "oracle" and util_timeline is None:
        raise ValueError("oracle mode requires util_timeline")

    orig_advance = env._advance_one_step

    def _make_forecast(t: int) -> np.ndarray:
        if mode == "zero":
            return np.zeros((channels, cfg.n_resources, cfg.time_horizon), dtype=np.float32)
        if mode == "oracle":
            return oracle_forecast_at(util_timeline, t, cfg, channels=channels)
        if mode == "reap" and reap_predict_fn is not None:
            return np.asarray(reap_predict_fn(t), dtype=np.float32)
        return diurnal_proxy_forecast(t, cfg, channels=channels)

    def _advance():
        orig_advance()
        fc = _make_forecast(env.t)
        if noise_pct > 0:
            fc = inject_forecast_noise(fc, noise_pct, rng=rng, mode=noise_mode)
        env.set_reap_forecast(fc)

    env._advance_one_step = _advance  # type: ignore[method-assign]
    # seed initial forecast
    fc0 = _make_forecast(env.t)
    if noise_pct > 0:
        fc0 = inject_forecast_noise(fc0, noise_pct, rng=rng, mode=noise_mode)
    env.set_reap_forecast(fc0)
    return env
