"""
Gym-compatible cluster scheduling environment for DeepRM_Plus.

State representation (image-like, as in Mao et al. 2016):
    - Cluster image     : (n_resources, time_horizon)         currently scheduled load
    - Job slots         : M jobs x (n_resources, time_horizon)  visible queue
    - Backlog count     : scalar  (jobs waiting beyond the visible window)
    - Optional: REAP demand forecast appended as an extra channel of the
                cluster image (filled by the integration layer).

Actions: discrete index in [0, M].
    - 0..M-1 : schedule the j-th visible job (skip if it does not fit)
    - M      : "no-op" -> advance one timestep

Reward (multi-objective, per time-advancing step):
    R = -Σ_j 1/T_j          (slowdown pressure on jobs in system)
      + α · ΔN_completed    (throughput bonus)
      - β · backlog_penalty (starvation / deferred-job penalty)
      - γ · wait_penalty    (age-weighted wait of queued jobs)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ClusterConfig:
    n_resources: int = 2
    res_capacity: int = 10           # capacity per resource
    time_horizon: int = 20           # T: how far ahead the image looks
    n_visible: int = 5               # M: number of visible job slots
    max_job_len: int = 15
    max_backlog: int = 60
    new_job_rate: float = 0.7        # P(new job per timestep when env steps)
    # Longer horizon forces deferred long jobs to eventually complete, giving
    # a steadier picture of throughput (old default 200 truncated too early).
    episode_max_steps: int = 2000
    reap_channels: int = 0           # appended channels for REAP forecast (optional)
    # Multi-objective reward weights
    reward_throughput_coef: float = 0.5   # α: bonus per newly completed job
    reward_backlog_coef: float = 0.05     # β: per backlog job
    reward_wait_coef: float = 0.01        # γ: age-weighted wait of queued jobs
    reward_slowdown_scale: float = 10.0   # divisor for Σ 1/T_j term
    reward_frag_coef: float = 0.02        # δ: multi-resource fragmentation penalty
    # SLA: max allowable waiting time (arrival → start); jobs above count as breach
    sla_max_wait: int = 30


@dataclass
class Job:
    job_id: int
    arrival_time: int
    duration: int
    demand: np.ndarray        # shape (n_resources,)
    started_time: int = -1    # -1 if not yet scheduled
    finished_time: int = -1


class ClusterEnv:
    """Lightweight gym-style env (no external Gym dep required)."""

    def __init__(
        self,
        cfg: ClusterConfig | None = None,
        job_trace: pd.DataFrame | None = None,
        seed: int = 0,
    ):
        self.cfg = cfg or ClusterConfig()
        self.rng = np.random.default_rng(seed)
        self._seed = seed
        self._job_trace = job_trace
        self.reset()

    # ------------------------------------------------------------------ public
    @property
    def action_dim(self) -> int:
        return self.cfg.n_visible + 1  # +1 for no-op

    @property
    def state_shape(self) -> tuple[int, int, int]:
        # combined image: cluster | M*job_slot | (optional REAP)
        h = self.cfg.n_resources
        w = self.cfg.time_horizon * (1 + self.cfg.n_visible)
        c = 1 + self.cfg.reap_channels
        return (c, h, w)

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self._seed = seed
        c = self.cfg
        # cluster_load[r, t]: load on resource r at offset t from current time
        self.cluster_load = np.zeros((c.n_resources, c.time_horizon), dtype=np.float32)
        self.visible: list[Job | None] = [None] * c.n_visible
        self.backlog: list[Job] = []
        self.in_progress: list[Job] = []
        self.finished: list[Job] = []
        self.t = 0
        self._next_job_id = 0
        self._steps = 0
        self._n_finished_prev = 0
        self._frag_samples: list[float] = []
        self.reap_forecast = np.zeros(
            (c.reap_channels, c.n_resources, c.time_horizon), dtype=np.float32
        )
        # if a fixed trace is supplied, sort by arrival_time
        if self._job_trace is not None:
            self._trace_idx = 0
            self._trace = self._job_trace.sort_values("arrival_time").reset_index(drop=True)
        else:
            self._trace = None
        # bootstrap the queue
        self._spawn_arrivals(initial=True)
        return self._observation()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        assert 0 <= action < self.action_dim
        info: dict = {"scheduled": None, "advanced": False}

        if action < self.cfg.n_visible:
            job = self.visible[action]
            if job is not None and self._can_schedule(job):
                self._schedule(job, slot=action)
                info["scheduled"] = job.job_id
                # IMPORTANT: do NOT advance time after a successful schedule;
                # let the agent fill more slots in the same timestep.
                obs = self._observation()
                reward = 0.0
                done = self._steps >= self.cfg.episode_max_steps and not self._has_work()
                self._steps += 1
                return obs, reward, done, info
            # invalid action -> no-op with small penalty
            info["invalid"] = True

        # no-op: advance one timestep
        info["advanced"] = True
        self._advance_one_step()
        reward = self._reward()
        done = self._steps >= self.cfg.episode_max_steps or not self._has_work()
        self._steps += 1
        return self._observation(), reward, done, info

    # ----------------------------------------------------------- core dynamics
    def _has_work(self) -> bool:
        return (
            any(s is not None for s in self.visible)
            or len(self.backlog) > 0
            or len(self.in_progress) > 0
            or (self._trace is not None and self._trace_idx < len(self._trace))
        )

    def _can_schedule(self, job: Job) -> bool:
        c = self.cfg
        if job.duration > c.time_horizon:
            return False
        # find the earliest start offset where capacity holds
        free = c.res_capacity - self.cluster_load
        for t0 in range(c.time_horizon - job.duration + 1):
            window = free[:, t0 : t0 + job.duration]
            if np.all(window >= job.demand[:, None]):
                job._start_offset = t0  # type: ignore[attr-defined]
                return True
        return False

    def _schedule(self, job: Job, slot: int) -> None:
        t0 = getattr(job, "_start_offset", 0)
        self.cluster_load[:, t0 : t0 + job.duration] += job.demand[:, None]
        job.started_time = self.t + t0
        self.in_progress.append(job)
        self.visible[slot] = None
        self._refill_visible()

    def _refill_visible(self) -> None:
        for i, slot in enumerate(self.visible):
            if slot is None and self.backlog:
                self.visible[i] = self.backlog.pop(0)

    def _advance_one_step(self) -> None:
        # roll cluster image left by 1
        self.cluster_load = np.roll(self.cluster_load, -1, axis=1)
        self.cluster_load[:, -1] = 0.0
        if self.cfg.reap_channels:
            self.reap_forecast = np.roll(self.reap_forecast, -1, axis=2)
            self.reap_forecast[:, :, -1] = 0.0
        self.t += 1
        # finish jobs whose end time has passed
        still = []
        for j in self.in_progress:
            if j.started_time + j.duration <= self.t:
                j.finished_time = j.started_time + j.duration
                self.finished.append(j)
            else:
                still.append(j)
        self.in_progress = still
        # spawn new arrivals
        self._spawn_arrivals()
        self._frag_samples.append(self._fragmentation())

    def _fragmentation(self) -> float:
        """
        Multi-dimensional fragmentation in [0, 1]: capacity left on each
        resource at the current slot that cannot host the average queued
        demand (waste due to imbalance across CPU vs memory).
        """
        c = self.cfg
        free = (c.res_capacity - self.cluster_load[:, 0]).astype(np.float32)
        util = self.cluster_load[:, 0] / max(c.res_capacity, 1)
        # imbalance across resources + unused capacity fraction
        imbalance = float(np.std(util)) if c.n_resources > 1 else 0.0
        unused = float(np.mean(free) / max(c.res_capacity, 1))
        return float(np.clip(0.5 * imbalance + 0.5 * unused * imbalance, 0.0, 1.0))

    def _reward(self) -> float:
        """
        Multi-objective reward:
            R = -Σ 1/T_j / scale
              + α · (# jobs completed this step)
              - β · |backlog|
              - γ · mean wait of queued (visible + backlog) jobs
              - δ · fragmentation
        This prevents the agent from hoarding long jobs indefinitely to
        optimize slowdown at the expense of throughput.
        """
        c = self.cfg
        slowdown_term = 0.0
        for j in self.in_progress:
            slowdown_term += 1.0 / max(j.duration, 1)
        for j in self.visible:
            if j is not None:
                slowdown_term += 1.0 / max(j.duration, 1)
        for j in self.backlog:
            slowdown_term += 1.0 / max(j.duration, 1)
        slowdown_term = -slowdown_term / max(c.reward_slowdown_scale, 1e-8)

        n_done_now = len(self.finished)
        delta_completed = n_done_now - self._n_finished_prev
        self._n_finished_prev = n_done_now
        throughput_term = c.reward_throughput_coef * float(delta_completed)

        backlog_term = -c.reward_backlog_coef * float(len(self.backlog))

        waits: list[float] = []
        for j in self.visible:
            if j is not None:
                waits.append(float(self.t - j.arrival_time))
        for j in self.backlog:
            waits.append(float(self.t - j.arrival_time))
        wait_term = 0.0
        if waits:
            # age-weighted: longer waits penalize more (encourages draining starved jobs)
            wait_term = -c.reward_wait_coef * float(np.mean(waits) / max(c.max_job_len, 1))

        frag_term = -c.reward_frag_coef * self._fragmentation()
        return slowdown_term + throughput_term + backlog_term + wait_term + frag_term

    # ---------------------------------------------------------------- arrivals
    def _spawn_arrivals(self, initial: bool = False) -> None:
        c = self.cfg
        if self._trace is not None:
            while (
                self._trace_idx < len(self._trace)
                and int(self._trace.iloc[self._trace_idx]["arrival_time"]) <= self.t
            ):
                row = self._trace.iloc[self._trace_idx]
                self._trace_idx += 1
                demand = np.array(
                    [int(row[f"res_{r}"]) for r in range(c.n_resources)],
                    dtype=np.float32,
                )
                self._enqueue(
                    Job(
                        job_id=int(row["job_id"]),
                        arrival_time=int(row["arrival_time"]),
                        duration=int(row["duration"]),
                        demand=demand,
                    )
                )
        else:
            n_new = 1 + (1 if initial else 0)
            for _ in range(n_new):
                if self.rng.random() > c.new_job_rate and not initial:
                    continue
                duration = int(self.rng.integers(1, c.max_job_len + 1))
                dom = int(self.rng.integers(0, c.n_resources))
                demand = np.zeros(c.n_resources, dtype=np.float32)
                for r in range(c.n_resources):
                    if r == dom:
                        demand[r] = self.rng.integers(int(0.25 * c.res_capacity) + 1, c.res_capacity + 1)
                    else:
                        demand[r] = self.rng.integers(1, int(0.25 * c.res_capacity) + 2)
                self._enqueue(
                    Job(
                        job_id=self._next_job_id,
                        arrival_time=self.t,
                        duration=duration,
                        demand=demand,
                    )
                )
                self._next_job_id += 1

    def _enqueue(self, job: Job) -> None:
        for i, slot in enumerate(self.visible):
            if slot is None:
                self.visible[i] = job
                return
        if len(self.backlog) < self.cfg.max_backlog:
            self.backlog.append(job)
        # else: drop (the original DeepRM also drops over-capacity backlog)

    # ----------------------------------------------------------- observation
    def _observation(self) -> np.ndarray:
        c = self.cfg
        # base channel: [cluster_image | job_1 | job_2 | ... | job_M]
        slots = []
        for j in self.visible:
            img = np.zeros((c.n_resources, c.time_horizon), dtype=np.float32)
            if j is not None:
                d = min(j.duration, c.time_horizon)
                img[:, :d] = j.demand[:, None]
            slots.append(img)
        base = np.concatenate([self.cluster_load] + slots, axis=1)  # (R, T*(M+1))
        base = base[None, :, :]  # (1, R, T*(M+1))

        if c.reap_channels:
            # tile the forecast across the same time axis as cluster image,
            # leaving the visible-slot region zero (it carries job demand only)
            tiled = np.zeros_like(base).repeat(c.reap_channels, axis=0)
            tiled[:, :, : c.time_horizon] = self.reap_forecast
            obs = np.concatenate([base, tiled], axis=0)
        else:
            obs = base
        return obs.astype(np.float32)

    # ----------------------------------------------------------------- helpers
    def set_reap_forecast(self, forecast: np.ndarray) -> None:
        """forecast: (reap_channels, n_resources, time_horizon)"""
        c = self.cfg
        assert forecast.shape == (c.reap_channels, c.n_resources, c.time_horizon)
        self.reap_forecast = forecast.astype(np.float32)

    def metrics(self) -> dict:
        n_remaining = (
            len(self.in_progress)
            + len(self.backlog)
            + sum(1 for s in self.visible if s is not None)
        )
        frag = float(np.mean(self._frag_samples)) if self._frag_samples else 0.0
        if not self.finished:
            return {
                "avg_slowdown": 0.0,
                "avg_completion": 0.0,
                "p95_slowdown": 0.0,
                "p99_slowdown": 0.0,
                "p95_wait": 0.0,
                "p99_wait": 0.0,
                "avg_wait": 0.0,
                "n_done": 0,
                "throughput": 0.0,
                "fragmentation": frag,
                "sla_breach_rate": 0.0,
                "n_remaining": n_remaining,
            }
        slow = []
        comp = []
        waits = []
        breaches = 0
        for j in self.finished:
            wait = max(j.started_time - j.arrival_time, 0)
            comp_time = j.finished_time - j.arrival_time
            slowdown = comp_time / max(j.duration, 1)
            slow.append(slowdown)
            comp.append(comp_time)
            waits.append(float(wait))
            if wait > self.cfg.sla_max_wait:
                breaches += 1
        n_done = len(self.finished)
        return {
            "avg_slowdown": float(np.mean(slow)),
            "avg_completion": float(np.mean(comp)),
            "p95_slowdown": float(np.percentile(slow, 95)),
            "p99_slowdown": float(np.percentile(slow, 99)),
            "p95_wait": float(np.percentile(waits, 95)),
            "p99_wait": float(np.percentile(waits, 99)),
            "avg_wait": float(np.mean(waits)),
            "n_done": n_done,
            "throughput": float(n_done) / max(self.t, 1),
            "fragmentation": frag,
            "sla_breach_rate": float(breaches) / max(n_done, 1),
            "n_remaining": n_remaining,
        }
