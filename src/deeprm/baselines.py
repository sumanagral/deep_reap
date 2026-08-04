"""
Heuristic baselines for the cluster scheduling environment.

All baselines expose the same `act(env)` interface: given the current
ClusterEnv, return an integer action in [0, env.action_dim). They never
mutate the environment.
"""

from __future__ import annotations

import numpy as np

from .env import ClusterEnv


def _fits(env: ClusterEnv, slot: int) -> bool:
    j = env.visible[slot]
    if j is None:
        return False
    return env._can_schedule(j)


def sjf_action(env: ClusterEnv) -> int:
    """Shortest-Job-First over the visible slots that currently fit."""
    best = None
    best_dur = None
    for i, j in enumerate(env.visible):
        if j is None or not _fits(env, i):
            continue
        if best_dur is None or j.duration < best_dur:
            best = i
            best_dur = j.duration
    return best if best is not None else env.cfg.n_visible  # no-op


def fifo_action(env: ClusterEnv) -> int:
    """First fitting visible job, in arrival order."""
    candidates = [
        (i, j) for i, j in enumerate(env.visible) if j is not None and _fits(env, i)
    ]
    if not candidates:
        return env.cfg.n_visible
    candidates.sort(key=lambda x: x[1].arrival_time)
    return candidates[0][0]


def packer_action(env: ClusterEnv) -> int:
    """Tetris-style Packer: pick the visible job whose demand vector
    aligns best (highest dot-product) with current free capacity."""
    free = (env.cfg.res_capacity - env.cluster_load[:, 0]).astype(np.float32)
    best = None
    best_score = -np.inf
    for i, j in enumerate(env.visible):
        if j is None or not _fits(env, i):
            continue
        score = float(np.dot(free, j.demand))
        if score > best_score:
            best_score = score
            best = i
    return best if best is not None else env.cfg.n_visible


BASELINES = {
    "sjf": sjf_action,
    "fifo": fifo_action,
    "packer": packer_action,
}


def run_baseline(env: ClusterEnv, name: str, max_steps: int = 5000) -> dict:
    fn = BASELINES[name]
    env.reset()
    steps = 0
    total_reward = 0.0
    while steps < max_steps:
        action = fn(env)
        _, r, done, info = env.step(action)
        total_reward += r
        steps += 1
        if done:
            break
    m = env.metrics()
    m["total_reward"] = total_reward
    m["steps"] = steps
    return m
