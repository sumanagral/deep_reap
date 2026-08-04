"""
Heuristic and optimization baselines for the cluster scheduling environment.

All baselines expose the same `act(env)` interface: given the current
ClusterEnv, return an integer action in [0, env.action_dim). They never
mutate the environment (except ILP which only reads state).
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


def drf_action(env: ClusterEnv) -> int:
    """
    Dominant Resource Fairness (Ghodsi et al., NSDI'11) over visible jobs.

    For each fitting job, compute its dominant share
        max_r (demand_r / capacity_r)
    and schedule the job with the *smallest* dominant share (most fair /
    least greedy next allocation under DRF progressive filling).
    """
    cap = float(env.cfg.res_capacity)
    best = None
    best_share = np.inf
    best_arrival = np.inf
    for i, j in enumerate(env.visible):
        if j is None or not _fits(env, i):
            continue
        share = float(np.max(j.demand / cap))
        # tie-break by arrival (FIFO among equal shares)
        if share < best_share - 1e-9 or (
            abs(share - best_share) <= 1e-9 and j.arrival_time < best_arrival
        ):
            best = i
            best_share = share
            best_arrival = j.arrival_time
    return best if best is not None else env.cfg.n_visible


def ilp_action(env: ClusterEnv, time_limit: float = 0.05) -> int:
    """
    Short-horizon ILP over the visible window.

    Selects at most one currently-fitting visible job that maximizes
    packing score (dot free·demand) − 0.01·duration, subject to capacity
    over the job's duration. Falls back to packer if PuLP/CBC is
    unavailable or the model is infeasible.

    This is an *online* one-step ILP reference (not a full offline
    optimum over the episode), suitable as a strong baseline.
    """
    try:
        import pulp
    except ImportError:
        return packer_action(env)

    candidates = [
        (i, j) for i, j in enumerate(env.visible) if j is not None and _fits(env, i)
    ]
    if not candidates:
        return env.cfg.n_visible
    if len(candidates) == 1:
        return candidates[0][0]

    free = (env.cfg.res_capacity - env.cluster_load).astype(np.float32)  # (R, T)
    prob = pulp.LpProblem("deeprm_step", pulp.LpMaximize)
    x = {i: pulp.LpVariable(f"job_{i}", cat="Binary") for i, _ in candidates}
    # at most one schedule action this step (matches env's single-action API)
    prob += pulp.lpSum(x.values()) <= 1

    # objective: prefer jobs that pack tightly and finish sooner
    obj = []
    for i, j in candidates:
        t0 = getattr(j, "_start_offset", 0)
        pack = float(np.dot(free[:, t0], j.demand))
        obj.append((pack - 0.01 * j.duration) * x[i])
    prob += pulp.lpSum(obj)

    # capacity constraints along the chosen job's window (only one job →
    # per-candidate local constraints suffice)
    for i, j in candidates:
        t0 = getattr(j, "_start_offset", 0)
        for r in range(env.cfg.n_resources):
            for tt in range(j.duration):
                if t0 + tt >= env.cfg.time_horizon:
                    break
                # if selected, demand must fit free capacity at that cell
                prob += j.demand[r] * x[i] <= float(free[r, t0 + tt]) + 1e-6

    try:
        # Prefer the bundled CBC solver shipped with PuLP.
        status = prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit))
    except Exception:
        return packer_action(env)
    if pulp.LpStatus.get(status, "") not in ("Optimal", "Not Solved"):
        # still accept a feasible integer solution if variables are set
        chosen = [i for i, v in x.items() if v.value() is not None and v.value() > 0.5]
        return int(chosen[0]) if chosen else packer_action(env)
    chosen = [i for i, v in x.items() if v.value() is not None and v.value() > 0.5]
    if not chosen:
        return env.cfg.n_visible
    return int(chosen[0])


BASELINES = {
    "sjf": sjf_action,
    "fifo": fifo_action,
    "packer": packer_action,
    "drf": drf_action,
    "ilp": ilp_action,
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
