"""
Genetic-algorithm feature selection for REAP.

Chromosome  : binary mask over candidate features.
Fitness     : negative cross-validated MSE of a fast Ridge regressor on
              the selected subset, with a small parsimony penalty.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


@dataclass
class GAConfig:
    pop_size: int = 30
    n_generations: int = 25
    crossover_rate: float = 0.7
    mutation_rate: float = 0.05
    elitism: int = 2
    cv_folds: int = 3
    parsimony: float = 0.01
    min_features: int = 2
    seed: int = 0


def _fitness(mask: np.ndarray, X: np.ndarray, y: np.ndarray, cfg: GAConfig) -> float:
    if mask.sum() < cfg.min_features:
        return -1e9
    Xs = X[:, mask.astype(bool)]
    kf = KFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed)
    errors = []
    for tr, va in kf.split(Xs):
        m = Ridge(alpha=1.0)
        m.fit(Xs[tr], y[tr])
        pred = m.predict(Xs[va])
        errors.append(float(np.mean((pred - y[va]) ** 2)))
    mse = float(np.mean(errors))
    # parsimony: encourage fewer features
    return -mse - cfg.parsimony * mask.sum()


def _tournament(pop: np.ndarray, fits: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.integers(0, len(pop), size=k)
    best = idx[np.argmax(fits[idx])]
    return pop[best].copy()


def _crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    point = int(rng.integers(1, len(a)))
    c1 = np.concatenate([a[:point], b[point:]])
    c2 = np.concatenate([b[:point], a[point:]])
    return c1, c2


def _mutate(ind: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    flips = rng.random(len(ind)) < rate
    out = ind.copy()
    out[flips] = 1 - out[flips]
    return out


def select_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str] | None = None,
    cfg: GAConfig | None = None,
    verbose: bool = False,
) -> tuple[np.ndarray, list[str], float]:
    """
    Returns
    -------
    mask : np.ndarray of {0,1} length n_features
    selected_names : list[str]
    best_fitness : float
    """
    cfg = cfg or GAConfig()
    rng = np.random.default_rng(cfg.seed)
    n_features = X.shape[1]
    feature_names = feature_names or [f"f{i}" for i in range(n_features)]

    pop = (rng.random((cfg.pop_size, n_features)) > 0.5).astype(np.int8)
    # ensure each individual has at least min_features active
    for ind in pop:
        if ind.sum() < cfg.min_features:
            ind[rng.choice(n_features, cfg.min_features, replace=False)] = 1

    fits = np.array([_fitness(ind, X, y, cfg) for ind in pop])

    for gen in range(cfg.n_generations):
        # elitism
        order = np.argsort(-fits)
        new_pop = [pop[i].copy() for i in order[: cfg.elitism]]
        while len(new_pop) < cfg.pop_size:
            p1 = _tournament(pop, fits, k=3, rng=rng)
            p2 = _tournament(pop, fits, k=3, rng=rng)
            if rng.random() < cfg.crossover_rate:
                c1, c2 = _crossover(p1, p2, rng)
            else:
                c1, c2 = p1, p2
            new_pop.append(_mutate(c1, cfg.mutation_rate, rng))
            if len(new_pop) < cfg.pop_size:
                new_pop.append(_mutate(c2, cfg.mutation_rate, rng))
        pop = np.stack(new_pop)
        fits = np.array([_fitness(ind, X, y, cfg) for ind in pop])
        if verbose:
            print(f"[GA] gen={gen:02d}  best={fits.max():.4f}  mean={fits.mean():.4f}")

    best = pop[int(np.argmax(fits))]
    selected = [n for n, b in zip(feature_names, best) if b]
    return best.astype(int), selected, float(fits.max())
