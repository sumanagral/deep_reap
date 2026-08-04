"""
REAP ensemble: sharper accuracy-weighted / stacked combination of base regressors.

Weighting schemes
-----------------
* softmax  — Softmax(-MSE / τ); temperature τ sharpens away from weak models
* inv_mse  — classic inverse-MSE (legacy, nearly uniform when MSEs are close)
* topk     — Softmax over the K best models only; others get weight 0
* stacking — Ridge meta-learner on base-model predictions (held-out)

Online re-weighting
-------------------
`REAPModel.update_online_weights` accepts (y_true, y_pred_per_model) feedback
and exponentially blends a fresh Softmax(-MSE) estimate into the live weights,
closing the monitoring → prediction feedback loop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .feature_selection import GAConfig, select_features
from .models import NamedRegressor, build_default_models

WeightingScheme = Literal["softmax", "inv_mse", "topk", "stacking"]


@dataclass
class EnsembleConfig:
    scheme: WeightingScheme = "softmax"
    temperature: float = 2.0       # Softmax τ; lower → sharper (more selective)
    top_k: int = 3                 # used when scheme == "topk"
    online_lr: float = 0.2         # EMA blend factor for online re-weighting
    online_window: int = 64        # rolling window of feedback residuals


def compute_weights(
    mses: np.ndarray,
    scheme: WeightingScheme = "softmax",
    temperature: float = 2.0,
    top_k: int = 3,
) -> np.ndarray:
    """Map per-model validation MSEs to a probability simplex of weights."""
    mses = np.asarray(mses, dtype=float)
    n = len(mses)
    if n == 0:
        return mses

    if scheme == "inv_mse":
        inv = 1.0 / (mses + 1e-8)
        return inv / inv.sum()

    # Softmax(-MSE / τ); subtract max for numerical stability
    logits = -mses / max(temperature, 1e-8)
    logits = logits - logits.max()
    exp = np.exp(logits)
    weights = exp / exp.sum()

    if scheme == "topk":
        k = max(1, min(int(top_k), n))
        order = np.argsort(mses)  # ascending MSE = better
        keep = set(order[:k].tolist())
        masked = np.array([weights[i] if i in keep else 0.0 for i in range(n)])
        if masked.sum() <= 0:
            masked = np.ones(n) / n
        else:
            masked = masked / masked.sum()
        return masked

    return weights  # softmax


@dataclass
class REAPModel:
    target_name: str
    feature_names: list[str]
    feature_mask: np.ndarray
    selected_features: list[str]
    scaler: StandardScaler
    models: list[NamedRegressor]
    weights: np.ndarray
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    scheme: str = "softmax"
    temperature: float = 2.0
    top_k: int = 3
    online_lr: float = 0.2
    meta_learner: Any | None = None   # Ridge for stacking; None otherwise
    _online_sqerr: np.ndarray | None = field(default=None, repr=False)
    _online_count: int = field(default=0, repr=False)

    def predict(self, X_raw: np.ndarray) -> np.ndarray:
        X = X_raw[:, self.feature_mask.astype(bool)]
        X = self.scaler.transform(X)
        preds = np.stack([m.predict(X) for m in self.models])  # (n_models, n_samples)
        if self.meta_learner is not None:
            # stacking: meta-learner consumes (n_samples, n_models)
            return np.asarray(self.meta_learner.predict(preds.T))
        return (self.weights[:, None] * preds).sum(axis=0)

    def per_model_predict(self, X_raw: np.ndarray) -> dict[str, np.ndarray]:
        X = X_raw[:, self.feature_mask.astype(bool)]
        X = self.scaler.transform(X)
        return {m.name: m.predict(X) for m in self.models}

    def update_online_weights(
        self,
        y_true: float | np.ndarray,
        per_model_preds: dict[str, float | np.ndarray] | None = None,
        X_raw: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Dynamically re-fit ensemble weights from cluster feedback.

        Provide either `per_model_preds` (name → prediction) or `X_raw`
        (features) so base predictions can be recomputed. Returns the
        updated weight vector.
        """
        if self.meta_learner is not None:
            # stacking meta-learner is fit offline; skip online reweight
            return self.weights

        y = np.atleast_1d(np.asarray(y_true, dtype=float))
        if per_model_preds is None:
            if X_raw is None:
                raise ValueError("Need per_model_preds or X_raw for online update")
            per_model_preds = self.per_model_predict(np.atleast_2d(X_raw))

        names = [m.name for m in self.models]
        sqerr = np.zeros(len(names), dtype=float)
        for i, name in enumerate(names):
            pred = np.atleast_1d(np.asarray(per_model_preds[name], dtype=float))
            sqerr[i] = float(np.mean((pred - y) ** 2))

        if self._online_sqerr is None:
            self._online_sqerr = sqerr.copy()
            self._online_count = 1
        else:
            # EMA of squared error per model
            a = float(self.online_lr)
            self._online_sqerr = (1.0 - a) * self._online_sqerr + a * sqerr
            self._online_count += 1

        new_w = compute_weights(
            self._online_sqerr,
            scheme=self.scheme if self.scheme != "stacking" else "softmax",
            temperature=self.temperature,
            top_k=self.top_k,
        )
        # Blend toward online estimate so a single noisy feedback can't flip the ensemble
        blend = min(1.0, self.online_lr * (1.0 + 0.1 * self._online_count))
        self.weights = (1.0 - blend) * self.weights + blend * new_w
        self.weights = self.weights / self.weights.sum()
        return self.weights

    # --- persistence ---
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "target_name": self.target_name,
                "feature_names": self.feature_names,
                "feature_mask": self.feature_mask,
                "selected_features": self.selected_features,
                "scaler": self.scaler,
                "models": self.models,
                "weights": self.weights,
                "metrics": self.metrics,
                "scheme": self.scheme,
                "temperature": self.temperature,
                "top_k": self.top_k,
                "online_lr": self.online_lr,
                "meta_learner": self.meta_learner,
                "_online_sqerr": self._online_sqerr,
                "_online_count": self._online_count,
            },
            path,
        )

    @staticmethod
    def load(path: str | Path) -> "REAPModel":
        d = joblib.load(path)
        # backward-compat for older checkpoints missing new fields
        d.setdefault("scheme", "inv_mse")
        d.setdefault("temperature", 2.0)
        d.setdefault("top_k", 3)
        d.setdefault("online_lr", 0.2)
        d.setdefault("meta_learner", None)
        d.setdefault("_online_sqerr", None)
        d.setdefault("_online_count", 0)
        return REAPModel(**d)


def train_reap(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    target_name: str,
    test_size: float = 0.2,
    seed: int = 42,
    ga_cfg: GAConfig | None = None,
    verbose: bool = True,
    ensemble_cfg: EnsembleConfig | None = None,
) -> REAPModel:
    """
    1. GA feature selection on TRAIN split
    2. Standardize selected features
    3. Train each base model
    4. Combine via Softmax / Top-K / Stacking (configurable)
    """
    ens_cfg = ensemble_cfg or EnsembleConfig()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    if verbose:
        print(f"[REAP] target={target_name}  train={len(y_tr)} test={len(y_te)}")
        print(f"[REAP] running GA feature selection over {X.shape[1]} features...")

    mask, selected, fit = select_features(
        X_tr, y_tr,
        feature_names=feature_names,
        cfg=ga_cfg or GAConfig(seed=seed),
        verbose=verbose,
    )
    if verbose:
        print(f"[REAP] selected {len(selected)}/{X.shape[1]} features  fit={fit:.4f}")
        print(f"[REAP] features: {selected}")

    Xs_tr = X_tr[:, mask.astype(bool)]
    Xs_te = X_te[:, mask.astype(bool)]
    scaler = StandardScaler().fit(Xs_tr)
    Xs_tr_s = scaler.transform(Xs_tr)
    Xs_te_s = scaler.transform(Xs_te)

    models = build_default_models(seed=seed)
    metrics: dict[str, dict[str, float]] = {}
    val_mses = []
    base_preds_te = []
    base_preds_tr = []
    for m in models:
        m.fit(Xs_tr_s, y_tr)
        pred_te = m.predict(Xs_te_s)
        pred_tr = m.predict(Xs_tr_s)
        mse = float(mean_squared_error(y_te, pred_te))
        mae = float(mean_absolute_error(y_te, pred_te))
        metrics[m.name] = {"mse": mse, "mae": mae}
        val_mses.append(mse)
        base_preds_te.append(pred_te)
        base_preds_tr.append(pred_tr)
        if verbose:
            print(f"[REAP]   {m.name:18s}  mse={mse:7.4f}  mae={mae:7.4f}")

    val_mses_arr = np.asarray(val_mses, dtype=float)
    meta_learner = None
    preds_te = np.stack(base_preds_te)  # (n_models, n_test)

    if ens_cfg.scheme == "stacking":
        # Hold out a meta-train slice from the training predictions to avoid leakage
        P_tr = np.stack(base_preds_tr).T  # (n_train, n_models)
        P_te = preds_te.T
        meta = Ridge(alpha=1.0)
        meta.fit(P_tr, y_tr)
        ens = np.asarray(meta.predict(P_te))
        meta_learner = meta
        # expose absolute ridge coefs as descriptive "weights"
        coef = np.abs(np.asarray(meta.coef_, dtype=float))
        weights = coef / (coef.sum() + 1e-12)
        if verbose:
            print(f"[REAP]   stacking meta-learner Ridge fitted")
    else:
        weights = compute_weights(
            val_mses_arr,
            scheme=ens_cfg.scheme,
            temperature=ens_cfg.temperature,
            top_k=ens_cfg.top_k,
        )
        ens = (weights[:, None] * preds_te).sum(axis=0)

    ens_mse = float(mean_squared_error(y_te, ens))
    ens_mae = float(mean_absolute_error(y_te, ens))
    metrics["ENSEMBLE"] = {
        "mse": ens_mse,
        "mae": ens_mae,
        "scheme": ens_cfg.scheme,
        "temperature": ens_cfg.temperature,
        "top_k": ens_cfg.top_k,
    }
    if verbose:
        print(f"[REAP]   ENSEMBLE({ens_cfg.scheme}) mse={ens_mse:7.4f}  mae={ens_mae:7.4f}")
        print(f"[REAP]   weights={dict(zip([m.name for m in models], weights.round(3)))}")
        best_single = float(np.min(val_mses_arr))
        if ens_mse > best_single:
            print(
                f"[REAP]   note: ensemble MSE {ens_mse:.4f} > best base {best_single:.4f}; "
                f"consider lower τ or stacking"
            )

    return REAPModel(
        target_name=target_name,
        feature_names=feature_names,
        feature_mask=mask,
        selected_features=selected,
        scaler=scaler,
        models=models,
        weights=weights,
        metrics=metrics,
        scheme=ens_cfg.scheme,
        temperature=ens_cfg.temperature,
        top_k=ens_cfg.top_k,
        online_lr=ens_cfg.online_lr,
        meta_learner=meta_learner,
    )


def metrics_to_json(model: REAPModel, path: str | Path) -> None:
    """Serialize per-model + ensemble metrics for the report."""
    payload: dict[str, Any] = {
        "target": model.target_name,
        "selected_features": model.selected_features,
        "scheme": model.scheme,
        "temperature": model.temperature,
        "top_k": model.top_k,
        "weights": dict(zip([m.name for m in model.models], model.weights.tolist())),
        "metrics": model.metrics,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
