"""
Regression model factory for REAP.

Each model is a thin wrapper around scikit-learn implementing a
common (.fit / .predict / .name) interface. The set covers the
algorithms named in the DeepREAP paper:
    - Linear Regression
    - Bayesian Ridge
    - Decision Tree Regressor
    - Random Forest Regressor
    - Support Vector Regressor
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


class RegressorLike(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> "RegressorLike": ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...


@dataclass
class NamedRegressor:
    name: str
    model: RegressorLike

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NamedRegressor":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(X))


def build_default_models(seed: int = 0) -> list[NamedRegressor]:
    return [
        NamedRegressor("LinearRegression", LinearRegression()),
        NamedRegressor("BayesianRidge", BayesianRidge()),
        NamedRegressor(
            "DecisionTree",
            DecisionTreeRegressor(max_depth=10, random_state=seed),
        ),
        NamedRegressor(
            "RandomForest",
            RandomForestRegressor(
                n_estimators=80, max_depth=12, n_jobs=-1, random_state=seed
            ),
        ),
        NamedRegressor("SVR", SVR(kernel="rbf", C=1.0, gamma="scale")),
    ]
