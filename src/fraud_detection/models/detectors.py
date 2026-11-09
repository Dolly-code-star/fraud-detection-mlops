"""Unsupervised anomaly detectors with a common interface: ``fit(X)`` / ``score(X)``.

``score`` always returns a value where HIGHER = MORE ANOMALOUS (more likely fraud).
Labels are never passed to ``fit`` - these are unsupervised models.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest


class BaseDetector:
    def fit(self, X: np.ndarray) -> "BaseDetector":
        raise NotImplementedError

    def score(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class IsolationForestDetector(BaseDetector):
    def __init__(self, seed: int = 42, **params):
        self.model = IsolationForest(random_state=seed, **params)

    def fit(self, X):
        self.model.fit(X)
        return self

    def score(self, X):
        return -self.model.score_samples(X)  # sklearn: lower = more abnormal


class PCAReconstructionDetector(BaseDetector):
    """Anomaly score = squared reconstruction error after projecting on the main components."""

    def __init__(self, seed: int = 42, n_components=0.90, **params):
        self.model = PCA(n_components=n_components, random_state=seed, **params)

    def fit(self, X):
        self.model.fit(X)
        return self

    def score(self, X):
        recon = self.model.inverse_transform(self.model.transform(X))
        return ((X - recon) ** 2).sum(axis=1)


class RandomBaselineDetector(BaseDetector):
    """Sanity floor: random scores. Its PR-AUC should be ~= fraud prevalence."""

    def __init__(self, seed: int = 42, **_):
        self.seed = seed

    def fit(self, X):
        return self

    def score(self, X):
        return np.random.default_rng(self.seed).random(len(X))


_REGISTRY = {
    "isolation_forest": IsolationForestDetector,
    "pca_reconstruction": PCAReconstructionDetector,
    "random": RandomBaselineDetector,
}


def build_detector(spec: dict, seed: int) -> BaseDetector:
    try:
        cls = _REGISTRY[spec["type"]]
    except KeyError as exc:
        raise ValueError(f"Unknown detector type {spec['type']!r}; options: {sorted(_REGISTRY)}") from exc
    return cls(seed=seed, **spec.get("params", {}))
