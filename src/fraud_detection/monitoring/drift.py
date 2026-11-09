"""Population Stability Index (PSI) drift detection - the monitoring baseline.

Reference bin edges/proportions are computed on the training features and persisted, so the
same reference can later be reused by the scoring API / scheduled monitoring jobs.
Rule of thumb: PSI < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 significant shift.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-4


def _proportions(col: np.ndarray, inner_edges: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(inner_edges, col, side="right")
    counts = np.bincount(idx, minlength=len(inner_edges) + 1).astype(float)
    return counts / max(counts.sum(), 1.0)


def build_reference(X: np.ndarray, names: list[str], bins: int = 10) -> dict:
    ref = {"bins": bins, "features": {}}
    for j, name in enumerate(names):
        edges = np.unique(np.quantile(X[:, j], np.linspace(0, 1, bins + 1)[1:-1]))
        ref["features"][name] = {
            "inner_edges": edges.tolist(),
            "expected": _proportions(X[:, j], edges).tolist(),
        }
    return ref


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e, a = np.clip(expected, _EPS, None), np.clip(actual, _EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def compute_drift(ref: dict, X: np.ndarray, names: list[str], warn: float = 0.10, alert: float = 0.25) -> dict:
    per_feature = {}
    for j, name in enumerate(names):
        spec = ref["features"][name]
        actual = _proportions(X[:, j], np.asarray(spec["inner_edges"]))
        per_feature[name] = psi(np.asarray(spec["expected"]), actual)
    values = np.array(list(per_feature.values()))
    return {
        "psi": per_feature,
        "max_psi": float(values.max()),
        "mean_psi": float(values.mean()),
        "n_warn": int((values >= warn).sum()),
        "n_alert": int((values >= alert).sum()),
        "status": "alert" if (values >= alert).any() else "warn" if (values >= warn).any() else "ok",
    }
