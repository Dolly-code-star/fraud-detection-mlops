"""Feature engineering + preprocessing + chronological split.

The preprocessor is a scikit-learn Pipeline (FeatureBuilder -> StandardScaler) that is fitted
on the training split only and persisted with the model, so training and serving share the
exact same transformation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Turn raw transaction columns into a numeric feature matrix.

    * V1..V28 pass through unchanged.
    * ``Amount``  -> ``log1p(Amount)`` (heavy right tail).
    * ``Time``    -> hour-of-day (seconds since start modulo 24h), encoded as sin/cos so
      23:59 and 00:01 are close. The raw, ever-growing ``Time`` is dropped.
    """

    def __init__(self, time_col="Time", amount_col="Amount", amount_log=True, cyclical_hour=True, v_prefix="V"):
        self.time_col = time_col
        self.amount_col = amount_col
        self.amount_log = amount_log
        self.cyclical_hour = cyclical_hour
        self.v_prefix = v_prefix

    def _v_cols(self, X: pd.DataFrame) -> list[str]:
        cols = [c for c in X.columns if c.startswith(self.v_prefix) and c[len(self.v_prefix):].isdigit()]
        return sorted(cols, key=lambda c: int(c[len(self.v_prefix):]))

    def fit(self, X: pd.DataFrame, y=None):
        names = self._v_cols(X)
        names.append("amount_log" if self.amount_log else self.amount_col)
        if self.cyclical_hour:
            names += ["hour_sin", "hour_cos"]
        self.feature_names_ = names
        self.v_cols_ = self._v_cols(X)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        parts = [X[self.v_cols_].to_numpy(dtype=float)]
        amount = X[self.amount_col].to_numpy(dtype=float)
        parts.append((np.log1p(amount) if self.amount_log else amount)[:, None])
        if self.cyclical_hour:
            hour = (X[self.time_col].to_numpy(dtype=float) / 3600.0) % 24
            angle = 2 * np.pi * hour / 24
            parts += [np.sin(angle)[:, None], np.cos(angle)[:, None]]
        return np.hstack(parts)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)


def build_preprocessor(cfg: dict) -> Pipeline:
    d, f = cfg["data"], cfg["features"]
    return Pipeline([
        ("features", FeatureBuilder(d["time_col"], d["amount_col"], f["amount_log"], f["cyclical_hour"])),
        ("scaler", StandardScaler()),
    ])


def chronological_split(df: pd.DataFrame, time_col: str, train_frac: float, val_frac: float):
    """Split by time: earliest -> train, then validation, latest -> test (mimics deployment)."""
    df = df.sort_values(time_col, kind="stable").reset_index(drop=True)
    n = len(df)
    i, j = int(n * train_frac), int(n * (train_frac + val_frac))
    return df.iloc[:i].copy(), df.iloc[i:j].copy(), df.iloc[j:].copy()
