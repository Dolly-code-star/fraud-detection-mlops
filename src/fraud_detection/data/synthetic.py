"""Synthetic transaction generator with the same schema as the Kaggle credit-card dataset.

Used as an offline fallback so the pipeline runs end-to-end with zero manual setup.
It is NOT a substitute for real data: it only mimics the layout (Time, V1..V28, Amount,
Class), extreme imbalance, and a plausible fraud signal so that every stage can be exercised.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_COMPONENTS = 28
SECONDS_PER_DAY = 86_400


def generate_synthetic(n_rows: int = 100_000, fraud_rate: float = 0.004, seed: int = 42, n_days: int = 14) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = max(int(round(n_rows * fraud_rate)), 1)
    y = np.zeros(n_rows, dtype=int)
    fraud_idx = rng.choice(n_rows, size=n_fraud, replace=False)
    y[fraud_idx] = 1
    is_fraud = y == 1

    # Anonymised components: decreasing variance, like PCA outputs.
    scales = np.linspace(1.9, 0.4, N_COMPONENTS)
    V = rng.normal(0.0, scales, size=(n_rows, N_COMPONENTS))

    # Fraud shifts a handful of components (V1, V3, V4, V10, V11, V12, V14, V17) and inflates variance.
    cols = np.array([0, 2, 3, 9, 10, 11, 13, 16])
    signs = np.array([-1, -1, 1, -1, 1, -1, -1, -1])
    shift = signs * scales[cols] * rng.uniform(0.8, 2.2, size=(n_fraud, len(cols)))
    V[np.ix_(fraud_idx, cols)] += shift
    V[fraud_idx] *= rng.uniform(1.0, 1.5, size=(n_fraud, 1))

    # Amount: log-normal; fraud is more heavy-tailed.
    amount = np.where(
        is_fraud,
        rng.lognormal(mean=3.6, sigma=1.8, size=n_rows),
        rng.lognormal(mean=3.2, sigma=1.4, size=n_rows),
    ).clip(0, 25_000).round(2)

    # Time (seconds over n_days days): legit follows a daily cycle, fraud skews to night hours.
    hour_w = 1.0 + 0.9 * np.sin((np.arange(24) - 8) / 24 * 2 * np.pi)
    hour_w /= hour_w.sum()
    legit_hour = rng.choice(24, size=n_rows, p=hour_w)
    night_hour = rng.choice(np.arange(0, 6), size=n_rows)
    fraud_hour = np.where(rng.random(n_rows) < 0.25, night_hour, rng.integers(0, 24, size=n_rows))
    hour = np.where(is_fraud, fraud_hour, legit_hour)
    day = rng.integers(0, n_days, size=n_rows)
    time = day * SECONDS_PER_DAY + hour * 3600 + rng.uniform(0, 3600, size=n_rows)

    df = pd.DataFrame(V, columns=[f"V{i}" for i in range(1, N_COMPONENTS + 1)])
    df.insert(0, "Time", time.round(0))
    df["Amount"] = amount
    df["Class"] = y
    return df.sort_values("Time", kind="stable").reset_index(drop=True)
