"""Imbalance-aware evaluation. Accuracy is deliberately NOT reported (99.6% by predicting 'legit')."""
from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def select_threshold(y: np.ndarray, scores: np.ndarray, strategy: str, alert_budget: float) -> float:
    """Pick the operating threshold on VALIDATION data. Flag a transaction if score >= threshold."""
    if strategy == "alert_budget":
        return float(np.quantile(scores, 1.0 - alert_budget))
    if strategy == "max_f1":
        p, r, thr = precision_recall_curve(y, scores)
        f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
        return float(thr[int(np.argmax(f1))])
    raise ValueError(f"Unknown threshold strategy {strategy!r}")


def compute_metrics(y: np.ndarray, scores: np.ndarray, threshold: float | None, alert_budget: float) -> dict:
    y = np.asarray(y).astype(int)
    out = {
        "n": int(len(y)),
        "n_fraud": int(y.sum()),
        "prevalence": float(y.mean()),
        "pr_auc": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
    }
    # Business view: how many frauds do we catch if analysts only review the top-k% riskiest?
    k = max(int(math.ceil(alert_budget * len(y))), 1)
    top = np.argsort(-scores, kind="stable")[:k]
    out[f"recall_at_{alert_budget:.0%}_budget"] = float(y[top].sum() / max(y.sum(), 1))
    out[f"precision_at_{alert_budget:.0%}_budget"] = float(y[top].mean())

    if threshold is not None:
        pred = (scores >= threshold).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        out.update(
            threshold=float(threshold), precision=prec, recall=rec,
            f1=2 * prec * rec / max(prec + rec, 1e-12), flagged_rate=float(pred.mean()),
            tp=tp, fp=fp, fn=fn, tn=tn,
        )
    return out
