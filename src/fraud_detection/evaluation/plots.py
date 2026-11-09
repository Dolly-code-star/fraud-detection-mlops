"""Evaluation figures saved to artifacts/figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402


def plot_pr_curves(val_results: dict, y_val, selected: str, y_test, test_scores, out: Path) -> None:
    """Left: PR curves for all candidates (validation). Right: selected model on the test set."""
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, scores in val_results.items():
        p, r, _ = precision_recall_curve(y_val, scores)
        axes[0].plot(r, p, label=name, lw=2 if name == selected else 1)
    axes[0].axhline(y_val.mean(), ls=":", c="grey", label="prevalence")
    axes[0].set(title="Validation PR curves (all candidates)", xlabel="Recall", ylabel="Precision")
    axes[0].legend(fontsize=8)

    p, r, _ = precision_recall_curve(y_test, test_scores)
    axes[1].plot(r, p, lw=2, label=selected)
    axes[1].axhline(y_test.mean(), ls=":", c="grey", label="prevalence")
    axes[1].set(title=f"Test PR curve - {selected}", xlabel="Recall", ylabel="Precision")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
