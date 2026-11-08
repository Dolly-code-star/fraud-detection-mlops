"""Data ingestion: load the Kaggle CSV if present, otherwise generate the synthetic dataset."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

from fraud_detection.config import resolve_path
from fraud_detection.data.synthetic import generate_synthetic

log = logging.getLogger(__name__)


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    """Content hash used to version the dataset in experiment tracking."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()[:16]


def load_dataset(cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Return ``(dataframe, info)`` where info records provenance (source, path, hash)."""
    source = cfg["data"]["source"]
    raw = resolve_path(cfg, "raw_data")
    synth_path = resolve_path(cfg, "synthetic_data")

    if source == "kaggle_csv" and not raw.exists():
        raise FileNotFoundError(f"data.source=kaggle_csv but {raw} does not exist (see README).")

    if source in ("auto", "kaggle_csv") and raw.exists():
        path, label = raw, "kaggle_creditcard"
        df = pd.read_csv(path)
    else:
        spec = cfg["data"]["synthetic"]
        df = generate_synthetic(spec["n_rows"], spec["fraud_rate"], cfg["project"]["seed"], spec.get("n_days", 14))
        synth_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(synth_path, index=False)
        path, label = synth_path, "synthetic"
        log.warning("Real dataset not found -> using SYNTHETIC data (%s rows).", len(df))

    info = {"source": label, "path": str(path), "sha256": file_sha256(path), "n_rows": int(len(df))}
    log.info("Loaded %s rows from %s (%s)", info["n_rows"], label, path)
    return df, info
