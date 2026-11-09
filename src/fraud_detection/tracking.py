"""Experiment tracking with a safe fallback.

* backend ``mlflow`` (or ``auto`` when mlflow is installed): logs to a local file store
  ``artifacts/mlruns`` (view with ``mlflow ui --backend-store-uri artifacts/mlruns``).
* backend ``local`` (or ``auto`` without mlflow): writes ``artifacts/experiments/<run_id>/`` and
  appends a summary line to ``artifacts/experiments/runs.jsonl``. No extra dependencies.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)


class ExperimentTracker:
    def __init__(self, backend: str, experiment: str, artifacts_dir: Path, run_tag: str = ""):
        self.dir = Path(artifacts_dir)
        self.experiment = experiment
        self.params: dict = {}
        self.metrics: dict = {}
        self.files: list[Path] = []
        self.run_id = f"run_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{run_tag}"
        self._mlflow = None

        if backend in ("auto", "mlflow"):
            try:
                import mlflow  # noqa: WPS433 (optional dependency)

                mlflow.set_tracking_uri((self.dir / "mlruns").resolve().as_uri())
                mlflow.set_experiment(experiment)
                mlflow.start_run(run_name=self.run_id)
                self._mlflow = mlflow
            except ImportError:
                if backend == "mlflow":
                    raise
                log.info("mlflow not installed -> using local JSON tracker")
        self.backend = "mlflow" if self._mlflow else "local"

    def log_params(self, params: dict) -> None:
        self.params.update({k: v for k, v in params.items()})
        if self._mlflow:
            self._mlflow.log_params({k: str(v)[:250] for k, v in params.items()})

    def log_metrics(self, metrics: dict, prefix: str = "") -> None:
        flat = {f"{prefix}{k}": float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        self.metrics.update(flat)
        if self._mlflow:
            self._mlflow.log_metrics({k.replace("%", "pct"): v for k, v in flat.items()})

    def log_artifact(self, path: Path) -> None:
        self.files.append(Path(path))
        if self._mlflow:
            self._mlflow.log_artifact(str(path))

    def finish(self) -> Path:
        run_dir = self.dir / "experiments" / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "run_id": self.run_id, "experiment": self.experiment, "backend": self.backend,
            "params": self.params, "metrics": self.metrics,
            "artifacts": [str(p) for p in self.files],
        }
        (run_dir / "run.json").write_text(json.dumps(record, indent=2, default=str))
        with open(self.dir / "experiments" / "runs.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
        if self._mlflow:
            self._mlflow.end_run()
        return run_dir
