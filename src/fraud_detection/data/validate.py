"""Schema and data-quality validation. Errors stop the pipeline; warnings are only reported."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


class DataValidationError(RuntimeError):
    """Raised when the dataset violates the schema in a way that makes training unsafe."""


@dataclass
class ValidationReport:
    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))


def load_schema(path: str | Path) -> dict:
    schema = yaml.safe_load(Path(path).read_text())
    cols = list(schema.get("columns", []))
    for grp in schema.get("column_groups", []):
        for i in range(grp["start"], grp["end"] + 1):
            spec = {k: v for k, v in grp.items() if k not in ("prefix", "start", "end")}
            spec["name"] = f"{grp['prefix']}{i}"
            cols.append(spec)
    schema["expanded_columns"] = cols
    return schema


def validate_dataframe(df: pd.DataFrame, schema: dict, vcfg: dict) -> ValidationReport:
    rep = ValidationReport()
    err, warn = rep.errors, rep.warnings
    target = schema["target"]["name"]

    # 1. Structure -----------------------------------------------------------------
    expected = [c["name"] for c in schema["expanded_columns"]] + [target]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        err.append(f"Missing required columns: {missing}")
    extra = [c for c in df.columns if c not in expected]
    if extra:
        warn.append(f"Unexpected extra columns (ignored): {extra}")
    if len(df) < vcfg["min_rows"]:
        err.append(f"Too few rows: {len(df)} < {vcfg['min_rows']}")

    # 2. Per-column checks (dtype, nulls, finiteness, ranges) ----------------------
    for spec in schema["expanded_columns"]:
        name = spec["name"]
        if name not in df.columns:
            continue
        s = df[name]
        if spec.get("kind") == "numeric" and (
            not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)
        ):
            err.append(f"{name}: expected numeric dtype, got {s.dtype}")
            continue
        n_null = int(s.isna().sum())
        if n_null and not spec.get("nullable", False):
            err.append(f"{name}: {n_null} null values but column is not nullable")
        vals = s.dropna().to_numpy(dtype=float)
        if vals.size and not np.isfinite(vals).all():
            err.append(f"{name}: contains non-finite values (inf)")
            continue
        if vals.size and "min" in spec and vals.min() < spec["min"]:
            err.append(f"{name}: min {vals.min():.4g} < allowed {spec['min']}")
        if vals.size and "max" in spec and vals.max() > spec["max"]:
            err.append(f"{name}: max {vals.max():.4g} > allowed {spec['max']}")

    # 3. Target ----------------------------------------------------------------------
    if target in df.columns:
        bad = set(pd.unique(df[target].dropna())) - set(schema["target"]["allowed"])
        if bad or df[target].isna().any():
            err.append(f"{target}: invalid or null labels {sorted(bad, key=str)}")
        else:
            rate = float(df[target].mean())
            lo, hi = vcfg["fraud_rate_bounds"]
            rep.stats["fraud_rate"] = rate
            rep.stats["n_fraud"] = int(df[target].sum())
            if not lo <= rate <= hi:
                err.append(f"Fraud rate {rate:.5f} outside expected bounds [{lo}, {hi}]")

    # 4. Soft checks -----------------------------------------------------------------
    n_dup = int(df.duplicated().sum())
    if n_dup:
        warn.append(f"{n_dup} fully duplicated rows")
    rep.stats.update(n_rows=int(len(df)), n_columns=int(df.shape[1]), n_duplicates=n_dup)

    rep.passed = not err
    return rep