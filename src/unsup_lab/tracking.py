"""A lightweight, dependency-free experiment tracker.

Full tracking systems (MLflow, Weights & Biases) are overkill for a portfolio
lab and add heavy dependencies. This module records each run as one JSON line in
an append-only log, which is enough to compare parameters and metrics across
runs, find the best run, and keep an audit trail - using only the standard
library.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_DEFAULT_LOG = Path("outputs/experiments/runs.jsonl")


@dataclass
class RunRecord:
    """A single experiment run."""

    name: str
    created_at: str
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def log_run(
    name: str,
    params: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    path: str | Path = _DEFAULT_LOG,
) -> RunRecord:
    """Append a run to the JSONL experiment log.

    Parameters
    ----------
    name:
        Short identifier for the run (for example the model kind).
    params, metrics:
        Optional parameter and metric dictionaries.
    path:
        Destination JSONL file; parent directories are created as needed.

    Returns
    -------
    RunRecord
        The record that was written.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("name must be a non-empty string.")

    record = RunRecord(
        name=name,
        created_at=_utc_now_iso(),
        params=params or {},
        metrics=metrics or {},
    )
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(asdict(record)) + "\n")
    return record


def load_runs(path: str | Path = _DEFAULT_LOG) -> pd.DataFrame:
    """Load the experiment log into a flat DataFrame.

    Parameters and metrics are flattened with ``param_`` and ``metric_``
    prefixes so runs with different keys line up in one table.

    Parameters
    ----------
    path:
        JSONL experiment log.

    Returns
    -------
    pandas.DataFrame
        One row per run; empty if the log does not exist.
    """
    log_path = Path(path)
    if not log_path.exists():
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        flat: dict[str, Any] = {
            "name": record.get("name"),
            "created_at": record.get("created_at"),
        }
        for key, value in (record.get("params") or {}).items():
            flat[f"param_{key}"] = value
        for key, value in (record.get("metrics") or {}).items():
            flat[f"metric_{key}"] = value
        rows.append(flat)
    return pd.DataFrame(rows)


def best_run(
    metric: str,
    path: str | Path = _DEFAULT_LOG,
    mode: str = "max",
) -> pd.Series:
    """Return the run that optimises a metric.

    Parameters
    ----------
    metric:
        Metric name *without* the ``metric_`` prefix.
    path:
        JSONL experiment log.
    mode:
        ``"max"`` to maximise the metric or ``"min"`` to minimise it.

    Returns
    -------
    pandas.Series
        The winning run as a row.
    """
    if mode not in {"max", "min"}:
        raise ValueError("mode must be 'max' or 'min'.")

    runs = load_runs(path)
    column = f"metric_{metric}"
    if runs.empty or column not in runs.columns:
        raise ValueError(f"no runs with metric {metric!r} were found.")

    valid = runs.dropna(subset=[column])
    if valid.empty:
        raise ValueError(f"no runs have a non-null value for metric {metric!r}.")

    values = valid[column].to_numpy()
    position = int(np.argmax(values)) if mode == "max" else int(np.argmin(values))
    return valid.iloc[position]
