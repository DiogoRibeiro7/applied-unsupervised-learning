"""Model artifact persistence with metadata.

A trained model is only useful in production if it can be saved, described and
reloaded reliably. Each artifact is stored as a pair of files: a joblib pickle
holding the fitted estimator and a sidecar JSON file holding human-readable
metadata (model kind, parameters, training shape, timestamp). The sidecar makes
it possible to audit a model without unpickling it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib


@dataclass
class ArtifactMetadata:
    """Human-readable description of a saved model artifact."""

    model_kind: str
    created_at: str
    parameters: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def save_artifact(
    model: Any,
    path: str | Path,
    model_kind: str,
    parameters: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Persist a fitted model and its metadata sidecar.

    Parameters
    ----------
    model:
        The fitted estimator or pipeline.
    path:
        Destination path for the joblib file. A ``.json`` sidecar with the same
        stem is written alongside it.
    model_kind:
        Short identifier such as ``"clustering"`` or ``"anomaly"``.
    parameters, metrics, extra:
        Optional metadata dictionaries.

    Returns
    -------
    pathlib.Path
        The path to the saved model file.
    """
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = ArtifactMetadata(
        model_kind=model_kind,
        created_at=_utc_now_iso(),
        parameters=parameters or {},
        metrics=metrics or {},
        extra=extra or {},
    )

    joblib.dump(model, model_path)
    metadata_path = model_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(asdict(metadata), indent=2), encoding="utf-8")
    return model_path


def load_artifact(path: str | Path) -> tuple[Any, ArtifactMetadata]:
    """Load a model and its metadata sidecar.

    Parameters
    ----------
    path:
        Path to the joblib model file.

    Returns
    -------
    tuple
        The loaded model and its :class:`ArtifactMetadata`.
    """
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"No model artifact at {model_path}.")

    metadata_path = model_path.with_suffix(".json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"No metadata sidecar at {metadata_path}.")

    model = joblib.load(model_path)
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata = ArtifactMetadata(**raw)
    return model, metadata
