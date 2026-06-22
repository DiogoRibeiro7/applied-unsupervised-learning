"""Typed YAML configuration for reproducible runs.

A small, validated configuration object keeps experiment settings out of the
code and in version-controlled YAML files. This is deliberately minimal: one
dataclass with sane defaults and explicit validation, loaded from YAML.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

_VALID_TASKS = {"clustering", "anomaly", "topic"}


@dataclass
class RunConfig:
    """Configuration for a single modelling run.

    Attributes
    ----------
    task:
        One of ``"clustering"``, ``"anomaly"`` or ``"topic"``.
    n_samples:
        Number of synthetic rows / documents to generate.
    n_clusters:
        Number of clusters (clustering task).
    contamination:
        Expected anomaly fraction (anomaly task).
    n_topics:
        Number of topics (topic task).
    random_state:
        Global random seed.
    """

    task: str = "clustering"
    n_samples: int = 2000
    n_clusters: int = 5
    contamination: float = 0.04
    n_topics: int = 4
    random_state: int = 42

    def __post_init__(self) -> None:
        """Validate the configuration values."""
        if self.task not in _VALID_TASKS:
            raise ValueError(f"task must be one of {sorted(_VALID_TASKS)}.")
        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        if self.n_clusters < 2:
            raise ValueError("n_clusters must be at least 2.")
        if not 0.0 < self.contamination < 0.5:
            raise ValueError("contamination must be between 0 and 0.5.")
        if self.n_topics < 2:
            raise ValueError("n_topics must be at least 2.")


def load_config(path: str | Path) -> RunConfig:
    """Load and validate a :class:`RunConfig` from a YAML file.

    Unknown keys are rejected so typos fail loudly rather than being ignored.

    Parameters
    ----------
    path:
        Path to a YAML file containing a mapping of config fields.

    Returns
    -------
    RunConfig
        The validated configuration.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"No config file at {config_path}.")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("config file must contain a mapping at the top level.")

    allowed = set(RunConfig.__dataclass_fields__)
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}.")

    return RunConfig(**raw)


def dump_config(config: RunConfig, path: str | Path) -> Path:
    """Write a :class:`RunConfig` to a YAML file.

    Parameters
    ----------
    config:
        The configuration to serialise.
    path:
        Destination YAML file.

    Returns
    -------
    pathlib.Path
        The path written.
    """
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = asdict(config)
    config_path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    return config_path
