from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from unsup_lab.artifacts import load_artifact
from unsup_lab.config import RunConfig, dump_config

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "scheduled_report.py"


def _load_script() -> ModuleType:
    """Import the scheduled-report script by path (it lives outside the package)."""
    spec = importlib.util.spec_from_file_location("scheduled_report", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scheduled_report_keeps_the_artifact_loadable(tmp_path: Path) -> None:
    # The report must not be written over the artifact's metadata sidecar,
    # which would leave the saved model unloadable.
    config_path = dump_config(
        RunConfig(task="clustering", n_samples=150, n_clusters=3), tmp_path / "run.yaml"
    )
    output_dir = tmp_path / "scheduled"

    report_path = _load_script().run_scheduled_report(str(config_path), output_dir)

    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["model_kind"] == "clustering"

    (model_path,) = output_dir.glob("*.joblib")
    _, metadata = load_artifact(model_path)
    assert metadata.model_kind == "clustering"
    assert metadata.parameters["k"] == 3


def test_scheduled_report_honours_the_config_seed(tmp_path: Path) -> None:
    # `--random-state` is a top-level CLI option; if the script appended it
    # after the subcommand (or omitted it) the config seed would be ignored.
    seeds = []
    for seed in (7, 99):
        config_path = dump_config(
            RunConfig(task="clustering", n_samples=150, n_clusters=3, random_state=seed),
            tmp_path / f"run_{seed}.yaml",
        )
        output_dir = tmp_path / f"scheduled_{seed}"
        _load_script().run_scheduled_report(str(config_path), output_dir)

        (model_path,) = output_dir.glob("*.joblib")
        _, metadata = load_artifact(model_path)
        seeds.append(metadata.parameters["random_state"])

    assert seeds == [7, 99]


def test_scheduled_report_rejects_a_missing_config(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _load_script().run_scheduled_report(str(tmp_path / "absent.yaml"), tmp_path)
