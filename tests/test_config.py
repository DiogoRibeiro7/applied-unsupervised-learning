from __future__ import annotations

from pathlib import Path

import pytest

from unsup_lab.config import RunConfig, dump_config, load_config


def test_load_valid_config(tmp_path: Path) -> None:
    path = tmp_path / "run.yaml"
    path.write_text("task: clustering\nn_clusters: 4\nrandom_state: 7\n", encoding="utf-8")

    config = load_config(path)

    assert config.task == "clustering"
    assert config.n_clusters == 4
    assert config.random_state == 7
    assert config.n_samples == 2000  # default preserved


def test_unknown_key_raises(tmp_path: Path) -> None:
    path = tmp_path / "run.yaml"
    path.write_text("task: clustering\ntypo_key: 1\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_config(path)


def test_invalid_task_raises(tmp_path: Path) -> None:
    path = tmp_path / "run.yaml"
    path.write_text("task: regression\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_config(path)


def test_invalid_values_raise() -> None:
    with pytest.raises(ValueError):
        RunConfig(n_clusters=1)
    with pytest.raises(ValueError):
        RunConfig(contamination=0.9)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "absent.yaml")


def test_dump_then_load_roundtrip(tmp_path: Path) -> None:
    config = RunConfig(task="anomaly", contamination=0.06)
    path = dump_config(config, tmp_path / "out.yaml")

    reloaded = load_config(path)

    assert reloaded == config


def test_example_config_is_valid() -> None:
    config = load_config(Path("configs/example_clustering.yaml"))

    assert config.task == "clustering"
