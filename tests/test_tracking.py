from __future__ import annotations

from pathlib import Path

import pytest

from unsup_lab.tracking import RunRecord, best_run, load_runs, log_run


def test_log_and_load_roundtrip(tmp_path: Path) -> None:
    log = tmp_path / "runs.jsonl"

    record = log_run("clustering", params={"k": 5}, metrics={"silhouette": 0.4}, path=log)
    log_run("clustering", params={"k": 6}, metrics={"silhouette": 0.5}, path=log)

    assert isinstance(record, RunRecord)
    runs = load_runs(log)
    assert len(runs) == 2
    assert list(runs["param_k"]) == [5, 6]
    assert "metric_silhouette" in runs.columns


def test_load_missing_log_returns_empty(tmp_path: Path) -> None:
    runs = load_runs(tmp_path / "absent.jsonl")

    assert runs.empty


def test_best_run_max_and_min(tmp_path: Path) -> None:
    log = tmp_path / "runs.jsonl"
    log_run("a", metrics={"silhouette": 0.4, "davies_bouldin": 1.2}, path=log)
    log_run("b", metrics={"silhouette": 0.6, "davies_bouldin": 0.8}, path=log)

    assert best_run("silhouette", path=log, mode="max")["name"] == "b"
    assert best_run("davies_bouldin", path=log, mode="min")["name"] == "b"


def test_best_run_unknown_metric_raises(tmp_path: Path) -> None:
    log = tmp_path / "runs.jsonl"
    log_run("a", metrics={"silhouette": 0.4}, path=log)

    with pytest.raises(ValueError):
        best_run("missing", path=log)


def test_best_run_invalid_mode_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        best_run("silhouette", path=tmp_path / "runs.jsonl", mode="highest")


def test_log_run_empty_name_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        log_run("", path=tmp_path / "runs.jsonl")
