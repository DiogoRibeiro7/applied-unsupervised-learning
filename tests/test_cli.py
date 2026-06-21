from __future__ import annotations

import json
from pathlib import Path

from unsup_lab.cli import main


def test_generate_data_writes_csv(tmp_path: Path) -> None:
    output = tmp_path / "customers.csv"

    exit_code = main(
        ["generate-data", "--dataset", "customers", "--n", "50", "--output", str(output)]
    )

    assert exit_code == 0
    assert output.exists()
    assert output.read_text(encoding="utf-8").count("\n") >= 50


def test_train_clustering_creates_artifact_and_report(tmp_path: Path) -> None:
    model_out = tmp_path / "clustering.joblib"
    report_out = tmp_path / "clustering.json"

    main(
        [
            "train-clustering",
            "--n",
            "150",
            "--k",
            "4",
            "--model-out",
            str(model_out),
            "--report-out",
            str(report_out),
        ]
    )

    assert model_out.exists()
    assert model_out.with_suffix(".json").exists()
    report = json.loads(report_out.read_text(encoding="utf-8"))
    assert report["model_kind"] == "clustering"
    assert report["k"] == 4
    assert len(report["cluster_sizes"]) == 4


def test_detect_anomalies_reports_precision(tmp_path: Path) -> None:
    report_out = tmp_path / "anomaly.json"

    main(
        [
            "detect-anomalies",
            "--n",
            "600",
            "--contamination",
            "0.05",
            "--top-k",
            "30",
            "--model-out",
            str(tmp_path / "anomaly.joblib"),
            "--report-out",
            str(report_out),
        ]
    )

    report = json.loads(report_out.read_text(encoding="utf-8"))
    assert report["model_kind"] == "anomaly"
    assert 0.0 <= report["precision_at_k"] <= 1.0
    assert len(report["top_anomaly_indices"]) == 30


def test_build_topic_model_report(tmp_path: Path) -> None:
    report_out = tmp_path / "topic.json"

    main(
        [
            "build-topic-model",
            "--n-topics",
            "4",
            "--model-out",
            str(tmp_path / "topic.joblib"),
            "--report-out",
            str(report_out),
        ]
    )

    report = json.loads(report_out.read_text(encoding="utf-8"))
    assert len(report["topics"]) == 4
