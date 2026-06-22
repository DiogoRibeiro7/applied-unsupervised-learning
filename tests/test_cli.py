from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from unsup_lab.cli import main
from unsup_lab.tracking import load_runs


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


def test_train_clustering_logs_run_when_tracked(tmp_path: Path) -> None:
    track = tmp_path / "runs.jsonl"

    main(
        [
            "train-clustering",
            "--n",
            "120",
            "--k",
            "3",
            "--model-out",
            str(tmp_path / "clustering.joblib"),
            "--report-out",
            str(tmp_path / "clustering.json"),
            "--track-path",
            str(track),
        ]
    )

    runs = load_runs(track)
    assert len(runs) == 1
    assert runs.iloc[0]["name"] == "clustering"
    assert runs.iloc[0]["param_k"] == 3


def test_batch_score_clustering(tmp_path: Path) -> None:
    model_out = tmp_path / "clustering.joblib"
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
            str(tmp_path / "report.json"),
        ]
    )

    # Build a small input CSV with the customer feature columns.
    input_csv = tmp_path / "input.csv"
    output_csv = tmp_path / "scored.csv"
    record = {
        "recency_days": 18.0,
        "purchase_frequency": 24.0,
        "avg_order_value": 190.0,
        "discount_ratio": 0.12,
        "email_engagement": 0.82,
        "product_diversity": 0.74,
    }
    pd.DataFrame([record, record]).to_csv(input_csv, index=False)

    main(["batch-score", "--model", str(model_out), "--input", str(input_csv),
          "--output", str(output_csv)])

    scored = pd.read_csv(output_csv)
    assert "cluster" in scored.columns
    assert len(scored) == 2
