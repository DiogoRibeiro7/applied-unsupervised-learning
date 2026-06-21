from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from unsup_lab import api  # noqa: E402
from unsup_lab.cli import main  # noqa: E402


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    clustering_path = str(tmp_path / "clustering.joblib")
    anomaly_path = str(tmp_path / "anomaly.joblib")
    main(["train-clustering", "--n", "150", "--k", "4", "--model-out", clustering_path])
    main(["detect-anomalies", "--n", "400", "--model-out", anomaly_path])

    monkeypatch.setenv("UNSUP_LAB_MODEL_DIR", str(tmp_path))
    api._load.cache_clear()
    return TestClient(api.app)


def test_health(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_cluster_assign(client: TestClient) -> None:
    record = {
        "recency_days": 18.0,
        "purchase_frequency": 24.0,
        "avg_order_value": 190.0,
        "discount_ratio": 0.12,
        "email_engagement": 0.82,
        "product_diversity": 0.74,
    }

    response = client.post("/cluster/assign", json={"records": [record, record]})

    assert response.status_code == 200
    labels = response.json()["labels"]
    assert len(labels) == 2
    assert labels[0] == labels[1]


def test_anomaly_score(client: TestClient) -> None:
    record = {
        "temperature": 21.0,
        "motion_count": 4.0,
        "power_usage": 1.8,
        "signal_strength": -65.0,
        "missing_ratio": 0.05,
    }

    response = client.post("/anomaly/score", json={"records": [record]})

    assert response.status_code == 200
    assert len(response.json()["scores"]) == 1


def test_cluster_assign_missing_model_returns_503(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UNSUP_LAB_MODEL_DIR", str(tmp_path / "empty"))
    api._load.cache_clear()
    client = TestClient(api.app)

    response = client.post("/cluster/assign", json={"records": [{"a": 1.0}]})

    assert response.status_code == 503
