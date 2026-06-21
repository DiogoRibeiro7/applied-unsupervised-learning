from __future__ import annotations

from pathlib import Path

import pytest

from unsup_lab.artifacts import ArtifactMetadata, load_artifact, save_artifact
from unsup_lab.data import make_customer_segmentation_data
from unsup_lab.service import train_clustering


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    dataset = make_customer_segmentation_data(n_customers=100, random_state=0)
    pipeline, _ = train_clustering(dataset.features, n_clusters=4, random_state=0)
    model_path = tmp_path / "clustering.joblib"

    returned = save_artifact(
        pipeline,
        model_path,
        model_kind="clustering",
        parameters={"k": 4},
        metrics={"silhouette": 0.5},
    )

    assert returned == model_path
    assert model_path.exists()
    assert model_path.with_suffix(".json").exists()

    loaded, metadata = load_artifact(model_path)
    assert isinstance(metadata, ArtifactMetadata)
    assert metadata.model_kind == "clustering"
    assert metadata.parameters["k"] == 4
    # The reloaded pipeline still predicts.
    assert loaded.predict(dataset.features.to_numpy()).shape == (100,)


def test_load_missing_model_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_artifact(tmp_path / "nope.joblib")


def test_load_missing_metadata_raises(tmp_path: Path) -> None:
    model_path = tmp_path / "model.joblib"
    import joblib

    joblib.dump({"x": 1}, model_path)

    with pytest.raises(FileNotFoundError):
        load_artifact(model_path)
