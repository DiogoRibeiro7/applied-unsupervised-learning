"""Minimal FastAPI service for cluster assignment and anomaly scoring.

The service loads model artifacts produced by the CLI (``applied-unsupervised-learning
train-clustering`` / ``applied-unsupervised-learning detect-anomalies``) and exposes them over HTTP.
Model paths are configured through the ``UNSUP_LAB_MODEL_DIR`` environment
variable and resolved lazily, so the app starts even before models are trained
and returns a clear error if a model is requested before it exists.

Run locally with::

    uvicorn unsup_lab.api:app --reload
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from unsup_lab.artifacts import load_artifact
from unsup_lab.service import anomaly_scores, assign_clusters

app = FastAPI(title="Applied Unsupervised Learning", version="0.2.0")


def _model_dir() -> Path:
    """Resolve the directory holding model artifacts."""
    return Path(os.environ.get("UNSUP_LAB_MODEL_DIR", "outputs/models"))


@lru_cache(maxsize=8)
def _load(model_name: str) -> Any:
    """Load and cache a model artifact by file name."""
    path = _model_dir() / model_name
    try:
        model, _ = load_artifact(path)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return model


class FeatureRequest(BaseModel):
    """A batch of feature rows as a list of column->value records."""

    records: list[dict[str, float]] = Field(..., min_length=1)


class ClusterResponse(BaseModel):
    """Cluster labels for each submitted row."""

    labels: list[int]


class AnomalyResponse(BaseModel):
    """Anomaly scores for each submitted row (higher means more anomalous)."""

    scores: list[float]


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/cluster/assign", response_model=ClusterResponse)
def cluster_assign(request: FeatureRequest) -> ClusterResponse:
    """Assign incoming rows to clusters using the trained clustering model."""
    pipeline = _load("clustering.joblib")
    frame = pd.DataFrame(request.records)
    try:
        labels = assign_clusters(pipeline, frame)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return ClusterResponse(labels=[int(label) for label in labels])


@app.post("/anomaly/score", response_model=AnomalyResponse)
def anomaly_score(request: FeatureRequest) -> AnomalyResponse:
    """Score incoming rows for anomalousness using the trained anomaly model."""
    pipeline = _load("anomaly.joblib")
    frame = pd.DataFrame(request.records)
    try:
        scores = anomaly_scores(pipeline, frame)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return AnomalyResponse(scores=[float(score) for score in scores])
