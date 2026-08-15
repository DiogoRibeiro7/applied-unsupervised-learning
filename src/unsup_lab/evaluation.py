"""Evaluation utilities for unsupervised learning workflows."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import ClusterMixin
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


@dataclass(frozen=True)
class ClusteringMetrics:
    """Internal clustering quality metrics."""

    n_clusters: int
    silhouette: float | None
    davies_bouldin: float | None
    calinski_harabasz: float | None


def _as_2d_array(data: ArrayLike) -> np.ndarray:
    """Convert input data to a checked 2D NumPy array."""
    array = np.asarray(data)
    if array.ndim != 2:
        raise ValueError("data must be a 2D array.")
    if array.shape[0] < 2:
        raise ValueError("data must contain at least two rows.")
    return array


def evaluate_clustering(data: ArrayLike, labels: ArrayLike) -> ClusteringMetrics:
    """Evaluate clustering labels using internal validation metrics.

    Points carrying the DBSCAN noise label ``-1`` are excluded before the
    metrics are computed: noise is not a cluster, and scoring it as one distorts
    the result (a handful of scattered noise points can become the "nearest
    cluster" for many real points and depress the silhouette). The reported
    ``n_clusters`` counts non-noise clusters for the same reason.

    Metrics that require at least two non-noise clusters return `None`
    when they are not well-defined.
    """
    x = _as_2d_array(data)
    y = np.asarray(labels)

    if y.ndim != 1:
        raise ValueError("labels must be a 1D array.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("data and labels must have the same number of rows.")

    unique_labels = set(y.tolist())
    non_noise = [label for label in unique_labels if label != -1]
    n_clusters = len(non_noise)

    keep = y != -1
    if n_clusters < 2 or int(keep.sum()) <= n_clusters:
        return ClusteringMetrics(
            n_clusters=n_clusters,
            silhouette=None,
            davies_bouldin=None,
            calinski_harabasz=None,
        )

    x_scored, y_scored = x[keep], y[keep]
    return ClusteringMetrics(
        n_clusters=n_clusters,
        silhouette=float(silhouette_score(x_scored, y_scored)),
        davies_bouldin=float(davies_bouldin_score(x_scored, y_scored)),
        calinski_harabasz=float(calinski_harabasz_score(x_scored, y_scored)),
    )


def evaluate_k_range(
    data: ArrayLike,
    estimator_factory: Callable[[int], ClusterMixin],
    k_values: list[int],
) -> pd.DataFrame:
    """Evaluate a clustering estimator across a range of cluster counts.

    Parameters
    ----------
    data:
        Feature matrix.
    estimator_factory:
        Function that receives `k` and returns a fitted-compatible estimator
        with a `fit_predict` method.
    k_values:
        Candidate numbers of clusters.

    Returns
    -------
    pandas.DataFrame
        One row per candidate k.
    """
    x = _as_2d_array(data)
    if not k_values:
        raise ValueError("k_values cannot be empty.")

    rows: list[dict[str, float | int | None]] = []
    for k in k_values:
        if not isinstance(k, int) or k <= 1:
            raise ValueError("all k values must be integers greater than 1.")

        estimator: ClusterMixin = estimator_factory(k)
        labels = estimator.fit_predict(x)
        metrics = evaluate_clustering(x, labels)
        rows.append(
            {
                "k": k,
                "n_clusters": metrics.n_clusters,
                "silhouette": metrics.silhouette,
                "davies_bouldin": metrics.davies_bouldin,
                "calinski_harabasz": metrics.calinski_harabasz,
            }
        )

    return pd.DataFrame(rows)
