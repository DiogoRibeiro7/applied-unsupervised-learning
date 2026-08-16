from __future__ import annotations

import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from unsup_lab.evaluation import evaluate_clustering, evaluate_k_range


def test_evaluate_clustering_returns_metrics() -> None:
    data = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]])
    labels = np.array([0, 0, 1, 1])

    metrics = evaluate_clustering(data, labels)

    assert metrics.n_clusters == 2
    assert metrics.silhouette is not None


def test_evaluate_clustering_single_cluster_has_none_metrics() -> None:
    data = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0]])
    labels = np.array([0, 0, 0])

    metrics = evaluate_clustering(data, labels)

    assert metrics.n_clusters == 1
    assert metrics.silhouette is None


def test_evaluate_clustering_excludes_noise_points() -> None:
    # Two tight clusters plus one far-away DBSCAN noise point.
    data = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0], [100.0, 100.0]])
    labels = np.array([0, 0, 1, 1, -1])

    metrics = evaluate_clustering(data, labels)

    assert metrics.n_clusters == 2
    # Noise is not a cluster: the score must match the noise-free subset.
    assert metrics.silhouette == pytest.approx(silhouette_score(data[:4], labels[:4]))


def test_evaluate_clustering_all_noise_but_two_points_has_none_metrics() -> None:
    data = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]])
    labels = np.array([-1, -1, 0, 1])

    metrics = evaluate_clustering(data, labels)

    assert metrics.n_clusters == 2
    assert metrics.silhouette is None


def test_evaluate_k_range() -> None:
    data = np.random.default_rng(1).normal(size=(50, 3))

    result = evaluate_k_range(
        data,
        estimator_factory=lambda k: KMeans(n_clusters=k, n_init=10, random_state=1),
        k_values=[2, 3],
    )

    assert list(result["k"]) == [2, 3]


def test_invalid_labels_shape_raises() -> None:
    data = np.random.default_rng(1).normal(size=(10, 2))

    with pytest.raises(ValueError):
        evaluate_clustering(data, np.array([[1, 2], [3, 4]]))
