from __future__ import annotations

import numpy as np
import pytest
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score

from unsup_lab.consensus import (
    ConsensusResult,
    build_coassociation_matrix,
    consensus_clustering,
)


def _three_blobs(random_state: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    centres = np.array([[0.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    blocks, truth = [], []
    for label, centre in enumerate(centres):
        blocks.append(centre + rng.normal(scale=0.4, size=(40, 2)))
        truth.extend([label] * 40)
    return np.vstack(blocks), np.array(truth)


def _kmeans_factory(k: int):
    return lambda seed: KMeans(n_clusters=k, n_init=10, random_state=seed)


def test_coassociation_matrix_is_valid() -> None:
    data, _ = _three_blobs()

    matrix = build_coassociation_matrix(data, _kmeans_factory(3), n_runs=10, random_state=0)

    assert matrix.shape == (120, 120)
    np.testing.assert_allclose(matrix, matrix.T)  # symmetric
    np.testing.assert_allclose(np.diag(matrix), 1.0)  # unit diagonal
    assert matrix.min() >= 0.0 and matrix.max() <= 1.0


def test_consensus_recovers_known_groups() -> None:
    data, truth = _three_blobs()

    result = consensus_clustering(data, n_clusters=3, base_factory=_kmeans_factory(3), n_runs=15)

    assert isinstance(result, ConsensusResult)
    assert adjusted_rand_score(truth, result.labels) > 0.95
    assert result.stability > 0.9


def test_consensus_is_deterministic() -> None:
    data, _ = _three_blobs()

    a = consensus_clustering(data, 3, _kmeans_factory(3), n_runs=8, random_state=1)
    b = consensus_clustering(data, 3, _kmeans_factory(3), n_runs=8, random_state=1)

    np.testing.assert_array_equal(a.labels, b.labels)
    np.testing.assert_allclose(a.coassociation, b.coassociation)


def test_consensus_handles_single_cluster_base() -> None:
    data, _ = _three_blobs()
    # eps huge -> every base run yields a single cluster; co-association is all 1.
    factory = lambda seed: DBSCAN(eps=1000.0, min_samples=2)  # noqa: E731

    result = consensus_clustering(data, n_clusters=3, base_factory=factory, n_runs=4)

    assert result.labels.shape == (120,)
    assert result.coassociation.min() >= 0.0


def test_invalid_n_clusters_raises() -> None:
    data, _ = _three_blobs()

    with pytest.raises(ValueError):
        consensus_clustering(data, n_clusters=1, base_factory=_kmeans_factory(3))


def test_invalid_linkage_raises() -> None:
    data, _ = _three_blobs()

    with pytest.raises(ValueError):
        consensus_clustering(data, n_clusters=3, base_factory=_kmeans_factory(3), linkage="ward")


def test_invalid_sample_fraction_raises() -> None:
    data, _ = _three_blobs()

    with pytest.raises(ValueError):
        build_coassociation_matrix(data, _kmeans_factory(3), sample_fraction=1.5)
