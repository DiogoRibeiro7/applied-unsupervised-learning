from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.metrics import adjusted_rand_score

from unsup_lab.graphs import (
    GraphCommunityResult,
    best_partition_by_modularity,
    build_knn_graph,
    detect_communities,
    make_block_graph,
    modularity,
)


def test_block_graph_is_symmetric_with_planted_labels() -> None:
    adjacency, labels = make_block_graph([20, 20, 20], p_in=0.5, p_out=0.02, random_state=0)

    dense = adjacency.toarray()
    assert adjacency.shape == (60, 60)
    np.testing.assert_array_equal(dense, dense.T)
    np.testing.assert_array_equal(np.diag(dense), 0)
    assert set(labels.tolist()) == {0, 1, 2}


def test_block_graph_invalid_probabilities_raise() -> None:
    with pytest.raises(ValueError):
        make_block_graph([10, 10], p_in=0.1, p_out=0.5)  # p_out > p_in


def test_modularity_single_community_is_zero() -> None:
    adjacency, _ = make_block_graph([15, 15], p_in=0.5, p_out=0.05, random_state=1)

    q = modularity(adjacency, np.zeros(30, dtype=int))

    assert q == pytest.approx(0.0, abs=1e-9)


def test_modularity_prefers_true_partition_over_random() -> None:
    adjacency, labels = make_block_graph([25, 25, 25], p_in=0.5, p_out=0.02, random_state=2)
    rng = np.random.default_rng(0)
    random_labels = rng.integers(0, 3, size=labels.size)

    assert modularity(adjacency, labels) > modularity(adjacency, random_labels)


def test_modularity_requires_sparse() -> None:
    with pytest.raises(TypeError):
        modularity(np.eye(3), [0, 1, 0])  # type: ignore[arg-type]


def test_modularity_no_edges_raises() -> None:
    empty = csr_matrix((4, 4))

    with pytest.raises(ValueError):
        modularity(empty, [0, 1, 0, 1])


def test_detect_communities_recovers_blocks() -> None:
    adjacency, labels = make_block_graph([30, 30, 30], p_in=0.45, p_out=0.02, random_state=3)

    result = detect_communities(adjacency, n_communities=3, random_state=0)

    assert isinstance(result, GraphCommunityResult)
    assert adjusted_rand_score(labels, result.labels) > 0.8
    assert result.modularity > 0.4


def test_best_partition_selects_true_count() -> None:
    adjacency, labels = make_block_graph([30, 30, 30], p_in=0.45, p_out=0.02, random_state=4)

    best = best_partition_by_modularity(adjacency, candidate_counts=[2, 3, 4, 5], random_state=0)

    assert best.n_communities == 3
    assert adjusted_rand_score(labels, best.labels) > 0.8


def test_detect_invalid_n_raises() -> None:
    adjacency, _ = make_block_graph([10, 10], p_in=0.5, p_out=0.05, random_state=0)

    with pytest.raises(ValueError):
        detect_communities(adjacency, n_communities=1)


def test_build_knn_graph_is_symmetric() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(size=(40, 3))

    graph = build_knn_graph(data, n_neighbors=5)

    dense = graph.toarray()
    assert graph.shape == (40, 40)
    np.testing.assert_array_equal(dense, dense.T)
    np.testing.assert_array_equal(np.diag(dense), 0)


def test_build_knn_graph_invalid_neighbors_raises() -> None:
    data = np.zeros((5, 2))

    with pytest.raises(ValueError):
        build_knn_graph(data, n_neighbors=10)
