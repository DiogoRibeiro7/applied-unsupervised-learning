from __future__ import annotations

import numpy as np
import pytest

from unsup_lab.graphs import (
    graph_anomaly_scores,
    make_block_graph,
    node_structural_features,
    spectral_node_embedding,
)


def _graph_with_bridge_nodes(
    n_bridges: int = 4, random_state: int = 0
) -> tuple[object, np.ndarray]:
    """An SBM plus a few 'bridge' nodes wired across all communities."""
    adjacency, _ = make_block_graph([40, 40, 40], p_in=0.4, p_out=0.02, random_state=random_state)
    dense = adjacency.toarray()
    n = dense.shape[0]
    rng = np.random.default_rng(random_state)

    is_anomaly = np.zeros(n, dtype=bool)
    bridges = rng.choice(n, size=n_bridges, replace=False)
    for node in bridges:
        # Connect the bridge node broadly across the whole graph.
        targets = rng.choice(n, size=25, replace=False)
        dense[node, targets] = 1
        dense[targets, node] = 1
        dense[node, node] = 0
        is_anomaly[node] = True

    from scipy.sparse import csr_matrix

    return csr_matrix(dense), is_anomaly


def test_structural_features_schema_and_values() -> None:
    adjacency, _ = make_block_graph([20, 20], p_in=0.5, p_out=0.05, random_state=0)

    features = node_structural_features(adjacency)

    assert list(features.columns) == [
        "degree",
        "clustering_coefficient",
        "avg_neighbor_degree",
    ]
    assert len(features) == 40
    assert (features["clustering_coefficient"] >= 0).all()
    assert (features["clustering_coefficient"] <= 1).all()


def test_structural_features_requires_sparse() -> None:
    with pytest.raises(TypeError):
        node_structural_features(np.eye(3))  # type: ignore[arg-type]


def test_graph_anomaly_ranks_bridges_high() -> None:
    adjacency, is_anomaly = _graph_with_bridge_nodes(n_bridges=4, random_state=1)

    scores = graph_anomaly_scores(adjacency, contamination=0.05, random_state=0)
    top = np.argsort(scores)[::-1][:4]
    precision_at_4 = is_anomaly[top].mean()

    assert precision_at_4 >= 0.75  # most of the top-scored nodes are real bridges


def test_graph_anomaly_invalid_contamination_raises() -> None:
    adjacency, _ = make_block_graph([10, 10], p_in=0.5, p_out=0.05, random_state=0)

    with pytest.raises(ValueError):
        graph_anomaly_scores(adjacency, contamination=0.8)


def test_spectral_embedding_shape() -> None:
    adjacency, _ = make_block_graph([20, 20, 20], p_in=0.4, p_out=0.03, random_state=0)

    embedding = spectral_node_embedding(adjacency, n_components=2, random_state=0)

    assert embedding.shape == (60, 2)


def test_spectral_embedding_invalid_components_raises() -> None:
    adjacency, _ = make_block_graph([10, 10], p_in=0.5, p_out=0.05, random_state=0)

    with pytest.raises(ValueError):
        spectral_node_embedding(adjacency, n_components=100)
