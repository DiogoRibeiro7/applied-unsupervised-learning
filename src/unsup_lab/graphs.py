"""Graph-based unsupervised learning: community detection.

Many datasets are naturally relational - a similarity graph over customers, a
co-purchase network, an interaction graph. Community detection finds groups of
nodes that are densely connected to each other and sparsely connected to the
rest, which is the graph analogue of clustering.

This module builds graphs (a planted-community stochastic block model for
demonstration, and a k-nearest-neighbour graph from feature vectors), scores a
partition with Newman modularity, and detects communities with spectral
clustering. The number of communities can be chosen by maximising modularity.

It depends only on NumPy, SciPy and scikit-learn - no dedicated graph library.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_matrix, issparse
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import kneighbors_graph


@dataclass(frozen=True)
class GraphCommunityResult:
    """Outcome of community detection.

    Attributes
    ----------
    labels:
        Community label per node.
    n_communities:
        Number of communities in the partition.
    modularity:
        Newman modularity of the partition, in roughly ``[-0.5, 1]``; higher
        means a stronger community structure.
    """

    labels: NDArray[np.int_]
    n_communities: int
    modularity: float


def make_block_graph(
    sizes: list[int],
    p_in: float,
    p_out: float,
    random_state: int = 0,
) -> tuple[csr_matrix, NDArray[np.int_]]:
    """Generate an undirected stochastic block model graph.

    Parameters
    ----------
    sizes:
        Number of nodes in each planted community.
    p_in:
        Probability of an edge between two nodes in the same community.
    p_out:
        Probability of an edge between two nodes in different communities.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    tuple
        The symmetric adjacency matrix (unweighted, zero diagonal) and the
        planted community label of each node.
    """
    if len(sizes) < 2:
        raise ValueError("sizes must describe at least two communities.")
    if any(size <= 0 for size in sizes):
        raise ValueError("every community size must be positive.")
    if not (0.0 <= p_out <= p_in <= 1.0):
        raise ValueError("probabilities must satisfy 0 <= p_out <= p_in <= 1.")

    rng = np.random.default_rng(random_state)
    labels = np.concatenate([np.full(size, index) for index, size in enumerate(sizes)])
    n_nodes = labels.size

    probabilities = np.where(labels[:, None] == labels[None, :], p_in, p_out)
    draws = rng.random((n_nodes, n_nodes))
    adjacency = (draws < probabilities).astype(float)

    adjacency = np.triu(adjacency, k=1)
    adjacency = adjacency + adjacency.T  # symmetric, zero diagonal
    return csr_matrix(adjacency), labels


def build_knn_graph(
    data: ArrayLike,
    n_neighbors: int = 10,
) -> csr_matrix:
    """Build a symmetric k-nearest-neighbour connectivity graph from features.

    Parameters
    ----------
    data:
        Feature matrix of shape ``(n_samples, n_features)``.
    n_neighbors:
        Number of neighbours per node.

    Returns
    -------
    scipy.sparse.csr_matrix
        A symmetric binary adjacency matrix.
    """
    array = np.asarray(data, dtype=float)
    if array.ndim != 2:
        raise ValueError("data must be a 2D array.")
    if not 1 <= n_neighbors < array.shape[0]:
        raise ValueError("n_neighbors must be between 1 and the number of samples.")

    graph = kneighbors_graph(array, n_neighbors=n_neighbors, mode="connectivity")
    # Symmetrise: keep an edge if either node lists the other as a neighbour.
    symmetric = graph.maximum(graph.T)
    symmetric.setdiag(0)
    symmetric.eliminate_zeros()
    return csr_matrix(symmetric)


def modularity(adjacency: csr_matrix, labels: ArrayLike) -> float:
    """Newman modularity of a partition of an undirected graph.

    Parameters
    ----------
    adjacency:
        Symmetric (weighted or binary) adjacency matrix.
    labels:
        Community label per node.

    Returns
    -------
    float
        The modularity ``Q``. ``Q = 0`` for a single all-encompassing community;
        higher values indicate stronger community structure than chance.
    """
    if not issparse(adjacency):
        raise TypeError("adjacency must be a scipy sparse matrix.")
    matrix = csr_matrix(adjacency).astype(float)
    label_array = np.asarray(labels)
    if label_array.shape[0] != matrix.shape[0]:
        raise ValueError("labels must have one entry per node.")

    total = matrix.sum()  # 2m for an undirected graph
    if total == 0:
        raise ValueError("the graph has no edges.")
    degrees = np.asarray(matrix.sum(axis=1)).ravel()

    score = 0.0
    for community in np.unique(label_array):
        members = np.flatnonzero(label_array == community)
        internal = matrix[members][:, members].sum()
        degree_sum = degrees[members].sum()
        score += internal / total - (degree_sum / total) ** 2
    return float(score)


def detect_communities(
    adjacency: csr_matrix,
    n_communities: int,
    random_state: int = 0,
) -> GraphCommunityResult:
    """Detect communities with spectral clustering on the adjacency matrix.

    Parameters
    ----------
    adjacency:
        Symmetric adjacency matrix treated as a precomputed affinity.
    n_communities:
        Number of communities to extract.
    random_state:
        Seed for the spectral embedding.

    Returns
    -------
    GraphCommunityResult
        Labels, community count and the partition's modularity.
    """
    if not issparse(adjacency):
        raise TypeError("adjacency must be a scipy sparse matrix.")
    matrix = csr_matrix(adjacency).astype(float)
    if not 2 <= n_communities <= matrix.shape[0]:
        raise ValueError("n_communities must be between 2 and the number of nodes.")

    model = SpectralClustering(
        n_clusters=n_communities,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=random_state,
    )
    labels = np.asarray(model.fit_predict(matrix), dtype=int)
    return GraphCommunityResult(
        labels=labels,
        n_communities=int(len(np.unique(labels))),
        modularity=modularity(matrix, labels),
    )


def best_partition_by_modularity(
    adjacency: csr_matrix,
    candidate_counts: list[int],
    random_state: int = 0,
) -> GraphCommunityResult:
    """Pick the community count that maximises modularity.

    Parameters
    ----------
    adjacency:
        Symmetric adjacency matrix.
    candidate_counts:
        Candidate numbers of communities to try.
    random_state:
        Seed for the spectral embedding.

    Returns
    -------
    GraphCommunityResult
        The highest-modularity partition found.
    """
    if not candidate_counts:
        raise ValueError("candidate_counts cannot be empty.")

    best: GraphCommunityResult | None = None
    for count in candidate_counts:
        result = detect_communities(adjacency, count, random_state=random_state)
        if best is None or result.modularity > best.modularity:
            best = result
    assert best is not None  # candidate_counts is non-empty
    return best
