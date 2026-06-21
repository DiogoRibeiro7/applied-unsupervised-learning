"""Consensus (ensemble) clustering.

A single KMeans fit depends on its random initialisation and on which points
happen to be present. Consensus clustering reduces that fragility: it runs many
base clusterings on resampled data, accumulates how often each pair of points
lands in the same cluster (the *co-association matrix*), and derives a final
partition from that agreement. Pairs that are repeatedly grouped together are
treated as genuinely similar, regardless of any single run.

The base clusterer is supplied as a factory ``Callable[[int], ClusterMixin]``
mapping a random seed to a fresh estimator, matching the convention used in
:mod:`unsup_lab.stability`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import ClusterMixin
from sklearn.cluster import AgglomerativeClustering

EstimatorFactory = Callable[[int], ClusterMixin]


@dataclass(frozen=True)
class ConsensusResult:
    """Outcome of a consensus clustering run.

    Attributes
    ----------
    labels:
        Final consensus cluster label for each point.
    coassociation:
        Symmetric ``(n, n)`` matrix; entry ``(i, j)`` is the fraction of base
        runs (in which both points were sampled) that placed ``i`` and ``j`` in
        the same cluster.
    stability:
        Mean intra-cluster co-association of the final partition, in ``[0, 1]``.
        Higher means the consensus clusters are internally well agreed upon.
    """

    labels: NDArray[np.int_]
    coassociation: NDArray[np.float64]
    stability: float


def _as_2d_array(data: ArrayLike) -> NDArray[np.float64]:
    """Convert input data to a validated 2D float array."""
    array = np.asarray(data, dtype=float)
    if array.ndim != 2:
        raise ValueError("data must be a 2D array.")
    if array.shape[0] < 2:
        raise ValueError("data must contain at least two rows.")
    return array


def build_coassociation_matrix(
    data: ArrayLike,
    base_factory: EstimatorFactory,
    n_runs: int = 25,
    sample_fraction: float = 0.8,
    random_state: int = 0,
) -> NDArray[np.float64]:
    """Accumulate pairwise co-clustering frequencies over resampled runs.

    Parameters
    ----------
    data:
        Feature matrix of shape ``(n, d)``.
    base_factory:
        Maps a random seed to a fresh base clustering estimator.
    n_runs:
        Number of base clusterings.
    sample_fraction:
        Fraction of rows drawn without replacement in each run, in ``(0, 1]``.
    random_state:
        Seed for reproducible subsampling and base-estimator seeding.

    Returns
    -------
    numpy.ndarray
        Symmetric ``(n, n)`` co-association matrix with unit diagonal.
    """
    x = _as_2d_array(data)
    if n_runs < 2:
        raise ValueError("n_runs must be at least 2.")
    if not 0.0 < sample_fraction <= 1.0:
        raise ValueError("sample_fraction must be in (0, 1].")

    n_rows = x.shape[0]
    sample_size = max(2, int(round(sample_fraction * n_rows)))
    rng = np.random.default_rng(random_state)

    co_occurrence = np.zeros((n_rows, n_rows), dtype=float)
    sampled_together = np.zeros((n_rows, n_rows), dtype=float)

    for run in range(n_runs):
        indices = rng.choice(n_rows, size=sample_size, replace=False)
        labels = np.asarray(base_factory(random_state + run).fit_predict(x[indices]))
        sampled_together[np.ix_(indices, indices)] += 1.0
        for cluster in np.unique(labels):
            if cluster == -1:  # DBSCAN noise is never "the same cluster".
                continue
            members = indices[labels == cluster]
            co_occurrence[np.ix_(members, members)] += 1.0

    with np.errstate(divide="ignore", invalid="ignore"):
        consensus = np.where(sampled_together > 0, co_occurrence / sampled_together, 0.0)
    np.fill_diagonal(consensus, 1.0)
    return consensus


def _intra_cluster_stability(
    coassociation: NDArray[np.float64],
    labels: NDArray[np.int_],
) -> float:
    """Mean co-association between distinct points sharing a consensus cluster."""
    values: list[float] = []
    for cluster in np.unique(labels):
        members = np.flatnonzero(labels == cluster)
        if members.size < 2:
            continue
        block = coassociation[np.ix_(members, members)]
        upper = block[np.triu_indices_from(block, k=1)]
        values.extend(upper.tolist())
    return float(np.mean(values)) if values else float("nan")


def consensus_clustering(
    data: ArrayLike,
    n_clusters: int,
    base_factory: EstimatorFactory,
    n_runs: int = 25,
    sample_fraction: float = 0.8,
    linkage: str = "average",
    random_state: int = 0,
) -> ConsensusResult:
    """Derive a stable partition from an ensemble of base clusterings.

    The co-association matrix is converted into a distance (``1 - agreement``)
    and partitioned with agglomerative clustering, so the final labels reflect
    how consistently points were grouped rather than any single fit.

    Parameters
    ----------
    data:
        Feature matrix.
    n_clusters:
        Number of consensus clusters to extract.
    base_factory:
        Maps a random seed to a fresh base clustering estimator.
    n_runs:
        Number of base clusterings.
    sample_fraction:
        Fraction of rows used per base run.
    linkage:
        Agglomerative linkage for the consensus step; one of ``"average"``,
        ``"complete"`` or ``"single"`` (``"ward"`` is unavailable for a
        precomputed distance).
    random_state:
        Seed for reproducibility.

    Returns
    -------
    ConsensusResult
        Final labels, the co-association matrix, and a stability score.
    """
    x = _as_2d_array(data)
    if not 2 <= n_clusters <= x.shape[0]:
        raise ValueError("n_clusters must be between 2 and the number of rows.")
    if linkage not in {"average", "complete", "single"}:
        raise ValueError("linkage must be 'average', 'complete' or 'single'.")

    coassociation = build_coassociation_matrix(
        x,
        base_factory,
        n_runs=n_runs,
        sample_fraction=sample_fraction,
        random_state=random_state,
    )
    distance = 1.0 - coassociation
    np.fill_diagonal(distance, 0.0)

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage=linkage,
    )
    labels = np.asarray(model.fit_predict(distance), dtype=int)
    stability = _intra_cluster_stability(coassociation, labels)
    return ConsensusResult(labels=labels, coassociation=coassociation, stability=stability)
