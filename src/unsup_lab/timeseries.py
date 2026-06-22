"""Time-series clustering with Dynamic Time Warping (DTW).

Euclidean distance compares two series point-by-point, so two signals with the
same shape but a small phase shift look far apart. DTW instead finds the optimal
alignment between the series before measuring distance, which makes it cluster by
*shape* rather than by exact timing.

This module provides a banded DTW distance, a pairwise distance matrix, and an
agglomerative clustering helper that consumes that matrix. It depends only on
NumPy and scikit-learn.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.cluster import AgglomerativeClustering


def _as_1d(series: ArrayLike, name: str) -> NDArray[np.float64]:
    """Convert a single series to a validated 1D float array."""
    array = np.asarray(series, dtype=float).ravel()
    if array.size < 1:
        raise ValueError(f"{name} must contain at least one value.")
    return array


def dtw_distance(
    first: ArrayLike,
    second: ArrayLike,
    band: int | None = None,
) -> float:
    """Dynamic Time Warping distance between two 1D series.

    Parameters
    ----------
    first, second:
        One-dimensional series (they may have different lengths).
    band:
        Optional Sakoe-Chiba band radius. When set, the warping path is
        constrained so aligned indices differ by at most ``band`` steps, which
        bounds the cost and prevents pathological alignments. ``None`` allows an
        unconstrained path.

    Returns
    -------
    float
        The DTW distance (square root of the accumulated squared alignment cost).
    """
    a = _as_1d(first, "first")
    b = _as_1d(second, "second")
    if band is not None and band < 0:
        raise ValueError("band must be non-negative.")

    n, m = a.size, b.size
    accumulated = np.full((n + 1, m + 1), np.inf)
    accumulated[0, 0] = 0.0

    for i in range(1, n + 1):
        if band is None:
            j_start, j_end = 1, m
        else:
            # Scale the band to the relative position when lengths differ.
            centre = int(round((i - 1) * m / n)) + 1
            j_start = max(1, centre - band)
            j_end = min(m, centre + band)
        for j in range(j_start, j_end + 1):
            cost = (a[i - 1] - b[j - 1]) ** 2
            accumulated[i, j] = cost + min(
                accumulated[i - 1, j],
                accumulated[i, j - 1],
                accumulated[i - 1, j - 1],
            )

    result = accumulated[n, m]
    if not np.isfinite(result):
        raise ValueError("no valid warping path; try a larger band.")
    return float(np.sqrt(result))


def dtw_distance_matrix(
    series: Iterable[ArrayLike],
    band: int | None = None,
) -> NDArray[np.float64]:
    """Compute the symmetric pairwise DTW distance matrix for a set of series.

    Parameters
    ----------
    series:
        Array of shape ``(n_series, length)`` (equal-length series) or a sequence
        of 1D arrays of possibly different lengths.
    band:
        Optional Sakoe-Chiba band radius passed to :func:`dtw_distance`.

    Returns
    -------
    numpy.ndarray
        A ``(n_series, n_series)`` symmetric distance matrix with a zero
        diagonal.
    """
    sequences = [_as_1d(row, "series row") for row in series]
    n_series = len(sequences)
    if n_series < 2:
        raise ValueError("series must contain at least two sequences.")

    matrix = np.zeros((n_series, n_series), dtype=float)
    for i in range(n_series):
        for j in range(i + 1, n_series):
            distance = dtw_distance(sequences[i], sequences[j], band=band)
            matrix[i, j] = distance
            matrix[j, i] = distance
    return matrix


def cluster_time_series(
    series: Iterable[ArrayLike],
    n_clusters: int,
    band: int | None = None,
    linkage: str = "average",
) -> NDArray[np.int_]:
    """Cluster series by shape using DTW distances and agglomerative clustering.

    Parameters
    ----------
    series:
        Array of shape ``(n_series, length)`` or a sequence of 1D arrays.
    n_clusters:
        Number of clusters to extract.
    band:
        Optional Sakoe-Chiba band radius for the DTW distance.
    linkage:
        Agglomerative linkage for the precomputed distance; one of ``"average"``,
        ``"complete"`` or ``"single"``.

    Returns
    -------
    numpy.ndarray
        Cluster label per series.
    """
    if linkage not in {"average", "complete", "single"}:
        raise ValueError("linkage must be 'average', 'complete' or 'single'.")

    distance = dtw_distance_matrix(series, band=band)
    if not 2 <= n_clusters <= distance.shape[0]:
        raise ValueError("n_clusters must be between 2 and the number of series.")

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage=linkage,
    )
    return np.asarray(model.fit_predict(distance), dtype=int)
