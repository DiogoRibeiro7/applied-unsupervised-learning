"""Time-series analysis: DTW clustering and the matrix profile.

Two complementary tools for time series, both pure NumPy/SciPy/scikit-learn:

* **Dynamic Time Warping (DTW)** compares whole series by *shape* regardless of
  small phase shifts, and feeds an agglomerative clustering of many series.
* **The matrix profile** scans a single long series and, for every subsequence,
  records the distance to its nearest neighbour elsewhere in the series. Low
  values mark *motifs* (repeated patterns); high values mark *discords*
  (anomalies). It needs no labels and finds shape anomalies a point-wise
  detector misses.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import cdist
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


def matrix_profile(
    series: ArrayLike,
    window: int,
    exclusion_fraction: float = 0.5,
) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """Compute the matrix profile of a single series.

    For each length-``window`` subsequence, the matrix profile records the
    z-normalised Euclidean distance to its nearest non-trivial neighbour
    elsewhere in the series, and the index of that neighbour. Z-normalisation
    makes the comparison about shape rather than offset or amplitude.

    Parameters
    ----------
    series:
        A single 1D time series.
    window:
        Subsequence length.
    exclusion_fraction:
        Trivial-match exclusion zone as a fraction of ``window``: neighbours
        within ``round(exclusion_fraction * window)`` positions of a subsequence
        are ignored so a subsequence does not match itself.

    Returns
    -------
    tuple
        ``(profile, profile_index)``, each of length ``len(series) - window + 1``.
    """
    values = _as_1d(series, "series")
    if window < 2:
        raise ValueError("window must be at least 2.")
    if window > values.size:
        raise ValueError("window cannot exceed the series length.")
    if not 0.0 <= exclusion_fraction < 1.0:
        raise ValueError("exclusion_fraction must be in [0, 1).")

    windows = sliding_window_view(values, window)
    n_windows = windows.shape[0]
    if n_windows < 2:
        raise ValueError("series is too short for more than one subsequence.")

    mean = windows.mean(axis=1, keepdims=True)
    std = windows.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    normalised = (windows - mean) / std

    distances = cdist(normalised, normalised)
    exclusion = int(round(exclusion_fraction * window))
    for i in range(n_windows):
        low = max(0, i - exclusion)
        high = min(n_windows, i + exclusion + 1)
        distances[i, low:high] = np.inf

    profile = distances.min(axis=1)
    profile_index = distances.argmin(axis=1).astype(int)
    return profile.astype(float), profile_index


def _greedy_pick(
    ordered_indices: NDArray[np.int_],
    window: int,
    k: int,
) -> list[int]:
    """Pick ``k`` indices in priority order, excluding near-duplicates."""
    picked: list[int] = []
    for index in ordered_indices.tolist():
        if all(abs(index - chosen) >= window for chosen in picked):
            picked.append(int(index))
        if len(picked) == k:
            break
    return picked


def top_discords(
    profile: ArrayLike,
    window: int,
    k: int = 1,
) -> list[int]:
    """Return the start indices of the top ``k`` discords (anomalies).

    Discords are the subsequences with the *highest* matrix-profile values - the
    ones least similar to anything else in the series.

    Parameters
    ----------
    profile:
        A matrix profile from :func:`matrix_profile`.
    window:
        Subsequence length, used as the exclusion gap between picks.
    k:
        Number of discords to return.

    Returns
    -------
    list of int
        Discord start indices, most anomalous first.
    """
    values = _as_1d(profile, "profile")
    if k < 1:
        raise ValueError("k must be a positive integer.")
    finite = np.where(np.isfinite(values), values, -np.inf)
    order = np.argsort(finite)[::-1].astype(int)
    return _greedy_pick(order, window, k)


def top_motifs(
    profile: ArrayLike,
    window: int,
    k: int = 1,
) -> list[int]:
    """Return the start indices of the top ``k`` motifs (repeated patterns).

    Motifs are the subsequences with the *lowest* matrix-profile values - the
    ones with a very close match elsewhere in the series.

    Parameters
    ----------
    profile:
        A matrix profile from :func:`matrix_profile`.
    window:
        Subsequence length, used as the exclusion gap between picks.
    k:
        Number of motifs to return.

    Returns
    -------
    list of int
        Motif start indices, strongest motif first.
    """
    values = _as_1d(profile, "profile")
    if k < 1:
        raise ValueError("k must be a positive integer.")
    finite = np.where(np.isfinite(values), values, np.inf)
    order = np.argsort(finite).astype(int)
    return _greedy_pick(order, window, k)
