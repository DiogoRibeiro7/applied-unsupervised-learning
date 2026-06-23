"""Time-series analysis: DTW clustering and the matrix profile.

Two complementary tools for time series, both pure NumPy/SciPy/scikit-learn:

* **Dynamic Time Warping (DTW)** compares whole series by *shape* regardless of
  small phase shifts, and feeds an agglomerative clustering of many series.
* **The matrix profile** scans a single long series and, for every subsequence,
  records the distance to its nearest neighbour elsewhere in the series. Low
  values mark *motifs* (repeated patterns); high values mark *discords*
  (anomalies). It needs no labels and finds shape anomalies a point-wise
  detector misses.
* **Unsupervised shapelets (u-shapelets)** are short subsequences whose distance
  to every series in a dataset splits it into a "contains this local pattern"
  group and a "does not" group with a large gap. They discover discriminative
  local shapes with no labels, which is useful when only part of each series
  carries the signal.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

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


@dataclass(frozen=True)
class Shapelet:
    """An unsupervised shapelet and how well it splits the dataset.

    Attributes
    ----------
    series_index:
        Index of the series the shapelet was extracted from.
    start:
        Start position of the shapelet within that series.
    values:
        The shapelet subsequence itself.
    gap:
        The separation gap score; higher means a cleaner split.
    threshold:
        The distance threshold separating the "near" and "far" groups.
    distances:
        The shapelet's distance to every series in the dataset.
    """

    series_index: int
    start: int
    values: NDArray[np.float64]
    gap: float
    threshold: float
    distances: NDArray[np.float64]


def _znorm(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Z-normalise a 1D array (zero mean, unit std; std 0 -> unchanged)."""
    std = values.std()
    return (values - values.mean()) / (std if std > 0 else 1.0)


def _znorm_rows(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Z-normalise each row of a 2D array."""
    mean = matrix.mean(axis=1, keepdims=True)
    std = matrix.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (matrix - mean) / std


def subsequence_distance(shapelet: ArrayLike, series: ArrayLike) -> float:
    """Minimum z-normalised Euclidean distance from a shapelet to a series.

    The shapelet is slid along the series and the closest match is returned, so
    the distance reflects whether the local *shape* appears anywhere in the
    series, regardless of where.

    Parameters
    ----------
    shapelet:
        The query subsequence.
    series:
        The series to search.

    Returns
    -------
    float
        The smallest z-normalised Euclidean distance over all alignments.
    """
    query = _as_1d(shapelet, "shapelet")
    target = _as_1d(series, "series")
    if query.size > target.size:
        raise ValueError("shapelet cannot be longer than the series.")

    windows = _znorm_rows(sliding_window_view(target, query.size))
    diffs = windows - _znorm(query)
    return float(np.sqrt((diffs**2).sum(axis=1)).min())


def shapelet_gap(distances: ArrayLike, min_fraction: float = 0.2) -> tuple[float, float]:
    """Best separation gap of a shapelet's distance vector.

    Sorting the distances, the gap score for a split is
    ``mean(far) - std(far) - mean(near) - std(near)``. The split is required to
    leave at least ``min_fraction`` of the series on each side, so a shapelet is
    only rewarded for separating a *substantial* group.

    Parameters
    ----------
    distances:
        Distances from one shapelet to every series.
    min_fraction:
        Minimum fraction of series required on each side of the split.

    Returns
    -------
    tuple
        ``(gap, threshold)``. ``gap`` is ``-inf`` and ``threshold`` is ``nan``
        when no valid split exists.
    """
    values = np.sort(np.asarray(distances, dtype=float).ravel())
    n = values.size
    if n < 2:
        raise ValueError("need at least two series to score a split.")
    if not 0.0 < min_fraction <= 0.5:
        raise ValueError("min_fraction must be in (0, 0.5].")

    min_count = max(1, int(round(min_fraction * n)))
    best_gap = -np.inf
    best_threshold = float("nan")
    for i in range(min_count - 1, n - min_count):
        near = values[: i + 1]
        far = values[i + 1 :]
        gap = float(far.mean() - far.std() - near.mean() - near.std())
        if gap > best_gap:
            best_gap = gap
            best_threshold = float((values[i] + values[i + 1]) / 2.0)
    return best_gap, best_threshold


def discover_shapelets(
    series: Iterable[ArrayLike],
    window: int,
    n_shapelets: int = 1,
    stride: int = 1,
    min_fraction: float = 0.2,
) -> list[Shapelet]:
    """Discover the top unsupervised shapelets in a dataset of series.

    Every length-``window`` subsequence (subject to ``stride``) is a candidate;
    each is scored by the gap it induces in its distance vector across the
    dataset. The highest-gap, non-overlapping candidates are returned.

    Parameters
    ----------
    series:
        Equal- or varying-length 1D series.
    window:
        Shapelet length.
    n_shapelets:
        Number of shapelets to return.
    stride:
        Step between candidate start positions (larger is faster, coarser).
    min_fraction:
        Minimum fraction of series on each side of a shapelet's split.

    Returns
    -------
    list of Shapelet
        Up to ``n_shapelets`` shapelets, highest gap first.
    """
    sequences = [_as_1d(row, "series row") for row in series]
    if len(sequences) < 2:
        raise ValueError("need at least two series.")
    if window < 2:
        raise ValueError("window must be at least 2.")
    if any(window > seq.size for seq in sequences):
        raise ValueError("window cannot exceed the shortest series length.")
    if stride < 1:
        raise ValueError("stride must be a positive integer.")

    candidates: list[Shapelet] = []
    for series_index, sequence in enumerate(sequences):
        for start in range(0, sequence.size - window + 1, stride):
            shapelet = sequence[start : start + window]
            distances = np.array([subsequence_distance(shapelet, seq) for seq in sequences])
            gap, threshold = shapelet_gap(distances, min_fraction=min_fraction)
            candidates.append(
                Shapelet(
                    series_index=series_index,
                    start=start,
                    values=shapelet,
                    gap=gap,
                    threshold=threshold,
                    distances=distances,
                )
            )

    candidates.sort(key=lambda item: item.gap, reverse=True)

    selected: list[Shapelet] = []
    for candidate in candidates:
        overlaps = any(
            candidate.series_index == chosen.series_index
            and abs(candidate.start - chosen.start) < window
            for chosen in selected
        )
        if not overlaps:
            selected.append(candidate)
        if len(selected) == n_shapelets:
            break
    return selected
