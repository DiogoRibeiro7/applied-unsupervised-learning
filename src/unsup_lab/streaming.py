"""Streaming clustering with drift monitoring.

In production, data arrives as a stream rather than a fixed table, and the
distribution can shift over time (concept drift). This module fits clusters
incrementally with mini-batch KMeans and tracks two complementary drift
signals per batch:

* **centroid shift** - how far the cluster centres move when the batch is
  absorbed; a large move means the model is chasing a changing distribution.
* **population stability index (PSI)** - a distributional distance between a
  reference batch and the current batch on a chosen feature. The common rule of
  thumb is ``PSI < 0.1`` stable, ``0.1-0.25`` moderate shift, ``> 0.25`` major
  shift.

Everything is built on NumPy and scikit-learn only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.cluster import MiniBatchKMeans


def _as_2d_array(data: ArrayLike) -> NDArray[np.float64]:
    """Convert input data to a validated 2D float array."""
    array = np.asarray(data, dtype=float)
    if array.ndim != 2:
        raise ValueError("data must be a 2D array.")
    if array.shape[0] < 2:
        raise ValueError("data must contain at least two rows.")
    return array


def iter_batches(data: ArrayLike, batch_size: int) -> list[NDArray[np.float64]]:
    """Split a feature matrix into consecutive batches.

    Parameters
    ----------
    data:
        Feature matrix of shape ``(n, d)``.
    batch_size:
        Number of rows per batch. The final batch may be smaller.

    Returns
    -------
    list of numpy.ndarray
        Ordered list of batch arrays.
    """
    x = _as_2d_array(data)
    if batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    return [x[start : start + batch_size] for start in range(0, x.shape[0], batch_size)]


def population_stability_index(
    expected: ArrayLike,
    actual: ArrayLike,
    n_bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Population Stability Index between a reference and a current sample.

    The reference sample defines quantile bin edges; the index sums the
    relative-frequency divergence across those bins. Returns ``0.0`` for
    identical distributions and grows as they diverge.

    Parameters
    ----------
    expected:
        Reference sample (1D).
    actual:
        Current sample (1D).
    n_bins:
        Number of quantile bins derived from the reference sample.
    epsilon:
        Floor applied to bin proportions to avoid division by zero.

    Returns
    -------
    float
        The PSI value (non-negative).
    """
    expected_array = np.asarray(expected, dtype=float).ravel()
    actual_array = np.asarray(actual, dtype=float).ravel()
    if expected_array.size < 2 or actual_array.size < 2:
        raise ValueError("expected and actual must each contain at least two values.")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(expected_array, quantiles))
    if edges.size < 2:
        # The reference is (near) constant; no meaningful binning is possible.
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    expected_counts = np.histogram(expected_array, bins=edges)[0] / expected_array.size
    actual_counts = np.histogram(actual_array, bins=edges)[0] / actual_array.size

    expected_counts = np.clip(expected_counts, epsilon, None)
    actual_counts = np.clip(actual_counts, epsilon, None)

    divergence = (actual_counts - expected_counts) * np.log(actual_counts / expected_counts)
    return float(np.sum(divergence))


def monitor_streaming_clusters(
    data: ArrayLike,
    n_clusters: int = 4,
    batch_size: int = 200,
    drift_column: int = 0,
    random_state: int = 0,
) -> pd.DataFrame:
    """Fit clusters incrementally and report per-batch drift diagnostics.

    A :class:`~sklearn.cluster.MiniBatchKMeans` is updated batch by batch with
    ``partial_fit``. For each batch the function records the centroid movement,
    the batch inertia, and the PSI of ``drift_column`` against the first batch
    used as the reference window.

    Parameters
    ----------
    data:
        Feature matrix, already in stream order.
    n_clusters:
        Number of clusters to maintain.
    batch_size:
        Rows per streaming batch.
    drift_column:
        Index of the feature monitored for distributional drift.
    random_state:
        Seed for the mini-batch estimator.

    Returns
    -------
    pandas.DataFrame
        One row per batch with ``batch``, ``n_seen``, ``centroid_shift``,
        ``inertia`` and ``psi`` columns.
    """
    x = _as_2d_array(data)
    if not 2 <= n_clusters <= x.shape[0]:
        raise ValueError("n_clusters must be between 2 and the number of rows.")
    if not 0 <= drift_column < x.shape[1]:
        raise ValueError("drift_column is out of range.")

    batches = iter_batches(x, batch_size)
    if len(batches) < 2:
        raise ValueError("need at least two batches; reduce batch_size.")

    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=3,
        batch_size=batch_size,
    )
    reference = batches[0][:, drift_column]

    rows: list[dict[str, float | int]] = []
    previous_centroids: NDArray[np.float64] | None = None
    seen = 0
    for index, batch in enumerate(batches):
        model.partial_fit(batch)
        seen += batch.shape[0]
        centroids = model.cluster_centers_.copy()

        if previous_centroids is None:
            centroid_shift = float("nan")
            psi = float("nan")
        else:
            centroid_shift = float(np.mean(np.linalg.norm(centroids - previous_centroids, axis=1)))
            psi = population_stability_index(reference, batch[:, drift_column])

        labels = model.predict(batch)
        inertia = float(np.sum((batch - centroids[labels]) ** 2))

        rows.append(
            {
                "batch": index,
                "n_seen": seen,
                "centroid_shift": centroid_shift,
                "inertia": inertia,
                "psi": psi,
            }
        )
        previous_centroids = centroids

    return pd.DataFrame(rows)


def detect_drift_points(
    psi_values: ArrayLike,
    threshold: float = 0.25,
) -> list[int]:
    """Return the batch indices whose PSI exceeds a drift threshold.

    Parameters
    ----------
    psi_values:
        Per-batch PSI values (``nan`` entries, such as the reference batch, are
        ignored).
    threshold:
        PSI value above which a batch is flagged as drifted. ``0.25`` is the
        conventional "major shift" cut-off.

    Returns
    -------
    list of int
        Indices of flagged batches.
    """
    values = np.asarray(psi_values, dtype=float)
    if threshold <= 0:
        raise ValueError("threshold must be positive.")
    flagged = np.flatnonzero((~np.isnan(values)) & (values > threshold))
    return [int(index) for index in flagged]
