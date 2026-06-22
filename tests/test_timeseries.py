from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from unsup_lab.timeseries import cluster_time_series, dtw_distance, dtw_distance_matrix


def test_dtw_identical_series_is_zero() -> None:
    series = np.array([0.0, 1.0, 2.0, 1.0, 0.0])

    assert dtw_distance(series, series) == pytest.approx(0.0)


def test_dtw_is_symmetric() -> None:
    a = np.array([0.0, 1.0, 2.0, 3.0])
    b = np.array([0.0, 0.0, 1.0, 2.0])

    assert dtw_distance(a, b) == pytest.approx(dtw_distance(b, a))


def test_dtw_handles_phase_shift_better_than_euclidean() -> None:
    t = np.linspace(0, 2 * np.pi, 60)
    base = np.sin(t)
    shifted = np.sin(t + 0.6)  # same shape, shifted in phase

    euclidean = float(np.sqrt(np.sum((base - shifted) ** 2)))
    warped = dtw_distance(base, shifted)

    assert warped < euclidean


def test_dtw_handles_unequal_lengths() -> None:
    a = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    b = np.array([0.0, 2.0, 4.0])

    assert dtw_distance(a, b, band=3) >= 0.0


def test_dtw_negative_band_raises() -> None:
    with pytest.raises(ValueError):
        dtw_distance([1.0, 2.0], [1.0, 2.0], band=-1)


def test_distance_matrix_shape_and_symmetry() -> None:
    rng = np.random.default_rng(0)
    series = rng.normal(size=(5, 20))

    matrix = dtw_distance_matrix(series, band=4)

    assert matrix.shape == (5, 5)
    np.testing.assert_allclose(matrix, matrix.T)
    np.testing.assert_allclose(np.diag(matrix), 0.0)


def _shape_dataset(random_state: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two shape families (sine vs sawtooth) each with random phase shifts."""
    rng = np.random.default_rng(random_state)
    t = np.linspace(0, 2 * np.pi, 50)
    rows, truth = [], []
    for _ in range(15):
        phase = rng.uniform(0, 1.0)
        rows.append(np.sin(t + phase) + rng.normal(scale=0.05, size=t.size))
        truth.append(0)
    for _ in range(15):
        phase = rng.uniform(0, 1.0)
        sawtooth = 2 * ((t + phase) / (2 * np.pi) % 1.0) - 1
        rows.append(sawtooth + rng.normal(scale=0.05, size=t.size))
        truth.append(1)
    return np.array(rows), np.array(truth)


def test_cluster_time_series_recovers_shape_families() -> None:
    series, truth = _shape_dataset()

    labels = cluster_time_series(series, n_clusters=2, band=8)

    assert labels.shape == (30,)
    assert adjusted_rand_score(truth, labels) > 0.7


def test_cluster_invalid_n_clusters_raises() -> None:
    series, _ = _shape_dataset()

    with pytest.raises(ValueError):
        cluster_time_series(series, n_clusters=1)


def test_cluster_invalid_linkage_raises() -> None:
    series, _ = _shape_dataset()

    with pytest.raises(ValueError):
        cluster_time_series(series, n_clusters=2, linkage="ward")
