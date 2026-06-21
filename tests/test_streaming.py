from __future__ import annotations

import numpy as np
import pytest

from unsup_lab.streaming import (
    detect_drift_points,
    iter_batches,
    monitor_streaming_clusters,
    population_stability_index,
)


def _drifting_stream(random_state: int = 0) -> np.ndarray:
    """A stream whose monitored feature shifts upward in the second half."""
    rng = np.random.default_rng(random_state)
    stable = rng.normal(loc=[0.0, 0.0], scale=0.5, size=(600, 2))
    drifted = rng.normal(loc=[8.0, 0.0], scale=0.5, size=(600, 2))
    return np.vstack([stable, drifted])


def test_iter_batches_covers_all_rows() -> None:
    data = np.arange(20).reshape(10, 2)

    batches = iter_batches(data, batch_size=3)

    assert len(batches) == 4  # 3 + 3 + 3 + 1
    assert sum(b.shape[0] for b in batches) == 10


def test_psi_zero_for_identical_distribution() -> None:
    rng = np.random.default_rng(0)
    sample = rng.normal(size=1000)

    assert population_stability_index(sample, sample) == pytest.approx(0.0, abs=1e-9)


def test_psi_grows_with_shift() -> None:
    rng = np.random.default_rng(0)
    reference = rng.normal(loc=0.0, size=1000)
    small = rng.normal(loc=0.3, size=1000)
    large = rng.normal(loc=3.0, size=1000)

    assert population_stability_index(reference, large) > population_stability_index(
        reference, small
    )


def test_psi_constant_reference_returns_zero() -> None:
    assert population_stability_index(np.ones(50), np.arange(50)) == 0.0


def test_monitor_schema_and_determinism() -> None:
    data = _drifting_stream()

    first = monitor_streaming_clusters(data, n_clusters=3, batch_size=200, random_state=0)
    second = monitor_streaming_clusters(data, n_clusters=3, batch_size=200, random_state=0)

    assert list(first.columns) == ["batch", "n_seen", "centroid_shift", "inertia", "psi"]
    assert first["n_seen"].iloc[-1] == data.shape[0]
    assert np.isnan(first["psi"].iloc[0])  # first batch is the reference
    np.testing.assert_allclose(first["psi"].to_numpy(), second["psi"].to_numpy(), equal_nan=True)


def test_monitor_detects_known_drift() -> None:
    data = _drifting_stream()

    report = monitor_streaming_clusters(data, n_clusters=3, batch_size=200, drift_column=0)
    flagged = detect_drift_points(report["psi"], threshold=0.25)

    # Drift is injected at row 600 -> batch index 3 with batch_size 200.
    assert any(index >= 3 for index in flagged)


def test_monitor_requires_multiple_batches() -> None:
    data = _drifting_stream()

    with pytest.raises(ValueError):
        monitor_streaming_clusters(data, batch_size=10_000)


def test_detect_drift_points_ignores_nan() -> None:
    flagged = detect_drift_points([np.nan, 0.05, 0.4, 0.1, 0.6])

    assert flagged == [2, 4]


def test_detect_drift_invalid_threshold_raises() -> None:
    with pytest.raises(ValueError):
        detect_drift_points([0.1, 0.2], threshold=0.0)
