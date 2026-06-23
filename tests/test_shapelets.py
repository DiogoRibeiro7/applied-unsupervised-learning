from __future__ import annotations

import numpy as np
import pytest

from unsup_lab.timeseries import (
    Shapelet,
    discover_shapelets,
    shapelet_gap,
    subsequence_distance,
)


def _two_class_dataset(random_state: int = 0) -> tuple[list[np.ndarray], np.ndarray]:
    """Class A series contain a local bump; class B series do not."""
    rng = np.random.default_rng(random_state)
    length = 80
    window = 20
    bump = np.sin(np.linspace(0, np.pi, window)) * 3.0  # a localised hump
    series: list[np.ndarray] = []
    labels: list[int] = []
    for _ in range(12):
        s = rng.normal(scale=0.3, size=length)
        start = rng.integers(0, length - window)  # bump at a random position
        s[start : start + window] += bump
        series.append(s)
        labels.append(1)
    for _ in range(12):
        series.append(rng.normal(scale=0.3, size=length))
        labels.append(0)
    return series, np.array(labels)


def test_subsequence_distance_zero_for_contained_shape() -> None:
    series = np.concatenate([np.zeros(20), np.array([0, 1, 2, 3, 2, 1, 0.0]), np.zeros(20)])
    shapelet = np.array([0, 1, 2, 3, 2, 1, 0.0])

    assert subsequence_distance(shapelet, series) == pytest.approx(0.0, abs=1e-9)


def test_subsequence_distance_longer_shapelet_raises() -> None:
    with pytest.raises(ValueError):
        subsequence_distance(np.zeros(10), np.zeros(5))


def test_shapelet_gap_positive_for_bimodal_distances() -> None:
    near = np.full(10, 0.1)
    far = np.full(10, 5.0)
    distances = np.concatenate([near, far])

    gap, threshold = shapelet_gap(distances, min_fraction=0.2)

    assert gap > 0
    assert 0.1 < threshold < 5.0


def test_shapelet_gap_low_for_unimodal() -> None:
    rng = np.random.default_rng(0)
    bimodal = np.concatenate([np.full(10, 0.1), np.full(10, 5.0)])
    unimodal = rng.normal(loc=1.0, scale=0.05, size=20)

    assert shapelet_gap(unimodal)[0] < shapelet_gap(bimodal)[0]


def test_shapelet_gap_invalid_fraction_raises() -> None:
    with pytest.raises(ValueError):
        shapelet_gap(np.arange(10.0), min_fraction=0.8)


def test_discover_shapelets_separates_hidden_classes() -> None:
    series, labels = _two_class_dataset(random_state=1)

    shapelets = discover_shapelets(series, window=20, n_shapelets=1, stride=4)

    assert len(shapelets) == 1
    best = shapelets[0]
    assert isinstance(best, Shapelet)
    # Series whose distance is below the threshold should be the bump (class 1).
    predicted = (best.distances <= best.threshold).astype(int)
    accuracy = max((predicted == labels).mean(), (predicted != labels).mean())
    assert accuracy >= 0.8


def test_discover_shapelets_respects_window_bounds() -> None:
    series, _ = _two_class_dataset(random_state=2)

    with pytest.raises(ValueError):
        discover_shapelets(series, window=1)


def test_discover_shapelets_returns_non_overlapping() -> None:
    series, _ = _two_class_dataset(random_state=3)

    shapelets = discover_shapelets(series, window=20, n_shapelets=3, stride=4)

    assert 1 <= len(shapelets) <= 3
    assert all(s.gap >= shapelets[i + 1].gap for i, s in enumerate(shapelets[:-1]))
