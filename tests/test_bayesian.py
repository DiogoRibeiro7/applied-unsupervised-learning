from __future__ import annotations

import numpy as np
import pytest

from unsup_lab.bayesian import (
    BayesianMixtureResult,
    assignment_uncertainty,
    component_weight_table,
    dirichlet_process_clustering,
    effective_components,
    fit_dirichlet_process_mixture,
)


def _three_blobs(random_state: int = 0) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    centres = np.array([[0.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    return np.vstack([centre + rng.normal(scale=0.3, size=(60, 2)) for centre in centres])


def test_dp_mixture_prunes_to_true_clusters() -> None:
    data = _three_blobs()

    result = dirichlet_process_clustering(data, max_components=10, random_state=0)

    assert isinstance(result, BayesianMixtureResult)
    # Three well-separated blobs -> three effective components out of ten allowed.
    assert result.effective_components == 3
    assert len(np.unique(result.labels)) == 3


def test_weights_are_sorted_descending() -> None:
    data = _three_blobs()

    result = dirichlet_process_clustering(data, max_components=8, random_state=0)

    assert np.all(np.diff(result.weights) <= 1e-12)
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)


def test_effective_components_counts_threshold() -> None:
    weights = np.array([0.5, 0.3, 0.15, 0.005, 0.045])

    assert effective_components(weights, threshold=0.01) == 4
    assert effective_components(weights, threshold=0.1) == 3


def test_effective_components_invalid_threshold_raises() -> None:
    with pytest.raises(ValueError):
        effective_components(np.array([0.5, 0.5]), threshold=1.0)


def test_dp_mixture_is_deterministic() -> None:
    data = _three_blobs()

    a = dirichlet_process_clustering(data, max_components=8, random_state=1)
    b = dirichlet_process_clustering(data, max_components=8, random_state=1)

    np.testing.assert_array_equal(a.labels, b.labels)
    np.testing.assert_allclose(a.weights, b.weights)


def test_component_weight_table_schema() -> None:
    model = fit_dirichlet_process_mixture(_three_blobs(), max_components=6, random_state=0)

    table = component_weight_table(model)

    assert list(table.columns) == ["component", "weight", "cumulative_weight"]
    assert table["weight"].is_monotonic_decreasing
    assert table["cumulative_weight"].iloc[-1] == pytest.approx(1.0, abs=1e-6)


def test_assignment_uncertainty_boundary_vs_core() -> None:
    data = _three_blobs()
    model = fit_dirichlet_process_mixture(data, max_components=8, random_state=0)

    # A point at a blob centre should be far more confident than one on the
    # midline between two blobs.
    core = np.array([[0.0, 0.0]])
    boundary = np.array([[5.0, 5.0]])
    core_unc = assignment_uncertainty(model, core)
    boundary_unc = assignment_uncertainty(model, boundary)

    assert core_unc["entropy"].iloc[0] < boundary_unc["entropy"].iloc[0]
    assert core_unc["max_responsibility"].iloc[0] > boundary_unc["max_responsibility"].iloc[0]


def test_max_components_cannot_exceed_rows() -> None:
    data = np.zeros((5, 2))

    with pytest.raises(ValueError):
        fit_dirichlet_process_mixture(data, max_components=10)


def test_invalid_data_shape_raises() -> None:
    with pytest.raises(ValueError):
        dirichlet_process_clustering(np.zeros((10,)))
