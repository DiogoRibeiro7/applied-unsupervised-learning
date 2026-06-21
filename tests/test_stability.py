from __future__ import annotations

import numpy as np
import pytest
from sklearn.cluster import DBSCAN, KMeans

from unsup_lab.data import make_customer_segmentation_data
from unsup_lab.preprocessing import scale_features
from unsup_lab.stability import (
    StabilitySummary,
    bootstrap_cluster_stability,
    outlier_sensitivity,
    pairwise_adjusted_mutual_information,
    repeated_run_labels,
    scaling_sensitivity,
    stability_report,
)


def _well_separated_data(random_state: int = 0) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    centres = np.array([[0.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    blocks = [centre + rng.normal(scale=0.3, size=(40, 2)) for centre in centres]
    return np.vstack(blocks)


def _kmeans_factory(k: int):
    return lambda seed: KMeans(n_clusters=k, n_init=10, random_state=seed)


def test_repeated_runs_are_deterministic_with_fixed_seed() -> None:
    data = _well_separated_data()
    first = repeated_run_labels(data, _kmeans_factory(3), n_runs=5, random_state=42)
    second = repeated_run_labels(data, _kmeans_factory(3), n_runs=5, random_state=42)

    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)


def test_pairwise_ami_high_on_separated_clusters() -> None:
    data = _well_separated_data()
    runs = repeated_run_labels(data, _kmeans_factory(3), n_runs=5, random_state=0)

    summary = pairwise_adjusted_mutual_information(runs)

    assert isinstance(summary, StabilitySummary)
    assert summary.n_pairs == 10  # C(5, 2)
    assert summary.mean_ami > 0.95


def test_bootstrap_stability_schema_and_value() -> None:
    data = _well_separated_data()

    summary = bootstrap_cluster_stability(
        data, _kmeans_factory(3), n_bootstrap=8, sample_fraction=0.8, random_state=0
    )

    assert summary.n_pairs == 28  # C(8, 2)
    assert summary.mean_ami > 0.9
    assert 2.5 <= summary.mean_n_clusters <= 3.0


def test_bootstrap_is_deterministic() -> None:
    data = _well_separated_data()
    a = bootstrap_cluster_stability(data, _kmeans_factory(3), n_bootstrap=6, random_state=1)
    b = bootstrap_cluster_stability(data, _kmeans_factory(3), n_bootstrap=6, random_state=1)

    assert a == b


def test_single_cluster_method_is_graceful() -> None:
    data = _well_separated_data()
    # eps huge -> DBSCAN assigns everything to one cluster.
    factory = lambda seed: DBSCAN(eps=1000.0, min_samples=2)  # noqa: E731

    summary = bootstrap_cluster_stability(data, factory, n_bootstrap=4, random_state=0)

    assert summary.n_pairs == 0
    assert np.isnan(summary.mean_ami)
    assert summary.mean_n_clusters == 1.0


def test_pairwise_ami_requires_two_runs() -> None:
    with pytest.raises(ValueError):
        pairwise_adjusted_mutual_information([np.array([0, 1, 0])])


def test_pairwise_ami_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        pairwise_adjusted_mutual_information([np.array([0, 1]), np.array([0, 1, 0])])


def test_scaling_sensitivity_schema() -> None:
    dataset = make_customer_segmentation_data(n_customers=200, random_state=3)

    result = scaling_sensitivity(dataset.features, _kmeans_factory(4), random_state=0)

    assert list(result.columns) == ["scaling", "n_clusters", "silhouette", "ami_vs_reference"]
    assert set(result["scaling"]) == {"none", "standard", "robust"}
    # The reference scaling agrees perfectly with itself.
    reference_row = result[result["scaling"] == "standard"].iloc[0]
    assert reference_row["ami_vs_reference"] == pytest.approx(1.0)


def test_scaling_sensitivity_invalid_reference_raises() -> None:
    dataset = make_customer_segmentation_data(n_customers=50, random_state=3)

    with pytest.raises(ValueError):
        scaling_sensitivity(
            dataset.features, _kmeans_factory(3), methods=("standard",), reference="robust"
        )


def test_outlier_sensitivity_schema_and_baseline() -> None:
    data = scale_features(
        make_customer_segmentation_data(n_customers=200, random_state=5).features
    ).to_numpy()

    result = outlier_sensitivity(
        data, _kmeans_factory(4), contamination_levels=(0.0, 0.05, 0.1), random_state=0
    )

    assert list(result.columns) == [
        "contamination",
        "n_clusters",
        "silhouette",
        "ami_vs_baseline",
    ]
    baseline_row = result[result["contamination"] == 0.0].iloc[0]
    assert baseline_row["ami_vs_baseline"] == pytest.approx(1.0)


def test_outlier_sensitivity_requires_baseline_level() -> None:
    data = _well_separated_data()

    with pytest.raises(ValueError):
        outlier_sensitivity(data, _kmeans_factory(3), contamination_levels=(0.05, 0.1))


def test_stability_report_is_markdown() -> None:
    dataset = make_customer_segmentation_data(n_customers=150, random_state=7)

    report = stability_report(
        dataset.features, _kmeans_factory(4), n_runs=4, n_bootstrap=4, random_state=0
    )

    assert isinstance(report, str)
    assert report.startswith("# Clustering stability report")
    assert "Bootstrap stability" in report
    assert "| scaling |" in report


def test_stability_report_rejects_non_dataframe() -> None:
    with pytest.raises(TypeError):
        stability_report(np.zeros((10, 2)), _kmeans_factory(2))  # type: ignore[arg-type]
