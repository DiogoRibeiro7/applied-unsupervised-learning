"""Model selection and stability tools for clustering.

These utilities help answer the question that matters most for unsupervised
learning: *is the discovered structure real, or an artefact of one particular
fit?* They quantify how much cluster assignments change under resampling,
repeated random seeds, alternative feature scaling, and outlier contamination.

All functions are deterministic when a ``random_state`` is supplied.

A clustering estimator is supplied as a factory ``Callable[[int], ClusterMixin]``
that maps a random seed to a fresh estimator, for example::

    factory = lambda seed: KMeans(n_clusters=4, n_init=10, random_state=seed)

The desired number of clusters is baked into the factory by the caller, which
keeps these tools agnostic to the clustering algorithm.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.base import ClusterMixin
from sklearn.metrics import adjusted_mutual_info_score

from unsup_lab.evaluation import evaluate_clustering
from unsup_lab.preprocessing import scale_features

EstimatorFactory = Callable[[int], ClusterMixin]


@dataclass(frozen=True)
class StabilitySummary:
    """Summary of clustering agreement across repeated fits.

    Attributes
    ----------
    mean_ami:
        Mean adjusted mutual information across all compared pairs of runs.
        ``nan`` when no pair could be compared (for example every run collapsed
        to a single cluster).
    std_ami:
        Standard deviation of the pairwise adjusted mutual information.
    n_pairs:
        Number of run pairs that were compared.
    mean_n_clusters:
        Mean number of non-noise clusters produced across runs.
    """

    mean_ami: float
    std_ami: float
    n_pairs: int
    mean_n_clusters: float


def _as_2d_array(data: ArrayLike) -> NDArray[np.float64]:
    """Convert input data to a validated 2D float array."""
    array = np.asarray(data, dtype=float)
    if array.ndim != 2:
        raise ValueError("data must be a 2D array.")
    if array.shape[0] < 2:
        raise ValueError("data must contain at least two rows.")
    return array


def _n_non_noise_clusters(labels: NDArray[np.int_]) -> int:
    """Count clusters, treating the DBSCAN noise label ``-1`` as not a cluster."""
    return len({label for label in labels.tolist() if label != -1})


def repeated_run_labels(
    data: ArrayLike,
    estimator_factory: EstimatorFactory,
    n_runs: int = 10,
    random_state: int = 0,
) -> list[NDArray[np.int_]]:
    """Cluster the same data ``n_runs`` times with different seeds.

    Parameters
    ----------
    data:
        Feature matrix.
    estimator_factory:
        Maps a random seed to a fresh clustering estimator.
    n_runs:
        Number of repeated fits.
    random_state:
        Base seed; run ``i`` uses seed ``random_state + i``.

    Returns
    -------
    list of numpy.ndarray
        One label vector per run.
    """
    x = _as_2d_array(data)
    if n_runs < 2:
        raise ValueError("n_runs must be at least 2.")

    runs: list[NDArray[np.int_]] = []
    for i in range(n_runs):
        estimator = estimator_factory(random_state + i)
        labels = np.asarray(estimator.fit_predict(x))
        runs.append(labels)
    return runs


def pairwise_adjusted_mutual_information(
    label_runs: list[NDArray[np.int_]],
) -> StabilitySummary:
    """Mean pairwise adjusted mutual information across repeated clusterings.

    A pair is skipped when either run produced fewer than two non-noise
    clusters, because adjusted mutual information is not informative against a
    degenerate single-cluster labelling.

    Parameters
    ----------
    label_runs:
        Label vectors of equal length, one per run.

    Returns
    -------
    StabilitySummary
        Aggregated agreement statistics.
    """
    if len(label_runs) < 2:
        raise ValueError("label_runs must contain at least two runs.")
    lengths = {len(run) for run in label_runs}
    if len(lengths) != 1:
        raise ValueError("all label runs must have the same length.")

    cluster_counts = [_n_non_noise_clusters(np.asarray(run)) for run in label_runs]

    scores: list[float] = []
    for left, right in combinations(label_runs, 2):
        if _n_non_noise_clusters(np.asarray(left)) < 2:
            continue
        if _n_non_noise_clusters(np.asarray(right)) < 2:
            continue
        scores.append(float(adjusted_mutual_info_score(left, right)))

    if scores:
        mean_ami = float(np.mean(scores))
        std_ami = float(np.std(scores))
    else:
        mean_ami = float("nan")
        std_ami = float("nan")

    return StabilitySummary(
        mean_ami=mean_ami,
        std_ami=std_ami,
        n_pairs=len(scores),
        mean_n_clusters=float(np.mean(cluster_counts)),
    )


def bootstrap_cluster_stability(
    data: ArrayLike,
    estimator_factory: EstimatorFactory,
    n_bootstrap: int = 20,
    sample_fraction: float = 0.8,
    random_state: int = 0,
) -> StabilitySummary:
    """Measure cluster stability under bootstrap subsampling.

    Each bootstrap draws a subsample without replacement, clusters it, and the
    resulting labels are compared between every pair of bootstraps on the points
    they share. High agreement means the structure does not depend on which
    points happen to be present.

    Parameters
    ----------
    data:
        Feature matrix.
    estimator_factory:
        Maps a random seed to a fresh clustering estimator.
    n_bootstrap:
        Number of bootstrap subsamples.
    sample_fraction:
        Fraction of rows drawn in each subsample, in ``(0, 1]``.
    random_state:
        Seed for reproducible subsampling and estimator seeding.

    Returns
    -------
    StabilitySummary
        Aggregated agreement statistics across bootstrap pairs.
    """
    x = _as_2d_array(data)
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least 2.")
    if not 0.0 < sample_fraction <= 1.0:
        raise ValueError("sample_fraction must be in (0, 1].")

    rng = np.random.default_rng(random_state)
    n_rows = x.shape[0]
    sample_size = max(2, int(round(sample_fraction * n_rows)))

    runs: list[tuple[NDArray[np.int_], dict[int, int]]] = []
    cluster_counts: list[int] = []
    for i in range(n_bootstrap):
        indices = rng.choice(n_rows, size=sample_size, replace=False)
        estimator = estimator_factory(random_state + i)
        labels = np.asarray(estimator.fit_predict(x[indices]))
        cluster_counts.append(_n_non_noise_clusters(labels))
        runs.append((indices, dict(zip(indices.tolist(), labels.tolist(), strict=True))))

    scores: list[float] = []
    for (idx_left, map_left), (idx_right, map_right) in combinations(runs, 2):
        shared = np.intersect1d(idx_left, idx_right, assume_unique=True)
        if shared.size < 2:
            continue
        left_labels = np.array([map_left[i] for i in shared.tolist()])
        right_labels = np.array([map_right[i] for i in shared.tolist()])
        if _n_non_noise_clusters(left_labels) < 2 or _n_non_noise_clusters(right_labels) < 2:
            continue
        scores.append(float(adjusted_mutual_info_score(left_labels, right_labels)))

    if scores:
        mean_ami = float(np.mean(scores))
        std_ami = float(np.std(scores))
    else:
        mean_ami = float("nan")
        std_ami = float("nan")

    return StabilitySummary(
        mean_ami=mean_ami,
        std_ami=std_ami,
        n_pairs=len(scores),
        mean_n_clusters=float(np.mean(cluster_counts)),
    )


def scaling_sensitivity(
    data: pd.DataFrame,
    estimator_factory: EstimatorFactory,
    methods: tuple[str, ...] = ("none", "standard", "robust"),
    reference: str = "standard",
    random_state: int = 0,
) -> pd.DataFrame:
    """Assess how feature scaling changes the clustering.

    The estimator is fitted on the data transformed by each scaling method, and
    the resulting labels are compared against a reference scaling using adjusted
    mutual information. A clustering that is sensitive to scaling is fragile.

    Parameters
    ----------
    data:
        Numeric feature table.
    estimator_factory:
        Maps a random seed to a fresh clustering estimator.
    methods:
        Scaling methods to compare. Supported values are ``"none"``,
        ``"standard"`` and ``"robust"``.
    reference:
        The scaling method whose labels are used as the agreement reference.
    random_state:
        Seed passed to the estimator factory.

    Returns
    -------
    pandas.DataFrame
        One row per scaling method with ``n_clusters``, ``silhouette`` and
        ``ami_vs_reference``.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if reference not in methods:
        raise ValueError("reference must be one of the requested methods.")

    transformed: dict[str, pd.DataFrame] = {}
    labels: dict[str, NDArray[np.int_]] = {}
    for method in methods:
        scaled = data if method == "none" else scale_features(data, method=method)
        transformed[method] = scaled
        estimator = estimator_factory(random_state)
        labels[method] = np.asarray(estimator.fit_predict(scaled.to_numpy()))

    reference_labels = labels[reference]
    rows: list[dict[str, float | int | str | None]] = []
    for method in methods:
        metrics = evaluate_clustering(transformed[method].to_numpy(), labels[method])
        if _n_non_noise_clusters(labels[method]) < 2 or _n_non_noise_clusters(reference_labels) < 2:
            ami = None
        else:
            ami = float(adjusted_mutual_info_score(reference_labels, labels[method]))
        rows.append(
            {
                "scaling": method,
                "n_clusters": metrics.n_clusters,
                "silhouette": metrics.silhouette,
                "ami_vs_reference": ami,
            }
        )
    return pd.DataFrame(rows)


def outlier_sensitivity(
    data: ArrayLike,
    estimator_factory: EstimatorFactory,
    contamination_levels: tuple[float, ...] = (0.0, 0.02, 0.05, 0.1),
    outlier_scale: float = 6.0,
    random_state: int = 0,
) -> pd.DataFrame:
    """Assess how injected outliers change the clustering of the original points.

    For each contamination level a number of synthetic outliers is appended far
    from the data centre, the augmented set is clustered, and the labels of the
    original points are compared against the zero-contamination baseline using
    adjusted mutual information.

    Parameters
    ----------
    data:
        Feature matrix.
    estimator_factory:
        Maps a random seed to a fresh clustering estimator.
    contamination_levels:
        Fractions of extra outliers to inject. Must include ``0.0`` as the
        baseline and contain no negative values.
    outlier_scale:
        How many standard deviations away the synthetic outliers sit.
    random_state:
        Seed for reproducible outlier generation and estimator seeding.

    Returns
    -------
    pandas.DataFrame
        One row per contamination level with ``n_clusters``, ``silhouette`` and
        ``ami_vs_baseline``.
    """
    x = _as_2d_array(data)
    if 0.0 not in contamination_levels:
        raise ValueError("contamination_levels must include 0.0 as the baseline.")
    if any(level < 0.0 for level in contamination_levels):
        raise ValueError("contamination_levels must be non-negative.")

    rng = np.random.default_rng(random_state)
    n_rows = x.shape[0]
    centre = x.mean(axis=0)
    spread = x.std(axis=0)
    spread[spread == 0] = 1.0

    baseline_labels = np.asarray(estimator_factory(random_state).fit_predict(x))

    rows: list[dict[str, float | int | None]] = []
    for level in contamination_levels:
        n_outliers = int(round(level * n_rows))
        if n_outliers == 0:
            augmented = x
        else:
            outliers = centre + outlier_scale * spread * rng.normal(size=(n_outliers, x.shape[1]))
            augmented = np.vstack([x, outliers])

        labels = np.asarray(estimator_factory(random_state).fit_predict(augmented))
        original_labels = labels[:n_rows]
        metrics = evaluate_clustering(augmented, labels)

        if _n_non_noise_clusters(original_labels) < 2 or _n_non_noise_clusters(baseline_labels) < 2:
            ami = None
        else:
            ami = float(adjusted_mutual_info_score(baseline_labels, original_labels))

        rows.append(
            {
                "contamination": level,
                "n_clusters": metrics.n_clusters,
                "silhouette": metrics.silhouette,
                "ami_vs_baseline": ami,
            }
        )
    return pd.DataFrame(rows)


def stability_report(
    data: pd.DataFrame,
    estimator_factory: EstimatorFactory,
    n_runs: int = 10,
    n_bootstrap: int = 20,
    random_state: int = 0,
) -> str:
    """Produce a Markdown summary of clustering stability diagnostics.

    Combines repeated-run agreement, bootstrap stability, scaling sensitivity
    and outlier sensitivity into a single human-readable report suitable for a
    notebook or a generated artefact.

    Parameters
    ----------
    data:
        Numeric feature table.
    estimator_factory:
        Maps a random seed to a fresh clustering estimator.
    n_runs:
        Number of repeated fits for the seed-agreement section.
    n_bootstrap:
        Number of bootstrap subsamples.
    random_state:
        Base seed for every diagnostic.

    Returns
    -------
    str
        Markdown report.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    matrix = data.to_numpy()
    repeated = pairwise_adjusted_mutual_information(
        repeated_run_labels(matrix, estimator_factory, n_runs=n_runs, random_state=random_state)
    )
    bootstrap = bootstrap_cluster_stability(
        matrix, estimator_factory, n_bootstrap=n_bootstrap, random_state=random_state
    )
    scaling = scaling_sensitivity(data, estimator_factory, random_state=random_state)
    outliers = outlier_sensitivity(matrix, estimator_factory, random_state=random_state)

    def _fmt(value: float) -> str:
        return "n/a" if np.isnan(value) else f"{value:.3f}"

    def _table(frame: pd.DataFrame) -> str:
        return _dataframe_to_markdown(frame)

    lines = [
        "# Clustering stability report",
        "",
        "## Seed agreement (repeated runs)",
        f"- Mean pairwise AMI: **{_fmt(repeated.mean_ami)}** "
        f"(std {_fmt(repeated.std_ami)}, {repeated.n_pairs} pairs)",
        f"- Mean clusters per run: {repeated.mean_n_clusters:.2f}",
        "",
        "## Bootstrap stability",
        f"- Mean pairwise AMI on shared points: **{_fmt(bootstrap.mean_ami)}** "
        f"(std {_fmt(bootstrap.std_ami)}, {bootstrap.n_pairs} pairs)",
        f"- Mean clusters per subsample: {bootstrap.mean_n_clusters:.2f}",
        "",
        "## Scaling sensitivity",
        _table(scaling),
        "",
        "## Outlier sensitivity",
        _table(outliers),
        "",
    ]
    return "\n".join(lines)


def _dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table without extra deps."""

    def _cell(value: object) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "n/a"
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    header = "| " + " | ".join(str(column) for column in frame.columns) + " |"
    divider = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = [
        "| " + " | ".join(_cell(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *body])
