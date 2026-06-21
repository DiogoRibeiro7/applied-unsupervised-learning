"""Reporting helpers for unsupervised learning notebooks."""

from __future__ import annotations

import pandas as pd


def cluster_profile(
    features: pd.DataFrame,
    labels: pd.Series | list[int],
) -> pd.DataFrame:
    """Create a cluster profile table using feature means.

    Parameters
    ----------
    features:
        Feature table used for clustering.
    labels:
        Cluster labels.

    Returns
    -------
    pandas.DataFrame
        Mean feature values by cluster and cluster size.
    """
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame.")
    if features.empty:
        raise ValueError("features cannot be empty.")

    label_series = pd.Series(labels, index=features.index, name="cluster")
    if len(label_series) != len(features):
        raise ValueError("labels must have the same length as features.")

    profiled = features.copy()
    profiled["cluster"] = label_series

    summary = profiled.groupby("cluster").mean(numeric_only=True)
    summary["cluster_size"] = profiled.groupby("cluster").size()
    return summary.sort_index()


def top_terms_by_component(
    components: pd.DataFrame,
    n_terms: int = 10,
) -> dict[str, list[str]]:
    """Return the top terms for each topic/component."""
    if n_terms <= 0:
        raise ValueError("n_terms must be positive.")

    result: dict[str, list[str]] = {}
    for component_name, row in components.iterrows():
        result[str(component_name)] = row.sort_values(ascending=False).head(n_terms).index.tolist()
    return result
