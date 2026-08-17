"""Turning a partition into something a person can act on.

A cluster label is an integer. Nobody acts on an integer. Between "the model
found five groups" and "here is what we will do about group three" sits work
this module does:

* :func:`surrogate_rules` fits a shallow decision tree *to the cluster labels*,
  so the partition can be read as if-then rules rather than distances. It
  reports its own fidelity, because a rule set that only half-reproduces the
  clustering is a misleading explanation rather than a simple one.
* :func:`distinctive_features` says what actually separates each cluster from
  the population, in units of population spread.
* :func:`name_clusters` turns that into a short readable name.
* :func:`cluster_exemplars` and :func:`boundary_cases` return real rows - the
  members that typify a cluster, and the members that only just belong to it.
* :func:`persona_cards` assembles the lot into a Markdown brief.

Two deliberate limits. Distances are computed on standardised features, because
a cluster centre is meaningless when one column is measured in hundreds and
another in tenths; the rows handed back are in original units, because that is
what a reader can interpret. And nothing here invents a business action - the
cards carry the evidence, and naming what to *do* about a segment is a judgement
that needs a domain, a cost and an owner, none of which live in the data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.tree import DecisionTreeClassifier, export_text


@dataclass(frozen=True)
class ClusterSurrogate:
    """A rule-based approximation of a clustering.

    Attributes
    ----------
    rules:
        The decision tree rendered as indented if-then text.
    fidelity:
        Fraction of points whose cluster the rules reproduce. This is measured
        on the same data the tree was fitted to, which is the right question for
        a surrogate: not "does it generalise" but "does it describe what the
        clustering did".
    depth:
        Depth of the fitted tree.
    n_leaves:
        Number of leaves, i.e. how many distinct rule paths a reader must hold.
    """

    rules: str
    fidelity: float
    depth: int
    n_leaves: int


def _validated(features: pd.DataFrame, labels: ArrayLike) -> tuple[pd.DataFrame, NDArray[np.int_]]:
    """Check the feature table and labels, dropping DBSCAN noise points."""
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame.")
    if features.empty:
        raise ValueError("features cannot be empty.")
    if not all(pd.api.types.is_numeric_dtype(features[column]) for column in features.columns):
        raise TypeError("all feature columns must be numeric.")

    label_array = np.asarray(labels)
    if label_array.ndim != 1:
        raise ValueError("labels must be a 1D array.")
    if label_array.shape[0] != features.shape[0]:
        raise ValueError("features and labels must have the same number of rows.")

    # Noise is not a persona: a DBSCAN -1 group has no centre worth describing.
    keep = label_array != -1
    kept_features = features.loc[keep]
    kept_labels = label_array[keep].astype(int)
    if len(set(kept_labels.tolist())) < 2:
        raise ValueError("need at least two non-noise clusters to explain.")
    return kept_features, kept_labels


def _standardised(features: pd.DataFrame) -> NDArray[np.float64]:
    """Standardise columns so distances are not dominated by units."""
    values = features.to_numpy(dtype=float)
    spread = values.std(axis=0)
    spread[spread == 0] = 1.0
    return (values - values.mean(axis=0)) / spread


def _centroids(scaled: NDArray[np.float64], labels: NDArray[np.int_]) -> dict[int, NDArray]:
    """Mean position of each cluster in standardised space."""
    return {int(c): scaled[labels == c].mean(axis=0) for c in np.unique(labels)}


def surrogate_rules(
    features: pd.DataFrame,
    labels: ArrayLike,
    max_depth: int = 3,
    random_state: int = 0,
) -> ClusterSurrogate:
    """Approximate a clustering with a shallow decision tree.

    The tree is fitted to predict the cluster label from the features, which
    converts an opaque partition into a handful of readable thresholds. Keeping
    it shallow is the point: a deep tree would reproduce the clustering exactly
    and explain nothing.

    Parameters
    ----------
    features:
        Numeric feature table, in the units you want the rules expressed in.
    labels:
        Cluster label per row. DBSCAN noise (``-1``) is excluded.
    max_depth:
        Depth limit for the surrogate tree.
    random_state:
        Seed for the tree.

    Returns
    -------
    ClusterSurrogate
        The rules and how faithfully they reproduce the partition.
    """
    kept, kept_labels = _validated(features, labels)
    if max_depth < 1:
        raise ValueError("max_depth must be at least 1.")

    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    tree.fit(kept, kept_labels)
    predicted = tree.predict(kept)

    return ClusterSurrogate(
        rules=export_text(tree, feature_names=list(kept.columns), decimals=2),
        fidelity=float(np.mean(predicted == kept_labels)),
        depth=int(tree.get_depth()),
        n_leaves=int(tree.get_n_leaves()),
    )


def distinctive_features(
    features: pd.DataFrame,
    labels: ArrayLike,
    n_features: int = 3,
) -> pd.DataFrame:
    """Rank what separates each cluster from the population.

    For every cluster and feature, the cluster mean is expressed as a deviation
    from the overall mean in units of the overall standard deviation, so
    features on different scales can be compared. The largest deviations are
    what makes a cluster that cluster.

    Parameters
    ----------
    features:
        Numeric feature table.
    labels:
        Cluster label per row; noise (``-1``) is excluded.
    n_features:
        How many features to report per cluster.

    Returns
    -------
    pandas.DataFrame
        Columns ``cluster``, ``feature``, ``cluster_mean``, ``overall_mean``,
        ``deviation`` (signed, in population standard deviations) and
        ``direction`` (``"high"`` or ``"low"``), most distinctive first.
    """
    kept, kept_labels = _validated(features, labels)
    if n_features < 1:
        raise ValueError("n_features must be at least 1.")

    overall_mean = kept.mean()
    overall_std = kept.std(ddof=0).replace(0.0, 1.0)

    rows: list[dict[str, object]] = []
    for cluster in sorted(set(kept_labels.tolist())):
        members = kept.loc[kept_labels == cluster]
        deviation = (members.mean() - overall_mean) / overall_std
        for feature in deviation.abs().sort_values(ascending=False).head(n_features).index:
            rows.append(
                {
                    "cluster": cluster,
                    "feature": str(feature),
                    "cluster_mean": float(members[feature].mean()),
                    "overall_mean": float(overall_mean[feature]),
                    "deviation": float(deviation[feature]),
                    "direction": "high" if deviation[feature] > 0 else "low",
                }
            )
    return pd.DataFrame(rows)


def name_clusters(
    features: pd.DataFrame,
    labels: ArrayLike,
    n_features: int = 2,
) -> dict[int, str]:
    """Generate a short readable name per cluster from its distinctive features.

    Names read like ``"high avg_order_value, low discount_ratio"``. They are
    descriptions of what the numbers do, not business identities: calling a
    group "loyal high-value customers" asserts a motive the data cannot support,
    and that leap belongs to a domain expert.

    Parameters
    ----------
    features:
        Numeric feature table.
    labels:
        Cluster label per row; noise (``-1``) is excluded.
    n_features:
        How many features to combine into each name.

    Returns
    -------
    dict
        Mapping from cluster label to name.
    """
    table = distinctive_features(features, labels, n_features=n_features)
    names: dict[int, str] = {}
    # Selecting per cluster rather than unpacking groupby keys, whose type a
    # checker can only see as "some scalar".
    for cluster in sorted(table["cluster"].unique().tolist()):
        group = table[table["cluster"] == cluster]
        parts = [f"{row.direction} {row.feature}" for row in group.itertuples()]
        names[int(cluster)] = ", ".join(parts)
    return names


def cluster_exemplars(
    features: pd.DataFrame,
    labels: ArrayLike,
    n_examples: int = 3,
) -> pd.DataFrame:
    """Return the rows that most typify each cluster.

    Typicality is nearness to the cluster centre in standardised space; the rows
    come back in their original units.

    Parameters
    ----------
    features:
        Numeric feature table.
    labels:
        Cluster label per row; noise (``-1``) is excluded.
    n_examples:
        How many exemplars to return per cluster.

    Returns
    -------
    pandas.DataFrame
        The selected rows with ``cluster`` and ``distance_to_centre`` columns,
        keeping the original index so they can be traced back.
    """
    kept, kept_labels = _validated(features, labels)
    if n_examples < 1:
        raise ValueError("n_examples must be at least 1.")

    scaled = _standardised(kept)
    centroids = _centroids(scaled, kept_labels)

    frames: list[pd.DataFrame] = []
    for cluster, centre in centroids.items():
        mask = kept_labels == cluster
        distances = np.linalg.norm(scaled[mask] - centre, axis=1)
        chosen = np.argsort(distances)[:n_examples]
        block = kept.loc[mask].iloc[chosen].copy()
        block.insert(0, "cluster", cluster)
        block["distance_to_centre"] = distances[chosen]
        frames.append(block)
    return pd.concat(frames)


def boundary_cases(
    features: pd.DataFrame,
    labels: ArrayLike,
    n_examples: int = 3,
) -> pd.DataFrame:
    """Return the rows that only just belong to their cluster.

    For each point, the margin is the distance to the nearest *other* cluster
    centre minus the distance to its own. Small margins mark members that a
    slightly different fit would have assigned elsewhere - the rows worth
    checking before a segmentation is acted upon, and the ones a profile of
    means hides completely.

    Parameters
    ----------
    features:
        Numeric feature table.
    labels:
        Cluster label per row; noise (``-1``) is excluded.
    n_examples:
        How many boundary cases to return per cluster.

    Returns
    -------
    pandas.DataFrame
        The selected rows with ``cluster``, ``nearest_other_cluster`` and
        ``margin`` columns, smallest margin first within each cluster.
    """
    kept, kept_labels = _validated(features, labels)
    if n_examples < 1:
        raise ValueError("n_examples must be at least 1.")

    scaled = _standardised(kept)
    centroids = _centroids(scaled, kept_labels)
    order = sorted(centroids)
    centre_matrix = np.vstack([centroids[c] for c in order])

    # Distance from every point to every cluster centre.
    distances = np.linalg.norm(scaled[:, None, :] - centre_matrix[None, :, :], axis=2)
    own_index = np.array([order.index(int(label)) for label in kept_labels])
    own_distance = distances[np.arange(distances.shape[0]), own_index]

    masked = distances.copy()
    masked[np.arange(masked.shape[0]), own_index] = np.inf
    nearest_other = masked.argmin(axis=1)
    margin = masked[np.arange(masked.shape[0]), nearest_other] - own_distance

    frames: list[pd.DataFrame] = []
    for position, cluster in enumerate(order):
        mask = own_index == position
        cluster_margin = margin[mask]
        chosen = np.argsort(cluster_margin)[:n_examples]
        block = kept.loc[mask].iloc[chosen].copy()
        block.insert(0, "cluster", cluster)
        block["nearest_other_cluster"] = [order[i] for i in nearest_other[mask][chosen]]
        block["margin"] = cluster_margin[chosen]
        frames.append(block)
    return pd.concat(frames)


def persona_cards(
    features: pd.DataFrame,
    labels: ArrayLike,
    n_features: int = 3,
    n_examples: int = 2,
) -> str:
    """Assemble one Markdown brief per cluster.

    Each card carries the size and share of the segment, its generated name,
    what distinguishes it, and real example members. It does *not* propose an
    action: what to do about a segment depends on cost, capacity and strategy
    that the feature table knows nothing about, so the card is written to be the
    input to that conversation rather than a substitute for it.

    Parameters
    ----------
    features:
        Numeric feature table.
    labels:
        Cluster label per row; noise (``-1``) is excluded.
    n_features:
        Distinctive features to list per card.
    n_examples:
        Exemplar rows to show per card.

    Returns
    -------
    str
        Markdown, one section per cluster.
    """
    kept, kept_labels = _validated(features, labels)
    names = name_clusters(kept, kept_labels, n_features=min(2, n_features))
    distinctive = distinctive_features(kept, kept_labels, n_features=n_features)
    exemplars = cluster_exemplars(kept, kept_labels, n_examples=n_examples)
    total = int(kept.shape[0])

    lines = ["# Segment cards", ""]
    for cluster in sorted(set(kept_labels.tolist())):
        size = int(np.sum(kept_labels == cluster))
        lines += [
            f"## Segment {cluster} - {names[cluster]}",
            "",
            f"**Size.** {size} of {total} ({size / total:.1%}).",
            "",
            "**What sets it apart** (deviation from the population, in standard deviations):",
            "",
        ]
        for row in distinctive[distinctive["cluster"] == cluster].itertuples():
            lines.append(
                f"- `{row.feature}` {row.direction}: {row.cluster_mean:.2f} "
                f"against {row.overall_mean:.2f} overall ({row.deviation:+.2f} sd)"
            )
        lines += ["", "**Example members** (closest to the segment centre, index shown):", ""]
        for row in exemplars[exemplars["cluster"] == cluster].itertuples():
            values = ", ".join(
                f"{column}={getattr(row, column):.2f}"
                for column in kept.columns
                if isinstance(getattr(row, column), (int, float, np.number))
            )
            lines.append(f"- `{row.Index}`: {values}")
        lines += [
            "",
            "**Action.** To be decided with the domain owner; the data supports the "
            "description above, not a decision about what to do with it.",
            "",
        ]
    return "\n".join(lines)
