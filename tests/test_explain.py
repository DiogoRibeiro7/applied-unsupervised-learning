from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from unsup_lab.explain import (
    ClusterSurrogate,
    boundary_cases,
    cluster_exemplars,
    distinctive_features,
    name_clusters,
    persona_cards,
    surrogate_rules,
)


def _planted_segments() -> tuple[pd.DataFrame, np.ndarray]:
    """Three segments separated on known features, on deliberately unequal scales."""
    rng = np.random.default_rng(0)
    blocks, labels = [], []
    # spend is in the hundreds, engagement in tenths: a distance on raw units
    # would be decided by spend alone.
    for cluster, (spend, visits, engagement) in enumerate(
        [(500.0, 5.0, 0.2), (100.0, 30.0, 0.8), (300.0, 15.0, 0.5)]
    ):
        blocks.append(
            pd.DataFrame(
                {
                    "spend": rng.normal(spend, 20.0, 80),
                    "visits": rng.normal(visits, 2.0, 80),
                    "engagement": rng.normal(engagement, 0.05, 80),
                }
            )
        )
        labels += [cluster] * 80
    return pd.concat(blocks, ignore_index=True), np.array(labels)


def test_surrogate_reproduces_a_separable_partition() -> None:
    features, labels = _planted_segments()

    surrogate = surrogate_rules(features, labels, max_depth=3)

    assert isinstance(surrogate, ClusterSurrogate)
    assert surrogate.fidelity > 0.95  # the segments are separable, so rules should describe them
    assert surrogate.depth <= 3
    assert "spend" in surrogate.rules or "visits" in surrogate.rules


def test_surrogate_fidelity_falls_when_the_partition_is_not_rule_shaped() -> None:
    # Labels assigned at random cannot be described by three thresholds, and the
    # reported fidelity has to say so rather than flattering the explanation.
    features, _ = _planted_segments()
    noise_labels = np.random.default_rng(1).integers(0, 3, size=features.shape[0])

    honest = surrogate_rules(features, noise_labels, max_depth=2)

    assert honest.fidelity < 0.6


def test_distinctive_features_finds_the_planted_separation() -> None:
    features, labels = _planted_segments()

    table = distinctive_features(features, labels, n_features=2)

    assert list(table.columns) == [
        "cluster",
        "feature",
        "cluster_mean",
        "overall_mean",
        "deviation",
        "direction",
    ]
    top = table[table["cluster"] == 0].iloc[0]
    assert top["direction"] == "high"  # cluster 0 has the highest spend
    assert abs(top["deviation"]) > 1.0


def test_names_describe_direction_and_feature() -> None:
    features, labels = _planted_segments()

    names = name_clusters(features, labels, n_features=2)

    assert set(names) == {0, 1, 2}
    assert all(word in names[0] for word in ("high", "low"))  # a name states directions
    assert any(column in names[0] for column in features.columns)


def test_exemplars_are_the_most_typical_members() -> None:
    features, labels = _planted_segments()

    exemplars = cluster_exemplars(features, labels, n_examples=2)

    assert len(exemplars) == 6  # 3 clusters x 2
    assert set(exemplars["cluster"]) == {0, 1, 2}
    # An exemplar must sit closer to its own segment's mean than a typical member.
    for cluster in (0, 1, 2):
        chosen = exemplars[exemplars["cluster"] == cluster]
        assert chosen["spend"].std() <= features.loc[labels == cluster, "spend"].std() + 1e-9


def test_boundary_cases_are_exactly_the_narrowest_margins() -> None:
    features, labels = _planted_segments()

    boundary = boundary_cases(features, labels, n_examples=3)

    assert set(boundary.columns) >= {"cluster", "nearest_other_cluster", "margin"}
    assert (boundary["margin"] >= 0).all()
    assert (boundary["cluster"] != boundary["nearest_other_cluster"]).all()

    # Independent oracle: recompute every point's margin and confirm the
    # function returned the narrowest ones, in order.
    values = features.to_numpy(dtype=float)
    scaled = (values - values.mean(axis=0)) / values.std(axis=0)
    centres = np.vstack([scaled[labels == c].mean(axis=0) for c in (0, 1, 2)])
    distances = np.linalg.norm(scaled[:, None, :] - centres[None, :, :], axis=2)
    own = distances[np.arange(len(labels)), labels]
    others = distances.copy()
    others[np.arange(len(labels)), labels] = np.inf
    margins = others.min(axis=1) - own

    for cluster in (0, 1, 2):
        expected = np.sort(margins[labels == cluster])[:3]
        returned = boundary[boundary["cluster"] == cluster]["margin"].to_numpy()
        np.testing.assert_allclose(returned, expected)  # narrowest three, ascending


def test_scaling_does_not_let_one_column_decide_the_geometry() -> None:
    # Multiplying a column by 1000 must not change which rows are exemplars,
    # because distances are taken on standardised features.
    features, labels = _planted_segments()
    rescaled = features.copy()
    rescaled["spend"] = rescaled["spend"] * 1000.0

    original = cluster_exemplars(features, labels, n_examples=2).index.tolist()
    after = cluster_exemplars(rescaled, labels, n_examples=2).index.tolist()

    assert original == after


def test_persona_cards_report_evidence_and_refuse_to_invent_actions() -> None:
    features, labels = _planted_segments()

    cards = persona_cards(features, labels, n_features=2, n_examples=1)

    assert cards.startswith("# Segment cards")
    for cluster in (0, 1, 2):
        assert f"## Segment {cluster}" in cards
    assert "Size." in cards and "sd)" in cards
    # The action line must defer rather than fabricate a recommendation.
    assert "To be decided with the domain owner" in cards


def test_noise_points_are_excluded_rather_than_described() -> None:
    features, labels = _planted_segments()
    with_noise = labels.copy()
    with_noise[:10] = -1

    names = name_clusters(features, with_noise)
    exemplars = cluster_exemplars(features, with_noise, n_examples=1)

    assert -1 not in names
    assert -1 not in set(exemplars["cluster"])


def test_invalid_inputs_raise() -> None:
    features, labels = _planted_segments()

    with pytest.raises(TypeError):
        surrogate_rules(features.to_numpy(), labels)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        surrogate_rules(features, labels[:-1])
    with pytest.raises(ValueError):
        surrogate_rules(features, np.zeros(len(features), dtype=int))  # only one cluster
    with pytest.raises(ValueError):
        cluster_exemplars(features, labels, n_examples=0)
    with pytest.raises(TypeError):
        distinctive_features(features.assign(note="x"), labels)
