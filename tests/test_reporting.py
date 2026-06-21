from __future__ import annotations

import pandas as pd
import pytest

from unsup_lab.reporting import cluster_profile, top_terms_by_component


def test_cluster_profile_means_and_sizes() -> None:
    features = pd.DataFrame(
        {
            "x": [0.0, 0.0, 10.0, 10.0],
            "y": [1.0, 3.0, 1.0, 3.0],
        }
    )
    labels = [0, 0, 1, 1]

    profile = cluster_profile(features, labels)

    assert list(profile.index) == [0, 1]
    assert profile.loc[0, "x"] == 0.0
    assert profile.loc[1, "x"] == 10.0
    assert profile.loc[0, "cluster_size"] == 2


def test_cluster_profile_length_mismatch_raises() -> None:
    features = pd.DataFrame({"x": [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError):
        cluster_profile(features, [0, 1])


def test_cluster_profile_non_dataframe_raises() -> None:
    with pytest.raises(TypeError):
        cluster_profile([[1, 2]], [0])  # type: ignore[arg-type]


def test_top_terms_by_component() -> None:
    components = pd.DataFrame(
        [[0.1, 0.9, 0.5], [0.8, 0.2, 0.7]],
        index=["topic_0", "topic_1"],
        columns=["alpha", "beta", "gamma"],
    )

    top = top_terms_by_component(components, n_terms=2)

    assert top["topic_0"] == ["beta", "gamma"]
    assert top["topic_1"] == ["alpha", "gamma"]


def test_top_terms_invalid_n_raises() -> None:
    components = pd.DataFrame([[1.0]], index=["topic_0"], columns=["alpha"])

    with pytest.raises(ValueError):
        top_terms_by_component(components, n_terms=0)
