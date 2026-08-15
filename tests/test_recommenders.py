from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from unsup_lab.data import make_user_item_interactions
from unsup_lab.recommenders import (
    MatrixFactorization,
    build_sparse_interactions,
    discover_item_groups,
    factorize_interactions,
    recommend_for_user,
    similar_items,
)


def test_build_sparse_interactions_shape_and_values() -> None:
    interactions = pd.DataFrame(
        {
            "user_id": [0, 0, 2],
            "item_id": [1, 3, 0],
            "interaction_strength": [1.5, 2.0, 0.5],
        }
    )

    matrix = build_sparse_interactions(interactions)

    assert isinstance(matrix, csr_matrix)
    assert matrix.shape == (3, 4)
    assert matrix[0, 1] == 1.5
    assert matrix[2, 0] == 0.5


def test_build_sparse_interactions_missing_column_raises() -> None:
    interactions = pd.DataFrame({"user_id": [0], "item_id": [1]})

    with pytest.raises(ValueError):
        build_sparse_interactions(interactions)


def test_build_sparse_interactions_explicit_dims_too_small_raises() -> None:
    interactions = pd.DataFrame(
        {"user_id": [0, 5], "item_id": [1, 2], "interaction_strength": [1.0, 1.0]}
    )

    with pytest.raises(ValueError):
        build_sparse_interactions(interactions, n_users=3)


def test_factorize_svd_schema() -> None:
    _, matrix = make_user_item_interactions(n_users=60, n_items=30, random_state=0)

    factors = factorize_interactions(matrix, n_components=5, method="svd", random_state=0)

    assert isinstance(factors, MatrixFactorization)
    assert factors.user_factors.shape == (60, 5)
    assert factors.item_factors.shape == (30, 5)
    assert factors.method == "svd"


def test_factorize_nmf_is_non_negative() -> None:
    _, matrix = make_user_item_interactions(n_users=60, n_items=30, random_state=1)

    factors = factorize_interactions(matrix, n_components=4, method="nmf", random_state=0)

    assert (factors.user_factors >= 0).all()
    assert (factors.item_factors >= 0).all()


def test_factorize_is_deterministic() -> None:
    _, matrix = make_user_item_interactions(n_users=50, n_items=25, random_state=2)

    a = factorize_interactions(matrix, n_components=4, random_state=7)
    b = factorize_interactions(matrix, n_components=4, random_state=7)

    np.testing.assert_allclose(a.item_factors, b.item_factors)


def test_factorize_too_many_components_raises() -> None:
    _, matrix = make_user_item_interactions(n_users=10, n_items=8, random_state=0)

    with pytest.raises(ValueError):
        factorize_interactions(matrix, n_components=8)


def test_factorize_invalid_method_raises() -> None:
    _, matrix = make_user_item_interactions(n_users=20, n_items=10, random_state=0)

    with pytest.raises(ValueError):
        factorize_interactions(matrix, n_components=3, method="pca")


def test_similar_items_recovers_latent_group() -> None:
    # Two clearly separated item groups in latent space.
    item_factors = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ]
    )

    neighbours = similar_items(item_factors, item_id=0, k=1)

    assert neighbours[0][0] == 1  # nearest neighbour shares the group
    assert 0 not in [index for index, _ in neighbours]  # query excluded


def test_similar_items_caps_k_at_the_catalogue_size() -> None:
    item_factors = np.eye(3)

    neighbours = similar_items(item_factors, item_id=0, k=10)

    # Only two other items exist, and the query is never one of them.
    assert len(neighbours) == 2
    assert 0 not in [index for index, _ in neighbours]


def test_similar_items_out_of_range_raises() -> None:
    item_factors = np.eye(3)

    with pytest.raises(ValueError):
        similar_items(item_factors, item_id=5)


def test_discover_item_groups_labels() -> None:
    _, matrix = make_user_item_interactions(n_users=80, n_items=40, random_state=3)
    factors = factorize_interactions(matrix, n_components=6, random_state=0)

    groups = discover_item_groups(factors.item_factors, n_groups=5, random_state=0)

    assert groups.shape == (40,)
    assert set(np.unique(groups)).issubset(set(range(5)))


def test_discover_item_groups_invalid_n_raises() -> None:
    factors = np.random.default_rng(0).normal(size=(5, 3))

    with pytest.raises(ValueError):
        discover_item_groups(factors, n_groups=10)


def test_recommend_for_user_excludes_seen_items() -> None:
    _, matrix = make_user_item_interactions(n_users=60, n_items=30, random_state=4)
    factors = factorize_interactions(matrix, n_components=5, random_state=0)

    recommendations = recommend_for_user(factors, matrix, user_id=0, k=5)
    seen = set(csr_matrix(matrix).getrow(0).indices.tolist())
    recommended = {index for index, _ in recommendations}

    assert len(recommendations) <= 5
    assert recommended.isdisjoint(seen)


def test_recommend_for_user_out_of_range_raises() -> None:
    _, matrix = make_user_item_interactions(n_users=10, n_items=8, random_state=0)
    factors = factorize_interactions(matrix, n_components=3, random_state=0)

    with pytest.raises(ValueError):
        recommend_for_user(factors, matrix, user_id=99)
