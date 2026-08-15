"""Recommender-style unsupervised learning helpers.

These utilities support a matrix-factorisation workflow over implicit
user-item interactions: build a sparse interaction matrix, learn latent factors
with TruncatedSVD or NMF, and explore the resulting embeddings through
similar-item search, product-group discovery and per-user recommendations.

Implicit feedback only records *that* an interaction happened, never a rating,
so the latent factors describe co-occurrence structure rather than preference
strength. The :func:`recommend_for_user` scores should be read as relevance
rankings, not predicted ratings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, TruncatedSVD


@dataclass(frozen=True)
class MatrixFactorization:
    """Latent factors learned from an interaction matrix.

    Attributes
    ----------
    user_factors:
        Array of shape ``(n_users, n_components)``.
    item_factors:
        Array of shape ``(n_items, n_components)``.
    method:
        Either ``"svd"`` or ``"nmf"``.
    n_components:
        Number of latent dimensions.
    """

    user_factors: NDArray[np.float64]
    item_factors: NDArray[np.float64]
    method: str
    n_components: int


def build_sparse_interactions(
    interactions: pd.DataFrame,
    n_users: int | None = None,
    n_items: int | None = None,
    user_column: str = "user_id",
    item_column: str = "item_id",
    value_column: str = "interaction_strength",
) -> csr_matrix:
    """Build a sparse user-item matrix from a long-form interaction table.

    Parameters
    ----------
    interactions:
        Long-form table with one row per interaction.
    n_users, n_items:
        Optional explicit dimensions. When omitted they are inferred as
        ``max index + 1`` so that the matrix covers every observed id.
    user_column, item_column, value_column:
        Column names in ``interactions``.

    Returns
    -------
    scipy.sparse.csr_matrix
        Matrix of shape ``(n_users, n_items)``.
    """
    if not isinstance(interactions, pd.DataFrame):
        raise TypeError("interactions must be a pandas DataFrame.")
    required = {user_column, item_column, value_column}
    missing = required - set(interactions.columns)
    if missing:
        raise ValueError(f"interactions is missing columns: {sorted(missing)}.")
    if interactions.empty:
        raise ValueError("interactions cannot be empty.")

    users = interactions[user_column].to_numpy(dtype=int)
    items = interactions[item_column].to_numpy(dtype=int)
    values = interactions[value_column].to_numpy(dtype=float)

    if (users < 0).any() or (items < 0).any():
        raise ValueError("user and item ids must be non-negative integers.")

    inferred_users = int(users.max()) + 1
    inferred_items = int(items.max()) + 1
    n_users = inferred_users if n_users is None else n_users
    n_items = inferred_items if n_items is None else n_items
    if n_users < inferred_users or n_items < inferred_items:
        raise ValueError("n_users/n_items are smaller than the observed ids.")

    return csr_matrix((values, (users, items)), shape=(n_users, n_items))


def factorize_interactions(
    matrix: csr_matrix | NDArray[np.float64],
    n_components: int = 10,
    method: str = "svd",
    random_state: int = 0,
) -> MatrixFactorization:
    """Learn latent user and item factors from an interaction matrix.

    Parameters
    ----------
    matrix:
        User-item matrix of shape ``(n_users, n_items)``.
    n_components:
        Number of latent factors. Must be smaller than both dimensions.
    method:
        ``"svd"`` for TruncatedSVD or ``"nmf"`` for non-negative matrix
        factorisation. NMF requires a non-negative matrix.
    random_state:
        Seed for the decomposition.

    Returns
    -------
    MatrixFactorization
        User and item latent factors.
    """
    array = csr_matrix(matrix)
    n_rows, n_cols = array.shape
    if n_components < 1:
        raise ValueError("n_components must be a positive integer.")
    if n_components >= min(n_rows, n_cols):
        raise ValueError("n_components must be smaller than both matrix dimensions.")

    if method == "svd":
        model = TruncatedSVD(n_components=n_components, random_state=random_state)
        user_factors = model.fit_transform(array)
        item_factors = model.components_.T
    elif method == "nmf":
        if array.min() < 0:
            raise ValueError("NMF requires a non-negative interaction matrix.")
        model = NMF(
            n_components=n_components,
            init="nndsvda",
            random_state=random_state,
            max_iter=500,
        )
        user_factors = model.fit_transform(array)
        item_factors = model.components_.T
    else:
        raise ValueError("method must be either 'svd' or 'nmf'.")

    return MatrixFactorization(
        user_factors=np.asarray(user_factors, dtype=float),
        item_factors=np.asarray(item_factors, dtype=float),
        method=method,
        n_components=n_components,
    )


def _l2_normalise(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return rows scaled to unit L2 norm, leaving zero rows untouched."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def similar_items(
    item_factors: NDArray[np.float64],
    item_id: int,
    k: int = 5,
) -> list[tuple[int, float]]:
    """Find the ``k`` items most similar to ``item_id`` by cosine similarity.

    Parameters
    ----------
    item_factors:
        Item latent factors of shape ``(n_items, n_components)``.
    item_id:
        Index of the query item.
    k:
        Number of neighbours to return.

    Returns
    -------
    list of tuple
        ``(item_id, cosine_similarity)`` pairs, most similar first, excluding
        the query item itself.
    """
    factors = np.asarray(item_factors, dtype=float)
    if factors.ndim != 2:
        raise ValueError("item_factors must be a 2D array.")
    n_items = factors.shape[0]
    if not 0 <= item_id < n_items:
        raise ValueError("item_id is out of range.")
    if k < 1:
        raise ValueError("k must be a positive integer.")

    normalised = _l2_normalise(factors)
    similarities = normalised @ normalised[item_id]
    similarities[item_id] = -np.inf
    # Never return more neighbours than exist, or the masked query item itself
    # would be handed back with a -inf similarity.
    top = np.argsort(similarities)[::-1][: min(k, n_items - 1)]
    return [(int(index), float(similarities[index])) for index in top]


def discover_item_groups(
    item_factors: NDArray[np.float64],
    n_groups: int = 5,
    random_state: int = 0,
) -> NDArray[np.int_]:
    """Cluster item embeddings into product groups using KMeans.

    Parameters
    ----------
    item_factors:
        Item latent factors of shape ``(n_items, n_components)``.
    n_groups:
        Number of product groups.
    random_state:
        Seed for KMeans.

    Returns
    -------
    numpy.ndarray
        Group label for each item.
    """
    factors = np.asarray(item_factors, dtype=float)
    if factors.ndim != 2:
        raise ValueError("item_factors must be a 2D array.")
    if not 2 <= n_groups <= factors.shape[0]:
        raise ValueError("n_groups must be between 2 and the number of items.")

    model = KMeans(n_clusters=n_groups, n_init=10, random_state=random_state)
    return np.asarray(model.fit_predict(_l2_normalise(factors)), dtype=int)


def recommend_for_user(
    factorization: MatrixFactorization,
    matrix: csr_matrix | NDArray[np.float64],
    user_id: int,
    k: int = 5,
) -> list[tuple[int, float]]:
    """Recommend items for a user from the reconstructed interaction scores.

    Items the user has already interacted with are excluded so that the output
    contains genuine discoveries.

    Parameters
    ----------
    factorization:
        Learned latent factors.
    matrix:
        The original interaction matrix, used to mask seen items.
    user_id:
        Index of the target user.
    k:
        Number of recommendations.

    Returns
    -------
    list of tuple
        ``(item_id, score)`` pairs, highest score first.
    """
    array = csr_matrix(matrix)
    n_users = factorization.user_factors.shape[0]
    if not 0 <= user_id < n_users:
        raise ValueError("user_id is out of range.")
    if k < 1:
        raise ValueError("k must be a positive integer.")

    scores = factorization.user_factors[user_id] @ factorization.item_factors.T
    seen = array.getrow(user_id).indices
    scores[seen] = -np.inf

    n_available = int(np.isfinite(scores).sum())
    top = np.argsort(scores)[::-1][: min(k, n_available)]
    return [(int(index), float(scores[index])) for index in top]
