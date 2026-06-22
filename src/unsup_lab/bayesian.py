"""Bayesian nonparametric clustering with Dirichlet-process mixtures.

Most clustering in this lab asks the analyst to choose the number of clusters
``k`` up front. A Dirichlet-process Gaussian mixture turns that around: you give
it a generous upper bound on components and the variational prior drives the
weight of the components it does not need toward zero, so the *effective* number
of clusters is inferred from the data.

This module wraps scikit-learn's
:class:`~sklearn.mixture.BayesianGaussianMixture` with helpers for counting
effective components, summarising component weights, and reporting per-point
soft-assignment uncertainty. Everything is deterministic for a fixed seed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.mixture import BayesianGaussianMixture


@dataclass(frozen=True)
class BayesianMixtureResult:
    """Outcome of fitting a Dirichlet-process Gaussian mixture.

    Attributes
    ----------
    model:
        The fitted :class:`~sklearn.mixture.BayesianGaussianMixture`.
    labels:
        Hard cluster assignment (the most probable component) per point.
    weights:
        Mixture weights in descending order.
    effective_components:
        Number of components whose weight exceeds the pruning threshold.
    """

    model: BayesianGaussianMixture
    labels: NDArray[np.int_]
    weights: NDArray[np.float64]
    effective_components: int


def _as_2d_array(data: ArrayLike) -> NDArray[np.float64]:
    """Convert input data to a validated 2D float array."""
    array = np.asarray(data, dtype=float)
    if array.ndim != 2:
        raise ValueError("data must be a 2D array.")
    if array.shape[0] < 2:
        raise ValueError("data must contain at least two rows.")
    return array


def fit_dirichlet_process_mixture(
    data: ArrayLike,
    max_components: int = 10,
    weight_concentration_prior: float | None = None,
    covariance_type: str = "full",
    max_iter: int = 200,
    random_state: int = 0,
) -> BayesianGaussianMixture:
    """Fit a Dirichlet-process Gaussian mixture to the data.

    Parameters
    ----------
    data:
        Feature matrix.
    max_components:
        Upper bound on the number of mixture components. The model may use
        fewer; unused components receive near-zero weight.
    weight_concentration_prior:
        Dirichlet-process concentration. Smaller values encourage fewer active
        components; ``None`` uses the scikit-learn default of ``1 / max_components``.
    covariance_type:
        One of ``"full"``, ``"tied"``, ``"diag"`` or ``"spherical"``.
    max_iter:
        Maximum number of variational EM iterations.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    sklearn.mixture.BayesianGaussianMixture
        The fitted estimator.
    """
    x = _as_2d_array(data)
    if max_components < 1:
        raise ValueError("max_components must be a positive integer.")
    if max_components > x.shape[0]:
        raise ValueError("max_components cannot exceed the number of rows.")

    model = BayesianGaussianMixture(
        n_components=max_components,
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=weight_concentration_prior,
        covariance_type=covariance_type,
        max_iter=max_iter,
        random_state=random_state,
    )
    model.fit(x)
    return model


def effective_components(weights: ArrayLike, threshold: float = 0.01) -> int:
    """Count mixture components whose weight exceeds a pruning threshold.

    Parameters
    ----------
    weights:
        Mixture weights (need not be sorted).
    threshold:
        Minimum weight for a component to count as active.

    Returns
    -------
    int
        The number of effective components.
    """
    weight_array = np.asarray(weights, dtype=float)
    if weight_array.ndim != 1:
        raise ValueError("weights must be a 1D array.")
    if not 0.0 <= threshold < 1.0:
        raise ValueError("threshold must be in [0, 1).")
    return int(np.sum(weight_array > threshold))


def component_weight_table(model: BayesianGaussianMixture) -> pd.DataFrame:
    """Summarise the mixture weights in descending order.

    Parameters
    ----------
    model:
        A fitted Bayesian Gaussian mixture.

    Returns
    -------
    pandas.DataFrame
        Columns ``component`` (original index), ``weight`` and
        ``cumulative_weight``.
    """
    weights = np.asarray(model.weights_, dtype=float)
    order = np.argsort(weights)[::-1]
    sorted_weights = weights[order]
    return pd.DataFrame(
        {
            "component": order.astype(int),
            "weight": sorted_weights,
            "cumulative_weight": np.cumsum(sorted_weights),
        }
    )


def assignment_uncertainty(
    model: BayesianGaussianMixture,
    data: ArrayLike,
) -> pd.DataFrame:
    """Report soft-assignment confidence and entropy for each point.

    A confidently assigned point has one responsibility near 1 (low entropy);
    a boundary point spreads its responsibility across components (high
    entropy).

    Parameters
    ----------
    model:
        A fitted Bayesian Gaussian mixture.
    data:
        Feature matrix.

    Returns
    -------
    pandas.DataFrame
        Columns ``dominant_component``, ``max_responsibility`` and ``entropy``.
    """
    x = np.asarray(data, dtype=float)
    if x.ndim != 2:
        raise ValueError("data must be a 2D array.")
    responsibilities = model.predict_proba(x)
    safe = np.clip(responsibilities, 1e-12, 1.0)
    entropy = -np.sum(responsibilities * np.log(safe), axis=1)
    return pd.DataFrame(
        {
            "dominant_component": responsibilities.argmax(axis=1).astype(int),
            "max_responsibility": responsibilities.max(axis=1).astype(float),
            "entropy": entropy.astype(float),
        }
    )


def dirichlet_process_clustering(
    data: ArrayLike,
    max_components: int = 10,
    weight_threshold: float = 0.01,
    weight_concentration_prior: float | None = None,
    random_state: int = 0,
) -> BayesianMixtureResult:
    """Fit a Dirichlet-process mixture and summarise the inferred clustering.

    Parameters
    ----------
    data:
        Feature matrix.
    max_components:
        Upper bound on the number of components.
    weight_threshold:
        Threshold used to count effective components.
    weight_concentration_prior:
        Dirichlet-process concentration prior.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    BayesianMixtureResult
        Fitted model, hard labels, sorted weights and effective component count.
    """
    x = _as_2d_array(data)
    model = fit_dirichlet_process_mixture(
        x,
        max_components=max_components,
        weight_concentration_prior=weight_concentration_prior,
        random_state=random_state,
    )
    labels = np.asarray(model.predict(x), dtype=int)
    sorted_weights = np.sort(np.asarray(model.weights_, dtype=float))[::-1]
    active = effective_components(model.weights_, threshold=weight_threshold)
    return BayesianMixtureResult(
        model=model,
        labels=labels,
        weights=sorted_weights,
        effective_components=active,
    )
