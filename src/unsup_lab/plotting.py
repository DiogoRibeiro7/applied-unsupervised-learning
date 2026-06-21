"""Plotting helpers for notebook workflows."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def plot_embedding(
    embedding: ArrayLike,
    labels: ArrayLike | None = None,
    title: str = "2D embedding",
) -> None:
    """Plot a two-dimensional embedding.

    Parameters
    ----------
    embedding:
        Two-dimensional matrix with exactly two columns.
    labels:
        Optional cluster labels.
    title:
        Plot title.
    """
    values = np.asarray(embedding)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("embedding must be a 2D array with exactly two columns.")

    plt.figure(figsize=(8, 6))
    if labels is None:
        plt.scatter(values[:, 0], values[:, 1], s=18, alpha=0.7)
    else:
        plt.scatter(values[:, 0], values[:, 1], c=np.asarray(labels), s=18, alpha=0.7)
    plt.title(title)
    plt.xlabel("component 1")
    plt.ylabel("component 2")
    plt.tight_layout()
    plt.show()


def plot_metric_curve(
    metrics: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
) -> None:
    """Plot a metric curve from a DataFrame."""
    if x_column not in metrics.columns or y_column not in metrics.columns:
        raise ValueError("x_column and y_column must exist in metrics.")

    plt.figure(figsize=(8, 5))
    plt.plot(metrics[x_column], metrics[y_column], marker="o")
    plt.title(title)
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.tight_layout()
    plt.show()
