from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # Non-interactive backend for headless test runs.

from unsup_lab.plotting import plot_embedding, plot_metric_curve  # noqa: E402


def test_plot_embedding_runs() -> None:
    embedding = np.random.default_rng(0).normal(size=(20, 2))
    labels = np.array([0, 1] * 10)

    plot_embedding(embedding, labels=labels, title="test")
    plot_embedding(embedding, labels=None)


def test_plot_embedding_wrong_shape_raises() -> None:
    with pytest.raises(ValueError):
        plot_embedding(np.zeros((10, 3)))


def test_plot_metric_curve_runs() -> None:
    metrics = pd.DataFrame({"k": [2, 3, 4], "silhouette": [0.4, 0.5, 0.45]})

    plot_metric_curve(metrics, "k", "silhouette", title="silhouette vs k")


def test_plot_metric_curve_missing_column_raises() -> None:
    metrics = pd.DataFrame({"k": [2, 3]})

    with pytest.raises(ValueError):
        plot_metric_curve(metrics, "k", "missing", title="bad")
