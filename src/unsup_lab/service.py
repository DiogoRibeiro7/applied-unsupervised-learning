"""Reusable modelling pipelines shared by the CLI and the API.

These functions wrap the notebook modelling choices into self-contained
scikit-learn pipelines so that the same code path trains a model, scores new
data, and powers the production endpoints. Keeping them here avoids duplicating
modelling logic across the command line and the web service.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_clustering_pipeline(n_clusters: int, random_state: int = 0) -> Pipeline:
    """Create a standardise-then-KMeans clustering pipeline."""
    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2.")
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)),
        ]
    )


def train_clustering(
    features: pd.DataFrame,
    n_clusters: int,
    random_state: int = 0,
) -> tuple[Pipeline, NDArray[np.int_]]:
    """Fit a clustering pipeline and return it together with the labels."""
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame.")
    if features.empty:
        raise ValueError("features cannot be empty.")

    pipeline = build_clustering_pipeline(n_clusters, random_state=random_state)
    labels = pipeline.fit_predict(features.to_numpy())
    return pipeline, np.asarray(labels, dtype=int)


def assign_clusters(pipeline: Pipeline, features: pd.DataFrame) -> NDArray[np.int_]:
    """Assign cluster labels to new rows using a fitted clustering pipeline."""
    return np.asarray(pipeline.predict(features.to_numpy()), dtype=int)


def build_anomaly_pipeline(
    contamination: float = 0.05,
    random_state: int = 0,
) -> Pipeline:
    """Create a standardise-then-IsolationForest anomaly pipeline."""
    if not 0.0 < contamination < 0.5:
        raise ValueError("contamination must be between 0 and 0.5.")
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "forest",
                IsolationForest(contamination=contamination, random_state=random_state),
            ),
        ]
    )


def train_anomaly_model(
    features: pd.DataFrame,
    contamination: float = 0.05,
    random_state: int = 0,
) -> Pipeline:
    """Fit an anomaly-detection pipeline on numeric features."""
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame.")
    if features.empty:
        raise ValueError("features cannot be empty.")

    pipeline = build_anomaly_pipeline(contamination=contamination, random_state=random_state)
    pipeline.fit(features.to_numpy())
    return pipeline


def anomaly_scores(pipeline: Pipeline, features: pd.DataFrame) -> NDArray[np.float64]:
    """Return anomaly scores where higher values mean more anomalous.

    Isolation Forest's ``score_samples`` returns higher values for inliers, so
    the sign is flipped to give an intuitive "higher is more anomalous" score.
    """
    raw = pipeline.named_steps["forest"].score_samples(
        pipeline.named_steps["scaler"].transform(features.to_numpy())
    )
    return np.asarray(-raw, dtype=float)


def train_topic_model(
    documents: list[str],
    n_topics: int = 4,
    n_terms: int = 10,
    random_state: int = 0,
) -> tuple[Pipeline, dict[str, list[str]]]:
    """Fit a TF-IDF + NMF topic model and return top terms per topic.

    Parameters
    ----------
    documents:
        Raw text documents.
    n_topics:
        Number of topics to learn.
    n_terms:
        Number of top terms reported per topic.
    random_state:
        Seed for NMF.

    Returns
    -------
    tuple
        The fitted pipeline and a mapping ``topic_i -> [terms]``.
    """
    if len(documents) < n_topics:
        raise ValueError("number of documents must be at least n_topics.")
    if n_topics < 2:
        raise ValueError("n_topics must be at least 2.")

    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(min_df=2, stop_words="english")),
            (
                "nmf",
                NMF(
                    n_components=n_topics,
                    init="nndsvda",
                    random_state=random_state,
                    max_iter=500,
                ),
            ),
        ]
    )
    pipeline.fit(documents)

    vocabulary = np.asarray(pipeline.named_steps["tfidf"].get_feature_names_out())
    components = pipeline.named_steps["nmf"].components_
    topics: dict[str, list[str]] = {}
    for index, weights in enumerate(components):
        top = np.argsort(weights)[::-1][:n_terms]
        topics[f"topic_{index}"] = vocabulary[top].tolist()
    return pipeline, topics
