from __future__ import annotations

import numpy as np
import pytest

from unsup_lab.data import (
    make_customer_segmentation_data,
    make_document_corpus,
    make_sensor_anomaly_data,
)
from unsup_lab.service import (
    anomaly_scores,
    assign_clusters,
    train_anomaly_model,
    train_clustering,
    train_topic_model,
)


def test_train_clustering_assigns_all_rows() -> None:
    dataset = make_customer_segmentation_data(n_customers=200, random_state=0)

    pipeline, labels = train_clustering(dataset.features, n_clusters=5, random_state=0)

    assert labels.shape == (200,)
    assert set(np.unique(labels)).issubset(set(range(5)))
    # The fitted pipeline can assign the same rows again.
    reassigned = assign_clusters(pipeline, dataset.features)
    np.testing.assert_array_equal(labels, reassigned)


def test_train_clustering_requires_two_clusters() -> None:
    dataset = make_customer_segmentation_data(n_customers=20, random_state=0)

    with pytest.raises(ValueError):
        train_clustering(dataset.features, n_clusters=1)


def test_anomaly_scores_rank_injected_anomalies_high() -> None:
    dataset = make_sensor_anomaly_data(n_points=800, contamination=0.05, random_state=0)
    numeric = dataset.features.drop(columns=["timestamp"])

    pipeline = train_anomaly_model(numeric, contamination=0.05, random_state=0)
    scores = anomaly_scores(pipeline, numeric)

    top_k = np.argsort(scores)[::-1][:40]
    hidden = dataset.hidden_labels.to_numpy()
    precision = (hidden[top_k] == "anomaly").mean()
    assert precision > 0.5


def test_train_topic_model_returns_terms() -> None:
    corpus = make_document_corpus(random_state=0)

    pipeline, topics = train_topic_model(corpus["text"].tolist(), n_topics=4, random_state=0)

    assert len(topics) == 4
    assert all(len(terms) > 0 for terms in topics.values())
    assert pipeline.named_steps["nmf"].n_components == 4


def test_train_topic_model_too_few_documents_raises() -> None:
    with pytest.raises(ValueError):
        train_topic_model(["one document"], n_topics=4)
