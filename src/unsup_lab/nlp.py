"""Helpers for unsupervised NLP topic modelling.

These functions cover the supporting steps around a TF-IDF + NMF / LSA topic
model: light text cleaning, turning component matrices into readable topic-term
tables and auto-generated labels, assigning documents to topics with a
confidence, and quality diagnostics (UMass coherence and topic diversity).

They depend only on NumPy, pandas, scipy and scikit-learn, and never call an
external service.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_matrix, issparse

_TOKEN_CLEANUP = re.compile(r"[^a-z\s]+")
_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase text and strip punctuation, digits and excess whitespace.

    Parameters
    ----------
    text:
        Raw document text.

    Returns
    -------
    str
        Cleaned text containing only lowercase words separated by single spaces.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    lowered = text.lower()
    without_symbols = _TOKEN_CLEANUP.sub(" ", lowered)
    return _WHITESPACE.sub(" ", without_symbols).strip()


def topic_term_table(
    components: NDArray[np.float64],
    feature_names: ArrayLike,
    n_terms: int = 10,
) -> pd.DataFrame:
    """Build a long-form table of the top terms for each topic.

    Parameters
    ----------
    components:
        Topic-term weight matrix of shape ``(n_topics, n_features)`` such as
        ``NMF.components_`` or ``TruncatedSVD.components_``.
    feature_names:
        Vocabulary aligned with the columns of ``components``.
    n_terms:
        Number of top terms reported per topic.

    Returns
    -------
    pandas.DataFrame
        Columns ``topic``, ``rank``, ``term`` and ``weight``.
    """
    weights = np.asarray(components, dtype=float)
    vocabulary = np.asarray(feature_names)
    if weights.ndim != 2:
        raise ValueError("components must be a 2D array.")
    if weights.shape[1] != vocabulary.shape[0]:
        raise ValueError("feature_names must align with the columns of components.")
    if n_terms <= 0:
        raise ValueError("n_terms must be positive.")

    rows: list[dict[str, object]] = []
    for topic_index, topic_weights in enumerate(weights):
        order = np.argsort(topic_weights)[::-1][:n_terms]
        for rank, column in enumerate(order):
            rows.append(
                {
                    "topic": f"topic_{topic_index}",
                    "rank": rank,
                    "term": str(vocabulary[column]),
                    "weight": float(topic_weights[column]),
                }
            )
    return pd.DataFrame(rows)


def label_topics(
    components: NDArray[np.float64],
    feature_names: ArrayLike,
    n_terms: int = 3,
) -> dict[str, str]:
    """Generate a short human-readable label per topic from its top terms.

    Parameters
    ----------
    components:
        Topic-term weight matrix.
    feature_names:
        Vocabulary aligned with the columns of ``components``.
    n_terms:
        Number of top terms combined into the label.

    Returns
    -------
    dict
        Mapping ``topic_i -> "term1 / term2 / term3"``.
    """
    table = topic_term_table(components, feature_names, n_terms=n_terms)
    labels: dict[str, str] = {}
    for topic, group in table.groupby("topic", sort=False):
        labels[str(topic)] = " / ".join(group.sort_values("rank")["term"])
    return labels


def assign_documents(document_topics: NDArray[np.float64]) -> pd.DataFrame:
    """Assign each document to its dominant topic with a confidence share.

    Parameters
    ----------
    document_topics:
        Document-topic matrix of shape ``(n_docs, n_topics)`` such as the output
        of ``NMF.fit_transform``.

    Returns
    -------
    pandas.DataFrame
        Columns ``dominant_topic`` (int) and ``confidence`` (the topic's share
        of the row mass, in ``[0, 1]``).
    """
    matrix = np.asarray(document_topics, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("document_topics must be a 2D array.")
    if matrix.shape[1] < 1:
        raise ValueError("document_topics must have at least one topic column.")

    dominant = matrix.argmax(axis=1)
    row_sums = matrix.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        confidence = np.where(row_sums > 0, matrix.max(axis=1) / row_sums, 0.0)
    return pd.DataFrame(
        {
            "dominant_topic": dominant.astype(int),
            "confidence": confidence.astype(float),
        }
    )


def umass_topic_coherence(
    top_terms: list[list[str]],
    document_term: csr_matrix,
    vocabulary: dict[str, int],
    epsilon: float = 1.0,
) -> list[float]:
    """Approximate per-topic UMass coherence from a binary document-term matrix.

    For each ordered pair of top terms ``(w_i, w_j)`` with ``i < j`` the score
    accumulates ``log((D(w_i, w_j) + epsilon) / D(w_j))`` where ``D`` counts
    documents. Higher (less negative) values indicate that a topic's top terms
    genuinely co-occur, i.e. a more coherent topic.

    Parameters
    ----------
    top_terms:
        Top terms per topic, most important first.
    document_term:
        Binary (or count) document-term matrix; non-zero entries are treated as
        presence.
    vocabulary:
        Mapping from term to column index in ``document_term``.
    epsilon:
        Smoothing constant.

    Returns
    -------
    list of float
        One coherence score per topic (``nan`` if a topic has fewer than two
        in-vocabulary terms).
    """
    if not issparse(document_term):
        raise TypeError("document_term must be a scipy sparse matrix.")
    presence = (document_term > 0).astype(int).tocsc()
    document_frequency = np.asarray(presence.sum(axis=0)).ravel()

    scores: list[float] = []
    for terms in top_terms:
        indices = [vocabulary[term] for term in terms if term in vocabulary]
        total = 0.0
        pairs = 0
        for j in range(1, len(indices)):
            column_j = presence[:, indices[j]]
            for i in range(j):
                co_document = int(column_j.multiply(presence[:, indices[i]]).sum())
                denominator = document_frequency[indices[j]] or epsilon
                total += float(np.log((co_document + epsilon) / denominator))
                pairs += 1
        scores.append(total / pairs if pairs else float("nan"))
    return scores


def topic_diversity(top_terms: list[list[str]]) -> float:
    """Fraction of unique terms across all topics' top-term lists.

    A value near ``1.0`` means topics describe distinct vocabulary; a low value
    means topics overlap and repeat the same words.

    Parameters
    ----------
    top_terms:
        Top terms per topic.

    Returns
    -------
    float
        Unique terms divided by total terms.
    """
    flattened = [term for terms in top_terms for term in terms]
    if not flattened:
        raise ValueError("top_terms must contain at least one term.")
    return len(set(flattened)) / len(flattened)
