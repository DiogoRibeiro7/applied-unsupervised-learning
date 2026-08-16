from __future__ import annotations

import numpy as np
import pytest
from sklearn.feature_extraction.text import CountVectorizer

from unsup_lab.nlp import (
    assign_documents,
    clean_text,
    label_topics,
    topic_diversity,
    topic_term_table,
    umass_topic_coherence,
)


def test_clean_text_strips_symbols_and_digits() -> None:
    assert clean_text("Hello, World! 123 -- TEST") == "hello world test"


def test_clean_text_non_string_raises() -> None:
    with pytest.raises(TypeError):
        clean_text(123)  # type: ignore[arg-type]


def test_topic_term_table_schema_and_order() -> None:
    components = np.array([[0.1, 0.9, 0.4], [0.7, 0.2, 0.6]])
    features = ["alpha", "beta", "gamma"]

    table = topic_term_table(components, features, n_terms=2)

    assert list(table.columns) == ["topic", "rank", "term", "weight"]
    topic0 = table[table["topic"] == "topic_0"].sort_values("rank")
    assert list(topic0["term"]) == ["beta", "gamma"]


def test_topic_term_table_misaligned_features_raises() -> None:
    with pytest.raises(ValueError):
        topic_term_table(np.zeros((2, 3)), ["a", "b"], n_terms=2)


def test_label_topics_joins_top_terms() -> None:
    components = np.array([[0.1, 0.9, 0.4], [0.7, 0.2, 0.6]])
    features = ["alpha", "beta", "gamma"]

    labels = label_topics(components, features, n_terms=2)

    assert labels["topic_0"] == "beta / gamma"
    assert labels["topic_1"] == "alpha / gamma"


def test_assign_documents_confidence() -> None:
    document_topics = np.array([[0.8, 0.2], [0.25, 0.75], [0.0, 0.0]])

    assignment = assign_documents(document_topics)

    assert list(assignment["dominant_topic"]) == [0, 1, 0]
    assert assignment["confidence"].iloc[0] == pytest.approx(0.8)
    assert assignment["confidence"].iloc[2] == 0.0  # zero row -> zero confidence


def test_umass_coherence_prefers_cooccurring_terms() -> None:
    # "apple"/"fruit" always co-occur; "apple"/"engine" never do.
    docs = ["apple fruit", "apple fruit", "apple fruit", "engine motor", "engine motor"]
    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform(docs)
    vocab = vectorizer.vocabulary_

    coherent, incoherent = umass_topic_coherence(
        [["apple", "fruit"], ["apple", "engine"]], matrix, vocab
    )

    assert coherent > incoherent


def test_umass_coherence_conditions_on_the_higher_ranked_term() -> None:
    # "common" appears in 10 documents, "rare" in 2 of them, so the topic
    # ["common", "rare"] scores log((2 + 1) / 10): the denominator is the
    # document frequency of the *higher ranked* term. Dividing by the rarer
    # term instead would give a positive "coherence", which a log conditional
    # probability can never be.
    docs = ["common"] * 8 + ["common rare"] * 2
    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform(docs)

    (score,) = umass_topic_coherence([["common", "rare"]], matrix, vectorizer.vocabulary_)

    assert score == pytest.approx(float(np.log(3 / 10)))
    assert score < 0


def test_umass_coherence_requires_sparse() -> None:
    with pytest.raises(TypeError):
        umass_topic_coherence([["a", "b"]], np.zeros((2, 2)), {"a": 0, "b": 1})  # type: ignore[arg-type]


def test_topic_diversity_bounds() -> None:
    assert topic_diversity([["a", "b"], ["c", "d"]]) == 1.0
    assert topic_diversity([["a", "b"], ["a", "b"]]) == 0.5


def test_topic_diversity_empty_raises() -> None:
    with pytest.raises(ValueError):
        topic_diversity([[]])
