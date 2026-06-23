from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from sklearn.metrics import adjusted_rand_score  # noqa: E402

from unsup_lab.deep import (  # noqa: E402
    DeepClusteringResult,
    autoencoder_embedding,
    contrastive_cluster,
    contrastive_embedding,
    deep_cluster,
    deep_embedded_clustering,
    train_autoencoder,
    train_contrastive,
)


def _blobs(random_state: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    centres = rng.normal(scale=5.0, size=(3, 8))
    rows, labels = [], []
    for label, centre in enumerate(centres):
        rows.append(centre + rng.normal(scale=0.4, size=(40, 8)))
        labels.extend([label] * 40)
    return np.vstack(rows), np.array(labels)


def test_training_reduces_reconstruction_loss() -> None:
    data, _ = _blobs()

    _, losses = train_autoencoder(data, latent_dim=2, epochs=120, random_state=0)

    assert len(losses) == 120
    assert losses[-1] < losses[0]


def test_embedding_shape() -> None:
    data, _ = _blobs()
    model, _ = train_autoencoder(data, latent_dim=3, epochs=20, random_state=0)

    embedding = autoencoder_embedding(model, data)

    assert embedding.shape == (data.shape[0], 3)


def test_deep_cluster_recovers_blobs() -> None:
    data, truth = _blobs(random_state=1)

    result = deep_cluster(data, n_clusters=3, latent_dim=2, epochs=150, random_state=0)

    assert isinstance(result, DeepClusteringResult)
    assert result.labels.shape == (data.shape[0],)
    assert adjusted_rand_score(truth, result.labels) > 0.8


def test_deep_cluster_is_deterministic() -> None:
    data, _ = _blobs(random_state=2)

    a = deep_cluster(data, n_clusters=3, epochs=60, random_state=7)
    b = deep_cluster(data, n_clusters=3, epochs=60, random_state=7)

    np.testing.assert_array_equal(a.labels, b.labels)
    np.testing.assert_allclose(a.embedding, b.embedding, rtol=1e-5, atol=1e-5)


def test_invalid_latent_dim_raises() -> None:
    data, _ = _blobs()

    with pytest.raises(ValueError):
        train_autoencoder(data, latent_dim=8)  # equals n_features


def test_invalid_n_clusters_raises() -> None:
    data, _ = _blobs()

    with pytest.raises(ValueError):
        deep_cluster(data, n_clusters=1)


def test_dec_recovers_blobs() -> None:
    data, truth = _blobs(random_state=3)

    result = deep_embedded_clustering(
        data, n_clusters=3, latent_dim=2, pretrain_epochs=120, cluster_epochs=40, random_state=0
    )

    assert isinstance(result, DeepClusteringResult)
    assert result.labels.shape == (data.shape[0],)
    assert adjusted_rand_score(truth, result.labels) > 0.8


def test_dec_is_deterministic() -> None:
    data, _ = _blobs(random_state=4)

    kwargs = {"n_clusters": 3, "latent_dim": 2, "pretrain_epochs": 40, "cluster_epochs": 20}
    a = deep_embedded_clustering(data, random_state=5, **kwargs)
    b = deep_embedded_clustering(data, random_state=5, **kwargs)

    np.testing.assert_array_equal(a.labels, b.labels)


def test_dec_invalid_n_clusters_raises() -> None:
    data, _ = _blobs()

    with pytest.raises(ValueError):
        deep_embedded_clustering(data, n_clusters=1)


def test_contrastive_training_reduces_loss() -> None:
    data, _ = _blobs()

    _, losses = train_contrastive(data, latent_dim=4, epochs=120, random_state=0)

    assert len(losses) == 120
    assert losses[-1] < losses[0]


def test_contrastive_embedding_shape() -> None:
    data, _ = _blobs()
    model, _ = train_contrastive(data, latent_dim=4, epochs=20, random_state=0)

    embedding = contrastive_embedding(model, data)

    assert embedding.shape == (data.shape[0], 4)


def test_contrastive_cluster_recovers_blobs() -> None:
    data, truth = _blobs(random_state=5)

    result = contrastive_cluster(data, n_clusters=3, latent_dim=4, epochs=200, random_state=0)

    assert isinstance(result, DeepClusteringResult)
    assert adjusted_rand_score(truth, result.labels) > 0.8


def test_contrastive_is_deterministic() -> None:
    data, _ = _blobs(random_state=6)

    kwargs = {"n_clusters": 3, "latent_dim": 4, "epochs": 40}
    a = contrastive_cluster(data, random_state=9, **kwargs)
    b = contrastive_cluster(data, random_state=9, **kwargs)

    np.testing.assert_array_equal(a.labels, b.labels)


def test_contrastive_invalid_params_raise() -> None:
    data, _ = _blobs()

    with pytest.raises(ValueError):
        train_contrastive(data, mask_prob=1.0)
    with pytest.raises(ValueError):
        train_contrastive(data, temperature=0.0)
