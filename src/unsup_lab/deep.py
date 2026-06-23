"""Deep clustering with an autoencoder representation.

When structure lives on a non-linear manifold tangled across many dimensions,
clustering the raw features struggles. An autoencoder learns a compact
representation by reconstructing its input through a bottleneck; clustering that
bottleneck embedding can recover structure the raw space hides.

This is the one place in the lab that uses a deep-learning dependency. PyTorch
is an *optional* extra (``poetry install --with deep``); the rest of the package
never imports this module, so the core install stays light. The autoencoder here
is deliberately small and trained full-batch for reproducibility - the point is
the representation-learning idea, not a production training loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.cluster import KMeans

try:  # PyTorch is an optional dependency.
    import torch
    from torch import nn
except ImportError as error:  # pragma: no cover - exercised only without torch
    raise ImportError(
        "unsup_lab.deep requires PyTorch. Install it with `poetry install --with deep`."
    ) from error


@dataclass
class DeepClusteringResult:
    """Outcome of autoencoder-based deep clustering.

    Attributes
    ----------
    labels:
        Cluster label per row, from KMeans on the learned embedding.
    embedding:
        The autoencoder bottleneck representation, shape ``(n_rows, latent_dim)``.
    losses:
        Reconstruction loss per training epoch.
    """

    labels: NDArray[np.int_]
    embedding: NDArray[np.float64]
    losses: list[float]


class _AutoEncoder(nn.Module):
    """A small symmetric multilayer-perceptron autoencoder."""

    def __init__(self, n_features: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_features),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the reconstruction and the latent code."""
        latent = self.encoder(x)
        return self.decoder(latent), latent


def _as_2d_array(data: ArrayLike) -> NDArray[np.float64]:
    """Convert input data to a validated 2D float array."""
    array = np.asarray(data, dtype=float)
    if array.ndim != 2:
        raise ValueError("data must be a 2D array.")
    if array.shape[0] < 2:
        raise ValueError("data must contain at least two rows.")
    return array


def train_autoencoder(
    data: ArrayLike,
    latent_dim: int = 2,
    hidden_dim: int = 32,
    epochs: int = 150,
    learning_rate: float = 1e-2,
    random_state: int = 0,
) -> tuple[_AutoEncoder, list[float]]:
    """Train a small autoencoder on the data (full-batch, deterministic).

    Parameters
    ----------
    data:
        Feature matrix; scale it beforehand for best results.
    latent_dim:
        Bottleneck dimensionality.
    hidden_dim:
        Hidden-layer width.
    epochs:
        Number of full-batch optimisation steps.
    learning_rate:
        Adam learning rate.
    random_state:
        Seed for reproducible initialisation and training.

    Returns
    -------
    tuple
        The trained model and the per-epoch reconstruction losses.
    """
    x = _as_2d_array(data)
    if not 1 <= latent_dim < x.shape[1]:
        raise ValueError("latent_dim must be between 1 and n_features - 1.")
    if hidden_dim < 1:
        raise ValueError("hidden_dim must be positive.")
    if epochs < 1:
        raise ValueError("epochs must be positive.")

    torch.manual_seed(random_state)
    tensor = torch.tensor(x, dtype=torch.float32)
    model = _AutoEncoder(x.shape[1], hidden_dim, latent_dim)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    losses: list[float] = []
    model.train()
    for _ in range(epochs):
        optimiser.zero_grad()
        reconstruction, _ = model(tensor)
        loss = loss_fn(reconstruction, tensor)
        loss.backward()
        optimiser.step()
        losses.append(float(loss.item()))
    return model, losses


def autoencoder_embedding(model: _AutoEncoder, data: ArrayLike) -> NDArray[np.float64]:
    """Return the autoencoder's latent embedding for the data."""
    x = _as_2d_array(data)
    model.eval()
    with torch.no_grad():
        _, latent = model(torch.tensor(x, dtype=torch.float32))
    return np.asarray(latent.numpy(), dtype=float)


def deep_cluster(
    data: ArrayLike,
    n_clusters: int,
    latent_dim: int = 2,
    hidden_dim: int = 32,
    epochs: int = 150,
    learning_rate: float = 1e-2,
    random_state: int = 0,
) -> DeepClusteringResult:
    """Cluster data in an autoencoder embedding space.

    The autoencoder is trained to reconstruct the input, then KMeans is run on
    the learned bottleneck representation.

    Parameters
    ----------
    data:
        Feature matrix (scale it beforehand for best results).
    n_clusters:
        Number of clusters.
    latent_dim, hidden_dim, epochs, learning_rate:
        Autoencoder hyper-parameters.
    random_state:
        Seed for the autoencoder and KMeans.

    Returns
    -------
    DeepClusteringResult
        Cluster labels, the learned embedding and the training losses.
    """
    x = _as_2d_array(data)
    if not 2 <= n_clusters <= x.shape[0]:
        raise ValueError("n_clusters must be between 2 and the number of rows.")

    model, losses = train_autoencoder(
        x,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        epochs=epochs,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    embedding = autoencoder_embedding(model, x)
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state).fit_predict(
        embedding
    )
    return DeepClusteringResult(
        labels=np.asarray(labels, dtype=int),
        embedding=embedding,
        losses=losses,
    )


def _student_t_assignment(
    embedding: torch.Tensor,
    centres: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Soft cluster assignment via a Student-t kernel (the DEC ``q``)."""
    squared = torch.cdist(embedding, centres) ** 2
    numerator = (1.0 + squared / alpha).pow(-(alpha + 1.0) / 2.0)
    return numerator / numerator.sum(dim=1, keepdim=True)


def _target_distribution(q: torch.Tensor) -> torch.Tensor:
    """Sharpened target distribution ``p`` that emphasises confident points."""
    weight = q**2 / q.sum(dim=0, keepdim=True)
    return weight / weight.sum(dim=1, keepdim=True)


def deep_embedded_clustering(
    data: ArrayLike,
    n_clusters: int,
    latent_dim: int = 10,
    hidden_dim: int = 64,
    pretrain_epochs: int = 150,
    cluster_epochs: int = 60,
    learning_rate: float = 1e-3,
    alpha: float = 1.0,
    random_state: int = 0,
) -> DeepClusteringResult:
    """Deep Embedded Clustering (DEC).

    First an autoencoder is pretrained for reconstruction, then KMeans on its
    embedding initialises cluster centres. From there the encoder *and* the
    centres are fine-tuned to minimise the KL divergence between a Student-t soft
    assignment and a sharpened target distribution. Unlike a plain autoencoder,
    this directly optimises for cluster separability.

    Parameters
    ----------
    data:
        Feature matrix (scale it beforehand).
    n_clusters:
        Number of clusters.
    latent_dim, hidden_dim:
        Autoencoder architecture.
    pretrain_epochs:
        Reconstruction pretraining epochs.
    cluster_epochs:
        DEC fine-tuning epochs.
    learning_rate:
        Adam learning rate for the fine-tuning phase.
    alpha:
        Degrees of freedom of the Student-t kernel.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    DeepClusteringResult
        Labels from the final soft assignment, the fine-tuned embedding and the
        per-epoch clustering (KL) losses.
    """
    x = _as_2d_array(data)
    if not 2 <= n_clusters <= x.shape[0]:
        raise ValueError("n_clusters must be between 2 and the number of rows.")
    if cluster_epochs < 1:
        raise ValueError("cluster_epochs must be positive.")

    model, _ = train_autoencoder(
        x,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        epochs=pretrain_epochs,
        random_state=random_state,
    )

    tensor = torch.tensor(x, dtype=torch.float32)
    initial = autoencoder_embedding(model, x)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state).fit(initial)
    centres = torch.nn.Parameter(torch.tensor(kmeans.cluster_centers_, dtype=torch.float32))

    optimiser = torch.optim.Adam([*model.encoder.parameters(), centres], lr=learning_rate)

    losses: list[float] = []
    model.train()
    for _ in range(cluster_epochs):
        optimiser.zero_grad()
        embedding = model.encoder(tensor)
        q = _student_t_assignment(embedding, centres, alpha=alpha)
        p = _target_distribution(q).detach()
        loss = torch.sum(p * torch.log((p + 1e-9) / (q + 1e-9)))
        loss.backward()
        optimiser.step()
        losses.append(float(loss.item()))

    model.eval()
    with torch.no_grad():
        embedding_final = model.encoder(tensor)
        q_final = _student_t_assignment(embedding_final, centres, alpha=alpha)
        labels = q_final.argmax(dim=1).numpy()

    return DeepClusteringResult(
        labels=np.asarray(labels, dtype=int),
        embedding=np.asarray(embedding_final.numpy(), dtype=float),
        losses=losses,
    )
