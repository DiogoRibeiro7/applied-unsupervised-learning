"""Export key project figures to ``outputs/figures`` as PNG screenshots.

Run with::

    python scripts/export_figures.py

The figures are deterministic (fixed seeds) and are used in the project summary
documentation. Uses a non-interactive backend so it runs headless / in CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from unsup_lab.consensus import build_coassociation_matrix  # noqa: E402
from unsup_lab.data import make_customer_segmentation_data  # noqa: E402
from unsup_lab.preprocessing import scale_features  # noqa: E402
from unsup_lab.streaming import monitor_streaming_clusters  # noqa: E402

FIGURES = ROOT / "outputs" / "figures"


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def figure_segments() -> None:
    """PCA scatter of customers coloured by KMeans segment."""
    dataset = make_customer_segmentation_data(n_customers=1_500, random_state=42)
    scaled = scale_features(dataset.features).to_numpy()
    labels = KMeans(n_clusters=5, n_init=10, random_state=42).fit_predict(scaled)
    coords = PCA(n_components=2, random_state=42).fit_transform(scaled)

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=12, alpha=0.7)
    ax.set_title("Customer segments (KMeans, PCA projection)")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    fig.colorbar(scatter, label="segment")
    _save(fig, "customer_segments_pca.png")


def figure_consensus() -> None:
    """Reordered co-association matrix heatmap."""
    dataset = make_customer_segmentation_data(n_customers=400, random_state=7)
    scaled = scale_features(dataset.features).to_numpy()
    matrix = build_coassociation_matrix(
        scaled,
        lambda seed: KMeans(n_clusters=5, n_init=1, random_state=seed),
        n_runs=40,
        random_state=0,
    )
    order = matrix.sum(axis=1).argsort()

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix[order][:, order], cmap="viridis", vmin=0, vmax=1)
    ax.set_title("Consensus co-association matrix")
    ax.set_xlabel("customer")
    ax.set_ylabel("customer")
    fig.colorbar(image, label="co-clustering frequency")
    _save(fig, "consensus_coassociation.png")


def figure_drift() -> None:
    """PSI and centroid shift over a drifting stream."""
    rng = np.random.default_rng(42)
    stable = np.vstack(
        [rng.normal(loc=c, scale=0.5, size=(400, 2)) for c in ([0.0, 0.0], [6.0, 6.0], [0.0, 6.0])]
    )
    drifted = rng.normal(loc=[12.0, 0.0], scale=0.6, size=(800, 2))
    stream = np.vstack([stable, drifted])
    report = monitor_streaming_clusters(stream, n_clusters=4, batch_size=200, random_state=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(report["batch"], report["psi"], marker="o")
    ax1.axhline(0.25, color="red", linestyle="--", label="major-shift threshold")
    ax1.set_title("PSI per batch")
    ax1.set_xlabel("batch")
    ax1.set_ylabel("PSI")
    ax1.legend()
    ax2.plot(report["batch"], report["centroid_shift"], marker="o", color="purple")
    ax2.set_title("Centroid shift per batch")
    ax2.set_xlabel("batch")
    ax2.set_ylabel("mean L2 move")
    fig.tight_layout()
    _save(fig, "streaming_drift.png")


def main() -> None:
    """Generate every project summary figure."""
    figure_segments()
    figure_consensus()
    figure_drift()


if __name__ == "__main__":
    main()
