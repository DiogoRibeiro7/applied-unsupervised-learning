"""Synthetic data generators for unsupervised learning case studies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass(frozen=True)
class GeneratedDataset:
    """Container for generated data and hidden labels.

    Hidden labels are useful for offline evaluation and notebook validation,
    but they should not be used during unsupervised model fitting.
    """

    features: pd.DataFrame
    hidden_labels: pd.Series


def _validate_positive_integer(value: int, name: str) -> None:
    """Validate that a value is a positive integer."""
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def make_customer_segmentation_data(
    n_customers: int = 2_000,
    random_state: int = 42,
) -> GeneratedDataset:
    """Generate customer behaviour data with latent segments.

    Parameters
    ----------
    n_customers:
        Number of customers to generate.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    GeneratedDataset
        Feature table and hidden segment labels.
    """
    _validate_positive_integer(n_customers, "n_customers")
    rng = np.random.default_rng(random_state)

    segment_names = np.array(
        [
            "premium_loyalists",
            "discount_seekers",
            "new_low_engagement",
            "high_frequency_low_value",
            "at_risk_previous_buyers",
        ]
    )
    probabilities = np.array([0.18, 0.24, 0.20, 0.22, 0.16])
    segment_ids = rng.choice(len(segment_names), size=n_customers, p=probabilities)

    means = np.array(
        [
            [18, 24, 190, 0.12, 0.82, 0.74],
            [34, 14, 72, 0.72, 0.61, 0.58],
            [12, 2, 35, 0.28, 0.22, 0.16],
            [9, 36, 28, 0.34, 0.73, 0.66],
            [110, 7, 115, 0.21, 0.38, 0.29],
        ]
    )
    scales = np.array([8, 5, 24, 0.08, 0.08, 0.10])

    raw = np.vstack(
        [
            rng.normal(loc=means[segment_id], scale=scales)
            for segment_id in segment_ids
        ]
    )

    features = pd.DataFrame(
        raw,
        columns=[
            "recency_days",
            "purchase_frequency",
            "avg_order_value",
            "discount_ratio",
            "email_engagement",
            "product_diversity",
        ],
    )

    features["recency_days"] = features["recency_days"].clip(lower=1)
    features["purchase_frequency"] = features["purchase_frequency"].clip(lower=1)
    features["avg_order_value"] = features["avg_order_value"].clip(lower=5)
    for column in ["discount_ratio", "email_engagement", "product_diversity"]:
        features[column] = features[column].clip(lower=0, upper=1)

    labels = pd.Series(segment_names[segment_ids], name="hidden_segment")
    return GeneratedDataset(features=features, hidden_labels=labels)


def make_sensor_anomaly_data(
    n_points: int = 3_000,
    contamination: float = 0.04,
    random_state: int = 42,
) -> GeneratedDataset:
    """Generate sensor-like operational data with injected anomalies."""
    _validate_positive_integer(n_points, "n_points")
    if not 0 < contamination < 0.5:
        raise ValueError("contamination must be between 0 and 0.5.")

    rng = np.random.default_rng(random_state)
    timestamps = pd.date_range("2026-01-01", periods=n_points, freq="15min")
    hour = timestamps.hour.to_numpy()
    daily_cycle = np.sin(2 * np.pi * hour / 24)

    temperature = 21 + 2.5 * daily_cycle + rng.normal(0, 0.6, n_points)
    motion_count = rng.poisson(4 + 3 * np.maximum(daily_cycle, 0), n_points)
    power_usage = 0.8 + 0.25 * motion_count + rng.normal(0, 0.4, n_points)
    signal_strength = -65 + rng.normal(0, 4, n_points)
    missing_ratio = rng.beta(1.5, 18, n_points)

    features = pd.DataFrame(
        {
            "timestamp": timestamps,
            "temperature": temperature,
            "motion_count": motion_count.astype(float),
            "power_usage": power_usage,
            "signal_strength": signal_strength,
            "missing_ratio": missing_ratio,
        }
    )

    n_anomalies = max(1, int(n_points * contamination))
    anomaly_indices = rng.choice(n_points, size=n_anomalies, replace=False)

    features.loc[anomaly_indices, "temperature"] += rng.normal(8, 1.5, n_anomalies)
    features.loc[anomaly_indices, "power_usage"] += rng.normal(5, 1.0, n_anomalies)
    features.loc[anomaly_indices, "signal_strength"] -= rng.normal(18, 4, n_anomalies)
    features.loc[anomaly_indices, "missing_ratio"] = rng.uniform(0.45, 0.95, n_anomalies)

    labels = pd.Series("normal", index=features.index, name="hidden_label")
    labels.iloc[anomaly_indices] = "anomaly"
    return GeneratedDataset(features=features, hidden_labels=labels)


def make_document_corpus(random_state: int = 42) -> pd.DataFrame:
    """Create a small synthetic document corpus with overlapping topics."""
    rng = np.random.default_rng(random_state)

    topic_terms = {
        "healthcare": [
            "clinic", "patient", "diagnosis", "treatment", "hospital",
            "nurse", "appointment", "triage", "capacity", "care",
        ],
        "finance": [
            "portfolio", "risk", "returns", "volatility", "market",
            "asset", "drawdown", "liquidity", "forecast", "capital",
        ],
        "retail": [
            "customer", "basket", "discount", "purchase", "campaign",
            "loyalty", "product", "segment", "conversion", "churn",
        ],
        "iot": [
            "sensor", "gateway", "signal", "battery", "motion",
            "temperature", "device", "stream", "event", "anomaly",
        ],
    }

    rows: list[dict[str, str]] = []
    for topic, terms in topic_terms.items():
        for index in range(35):
            chosen = rng.choice(terms, size=18, replace=True)
            noise_topic = rng.choice([name for name in topic_terms if name != topic])
            noise = rng.choice(topic_terms[noise_topic], size=4, replace=True)
            words = np.concatenate([chosen, noise])
            rng.shuffle(words)
            rows.append(
                {
                    "document_id": f"{topic}_{index:03d}",
                    "text": " ".join(words),
                    "hidden_topic": topic,
                }
            )

    return pd.DataFrame(rows)


def make_user_item_interactions(
    n_users: int = 600,
    n_items: int = 120,
    n_latent_groups: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, NDArray[np.float64]]:
    """Generate implicit user-item interactions and the dense matrix.

    Returns
    -------
    tuple[pandas.DataFrame, numpy.ndarray]
        Long-form interaction table and dense user-item matrix.
    """
    _validate_positive_integer(n_users, "n_users")
    _validate_positive_integer(n_items, "n_items")
    _validate_positive_integer(n_latent_groups, "n_latent_groups")

    rng = np.random.default_rng(random_state)
    user_groups = rng.integers(0, n_latent_groups, size=n_users)
    item_groups = rng.integers(0, n_latent_groups, size=n_items)

    matrix = np.zeros((n_users, n_items), dtype=float)
    for user_id in range(n_users):
        affinity = np.where(item_groups == user_groups[user_id], 0.35, 0.04)
        interactions = rng.binomial(1, affinity)
        strength = rng.gamma(shape=2.0, scale=1.5, size=n_items)
        matrix[user_id] = interactions * strength

    rows = []
    user_idx, item_idx = np.nonzero(matrix)
    for user_id, item_id in zip(user_idx, item_idx, strict=True):
        rows.append(
            {
                "user_id": int(user_id),
                "item_id": int(item_id),
                "interaction_strength": float(matrix[user_id, item_id]),
            }
        )

    return pd.DataFrame(rows), matrix
