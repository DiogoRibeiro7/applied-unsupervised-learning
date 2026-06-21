from __future__ import annotations

import pandas as pd
import pytest

from unsup_lab.data import (
    make_customer_segmentation_data,
    make_sensor_anomaly_data,
    make_user_item_interactions,
)


def test_customer_data_shape() -> None:
    dataset = make_customer_segmentation_data(n_customers=100, random_state=7)

    assert isinstance(dataset.features, pd.DataFrame)
    assert dataset.features.shape[0] == 100
    assert len(dataset.hidden_labels) == 100


def test_sensor_data_contains_anomalies() -> None:
    dataset = make_sensor_anomaly_data(n_points=500, contamination=0.05, random_state=7)

    assert "timestamp" in dataset.features.columns
    assert (dataset.hidden_labels == "anomaly").sum() > 0


def test_user_item_interactions_shape() -> None:
    interactions, matrix = make_user_item_interactions(
        n_users=20,
        n_items=10,
        n_latent_groups=3,
        random_state=7,
    )

    assert matrix.shape == (20, 10)
    assert {"user_id", "item_id", "interaction_strength"}.issubset(interactions.columns)


def test_invalid_customer_count_raises() -> None:
    with pytest.raises(ValueError):
        make_customer_segmentation_data(n_customers=0)
