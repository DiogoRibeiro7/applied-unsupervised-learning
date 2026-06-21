"""Preprocessing helpers for notebooks."""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler


def scale_features(
    data: pd.DataFrame,
    method: str = "standard",
) -> pd.DataFrame:
    """Scale numeric features and return a DataFrame with the original columns.

    Parameters
    ----------
    data:
        Numeric feature table.
    method:
        Either `"standard"` or `"robust"`.

    Returns
    -------
    pandas.DataFrame
        Scaled feature table.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if data.empty:
        raise ValueError("data cannot be empty.")
    if not all(pd.api.types.is_numeric_dtype(data[column]) for column in data.columns):
        raise TypeError("all columns must be numeric.")

    if method == "standard":
        scaler = StandardScaler()
    elif method == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError("method must be either 'standard' or 'robust'.")

    scaled = scaler.fit_transform(data)
    return pd.DataFrame(scaled, columns=data.columns, index=data.index)
