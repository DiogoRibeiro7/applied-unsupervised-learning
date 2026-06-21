from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from unsup_lab.preprocessing import scale_features


def _frame() -> pd.DataFrame:
    return pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 20.0, 30.0, 40.0]})


def test_standard_scaling_zero_mean() -> None:
    scaled = scale_features(_frame(), method="standard")

    assert list(scaled.columns) == ["a", "b"]
    np.testing.assert_allclose(scaled.mean().to_numpy(), [0.0, 0.0], atol=1e-9)


def test_robust_scaling_preserves_index() -> None:
    data = _frame()
    data.index = [10, 11, 12, 13]

    scaled = scale_features(data, method="robust")

    assert list(scaled.index) == [10, 11, 12, 13]


def test_invalid_method_raises() -> None:
    with pytest.raises(ValueError):
        scale_features(_frame(), method="minmax")


def test_non_numeric_raises() -> None:
    data = pd.DataFrame({"a": ["x", "y"], "b": [1, 2]})

    with pytest.raises(TypeError):
        scale_features(data)


def test_empty_frame_raises() -> None:
    with pytest.raises(ValueError):
        scale_features(pd.DataFrame())


def test_non_dataframe_raises() -> None:
    with pytest.raises(TypeError):
        scale_features([[1, 2], [3, 4]])  # type: ignore[arg-type]
