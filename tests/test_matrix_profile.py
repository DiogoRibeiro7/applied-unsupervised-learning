from __future__ import annotations

import numpy as np
import pytest

from unsup_lab.timeseries import matrix_profile, top_discords, top_motifs


def _repetitive_series() -> tuple[np.ndarray, int, int]:
    """A repeating sine background (so windows have neighbours) with one break.

    The matrix profile is designed for repetitive signals: normal windows match
    each other (low profile), and a region that breaks the repetition stands out
    as a discord (high profile).
    """
    rng = np.random.default_rng(0)
    window = 40
    n = 600
    t = np.arange(n)
    series = np.sin(2 * np.pi * t / window) + rng.normal(scale=0.05, size=n)
    discord = 300
    series[discord : discord + window] = rng.normal(scale=0.05, size=window)  # flat break
    return series, window, discord


def test_matrix_profile_shape() -> None:
    series, window, _ = _repetitive_series()

    profile, index = matrix_profile(series, window)

    assert profile.shape == (series.size - window + 1,)
    assert index.shape == profile.shape
    assert np.all(np.isfinite(profile))


def test_motif_is_more_similar_than_discord() -> None:
    series, window, _ = _repetitive_series()
    profile, _ = matrix_profile(series, window)

    motif = top_motifs(profile, window, k=1)[0]
    discord = top_discords(profile, window, k=1)[0]

    # A motif has a very close neighbour; a discord does not.
    assert profile[motif] < profile[discord]
    assert profile[motif] < np.median(profile)


def test_top_discord_finds_injected_anomaly() -> None:
    series, window, discord = _repetitive_series()
    profile, _ = matrix_profile(series, window)

    discords = top_discords(profile, window, k=1)

    assert abs(discords[0] - discord) <= window


def test_matrix_profile_invalid_window_raises() -> None:
    with pytest.raises(ValueError):
        matrix_profile(np.zeros(50), window=1)
    with pytest.raises(ValueError):
        matrix_profile(np.zeros(10), window=20)


def test_matrix_profile_invalid_exclusion_raises() -> None:
    with pytest.raises(ValueError):
        matrix_profile(np.zeros(100), window=10, exclusion_fraction=1.0)


def test_matrix_profile_series_too_short_for_exclusion_zone_raises() -> None:
    # 11 windows of length 10 with a 5-position exclusion zone leaves the middle
    # subsequences without a single valid neighbour, which used to return a
    # profile full of infinities instead of failing.
    with pytest.raises(ValueError, match="exclusion zone"):
        matrix_profile(np.arange(20.0), window=10)


def test_discords_respect_exclusion_gap() -> None:
    series, window, _ = _repetitive_series()
    profile, _ = matrix_profile(series, window)

    discords = top_discords(profile, window, k=3)

    for i in range(len(discords)):
        for j in range(i + 1, len(discords)):
            assert abs(discords[i] - discords[j]) >= window


def test_top_motifs_invalid_k_raises() -> None:
    profile, _ = matrix_profile(np.random.default_rng(0).normal(size=100), window=10)

    with pytest.raises(ValueError):
        top_motifs(profile, window=10, k=0)
