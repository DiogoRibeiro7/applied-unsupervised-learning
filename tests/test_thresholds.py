from __future__ import annotations

import numpy as np
import pytest

from unsup_lab.thresholds import (
    ThresholdChoice,
    compare_thresholds,
    evaluate_threshold,
    knee_threshold,
    mad_threshold,
    otsu_threshold,
    quantile_threshold,
)


def _bimodal_scores() -> tuple[np.ndarray, np.ndarray]:
    """950 normal scores near 0 and 50 clear anomalies near 8."""
    rng = np.random.default_rng(0)
    normal = rng.normal(0.0, 1.0, size=950)
    anomalous = rng.normal(8.0, 1.0, size=50)
    scores = np.concatenate([normal, anomalous])
    truth = np.concatenate([np.zeros(950, dtype=bool), np.ones(50, dtype=bool)])
    return scores, truth


def test_quantile_flags_the_requested_fraction() -> None:
    scores, _ = _bimodal_scores()

    choice = quantile_threshold(scores, contamination=0.05)

    assert isinstance(choice, ThresholdChoice)
    assert choice.strategy == "quantile"
    assert choice.n_flagged == pytest.approx(50, abs=2)
    assert choice.flagged_fraction == pytest.approx(0.05, abs=0.005)


def test_every_strategy_lands_between_the_two_populations() -> None:
    # The strategies disagree on exactly where to cut, but a cut that fails to
    # sit between the populations has not found the structure at all.
    scores, truth = _bimodal_scores()

    for choice in (
        quantile_threshold(scores),
        mad_threshold(scores),
        knee_threshold(scores),
        otsu_threshold(scores),
    ):
        assert scores[~truth].mean() < choice.threshold < scores[truth].mean(), choice.strategy
        assert 0 < choice.n_flagged < scores.size


def test_mad_resists_the_anomalies_inflating_the_spread() -> None:
    # The mean/std cut is dragged upward by the very points it should find;
    # the robust cut should sit lower and so flag at least as many.
    scores, _ = _bimodal_scores()

    robust = mad_threshold(scores, n_deviations=3.0)
    naive = float(scores.mean() + 3.0 * scores.std())

    assert robust.threshold < naive
    assert robust.n_flagged >= int(np.sum(scores >= naive))


def test_knee_needs_no_parameters_and_finds_the_bend() -> None:
    scores, truth = _bimodal_scores()

    choice = knee_threshold(scores)

    assert choice.strategy == "knee"
    # The bend should recover most of the injected anomalies.
    assert evaluate_threshold(scores, choice.threshold, truth)["recall"] > 0.8


def test_compare_thresholds_tabulates_every_strategy() -> None:
    scores, _ = _bimodal_scores()

    table = compare_thresholds(scores)

    assert list(table.columns) == ["strategy", "threshold", "n_flagged", "flagged_fraction"]
    assert set(table["strategy"]) == {"quantile", "mad", "knee", "otsu"}
    assert (table["flagged_fraction"] > 0).all()


def test_evaluate_threshold_matches_a_hand_computed_case() -> None:
    scores = np.array([0.0, 1.0, 2.0, 3.0])
    truth = np.array([False, False, True, True])

    # Cutting at 2.0 flags the two highest scores, both of them anomalies.
    perfect = evaluate_threshold(scores, 2.0, truth)
    assert perfect == pytest.approx({"precision": 1.0, "recall": 1.0, "f1": 1.0, "n_flagged": 2.0})

    # Cutting at 1.0 also flags one normal point.
    loose = evaluate_threshold(scores, 1.0, truth)
    assert loose["precision"] == pytest.approx(2 / 3)
    assert loose["recall"] == pytest.approx(1.0)


def test_evaluate_threshold_reports_undefined_rather_than_zero() -> None:
    scores = np.array([0.0, 1.0, 2.0])
    truth = np.array([False, False, True])

    # Nothing flagged: precision is undefined, not zero.
    empty = evaluate_threshold(scores, 99.0, truth)
    assert np.isnan(empty["precision"])
    assert empty["recall"] == 0.0

    # Nothing to find: recall is undefined, not zero.
    nothing_to_find = evaluate_threshold(scores, 1.0, np.zeros(3, dtype=bool))
    assert np.isnan(nothing_to_find["recall"])


def test_constant_scores_raise_rather_than_inventing_a_cut() -> None:
    constant = np.full(20, 3.0)

    with pytest.raises(ValueError):
        mad_threshold(constant)
    with pytest.raises(ValueError):
        otsu_threshold(constant)


def test_invalid_inputs_raise() -> None:
    scores, truth = _bimodal_scores()

    with pytest.raises(ValueError):
        quantile_threshold(scores, contamination=0.0)
    with pytest.raises(ValueError):
        mad_threshold(scores, n_deviations=0)
    with pytest.raises(ValueError):
        otsu_threshold(scores, n_bins=1)
    with pytest.raises(ValueError):
        quantile_threshold([1.0])
    with pytest.raises(ValueError):
        knee_threshold([1.0, np.inf, 2.0])
    with pytest.raises(ValueError):
        evaluate_threshold(scores, 1.0, truth[:-1])
