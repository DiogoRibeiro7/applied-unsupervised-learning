"""Choosing an operating point on anomaly scores, without labels.

A detector produces a ranking; an operational system needs a *cut*. Deciding
where to cut is the step most anomaly-detection write-ups skip, and it is not a
modelling detail - it is the decision that sets how many alerts an analyst
receives tomorrow morning.

Each strategy here encodes a different assumption about what "anomalous" means,
and none of them looks at labels:

* :func:`quantile_threshold` assumes you already know the contamination rate.
* :func:`mad_threshold` assumes normal scores are unimodal, and cuts a robust
  number of deviations above the middle.
* :func:`knee_threshold` assumes the sorted score curve bends where the
  population changes, and finds that bend geometrically.
* :func:`otsu_threshold` assumes the scores are two populations, and picks the
  cut that separates them best.

:func:`compare_thresholds` runs them side by side, because the honest output is
not one number but a view of how much the alert volume depends on the
assumption. :func:`evaluate_threshold` is the only function that touches labels,
and it exists for *offline* checking after the fact - never for choosing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

_MAD_TO_SIGMA = 1.4826  # scales the median absolute deviation to a normal sigma


@dataclass(frozen=True)
class ThresholdChoice:
    """An operating point chosen by one strategy.

    Attributes
    ----------
    strategy:
        Name of the strategy that produced the cut.
    threshold:
        Scores greater than or equal to this value are flagged.
    n_flagged:
        How many points the cut flags.
    flagged_fraction:
        ``n_flagged`` as a fraction of all points - the alert volume.
    """

    strategy: str
    threshold: float
    n_flagged: int
    flagged_fraction: float


def _as_scores(scores: ArrayLike) -> NDArray[np.float64]:
    """Convert anomaly scores to a validated 1D float array."""
    values = np.asarray(scores, dtype=float).ravel()
    if values.size < 2:
        raise ValueError("scores must contain at least two values.")
    if not np.all(np.isfinite(values)):
        raise ValueError("scores must be finite; drop or impute non-finite values first.")
    return values


def _choice(strategy: str, threshold: float, values: NDArray[np.float64]) -> ThresholdChoice:
    """Package a threshold together with the alert volume it implies."""
    flagged = int(np.sum(values >= threshold))
    return ThresholdChoice(
        strategy=strategy,
        threshold=float(threshold),
        n_flagged=flagged,
        flagged_fraction=float(flagged / values.size),
    )


def quantile_threshold(scores: ArrayLike, contamination: float = 0.05) -> ThresholdChoice:
    """Cut at the top ``contamination`` fraction of scores.

    The simplest strategy, and the most honest about what it assumes: you are
    not discovering how many anomalies there are, you are declaring it. Useful
    when the alert budget is fixed by how many cases a team can review.

    Parameters
    ----------
    scores:
        Anomaly scores, higher meaning more anomalous.
    contamination:
        Fraction of points to flag, in ``(0, 1)``.

    Returns
    -------
    ThresholdChoice
        The chosen operating point.
    """
    values = _as_scores(scores)
    if not 0.0 < contamination < 1.0:
        raise ValueError("contamination must be between 0 and 1.")
    return _choice("quantile", float(np.quantile(values, 1.0 - contamination)), values)


def mad_threshold(scores: ArrayLike, n_deviations: float = 3.0) -> ThresholdChoice:
    """Cut a robust number of deviations above the median score.

    Uses the median and the median absolute deviation rather than the mean and
    standard deviation, because the anomalies themselves inflate both of the
    latter - the very points being looked for would raise the bar against
    finding them.

    Parameters
    ----------
    scores:
        Anomaly scores, higher meaning more anomalous.
    n_deviations:
        How many robust deviations above the median to cut, positive.

    Returns
    -------
    ThresholdChoice
        The chosen operating point.
    """
    values = _as_scores(scores)
    if n_deviations <= 0:
        raise ValueError("n_deviations must be positive.")

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad == 0.0:
        # More than half the scores are identical; fall back to the standard
        # deviation so the strategy degrades rather than dividing by nothing.
        spread = float(values.std())
        if spread == 0.0:
            raise ValueError("scores are constant; no threshold separates them.")
    else:
        spread = mad * _MAD_TO_SIGMA
    return _choice("mad", median + n_deviations * spread, values)


def knee_threshold(scores: ArrayLike) -> ThresholdChoice:
    """Cut where the sorted score curve bends upward.

    Sorting the scores ascending, a detector that has found something produces a
    long flat run of normal scores and a short steep tail. The bend between them
    is the point furthest below the straight line joining the curve's ends,
    which needs no parameter at all - its cost is that on scores with no real
    tail it still returns a bend.

    Parameters
    ----------
    scores:
        Anomaly scores, higher meaning more anomalous.

    Returns
    -------
    ThresholdChoice
        The chosen operating point.
    """
    values = _as_scores(scores)
    curve = np.sort(values)
    chord = np.linspace(curve[0], curve[-1], curve.size)
    return _choice("knee", float(curve[int(np.argmax(chord - curve))]), values)


def otsu_threshold(scores: ArrayLike, n_bins: int = 128) -> ThresholdChoice:
    """Cut where the scores split best into two populations.

    Otsu's method, borrowed from image binarisation: sweep every candidate cut
    and keep the one maximising the variance *between* the two groups, which is
    equivalent to minimising the variance within them. It assumes the scores are
    two populations rather than one continuum, and says nothing sensible when
    they are not.

    Parameters
    ----------
    scores:
        Anomaly scores, higher meaning more anomalous.
    n_bins:
        Histogram resolution for the sweep; at least 2.

    Returns
    -------
    ThresholdChoice
        The chosen operating point.
    """
    values = _as_scores(scores)
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")
    if values.min() == values.max():
        raise ValueError("scores are constant; no threshold separates them.")

    counts, edges = np.histogram(values, bins=n_bins)
    centres = (edges[:-1] + edges[1:]) / 2.0
    weights = counts / counts.sum()

    # Cumulative class probabilities and means for every candidate split.
    low_weight = np.cumsum(weights)
    low_mean = np.cumsum(weights * centres)
    total_mean = low_mean[-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        between = (total_mean * low_weight - low_mean) ** 2 / (low_weight * (1.0 - low_weight))
    between[~np.isfinite(between)] = -np.inf

    return _choice("otsu", float(edges[int(np.argmax(between)) + 1]), values)


def compare_thresholds(
    scores: ArrayLike,
    contamination: float = 0.05,
    n_deviations: float = 3.0,
) -> pd.DataFrame:
    """Run every strategy on the same scores and tabulate what each would flag.

    The point of the table is the spread: when four label-free rules disagree by
    an order of magnitude in alert volume, that disagreement is the finding, and
    the choice belongs to whoever owns the review capacity.

    Parameters
    ----------
    scores:
        Anomaly scores, higher meaning more anomalous.
    contamination:
        Passed to :func:`quantile_threshold`.
    n_deviations:
        Passed to :func:`mad_threshold`.

    Returns
    -------
    pandas.DataFrame
        One row per strategy with ``threshold``, ``n_flagged`` and
        ``flagged_fraction``.
    """
    values = _as_scores(scores)
    choices = [
        quantile_threshold(values, contamination=contamination),
        mad_threshold(values, n_deviations=n_deviations),
        knee_threshold(values),
        otsu_threshold(values),
    ]
    return pd.DataFrame(
        [
            {
                "strategy": choice.strategy,
                "threshold": choice.threshold,
                "n_flagged": choice.n_flagged,
                "flagged_fraction": choice.flagged_fraction,
            }
            for choice in choices
        ]
    )


def evaluate_threshold(
    scores: ArrayLike,
    threshold: float,
    is_anomaly: ArrayLike,
) -> dict[str, float]:
    """Score an operating point against known labels, *after* it was chosen.

    This is the only function here that sees labels, and it is for offline
    validation of a cut that was already made - using it to select a threshold
    would leak exactly the supervision the rest of the module avoids.

    Parameters
    ----------
    scores:
        Anomaly scores, higher meaning more anomalous.
    threshold:
        The operating point to assess; scores at or above it are flagged.
    is_anomaly:
        Boolean-like ground truth, aligned with ``scores``.

    Returns
    -------
    dict
        ``precision``, ``recall``, ``f1`` and ``n_flagged``. Precision is
        ``nan`` when nothing is flagged, recall ``nan`` when there is nothing to
        find; neither is reported as zero, because "undefined" and "wrong" are
        different answers.
    """
    values = _as_scores(scores)
    truth = np.asarray(is_anomaly).ravel().astype(bool)
    if truth.shape[0] != values.shape[0]:
        raise ValueError("scores and is_anomaly must have the same length.")

    flagged = values >= threshold
    true_positives = float(np.sum(flagged & truth))
    n_flagged = float(np.sum(flagged))
    n_actual = float(np.sum(truth))

    precision = true_positives / n_flagged if n_flagged else float("nan")
    recall = true_positives / n_actual if n_actual else float("nan")
    if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0.0:
        f1 = float("nan") if np.isnan(precision) or np.isnan(recall) else 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "n_flagged": float(n_flagged),
    }
