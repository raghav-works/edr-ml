"""Tests for scripts/derive_static_thresholds.py's pure sweep logic. No
model, no parquet, no big arrays -- the sweep is factored into
sweep_thresholds() specifically so it's testable this way."""
import numpy as np

from scripts.derive_static_thresholds import (
    clopper_pearson_ci,
    sweep_thresholds,
    three_way_confusion,
)


def _hand_made_example():
    """20 benign (label 0) + 20 malicious (label 1), scores hand-chosen so
    the achieved FPR/threshold at each target is easy to verify by eye.
    Benign scores: 18 at 0.10, 2 at 0.96 (the only benign rows that could
    ever be false positives at a high threshold).
    Malicious scores: 20 rows evenly spread 0.05..1.00 in steps of 0.05.
    """
    y = np.array([0] * 20 + [1] * 20, dtype=np.int32)
    benign_scores = np.array([0.10] * 18 + [0.96, 0.96], dtype=np.float64)
    malicious_scores = np.linspace(0.05, 1.00, 20)
    proba = np.concatenate([benign_scores, malicious_scores])
    return y, proba


def test_thresholds_achieve_actual_fpr_at_or_below_target():
    y, proba = _hand_made_example()
    rows = sweep_thresholds(y, proba, [0.5, 0.1, 0.05])
    for r in rows:
        assert r["actual_fpr"] <= r["target_fpr"] + 1e-12, r


def test_thresholds_non_increasing_as_target_fpr_grows():
    y, proba = _hand_made_example()
    # Ascending target order -- threshold should be non-increasing.
    rows = sweep_thresholds(y, proba, [0.05, 0.1, 0.5, 1.0])
    thresholds = [r["threshold"] for r in rows]
    assert all(thresholds[i] >= thresholds[i + 1] for i in range(len(thresholds) - 1)), thresholds


def test_tie_count_and_detection_rate_on_hand_made_example():
    y, proba = _hand_made_example()
    # target_fpr=0.1 on 20 benign rows allows at most 2 FP (10%) -- the 2
    # rows at 0.96. find_threshold_for_fpr() picks the LOWEST threshold
    # that still keeps FPR at or below target (maximizing detection at
    # that FPR, not just the first threshold that reaches it) -- between
    # 0.96 and 0.15 no further benign rows are crossed (the next benign
    # cluster is at 0.10), so FPR stays flat at 0.1 while more malicious
    # rows get captured going lower; the chosen threshold is therefore
    # 0.15, the lowest score before the next benign row (0.10) would push
    # FPR above target. Verified directly (not hand-derived) before being
    # hardcoded here.
    rows = sweep_thresholds(y, proba, [0.1])
    r = rows[0]
    assert r["n_fp"] == 2
    assert r["threshold"] == 0.15
    assert r["n_tied"] == 1  # exactly one malicious row (0.15 itself) ties the threshold
    # Malicious scores >= 0.15 out of the 0.05..1.00/step-0.05 sequence:
    # 0.15, 0.20, ..., 1.00 = 18 of the 20 rows.
    assert r["n_tp"] == 18
    assert r["detection_rate"] == 18 / 20


def test_clopper_pearson_ci_matches_known_values():
    # 0 successes out of 10 -- lower bound must be exactly 0.
    lo, hi = clopper_pearson_ci(0, 10)
    assert lo == 0.0
    assert 0.0 < hi < 1.0
    # All successes -- upper bound must be exactly 1.
    lo, hi = clopper_pearson_ci(10, 10)
    assert hi == 1.0
    assert 0.0 < lo < 1.0
    # n=0 -- undefined, both NaN.
    lo, hi = clopper_pearson_ci(0, 0)
    assert lo != lo and hi != hi  # NaN != NaN
    # A known textbook case: 5/20 successes, 95% CP interval is
    # approximately [0.0866, 0.4910] (standard reference value).
    lo, hi = clopper_pearson_ci(5, 20)
    assert abs(lo - 0.0866) < 0.001
    assert abs(hi - 0.4910) < 0.001


def test_three_way_confusion_on_hand_made_example():
    y, proba = _hand_made_example()
    # allow_max below the 0.10 benign cluster -> those 18 rows are ALERT, not ALLOW.
    tw = three_way_confusion(y, proba, allow_max=0.05, block_min=0.97)
    assert tw["benign"]["ALLOW"] == 0
    assert tw["benign"]["ALERT"] == 20  # none reach block_min=0.97 (max benign score is 0.96)
    assert tw["benign"]["BLOCK"] == 0
    assert tw["benign"]["n"] == 20
    assert tw["malicious"]["n"] == 20
    assert (tw["malicious"]["ALLOW"] + tw["malicious"]["ALERT"] + tw["malicious"]["BLOCK"]) == 20
