"""
Unit tests for the deployment-prevalence projection math in
scripts/evaluate_all_models.py (review item 4).

Pure arithmetic -- no model artifacts, no parquet, NOT marked slow. A wrong
formula here produces confidently-wrong numbers in the report, which is worse
than not having them, so these assertions are hard, not smoke.
"""
from __future__ import annotations

import argparse

import pytest

from scripts.evaluate_all_models import (
    BinaryEval,
    PrevalenceConfig,
    _parse_float_list,
    _parse_int_list,
    alert_rate_at_prevalence,
    expected_fp,
    expected_tp,
    ppv_at_prevalence,
    project_to_prevalence,
    threshold_for_target_ppv,
)


# ------------------------------------------------------------- PPV (Bayes)
def test_ppv_round_trips_measured_precision_at_the_test_prevalence():
    """Feeding the measured FPR/TPR and the TEST split's own prevalence back
    into the Bayes formula must reproduce the measured precision. Anchor case:
    Cortex-Memory test (EVAL_ALL_MODELS_RESULTS.txt) -- FPR=0.0109, TPR=1.0,
    3000 malicious / 5930 total -> reported precision 0.9894."""
    fpr, tpr = 32 / 2930, 1.0            # TN=2898 FP=32 ; TP=3000 FN=0
    pi_test = 3000 / 5930
    assert ppv_at_prevalence(fpr, tpr, pi_test) == pytest.approx(0.9894, abs=5e-4)


def test_ppv_hand_computed_low_prevalence():
    # 1e-4 / (1e-4 + 0.01 * (1 - 1e-4))
    assert ppv_at_prevalence(0.01, 1.0, 1e-4) == pytest.approx(1e-4 / (1e-4 + 0.01 * 0.9999))
    assert ppv_at_prevalence(0.01, 1.0, 1e-4) == pytest.approx(0.00990198, abs=1e-8)


def test_ppv_is_monotone_decreasing_as_malware_gets_rarer():
    fpr, tpr = 0.01, 0.95
    vals = [ppv_at_prevalence(fpr, tpr, pi) for pi in (1e-2, 1e-3, 1e-4, 1e-5)]
    assert all(a > b for a, b in zip(vals, vals[1:]))


def test_ppv_edges():
    assert ppv_at_prevalence(0.01, 1.0, 0.0) == 0.0        # no malware -> no true positives
    assert ppv_at_prevalence(0.01, 1.0, 1.0) == 1.0        # all malware -> every flag is right
    assert ppv_at_prevalence(0.0, 1.0, 1e-4) == 1.0        # zero FPR -> perfect precision
    assert ppv_at_prevalence(0.0, 0.0, 1e-4) == 0.0        # nothing flagged -> defined as 0, no div0


# ---------------------------------------------------- alert rate / FP / TP
def test_alert_rate_identity_and_limit():
    fpr, tpr, pi = 0.01, 0.95, 1e-4
    assert alert_rate_at_prevalence(fpr, tpr, pi) == pytest.approx(tpr * pi + fpr * (1 - pi))
    # as malware -> 0 the population flag rate -> FPR
    assert alert_rate_at_prevalence(fpr, tpr, 1e-9) == pytest.approx(fpr, abs=1e-6)


def test_expected_fp_scales_linearly_with_population():
    fpr, pi = 0.0109, 1e-4
    assert expected_fp(fpr, pi, 200_000) == pytest.approx(2 * expected_fp(fpr, pi, 100_000))
    assert expected_fp(fpr, pi, 100_000) == pytest.approx(100_000 * (1 - pi) * fpr)
    # ~1090 false positives per 100k files for the memory signal
    assert expected_fp(fpr, pi, 100_000) == pytest.approx(1090, abs=1.0)


def test_expected_tp_matches_prevalence_times_recall():
    tpr, pi = 0.9752, 1e-4
    assert expected_tp(tpr, pi, 100_000) == pytest.approx(100_000 * pi * tpr)
    assert expected_tp(tpr, pi, 100_000) == pytest.approx(9.752, abs=1e-3)


# ------------------------------------------------ project_to_prevalence()
def _ev(fp: int, tn: int, tp: int, fn: int) -> BinaryEval:
    nb, nm = fp + tn, tp + fn
    return BinaryEval(
        threshold=0.5, n=nb + nm, n_benign=nb, n_malicious=nm,
        tn=tn, fp=fp, fn=fn, tp=tp,
        precision=tp / (tp + fp) if tp + fp else 0.0,
        recall=tp / (tp + fn) if tp + fn else 0.0,
        f1=0.0,
        fpr=fp / (fp + tn) if fp + tn else 0.0,
        auc_roc=float("nan"),
    )


def test_projection_rows_follow_config_and_stay_in_range():
    ev = _ev(fp=32, tn=2898, tp=3000, fn=0)
    cfg = PrevalenceConfig(prevalences=(1e-3, 1e-4, 1e-5), fp_per=(10_000, 100_000))
    proj = project_to_prevalence(ev, cfg)

    assert [r.prevalence for r in proj.rows] == [1e-3, 1e-4, 1e-5]
    assert proj.fpr == pytest.approx(ev.fpr)
    assert proj.tpr == pytest.approx(ev.recall)
    assert proj.fpr_ci95_upper is None           # 32 FP observed, not zero
    for r in proj.rows:
        assert 0.0 <= r.ppv <= 1.0
        assert 0.0 <= r.alert_rate <= 1.0
        assert set(r.fp_per) == {10_000, 100_000}
        assert all(v >= 0.0 for v in r.fp_per.values())
        assert all(v >= 0.0 for v in r.tp_per.values())


def test_projection_flags_zero_fp_with_rule_of_three_upper_bound():
    """0 observed false positives -> fpr_ci95_upper = 3 / n_benign, so a
    clean-looking PPV can't hide a thin benign set."""
    ev = _ev(fp=0, tn=126, tp=765, fn=11)     # Cortex-Behavioral test deployment total
    proj = project_to_prevalence(ev, PrevalenceConfig())
    assert proj.fpr == 0.0
    assert proj.fpr_ci95_upper == pytest.approx(3.0 / 126)
    assert all(r.ppv == 1.0 for r in proj.rows)   # the number itself is 1.0 -- the caveat carries the doubt


# --------------------------------------------- threshold_for_target_ppv()
def test_target_ppv_finds_a_higher_recall_threshold_and_respects_the_bar():
    # separable-ish scores: benign near 0, malicious near 1, a few overlaps
    import numpy as np
    rng = np.random.default_rng(0)
    y = np.array([0] * 5000 + [1] * 5000)
    proba = np.concatenate([
        rng.uniform(0.0, 0.6, 5000),          # benign
        rng.uniform(0.4, 1.0, 5000),          # malicious
    ])
    hit = threshold_for_target_ppv(y, proba, target_ppv=0.5, prevalence=1e-3)
    assert hit is not None
    _th, fpr, recall = hit
    # the reported operating point must actually meet the PPV bar
    assert ppv_at_prevalence(fpr, recall, 1e-3) >= 0.5
    assert 0.0 <= recall <= 1.0


def test_target_ppv_none_when_no_roc_point_clears_the_bar():
    """Worse-than-chance classifier (AUC ~0.35): malicious score LOWER on
    average, so the single highest score is benign and there is no
    (FPR=0, TPR>0) point. The search walks a rich ROC -- thousands of real
    operating points -- and none reaches PPV>=0.5, so it returns None. Guards
    that None comes from a genuine empty candidate set, not a coarse ROC or
    the both-zero edge."""
    import numpy as np
    from sklearn.metrics import roc_curve

    rng = np.random.default_rng(11)
    benign = np.clip(rng.normal(0.55, 0.20, 6000), 0, 1)
    malicious = np.clip(rng.normal(0.45, 0.20, 6000), 0, 1)   # lower on average
    y = np.array([0] * 6000 + [1] * 6000)
    proba = np.concatenate([benign, malicious])

    fpr_a, tpr_a, _ = roc_curve(y, proba)
    searched = [(f, t) for f, t in zip(fpr_a, tpr_a) if t > 0.0]
    assert len(searched) >= 500, "ROC too coarse -- test would pass for a trivial reason"
    assert max(ppv_at_prevalence(f, t, 1e-3) for f, t in searched) < 0.5

    assert threshold_for_target_ppv(y, proba, target_ppv=0.5, prevalence=1e-3) is None


def test_target_ppv_returns_a_brutal_low_recall_point_when_only_degenerately_reachable():
    """A decent classifier (AUC ~0.92) at a very rare prevalence: PPV>=0.5 is
    only reachable at the near-top of the ROC (FPR~0, tiny recall). The
    function returns THAT point -- a true 'you would sacrifice almost all
    recall' answer, which the report renders with a large negative delta --
    not None. The returned point must still genuinely meet the PPV bar."""
    import numpy as np

    rng = np.random.default_rng(7)
    benign = np.clip(rng.normal(0.35, 0.15, 5000), 0, 1)
    malicious = np.clip(rng.normal(0.65, 0.15, 5000), 0, 1)
    y = np.array([0] * 5000 + [1] * 5000)
    proba = np.concatenate([benign, malicious])

    hit = threshold_for_target_ppv(y, proba, target_ppv=0.5, prevalence=1e-5)
    assert hit is not None
    _th, fpr, recall = hit
    assert ppv_at_prevalence(fpr, recall, 1e-5) >= 0.5      # bar genuinely met
    assert recall < 0.1                                      # ... only by gutting recall


def test_target_ppv_none_for_single_class_input():
    import numpy as np
    y = np.zeros(100, dtype=int)
    proba = np.random.default_rng(2).uniform(0, 1, 100)
    assert threshold_for_target_ppv(y, proba, target_ppv=0.5, prevalence=1e-3) is None


# ------------------------------------------------------- CLI arg parsing
def test_parse_float_list_accepts_scientific_and_decimal():
    ap = argparse.ArgumentParser()
    assert _parse_float_list("1e-3,1e-4,0.00001", "prevalence", ap, lo=0.0, hi=1.0) == (1e-3, 1e-4, 1e-5)


def test_parse_float_list_rejects_out_of_range_and_garbage():
    ap = argparse.ArgumentParser()
    with pytest.raises(SystemExit):
        _parse_float_list("1.5", "prevalence", ap, lo=0.0, hi=1.0)     # > 1
    with pytest.raises(SystemExit):
        _parse_float_list("0", "prevalence", ap, lo=0.0, hi=1.0)       # not > 0
    with pytest.raises(SystemExit):
        _parse_float_list("abc", "prevalence", ap, lo=0.0, hi=1.0)


def test_parse_int_list_ok_and_rejects_nonpositive():
    ap = argparse.ArgumentParser()
    assert _parse_int_list("10000,100000", "fp-per", ap) == (10_000, 100_000)
    with pytest.raises(SystemExit):
        _parse_int_list("0,100", "fp-per", ap)
    with pytest.raises(SystemExit):
        _parse_int_list("-5", "fp-per", ap)
