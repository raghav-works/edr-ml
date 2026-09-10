"""
Policy engine — sequential-gate decision logic.

Exactly reproduces the architecture: static and behavioral scores are never
averaged/ensembled. Static runs first; only a static ALLOW allows the file to
proceed to behavioral analysis. Final decision is rule-priority, not blended.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
# config/thresholds.yaml is the SINGLE SOURCE OF TRUTH for every decision
# threshold below. This module loads it at import time and fails loudly if
# the file is missing, malformed, or missing a key -- a silently-wrong
# threshold is exactly the failure class this project keeps hardening
# against. The long comment on each constant documents how that value was
# derived; the value itself lives only in the YAML, not here.
_THRESHOLDS_PATH = Path(__file__).resolve().parents[1] / "config" / "thresholds.yaml"


def _load_thresholds() -> dict:
    try:
        with open(_THRESHOLDS_PATH, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"decision thresholds not found: {_THRESHOLDS_PATH}. config/thresholds.yaml is "
            "the source of truth for policy_engine and must ship with the package."
        ) from exc
    except yaml.YAMLError as exc:
        raise RuntimeError(f"malformed threshold file {_THRESHOLDS_PATH}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"{_THRESHOLDS_PATH} did not parse to a mapping")
    return data


def _thr(data: dict, *path: str) -> float:
    node: object = data
    for key in path:
        if not isinstance(node, dict) or key not in node:
            raise RuntimeError(f"{_THRESHOLDS_PATH} is missing required key: {'.'.join(path)}")
        node = node[key]
    if isinstance(node, bool) or not isinstance(node, (int, float)):
        raise RuntimeError(f"{_THRESHOLDS_PATH}:{'.'.join(path)} must be a number, got {node!r}")
    return float(node)


_THRESHOLDS = _load_thresholds()

# Static thresholds re-derived from data/models/cortex_static (LightGBM,
# trained on the deduplicated EMBER2024 PE-format train split) via
# models.static_lgbm.find_threshold_for_fpr() against the calibrated
# probabilities on the deduplicated EMBER2024 test split (539,940 rows).
# STATIC_BLOCK_MIN hits the architecture doc's max_fpr@0.1% = 0.001 bar
# (target_fpr=0.001 -> actual FPR=0.000997, detection_rate=0.9168 at this
# threshold). STATIC_ALLOW_MAX uses target_fpr=0.01 (actual FPR=0.009999,
# detection_rate=0.9803).
#
# RE-DERIVED 2026-09-08 after the Platt-calibrator fix (calibrators now fit
# on raw booster margins, not sigmoid probabilities). The booster is
# unchanged, so AUC-ROC (0.9988) and the operating points above are
# unchanged -- only the calibrated score scale moved, so these raw values
# differ from the pre-fix ones (ALLOW was 0.6163460957, BLOCK 0.9950119117).
# The old values must NOT be used with the recalibrated model: at the old
# BLOCK value the recalibrated model blocks at only ~0.03% FPR / 86% det.
STATIC_ALLOW_MAX = _thr(_THRESHOLDS, "static", "allow_below")        # score <  this -> ALLOW
STATIC_BLOCK_MIN = _thr(_THRESHOLDS, "static", "block_at_or_above")  # score >= this -> BLOCK  (between the two -> ALERT)
# INTERIM CAP (2026-09-08): a score >= STATIC_BLOCK_MIN still yields
# StaticVerdict.BLOCK, but decide() demotes that to a final ALERT (not BLOCK)
# -- 2/5 confirmed-benign real binaries still cross this line post-
# calibration-fix. See decide()'s "INTERIM CAP" docstring section for the
# evidence and the removal criteria.

# Behavioral threshold re-derived against data/models/cortex_behavioral_best.pt
# (1D-CNN+attention, trained on the deduplicated Mal-API-2019+MalbehavD-V1+
# Carpenter behavioral dataset) via a threshold sweep (0.10-0.99) on the
# val+test splits combined (1,835 rows, 274 benign) for more statistical
# power than test alone (137 benign). NOT derived via find_threshold_for_fpr()
# at a precise target like static's 0.1%/1% -- 274 benign samples (~0.36%
# resolution per sample) can't support a precise-sounding FPR claim.
#
# 0.60 rather than the higher end of the zero-observed-FP range (~0.922+):
# there is exactly one persistent false positive across the whole sweep, a
# MalbehavD-V1 benign sample whose trace includes networking-setup calls
# (setsockopt, ioctlsocket, wsastartup, getsockname) alongside routine
# registry/system calls -- plausibly confusable with C2 setup, and the model
# is confidently wrong about it (scores 0.921006, not a wobbly near-threshold
# case). Below its exact score, ANY threshold produces this one false
# positive; from ~0.922 up, false positives disappear on the data we have.
# 0.60 deliberately does NOT clear that single case. Malicious recall climbs
# steadily as the threshold drops (FN count on val+test: 57 at 0.95, 49 at
# 0.90, ~38 at 0.60, 36 at 0.50), and n=1 evidence isn't a sound basis for a
# permanent recall tradeoff across the whole malicious population --
# especially since behavioral currently has no downstream backstop in this
# repo (no rule-based overlay for high-risk call combinations independent of
# the ML score). If/when more real benign network-adjacent software traces
# are added to training data, or the architecture research's rule-based
# overlay for high-risk call combinations gets built, that's the more
# targeted fix for this class of ambiguity -- not blanket threshold tuning
# in response to a single hard example.
BEHAVIORAL_MALICIOUS_MIN = _thr(_THRESHOLDS, "behavioral", "malicious_at_or_above")  # >= this -> MALICIOUS

# Memory threshold -- RE-DERIVED 2026-09-10 by the split-discipline retrain
# in OPEN_ITEMS.md's "retrain cluster" section (PDF review items 2 and 3).
# data/models/cortex_memory was retrained on a new 4-way split
# (scripts/split_memory.py --cal-frac 0.2: 34,949 train / 5,930 val / 11,787
# cal / 5,930 test; val and test are byte-identical to the prior 3-way
# split). The Platt calibrator is now fit on `cal` -- a split held out of
# BOTH the booster fit and early stopping, not the val split -- and this
# operating point is derived on `cal` alone via
# models.memory_lgbm.find_threshold_for_fpr(); `test` was not read at any
# point in training or threshold selection (it is consumed once, at the
# end, by scripts/evaluate_all_models.py). This removes the item-2
# calibration optimism (calibrator previously fit on the early-stopping val
# split) and the item-3 threshold optimism (operating point previously
# chosen on the same val+test pool it was then scored against).
#
# target_fpr=0.01 (1%) on cal -> threshold 0.0024964628, actual FPR 0.9556%
# (56 / 5,860 benign FP), detection_rate 0.9993. Same target-FPR choice the
# pre-fix derivation made, deliberately kept rather than re-picked from the
# new numbers: 56 benign FP is well-supported (one FP moves observed FPR by
# ~0.017%), detection is ~99.9% here and at every looser target, and
# tighter targets (29 FP at 0.5%, 5 FP at 0.1%) are thinner evidence for
# the same recall. The honest detection_rate is 0.9993, not the 1.0000 the
# old val+test derivation reported -- the ~0.1 pp is the optimism the
# split-discipline fix removes, not a regression. Held-out test metrics at
# this threshold are regenerated in EVAL_ALL_MODELS_RESULTS.txt.
#
# IMPORTANT -- CIC-MalMem-2022 metrics are internally valid but likely
# optimistic for production, for a specific, diagnosed reason: the dataset's
# benign class is a repeated capture of "normal user behavior" on a single
# baseline Windows 10 VM, while malicious samples span far more varied
# executions. A large fraction of the model's features are individually
# near-separating on this data (the prior pass measured 22 of 62 above 0.95
# AUC on their own, on both train and held-out data -- a property of the
# dataset, not of any one trained model; the README's separability table is
# the reference). The model may be learning "does this look like that one
# baseline VM," not "is malicious behavior present" -- a distinction this
# dataset alone cannot resolve. This is exactly why memory's authority
# stays capped at ALERT below (see decide()'s docstring): the split-
# discipline retrain fixes calibration and threshold honesty, not feature
# fidelity, so it is a reason to KEEP the cap, not lift it. Real-world
# validation needs benign captures from multiple genuinely different
# (non-baseline, non-single-VM) machines.
#
# Superseded values -- never reuse, each calibrated score scale is specific
# to its calibrator: 0.0006464189644018 (2026-09-08 recalibration on the
# old 3-way split), 0.0005358335957155212 (the pre-calibrator-fix value).
MEMORY_MALICIOUS_MIN: Optional[float] = _thr(_THRESHOLDS, "memory", "malicious_at_or_above")

# Network threshold -- RE-DERIVED 2026-09-10 by the split-discipline retrain
# in OPEN_ITEMS.md's "retrain cluster" section (PDF review items 2 and 3).
# data/models/cortex_network was retrained on a new 4-way split
# (scripts/split_network.py --cal-frac 0.1: 1,331,042 train / 211,697 val /
# 195,715 cal / 213,217 test; val and test byte-identical to the prior
# 3-way split, ambiguous-group exclusion unchanged). The Platt calibrator
# is now fit on `cal` -- held out of BOTH the booster fit and early
# stopping, not the val split -- and this operating point is derived on
# `cal` alone via models.network_lgbm.find_threshold_for_fpr(); `test` was
# not read during training or threshold selection (consumed once, at the
# end, by scripts/evaluate_all_models.py). Removes the item-2 calibration
# optimism (calibrator previously fit on the early-stopping val split) and
# the item-3 threshold optimism (operating point previously chosen on the
# same val+test pool it was then scored against).
#
# target_fpr=0.001 (0.1%) on cal -> threshold 0.6672636218, actual FPR
# 0.0988% (159 / 160,863 benign FP), detection_rate 0.9626. Same target-FPR
# choice as the pre-fix derivation, kept rather than re-picked: 159 benign
# FP is ample statistical power (one FP moves observed FPR by ~0.0006%), and
# loosening to 0.5% buys only +0.8 pp detection for 5x the FPR on an ALERT-
# capped signal. The threshold is not sitting on a discontinuity -- FPR and
# detection move smoothly and monotonically across a +/-0.01 sweep around
# it, and only 5 benign scores lie within +/-0.001. The honest
# detection_rate is 0.9626, not the 0.9752 the old val+test derivation
# reported as held-out recall -- that ~1.3 pp gap is the calibration/
# threshold optimism the fix removes. A single-feature-AUC sweep on train
# and cal found 0 of 78 features above 0.95 AUC (max ~0.77) -- no fixed-
# testbed shortcut, same result as before. Held-out test metrics at this
# threshold are regenerated in EVAL_ALL_MODELS_RESULTS.txt.
#
# IMPORTANT -- reasons this threshold should NOT be read as "solved,"
# unchanged by the retrain (they are dataset/representation properties, not
# calibration artifacts):
#  1. Infiltration attacks are slow/low-volume and lack the flow-statistics
#     signature CICFlowMeter captures for DDoS/brute-force -- a feature-
#     representation ceiling (the prior test pass measured ~10.75% detection
#     on that class at n_train=11,463, i.e. not under-training). Deliberately
#     NOT chased by lowering the threshold, which would spike FPR across
#     every other class. The step-f per-attack-type table (now produced by
#     scripts/evaluate_all_models.py) is the reference.
#  2. CSE-CIC-IDS2018 has independently reported label noise up to ~7.5%
#     (time-window labeling, not per-flow human review).
#  3. Models "nearly perfect" on this dataset -- which this one is -- have
#     been shown to degrade toward random on external traffic.
# These are why network's authority is capped at ALERT below (see decide()'s
# docstring), on two independent legs: the Infiltration blind spot and the
# external-validation-collapse pattern.
#
# Superseded values -- never reuse: 0.5883628015255921 (2026-09-08
# recalibration on the old 3-way split), 0.9441855970306654 (the
# pre-calibrator-fix value).
NETWORK_MALICIOUS_MIN: Optional[float] = _thr(_THRESHOLDS, "network", "malicious_at_or_above")

# Emulation threshold -- derived, but for LOGGING / TELEMETRY ONLY. This is
# the target_fpr=1% point from the val+test sweep against the trained e64
# checkpoint (data/models/cortex_emulation_best.pt): threshold 0.999358594,
# ~24 expected FP on 2,386 combined benign, ~50% pooled detection. See the
# README's "Cortex-Emulation" section for the full sweep table.
#
# Cortex-Emulation is deliberately NOT wired into decide()'s priority chain
# (grep: there is no `emulation_verdict ==` branch below, by design).
# Its deployment status is report-only / additive -- the same disposition
# malware-ml gave its own Behavioral v2 category signal
# (deployment_status: additive_report_only_not_in_policy_decision). The
# reason is stricter than memory's or network's ALERT cap: an ablation-
# confirmed Jan->Apr 2022 temporal concept-drift collapse (malicious recall
# 70.7% -> 41.5% at embed_dim=64, 68.0% -> 38.3% at embed_dim=32, i.e.
# unchanged by capacity reduction), plus a margin of only ~+3-4 pp test
# accuracy over a trivial train-sequence duplicate-lookup baseline. A signal
# that recovers ~40% of malware on the next collection era, and barely beats
# a lookup table, cannot carry autonomous OR ALERT authority yet. Use this
# constant to record an EmulationVerdict alongside a scan for later
# analysis; do not branch policy on it. Revisit only with a materially
# different model (see the README's future-work note: engineered
# behavioural-category features, or training data spanning more collection
# dates -- not further hyperparameter tuning, which the ablation ruled out).
EMULATION_MALICIOUS_MIN: Optional[float] = _thr(_THRESHOLDS, "emulation", "malicious_at_or_above")


class StaticVerdict(str, enum.Enum):
    ALLOW = "ALLOW"
    ALERT = "ALERT"
    BLOCK = "BLOCK"
    ERROR = "ERROR"


class BehavioralVerdict(str, enum.Enum):
    NOT_PROVIDED = "NOT_PROVIDED"
    PENDING = "PENDING"
    BENIGN = "BENIGN"
    MALICIOUS = "MALICIOUS"
    ERROR = "ERROR"


class MemoryVerdict(str, enum.Enum):
    NOT_PROVIDED = "NOT_PROVIDED"
    BENIGN = "BENIGN"
    MALICIOUS = "MALICIOUS"
    ERROR = "ERROR"


class NetworkVerdict(str, enum.Enum):
    NOT_PROVIDED = "NOT_PROVIDED"
    BENIGN = "BENIGN"
    MALICIOUS = "MALICIOUS"
    ERROR = "ERROR"


class EmulationVerdict(str, enum.Enum):
    NOT_PROVIDED = "NOT_PROVIDED"
    BENIGN = "BENIGN"
    MALICIOUS = "MALICIOUS"
    ERROR = "ERROR"


class FinalDecision(str, enum.Enum):
    ALLOW = "ALLOW"
    # NEEDS_REVIEW: analysis could not complete for at least one signal
    # (non-PE / missing / unreadable / oversized file, or an exception during
    # feature extraction or scoring) AND no channel that DID complete produced
    # malicious or suspicious evidence. Distinct from ALERT ("we found
    # suspicious evidence") -- "we could not analyze this" must not be
    # reported as a malware finding. Sits between ALLOW and ALERT in severity.
    NEEDS_REVIEW = "NEEDS_REVIEW"
    ALERT = "ALERT"
    BLOCK = "BLOCK"
    TERMINATE = "TERMINATE"


def static_verdict_from_score(score: float) -> StaticVerdict:
    if score < STATIC_ALLOW_MAX:
        return StaticVerdict.ALLOW
    if score < STATIC_BLOCK_MIN:
        return StaticVerdict.ALERT
    return StaticVerdict.BLOCK


def behavioral_verdict_from_score(score: float) -> BehavioralVerdict:
    return BehavioralVerdict.MALICIOUS if score >= BEHAVIORAL_MALICIOUS_MIN else BehavioralVerdict.BENIGN


def memory_verdict_from_score(score: float) -> MemoryVerdict:
    if MEMORY_MALICIOUS_MIN is None:
        raise NotImplementedError(
            "MEMORY_MALICIOUS_MIN has not been derived yet -- Cortex-Memory isn't "
            "trained. Train it (scripts/train_memory.py, not yet written) and set "
            "the threshold above before calling this."
        )
    return MemoryVerdict.MALICIOUS if score >= MEMORY_MALICIOUS_MIN else MemoryVerdict.BENIGN


def network_verdict_from_score(score: float) -> NetworkVerdict:
    if NETWORK_MALICIOUS_MIN is None:
        raise NotImplementedError(
            "NETWORK_MALICIOUS_MIN has not been derived yet -- train Cortex-Network "
            "(scripts/train_network.py) and set the threshold above before calling this."
        )
    return NetworkVerdict.MALICIOUS if score >= NETWORK_MALICIOUS_MIN else NetworkVerdict.BENIGN


def emulation_verdict_from_score(score: float) -> EmulationVerdict:
    """Report-only / additive. The returned verdict is for logging and later
    analysis, NOT a policy input -- decide() never receives it (see
    EMULATION_MALICIOUS_MIN's comment for why). Callers that surface it in a
    scan record must keep it isolated from the final decision."""
    if EMULATION_MALICIOUS_MIN is None:
        raise NotImplementedError(
            "EMULATION_MALICIOUS_MIN is unset -- train Cortex-Emulation "
            "(scripts/train_emulation.py) and set the telemetry threshold above."
        )
    return EmulationVerdict.MALICIOUS if score >= EMULATION_MALICIOUS_MIN else EmulationVerdict.BENIGN


@dataclass
class ScanResult:
    file_path: str
    sha256: Optional[str] = None
    static_score: Optional[float] = None
    static_verdict: StaticVerdict = StaticVerdict.ERROR
    behavioral_score: Optional[float] = None
    behavioral_verdict: BehavioralVerdict = BehavioralVerdict.NOT_PROVIDED
    memory_score: Optional[float] = None
    memory_verdict: MemoryVerdict = MemoryVerdict.NOT_PROVIDED
    network_score: Optional[float] = None
    network_verdict: NetworkVerdict = NetworkVerdict.NOT_PROVIDED
    # Default is NEEDS_REVIEW, not ALERT: an un-populated result means "nothing
    # ran to completion", which is precisely the NEEDS_REVIEW case, not a
    # malware finding. static_verdict stays ERROR ("nothing ran yet").
    final_decision: FinalDecision = FinalDecision.NEEDS_REVIEW
    reason_codes: list[str] = field(default_factory=list)
    # Analyzer/model health, kept SEPARATE from the security verdict (review
    # item 10). Sparse: signal name -> problem string, populated only when a
    # signal is not healthy; an absent key means "healthy or not applicable".
    # Current values: "model_not_configured" (features supplied but no model
    # wired -- neutral, does not affect the decision) and "model_error" (a
    # configured model raised at runtime -- routes to NEEDS_REVIEW via the
    # signal's ERROR verdict, same as any other analysis failure). This is
    # diagnostic output only; decide() never sees it. Item 6 may extend this
    # with a "degraded" entry for partial feature extraction.
    signal_health: dict[str, str] = field(default_factory=dict)
    # Feature groups that had to be degraded (zero-filled) during static
    # extraction (review item 6). Empty == clean. A degraded CRITICAL group
    # (features.pe_features.CRITICAL_FEATURE_GROUPS) makes pipeline.scan() set
    # static_verdict = ERROR (-> NEEDS_REVIEW) and adds signal_health["static"]
    # = "degraded"; a degraded non-critical group only sets the health flag.
    # decide() never sees this list.
    degraded_groups: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "sha256": self.sha256,
            "static": {"score": self.static_score, "verdict": self.static_verdict.value},
            "behavioral": {"score": self.behavioral_score, "verdict": self.behavioral_verdict.value},
            "memory": {"score": self.memory_score, "verdict": self.memory_verdict.value},
            "network": {"score": self.network_score, "verdict": self.network_verdict.value},
            "final_decision": self.final_decision.value,
            "reason_codes": self.reason_codes,
            "signal_health": self.signal_health,
            "degraded_groups": self.degraded_groups,
        }


def decide(
    static_verdict: StaticVerdict,
    behavioral_verdict: BehavioralVerdict,
    memory_verdict: MemoryVerdict = MemoryVerdict.NOT_PROVIDED,
    network_verdict: NetworkVerdict = NetworkVerdict.NOT_PROVIDED,
) -> tuple[FinalDecision, list[str]]:
    """
    Priority order (never averaged):
        1. behavioral MALICIOUS      -> TERMINATE
        2. memory MALICIOUS          -> ALERT           (capped -- see below)
        3. network MALICIOUS         -> ALERT           (capped -- see below)
        4. static ALERT  or  BLOCK   -> ALERT           (BLOCK is INTERIM-
                                                         CAPPED to ALERT --
                                                         see below)
        5. any static / behavioral / memory / network ERROR -> NEEDS_REVIEW
        6. otherwise                 -> ALLOW

    Reason-code accumulation
    -----------------------
    The ERROR check at rung 5 does NOT drive the outcome on its own when a
    higher rung fires -- but the failed-signal reason code(s) are still
    collected and returned. Every signal in the ERROR state contributes its
    code (`static_scan_error`, `behavioral_scan_error`, `memory_scan_error`,
    `network_scan_error`) to the returned `reasons` list regardless of which
    rung decides the outcome; the code for the rung that actually fired is
    listed first. So `decide(ERROR, MALICIOUS, ...)` still returns TERMINATE,
    but `reasons` is `["behavioral_malicious", "static_scan_error"]` rather
    than dropping the static failure. This is audit-trail only -- it never
    changes the FinalDecision.

    "Cannot analyze" is not "malware" (review item 9)
    ------------------------------------------------
    Rung 5 returns NEEDS_REVIEW, not ALERT: a non-PE / missing / unreadable /
    oversized file, or an exception during feature extraction or scoring,
    means the analyzer could not reach a verdict -- it is NOT evidence of
    maliciousness. NEEDS_REVIEW keeps those out of the malware-alert stream
    while still routing them to a human / review queue (it must not be
    treated as quieter than ALERT operationally, only as a separate queue).
    A signal that DID complete and found something (rungs 1-4) always wins
    over a different signal's failure -- positive evidence outranks absence
    of evidence. Items 6 (feature-extraction `degraded_groups`) and 10
    (memory model availability as a system-health signal) layer on top of
    this state; they are not implemented here.

    INTERIM CAP on Cortex-Static's BLOCK authority (added 2026-09-08)
    ---------------------------------------------------------------
    Static's BLOCK verdict is demoted to ALERT here: a static score above
    STATIC_BLOCK_MIN can no longer autonomously block a file, only raise the
    outcome to ALERT (the same rung as static's own ALERT and as
    memory/network MALICIOUS). `static_verdict` still reports BLOCK for
    telemetry/audit; only the policy consequence is capped, and the reason
    code `static_block_capped_at_alert` records each time it happens.

    Why: BLOCK was the one uncapped, autonomous verdict in this function,
    and validation shows it is not safe to trust yet.
    - The Platt-calibration bug is fixed (calibrators now fit on raw booster
      margins, not sigmoid probabilities) and STATIC_ALLOW_MAX/STATIC_BLOCK_MIN
      were re-derived -- but Platt scaling is a monotonic transform, so it
      cannot change the booster's *ranking* of files.
    - In the post-fix 5-file check, 2 of 5 confirmed-benign Windows binaries
      still land at/above STATIC_BLOCK_MIN and would still be wrongly BLOCKed:
      benign_test_50mb.exe (raw booster margin +3.81 -> calibrated 0.980) and
      extractor.exe (+4.48 -> 0.990). Their raw margins rank them among the
      most-malicious-looking files in the set -- the known PyInstaller /
      atypical-large-PE false-positive pattern, compounded by the documented
      pe_features.py-vs-thrember feature-fidelity skew (cortex-ml's live
      feature vector skews toward "malicious").
    - No calibration or threshold change fixes a ranking problem; only the
      feature-fidelity work does.

    Removal criteria (ALL required):
    - the pe_features.py feature-parity test exists and passes against a
      reference extractor (thrember) on real PEs incl. a signed binary
      (README open item #1); AND
    - the residual pe_features.py-vs-thrember skew is closed (open item #2);
      AND
    - Cortex-Static is re-validated on a real-world confirmed-label file set
      (benign + malicious) with no confirmed-benign file at or above
      STATIC_BLOCK_MIN.
    Until then, do not restore the autonomous BLOCK branch.

    Memory is evaluated as an independent third signal, NOT gated behind a
    static ALLOW the way behavioral is (see pipeline.py: memory runs
    whenever features are supplied, regardless of the static verdict, and
    even when static/PE validation failed outright). This is deliberate:
    Cortex-Memory's entire purpose is catching injected/fileless malicious
    activity that structurally has no on-disk file for static to see in the
    first place -- gating it behind static's verdict would defeat that
    purpose for exactly the threat class it exists to catch.

    However, memory's ceiling is capped at ALERT, one full rung below
    behavioral's TERMINATE, deliberately: Cortex-Memory is trained and
    threshold-derived against CIC-MalMem-2022 only (a fixed, lab-collected
    memory-dump research corpus) and has not yet been validated against real
    injected-process samples outside that test set. Letting an unvalidated
    signal autonomously TERMINATE/BLOCK risks killing a legitimate process on
    a false positive we have no field evidence to rule out. This mirrors the
    same caution already applied to behavioral's own threshold pick in this
    file (see BEHAVIORAL_MALICIOUS_MIN's comment above): don't extend a
    model's authority past what its validation evidence actually supports.
    Revisit this cap once real-world injected-process validation data exists
    for memory -- flagged as an open item in the README. The trained
    model's own held-out test performance reinforces keeping this cap
    rather than raising it: 22 of its 62 features individually separate
    the classes with >0.95 AUC, consistent with CIC-MalMem-2022's benign
    samples all coming from one baseline VM (see MEMORY_MALICIOUS_MIN's
    comment for the full finding) -- a strong signal the near-perfect
    test-set metrics are partly an artifact of that single-VM collection
    methodology, not proof the model generalizes to real injected-process
    detection.

    Network is evaluated the same way memory is -- an independent signal,
    capped at ALERT, never autonomous TERMINATE/BLOCK -- but for two
    separate, evidence-backed reasons rather than one (see
    NETWORK_MALICIOUS_MIN's comment for the full derivation each rests on):

    (a) Infiltration blind spot: Cortex-Network's held-out test detection
        rate for Infiltration attacks is 10.75%, not an under-training
        artifact (11,463 train rows -- comparable classes with far less
        train data still scored >=91%). Independently confirmed as a known
        weak spot of CICFlowMeter's flow-statistics representation
        specifically (infiltration is slow/low-volume, and a comparative
        study found it's only well-detected via NetFlow-derived features,
        not CICFlowMeter's). This means Network structurally cannot be
        trusted alone for exactly the threat class that behaves like a
        slow, quiet process rather than a loud flood -- which is precisely
        the class of activity Behavioral (API-call sequences) and Memory
        (process/injection artifacts) have process-level visibility into
        that Network's flow-only view does not. The system's overall
        coverage of infiltration-style threats depends on that
        complementary visibility, not on Network alone closing the gap.
    (b) External-validation-collapse pattern: models reported as "nearly
        perfect" on CSE-CIC-IDS2018 -- which Cortex-Network's test-set
        numbers are -- have been independently shown elsewhere to degrade
        toward random performance on external/real-world traffic. This is
        a documented pattern for this specific dataset, not a generic
        "may not generalize" hedge, and it means the clean test metrics
        here should not be assumed predictive of production performance
        without independent validation on traffic this dataset didn't
        generate.

    Either reason alone would justify the same cap already applied to
    memory; having both independently is why this cap should be treated as
    at least as firm as memory's, not a formality to relax once Network's
    numbers look good -- they already do, and that's exactly the case this
    cap exists for.
    """
    # Record every signal currently in the ERROR state, up-front and
    # independent of which rung drives the outcome. decide() returns on first
    # match, so without collecting these first a higher-priority result (e.g.
    # behavioral TERMINATE) would silently drop the fact that a lower-priority
    # signal also failed to run. This affects the audit trail ONLY -- the
    # priority chain below consults `reasons` for the OUTCOME exactly where
    # the old per-signal ERROR rungs sat (after every malicious/suspicious
    # check, before ALLOW), and now yields NEEDS_REVIEW instead of ALERT.
    #
    # Each signal's verdict is a single enum value, and ERROR is mutually
    # exclusive with MALICIOUS / ALERT / BLOCK, so a given signal contributes
    # to at most one of "driver reason" or this list -- never both.
    reasons: list[str] = []
    if static_verdict == StaticVerdict.ERROR:
        reasons.append("static_scan_error")
    if behavioral_verdict == BehavioralVerdict.ERROR:
        reasons.append("behavioral_scan_error")
    if memory_verdict == MemoryVerdict.ERROR:
        reasons.append("memory_scan_error")
    if network_verdict == NetworkVerdict.ERROR:
        reasons.append("network_scan_error")

    # Priority chain -- first match decides the OUTCOME. The reason code for
    # the rung that fired is listed first, then any error codes from above.
    #
    # Cortex-Static's BLOCK verdict is NOT an autonomous top-priority block.
    # It is interim-capped to ALERT (see this function's docstring for the
    # evidence and the removal criteria) and handled together with static
    # ALERT below -- after the behavioral MALICIOUS check, so a validated
    # behavioral signal can still escalate the same file to TERMINATE.

    if behavioral_verdict == BehavioralVerdict.MALICIOUS:
        return FinalDecision.TERMINATE, ["behavioral_malicious", *reasons]

    if memory_verdict == MemoryVerdict.MALICIOUS:
        return FinalDecision.ALERT, ["memory_malicious", *reasons]

    if network_verdict == NetworkVerdict.MALICIOUS:
        return FinalDecision.ALERT, ["network_malicious", *reasons]

    if static_verdict in (StaticVerdict.ALERT, StaticVerdict.BLOCK):
        # static BLOCK demoted to ALERT (interim cap); the distinct reason
        # code keeps the demotion visible in the audit trail.
        driver = (
            "static_block_capped_at_alert"
            if static_verdict == StaticVerdict.BLOCK
            else "static_alert"
        )
        return FinalDecision.ALERT, [driver, *reasons]

    # No malicious or suspicious evidence from any channel that completed.
    if reasons:
        # One or more signals could not be analyzed at all -- route to a
        # human / review queue, NOT the malware-alert stream (review item 9).
        return FinalDecision.NEEDS_REVIEW, reasons

    return FinalDecision.ALLOW, ["no_malicious_evidence"]
