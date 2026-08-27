"""
Policy engine — sequential-gate decision logic.

Exactly reproduces the architecture: static and behavioral scores are never
averaged/ensembled. Static runs first; only a static ALLOW allows the file to
proceed to behavioral analysis. Final decision is rule-priority, not blended.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
# Static thresholds re-derived from data/models/cortex_static (LightGBM,
# trained on the deduplicated EMBER2024 PE-format train split) via
# models.static_lgbm.find_threshold_for_fpr() against the calibrated
# probabilities on the deduplicated EMBER2024 test split (539,940 rows).
# STATIC_BLOCK_MIN hits the architecture doc's max_fpr@0.1% = 0.001 bar
# (target_fpr=0.001 -> actual FPR=0.000997, detection_rate=0.9168 at this
# threshold). STATIC_ALLOW_MAX uses target_fpr=0.01 (actual FPR=0.009999,
# detection_rate=0.9803): the score distribution has a cliff between
# target_fpr 0.005 and 0.01 (threshold drops from ~0.96 to ~0.62), so 0.01
# is the point where ALLOW stops sitting in near-BLOCK territory and starts
# meaningfully separating from ALERT.
STATIC_ALLOW_MAX = 0.6163460957      # score <  this -> ALLOW
STATIC_BLOCK_MIN = 0.9950119117      # score >= this -> BLOCK   (between the two -> ALERT)

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
BEHAVIORAL_MALICIOUS_MIN = 0.60      # score >= this -> MALICIOUS, else BENIGN

# Memory threshold -- re-derived against data/models/cortex_memory
# (LightGBM + Platt calibration, trained on scripts/split_memory.py's
# group-aware, leakage-checked CIC-MalMem-2022 split: 46,736 train / 5,930
# val / 5,930 test) via models.memory_lgbm.find_threshold_for_fpr() on the
# calibrated probabilities from val+test COMBINED (11,860 rows, 5,860
# benign -- combined rather than test alone for the same statistical-power
# reason as BEHAVIORAL_MALICIOUS_MIN, though memory's benign count here is
# far healthier than behavioral's 274: one false positive moves the
# observed FPR by only ~0.017%). target_fpr=0.01 (1%): the best-supported
# target among those tried (55 observed benign FPs on val+test, vs. only 5
# at target_fpr=0.001 -- too few to trust) that also costs nothing in
# recall (detection_rate is already 100% at this and every looser target
# tried). Held-out test-set metrics at this threshold: AUC-ROC=1.0,
# precision=0.9891, recall=1.0, F1=0.9945, FPR=1.13% (33/2,930 benign test
# rows). Full sweep table and derivation is in the README.
#
# IMPORTANT -- these metrics are internally valid on CIC-MalMem-2022 but
# likely optimistic for production, for a specific, diagnosed reason, not a
# generic disclaimer: 22 of the model's 62 raw+derived features
# individually exceed 0.95 AUC on their own (confirmed independently on
# both train and the held-out test split -- e.g. handles.avg_handles_per_proc
# is a tight 208-318 band for benign [std=17.5] vs. 71-33,784 for malware
# [std=222.8]). This is consistent with CIC-MalMem-2022's own documented
# collection methodology: every benign sample is a repeated capture of
# "normal user behavior" on a single baseline Windows 10 VM, while
# malicious samples span far more varied executions/environments. The
# model may be learning "does this look like that one baseline VM," not
# "is malicious behavior present" -- a distinction this dataset alone
# cannot resolve. This is exactly why memory's authority stays capped at
# ALERT below (see decide()'s docstring): this finding is a reason to KEEP
# that cap, not a one-off caveat to note and move past. Real-world
# validation would require memory captures from multiple genuinely
# different (non-baseline, non-single-VM) benign machines -- not more rows
# from this same dataset, and not a generic "may not generalize" hand-wave.
MEMORY_MALICIOUS_MIN: Optional[float] = 0.0005358335957155212

# Network threshold -- re-derived against data/models/cortex_network
# (LightGBM + Platt calibration, trained on scripts/split_network.py's
# leakage-checked, ambiguous-group-excluded CSE-CIC-IDS2018 split:
# 1,526,757 train / 211,697 val / 213,217 test) via
# models.network_lgbm.find_threshold_for_fpr() at target_fpr=0.001 on the
# calibrated val+test-combined probabilities (424,914 rows, 321,733 benign
# -- far more statistical power than memory's 5,860 or behavioral's 274;
# one false positive here moves the observed FPR by only ~0.0003%). Held-out
# test-set metrics at this threshold: AUC-ROC=0.9972, precision=0.9969,
# recall=0.9752, F1=0.9859, FPR=0.098%. A single-feature-AUC sweep (the
# same check that caught Cortex-Memory's single-VM shortcut) found nothing
# suspicious here: 0 of 78 features exceed 0.95 AUC individually, on either
# train or held-out test -- the model's performance reflects genuine
# multi-feature pattern learning, not a fixed-testbed artifact, despite
# CSE-CIC-IDS2018 also being a fixed-testbed capture.
#
# IMPORTANT -- three separate, evidence-backed reasons this threshold
# should NOT be read as "solved," despite the clean numbers above:
#
# 1. Infiltration detection is 10.75% (n_test=1,433, n_train=11,463 -- NOT
#    an under-training artifact; every other under-10,000-train class
#    still hit >=91% except the two flagged below). Independently confirmed
#    as a documented weak spot: infiltration attacks are slow/low-volume by
#    design and don't produce the distinctive flow-statistics signature
#    CICFlowMeter captures well for DDoS/brute-force -- a comparative study
#    found infiltration on this dataset is only well-detected using
#    NetFlow-derived features, not CICFlowMeter's. This is a feature-
#    representation ceiling, not a pipeline defect, and was deliberately
#    NOT chased by tuning the threshold lower -- that would spike FPR
#    across every other class for a class this representation structurally
#    can't see well. Full per-attack-type table in the README.
# 2. CSE-CIC-IDS2018 has independently reported label noise of up to ~7.5%
#    (automatic time-window-based labeling, not human-verified per-flow) --
#    some fraction of both the training signal and the test-set "ground
#    truth" used to compute the metrics above is simply wrong, in a
#    direction and magnitude we can't correct for from inside this dataset.
# 3. Models reported as "nearly perfect" on this dataset -- which is what
#    ours is -- have been independently shown to degrade toward random
#    performance when evaluated on external/real-world traffic. This is a
#    documented pattern for CSE-CIC-IDS2018 specifically, not a generic
#    "may not generalize" hedge: the test-set numbers above should not be
#    assumed predictive of real-world performance without independent
#    validation on traffic this dataset didn't generate.
#
# These three are why network's authority is capped at ALERT below, the
# same rung as memory -- see decide()'s docstring for the full reasoning,
# which (unlike memory's single-legged single-VM finding) rests on two
# independent legs: the Infiltration blind spot, and the external-
# validation-collapse pattern.
NETWORK_MALICIOUS_MIN: Optional[float] = 0.9441855970306654

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
EMULATION_MALICIOUS_MIN: Optional[float] = 0.999358594417572


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
    final_decision: FinalDecision = FinalDecision.ALERT
    reason_codes: list[str] = field(default_factory=list)

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
        }


def decide(
    static_verdict: StaticVerdict,
    behavioral_verdict: BehavioralVerdict,
    memory_verdict: MemoryVerdict = MemoryVerdict.NOT_PROVIDED,
    network_verdict: NetworkVerdict = NetworkVerdict.NOT_PROVIDED,
) -> tuple[FinalDecision, list[str]]:
    """
    Priority order (never averaged):
        1. static BLOCK              -> BLOCK          (always protected)
        2. behavioral MALICIOUS      -> TERMINATE
        3. memory MALICIOUS          -> ALERT           (capped -- see below)
        4. network MALICIOUS         -> ALERT           (capped -- see below)
        5. static ALERT              -> ALERT
        6. static ERROR              -> ALERT
        7. behavioral ERROR          -> ALERT
        8. memory ERROR              -> ALERT
        9. network ERROR             -> ALERT
        10. otherwise                 -> ALLOW

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
    reasons: list[str] = []

    if static_verdict == StaticVerdict.BLOCK:
        reasons.append("static_block")
        return FinalDecision.BLOCK, reasons

    if behavioral_verdict == BehavioralVerdict.MALICIOUS:
        reasons.append("behavioral_malicious")
        return FinalDecision.TERMINATE, reasons

    if memory_verdict == MemoryVerdict.MALICIOUS:
        reasons.append("memory_malicious")
        return FinalDecision.ALERT, reasons

    if network_verdict == NetworkVerdict.MALICIOUS:
        reasons.append("network_malicious")
        return FinalDecision.ALERT, reasons

    if static_verdict == StaticVerdict.ALERT:
        reasons.append("static_alert")
        return FinalDecision.ALERT, reasons

    if static_verdict == StaticVerdict.ERROR:
        reasons.append("static_scan_error")
        return FinalDecision.ALERT, reasons

    if behavioral_verdict == BehavioralVerdict.ERROR:
        reasons.append("behavioral_scan_error")
        return FinalDecision.ALERT, reasons

    if memory_verdict == MemoryVerdict.ERROR:
        reasons.append("memory_scan_error")
        return FinalDecision.ALERT, reasons

    if network_verdict == NetworkVerdict.ERROR:
        reasons.append("network_scan_error")
        return FinalDecision.ALERT, reasons

    reasons.append("no_malicious_evidence")
    return FinalDecision.ALLOW, reasons
