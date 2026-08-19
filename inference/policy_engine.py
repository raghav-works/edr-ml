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


@dataclass
class ScanResult:
    file_path: str
    sha256: Optional[str] = None
    static_score: Optional[float] = None
    static_verdict: StaticVerdict = StaticVerdict.ERROR
    behavioral_score: Optional[float] = None
    behavioral_verdict: BehavioralVerdict = BehavioralVerdict.NOT_PROVIDED
    final_decision: FinalDecision = FinalDecision.ALERT
    reason_codes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "sha256": self.sha256,
            "static": {"score": self.static_score, "verdict": self.static_verdict.value},
            "behavioral": {"score": self.behavioral_score, "verdict": self.behavioral_verdict.value},
            "final_decision": self.final_decision.value,
            "reason_codes": self.reason_codes,
        }


def decide(
    static_verdict: StaticVerdict,
    behavioral_verdict: BehavioralVerdict,
) -> tuple[FinalDecision, list[str]]:
    """
    Priority order (never averaged):
        1. static BLOCK            -> BLOCK          (always protected)
        2. behavioral MALICIOUS    -> TERMINATE
        3. static ALERT            -> ALERT
        4. static ERROR            -> ALERT
        5. behavioral ERROR        -> ALERT
        6. otherwise                -> ALLOW
    """
    reasons: list[str] = []

    if static_verdict == StaticVerdict.BLOCK:
        reasons.append("static_block")
        return FinalDecision.BLOCK, reasons

    if behavioral_verdict == BehavioralVerdict.MALICIOUS:
        reasons.append("behavioral_malicious")
        return FinalDecision.TERMINATE, reasons

    if static_verdict == StaticVerdict.ALERT:
        reasons.append("static_alert")
        return FinalDecision.ALERT, reasons

    if static_verdict == StaticVerdict.ERROR:
        reasons.append("static_scan_error")
        return FinalDecision.ALERT, reasons

    if behavioral_verdict == BehavioralVerdict.ERROR:
        reasons.append("behavioral_scan_error")
        return FinalDecision.ALERT, reasons

    reasons.append("no_malicious_evidence")
    return FinalDecision.ALLOW, reasons
