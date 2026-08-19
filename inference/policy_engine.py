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
BEHAVIORAL_MALICIOUS_MIN = 0.9910649657  # score >= this -> MALICIOUS, else BENIGN (behavioral model not yet trained -- placeholder, unchanged)


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
