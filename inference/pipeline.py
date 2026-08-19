"""
End-to-end scan pipeline — mirrors the architecture flow diagram exactly:

  1. path validation (exists, is file, readable, <= 100 MiB)
  2. rule-based Windows PE validation (not an ML prediction)
  3. SHA-256 + feature extraction (2568-dim EMBER2024-compatible)
  4. Cortex-Static LightGBM -> static score -> ALLOW/ALERT/BLOCK
  5. gate: only a static ALLOW proceeds to behavioral analysis
  6. (if API-call JSON supplied) Cortex-Behavioral -> BENIGN/MALICIOUS/PENDING
  7. policy engine -> final decision
  8. structured JSON security event
"""

from __future__ import annotations

import datetime
import hashlib
import logging
import uuid
from pathlib import Path
from typing import Optional

import numpy as np

from features.pe_features import PEFeatureExtractor
from inference.policy_engine import (
    BehavioralVerdict, FinalDecision, ScanResult, StaticVerdict,
    behavioral_verdict_from_score, decide, static_verdict_from_score,
)
from models.static_lgbm import LGBMModel
from tokenizer.api_tokenizer import ApiTokenizer, load_api_calls_json

logger = logging.getLogger("cortex.pipeline")

MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MiB


class CortexPipeline:
    def __init__(self, static_model: LGBMModel, behavioral_model=None,
                 tokenizer: Optional[ApiTokenizer] = None, device: str = "cpu"):
        self.static_model = static_model
        self.behavioral_model = behavioral_model
        self.tokenizer = tokenizer
        self.device = device
        self.feature_extractor = PEFeatureExtractor()

    # ------------------------------------------------------------------
    def _validate_path(self, path: Path) -> Optional[str]:
        if not path.exists():
            return "path_not_found"
        if not path.is_file():
            return "not_a_file"
        try:
            size = path.stat().st_size
        except OSError:
            return "unreadable"
        if size > MAX_FILE_SIZE_BYTES:
            return "file_too_large"
        return None

    # ------------------------------------------------------------------
    def scan(self, file_path: str, api_calls_json_path: Optional[str] = None) -> ScanResult:
        path = Path(file_path)
        result = ScanResult(file_path=str(path))

        # 2. basic input validation
        err = self._validate_path(path)
        if err is not None:
            result.static_verdict = StaticVerdict.ERROR
            result.reason_codes.append(err)
            final, reasons = decide(result.static_verdict, result.behavioral_verdict)
            result.final_decision, result.reason_codes = final, result.reason_codes + reasons
            return result

        bytez = path.read_bytes()
        result.sha256 = hashlib.sha256(bytez).hexdigest()

        # 3. rule-based Windows PE validation
        if not self.feature_extractor.is_valid_pe(bytez):
            result.static_verdict = StaticVerdict.ERROR
            result.reason_codes.append("invalid_or_non_pe_file")
            final, reasons = decide(result.static_verdict, result.behavioral_verdict)
            result.final_decision, result.reason_codes = final, result.reason_codes + reasons
            return result

        # 4-5. static feature extraction + LightGBM
        try:
            feats = self.feature_extractor.feature_vector(bytez).reshape(1, -1)
            static_score = float(self.static_model.predict_proba(feats)[0])
            result.static_score = static_score
            result.static_verdict = static_verdict_from_score(static_score)
        except Exception:
            logger.exception("static scan failed for %s", file_path)
            result.static_verdict = StaticVerdict.ERROR
            result.reason_codes.append("static_scan_exception")

        # 6. gate: only ALLOW proceeds to behavioral
        if result.static_verdict == StaticVerdict.ALLOW and api_calls_json_path is not None:
            result.behavioral_score, result.behavioral_verdict = self._run_behavioral(api_calls_json_path)
        elif result.static_verdict != StaticVerdict.ALLOW:
            # per architecture: BLOCK short-circuits immediately; ALERT still
            # "preserves alert and examines behavior" in the prose flow, but
            # the decision table shows Alert+Behavioral only matters if
            # MALICIOUS upgrades it to TERMINATE, so we still run behavioral
            # for ALERT (not BLOCK) when data is supplied.
            if result.static_verdict == StaticVerdict.ALERT and api_calls_json_path is not None:
                result.behavioral_score, result.behavioral_verdict = self._run_behavioral(api_calls_json_path)
            # BLOCK: never run behavioral, static block is always protected
        # else: no api_calls_json_path supplied -> behavioral stays NOT_PROVIDED

        # 7. policy engine
        final, reasons = decide(result.static_verdict, result.behavioral_verdict)
        result.final_decision = final
        result.reason_codes += reasons
        return result

    # ------------------------------------------------------------------
    def _run_behavioral(self, api_calls_json_path: str) -> tuple[Optional[float], BehavioralVerdict]:
        if self.behavioral_model is None or self.tokenizer is None:
            return None, BehavioralVerdict.ERROR
        try:
            calls = load_api_calls_json(api_calls_json_path)
            token_ids, status = self.tokenizer.encode(calls)
            if status in ("empty", "insufficient"):
                return None, BehavioralVerdict.PENDING

            import torch
            with torch.no_grad():
                x = torch.from_numpy(token_ids).long().unsqueeze(0).to(self.device)
                score = torch.sigmoid(self.behavioral_model(x)).item()
            return score, behavioral_verdict_from_score(score)
        except Exception:
            logger.exception("behavioral scan failed for %s", api_calls_json_path)
            return None, BehavioralVerdict.ERROR

    # ------------------------------------------------------------------
    def to_security_event(self, result: ScanResult) -> dict:
        event = result.to_dict()
        event["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        event["correlation_id"] = str(uuid.uuid4())
        return event
