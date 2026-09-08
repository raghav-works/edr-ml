"""
End-to-end scan pipeline — mirrors the architecture flow diagram, plus
Cortex-Memory and Cortex-Network as independent signals:

  1. path validation (exists, is file, readable, <= 100 MiB)
  2. rule-based Windows PE validation (not an ML prediction)
  3. SHA-256 + feature extraction (2568-dim EMBER2024-compatible)
  4. Cortex-Static LightGBM -> static score -> ALLOW/ALERT/BLOCK
  5. gate: a static ALLOW / ALERT / BLOCK proceeds to behavioral analysis
     (only a static scan ERROR skips it)
  6. (if API-call JSON supplied) Cortex-Behavioral -> BENIGN/MALICIOUS/PENDING
  7.  (if memory features supplied)  Cortex-Memory  -> BENIGN/MALICIOUS
  7b. (if network flow features supplied) Cortex-Network -> BENIGN/MALICIOUS
     Memory and Network both run independently of steps 1-6 -- NOT gated
     behind static's verdict, and not skipped even if path/PE validation
     failed. Their purpose (catching injected/fileless activity, and
     network-only C2/exfil) requires them to run when there is no valid
     on-disk file for steps 1-4 to evaluate. Neither derives its own
     features: the caller supplies them (no live memory or traffic capture
     exists in this repo), the same as behavioral's API-call trace.
  8. policy engine -> final decision. Memory's and Network's authority is
     each capped at ALERT, one rung below behavioral's TERMINATE; see
     inference/policy_engine.py::decide() for the full reasoning.
  9. structured JSON security event
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
    BehavioralVerdict, FinalDecision, MemoryVerdict, NetworkVerdict, ScanResult, StaticVerdict,
    behavioral_verdict_from_score, decide, memory_verdict_from_score,
    network_verdict_from_score, static_verdict_from_score,
)
from models.static_lgbm import LGBMModel
from tokenizer.api_tokenizer import ApiTokenizer, load_api_calls_json

logger = logging.getLogger("cortex.pipeline")

MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MiB


class CortexPipeline:
    def __init__(self, static_model: LGBMModel, behavioral_model=None,
                 tokenizer: Optional[ApiTokenizer] = None, memory_model=None,
                 network_model=None, device: str = "cpu"):
        self.static_model = static_model
        self.behavioral_model = behavioral_model
        self.tokenizer = tokenizer
        # memory_model / network_model are optional. Each is a trained
        # LightGBM model (models/memory_lgbm.py, models/network_lgbm.py) that
        # scores a caller-supplied feature vector -- there is no live memory
        # or traffic capture in this repo. If one is left None, scan()
        # produces NOT_PROVIDED for that signal (neutral in decide()), even
        # when the corresponding *_features argument is passed.
        self.memory_model = memory_model
        self.network_model = network_model
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
    def scan(self, file_path: str, api_calls_json_path: Optional[str] = None,
              memory_features: Optional[np.ndarray] = None,
              network_features: Optional[np.ndarray] = None) -> ScanResult:
        path = Path(file_path)
        result = ScanResult(file_path=str(path))

        # 7 / 7b. Memory and Network (deliberately run first, before step 1's
        # own validation) independently of every file-validation check below
        # -- unlike behavioral they are not gated behind a static ALLOW, and
        # unlike static/behavioral they do not even depend on file_path
        # resolving to a valid on-disk PE (a fileless/injected threat, or a
        # network-only C2/exfil flow, has no such file to resolve). See
        # policy_engine.decide() for why their authority is nonetheless
        # capped at ALERT. Neither captures its own data: the caller supplies
        # the feature vectors.
        if memory_features is not None:
            result.memory_score, result.memory_verdict = self._run_memory(memory_features)
        if network_features is not None:
            result.network_score, result.network_verdict = self._run_network(network_features)

        # 2. basic input validation
        err = self._validate_path(path)
        if err is not None:
            result.static_verdict = StaticVerdict.ERROR
            result.reason_codes.append(err)
            final, reasons = decide(result.static_verdict, result.behavioral_verdict,
                                    result.memory_verdict, result.network_verdict)
            result.final_decision, result.reason_codes = final, result.reason_codes + reasons
            return result

        bytez = path.read_bytes()
        result.sha256 = hashlib.sha256(bytez).hexdigest()

        # 3. rule-based Windows PE validation
        if not self.feature_extractor.is_valid_pe(bytez):
            result.static_verdict = StaticVerdict.ERROR
            result.reason_codes.append("invalid_or_non_pe_file")
            final, reasons = decide(result.static_verdict, result.behavioral_verdict,
                                    result.memory_verdict, result.network_verdict)
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

        # 6. gate: behavioral runs for any static verdict that produced a
        # usable score -- ALLOW, ALERT, or BLOCK -- when an API-call trace is
        # supplied. A file scoring above STATIC_BLOCK_MIN now gets the SAME
        # behavioral scrutiny as one scoring just below it: static BLOCK is
        # interim-capped to ALERT in the policy engine (see
        # policy_engine.decide()'s "INTERIM CAP" section), so skipping
        # behavioral on BLOCK would give the higher-scoring file LESS
        # scrutiny -- and behavioral MALICIOUS is the one signal that can
        # still escalate it to TERMINATE. Only a static ERROR skips
        # behavioral (no reliable static evidence to combine it with).
        if (result.static_verdict in (StaticVerdict.ALLOW, StaticVerdict.ALERT, StaticVerdict.BLOCK)
                and api_calls_json_path is not None):
            (result.behavioral_score, result.behavioral_verdict,
             beh_note) = self._run_behavioral(api_calls_json_path)
            if beh_note is not None:
                result.reason_codes.append(beh_note)
        # else: static ERROR, or no api_calls_json_path -> behavioral stays NOT_PROVIDED

        # 8. policy engine
        final, reasons = decide(result.static_verdict, result.behavioral_verdict,
                                result.memory_verdict, result.network_verdict)
        result.final_decision = final
        result.reason_codes += reasons
        return result

    # ------------------------------------------------------------------
    def _run_memory(self, memory_features: np.ndarray) -> tuple[Optional[float], MemoryVerdict]:
        if self.memory_model is None:
            # No model configured is a config gap, not a scan failure.
            # NOT_PROVIDED is neutral in decide(); ERROR would wrongly force
            # a fail-closed ALERT (that is reserved for real runtime failures).
            logger.warning("memory_features supplied but no memory_model configured; skipping memory signal")
            return None, MemoryVerdict.NOT_PROVIDED
        try:
            score = float(self.memory_model.predict_proba(memory_features.reshape(1, -1))[0])
            return score, memory_verdict_from_score(score)
        except Exception:
            logger.exception("memory scan failed")
            return None, MemoryVerdict.ERROR

    # ------------------------------------------------------------------
    def _run_network(self, network_features: np.ndarray) -> tuple[Optional[float], NetworkVerdict]:
        """Score a caller-supplied CICFlowMeter flow-stats vector. Mirrors
        _run_memory: NOT_PROVIDED (neutral in decide()) when no model is
        configured, ERROR only on an actual runtime failure. Network's
        MALICIOUS verdict is capped at ALERT in decide() -- never autonomous
        BLOCK/TERMINATE; see NETWORK_MALICIOUS_MIN's comment and decide()'s
        docstring for the two-legged reasoning (Infiltration blind spot +
        external-validation-collapse pattern)."""
        if self.network_model is None:
            logger.warning("network_features supplied but no network_model configured; skipping network signal")
            return None, NetworkVerdict.NOT_PROVIDED
        try:
            score = float(self.network_model.predict_proba(network_features.reshape(1, -1))[0])
            return score, network_verdict_from_score(score)
        except Exception:
            logger.exception("network scan failed")
            return None, NetworkVerdict.ERROR

    # ------------------------------------------------------------------
    def _run_behavioral(self, api_calls_json_path: str) -> tuple[Optional[float], BehavioralVerdict, Optional[str]]:
        """Returns (score, verdict, note).

        `note` is "behavioral_short_trace" when the verdict came from a short
        (10-99 call) padded sequence -- the model is validated there (test
        AUC 0.9921, 0 FP on 118 benign short rows) but short-benign coverage
        rests almost entirely on one dataset (MalbehavD-V1), so a caller
        should surface this in the scan's audit trail, the same caution
        applied to memory/network authority. `note` is None otherwise.

        Sequences with < MIN_SEQ_LEN (10) real calls stay PENDING: below that
        the model is non-discriminative (val+test AUC 0.66, 0/4 malicious
        detected), so PENDING is the honest answer, not a limitation.
        """
        if self.behavioral_model is None or self.tokenizer is None:
            return None, BehavioralVerdict.ERROR, None
        try:
            calls = load_api_calls_json(api_calls_json_path)
            token_ids, status = self.tokenizer.encode(calls)
            if status in ("empty", "too_short"):
                return None, BehavioralVerdict.PENDING, None

            import torch
            with torch.no_grad():
                x = torch.from_numpy(token_ids).long().unsqueeze(0).to(self.device)
                score = torch.sigmoid(self.behavioral_model(x)).item()
            note = "behavioral_short_trace" if status == "short" else None
            return score, behavioral_verdict_from_score(score), note
        except Exception:
            logger.exception("behavioral scan failed for %s", api_calls_json_path)
            return None, BehavioralVerdict.ERROR, None

    # ------------------------------------------------------------------
    def to_security_event(self, result: ScanResult) -> dict:
        event = result.to_dict()
        event["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        event["correlation_id"] = str(uuid.uuid4())
        return event
