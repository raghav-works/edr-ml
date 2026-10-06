"""
End-to-end scan pipeline — mirrors the architecture flow diagram, plus
Cortex-Memory and Cortex-Network as independent signals:

  1. path validation (exists, is file, readable, <= 100 MiB)
  2. rule-based Windows PE validation (not an ML prediction)
  3. SHA-256 + feature extraction (2568-dim EMBER2024-compatible). A feature
     group that fails extraction is recorded in ScanResult.degraded_groups
     instead of silently becoming a zero vector (review item 6).
  3b. known-file allowlist (OPEN_ITEMS.md "Static false-positive severity
     cluster"): an NSRL SHA-256 hash match or a real Authenticode
     chain-verification pass (features/authenticode_trust.py -- NOT the
     presence-only authenticode ML feature) sets static_verdict = ALLOW
     directly and skips step 4 entirely for this file. Deliberately bypasses
     ONLY static's own judgment (both its ML score and its feature-
     degradation check below); memory, network, and behavioral all run
     exactly as they do on any other file, allowlisted or not.
  4. Cortex-Static LightGBM -> static score -> ALLOW/ALERT/BLOCK. If a
     CRITICAL feature group degraded, the score is not trusted: static_verdict
     becomes ERROR (-> NEEDS_REVIEW) with reason static_features_degraded.
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
     With file_scan.attach_memory_network false (docs/CODE_REVIEW.md F11),
     steps 7/7b are skipped for file scans: both stay NOT_PROVIDED and a
     supplied vector is recorded as signal_health "detached_by_config".
  8. policy engine -> final decision (ALLOW / NEEDS_REVIEW / ALERT / BLOCK /
     TERMINATE). Memory's and Network's authority is each capped at ALERT,
     one rung below behavioral's TERMINATE; see
     inference/policy_engine.py::decide() for the full reasoning. A file that
     cannot be analyzed at all (non-PE, missing, unreadable, oversized, or an
     extraction/scoring exception) with no malicious/suspicious signal from
     any other channel resolves to NEEDS_REVIEW, not ALERT -- "cannot
     analyze" is not a malware finding.
     Model health (a supplied memory/network signal whose model is not wired,
     or a configured model that raised) is recorded on
     ScanResult.signal_health -- kept SEPARATE from the security verdict
     (review item 10). A not-yet-deployed model stays neutral; a runtime
     model error routes to NEEDS_REVIEW like any other analysis failure, but
     its cause is visible in signal_health rather than only in reason_codes.
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

from features.authenticode_trust import verify_trusted_chain
from features.nsrl_allowlist import NSRLAllowlist
from features.pe_features import CRITICAL_FEATURE_GROUPS, PEFeatureExtractor, truncation_findings
from inference.policy_engine import (
    BEHAVIORAL_MAX_UNK_RATE, FILE_SCAN_ATTACH_MEMORY_NETWORK, BehavioralVerdict, FinalDecision, MemoryVerdict, NetworkVerdict, ScanResult, StaticVerdict,
    behavioral_verdict_from_score, decide, memory_verdict_from_score,
    network_verdict_from_score, static_verdict_from_score,
)
from models.static_lgbm import LGBMModel
from tokenizer.api_tokenizer import (
    ApiTokenizer, input_diagnostics, load_api_calls_json, scored_unk_rate,
)

logger = logging.getLogger("cortex.pipeline")

MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MiB


class CortexPipeline:
    def __init__(self, static_model: LGBMModel, behavioral_model=None,
                 tokenizer: Optional[ApiTokenizer] = None, memory_model=None,
                 network_model=None, nsrl_allowlist: Optional[NSRLAllowlist] = None,
                 device: str = "cpu", self_test: bool = True):
        self.static_model = static_model
        self.behavioral_model = behavioral_model
        self.tokenizer = tokenizer
        # docs/CODE_REVIEW.md F21: dropout and batch-norm must be in inference
        # mode, or a model handed over in train mode scores non-
        # deterministically. Re-checked before every scoring call as well.
        self._ensure_eval()
        # memory_model / network_model are optional. Each is a trained
        # LightGBM model (models/memory_lgbm.py, models/network_lgbm.py) that
        # scores a caller-supplied feature vector -- there is no live memory
        # or traffic capture in this repo. If one is left None, scan()
        # produces NOT_PROVIDED for that signal (neutral in decide()), even
        # when the corresponding *_features argument is passed.
        self.memory_model = memory_model
        self.network_model = network_model
        # nsrl_allowlist is optional (data/download_nsrl.py builds the
        # artifact, features/nsrl_allowlist.py::NSRLAllowlist.load() loads
        # it). None means that leg of the known-file allowlist (step 3b in
        # scan(), OPEN_ITEMS.md "Static false-positive severity cluster")
        # is skipped -- the Authenticode chain-verification leg still runs
        # (it needs no external artifact), and if that also misses, static
        # falls through to its normal ML judgment, exactly as today.
        self.nsrl_allowlist = nsrl_allowlist
        self.device = device
        self.feature_extractor = PEFeatureExtractor()

        # Startup self-test (review item 6): run the feature extractor against
        # a known-good signed PE once, here, so a pefile/signify API change
        # that silently disables a feature group is caught at construction
        # rather than as skewed production scores. A critical failure (a
        # critical group degraded, a malformed vector, authenticode all-zero
        # on a signed binary) means the extractor is broken for EVERY scan ->
        # raise. Non-critical noise (reference PE absent, a non-critical group
        # degraded) is logged, not fatal. Pass self_test=False in tests /
        # offline tooling that build a pipeline without needing the check.
        if self_test:
            failures = self.feature_extractor.self_test()
            blocking = [
                f for f in failures
                if not f.startswith("reference_pe_not_found")
                and not (f.startswith("degraded_group:")
                         and f.split(":", 1)[1] not in CRITICAL_FEATURE_GROUPS)
            ]
            if blocking:
                raise RuntimeError(
                    "PEFeatureExtractor.self_test() failed at startup -- the feature "
                    f"extractor is not producing trustworthy vectors: {failures}"
                )
            if failures:
                logger.warning("PEFeatureExtractor.self_test() non-blocking warnings: %s", failures)

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
        #
        # docs/CODE_REVIEW.md F11: with file_scan.attach_memory_network false
        # (config/thresholds.yaml), a supplied vector is NOT scored for this
        # file's decision -- both stay NOT_PROVIDED -- and the detachment is
        # recorded in signal_health and logged rather than dropped silently.
        # Read at call time so tests can monkeypatch it.
        attach = FILE_SCAN_ATTACH_MEMORY_NETWORK
        if memory_features is not None:
            if attach:
                result.memory_score, result.memory_verdict = self._run_memory(memory_features)
                self._record_signal_health(result, "memory", result.memory_verdict, self.memory_model)
            else:
                self._detach_signal(result, "memory")
        if network_features is not None:
            if attach:
                result.network_score, result.network_verdict = self._run_network(network_features)
                self._record_signal_health(result, "network", result.network_verdict, self.network_model)
            else:
                self._detach_signal(result, "network")

        # 2. basic input validation
        err = self._validate_path(path)
        if err is not None:
            result.static_verdict = StaticVerdict.ERROR
            result.reason_codes.append(err)
            final, reasons = decide(result.static_verdict, result.behavioral_verdict,
                                    result.memory_verdict, result.network_verdict)
            result.final_decision, result.reason_codes = final, result.reason_codes + reasons
            return result

        # _validate_path() already checked exists / is_file / stat-readable /
        # size, but the file can still vanish or become unreadable between
        # that check and this read (TOCTOU). Treat that as "cannot analyze"
        # (NEEDS_REVIEW via decide()), not an unhandled exception that would
        # produce no ScanResult at all. Reuses the "unreadable" reason code
        # from _validate_path so triage reads both the same way.
        try:
            bytez = path.read_bytes()
        except OSError:
            logger.warning("could not read %s after path validation (I/O race?)", file_path)
            result.static_verdict = StaticVerdict.ERROR
            result.reason_codes.append("unreadable")
            final, reasons = decide(result.static_verdict, result.behavioral_verdict,
                                    result.memory_verdict, result.network_verdict)
            result.final_decision, result.reason_codes = final, result.reason_codes + reasons
            return result
        result.sha256 = hashlib.sha256(bytez).hexdigest()

        # 3. rule-based Windows PE validation. The file is parsed exactly once
        # here; the same pefile object feeds the truncation check (3a) and
        # feature extraction (4) and is closed when this scan is done.
        pe = self.feature_extractor.parse(bytez)
        if pe is None:
            result.static_verdict = StaticVerdict.ERROR
            result.reason_codes.append("invalid_or_non_pe_file")
            final, reasons = decide(result.static_verdict, result.behavioral_verdict,
                                    result.memory_verdict, result.network_verdict)
            result.final_decision, result.reason_codes = final, result.reason_codes + reasons
            return result
        try:
            return self._scan_parsed(result, path, bytez, pe, api_calls_json_path)
        finally:
            try:
                pe.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    def _scan_parsed(self, result: ScanResult, path: Path, bytez: bytes, pe,
                     api_calls_json_path: Optional[str]) -> ScanResult:
        file_path = str(path)

        # 3a. truncation (docs/CODE_REVIEW.md F17): a PE whose headers,
        # section raw data, or certificate table extend past EOF is cut off.
        # Its features describe a file that does not exist, so static must not
        # score it -> ERROR -> NEEDS_REVIEW. Like an invalid PE, this ends the
        # static path (behavioral is gated on a usable static verdict).
        truncated = truncation_findings(pe, len(bytez))
        if truncated:
            logger.warning("truncated PE %s: %s", file_path, truncated)
            result.static_verdict = StaticVerdict.ERROR
            result.reason_codes += ["static_pe_truncated", *truncated]
            final, reasons = decide(result.static_verdict, result.behavioral_verdict,
                                    result.memory_verdict, result.network_verdict)
            result.final_decision, result.reason_codes = final, result.reason_codes + reasons
            return result

        # 3b. known-file allowlist (OPEN_ITEMS.md "Static false-positive
        # severity cluster", addition A) -- a rule-based check that runs
        # BEFORE static's ML judgment, not an ML feature. A match sets
        # static_verdict = ALLOW directly and skips step 4-5 (feature
        # extraction + LightGBM scoring) ENTIRELY for this file: static's
        # own judgment -- both its ML score AND its feature-degradation
        # check below -- is deliberately set aside for a file independently
        # verified trustworthy, not because degradation doesn't matter in
        # general. static_score and degraded_groups stay unset/empty in
        # this branch; there is nothing to report because nothing was
        # extracted.
        #
        # CRITICAL CONSTRAINT: this bypasses ONLY static's own judgment.
        # Memory and network (steps 7/7b, above) already ran independently
        # of this branch entirely -- they do not even depend on
        # static_verdict -- and behavioral's gate below
        # (static_verdict in (ALLOW, ALERT, BLOCK)) already includes ALLOW,
        # so an allowlist-driven ALLOW still lets a supplied API trace be
        # scored exactly as an ML-derived ALLOW would. A legitimate signed
        # binary can still be abused at runtime (DLL injection, process
        # hollowing, living-off-the-land abuse), which is exactly what
        # those other signals -- not static -- exist to catch.
        allowlist_reason = self._check_allowlist(bytez, result.sha256)
        if allowlist_reason is not None:
            result.static_verdict = StaticVerdict.ALLOW
            result.reason_codes.append(allowlist_reason)
        else:
            # 4-5. static feature extraction + LightGBM
            try:
                vec, degraded = self.feature_extractor.feature_vector_with_report(bytez, pe=pe)
                result.degraded_groups = degraded
                static_score = float(self.static_model.predict_proba(vec.reshape(1, -1))[0])
                result.static_score = static_score
                result.static_verdict = static_verdict_from_score(static_score)
            except Exception:
                logger.exception("static scan failed for %s", file_path)
                result.static_verdict = StaticVerdict.ERROR
                result.reason_codes.append("static_scan_exception")

        # 5b. feature-group degradation telemetry (review item 6). ANY degraded
        # group -- critical or not -- is recorded on signal_health for ops
        # visibility (and on result.degraded_groups, set above). Whether a
        # degraded group also overrides the verdict is decided in step 6b,
        # AFTER behavioral has had its chance to run.
        if result.degraded_groups:
            result.signal_health["static"] = "degraded"

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
             beh_notes, result.behavioral_input_diagnostics) = self._run_behavioral(api_calls_json_path)
            # _record_signal_health's "not configured" check is `model is
            # None`; behavioral's not-configured condition is
            # `behavioral_model is None or tokenizer is None` (see
            # _run_behavioral), so mirror that here rather than passing
            # self.behavioral_model alone -- a tokenizer-only config gap
            # must be flagged too, not just a missing model.
            configured_behavioral = (
                self.behavioral_model if self.tokenizer is not None else None
            )
            self._record_signal_health(result, "behavioral", result.behavioral_verdict, configured_behavioral)
            result.reason_codes += beh_notes
        # else: static ERROR, or no api_calls_json_path -> behavioral stays NOT_PROVIDED

        # 6b. If a CRITICAL feature group degraded (features.pe_features
        # .CRITICAL_FEATURE_GROUPS), the static score cannot be trusted in a
        # known direction -- zeros can fabricate maliciousness (an empty import
        # table) OR erase it (dropped IOC strings). Set the score-derived
        # verdict aside: StaticVerdict.ERROR routes the scan to NEEDS_REVIEW
        # via decide(). This runs AFTER the behavioral gate on purpose -- a
        # degraded static extraction must not suppress a caller-supplied API
        # trace, and a completed behavioral MALICIOUS on the same file still
        # wins (decide() rung 1 -> TERMINATE). A degraded NON-critical group
        # (exports / richheader / pefilewarnings) is in-distribution and does
        # not reach here. static_score stays populated for telemetry.
        critical_degraded = set(result.degraded_groups) & CRITICAL_FEATURE_GROUPS
        if critical_degraded:
            logger.warning("critical feature group(s) degraded on %s: %s -> static verdict set aside",
                           file_path, sorted(critical_degraded))
            result.static_verdict = StaticVerdict.ERROR
            if "static_features_degraded" not in result.reason_codes:
                result.reason_codes.append("static_features_degraded")

        # 8. policy engine
        final, reasons = decide(result.static_verdict, result.behavioral_verdict,
                                result.memory_verdict, result.network_verdict)
        result.final_decision = final
        result.reason_codes += reasons
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _detach_signal(result: ScanResult, name: str) -> None:
        """F11: `name`_features were supplied to a file scan while
        file_scan.attach_memory_network is false. Not scored, verdict stays
        NOT_PROVIDED; visible in signal_health and the log."""
        result.signal_health[name] = "detached_by_config"
        logger.warning("%s_features supplied to a file scan of %s but file_scan.attach_memory_network "
                       "is false: not scored and not used in the decision", name, result.file_path)

    # ------------------------------------------------------------------
    @staticmethod
    def _record_signal_health(result: ScanResult, name: str, verdict, model) -> None:
        """Record model health for `name` on result.signal_health, SEPARATE
        from the security verdict (review item 10). Only non-healthy states
        are written; an absent key means healthy / not applicable.

          - configured model raised at runtime  -> "model_error"
            (the signal's ERROR verdict already routes the scan to
             NEEDS_REVIEW via decide(); this just makes "which model, why"
             visible outside the flat reason_codes list)
          - features supplied but no model wired -> "model_not_configured"
            (NOT_PROVIDED: deliberately neutral -- a not-yet-deployed signal
             must not push every scan to NEEDS_REVIEW -- but no longer
             silent: a caller that wants fail-closed behaviour can key off
             this field itself)
          - memory/network features supplied while file_scan
            .attach_memory_network is false -> "detached_by_config"
            (written by _detach_signal, not here; docs/CODE_REVIEW.md F11)

        decide() never receives signal_health; this is diagnostic output only.
        """
        if verdict.name == "ERROR":
            result.signal_health[name] = "model_error"
        elif verdict.name == "NOT_PROVIDED" and model is None:
            result.signal_health[name] = "model_not_configured"

    # ------------------------------------------------------------------
    def _check_allowlist(self, bytez: bytes, sha256_hex: Optional[str]) -> Optional[str]:
        """Known-file allowlist (scan()'s step 3b). Two independent checks,
        either sufficient -- returns the reason code for whichever matched
        first, or None if neither did:

          1. NSRL hash match, if an allowlist artifact is configured
             (self.nsrl_allowlist is not None). Cheap (O(log N) binary
             search); checked first.
          2. Authenticode chain verification to a genuinely trusted root
             (features/authenticode_trust.py::verify_trusted_chain) --
             needs no external artifact, always attempted regardless of
             whether NSRL is configured.

        Returns None (never raises) on no match, missing sha256, or any
        internal failure of either check -- an allowlist check must never
        be the reason a scan fails or produces a false ALLOW; static's
        normal ML judgment is always the safe fallback.
        """
        if sha256_hex is not None and self.nsrl_allowlist is not None and self.nsrl_allowlist.contains(sha256_hex):
            return "static_allowlisted_nsrl"
        if verify_trusted_chain(bytez).trusted:
            return "static_allowlisted_authenticode_chain"
        return None

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
    def _ensure_eval(self) -> None:
        if self.behavioral_model is not None and getattr(self.behavioral_model, "training", False):
            self.behavioral_model.eval()

    # ------------------------------------------------------------------
    def _run_behavioral(self, api_calls_json_path: str
                        ) -> tuple[Optional[float], BehavioralVerdict, list[str], Optional[dict]]:
        """Returns (score, verdict, notes, input_diagnostics).

        `notes` (reason codes for the audit trail, possibly empty):
          - "behavioral_short_trace": the verdict came from a short (10-99
            call) padded sequence -- the model is validated there (test
            AUC 0.9921, 0 FP on 118 benign short rows) but short-benign
            coverage rests almost entirely on one dataset (MalbehavD-V1).
          - "behavioral_unk_rate_high": more than BEHAVIORAL_MAX_UNK_RATE of
            the scored window is <UNK> -> PENDING, not scored (F3).
          - "behavioral_logit_non_finite": the model returned NaN/inf ->
            ERROR. Checked on the raw logit because sigmoid(+-inf) is a
            normal-looking 1.0/0.0 (F12 follow-up).

        `input_diagnostics`: input_diagnostics() counts plus "unk_rate" for
        any trace that was read; None if the model is not configured or the
        file could not be read. Logged at WARNING when a count is non-zero.

        Sequences with < MIN_SEQ_LEN (10) real calls stay PENDING: below that
        the model is non-discriminative (val+test AUC 0.66, 0/4 malicious
        detected), so PENDING is the honest answer, not a limitation.

        No behavioral_model/tokenizer configured is a config gap, not a scan
        failure -- mirrors _run_memory/_run_network: NOT_PROVIDED is neutral
        in decide() (a not-yet-deployed signal must not force every scan to
        NEEDS_REVIEW); ERROR is reserved for a real runtime failure below.
        """
        if self.behavioral_model is None or self.tokenizer is None:
            return None, BehavioralVerdict.NOT_PROVIDED, [], None
        diagnostics: Optional[dict] = None
        try:
            calls = load_api_calls_json(api_calls_json_path)
            diagnostics = input_diagnostics(calls)
            if any(diagnostics.values()):
                logger.warning("behavioral trace %s has input-format problems: %s",
                               api_calls_json_path, diagnostics)
            token_ids, status = self.tokenizer.encode(calls)
            diagnostics["unk_rate"] = scored_unk_rate(token_ids, len(calls))
            if status in ("empty", "too_short"):
                return None, BehavioralVerdict.PENDING, [], diagnostics
            if diagnostics["unk_rate"] > BEHAVIORAL_MAX_UNK_RATE:
                logger.warning("behavioral trace %s: <UNK> rate %.3f > %.3f -> PENDING",
                               api_calls_json_path, diagnostics["unk_rate"], BEHAVIORAL_MAX_UNK_RATE)
                return None, BehavioralVerdict.PENDING, ["behavioral_unk_rate_high"], diagnostics

            import torch
            self._ensure_eval()
            with torch.inference_mode():
                x = torch.from_numpy(token_ids).long().unsqueeze(0).to(self.device)
                logit = self.behavioral_model(x)
                if not bool(torch.isfinite(logit).all()):
                    logger.error("behavioral model returned a non-finite logit for %s", api_calls_json_path)
                    return None, BehavioralVerdict.ERROR, ["behavioral_logit_non_finite"], diagnostics
                score = torch.sigmoid(logit).item()
            notes = ["behavioral_short_trace"] if status == "short" else []
            return score, behavioral_verdict_from_score(score), notes, diagnostics
        except Exception:
            logger.exception("behavioral scan failed for %s", api_calls_json_path)
            return None, BehavioralVerdict.ERROR, [], diagnostics

    # ------------------------------------------------------------------
    def to_security_event(self, result: ScanResult) -> dict:
        event = result.to_dict()
        event["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        event["correlation_id"] = str(uuid.uuid4())
        return event
