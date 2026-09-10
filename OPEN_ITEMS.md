# Open items

Running tracker for known-but-deferred work. Blocking review items 1–6 are
done (see git log); this is what remains.

## PDF review item 9 — `NEEDS_REVIEW` state — DONE

`FinalDecision.NEEDS_REVIEW` now sits between `ALLOW` and `ALERT`. The four
`*Verdict.ERROR` rungs in `decide()` return `NEEDS_REVIEW` instead of `ALERT`
(unchanged priority position — a completed malicious/suspicious finding still
wins). `decide()` now accumulates every failed-signal reason code even when a
higher rung drives the outcome. `pipeline.scan()`'s `path.read_bytes()` is
guarded (TOCTOU → `NEEDS_REVIEW`, not an unhandled crash). Covered by
`tests/test_policy_engine.py` (truth table + invariants) and the new
`tests/test_pipeline.py` (non-PE / missing / directory / oversized).

## PDF review item 10 — model health separate from the security verdict — DONE

`ScanResult.signal_health` (a sparse `{signal: problem}` map, serialized in
`to_dict()` / `to_security_event()`) records analyzer/model health without
touching `decide()`. `pipeline.scan()` populates it for the two supplied
signals: a configured memory/network model that raises → `"model_error"`
(its `ERROR` verdict still routes the scan to `NEEDS_REVIEW`); features
supplied with no model wired → `"model_not_configured"`, which stays
**neutral** (verdict `NOT_PROVIDED`, decision unchanged) — a not-yet-deployed
signal is made visible, not escalated. Covered by `tests/test_pipeline.py`
(model-error, unconfigured-but-neutral, and a guard that `signal_health`
never reaches `decide()`).

### Known inconsistency — behavioral config-gap (tracked, not yet decided)

Memory and network treat "features supplied, no model configured" as
**neutral** (`NOT_PROVIDED`). Behavioral does not: `_run_behavioral` returns
`BehavioralVerdict.ERROR` when `behavioral_model` or `tokenizer` is `None`,
which now routes to `NEEDS_REVIEW`. So the same operator mistake — supplying
evidence for a signal whose model was never wired — is silently ignored for
two signals and forces a review queue entry for the third. This asymmetry
predates items 9/10 and was deliberately left in place to keep item 10
scoped. Someone should decide, eyes open, whether behavioral's config-gap
should also become neutral `NOT_PROVIDED` (consistent, but a not-yet-deployed
behavioral model then stops being fail-closed) or whether memory/network
should instead fail-closed like behavioral. Either direction is a one-signal
change plus test updates; the point is that it should be a decision, not an
accident.

## PDF review item 4 — deployment-prevalence evaluation — DONE (per-signal)

`scripts/evaluate_all_models.py` prints a deployment-prevalence projection
after every confusion matrix: `PPV(π) = TPR·π / (TPR·π + FPR·(1−π))`, alert
rate, and false/true positives per 10k/100k files, at assumed malicious base
rates (default 1 in 1,000 / 10,000 / 100,000, `--prevalence` to override).
The projection math (`ppv_at_prevalence` / `alert_rate_at_prevalence` /
`expected_fp` / `expected_tp` / `project_to_prevalence`) is a pure function
of the measured FPR/TPR and is unit-tested in `tests/test_prevalence.py`
(round-trip to the measured precision at the test split's own prevalence,
hand-computed low-π cases, monotonicity, edge cases). A block computed from
0 observed false positives prints its `n_benign` and a rule-of-three 95% CI
upper bound on the true FPR, so a clean-looking PPV can't hide a thin benign
set. `--target-ppv` reports, read-only against each test ROC, the
highest-recall threshold reaching a target PPV (no threshold change).

**Explicit non-goal (stated in the script output, not just here):** the
COMBINED pipeline's ALERT / NEEDS_REVIEW / TERMINATE volume through
`decide()` is not modelled. That needs a file-population model — what
fraction of endpoint files are non-PE, carry an API trace / memory vector /
network flow, or fail extraction — which this repo does not have. NEEDS_REVIEW
volume in particular is dominated by the non-PE fraction (item 9 moved that
out of the ALERT stream), which no model test set can estimate.

The **modeling-side** prior correction the PDF also mentions (shifting the
calibrated probability / threshold to target a precision at π, which edits
`config/thresholds.yaml`) is deliberately NOT done here — it belongs in the
retrain cluster. `--target-ppv` shows what such a threshold would cost in
recall without making the change.

## PDF review item 6 — feature-extraction degradation is explicit — DONE

`PEFeatureExtractor` no longer swallows a failed feature group into a silent
zero vector. `feature_vector_with_report()` returns the list of degraded
groups; `raw_features()` / `process_raw_features()` log at WARNING and record
each degradation; `process_raw_features()` also validates every group's
output shape/finiteness so a malformed part (e.g. a `None` raw group making
`ByteHistogram` return a 0-d NaN) is caught and zero-filled rather than
producing a short, non-finite vector. `HeaderFileInfo`'s internal
`try/except` was removed so a header-parse failure actually surfaces;
`authenticode`'s `parse_error=1` is surfaced as a degradation too.

`ScanResult.degraded_groups` carries the list. `pipeline.scan()` sets
`signal_health["static"] = "degraded"` for any degradation, and for a
**critical** group (`CRITICAL_FEATURE_GROUPS` — the nine whose all-zero fill
fabricates or erases a primary maliciousness signal) also sets
`static_verdict = ERROR` (→ `NEEDS_REVIEW`, reason `static_features_degraded`)
— *after* the behavioral gate, so a caller-supplied API trace still runs and
a completed behavioral `MALICIOUS` still wins. Non-critical groups
(`exports`, `richheader`, `pefilewarnings` — where all-zero is also a common
legitimate value) keep the score-derived verdict.

`PEFeatureExtractor.self_test()` runs the extractor against the committed
signed fixture (`tests/fixtures/pe_samples/sample_signed64.exe` — single
source of truth, not a duplicated copy) and `CortexPipeline(self_test=True)`
(default) raises at construction if a critical group is broken, warns on
non-critical noise. Covered by `tests/test_pipeline.py` and
`tests/test_static_feature_parity.py`.

### Dependency pinning — deferred to item 7's parity cluster

The PDF's item 6 also says "pin dependency versions". Only `signify` has an
upper bound today; `pefile` and `scikit-learn` (whose `FeatureHasher` hashing
is parity-critical) are lower-bound only. Tightening `scikit-learn` needs a
`FeatureHasher` hash-stability re-check, which belongs with the `thrember`
parity cross-check (item 7 in the Structural section below), not a standalone
change. Tracked there, not forgotten.

## Structural

- **Feature-parity harness for `pe_features.py`** — MVP DONE
  (`tests/test_static_feature_parity.py`, item 7): vector contract,
  determinism, signed-binary authenticode, ExportsInfo count slot, and
  adapter-passthrough parity, on committed synthetic PE fixtures. Two
  sub-items remain:
  - **Real EMBER2024 record schema check.** The adapter assumes an
    EMBER2024 record's group dicts (`record["general"]`,
    `record["header"]["coff"]`, ...) have the exact keys/shape that
    `pe_features.raw_features()` produces. The MVP only proves the adapter
    faithfully processes a dict `raw_features` itself made. Verifying
    against a *real* record needs a small (~100-record) pull from
    `joyce8/EMBER2024` on HF — NOT a full parquet regeneration. Needed
    before trusting the adapter beyond "faithful passthrough".
  - **`thrember` skew-quantification cross-check.** Compare the live
    `pe_features` vector against the gold-standard `thrember` extractor on
    real PEs, per group, to quantify the documented "skews toward
    malicious" gap. BLOCKED in this repo's main venv: `thrember` (a git
    install from the EMBER2024 repo) imports the pre-0.9 `signify`
    authenticode API, and item 4 pinned `signify>=0.9,<0.10`. The two
    cannot coexist — this check needs a separate venv (or to run inside
    the `malware-ml` venv, which already has `thrember` + old `signify`).
- **Baseline `pytest` suite** — item 8. `tests/` scaffold (`conftest.py`,
  `pytest.ini`) exists from item 7. Still to add:
  `tests/test_policy_engine.py` (the `decide()` truth table run by hand all
  session) and a pytest wrapper around `scripts/verify_onnx_parity.py`.

## Deferred to a deliberate retrain pass

- **Threshold-selection optimistic bias** — memory/network operating
  thresholds are derived on `val+test` combined and then held-out metrics
  are quoted on that same pool. Fix = a 3-way split (separate
  threshold-selection set). Fold into the next retrain, not a standalone
  re-run.
- **`pe_features.py`-vs-thrember feature-fidelity skew** — cortex-ml's live
  vector skews toward "malicious"; this is the real cause behind the interim
  cap on Cortex-Static's BLOCK authority (`policy_engine.decide()` "INTERIM
  CAP"). Closing it needs the parity harness first, then a static retrain +
  threshold re-derivation + real-world re-validation.
- **`ExportsInfo` count fix** (commit `bb1ade0`) delivers no model change
  until the next static retrain regenerates the EMBER2024 parquets.

## Cleanup

- **Unreachable `None` guards** — `MEMORY/NETWORK/EMULATION_MALICIOUS_MIN`
  keep `Optional[float]` typing and `is None -> NotImplementedError` guards
  in `*_verdict_from_score()`. Since `policy_engine` now loads thresholds
  from YAML (loader yields a float or raises), those paths are dead.
- **`config/` packaging** — `policy_engine` loads `config/thresholds.yaml`
  via `parents[1]/config`; fine running from the tree, but there is no
  `pyproject.toml`/`setup.py`, so packaging would need to bundle `config/`.
- **`thresholds.yaml` `sequence.*` / `file.*`** keys are informational only;
  `api_tokenizer.MAX_SEQ_LEN` / `pipeline.MAX_FILE_SIZE_BYTES` still hold
  their own constants.

## Docs pass (batch, not piecemeal)

- `README.md` — calibration-saturation section (~lines 176/852) still
  describes pre-fix behaviour; head/flow already updated.
- `PROJECT_HISTORY_REPORT.md` — uncommitted edits pending.
