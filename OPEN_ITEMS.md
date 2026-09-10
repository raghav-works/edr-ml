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

## The retrain cluster (PDF items 2, 3, and item 8's residue) — design agreed 2026-09-10

Three coupled defects, all fixed by one retrain pass with a proper split
discipline. The split-scheme design below was reviewed and agreed on
2026-09-10; it is the plan of record for the implementation sessions, which
have **not** started.

### The defects

- **Item 2 — calibration optimism.** The Platt calibrator for static,
  memory, and network is fit on `X_val` — the *same* split that drives early
  stopping (`best_iteration`). `best_iteration` is chosen to maximize val
  separation, so the booster's margins on val are optimistically separated,
  and a calibrator fit on them produces over-confident probabilities on
  genuinely held-out data. Verified in `models/static_lgbm.py::train()`,
  `models/memory_lgbm.py::train()`, `scripts/train_network.py`.
- **Item 3 — threshold optimism.** Static picks its operating thresholds on
  `ember2024_test` itself (`config/thresholds.yaml` header says so
  explicitly) and then reports metrics on that same split. Memory and
  network pick thresholds on `val + test` combined and quote held-out
  metrics on that same pool (`scripts/train_memory.py`,
  `scripts/train_network.py`). The operating point is chosen on data it is
  later scored against.
- **Item 8 residue.** The `ExportsInfo` count fix (commit `bb1ade0`) is
  correct in `features/pe_features.py` / `features/ember2024_adapter.py` but
  the deployed `data/models/cortex_static.lgbm` (Aug 18) was trained on
  parquets carrying the old constant-128 bug. It lands only when the
  EMBER2024 parquets are regenerated and static is retrained — which this
  pass does anyway.

### Target discipline — four roles, consumed in this order

1. `train` — booster fit.
2. `val` — `best_iteration` only.
3. `cal` — Platt fit **and** `config/thresholds.yaml` re-derivation. One
   split covers both: each is a "fit a monotone map / pick an operating
   point" task, neither is scored against, and a 5th split only fragments
   thin data (memory) for no gain.
4. `test` — read exactly once, at the very end, for reported numbers only.

EMBER2024 upstream ships only `train` / `test` zips (+ an unlabeled
`challenge.zip`) — checked against the HF file list. There is no free
upstream validation split; static's `cal` must be carved locally.

### Agreed carve: freeze `val` + `test`, slice `cal` from the front of `train`

The greedy allocators in `scripts/split_memory.py` /
`scripts/split_network.py` fill val, then test, then train from **one seeded
permutation** of the group list. Insert a `cal` phase between `test` and
`train`: with the seed unchanged, val and test receive the same leading
groups they get today (**byte-identical**), and `cal` is a deterministic
slice of what would have been train. `scripts/train_static.py` has no split
script — `_load_train_val_split` does the same with `rng.permutation(n_rows)`
and slicing; `perm[:n_val]` stays val, `perm[n_val:n_val+n_cal]` becomes
`cal`, the remainder is train.

Why this and not a fresh 4-way re-split: every historical `val` / `test`
metric stays comparable (only the model changes — retrained on less data,
recalibrated on `cal`, re-thresholded on `cal`); the group/dedup
leakage invariants extend to `cal` for free (groups stay atomic units); and
churn in the maintained record is minimal. It is **per-model** — the four
datasets share no rows, so a shared `cal` split is not meaningful; the
recipe is what is shared.

| Model | train → | `val` (frozen) | `cal` (new) | `test` (frozen) | `cal` benign ≈ | Notes |
|---|---|---|---|---|---|---|
| Static | 2,340,000 → ~1,872,000 (80%) | 234,000 (10%) | ~234,000 (10%) | 539,940 | ~117,000 | static threshold **moves off `test`** onto `cal` |
| Memory | 46,736 → ~35,000 | 5,930 | **~11,700** (enlarged) | 5,930 | ~5,800 | `cal` deliberately enlarged to match the benign count the current `val+test` derivation used; train at ~35k still ample for 62 features |
| Network | 1,526,757 → ~1,320,000 | 211,697 | ~211,700 | 213,217 | ~178,000 | `split_network.py`'s per-group Python loop (~25 min) re-runs once |

### Implementation sequence (agreed)

**Memory + network first, as one session; static as its own later session.**
Memory/network are minutes of compute, low RAM, fully reversible — they
prove the split + calibration + threshold pattern and get it committed
before the expensive half. Static's parquet regen (many GB from HF; also
where item 8 lands) + retrain is **~2 h and has previously OOM-killed this
38 GB machine**; the memory-management mitigations are all in place and the
2.34M-row train has completed post-fix since, and carving `cal` reduces
train rows, but it stays the long pole and the only real failure mode.
124 GB disk free, 20 cores.

Per session, in order:
- `b.` add a `--cal-frac` / `cal` phase to the split path (per model above).
- `c.` refit the calibrator on `cal`, not on the early-stopping `val`.
- `d.` re-derive every value in `config/thresholds.yaml` from `cal` only —
  `test` stays untouched until everything is frozen.
- `e.` (static session only) regenerate the EMBER2024 parquets via
  `data/download_ember2024.py` — this also lands item 8's `ExportsInfo`
  fix — then retrain static.
- `f.` re-run `scripts/evaluate_all_models.py` (now with item 4's
  deployment-prevalence projection) on the clean splits and regenerate
  `EVAL_ALL_MODELS_RESULTS.txt`.
- `g.` re-check item 2's interim BLOCK cap — see below.

### Item 2 interim BLOCK cap — this pass does NOT lift it

`policy_engine.decide()`'s docstring lists three removal criteria, ALL
required, and all three are the thrember feature-parity cluster (item 7,
still blocked on the `signify` version conflict): parity test passing on
real PEs, residual `pe_features`-vs-thrember skew closed, real-world
confirmed-label re-validation with no confirmed-benign file at/above
`STATIC_BLOCK_MIN`. The retrain fixes *calibration and threshold
discipline*, not *feature fidelity* or *booster ranking* — the docstring is
explicit that "no calibration or threshold change fixes a ranking problem".
The pass lands item 8's `ExportsInfo` fix (one small input-fidelity gain)
but the cap stays until item 7 is done.

### `pe_features.py`-vs-thrember feature-fidelity skew (separate track)

cortex-ml's live vector skews toward "malicious"; this is the real cause
behind the BLOCK cap. Closing it needs the parity harness first (item 7),
then its own static retrain + threshold re-derivation + real-world
re-validation — separate from the split-discipline retrain above.

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
