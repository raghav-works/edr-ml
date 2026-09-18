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
2026-09-10; it is the plan of record for the implementation sessions.

**Status (2026-09-10): memory + network half DONE. Static half still OPEN**
as its own later session (EMBER2024 parquet regen + ~2 h retrain that has
previously OOM-killed this machine — see "Implementation sequence" below).
The static session also lands item 8's residue and re-checks the BLOCK cap
(item 2 interim cap NOT lifted — see below). See "Progress" under
"Implementation sequence" for exactly what the memory/network half changed.

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

### Progress — memory + network half (2026-09-10)

Done in one session, exactly the agreed sequence (steps `b`, `c`, `d`, `f`;
`e` and `g` are static-only). `val` and `test` for both models are
**byte-identical** to the pre-retrain 3-way splits (SHA-256 verified) — only
`train` shrank and `cal` was carved between `test` and `train` from the same
seeded per-bucket permutation.

- `b.` `scripts/split_memory.py` / `scripts/split_network.py` gained a
  required `--cal-frac` / `--cal-out` and a `cal` phase inserted between the
  `test` and `train` fills. Memory `--cal-frac 0.2` → cal 11,787 rows /
  5,860 benign (enlarged, matches the old val+test benign count); network
  `--cal-frac 0.1` → cal 195,715 rows / 160,863 benign (the design table's
  "~211,700" over-projected; 0.1 hits the stated "~10%" spec and the benign
  count is far more than the `target_fpr=0.001` derivation needs). Leakage /
  group / ambiguous-group invariants all re-passed.
- `c.` `models/memory_lgbm.py::train()` / `models/network_lgbm.py::train()`
  take `X_cal, y_cal` and fit the Platt calibrator on `cal` margins at
  `best_iteration` (not `X_val`). `scripts/train_memory.py` /
  `scripts/train_network.py` dropped `--test` entirely — they take
  `--train --val --cal --out`, so "`test` untouched during training" is
  structural, not disciplinary. Network's per-attack-type test breakdown
  moved to `scripts/evaluate_all_models.py` (step `f`); its single-feature
  AUC leakage check now runs train-vs-`cal`. Retrained: memory
  `best_iteration` 71→127 (0.7 s), network 760→879 (54 s). ONNX re-exported
  for both; `verify_onnx_parity` clean (0 verdict flips at the new
  thresholds). `models/static_lgbm.py` is untouched — the static session
  mirrors this change there.
- `d.` `config/thresholds.yaml`: `memory.malicious_at_or_above`
  0.0006464189644018 → **0.0024964628** (cal `target_fpr=0.01`: 56/5,860
  benign FP, detection 0.9993); `network.malicious_at_or_above`
  0.5883628015255921 → **0.6672636218** (cal `target_fpr=0.001`: 159/160,863
  benign FP, detection 0.9626). Same target-FPR choices as the superseded
  derivation, re-measured on `cal` — not re-picked from the new numbers.
  `inference/policy_engine.py` constant comments rewritten to match; the
  ALERT-cap rationale for both signals is retained (dataset/representation
  properties, unaffected by the retrain).
- `f.` `EVAL_ALL_MODELS_RESULTS.txt` sections 2 (memory) and 3 (network)
  regenerated on the frozen `test` splits, now with the item-4 prevalence
  projection and network's per-attack-type table. Held-out `test` at the new
  thresholds is essentially unchanged from the old committed numbers (memory
  FPR 0.96%/detection 0.9997; network FPR 0.09%/recall 0.9750/AUC 0.99718)
  — the operating point was re-derived without ever reading `test`, and it
  still lands well there. Sections 1/4/5 (static / behavioral / emulation)
  left as-is. Full suite green at every step (117 passed).

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

## Static false-positive severity cluster — allowlist + corroboration-aware decisioning — design agreed 2026-09-18

**Status (2026-09-18): Addition A DONE (see "Addition A — DONE" below).
B1 in progress next. B2's mechanism is decided (see "Addition B" below) but
not yet implemented.** Two additions, agreed in this order, motivated by a
severity distinction the current system doesn't make: static's known false positives
(the PyInstaller-packed / atypical-large-PE pattern behind item 2's BLOCK
cap, see "Cortex-Static: PE-file model" above) sometimes land on files the
system needs to keep running — e.g. core Windows system binaries — which is
a materially worse failure mode than a false positive on an arbitrary
third-party file. Both additions are scoped below; addition B's second half
is explicitly flagged as needing the maintainer's confirmation before any
implementation, per instruction.

**Emulation is excluded from both additions**, deliberately, and stays
excluded until its own separate future item: it is already telemetry-only
(`decide()` never receives it — see "Cortex-Emulation" above) because of a
documented temporal concept-drift collapse (recall dropped from ~70% to
~41% on data only 3 months newer than its training set) and only marginal
improvement over a trivial baseline. Folding it into either addition below
would mean trusting exactly the signal already shown not to generalize
forward in time; it needs its own retrain and re-validation before being
considered for *any* decision tier, not just this cluster.

### Addition A — known-file allowlist (pre-empts static's ML judgment) — DONE

A rule-based check that runs **before** static's LightGBM model sees a file,
short-circuiting to `ALLOW` on a match and skipping static's ML judgment
entirely for that file — while every other signal keeps running exactly as
it does today, allowlisted or not, because a legitimate signed binary can
still be abused at runtime (DLL injection, process hollowing, living-off-
the-land) and memory/behavioral/network are what catch that.

**Confirmed nothing like this exists today.** A repo-wide search for
allowlist/whitelist/NSRL/trusted-root/known-good-hash-database turns up only
unrelated hits: a CICIDS *label*-value allowlist in the network dataset
loader (nothing to do with files or trust), and `pe_features.py`'s
`self_test()` "known-good signed PE" — a bundled *test fixture* used to
catch a `pefile`/`signify` API regression at startup, not a runtime trust
mechanism. Addition A is genuinely new.

**Where it sits (confirmed against the current code):**
`pipeline.scan()` today runs, in order: (2) path validation, (3)
`is_valid_pe()`, (4) `feature_vector_with_report()` + `static_model
.predict_proba()`. The SHA-256 needed for a hash-based lookup is already
computed at step 2's `hashlib.sha256(bytez)` for every scan — free reuse.
Addition A becomes a new step **3b**, between PE validation and feature
extraction: on a match, set `static_verdict = StaticVerdict.ALLOW` directly,
record which check matched (a reason code, e.g.
`static_allowlisted_nsrl` / `static_allowlisted_authenticode_chain`, plus
possibly a dedicated `ScanResult` field for the matched source — exact shape
TBD at implementation time), and **skip step 4 entirely** — no feature
extraction, no model call, for static only.

The CRITICAL CONSTRAINT the maintainer stated is already satisfied by the
existing control flow, not something addition A has to newly enforce:
memory (`_run_memory`) and network (`_run_network`) both run in `scan()`
before the static branch even begins and are wholly independent of
`static_verdict`; behavioral's gate is `static_verdict in (ALLOW, ALERT,
BLOCK)`, which already includes `ALLOW` — so an allowlist-driven `ALLOW`
still lets behavioral score a supplied API trace exactly as it would for an
ML-derived `ALLOW`. Addition A only needs to make sure step 3b sets
`static_verdict` to a value already inside that existing gate; it doesn't
touch memory/network/behavioral's code paths at all.

**Two independent match conditions, confirmed feasible, not yet built:**

1. **NIST NSRL hash match.** NSRL's Reference Data Set publishes hash
   lists per category (the "Modern" category — current, actively-used
   software — is the one that matches "core Windows system binaries," and
   is the only category NIST is currently planning a trimmed "minimal"
   database for). The minimal set is a bulk download (tens of millions of
   rows, roughly 1–2 GB compressed depending on category/format) containing
   SHA-1/MD5/SHA-256/filename/product metadata; the newer RDSv3 format ships
   as a SQLite DB, the legacy minimal format as a flat `NSRLFile.txt`.
   Two integration shapes exist: (i) a downloaded/prebuilt local artifact —
   e.g. a new `data/download_nsrl.py` mirroring the existing
   `data/download_ember2024.py` pattern, indexing the SHA-256 column into a
   compact local store `pipeline.py` loads at construction time; or (ii) a
   live `nsrlsvr`-style lookup daemon queried per scan. **Recommend (i)**:
   ARCHITECTURE.md is explicit that "there is no live traffic capture, ...
   model-serving API, queue, database, or web service in this repository,"
   and a runtime daemon dependency for every scan breaks that property for
   no clear benefit over a prebuilt local index. The full corpus is too
   large to vendor in git and needs the same "download script produces a
   gitignored artifact, checked at runtime" treatment already used for
   EMBER2024.
2. **Authenticode chain verification to a real trusted root.** Confirmed
   against the `signify` 0.9.2 already pinned and installed in this venv:
   `AuthenticodeFile.verify()` / `AuthenticodeSignature.verify()` perform
   **real chain verification** — not presence-detection — returning valid
   certificate chains or raising `AuthenticodeVerificationError`, checking
   digest match, key usage, and chain validity against a
   `trusted_certificate_store` argument. This venv already carries `signify`'s
   transitive `mscerts` dependency, which builds
   `signify.authenticode.TRUSTED_CERTIFICATE_STORE` from a bundled
   `authroot.stl` — Microsoft's actual Authenticode root Certificate Trust
   List. Confirmed non-empty and genuine: 566 real roots (Microsoft Root CA
   variants, VeriSign, etc. — the same trust list Windows itself uses for
   Authenticode), not a stand-in. **No new dependency is needed.** What's
   missing is a *call* to `.verify()` — today, `pe_features.py`'s
   `AuthenticodeSignature.raw_features()` (the ML feature, unaffected by
   this work) only calls `iter_signatures()` and enumerates certs for
   summary stats (`num_certs`, `self_signed` via issuer==subject heuristic,
   etc.); it never calls `.verify()` and so never asserts real chain
   validity. Addition A needs a **standalone function**, structurally
   separate from that ML feature group — proposed home: a new module (e.g.
   `features/authenticode_trust.py`) exposing something like
   `verify_trusted_chain(bytez) -> bool` (or a small result object), called
   only from `pipeline.py`'s new step 3b, never from
   `PEFeatureExtractor`/`AuthenticodeSignature`.

Match condition for step 3b := SHA-256 found in the local NSRL store **OR**
`verify_trusted_chain()` succeeds. Exact match logic (OR vs. requiring both,
whether a chain-verified-but-NSRL-absent file gets the same treatment as an
NSRL-present one) is an implementation-time decision, not resolved here.

#### Addition A — implementation summary (2026-09-18)

Built and merged exactly as scoped above, both match conditions:

- **`features/authenticode_trust.py`** (new) — `verify_trusted_chain(bytez)`
  returns an `AuthenticodeTrustResult(trusted, reason)`, `reason` one of
  `verified` / `not_signed` / `parse_error` / `chain_untrusted`. Calls
  `AuthenticodeFile.verify(trusted_certificate_store=TRUSTED_CERTIFICATE_STORE)`
  — genuinely never touches `pe_features.py`/`AuthenticodeSignature`.
  Verified against this repo's own fixtures during development, not just
  assumed: the repo's one signed fixture (`sample_signed64.exe`) is
  self-signed and correctly returns `trusted=False, reason=chain_untrusted`
  — the load-bearing regression case, since a false "verified" there would
  mean the allowlist trusts every signed binary this repo's own fixture
  generator produces. Never raises.
- **`data/download_nsrl.py`** (new) — offline/manual script (not run in CI
  or at scan time, same posture as `download_ember2024.py`): fetches NIST's
  version-independent "current" Modern-minimal RDSv3 alias, queries the
  `FILE` view's `sha256` column, and writes a sorted flat binary artifact of
  raw 32-byte digests. `--sqlite-path` lets the extraction/build logic be
  exercised in tests against a small synthetic SQLite fixture, without any
  network access.
- **`features/nsrl_allowlist.py`** (new) — `NSRLAllowlist.load(path)`
  memory-maps that artifact and answers `.contains(sha256_hex)` via
  `np.searchsorted`, O(log N). Returns `None` (not an error) when the
  artifact isn't present — mirrors how an unconfigured memory/network model
  is already neutral rather than a failure.
- **`inference/pipeline.py`** — new step 3b (`_check_allowlist`, called
  between PE validation and static feature extraction) and a new
  `nsrl_allowlist=None` constructor kwarg, wired the same way
  `memory_model`/`network_model` already are. A match sets
  `static_verdict = ALLOW` directly, skips static's feature extraction and
  LightGBM call entirely, and records a reason code
  (`static_allowlisted_nsrl` / `static_allowlisted_authenticode_chain`) —
  no new `ScanResult` field, reusing `reason_codes` the same way
  `static_block_capped_at_alert` / `static_features_degraded` already do.
  The module docstring's step list and the step-3b code comment both say
  plainly that this skips BOTH static's ML score AND its feature-
  degradation check, deliberately — static's own judgment is being set
  aside for an independently-verified file, not because degradation stops
  mattering in general.

**A real bug caught and fixed during development, worth recording
explicitly:** the first on-disk format considered for the NSRL digest
array used numpy's fixed-width byte-string dtype (`'S32'`). Direct testing
(not assumption) showed `'S32'` silently strips trailing `0x00` bytes on
comparison and storage — confirmed with `np.frombuffer(...).item()`/
equality checks before any test was written against it. Against tens of
millions of real SHA-256 digests, a hash ending in `0x00` is a ~1-in-256
event, not a corner case: under `'S32'` this would have produced both false
negatives (a genuinely allowlisted hash silently never matching itself) and
false positives (two unrelated digests sharing their first 31 bytes and
both ending in `0x00` comparing equal). Switched to numpy's void (`'V32'`)
dtype, verified byte-for-byte exact under sort/searchsorted/equality, and
added `test_digest_ending_in_null_byte_round_trips_correctly` as a
permanent regression test. Same category of finding as the double-failure
mode caught during the earlier feature-degradation work (review item 6) —
a plausible-looking implementation that is silently wrong on a real,
non-rare slice of the input space — surfaced here before it ever shipped
rather than after.

**Critical-constraint proof, not just an assertion:** `tests/test_pipeline.py`
carries three dedicated tests —
`test_allowlist_hit_does_not_suppress_memory_malicious`,
`..._network_malicious`, `..._behavioral_terminate` — each configuring the
static model as one that raises `AssertionError` if ever invoked, so a
regression that let the allowlist branch fall through to real ML scoring
would fail loudly via that assertion, not silently pass. All three confirm
memory/network/behavioral reach their own independent verdict (`ALERT`,
`ALERT`, `TERMINATE` respectively) — and `decide()`'s own outcome — on the
same allowlisted scan, unaffected by static's bypass.

Full suite green throughout (142 passed at completion: 117 baseline + 25
new — `test_authenticode_trust.py` (6), `test_nsrl_allowlist.py` (7),
`test_download_nsrl.py` (6), and `test_pipeline.py`'s new allowlist section
(6) — verified by exact count against the baseline commit, not estimated).

### Addition B — corroboration-aware decision logic (after A)

**B1 — corroboration as a reason-code signal, not a blended score.**
`decide()`'s existing priority rungs already let memory `MALICIOUS` and
network `MALICIOUS` fire independently, each producing an identical `ALERT`
outcome with no distinction from the other, or from just one of them being
true. Proposal: after the existing priority chain picks its rung, detect
whether **two or more independently-capped signals** (memory, network, and
static ALERT/BLOCK) agree at the same time and, if so, append an additional
reason code (e.g. `corroborated_multi_signal`, or specific pairwise codes)
to the existing `reasons` list — **without changing `FinalDecision`** (still
`ALERT`; the priority/rung model is untouched). This is structurally
identical to how `decide()` already accumulates ERROR reason codes up front
today regardless of which rung fires (see the `reasons: list[str] = []`
block near the top of `decide()`) — corroboration detection is the same
pattern, computed up front and appended after the driving rung's own code.
A downstream consumer (e.g. a review-queue prioritizer) can then treat a
corroborated `ALERT` as higher-priority than a single-signal one, entirely
from `reason_codes` — no numeric confidence/blended score is introduced,
per instruction, since that would reintroduce exactly the failure mode this
architecture's severity-hierarchy design already avoids.

**B2 — static's BLOCK authority, permanently non-unilateral — DECIDED
2026-09-18.** Today's item-2 BLOCK cap is *interim*: `decide()`'s docstring
lists three removal criteria (feature-parity test passing, residual skew
closed, real-world re-validation with no confirmed-benign file at/above
`STATIC_BLOCK_MIN`) that, once ALL met, were designed to let static regain
autonomous `BLOCK`. This changes that destination: static moves toward
**never** regaining unilateral `BLOCK` authority again, even after its
eventual retrain.

**Mechanism (maintainer's mechanism (ii), confirmed):** static `BLOCK`
continues to demote to `ALERT` by default, exactly as it does today — that
default does not change. It escalates to `FinalDecision.BLOCK` **only** when
corroborated. **`corroborated` is defined precisely as:** `memory_verdict ==
MemoryVerdict.MALICIOUS OR network_verdict == NetworkVerdict.MALICIOUS`,
evaluated independently of static, on the same scan. Behavioral `MALICIOUS`
needs no special-casing in this mechanism at all — it already sits at rung 1
and produces `TERMINATE` unconditionally, which outranks `BLOCK` in the
`FinalDecision` ordering regardless of what static or this new rung does.

**Critical constraint (to be written verbatim into `decide()`'s docstring at
implementation time):** corroboration unlocks *only* static's own `BLOCK`
verdict. It must never let memory's or network's own authority escalate past
their existing `ALERT` cap. Concretely: a scan where memory is `MALICIOUS`
and static is merely `ALLOW`/`ALERT` (not `BLOCK`) still resolves to `ALERT`
via memory's own rung, exactly as today — memory does not borrow static's
`BLOCK` tier just because they happen to co-occur, and there is no path in
this mechanism by which memory's or network's own driving rung can produce
anything other than `ALERT`. The escalation to `BLOCK` happens on *static's*
rung, using memory/network's verdicts only as corroborating evidence for
*static's* verdict — not the reverse.

**Implementation-time correctness note (recorded now so it isn't
rediscovered as a bug later):** `decide()`'s current priority chain is
strict first-match: rung 2 (memory `MALICIOUS` → `ALERT`) and rung 3
(network `MALICIOUS` → `ALERT`) both fire, and `return`, *before* rung 4
(static `ALERT`/`BLOCK`) is ever reached. As written today, a scan with
static `BLOCK` AND memory `MALICIOUS` never actually reaches the static
rung — `decide()` returns at rung 2 first. So corroboration cannot be
"checked at rung 4" as a local addition to the existing static branch; the
corroboration condition (`static_verdict == BLOCK and (memory ==
MALICIOUS or network == MALICIOUS)`) must be computed up front — the same
"compute before the priority chain, independent of which rung fires" pattern
already used for the ERROR reason codes at the top of `decide()` today — and
consulted *before* (or folded into) the memory/network rungs, so a
corroborated static `BLOCK` produces `FinalDecision.BLOCK` rather than being
pre-empted by memory's or network's own `ALERT` returning first. This is an
implementation detail of *how* to realize the mechanism above, not a change
to the mechanism itself.

**Stack, don't replace (confirmed):** corroboration is *additive* to item
2's existing removal criteria (thrember parity + skew closure + real-world
validation) — it is not a substitute path to trusting static's `BLOCK`
verdict in general. Reason: corroboration only helps when another signal is
actually available and positive at scan time. A static-only scan — no API
trace supplied (behavioral `NOT_PROVIDED`/`PENDING`), no memory or network
features supplied (both `NOT_PROVIDED`) — gets **zero** benefit from this
mechanism: `corroborated` is false by construction whenever memory and
network are both absent, so static `BLOCK` demotes to `ALERT` exactly as it
does today, with no escalation path at all. Item 2's three removal criteria
remain the *only* route to trusting static's `BLOCK` when it is acting
alone, which is also the most common case in practice (most files are
scanned with static evidence only). Corroboration and the removal criteria
solve different problems — the former lets *additional, independently
positive* evidence unlock `BLOCK` sooner on files where multiple signals
happen to be available and agree; the latter is what would eventually let
`BLOCK` stand on static's evidence alone.

### Next steps

- Addition A: `data/download_nsrl.py` (or equivalent), the standalone
  `verify_trusted_chain()` function, `pipeline.py` step 3b, and tests
  (allowlist hit/miss, chain-verify success/failure/parse-error, and a check
  that memory/network/behavioral are provably unaffected by an allowlist
  `ALLOW`). **Proceeding to implementation scoping now.**
- Addition B1: additive, low-risk, no structural-policy question attached.
  **Proceeding to implementation scoping now**, alongside A.
- Addition B2: mechanism decided (above) but implementation not yet
  requested — the priority-chain restructuring note above should be
  reread at that time; not scoped further this pass.

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
  describes pre-fix behaviour; head/flow already updated. Also now: the
  Cortex-Memory / Cortex-Network sections still quote the pre-2026-09-10
  thresholds, the old `val+test`-combined derivation, and the old held-out
  numbers — supersede with the `cal`-split derivation and the regenerated
  `EVAL_ALL_MODELS_RESULTS.txt` figures (memory 0.0024964628 @ `target_fpr`
  0.01; network 0.6672636218 @ `target_fpr` 0.001). Static's section stays
  as-is until its retrain session.
- `PROJECT_HISTORY_REPORT.md` — uncommitted edits pending.
