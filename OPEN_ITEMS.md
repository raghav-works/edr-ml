# Open items

Running tracker for known-but-deferred work. Blocking review items 1–6 are
done (see git log); this is what remains.

## Senior code review — Phase 1 status (2026-10-06)

Review findings F1–F30 live in `docs/CODE_REVIEW.md`. Its last section,
**"Phase 1 status (2026-10-06)"**, is the current tracker for them. It has
three parts:
- **Status per finding.** Fixed: F2, F3, F12, F13, F17, F21, F24. Partly fixed:
  F4, and F11 (flag only, default unchanged). All others are open.
- **Follow-ups found during Phase 1.**
  - Fixed in `a4dcfa9`: the emulation checkpoint is now pinned to its
    pre-masking forward pass, and empty traces are PENDING for both sequence
    models.
  - Still open: F23; a re-check of the F17 rules; model metadata files that must
    ship with the models; and emulation scoring 1–9-call traces.
- **Decisions waiting on the manager.** The BLOCK path, `ALLOW_UNVERIFIED`, and
  the alert budget and deployment surface.

Final checks: `pytest` 403 + `pytest -m slow` 14 passed, 0 failed, 0 errors;
`scripts.evaluate_all_models` matches the records for all five models
(`reports/phase1_final_eval_2026-10-06.txt`); ONNX parity has 0 flips.

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

### Known inconsistency — behavioral config-gap — RESOLVED 2026-09-18

Memory and network treated "features supplied, no model configured" as
**neutral** (`NOT_PROVIDED`); behavioral instead returned
`BehavioralVerdict.ERROR` when `behavioral_model` or `tokenizer` was `None`,
which routed to `NEEDS_REVIEW` — the same operator mistake (supplying
evidence for a signal whose model was never wired) was silently ignored for
two signals and forced a review queue entry for the third. This asymmetry
predated items 9/10 and was deliberately left in place at the time to keep
item 10 scoped.

**Decided: behavioral becomes neutral, matching memory/network** — not the
reverse. `_run_behavioral`'s no-model-or-tokenizer branch now returns
`BehavioralVerdict.NOT_PROVIDED` instead of `ERROR` (the actual-exception
branch is untouched and still returns `ERROR`, matching memory/network's own
exception handling); `scan()` now also calls `_record_signal_health` for
behavioral (it never did before this fix), so a behavioral config gap is
visible via `signal_health["behavioral"] = "model_not_configured"` the same
way memory/network already were. Behavioral's *uncapped* authority when it
does run (`MALICIOUS` -> `TERMINATE`, no ALERT cap) is unaffected — this only
changes what "not deployed yet" means, not what "deployed and positive"
means. Covered by
`tests/test_pipeline.py::test_behavioral_trace_without_model_is_visible_but_neutral`,
mirroring the existing memory/network item-10 tests.

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

### Dependency pinning — RESOLVED 2026-09-18

The PDF's item 6 also says "pin dependency versions". Only `signify` had an
upper bound; `pefile` and `scikit-learn` (whose `FeatureHasher` hashing is
parity-critical) were lower-bound only. This was originally deferred to item
7's `thrember` parity cluster since tightening `scikit-learn` needs a
`FeatureHasher` hash-stability re-check — but that re-check doesn't actually
depend on `thrember`/`signify` availability, so it was done standalone
instead of waiting on item 7's unrelated `signify` version conflict:

- **`scikit-learn>=1.4,<2.0`** — the hash-stability re-check found no risk:
  `FeatureHasher` (the exact call shapes `pe_features.py` uses) produces
  byte-identical output across 1.4.0..1.7.2, verified directly in isolated
  venvs, not assumed. The upper bound is precautionary for the untested next
  major line, not a known incompatibility.
- **`pefile>=2023.2.7,<2025`** — a real, different finding: pefile 2024.8.26
  (current latest, no newer release since Aug 2024) added an unconditional
  `gc.collect()` to `PE.close()`, which `pe_features.py` calls on every scan
  reaching static feature extraction. Confirmed by reading pefile's own
  source (not assumed) and independently corroborated
  (`erocarrera/pefile#420`, `pyinstaller#8762`). Measured directly against
  this repo's own fixture: ~1.04x overhead per scan (~18.2ms vs ~17.5ms) --
  real but modest at this process's object-graph size, smaller than the
  externally-reported regression (measured in a much larger, longer-lived
  object graph). Pinned with an upper bound and the finding documented in
  `requirements.txt` so a future bump is a deliberate re-check, same posture
  as the existing `signify` pin.

Full requirements.txt reasoning lives as comments next to each pin, not just
here.

## Structural

- **Feature-parity harness for `pe_features.py`** — MVP DONE
  (`tests/test_static_feature_parity.py`, item 7): vector contract,
  determinism, signed-binary authenticode, ExportsInfo count slot, and
  adapter-passthrough parity, on committed synthetic PE fixtures. Two
  sub-items were identified; one is now closed:
  - **Real EMBER2024 record schema check — DONE, 2026-09-22, one-off
    investigation (not an automated test — see below for why).** The
    adapter assumes an EMBER2024 record's group dicts (`record["general"]`,
    `record["header"]["coff"]`, ...) have the exact keys/shape that
    `pe_features.raw_features()` produces; the MVP above only proves the
    adapter faithfully processes a dict `raw_features` itself made, not a
    real HF record. Verified directly: pulled 140 real records from
    `joyce8/EMBER2024` via the same `hf_hub_download` streaming method
    `data/download_ember2024.py` uses (100 `Win32_test.zip`, 20
    `Win64_test.zip`, 20 `Dot_Net_test.zip` — all three PE-container
    formats the training pipeline actually consumes), saved outside the
    repo tree, no parquet regeneration. Tested two independent ways: (1)
    ran every record through `features.ember2024_adapter.record_to_vector()`
    with the `cortex.features.pe` logger instrumented to catch
    `PEFeatureExtractor.process_raw_features()`'s silent per-group
    zero-fill-on-exception path (`pe_features.py:846–851`) — a naive
    "does it crash?" check would miss this, since that path never raises,
    it silently degrades; the instrumentation itself was verified working
    by deliberately deleting a real key (`header.coff.timestamp`) from a
    copy of one record and confirming the expected `KeyError` and
    zero-fill warning fired; (2) an exhaustive key-set diff of every real
    record's `general`, `strings`, `header.coff`, `header.optional`,
    `header.dos`, `section`, `authenticode`, and `datadirectories[0]`/
    `[i]` keys against exactly what `pe_features.py`'s corresponding
    `process_raw_features()` reads. Result across all 140 records, all
    three formats: 0 exceptions, 0 shape mismatches, 0 non-finite values,
    0 zero-fill warnings, 0 missing keys. One harmless finding:
    `header.optional` carries an extra real key, `base_of_data`, that the
    code correctly never reads (dead-weight field, not a bug). This also
    closes two narrower residual questions flagged in code comments: (a)
    `pe_features.py`'s `ExportsInfo` TODO about whether EMBER2024's
    `exports` field is really a list of symbol-name strings — confirmed
    directly (5 non-empty real examples, all plain string lists); (b) the
    `authenticode` group's field types on genuinely signed real records —
    confirmed directly (5 signed examples, `num_certs>0`, all 8 fields
    present with correct numeric types). **This does NOT close the other
    sub-item below** (the `thrember` skew-quantification cross-check) —
    that remains open and blocked by the signify version conflict; this
    investigation did not touch it.
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

**What "frozen val/test" means for static, recorded before the static
session touches anything (2026-09-18):** memory/network's `val`/`test`
parquets are frozen **byte-identical** — nothing about their generation
changed, so a SHA-256 of the files themselves matches pre- and post-retrain.
Static cannot use that same literal test, because step (a) of this session
regenerates the EMBER2024 parquets specifically to land item 8's
`ExportsInfo` fix — which changes the *feature values* for every row,
`val`/`test` included, by design. For static, "frozen" instead means: the
same **row identity** (the deduped SHA-256 set, produced by the same
deterministic first-occurrence dedup order over an unchanged local HF
cache) stays in `val`/`test` throughout, and that split is never read by
calibration or threshold-derivation (`cal` only) until everything is
frozen and `test` is read exactly once at the end for reported numbers.
What changes deliberately is the feature vector under each row, not which
rows are in the split. This is proven, not assumed, by comparing the
regenerated `ember2024_test.parquet`'s `sha256` column (as a set) against
the pre-regeneration file's, not by comparing file bytes.

**Static retrain OOM investigation, in progress (2026-09-21) — see
correction below before trusting any max_bin number.** Three real
`scripts.train_static` attempts on the new `cal`-carved split all failed
during `lgb.Dataset.construct()` (2568 features), not during boosting.
Fixed so far, each verified: (1) `lgb.Dataset(..., params={"two_round":
True})` never inherited `DEFAULT_PARAMS["n_jobs"]` — construction ran with
this machine's full uncapped thread count; now passes `num_threads`
explicitly. **Real-data Attempt 3** (task `bkjigfpc4`/`bwu5jrb3z`, PID
80149, `python -m scripts.train_static` on the actual EMBER2024 train
parquet, properly-fixed v3 watchdog) tested fix (1) alone and shows it is
**not sufficient by itself**: `proc_rss_kb` climbed 19.31 GB → 36.58 GB
over 123 seconds, still rising, before the watchdog killed it (raw lines
in `mem_watchdog_train_v3.log`). (2) `lgb.Dataset()` is lazy (`ds.data is
X_train` stays true until `.construct()` actually runs, confirmed via
`sys.getrefcount()`) — an earlier `del X_train, X_val` placed before
`lgb.train()` was a no-op, since construction hadn't happened yet; fixed
by calling `.construct()` explicitly and deleting each raw array
immediately after its own Dataset is constructed, before the next one is
built. **Fix (2) has only been verified mechanistically on small
synthetic arrays (confirms del actually frees memory once construct() has
run) — it has never been tested against real data at real scale, alone or
combined with fix (1).** The real-scale test that would have answered
that question used the contaminated synthetic generator (see correction
below) and is invalid — so whether fixes (1)+(2) together are sufficient
at real scale is still genuinely open, not "neither fix was sufficient"
as an earlier version of this note incorrectly implied by citing the
invalidated synthetic reading as if it were that test.

**Correction (2026-09-21): the two real-scale `max_bin` readings (255 →
"31.7 GB", 127 → "35.8 GB") are INVALID, not evidence either way.** Both
test scripts generated synthetic data via the legacy `np.random.random(
(1_872_000, 2568)).astype(np.float32)`, which always returns float64 --
the `.astype(np.float32)` conversion is a second step. At this scale the
float64 temporary alone is 35.82 GiB (`1_872_000 * 2568 * 8 / 1024**3`,
confirmed by direct computation), large enough to trigger the watchdog by
itself, before `lgb.Dataset.construct()` was ever reached. The `max_bin=
127` run's log shows only `"baseline: 165 MB"` -- the next print
(`"after allocating X_train"`) never appears, and the process exited 143
(SIGTERM), consistent with being killed mid-generation. **`max_bin` is
therefore UNRESOLVED, not ruled out**, and needs re-testing with
float32-native generation (`Generator.random(shape, dtype=np.float32)`,
chunked if needed) before any conclusion is drawn.

**What remains valid from this investigation (measured at a smaller,
uncontaminated 500,000-row scale, where the generator's float64 temporary
is freed by Python's refcounting before the next statement samples RSS):**
- Thread count does not meaningfully affect construction overhead:
  8,055 MB (`num_threads=6`) vs 8,038 MB (`num_threads=2`) -- ruling out
  further thread-count tuning as a lever.
- Constructing a `Dataset` from a pre-built `save_binary()` file needs far
  less memory than constructing from an in-memory array at the same
  scale: 1,237 MB vs 8,055 MB (~6.5x). Not directly applicable yet --
  building that binary file still requires the expensive in-memory
  construction once -- but confirms the *binned* representation itself is
  cheap; the cost is in binning from a raw array.
- LightGBM's own construction warning states `two_round` only helps when
  loading directly from a text file, not from an in-memory array --
  meaning it may never have reduced peak memory for this codebase's
  in-memory-array construction path. The follow-up text-file experiment
  was abandoned (a naive per-row Python writer was far too slow to
  time-box, and a real text file at 2,568 columns would run ~35-45 GB
  on disk) -- deprioritized in favor of investigating `lgb.Sequence`
  (LightGBM's streaming/batched construction API, reads from a source
  like a memmap instead of a fully-materialized array) as the next lead.

Hard constraints for the remainder of this investigation: no system swap
changes, no `num_leaves`/learning-hyperparameter changes, no full real
training run, until a verified, time-boxed fix is agreed. `max_bin` and
`bin_construct_sample_cnt` are measured and reported, never applied to the
real pipeline without explicit sign-off.

### Step 1 — loader audit (2026-09-21): the loader is clean — MEASURED

Instrumented `scripts/train_static.py::_load_train_val` with temporary
RssAnon markers (removed after), run on the real train parquet under the
v3 watchdog. **PASS** — `X_train`/`X_val` are float32 (not float64),
C-contiguous (not Fortran-ordered), and RssAnon right before any
`lgb.Dataset` call (21,353 MB) is within ~3.5% of the theoretical minimum
for holding both raw arrays once (20,634.7 MB) — not a double-hold. The
one plausible double-copy line found while reading the code
(`X_train[tr_mask]`, boolean-mask fancy indexing) does not fire on real
data: the real `ember2024_train.parquet` has zero `label == -1` rows
(MEASURED: 1,170,000/1,170,000 exact split), confirmed by marker(a) ==
marker(b) (21,595 MB both) showing the mask-filter step allocated nothing.
Conclusion: the growth happens inside `lgb.train()`, not the loader.

### Step 2 — scaling/lever investigation (2026-09-21)

Full real-data (not synthetic) scaling curves at 250K/500K/1M rows,
production Dataset/booster params, RssAnon via a 20ms internal poller
thread + the v3 watchdog for the 1M runs. Real EMBER2024 data is ~30%
non-zero (MEASURED: `nonzero_frac` 0.2997–0.3019 across all three sizes)
— a real, substantive difference from the earlier invalidated
100%-non-zero synthetic tests, alongside the RssAnon-vs-plain-RSS fix.

**Numpy path (production params, MEASURED):**

| N_ROWS | (a) baseline MB | (b) peak construct MB | (b) peak-above-baseline MB | (c) steady after free MB | (d) peak train(2 rounds) MB |
|---|---|---|---|---|---|
| 250,000 | 3,626.2 | 5,920.9 | 2,294.7 | 3,164.6 | 4,231.9 |
| 500,000 | 6,012.1 | 8,583.5 | 2,571.4 | 3,685.5 | 6,658.5 |
| 1,000,000 | 10,898.9 | 14,628.1 | 3,729.1 | 4,832.0 | 10,908.3 |

**`lgb.Sequence` path (disk-backed `.npy`, `mmap_mode="r"`, production
params, MEASURED):**

| N_ROWS | (a) baseline MB | (b) peak construct MB | (b) peak-above-baseline MB | (c) steady MB | (d) peak train(2 rounds) MB |
|---|---|---|---|---|---|
| 500,000 | 896.1 | 8,740.5 | 7,844.4 | 5,863.7 | 7,298.1 |
| 1,000,000 | 907.7 | 8,751.2 | 7,843.5 | 6,894.8 | 11,306.3 |

**MEASURED finding:** Sequence-path peak-above-baseline is essentially
flat between 500K and 1M rows (7,844.4 → 7,843.5 MB) — consistent with
LightGBM's default `bin_construct_sample_cnt=200,000` (a fixed row count,
required as float64 per a directly-reproduced error:
`ValueError('sample_data[0] type float32 is not double')`) dominating the
construct-phase cost, not the actual data volume.

**`bin_construct_sample_cnt=50,000` vs default 200,000, Sequence path, 500K
rows only, MEASURED, NOT applied anywhere:** construct peak 8,740.5 MB →
2,826.5 MB, a 5,914 MB (~68%) reduction. LightGBM printed
`"Using too small bin_construct_sample_cnt may encounter unexpected errors
and poor accuracy"` — a real accuracy tradeoff, needs explicit sign-off
before ever being applied.

**`max_bin=127` vs production default 255, numpy path, 500K rows,
MEASURED with a corrected float32-native generator:** peak-above-baseline
2,563.0 MB vs 2,571.4 MB — an 8.4 MB (0.3%) difference. Confirms the
prediction that 1-byte-per-bin-index storage makes bin count nearly
irrelevant to construct-time memory; **`max_bin` is not a useful lever**
(this supersedes the two earlier invalidated real-scale readings, which
used a contaminated float64-producing generator).

**Extrapolation to the real split (1,872,000 train / 234,000 val),
INFERRED from linear fits on the above (thin: 3 points numpy, 2 points
Sequence):** numpy-path peak during `train_set.construct()` ≈ 26,035 MB
(≈25.4 GiB); Sequence-path construct peak ≈ 8,752 MB (≈8.55 GiB). Danger
line at time of estimate: 15% of 38.72 GiB total = 5.81 GiB available;
~30.66 GiB of headroom given other processes' usage at that moment (this
drifts with other processes' load and is not a fixed number).

### Step 3 — root cause found: a lingering caller-side reference — MEASURED

Read-only audit (Step 3-0) of `scripts/train_static.py::main()` and
`models/static_lgbm.py::train()` found that `main()`'s own
`X_train`/`X_val` local bindings stay alive for `train()`'s **entire**
call — `del X_train` inside `train()` only removes `train()`'s own local
name, which has no effect on a separate binding of the same name in the
caller's frame. This means the earlier `.construct()`-then-`del` fix
(Step 2 precursor) never actually freed anything when called through
`main()` — both raw arrays (~20.6 GiB combined) stayed resident through
both `Dataset()` constructs *and* the full boosting call. **INFERRED**:
this plausibly explains Attempt 3's 36.6 GiB peak (raw ~20.6 GiB never
freed + binned ~5.4 GiB + boosting overhead ~11.9 GiB ≈ 38 GiB, close to
the 36.6 GiB kill) — that arithmetic draws on the Step 2 fits, not a
direct measurement of Attempt 3 itself.

**Step A, MEASURED on this exact interpreter (Python 3.10.12):** tested
three calling patterns with `weakref.ref()` + RssAnon. The originally
proposed one-line fix (`callee(*producer())`) does **not** work — weakref
stays alive, RssAnon drop = 0.0 MB, identical to the buggy pattern. Only a
holder-dict pattern (caller passes a dict, callee `.pop()`s each array out
before its own `del`) reliably frees the array mid-call: weakref dies,
RssAnon drop = 76.3 MB, matching the test array's exact size. CPython's
evaluation stack retains a reference to a call's arguments for the call's
full duration regardless of calling convention (named variable or
unpacked temporary) — only removing the array from an intermediate
container the callee empties avoids this.

**Fix implemented:** new `models/static_lgbm.py::train_from_holder(holder,
...)` pops `X_train`/`y_train`/`X_val`/`y_val` out of a plain dict instead
of receiving them as positional arguments; `train()` becomes a thin
backward-compatible wrapper (unchanged signature, still used by any future
small-scale caller) that just builds a holder and delegates.
`scripts/train_static.py::main()` now builds its holder directly from
`_load_train_val()`'s return tuple via `dict(zip(...))`, never binding
`X_train`/`X_val` to a name of its own. New regression test
`tests/test_static_lgbm_train.py::test_train_from_holder_frees_raw_arrays`
asserts both weakrefs are dead after the call — proven, not assumed, to
catch a reversion (Step A's mechanism is identical to what this test
checks).

**Step C, MEASURED (20,000-row real slice, `n_estimators=2`):** OLD
calling pattern run twice (`model_hash=1cda50972cacd74b`,
`best_iteration=2`, identical both times — baseline is deterministic) vs
NEW pattern (`model_hash=1cda50972cacd74b`, `best_iteration=2`) —
identical. No behavior change from the fix.

**Step D, MEASURED (500,000-row real slice, `n_estimators=2`, 450K
train / 50K val):**

| | OLD (plain bound names) | NEW (holder pattern) |
|---|---|---|
| RssAnon after train construct | 8,241.1 MB | 8,219.9 MB |
| RssAnon at next construct call (= after del+gc.collect()) | 8,241.1 MB (**0.0 MB drop**) | 3,811.7 MB (**4,408.2 MB drop**, expected 4,408.3 MB for this split's 450K-row train) |
| lgb.train peak | 10,771.5 MB | 6,025.3 MB |
| Final RssAnon | 8,138.5 MB | 3,101.5 MB |
| `model_hash` / `best_iteration` | `cd2205801c55e20c` / 2 | `cd2205801c55e20c` / 2 (identical) |

The OLD pattern's exact **zero** drop at the freed-array checkpoint is
direct proof of the bug; the NEW pattern's 4,408.2 MB drop (matching the
450K-row `X_train`'s expected 4,408.3 MB almost exactly) is direct proof
of the fix, with identical `model_hash`/`best_iteration` confirming no
behavior change at this larger scale either.

**Follow-ups (2026-09-21), MEASURED:** a negative-control scratchpad check
(same weakref assertions as the regression test, called through `train()`
with the arrays bound to caller names) confirmed both weakrefs stay
**alive** — the opposite of the real test's result — proving the test's
assertions genuinely discriminate between the two patterns rather than
being a tautology. `train()`'s wrapper docstring corrected: it previously
claimed to behave "identically to the pre-refactor train()", which was
imprecise — committed HEAD's `train()` took a `calibrate: bool = True`
kwarg and returned a calibrated model; this session's earlier
calibrate-later redesign (train()/calibrate() split, prior to today's
holder-pattern fix) already removed that. The docstring now states
`train()` is unchanged only relative to the calibrate-later version that
immediately preceded today's fix, not relative to committed HEAD.

### Full-scale (1,872,000 train / 234,000 val) dry run — MEASURED, 2026-09-21

Real `scripts.train_static.main()`, real split, `n_estimators=2` (the only
behavior-changing monkeypatch — everything else was a transparent
before/peak/after/wall-time logging wrapper), outputs to scratchpad, run
under the corrected v3 watchdog. **Completed successfully, no watchdog
ALERT, minimum available memory during the whole run was 10.10 GiB**
(never approached the 5.81 GiB kill line). Confirmed via git status and
file mtimes: nothing under `data/models/cortex_static*` or `config/`
changed.

| Phase | wall_s | RssAnon before | RssAnon peak | RssAnon after |
|---|---|---|---|---|
| `_load_train_val` | 59.799 | 91.9 MB | 22,022.5 MB | 21,352.0 MB |
| `lgb.Dataset.construct` (train) | 53.032 | 21,352.0 MB | **27,752.1 MB** | 27,754.4 MB |
| *(gap: `del X_train; gc.collect()`)* | | | | **→ 9,416.0 MB** |
| `lgb.Dataset.construct` (val) | 7.278 | 9,416.0 MB | 9,434.9 MB | 9,434.9 MB |
| `lgb.train` (2 rounds) | 28.526 | 7,142.8 MB | **19,837.0 MB** | 7,981.8 MB |
| `_load_cal` | 19.109 | 7,981.8 MB | 12,642.8 MB | 10,372.2 MB |
| `calibrate` | 0.563 | 10,372.2 MB | 10,372.2 MB | 10,372.2 MB |
| process end (after `main()` + `gc.collect()`) | — | — | — | 8,080.0 MB |

**Drop when the raw train array is freed: 27,754.4 → 9,416.0 MB =
18,338.4 MB (17.91 GiB)** — matches the INFERRED expectation
(`1,872,000×2568×4/1024²` = 18,340.1 MB = 17.91 GiB) almost exactly (off
by 1.7 MB, 0.009%). The fix holds at full real scale, not just at the
500K-row scale Step D measured.

**Result vs. the stated PASS bars:**

| Bar | Threshold | Measured | Status |
|---|---|---|---|
| Freed-array drop | ~17–18 GiB | 17.91 GiB | clears |
| RssAnon after train construct+free | ≤~11 GiB | 9.20 GiB | clears |
| Boosting-phase peak | ≤~20 GiB | 19.37 GiB | clears |
| **Overall process peak** | **at or below ~27 GiB** | **27.10 GiB** | **exceeds by ~100 MB (0.4%)** |

3 of 4 bars clear; the overall-peak bar is marginally exceeded. Per
instruction, this is reported as NOT a clean PASS rather than rounded
into one — flagged for discussion, not treated as blocking on my own
judgment given how small the margin is relative to the ~30 GiB of
headroom that was never approached.

### Real full-scale training run — completed 2026-09-21, MEASURED

Real `scripts.train_static.main()`, full 3000-round budget (production
`DEFAULT_PARAMS`, unchanged), the real split (1,872,000 train / 234,000
val, `cal` loaded after training), launched detached (`setsid`+`nohup`),
monitored by the memory logger and v3 watchdog for its entire run.
**Completed cleanly — 0 watchdog ALERT lines for the whole run.**

```
INFO:cortex.scripts.train_static:train=1872000 val=234000 features=2568 ...
INFO:cortex.static.train:Training done in 6040.8s, best_iteration=3000
INFO:cortex.scripts.train_static:cal=234000
[3000]  train's auc: 0.999999  val's auc: 0.999283
Did not meet early stopping. Best iteration is:
[3000]  train's auc: 0.999999  val's auc: 0.999283
```

- **Wall time: 6,040.8 s = 100.68 min.** `best_iteration=3000` — ran the
  full budget without early stopping, same pattern as the deployed model
  (2996/3000).
- **Seconds/round, MEASURED: 6040.8 / 3000 ≈ 2.01 s/round.** This
  *corrects* the earlier dry-run-based estimate of ~14.3 s/round, which
  was INFERRED from a 2-round test — that figure was dominated by fixed
  per-call startup cost (Dataset re-validation, first-round histogram
  setup) that doesn't repeat per round; it was never a valid basis for
  extrapolating to thousands of rounds, and is now superseded by this
  direct measurement.
- **Memory story:** dry-run construct peak 27.10 GiB (MEASURED, from the
  earlier dry run). Real run's sampled maximum was 25.72 GiB
  (`proc_rss_kb=26967196` at 07:32:44Z, watchdog's 20s interval) —
  **INFERRED lower only due to 20–30s sampling being unable to see a
  peak briefer than the sampling interval**, not evidence the real peak
  was actually lower. Boosting itself ran flat at ~9.1–9.5 GiB for the
  entire ~90+ minutes of sustained computation (confirmed via a 60s
  per-thread CPU-time snapshot: 6 threads at 97–100% of one core each,
  matching `n_jobs=6`) — notably, boosting never approached the dry run's
  19.37 GiB `lgb.train` peak at any sampled point, a real but
  INFERRED-explanation difference (plausibly a one-time setup cost tied
  to the dry run's specific 2-round call, not confirmed). A brief bump to
  10.09 GiB appeared during `cal` loading near the end, then the process
  exited cleanly.
- **Artifacts:** `data/models/cortex_static_retrain.lgbm`
  (`acfa5a757aa13cd0708c00ca97f445ed86ab54d9a4e80505c7fd733157b02bea`,
  78,558,257 bytes) + `.meta`
  (`abef687e8ce6c7523918d85bbd0352bac5633502a1be9aadc1e45f2682b630bc`,
  786 bytes) — saved to `~/cortex_static_retrain_run/retrain_sha256.txt`.
  Deployed `data/models/cortex_static.{lgbm,meta,onnx}` re-verified
  unchanged against the `~/cortex_static_backup_20260921` backup (exact
  sha256 match on all three) throughout.
- **Calibrator sanity, chronology corrected 2026-09-21 (MEASURED, via git
  log, not the earlier imprecise version of this note):** new model's
  Platt calibrator (`coef_≈0.991, intercept_≈0.139`) is essentially the
  same shape/magnitude as the deployed model's (`coef_≈1.004,
  intercept_≈0.087`) — both far from the historical saturated-calibrator
  pattern (`coef_=10.66, intercept_=-5.13`) that `PROJECT_HISTORY_REPORT.md`
  documents. The earlier version of this note said that pattern "predates
  the currently-deployed model" -- imprecise. The exact chronology: the
  deployed **booster** (`.lgbm`) was trained Aug 18, 14:22:54 (commit
  `6cd4131`), *before* the raw-margin calibrator fix. That fix landed in
  commit `6eed759` ("fit Platt calibrators on raw margins, not
  probabilities") at Sep 8, 07:17:38 -- three weeks later. The deployed
  **`.meta`** (calibrator) was regenerated at Sep 8, 07:22:54, only ~5m16s
  after that fix commit -- a post-hoc recalibration of the *same,
  unchanged* Aug 18 booster's margins using the newly-fixed code, not a
  retrain. Threshold re-derivation (`6e8d50c`) followed ~12 minutes later
  (07:34:51). So: the booster predates the fix by ~3 weeks; the calibrator
  was refit after the fix, before ever being committed. The deployed model
  was never the saturated 10.66/-5.13 calibrator -- that number describes
  a different, already-superseded model version -- so "new vs. deployed"
  here is a same-shape comparison between two already-fixed calibrators,
  not a saturation-fix comparison.
- **Hyperparameter proof:** every `DEFAULT_PARAMS` value (`num_leaves`,
  `max_depth`, `learning_rate`, `min_child_samples`, `subsample`,
  `colsample_bytree`, `reg_alpha`, `reg_lambda`, `min_split_gain`,
  `is_unbalance`, `seed`, `histogram_pool_size`) matches the saved
  booster's own recorded params exactly (via LightGBM's parameter
  aliases) — confirmed by reading `model_to_string()`'s params directly,
  not assumed. `max_bin=255` and `bin_construct_sample_cnt=200000` are
  both LightGBM's own defaults, confirming neither of the two
  measured-only levers from the OOM investigation (`max_bin=127`,
  `bin_construct_sample_cnt=50000`) was ever applied to this run. Proves
  only the memory-lifecycle fix changed — model configuration is
  unchanged from the design.
- **Val AUC caveat, stated plainly:** 0.999283 (new) vs. 0.999319
  (deployed, from `EVAL_ALL_MODELS_RESULTS.txt`), a difference of
  0.000036. **This is not apples-to-apples** — features changed
  (`ExportsInfo` fix), training rows are ~11% fewer (`cal` now carved out
  of what was train), and LightGBM's own per-round val AUC is on raw
  scores while the evaluation harness reports calibrated-probability
  AUC (mathematically identical ranking, same AUC value, but worth
  naming since the two AUCs come from different code paths). The
  difference is INFERRED to be within sampling noise — **unquantified**,
  not proven. The frozen test-set read (once, at the end) is the real
  comparison, not this val figure.

**Cal-vs-test caveat, INFERRED:** the deployed model's own recorded
numbers (`EVAL_ALL_MODELS_RESULTS.txt`, MEASURED) show val AUC 0.999319
vs. test AUC 0.998841 -- test is measurably harder than val for this
model family. If that pattern holds for the retrained model too, the
cal-derived thresholds (also close to val in difficulty, per Step D
below) are INFERRED likely to show a somewhat higher FPR on the frozen
test split than their cal targets -- unquantified until the one real test
read happens.

**Old-model-must-not-be-rescored note:** the deployed model must NOT be
re-scored against the regenerated (`ExportsInfo`-fixed) parquets for an
old-vs-new comparison -- OPEN_ITEMS.md's own "frozen val/test" note
above states the fix changes feature values for every row, so scoring the
old model on new features would be a confounded comparison (different
inputs, not just a different model). Old-vs-new must use the old model's
**already-recorded** numbers (`EVAL_ALL_MODELS_RESULTS.txt`), not a fresh
scoring run.

### Threshold derivation tool: results against the real retrained model (2026-09-21)

`scripts/derive_static_thresholds.py` (new) run against
`data/models/cortex_static_retrain`, cal split only (234,000 rows,
benign=116,862, malicious=117,138) -- MEASURED, raw sweep table with
full-precision (`repr()`) thresholds, each reproduced-and-verified against
the reported FP/TP counts before being printed (an assertion, not a
claim):

| target FPR | full-precision threshold | actual FPR (95% CI) | detection (95% CI) |
|---|---|---|---|
| 0.0001 | 0.9977410259813835 | 0.000094 ([0.000047,0.000168], 11/116862 FP) | 0.8489 ([0.846849,0.850960]) |
| 0.0005 | 0.9811748406902472 | 0.000496 ([0.000377,0.000642], 58/116862 FP) | 0.9311 ([0.929659,0.932568]) |
| 0.0010 | 0.9575215714986189 | 0.000993 ([0.000820,0.001190], 116/116862 FP) | 0.9518 ([0.950593,0.953054]) |
| 0.0050 | 0.7320628018088604 | 0.004997 ([0.004601,0.005418], 584/116862 FP) | 0.9816 ([0.980773,0.982323]) |
| 0.0100 | 0.4789517595186417 | 0.009995 ([0.009432,0.010582], 1168/116862 FP) | 0.9887 ([0.988040,0.989261]) |
| 0.0200 | 0.22744200804171985 | 0.019955 ([0.019161,0.020773], 2332/116862 FP) | 0.9932 ([0.992726,0.993675]) |

Cliffs (jump > 0.1) between 0.001→0.005, 0.005→0.01, and 0.01→0.02 -- the
same cliff pattern the deployed model's own comments document (between
0.005 and 0.01) -- recurring, not new. Thresholds non-increasing as
target FPR grows: confirmed True. Peak RssAnon during the run: 5.17 GiB
(manual ~5s-interval polling, may have missed a brief higher peak),
comfortably under the 6 GiB guard throughout.

**Candidate pair (same target-FPR precedent as the deployed model's own
derivation, not a fresh choice): ALLOW at target 0.01 = `0.4789517595186417`,
BLOCK at target 0.001 = `0.9575215714986189`.** This is a MENU item, not
an applied decision -- `config/thresholds.yaml` is untouched.

**ONNX export (MEASURED):** `data/models/cortex_static_retrain.onnx`,
61,450,626 bytes, sha256
`636c77c5c536bf3ba2996205206c718d481708b39e7bb0007e4b3535874b4944`, wall
time 44.02s, peak RssAnon 1.79 GiB (manual polling).

**ONNX parity on the first 50,000 cal rows, new ONNX file (MEASURED):**
`max_abs_err=3.185e-2`, `mean_abs_err=4.374e-6`, `median=3.741e-8` --
**0 verdict flips at both candidate thresholds**, matching the "0 flips"
precedent from the deployed model's own parity check. Largest errors
concentrate in mid-range score buckets ([0.05,0.5): max 3.19e-2;
[0.5,0.95): max 1.08e-2), the same pattern as the documented pre-existing
float32 TreeEnsemble limitation (roughly 3x the deployed model's ~1e-2
max, same order of magnitude, same concentration). Deep-benign/malicious
buckets show tiny errors (~3-4e-4), as expected far from any boundary.

**Val-only validation of the candidate thresholds (MEASURED, 234,000 val
rows, benign=117,001 malicious=116,999):** val AUC on calibrated
probabilities = 0.999283, exactly matching LightGBM's own internal val
AUC from training (difference -0.000000) -- confirms the Platt calibrator
is monotone, as expected. ALLOW boundary: val FPR=0.010231
([0.009662,0.010824]) vs. cal-derived CI [0.009432,0.010582] -- val point
estimate falls inside the cal CI. BLOCK boundary: val FPR=0.001171
([0.000983,0.001384]) vs. cal-derived CI [0.000820,0.001190] -- val point
estimate falls inside the cal CI (near its edge). Both val CIs overlap
their respective cal CIs substantially. This is validation only -- no
threshold was adjusted based on this result. Val informed early stopping
during training, though `best_iteration` hit the 3000-round cap without
ever triggering early stopping, so val was read but never actually
influenced the final model choice.

**Read-only observation, not acted on (2026-09-21):** the saved booster's
`bagging_fraction=0.8` with `bagging_freq=0` -- confirmed identical in
the deployed model's own saved params. Checked the installed LightGBM
package for `bagging_freq` documentation text: the pip wheel does not
ship `Parameters.rst` (no `.rst` files anywhere in the installed
package); the only occurrence of the string is inside the compiled
`lib_lightgbm.so` binary, as an internal assertion:
`"Check failed: (config->bagging_freq > 0 && config->bagging_fraction <
1.0f && config->bagging_fraction > 0.0f) || (config->feature_fraction <
1.0f && config->feature_fraction > 0.0f) ..."` -- MEASURED, extracted via
`strings` on the actual installed binary, not fabricated prose. **This
string only proves `bagging_freq` is a real, internally-checked
parameter -- it does NOT itself demonstrate that `bagging_freq=0`
disables bagging.** That specific behavioral claim is INFERRED from
general LightGBM library knowledge/community documentation not shipped
with this installed wheel (no `.rst` docs present) -- **unverified,
correctly flagged by the maintainer as INFERRED until directly measured.**

**P5 runtime test (2026-09-21), MEASURED -- not inferred:** trained three
real boosters on the same 20,000-row real slice, same seed, same base
params, `deterministic=True`, `num_boost_round=10`: (A) production values
(`bagging_fraction=0.8, bagging_freq=0`); (B) `bagging_fraction=1.0,
bagging_freq=0`; (C) positive control, `bagging_fraction=0.8,
bagging_freq=1`. Compared `booster.model_to_string()`'s tree-structure
section (everything before the `parameters:` block, which would trivially
differ by the params text itself). **A's trees == B's trees, byte-for-byte
identical** (122,607 characters each, exact string equality `True`) --
changing `bagging_fraction` from 0.8 to 1.0 produced the *exact same
model* when `bagging_freq=0`. **A's trees != C's trees** (122,607 vs
113,155 characters) -- confirms the comparison method genuinely detects
real differences when `bagging_freq` is actually nonzero, so A==B is not
a testing artifact. **Conclusion: `subsample=0.8` (`bagging_fraction`)
is confirmed inactive in both the deployed and retrained models --
MEASURED directly, not inferred from documentation or a binary string.**
**Nothing changed here** -- flagged for a future, separately-discussed
change with before/after accuracy numbers, per instruction.

### Chosen candidate thresholds: full parity + val checks, and pre-registration (2026-09-21)

**P3, full 234,000-row cal ONNX parity (MEASURED) -- the 50,000-row
sample's "0 flips" did not hold at full scale:** `max_abs_err=3.571906e-02
mean_abs_err=2.979517e-06`. 4 flips total, all within thousandths of
their threshold: ALLOW (`0.4789517595186417`) 1 flip (benign row 78357,
Python allows/ONNX alerts -- not safety-degrading); BLOCK@0.0005
(`0.9811748406902472`) 1 flip (malicious row 145478, ONNX more
aggressive); BLOCK@0.001 (`0.9575215714986189`) 2 flips (malicious rows
111253/130382, one each direction). **Correction (2026-09-21):** an
earlier draft of this note called the BLOCK@0.001 pair "symmetric, not a
directional bias" -- that is an over-read from n=2; two events split one
each way is not evidence of symmetry (or of its absence), and the phrase
is retracted.
Consistent with the already-documented, pre-existing float32 TreeEnsemble
ONNX limitation this repo already accepted for network (commit `6e8d50c`:
"3 flips / 213,217") -- not a new problem from this retrain.

**P4, val-only check at the chosen BLOCK threshold (MEASURED, 234,000 val
rows):** `FPR=0.000615 (95% CI [0.000482,0.000775], 72/117001 FP)
detection=0.9326 (95% CI [0.931155,0.934036], 109114/116999 TP)` at
`0.9811748406902472` (target 0.0005). Falls inside the cal-derived CI at
that target (`[0.000377,0.000642]`... actually the val point estimate
0.000615 sits inside that interval), CIs overlap substantially. Not used
to adjust anything.

**P5 bagging test:** see the `bagging_freq` observation above -- MEASURED
directly, `subsample=0.8` confirmed inactive.

**DECISION (user-chosen, 2026-09-21):** ALLOW at cal target FPR 0.01 ->
`0.4789517595186417`. BLOCK at cal target FPR 0.0005 ->
`0.9811748406902472` (deviates from the 0.001 precedent -- see
pre-registration rationale below). Full pre-registration document written
to `~/cortex_static_retrain_run/PREREGISTRATION.txt` before test is ever
read:

```
CORTEX-STATIC RETRAIN: PRE-REGISTRATION, BEFORE TEST IS EVER READ
Timestamp: Mon Sep 21 10:04:18 AM UTC 2026

============================================================
1. CHOSEN THRESHOLDS AND TARGETS
============================================================

  ALLOW (allow_below): target cal FPR 0.01 -> 0.4789517595186417
  BLOCK (block_at_or_above): target cal FPR 0.0005 -> 0.9811748406902472

  Rationale for the 0.0005 BLOCK target (deviating from the deployed
  model's 0.001 precedent): the deployed model's own recorded val-vs-test
  behavior shows honest cal/val-derived FPRs understate FPR on the later
  test split (see prediction arithmetic, section 3). A lower BLOCK target
  is chosen in advance for this reason, not picked after seeing test.

  Artifacts these thresholds belong to (sha256):
    data/models/cortex_static_retrain.lgbm
      acfa5a757aa13cd0708c00ca97f445ed86ab54d9a4e80505c7fd733157b02bea
    data/models/cortex_static_retrain.meta
      abef687e8ce6c7523918d85bbd0352bac5633502a1be9aadc1e45f2682b630bc
    data/models/cortex_static_retrain.onnx
      636c77c5c536bf3ba2996205206c718d481708b39e7bb0007e4b3535874b4944

============================================================
2. ACCEPTANCE GATES FOR THE SINGLE TEST READ
============================================================

  (a) REGRESSION GATE (hard): test AUC on calibrated probabilities
      >= 0.9980. Deployed model's recorded test AUC (EVAL_ALL_MODELS_
      RESULTS.txt): 0.998841.

  (b) No hard gate on FPR. Report FPR and detection at both chosen
      thresholds (ALLOW and BLOCK) with 95% Clopper-Pearson CIs. This is
      read-only reporting, not a pass/fail condition in itself.

============================================================
3. PREDICTIONS (INFERRED, crude one-model ratio, uncertain by
   roughly +-30% -- shown BEFORE test is read)
============================================================

  Deployed model's own recorded numbers (EVAL_ALL_MODELS_RESULTS.txt,
  MEASURED):
    val  @ ALLOW boundary: FP=928,  n_benign=117001  -> FPR=0.007932
    val  @ BLOCK boundary: FP=65,   n_benign=117001  -> FPR=0.000556
    test @ ALLOW boundary: FP=2699, n_benign=269940  -> FPR=0.009999
    test @ BLOCK boundary: FP=269,  n_benign=269940  -> FPR=0.000997

  val-to-test FPR ratios (deployed model):
    ALLOW: 0.009999 / 0.007932 = 1.2606
    BLOCK: 0.000997 / 0.000556 = 1.7937

  New (retrained) model's val FPRs at the chosen thresholds (Step D / P4,
  MEASURED):
    val @ ALLOW (0.4789517595186417):  FPR=0.010231 (1197/117001 FP)
    val @ BLOCK (0.9811748406902472):  FPR=0.000615 (72/117001 FP)

  Applying the deployed model's val-to-test ratios to the new model's val
  FPRs (arithmetic, INFERRED prediction, not a measurement):
    predicted test FPR @ ALLOW = 0.010231 * 1.2606 = 0.012897 (~1.29%)
    predicted test FPR @ BLOCK = 0.000615 * 1.7937 = 0.001103 (~0.110%)

  +-30% uncertainty band on those predictions:
    ALLOW: [0.009028, 0.016766]
    BLOCK: [0.000772, 0.001434]

  This is a crude, one-model-family ratio applied to a different model's
  val numbers -- not a statistically rigorous forecast. It exists so a
  large, surprising deviation on the real test read is visible as a
  deviation from a stated prior, not rationalized after the fact.

============================================================
4. RULES (binding for after the test read)
============================================================

  - No threshold or model change may be made in response to test results
    OTHER THAN an explicit accept/rollback decision by the user.
  - Test is never used to pick between candidate thresholds -- the
    candidates were already chosen (section 1) before test is read.
  - A large drift from the section-3 prediction is DOCUMENTED and
    DISCUSSED, not tuned away by picking a different threshold from test.

============================================================
ADDENDUM 1 (appended Mon Sep 21 10:12:10 AM UTC 2026, before any test read)
============================================================

  - Gate (a) is replaced: test AUC on calibrated probabilities >= 0.9985.
    Reason: expected test AUC is about 0.9988 (new val AUC 0.999283 minus
    the deployed model's recorded val-to-test gap 0.999319 - 0.998841 =
    0.000478). The original 0.9980 floor would tolerate roughly 70% more
    AUC error (1-AUC from about 0.0012 to 0.0020) and was not a
    meaningful regression gate.

  - Outcome rule: AUC >= 0.9985 -> the candidate may be proposed for
    promotion (the user decides). AUC < 0.9985 -> NOT promoted; results
    documented and discussed.

  - Uncertainty note: the +-30% band in section 3 is too narrow for
    BLOCK. Poisson noise alone on the three FP counts involved (65, 269,
    72) gives about +-36% at 95%, before any model-to-model difference;
    for ALLOW (928, 2699, 1197) it is about +-9%. Read the BLOCK
    prediction as 0.110% with a plausible range of about 0.07%-0.15%,
    which straddles the 0.1% architecture bar. No hard FPR gate.

  data/processed/ember2024_test.parquet stat (informational only, atime
  may be unreliable under relatime):
    access: 2026-09-21 03:49:43.673704976 +0000
    modify: 2026-09-18 09:48:57.278474058 +0000
```

### The one authorized test read (2026-09-21, MEASURED)

Run once, via a scratchpad driver (`~/cortex_static_retrain_run/` --
not committed to the repo) that overrode `scripts.evaluate_all_models.
STATIC_MODEL` and `inference.policy_engine.STATIC_ALLOW_MAX` /
`STATIC_BLOCK_MIN` as module attributes at runtime -- no repo file was
edited to do this. A rehearsal (same driver, `_load_ember_test`
substituted with a val-returning function, `pyarrow`/`pandas` parquet
readers wrapped to raise on any path ending in `ember2024_test.parquet`)
was run first and confirmed the override took effect (printed header
showed the candidate path and the exact DECISION thresholds) and that no
attempt was made to open the test parquet. The real run then used the
unmodified `_load_ember_test` -- full output, raw, in
`~/cortex_static_retrain_run/eval_static_retrain_test.txt`
(sha256 `fd8869fe3f59d738bc7b47c9f44fb94ca83f0adf7d1c5614d21b1ecc144c27ab`).
The test parquet was opened exactly 1 time (counted via a wrapped
`pyarrow.parquet.ParquetFile`, printed in the output). Peak RssAnon during
the run: ~18.4 GiB (well under the shared-machine guard), wall time ~3
minutes, `free -k` available memory checked immediately before launch
(37.3 GiB available).

Loaded split sizes matched the pre-registered expectation exactly (no
STOP condition triggered): val n=234,000 (117,001 benign / 116,999
malicious), test n=539,940 (269,940 benign / 270,000 malicious).

**Test-split results (candidate model, DECISION thresholds
ALLOW=`0.4789517595186417` BLOCK=`0.9811748406902472`):**

  3-way confusion:
  ```
  true\pred    ALLOW   ALERT   BLOCK      n
  benign      266666    3001     273  269940
  malicious     4821   18162  247017  270000
  ```
  @ ALLOW boundary: FP=3274 FN=4821 TP=265179 TN=266666 | FPR=0.012129
  (95% Clopper-Pearson CI [0.011719, 0.012549]) | R=0.9821
  @ BLOCK boundary: FP=273 FN=22983 TP=247017 TN=269667 | FPR=0.001011
  (95% CP CI [0.000895, 0.001139]) | R=0.9149
  AUC-ROC (test, calibrated): **0.998778**

**Vs. pre-registration (report only -- no interpretation applied to
change thresholds or the model):**

  | boundary | predicted FPR | band | measured FPR | CI | verdict |
  |---|---|---|---|---|---|
  | ALLOW | 0.012897 | [0.009028, 0.016766] | 0.012129 | [0.011719, 0.012549] | HIT |
  | BLOCK | 0.001103 | [0.0007, 0.0015] | 0.001011 | [0.000895, 0.001139] | HIT |

  AUC 0.998778 >= 0.9985 (ADDENDUM 1 amended gate): **PASS**.
  AUC 0.998778 >= 0.9980 (original gate): **PASS**.
  Both measured FPRs land inside their pre-registered bands -- no large
  deviation to document per the pre-registration's own rule.

**Side-by-side with the deployed model's recorded test numbers**
(`EVAL_ALL_MODELS_RESULTS.txt` lines 46-54):

  | metric | deployed | candidate |
  |---|---|---|
  | ALLOW FPR | 0.009999 [0.009627, 0.010381] | 0.012129 [0.011719, 0.012549] |
  | ALLOW recall | 0.9803 | 0.9821 |
  | BLOCK FPR | 0.000997 [0.000881, 0.001123] | 0.001011 [0.000895, 0.001139] |
  | BLOCK recall | 0.9168 | 0.9149 |
  | AUC | 0.998841 | 0.998778 |

  **Correction (2026-09-21):** an earlier draft of this note attributed the
  candidate's measurably higher ALLOW FPR mainly to the `ExportsInfo`
  feature regeneration. That is at best a partial explanation. The larger
  cause is a methodology difference: `config/thresholds.yaml`'s prior
  static thresholds were derived by running `find_threshold_for_fpr()`
  directly against the calibrated *test*-split probabilities (see
  `config/thresholds.yaml`'s pre-2026-09-21 comment) -- which is why the
  deployed model's recorded test FPRs (0.009999, 0.000997) equal their
  0.01/0.001 targets almost exactly: that agreement is true by
  construction, not an out-of-sample result. The candidate's thresholds
  were derived honestly on `cal` and read against `test` exactly once, so
  the candidate's test FPR was never going to reproduce its own target as
  tightly -- some drift from the val-measured FPR (0.010231; the cal FPR
  the threshold was actually derived from is 0.009995, 1168/116,862) is
  the expected, correct behaviour of an honest derivation, not a symptom
  of the retrain or the feature change. The `ExportsInfo` regeneration
  being a real but secondary factor on top of that is **INFERRED, not
  measured** -- no ablation isolating it from the derivation-methodology
  difference above has been run.

  BLOCK FPR is statistically indistinguishable between the two models
  (overlapping CIs, FP=273 vs 269) -- and separately, the candidate's
  BLOCK 95% CI ([0.000895, 0.001139]) contains the architecture doc's
  0.001 (0.1%) bar, so BLOCK is **at** that bar on held-out data, not
  demonstrably below it; this note previously risked being read as
  "meets" or "is under" the bar, which the CI does not support. ALLOW FPR
  is measurably higher for the candidate (non-overlapping CIs) -- a real
  difference, explained above, not evidence of a bug. AUC is marginally
  lower for the candidate (by 0.000063), well inside both gates. BLOCK
  recall is ~0.2 points lower for the candidate; ALLOW recall is ~0.2
  points higher -- neither is large relative to the derivation-methodology
  and feature-regeneration differences above.

**No threshold or model was changed in response to test results.**
`config/thresholds.yaml`, `data/models/cortex_static.{lgbm,meta,onnx}`
were not touched by this step. Final verification (immediately after the
read, 2026-09-21): deployed artifact hashes unchanged and match the
`~/cortex_static_backup_20260921/` backups exactly (`.lgbm`
`277489bee1b9d0828a5ab117d3498cc5902684bcda6b8d10dc2c4098fef71b34`,
`.meta` `0f668725f9e5768718fb2cb34f5a98da70a02f163c8b1fa4850dd00d6869bfce`,
`.onnx` `a9ae8f485fd9019c5f9c392395c20d6072366ff08b4a9414ed98219a5482a857`);
`git status` shows no repo file changed by this step other than this
`OPEN_ITEMS.md` edit itself; full test suite still green, **165 passed,
0 failed, 0 skipped**. The candidate model (`data/models/
cortex_static_retrain.*`) was NOT promoted -- promotion is a separate,
explicitly authorized action not taken here.

### Promotion (2026-09-21, MEASURED)

User accepted the candidate for deployment after the pre-registered
outcome rule was met (ADDENDUM 1: test AUC >= 0.9985; measured 0.998778 --
PASS), in a separately, explicitly authorized step following the test
read recorded above.

**What changed:** `data/models/cortex_static.{lgbm,meta,onnx}` overwritten
with the candidate (`cp -p`, timestamp `Mon Sep 21 10:31:50 AM UTC 2026`);
post-copy sha256 verified to equal the candidate's recorded hashes exactly
(`.lgbm acfa5a757aa13cd0708c00ca97f445ed86ab54d9a4e80505c7fd733157b02bea`,
`.meta abef687e8ce6c7523918d85bbd0352bac5633502a1be9aadc1e45f2682b630bc`,
`.onnx 636c77c5c536bf3ba2996205206c718d481708b39e7bb0007e4b3535874b4944`).
`config/thresholds.yaml`'s `static:` block updated to `allow_below
0.4789517595186417` / `block_at_or_above 0.9811748406902472` plus a
rewritten static.* comment (methodology, model/threshold pairing, test
results, the corrected causal explanation above, and the BLOCK-cap note
below) -- verified via `yaml.safe_load()` deep-compare against the
pre-promotion backup copy that these were the **only 2 of 8 top-level
threshold keys** that changed; the diff is recorded by the
`config(thresholds)` commit of this work. Fresh-process assertions after
the edit:
`inference.policy_engine.STATIC_ALLOW_MAX ==
0.4789517595186417`, `STATIC_BLOCK_MIN == 0.9811748406902472`,
`LGBMModel.load("data/models/cortex_static").model_hash ==
"acfa5a757aa13cd0"`, `num_trees() == 3000` -- all passed.

**How to roll back:** `bash ~/cortex_static_backup_20260921/rollback.sh`
(real mode, no `--dry-run`) restores `cortex_static.{lgbm,meta,onnx}` and
`config/thresholds.yaml` from the pre-promotion backup and re-verifies
their hashes; a `--dry-run` pass was re-confirmed immediately before this
promotion (exit 0, no processes had the deployed files open per `lsof`).

**Post-promotion smoke test (cal split only, 234,000 rows, test not
reopened):** reused `scripts.train_static._load_cal` and
`scripts.verify_onnx_parity._onnx_run`/`_report` by import, against the
now-deployed `data/models/cortex_static{,.onnx}` paths -- did not call
`check_static()`/`main()` from that module, which would have opened
`ember2024_test.parquet`. Result: `max_abs_err=3.571906e-02
mean_abs_err=2.979517e-06`, 1 flip at ALLOW, 1 flip at BLOCK -- matches
the earlier full-scale cal parity check exactly, no regression from
promotion. Cal 3-way confusion (deployed thresholds, Python path, n=234,000):
benign 115694/1110/58 (ALLOW/ALERT/BLOCK), malicious 1328/6740/109070.

**Full test suite after promotion:** `165 passed, 0 failed, 0 skipped` --
unchanged from pre-promotion.

**Optional real-world sanity check (Step G, read-only, non-gating):** 5
local validation files named in `PROJECT_HISTORY_REPORT.md`
(`svchost.exe`, `notepad_test.exe`, `benign_test_50mb.exe`,
`notepadd.exe`, `extractor.exe`) found on this machine. Features extracted
once per file via the exact call `inference/pipeline.py` uses
(`features.pe_features.PEFeatureExtractor().feature_vector_with_report(
bytez)`, static parsing only, no file executed), then scored with both the
OLD model (`~/cortex_static_backup_20260921/cortex_static`) and the NEW
deployed model on the identical feature vector:

  | file | sha256[:16] | old score | new score | old verdict | new verdict |
  |---|---|---|---|---|---|
  | svchost.exe | 75772da68f23bee2 | 0.877460 | 0.942081 | ALERT | ALERT |
  | notepad_test.exe | ab15a95de88ab062 | 0.005642 | 0.023364 | ALLOW | ALLOW |
  | benign_test_50mb.exe | fce08e3382e46073 | 0.980467 | 0.984642 | BLOCK | BLOCK |
  | notepadd.exe | d38ba15bd8df9bdd | 0.935731 | 0.960786 | ALERT | ALERT |
  | extractor.exe | 71506a193d361743 | 0.989866 | 0.993514 | BLOCK | BLOCK |

  All 5 files are documented as **benign** at `PROJECT_HISTORY_REPORT.md:
  218-222` (svchost.exe: benign, signed MS; notepad_test.exe: benign;
  benign_test_50mb.exe: benign; notepadd.exe: benign; extractor.exe:
  benign, PyInstaller). Two of them -- `benign_test_50mb.exe` (new score
  0.984642) and `extractor.exe` (new score 0.993514) -- score at or above
  the new BLOCK threshold (`0.9811748406902472`); both were also at or
  above the OLD BLOCK threshold (`0.9798998555119341`) before this
  retrain. All 5 verdicts are identical old vs. new model (no verdict
  changed); no degraded feature groups on any file. All five scores rose:
  svchost.exe +0.0646, notepad_test.exe +0.0177, benign_test_50mb.exe
  +0.0042, notepadd.exe +0.0251, extractor.exe +0.0036. The cause of that
  uniform increase was **not investigated** in this step -- **INFERRED**:
  a combination of a different booster and a different calibrator, not
  verified by any ablation isolating one from the other. Five files carry
  no statistical weight -- report only, nothing acted on. This retrain
  does not address the known real-world false-positive / feature-fidelity
  problem (open item 7, `pe_features.py`-vs-thrember skew) that these two
  BLOCK-scoring benign files are consistent with -- which is exactly why
  the static BLOCK cap (item 2's interim corroboration gate) stays in
  place, unchanged by this promotion.

**Stale documentation references -- current state (updated across several
follow-up sessions; as of this commit):**

  **Fixed:** `ARCHITECTURE.md`'s threshold table (both its 3-line
  ALLOW/ALERT/BLOCK band and its split-protection table's Static row are
  updated to the current values and the train/val/cal carve).
  `EVAL_ALL_MODELS_RESULTS.txt` received a one-line SUPERSEDED note
  directly above its static section -- the section's own numbers are
  untouched (see the paragraph below for why). `inference/policy_engine.py`'s
  static-threshold comment is updated (comment lines only; no code
  changed -- see this record's "Promotion" addenda).

  **Still stale, NOT edited this session (out of this session's allowed
  scope):** `README.md` (as of this commit: line 110 still shows
  `--test data/processed/ember2024_test.parquet` in the `train_static`
  example, and line 113 still names the removed `_split_xy` function --
  the script has neither any more) and `PROJECT_HISTORY_REPORT.md` (as of
  this commit: line 208 still quotes the pre-2026-09-08
  pre-calibration-fix pair `0.6163460957`/`0.9950119117`) -- both files
  currently hold **unrelated, uncommitted local edits by the maintainer**
  (`git status --short` shows both `M`), so editing them here risked
  colliding with in-progress work; `docs/Cortex_Pipeline_Report.html`
  (lines 422-423, 531-532, 1301-1302, same pre-2026-09-08 pair) -- the
  matching `docs/Cortex_Pipeline_Report.pdf` is **unverified**, because
  `pdftotext` is not installed on this machine and `strings` cannot see
  compressed PDF text streams. A ready patch for the README
  `train_static` example (and the `_split_xy` sentence) is at
  `~/cortex_static_followups/README_static_train.patch`, built and
  dry-run-verified against both HEAD's and the current working-tree
  README in a follow-up session; it is NOT applied to the repo.

**`EVAL_ALL_MODELS_RESULTS.txt`'s static section is now superseded**, not
regenerated: its numbers are untouched; a one-line SUPERSEDED note was
added above the static section (re-running `scripts/evaluate_all_models.py`
against `--only static` would reopen `ember2024_test.parquet`, spending
the one authorized test read a second time -- that is why the numbers
themselves were left alone). The current, authoritative
static test numbers live in
`reports/static_retrain_20260921/eval_static_retrain_test.txt` and this
file's "The one authorized test read" section above.

**Static BLOCK verdicts remain capped to ALERT** by the interim
corroboration-gated policy (`inference/policy_engine.py::decide()`, item 2)
-- unchanged by this promotion.

**Durable record:** `reports/static_retrain_20260921/` (new, committed in
the `docs(OPEN_ITEMS)` commit of this work) holds `PREREGISTRATION.txt`,
`eval_static_retrain_test.txt`, `threshold_report_v2.txt`,
`retrain_sha256.txt`, `MANIFEST.sha256` (sha256 of the four preceding
files), and `DELIVERY_MANIFEST.txt` (artifact hashes/sizes, full-precision
thresholds and their pairing, the ONNX input/output contract read live
from `onnxruntime`, verdict semantics, and the known ONNX-vs-Python
behaviour -- for the agent/endpoint integration team).

**Commits:** the code for this work landed as two local commits:
`8192452` (static split-discipline retrain + memory-lifecycle fix +
regression test) and `48f55f0` (derive-thresholds tool + its tests).
Worktree tests against the COMMITTED content (not working-tree overlays)
measured **157** passed after `8192452` and **162** passed after
`48f55f0`. The `config(thresholds)` and `docs(OPEN_ITEMS)` commits follow
this edit. To undo the whole series before any push:
`git reset --mixed 388d6df` (leaves the working tree exactly as it is,
just unstages/uncommits; nothing here was ever pushed).

**Note on this section's provenance:** this file's static-retrain
material (the "retrain cluster" section starting above and running
through this Promotion record) also contains one note written in an
earlier, separate 2026-09-18 session and left uncommitted when this
session's static-retrain work began -- the bold-labelled paragraph
`**What "frozen val/test" means for static, recorded before the static
session touches anything (2026-09-18):**` (not a `###` heading; no
`###`-level heading in this section carries that date). It predates and
is unrelated to the split-discipline retrain itself but was already
sitting in this same working-tree file.

### Session bookkeeping (not part of the promotion record)

**Baseline test count for this session (2026-09-18): 159 passed, 0 failed,
0 skipped** — re-verified at the start of this session, not carried
forward from an earlier note. A task brief for this session stated an
expected baseline of "157 passed"; that number does not appear anywhere in
this file's git history and was traced to a stale, never-re-verified
estimate rather than a real prior measurement. The actual figures at each
point since the "155 (post-B1+B2) passed" line above (verified true at
commit `ac132d2` by checking it out into an isolated worktree with real
model/parquet artifacts present): +1 at `4f45630` (one new test,
`test_behavioral_trace_without_model_is_visible_but_neutral`), unchanged
at `388d6df` (docs/requirements only) → 156 from committed code, +3 from
`tests/test_behavioral_model.py` (untracked, pre-existing separate
work-in-progress, not part of any commit) → **159** observed today.
Going forward in this repo, an "N passed" figure anywhere in this file is
an **informational snapshot of when it was written, not a hard assertion
to match** — the real gate at any point is 0 failed / 0 errors when
actually run, not agreement with a previously-recorded digit, since the
suite's size legitimately changes commit to commit.

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
script — `_load_train_val_split` (since renamed to `_load_train_val`) does
the same with `rng.permutation(n_rows)`
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

**Status (2026-09-18): DONE — Addition A, B1, and B2 all implemented, tested,
and committed** (see "Addition A — DONE" and "Addition B — DONE" below for
the full write-ups; commits `383edf1`, `3749989`, `3449503`, `b6efb22`,
`ac132d2`, in that order). Two additions, agreed in this order, motivated by a
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

### Addition B — corroboration-aware decision logic (after A) — DONE

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

#### Addition B — implementation summary (2026-09-18)

Built exactly as designed above, in two commits:

- **B1** (`b6efb22`) — `decide()` computes `corroborating_signals` up front
  (the same "before the priority chain, independent of which rung fires"
  pattern as the existing ERROR-reason collection): `static` counted when
  `static_verdict in (ALERT, BLOCK)`, `memory`/`network` counted when
  `MALICIOUS`. `len(...) >= 2` appends `corroborated_multi_signal` to
  `reasons` at all four malicious/suspicious return points (including the
  `TERMINATE` one, for audit completeness — corroboration cannot change
  that outcome but is still useful information there). Purely additive to
  `reason_codes`; behavioral is deliberately excluded from the set, since
  it already has independent uncapped authority and needs neither to
  corroborate nor be corroborated.
- **B2** (`ac132d2`) — one new rung reusing B1's `corroborating_signals`
  directly: `if static_verdict == StaticVerdict.BLOCK and
  len(corroborating_signals) >= 2: return FinalDecision.BLOCK, [...]`,
  inserted immediately after the behavioral-`TERMINATE` check and
  **before** the memory/network `ALERT` rungs. That placement is required,
  not stylistic: `decide()` returns on first match, and memory/network's
  own rungs would otherwise silently pre-empt a corroborated static
  `BLOCK` by returning `ALERT` first — exactly the bug the "implementation
  correctness note" above was recorded to prevent.

**Critical constraint — proven, not just implemented:** the escalation
rung is gated on `static_verdict == StaticVerdict.BLOCK`, so memory/network
`MALICIOUS` with static anything else falls through unchanged to their own
`ALERT`-only rungs. `tests/test_policy_engine.py::
test_memory_or_network_alone_can_never_reach_block` checks this across the
**entire** verdict cartesian product (`_ALL`, every combination of all four
signals), not hand-picked cases — there is no combination anywhere in the
state space where memory or network reaches `BLOCK`/`TERMINATE` on their
own. Companion load-bearing tests: `test_memory_corroborates_static_block`,
`test_network_corroborates_static_block`,
`test_both_memory_and_network_corroborate_static_block_once`,
`test_behavioral_terminate_still_outranks_corroborated_block`,
`test_static_alert_never_escalates_to_block_even_when_corroborated`, and
`test_uncorroborated_block_still_demotes_to_alert` (the static-only,
zero-benefit case, including the exact `(NOT_PROVIDED, NOT_PROVIDED)`
no-other-signals-supplied row).

**Pre-existing tests that had to change, not just new ones added:** four
tests encoded the pre-B2 "`decide()` never returns `BLOCK`" invariant
across the full verdict product and would have silently masked a real
regression if left alone: `test_decide_never_returns_block` (rewritten as
`test_decide_returns_block_iff_static_block_is_corroborated`),
`test_network_alone_never_escalates_past_alert`,
`test_memory_alone_never_escalates_past_alert`, and two `CASES`
truth-table rows. One B1-era test
(`test_two_signals_corroborate_without_changing_outcome`) had its
static-BLOCK-plus-network case swapped for a static-ALERT one, since that
specific combination is exactly what B2 now intentionally changes.

Full suite green throughout: 148 (post-A) → 155 (post-B1+B2) passed, zero
regressions.

### Next steps

None remaining for this tracked item — Addition A, B1, and B2 are complete.
Two things intentionally NOT done here, both flagged above as deliberate:
emulation stays excluded from all of this pending its own retrain/
re-validation, and item 2's original removal criteria (thrember parity +
skew closure + real-world validation) remain the only path to static
regaining trust when acting alone with no corroboration available — this
work does not touch or shortcut those criteria.

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

## Known-but-accepted limitations

- **Behavioral sentinel-token dependence — INVESTIGATED, accepted 2026-09-22, no fix applied.** `data/download_behavioral.py`'s vocabulary-check docstring incorrectly claimed the Cuckoo sentinel tokens `__anomaly__`/`__exception__` were absent from MalbehavD-V1/Carpenter (corrected in that file); token-ablation on the held-out test split found aggregate model dependence on these tokens is low (not systemic shortcut-learning) but two individual test rows show real per-row dependence, including one live false positive. See README.md's "Known limitation (behavioral sentinel tokens)" section for the full investigation and numbers.
