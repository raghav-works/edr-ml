# cortex-ml — Static (LightGBM) + Behavioral (1D-CNN + Attention) reproduction

Independent reimplementation, built to the architecture you specified
(`ARCHITECTURE.md`): **sequential gating, no ensembling**. Static runs
first; a static **ALLOW, ALERT, or BLOCK** verdict lets a file proceed to
behavioral analysis (only a static scan ERROR skips it). Static's BLOCK is
currently interim-capped to a final ALERT (see `policy_engine.decide()`'s
"INTERIM CAP"). Scores are never averaged — the policy engine combines
*verdicts* by fixed priority rules.

```
path validation → PE validation → 2568-dim feature extraction → LightGBM
        → static verdict (ALLOW / ALERT / BLOCK)
        → [ALLOW / ALERT / BLOCK] → API-call tokenization → 1D-CNN+Attention
              → behavioral verdict (BENIGN / MALICIOUS / PENDING)
                (PENDING if < 10 API calls; a 10–99 call verdict is emitted
                 but flagged "behavioral_short_trace")
        → policy engine → final decision
              (ALLOW / NEEDS_REVIEW / ALERT / BLOCK / TERMINATE)
        → structured JSON security event
```

A file that **cannot be analyzed** (non-PE / missing / unreadable / oversized,
or an exception during feature extraction or scoring) with no malicious or
suspicious signal from any other channel resolves to **`NEEDS_REVIEW`**, not
`ALERT` (review item 9): "could not analyze" is not a malware finding. A signal
that *did* complete with a finding still wins — `decide(static=ERROR,
behavioral=MALICIOUS)` is `TERMINATE` — and the failed-signal reason code is
kept in `reason_codes` regardless.

Model/analyzer health is reported on `ScanResult.signal_health`, kept
**separate from the security verdict** (review item 10): a configured
memory/network model that raises at runtime is flagged `"model_error"` (and
routes to `NEEDS_REVIEW` via its `ERROR` verdict); memory/network features
supplied with no model wired are flagged `"model_not_configured"` **without**
changing the decision — a not-yet-deployed signal is made visible, not
escalated.

A feature group that fails extraction is listed in `ScanResult.degraded_groups`
instead of silently becoming a zero vector (review item 6). A degraded
**critical** group (`features.pe_features.CRITICAL_FEATURE_GROUPS`) sets
`static_verdict = ERROR` → `NEEDS_REVIEW` (reason `static_features_degraded`),
because an all-zero group can fabricate maliciousness (an empty import table)
or erase it (dropped IOC strings); a non-critical group only sets
`signal_health["static"] = "degraded"`. `CortexPipeline()` runs a startup
self-test against a bundled signed PE and raises if a critical group is broken
(pass `self_test=False` to skip).

## Layout
```
cortex/
├── features/
│   ├── pe_features.py             # 2568-dim EMBER2024-compatible extractor (pefile-based)
│   └── memory_features.py        # 7 derived ratio features + train-only-fit scaler over VolMemLyzer's 55
├── models/
│   ├── static_lgbm.py            # LightGBM train/eval/calibrate/persist
│   ├── memory_lgbm.py            # LightGBM train/eval/calibrate/persist (mirrors static_lgbm.py)
│   ├── network_lgbm.py           # LightGBM train/eval/calibrate/persist (mirrors memory_lgbm.py)
│   ├── behavioral_cnn.py         # 1D-CNN + multi-head attention, single sigmoid score
│   ├── train_behavioral.py       # training loop (AdamW, cosine restarts, early stop)
│   ├── emulation_cnn.py          # Cortex-Emulation net (mirrors behavioral_cnn.py; embed_dim=64, seq 500)
│   └── train_emulation.py        # training loop + eval + threshold sweep (val-AUC early stop, AUC-gap overfit monitor)
├── tokenizer/
│   ├── api_tokenizer.py          # API-name → token-ID, fixed length 100, <PAD>/<UNK>
│   └── emulation_tokenizer.py    # same design, fixed length 500, vocab 3,154
├── data/
│   ├── download_ember2024.py     # HF download + de-dup (see note below)
│   ├── download_memory.py        # CIC-MalMem-2022 CSV loader + schema validation (manual download)
│   └── download_network.py       # CSE-CIC-IDS2018 loader: schema align, cleaning, memory-safe stratified sampling
├── export/
│   ├── export_onnx.py            # LightGBM→ONNX (onnxmltools) for static/memory/network, PyTorch→ONNX for behavioral
│   └── quantize.py               # dynamic INT8 quantization + FP32/INT8 comparison + latency bench
├── inference/
│   ├── policy_engine.py          # exact threshold + priority-rule decision logic
│   └── pipeline.py               # end-to-end scan() matching the architecture flow
├── scripts/
│   ├── train_static.py
│   ├── train_behavioral.py
│   ├── split_memory.py           # group-aware, leakage-checked memory-dataset split
│   ├── train_memory.py           # trains, evaluates, sweeps threshold candidates, saves
│   ├── split_network.py          # ambiguous-group exclusion, stratified split, leakage check
│   ├── train_network.py          # trains, single-feature-AUC check, per-attack-type breakdown, saves
│   ├── split_emulation.py        # module_entry restriction, per-era duplicate collapse, leakage check
│   └── train_emulation.py        # trains, threshold sweep (pooled + era-split), per-family recall, baselines
└── config/thresholds.yaml
```

## EMBER2024 duplication issue
You noted the downloaded EMBER2024 PE dataset comes out roughly **2x** the
expected row count. `data/download_ember2024.py` handles this: it
de-duplicates on `sha256` (falling back to a hash of the raw feature vector
if no id column exists) immediately after download, logs the before/after
row counts, and flags the case explicitly if the drop is close to 50% —
which is the signature of "every sample present twice." Always run this
before anything else touches the data; don't dedup after merging with other
sources, since that can hide the same issue inside a bigger table.

```bash
python -m data.download_ember2024 --split train --out data/processed/ember2024_train.parquet
python -m data.download_ember2024 --split test  --out data/processed/ember2024_test.parquet
```

## Training

**Static:**
```bash
python -m scripts.train_static \
  --train data/processed/ember2024_train.parquet \
  --test  data/processed/ember2024_test.parquet \
  --out   data/models/cortex_static
```
`scripts/train_static.py::_split_xy` assumes feature columns are named
`feature_0..feature_2567` (or plain digit strings) and a `label` column with
EMBER's convention (`0`=benign, `1`=malicious, `-1`=unlabeled, dropped
automatically). Adjust column detection if the HF parquet schema differs —
check `df.columns` after downloading before your first real run.

**Behavioral:** needs a dataset with `api_calls` (list[str]) + `label`
(0/1) columns — build this from whatever API-trace corpus you're pairing
with the CNN (Mal-API-2019 / MalbehavD-V1 / your own captures).
```bash
python -m scripts.train_behavioral \
  --data data/processed/behavioral_dataset.parquet \
  --vocab-out data/models/api_vocab.json \
  --checkpoint-out data/models/cortex_behavioral_best.pt
```

## Export + quantize
```python
from export.export_onnx import export_static_lgbm_to_onnx, export_behavioral_to_onnx
from export.quantize import quantize_behavioral_int8, compare_accuracy, benchmark_latency

# Takes the base model path (LGBMModel.save()'s <path>.lgbm + <path>.meta),
# not a bare .lgbm path -- the calibrator lives in the .meta file and gets
# chained into the ONNX graph itself (see the export_static_lgbm_to_onnx
# docstring): the exported model's one output IS the Platt-calibrated
# probability, matching LGBMModel.predict_proba() exactly. Exporting just
# the raw booster would silently diverge from the calibrated distribution
# the thresholds above were derived against (raw vs. calibrated differ by
# up to ~0.21 on real test data -- the raw booster saturates hard at
# 0.0/1.0 for many samples).
export_static_lgbm_to_onnx("data/models/cortex_static", "data/models/cortex_static.onnx")
# static model: NO INT8 quantization (tree ensemble — no effect on split thresholds)

from export.export_onnx import export_memory_lgbm_to_onnx
export_memory_lgbm_to_onnx("data/models/cortex_memory", "data/models/cortex_memory.onnx")
# memory model: NO INT8 quantization, same reason as static (tree ensemble).
# Verified against the real cortex_memory model and the actual 5,930-row
# CIC-MalMem-2022 test split (2026-08-25): mean abs error 6.7e-9, max abs
# error 2.0e-6 vs. MemoryLGBMModel.predict_proba(), zero prediction
# mismatches at MEMORY_MALICIOUS_MIN across all 5,930 test rows.

from export.export_onnx import export_network_lgbm_to_onnx
export_network_lgbm_to_onnx("data/models/cortex_network", "data/models/cortex_network.onnx")
# network model: NO INT8 quantization, same reason as static/memory.
# Verified against the real cortex_network model and the actual 213,217-row
# CSE-CIC-IDS2018 test split (2026-08-25): mean abs error 2.1e-4, max abs
# error 0.5465 -- three orders of magnitude looser than static's/memory's,
# root-caused to onnxmltools' hard float32-only input constraint for
# LightGBM interacting with network's much larger feature dynamic range.
# Practical impact is small (3/213,217 prediction flips at the operating
# threshold, aggregate FPR/detection within noise) but NOT resolved -- see
# the Cortex-Network section's "ONNX export" subsection before treating
# this file as fully interchangeable with the Python model.

export_behavioral_to_onnx(model, "data/models/cortex_behavioral.onnx")
quantize_behavioral_int8("data/models/cortex_behavioral.onnx", "data/models/cortex_behavioral_int8.onnx")
compare_accuracy("data/models/cortex_behavioral.onnx", "data/models/cortex_behavioral_int8.onnx")
benchmark_latency("data/models/cortex_behavioral_int8.onnx")
```

## Thresholds
Hard-coded in `inference/policy_engine.py` (mirrored in
`config/thresholds.yaml`), re-derived against the models actually trained in
this repo (not carried over from the prior project's placeholder values):
- static score `< 0.6163460957` → ALLOW, `< 0.9950119117` → ALERT, else BLOCK
  (`models.static_lgbm.find_threshold_for_fpr()` against `cortex_static`'s
  calibrated test-set probabilities; BLOCK hits the architecture doc's
  max_fpr@0.1% bar, ALLOW uses target_fpr=0.01 -- see the code comments in
  `policy_engine.py`/`thresholds.yaml` for the full derivation).
- behavioral score `>= 0.60` → MALICIOUS, else BENIGN (a threshold sweep
  against `cortex_behavioral_best.pt`'s val+test scores, not a precise FPR
  target -- 274 benign samples can't support one. See the known-limitation
  note below.)
- memory score `>= 0.0005358335957155212` → MALICIOUS, else BENIGN
  (`models.memory_lgbm.find_threshold_for_fpr()` at target_fpr=0.01 against
  `cortex_memory`'s calibrated val+test-combined probabilities, 5,860
  benign samples -- see "Training results and the separability finding"
  above for the full sweep table and why this specific target was chosen).
  Unlike static/behavioral, memory's MALICIOUS verdict is capped at ALERT
  in the policy engine regardless of score, not wired to BLOCK/TERMINATE --
  see the "Known limitation (memory authority)" note above.
- network score `>= 0.9441855970306654` → MALICIOUS, else BENIGN
  (`find_threshold_for_fpr()` at target_fpr=0.001 on `cortex_network`'s
  calibrated val+test-combined probabilities). Also capped at ALERT — see
  the Cortex-Network section.
- emulation score `>= 0.999358594417572` → MALICIOUS, else BENIGN
  (target_fpr=1% sweep point against `cortex_emulation_best.pt`).
  **Logging/telemetry only** — Cortex-Emulation is not in `decide()` at all
  (report-only / additive). See the Cortex-Emulation section for the
  ablation-confirmed concept-drift collapse behind that decision.

If you retrain any model from scratch, the raw score distribution will
differ and these need to be re-derived again, not assumed to still hold.

**Known limitation (behavioral threshold):** across the full val+test sweep,
exactly one benign sample is misclassified at every threshold below ~0.922 --
a MalbehavD-V1 sample whose trace includes networking-setup calls
(`setsockopt`, `ioctlsocket`, `wsastartup`, `getsockname`) alongside routine
registry/system calls, plausibly confusable with C2 setup. The model scores
it 0.921006, confidently wrong, not a wobbly near-threshold case. We
deliberately kept the threshold at 0.60 rather than raising it past 0.922 to
clear this one case: n=1 evidence isn't a sound basis for a permanent recall
tradeoff across the whole malicious population, especially since behavioral
has no downstream backstop in this repo yet (no rule-based overlay for
high-risk call combinations independent of the ML score). The more targeted
fix is either more real benign network-adjacent software traces in future
training data, or the architecture research's rule-based overlay for
high-risk call combinations -- not blanket threshold tuning in response to a
single hard example.

## Cortex-Memory
Third signal, added independently of static/behavioral: LightGBM binary
classifier over 55 VolMemLyzer-derived memory-forensics features plus 7
derived features (CIC-MalMem-2022, Carrier et al., ICISSP 2022). Evaluated
as an **independent** signal in the policy engine, not gated behind
static's ALLOW like behavioral is — its purpose is catching
injected/fileless malicious activity that static structurally cannot see
(there is no malicious file on disk for static to score in the first
place), so gating it behind static would defeat that purpose. See
`inference/policy_engine.py::decide()`'s docstring for the full reasoning.

**Known limitation (memory authority — revisit before relying on this in
production):** memory's authority is deliberately capped at ALERT — it can
escalate a scan's final decision, but it cannot autonomously TERMINATE/BLOCK
the way behavioral can, even at MALICIOUS confidence. This mirrors the same
caution already documented above for behavioral's own threshold choice:
Cortex-Memory is trained and threshold-derived against CIC-MalMem-2022 only,
a fixed lab-collected memory-dump corpus, and has not been validated against
real injected-process samples outside that test set. Extending its authority
to TERMINATE/BLOCK should wait until real-world injected-process validation
evidence exists — don't lift this cap based on CIC-MalMem-2022 test-set
metrics alone. **The training results below are a specific, diagnosed
reason to keep this cap, not just a generic caveat** — see "Training
results and the separability finding" further down.

Layout: `data/download_memory.py` (schema-validated CSV loader),
`scripts/split_memory.py` (group-aware, leakage-checked split),
`features/memory_features.py` (7 derived ratio/aggregate features + a
train-only-fit scaler, see "Feature engineering" below),
`models/memory_lgbm.py` + `scripts/train_memory.py` (LightGBM trainer with
Platt calibration, mirroring `models/static_lgbm.py`'s shape). ONNX export
is not built yet.

**Data provenance:** the working copy (`data/raw/memory/`, gitignored, not
committed) was pulled from the Kaggle mirror
`luccagodoy/obfuscated-malware-memory-2022-cic` on 2026-08-25, not the
official UNB/CIC portal — at the time of this pull, the official portal's
download was missing 3 of the dataset's 4 top-level categories.

**Real split results (2026-08-25, seed=42, val/test=10%/10%):**
train=46,736 (benign=23,438 / malware=23,298), val=5,930 (2,930 / 3,000),
test=5,930 (2,930 / 3,000). All 15 malware families proportionally
represented in every split. Zero sample-group leakage (0/32,106 groups span
more than one split) and zero exact-duplicate-feature-vector leakage (0
hashes span more than one split, verified against all 58,596 rows).

**Known limitation (benign near-duplicates — not fully solved):**
`Category` for every benign row is the literal constant string `"Benign"`
— there is no per-sample identifier anywhere in the raw schema for benign
captures, unlike malicious rows (which encode `<Type>-<Family>-<sha256
hash>-<dump#>.raw`, letting `scripts/split_memory.py` group a malware
sample's ~10 dumps together). The split script's
`verify_zero_duplicate_feature_hashes` check does catch and fix *exact*
duplicate benign feature vectors landing across splits (confirmed present
in the real data: 8 groups / 55 rows, now merged into single groups before
splitting — see `_assign_merged_groups`'s docstring in
`scripts/split_memory.py`). It does **not** catch *near*-duplicates — two
benign captures that are extremely similar but not byte-identical (e.g.
repeated snapshots of the same idle baseline system state with minor
timing-driven variation) can still land across train and test, since
nothing in this dataset's schema lets us tell "two genuinely independent
benign captures" apart from "two captures of the same underlying system
state." This is an open item, not a solved problem — flagging it here
rather than assuming grouped splitting on the malicious side implies the
same protection exists for benign.

**Also observed (informational, not a leakage risk):** 11 of the exact
duplicate-feature-vector groups found in the real data involve *different*
malware samples (different Category, different sha256 hash) that happen to
produce byte-identical 55-feature vectors — most likely because
VolMemLyzer's features are coarse count/ratio summary statistics with
limited cardinality, not a data-pipeline artifact (no cross-label
duplicates were found; every duplicate group is 100% benign or 100%
malicious). `_assign_merged_groups` unions these into shared groups the
same way, so they don't span splits either, but it's worth knowing the
feature space has real collisions independent of sample identity — a
signal worth revisiting once feature engineering starts.

### Feature engineering
`features/memory_features.py` adds 7 derived ratio/aggregate features on
top of the 55 raw VolMemLyzer columns (62 total), each justified by a
specific memory-forensics indicator rather than picked to hit a round
number: `callbacks_anonymous_ratio`, `svcscan_driver_ratio`,
`handles_file_ratio`, `handles_mutant_ratio`, `psxview_hiding_score`,
`ldrmodules_hidden_ratio`, `malfind_injection_rate` — see the module
docstring for the one-line justification behind each. Handles-per-process
and DLL-per-process ratios were deliberately **not** added as derived
features: VolMemLyzer already provides both directly
(`handles.avg_handles_per_proc`, `dlllist.avg_dlls_per_proc`), so
duplicating them would pad the feature count without adding information.

`MemoryFeatureScaler` standardizes the full 62-column matrix, fit on the
train split's mean/std only (never val/test, avoiding the same leakage
`scripts/split_memory.py`'s vocabulary/pos_weight discipline avoids
elsewhere). It is built and tested, but **not applied by the LightGBM
trainer**: tree-ensemble splits (and therefore predictions) are provably
invariant to any monotonic per-feature transform, including
standardization, so scaling changes every feature value but not one split
decision. Applying it here would add a second fitted artifact -- needing
its own place in the eventual ONNX export -- for zero effect on the model.
It stays available in `features/memory_features.py` for a future non-tree
Cortex-Memory variant, where scaling would actually matter.

### Training results and the separability finding
`models/memory_lgbm.py` (LightGBM + Platt calibration, mirroring
`models/static_lgbm.py`'s shape exactly: `PlattCalibrator`,
`MetricsReport`, `find_threshold_for_fpr()`) + `scripts/train_memory.py`.
Trained on `memory_train` only; `memory_val` drives early stopping (best
iteration 71) and Platt calibration; `memory_test` untouched until final
evaluation. Saved to `data/models/cortex_memory.lgbm` / `.meta`
(gitignored, like the other trained models).

**Threshold derivation (2026-08-25, seed=42):** benign counts — val=2,930,
test=2,930, **combined=5,860** — are far healthier than behavioral's 274;
one false positive on the combined set moves the observed FPR by only
~0.017%. Swept on val+test combined (11,860 rows):

| target FPR | threshold | actual FPR | benign FPs | detection rate |
|---|---|---|---|---|
| 0.10% | 0.100225 | 0.085% | 5 / 5,860 | 99.95% |
| 0.50% | 0.000627 | 0.478% | 28 / 5,860 | 100.00% |
| **1.00%** | **0.000536** | **0.939%** | **55 / 5,860** | **100.00%** |
| 2.00% | 0.000502 | 1.724% | 101 / 5,860 | 100.00% |
| 5.00% | 0.000446 | 4.744% | 278 / 5,860 | 100.00% |

Chosen: **target_fpr=1% → MEMORY_MALICIOUS_MIN=0.0005358335957155212**
(`inference/policy_engine.py`) — the best-supported target among those
tried (55 observed FPs, vs. only 5 at 0.1% — too few to trust) that also
costs nothing in recall, since detection is already 100% at this and every
looser target tried. **Test-set metrics at this threshold:** AUC-ROC=1.0,
precision=0.9891, recall=1.0, F1=0.9945, FPR=1.13% (33/2,930 benign test
rows), detection rate=100%.

**These numbers are too clean to accept at face value, and were checked,
not just reported.** Feature importances showed one derived feature
(`svcscan_driver_ratio`) with 5.8x the gain of the next-best feature —
reason enough to suspect it on its own. Checking single-feature AUC for
every one of the 62 raw+derived columns (not just the top-gain one) found
**22 of 62 individually exceed 0.95 AUC**, confirmed independently on both
the train split and the held-out test split (never touched during
training) — e.g. on train, `handles.avg_handles_per_proc` is a tight
208–318 band for benign (std=17.5) but ranges 71–33,784 for malware
(std=222.8); `dlllist.avg_dlls_per_proc` is 34.5–53.2 for benign but drops
as low as 7.3 for malware. Pervasive multi-feature separability, present
identically on data the model never trained on, rules out our own
train/test split leaking (already independently verified: 0/32,106 groups
and 0/58,596 duplicate feature hashes span splits) — this is a property of
the dataset itself.

This is **consistent with CIC-MalMem-2022's own documented single-VM
benign collection methodology**: every benign sample is a repeated capture
of "normal user behavior" on one baseline Windows 10 VM, while malicious
samples span far more varied executions — exactly the narrow-benign/
wide-malware pattern found above. The model may be learning "does this
look like that one baseline VM" more than "is malicious behavior present,"
a distinction this dataset alone cannot resolve. **This is why memory's
policy-engine authority stays capped at ALERT** (see the "Known limitation
(memory authority)" note above and `MEMORY_MALICIOUS_MIN`'s comment in
`inference/policy_engine.py`) — the near-perfect test metrics are a
specific reason to keep that cap, not evidence it can be loosened. Real
validation would require memory captures from multiple genuinely different
(non-baseline, non-single-VM) benign machines, not more rows from this
same dataset.

## Cortex-Network
Fourth signal: LightGBM binary classifier over CSE-CIC-IDS2018 network-flow
features (Sharafaldin et al., CIC, hosted on AWS Open Data --
`s3://cse-cic-ids2018/`, no account/form required, unlike CIC-MalMem-2022).
Architecture mirrors Cortex-Static/Cortex-Memory exactly (train/val/test
split → Platt calibration → threshold derivation from our own data → ONNX
export with calibration merged into the graph) — deliberately NOT the
autoencoder+classifier hybrid some reference designs use for network
anomaly detection; that's a legitimate future capability, not a starting
point. Training is not built yet; this section covers data acquisition and
the split, both run against the real dataset.

**Layout:** `data/download_network.py` (loads/validates/cleans/samples the
10 already-synced CSVs) + `scripts/split_network.py` (ambiguous-group
exclusion, stratified split, leakage check).

**Real bucket structure** (verified, not assumed from the docs): two
top-level prefixes — `Original Network Traffic and Log data/` (raw PCAPs +
logs, per-day subfolders, NOT used) and `Processed Traffic Data for ML
Algorithms/` (10 CICFlowMeter-extracted CSVs, one per capture day, 6.4 GiB
— this is what's loaded). Sync command:
```bash
aws s3 sync --no-sign-request --region ca-central-1 \
  "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/" \
  data/raw/network/
```

**Real schema issues found and handled** (`data/download_network.py`):
- **Column count is not uniform**: 9 of 10 files have 80 columns; the
  `Thuesday-20-02-2018` file (also the size outlier, ~4 GiB vs ~330 MiB
  typical) has 84 — 4 extra leading identity columns (`Flow ID`, `Src IP`,
  `Src Port`, `Dst IP`). **Dropped, not aligned-and-kept**, for a specific
  reason: CSE-CIC-IDS2018 was captured on a small, fixed testbed (a
  handful of specific attacker/victim machines reused across the whole
  multi-day capture). A model with access to these columns could trivially
  learn "traffic to/from this specific IP is malicious" — memorizing the
  testbed's fixed machines — instead of learning a traffic pattern that
  would transfer to a real network with different machines at different
  addresses. The same class of shortcut-learning risk already documented
  for Cortex-Memory (single-VM benign collection), eliminated at the
  source here instead of just noted.
- **`Timestamp` excluded from the feature set** (kept only as metadata),
  for the same reason: each capture day is (almost) entirely one attack
  scenario, so a raw wall-clock timestamp — or even the derived
  `capture_day` column — would let a model learn "which day this is" as a
  near-perfect label proxy instead of learning the actual traffic pattern.
- **59 rows (of 16,233,002, across 3 files) are literal header-repeats** —
  the header line reappears as a data row. Filtered via an `ALLOWED_LABELS`
  allowlist built by scanning the real Label values across all 10 files
  (16.2M values), not copied from the docs.
- **`Flow Byts/s` / `Flow Pkts/s` contain real NaN/Infinity values**
  (division by zero when Flow Duration=0) — confirmed present on a file
  with no other schema issues, so this is a genuine data property. Rows
  with either dropped, not imputed: a small, known fraction (~0.3-0.6% per
  file), not worth inventing a fill value for.
- **Label taxonomy is inconsistent in casing/wording across files**
  ("DDoS attacks-LOIC-HTTP" vs "DDOS attack-HOIC" / "DDOS attack-LOIC-UDP").
  Sidestepped for the binary target via `label.strip().lower() != "benign"`
  rather than an exact-match allowlist; the raw string is kept in
  `label_raw` so a future multi-class model isn't locked out.

**Memory-safe loading**: the raw dataset (16,233,002 rows) is far larger
than what should be held fully in memory at once, given static_lgbm.py's
own history of an OOM from a similar full-copy-everything pattern. Two-phase
design: a cheap Label-only scan computes an exact per-(capture_day,
label_raw) stratified sampling plan first, then each file is fully loaded,
cleaned, and immediately downsampled to its planned size before the next
file loads — never more than one file's full data in memory at once (the
Tuesday file alone is 4 GiB of CSV text). Strata at or below 15,000 rows
are kept in full (protects rare attack types — e.g. SQL Injection is 87
rows total across the whole dataset); everything else is downsampled
proportionally to a **2,000,000-row target**, chosen to stay safely under
budget with margin for multiple in-memory copies during split/training,
while giving far more statistical power than Cortex-Memory needed for
strong results. **Measured**: peak RSS 22.8 GiB, ~8.5 min wall clock.

**Leakage check — a much bigger finding than Memory's**: run before
finalizing the split, not assumed absent just because network flow data
hadn't been checked before. **495,181 of 2,000,000 rows (24.75%) were
exact duplicates of another row** — 91,221 groups, sizes 2 to 18,376 — far
higher than Memory's 0.9% (517/58,596), consistent with how repetitive
automated attack traffic and simple/idle benign flows are. The largest
group (18,376 rows) spans two different attack types (FTP-BruteForce and
DoS-SlowHTTPTest), which apparently produce a statistically identical
"minimal/degenerate flow" signature in CICFlowMeter's 78-feature space.
Same union-merge approach as Memory (rows sharing a hash become one atomic
group before split allocation) — **verified: 0 duplicate hashes span
multiple splits**.

**Known limitation (CICFlowMeter feature ambiguity — named, not silently
worked around):** of the 91,221 duplicate groups, **4,742 contained both
benign and malicious rows** — the identical 78-feature vector observed
with contradictory ground truth. This is not the cross-split leakage
problem above (nothing here spans splits); it's a property of
CICFlowMeter's feature representation itself: these specific feature
values genuinely do not determine the label. No classifier operating on
these features could resolve them — keeping them in train would teach
contradictory signal for identical inputs, and keeping them in val/test
would penalize any model, including a hypothetically perfect one, for an
unwinnable case. `scripts/split_network.py::exclude_ambiguous_groups()`
removes every row in an affected group from train, val, AND test alike
(not just train), and saves them to
`data/processed/network_excluded_ambiguous.parquet` for inspection rather
than discarding them silently.

**Known limitation (FTP-BruteForce / DoS-SlowHTTPTest train support — a
direct, correct consequence of the leakage guarantee, not a bug):** the
18,376-row duplicate group mentioned above is one indivisible block that
had to land entirely in one split. The stratified allocation put most of
it in val/test, leaving train with very few examples of these two attack
types relative to val/test (see the exact per-split counts below). This
was left as-is deliberately: forcing large duplicate groups toward train
would defeat the leakage guarantee for an arbitrary, non-principled reason
(favoring training-set size over correctness). **When training-report
per-class metrics eventually cover these two classes, "hard to detect"
must not be presented without the accompanying "and hard to learn from,
given how few train examples exist"** — the two are entangled here, not
independent findings.

**Real split results (seed=42, val/test=10%/10%):** starting from the
2,000,000-row canonical dataset, 48,329 rows (2.42%) across 4,742
ambiguous groups were excluded per the limitation above, leaving
**1,951,671 rows** to split:

| split | rows | % (target) | benign | malicious |
|---|---|---|---|---|
| train | 1,526,757 | 78.23% (80%) | 1,286,894 | 239,863 |
| val | 211,697 | 10.85% (10%) | 160,864 | 50,833 |
| test | 213,217 | 10.92% (10%) | 160,869 | 52,348 |

All 27 (day, attack-type) combinations proportionally represented in every
split, aggregate level. **FTP-BruteForce train support is severe**: 1,007
train rows vs. 9,669 val / 13,005 test — a direct, correct consequence of
the 18,376-row duplicate group (mostly FTP-BruteForce + DoS-SlowHTTPTest)
landing predominantly in val/test, not a bug. DoS-SlowHTTPTest is
similarly skewed (1,302 train vs. 6,725 val / 9,105 test). **Both hard
invariants verified on this real run**: 0 duplicate-feature-vector groups
span both benign and malicious labels (post-exclusion), and 0 duplicate
feature-vector hashes span more than one split (1,591,298 unique vectors /
1,951,671 rows).

Excluded ambiguous rows are saved to
`data/processed/network_excluded_ambiguous.parquet` (48,329 rows) for
inspection, not discarded.

### Training
`models/network_lgbm.py` (LightGBM + Platt calibration, mirroring
`models/static_lgbm.py`/`models/memory_lgbm.py`'s shape exactly) +
`scripts/train_network.py`. One deliberate hyperparameter deviation from
memory's config: `is_unbalance=True` -- network's train split is 84.3%
benign / 15.7% malicious (a real ~5.4:1 imbalance, unlike memory's
near-50/50 split, so blindly copying memory's omission of this flag would
have reused a choice that was right for a different class balance).

**Scale check, not assumed safe by analogy**: train is 1,526,757 rows × 78
features -- far bigger than memory's 46,736 × 62, and in the same order of
magnitude as static's 2,342,321-row EMBER2024 problem (though static's
2,568 features make its total cell count ~50x larger than network's).
Measured: **54.2s** LightGBM fit (760 boosting iterations, early stopping
at round 760/810), **72.65s** total script wall-clock, **peak RSS 2.8 GiB**.
Default (memory-config-style) LightGBM settings were fine at this scale
without any of static's memory-management tricks (n_jobs cap,
`histogram_pool_size`, `two_round` Dataset construction) -- checked
empirically, not assumed.

**Single-feature AUC sanity check — clean, unlike memory's finding**: the
same check that caught Cortex-Memory's single-VM shortcut (CSE-CIC-IDS2018
is also a fixed-testbed capture, the same category of dataset that made
memory's near-perfect metrics suspicious) found **0 of 78 features exceed
0.95 AUC individually**, on either train or held-out test. Top feature:
`Fwd Seg Size Min` (train AUC 0.736, test AUC 0.848) -- informative, not a
shortcut. `Dst Port` (a plausible shortcut given the fixed testbed, since
specific attack tooling could correlate with specific ports) shows up
mid-table (test AUC 0.761), consistent with legitimate signal rather than
a dominant giveaway. This model's performance reflects genuine
multi-feature pattern learning.

**Threshold derivation**: val+test combined benign=321,733 -- far more
statistical power than memory's 5,860 or behavioral's 274 (one false
positive moves the observed FPR by only ~0.0003%), so FPR targets well
below 0.1% are defensible here without the small-sample caution applied to
those two. Chosen: **target_fpr=0.001 (0.1%) → NETWORK_MALICIOUS_MIN=0.9441855970306654**.
**Test-set metrics at this threshold**: AUC-ROC=0.9972, AUC-PR=0.9951,
precision=0.9969, recall=0.9752, F1=0.9859, FPR=0.098%.

**Per-attack-type detection rate on test** (not just aggregate recall --
train support is severely skewed, so an aggregate number dominated by
high-support classes could mask near-total failure on low-support ones):

| label_raw | n_test | n_train | detection rate | read |
|---|---|---|---|---|
| Bot | 3,517 | 27,882 | 99.94% | — |
| Brute Force -Web | 64 | 484 | 75.00% | under-trained |
| Brute Force -XSS | 23 | 183 | 91.30% | under-trained |
| DDOS attack-HOIC | 8,402 | 67,211 | 100.00% | — |
| DDOS attack-LOIC-UDP | 173 | 1,384 | 100.00% | — |
| DDoS attacks-LOIC-HTTP | 7,056 | 56,453 | 100.00% | — |
| DoS attacks-GoldenEye | 508 | 4,067 | 100.00% | — |
| DoS attacks-Hulk | 5,657 | 45,255 | 100.00% | — |
| DoS attacks-SlowHTTPTest | 9,105 | 1,302 | 100.00% | low train support, detected anyway (duplicate-signature effect, see below) |
| DoS attacks-Slowloris | 1,099 | 8,792 | 99.91% | — |
| FTP-BruteForce | 13,005 | 1,007 | 100.00% | low train support, detected anyway (duplicate-signature effect, see below) |
| **Infilteration** | 1,433 | 11,463 | **10.75%** | **genuinely hard, not under-trained -- see below** |
| SQL Injection | 8 | 72 | 100.00% | tiny sample, treat with caution |
| SSH-Bruteforce | 2,298 | 14,308 | 100.00% | — |
| Benign | 160,869 | 1,286,894 | FPR=0.10% | — |

FTP-BruteForce and DoS-SlowHTTPTest hit 100% test detection despite very
low train support, most likely because the earlier duplicate-group
analysis (see the leakage-check section above) found these two share a
highly repetitive, near-identical flow signature -- even a handful of
train examples generalizes well when the attack traffic itself is that
uniform. Brute Force -Web/-XSS, with similarly low support but more varied
flow shapes, show the expected degradation instead -- the two patterns
("low support but uniform enough to still learn" vs. "low support and
degraded") are both real and distinguishable from the data, not guessed.

**Known limitation (Infiltration detection -- a feature-representation
ceiling, not a pipeline defect):** 10.75% detection despite 11,463 train
rows rules out under-training as the explanation (every other class with
under 10,000 train rows still scored ≥91% except the two flagged above).
Independently confirmed as a documented weak spot: infiltration attacks
are slow/low-volume by design and don't produce the distinctive flow-
statistics signature CICFlowMeter captures well for DDoS/brute-force
traffic; a comparative study found infiltration on this dataset is only
well-detected using NetFlow-derived features, not CICFlowMeter's. This was
deliberately **not** chased by lowering the threshold -- that would spike
FPR across every other (well-detected) class for a class this
representation structurally can't see well.

**Known limitation (CSE-CIC-IDS2018 dataset-level ceilings -- apply to
every model trained on this data, not fixable from inside this pipeline):**
- **Up to ~7.5% label noise**, independently reported: the dataset's labels
  come from automatic time-window-based labeling, not per-flow human
  verification. Some fraction of both the training signal and the "ground
  truth" used to compute every metric above is simply wrong, in a
  direction and magnitude this pipeline cannot detect or correct from
  inside the dataset itself.
- **External-validation-collapse pattern**: models reported as "nearly
  perfect" on CSE-CIC-IDS2018 -- which Cortex-Network's numbers above are
  -- have been independently shown elsewhere to degrade toward random
  performance on external/real-world traffic. This is a documented pattern
  for this specific dataset, not a generic "may not generalize" hedge.

**These two dataset-level ceilings, plus the Infiltration blind spot, are
why Cortex-Network's policy-engine authority is capped at ALERT** (same
rung as memory, `inference/policy_engine.py::NETWORK_MALICIOUS_MIN`'s
comment and `decide()`'s docstring have the full two-legged reasoning):
(a) the Infiltration blind spot means Network structurally can't be
trusted alone for slow/low-volume attacks -- exactly the threat class
Behavioral (API-call sequences) and Memory (process/injection artifacts)
have process-level visibility into that Network's flow-only view doesn't,
so the system's overall coverage still holds even though Network alone
doesn't; (b) the external-validation-collapse pattern means these test-set
numbers shouldn't be assumed to predict real-world performance without
independent validation on traffic this dataset didn't generate.

### ONNX export — shipped, with a documented external precision limitation
`export.export_onnx.export_network_lgbm_to_onnx()`, mirroring
`export_static_lgbm_to_onnx()`/`export_memory_lgbm_to_onnx()` exactly
(calibrator merged into the same graph, single `malicious_probability`
output). Verified against the real `cortex_network` model and the actual
213,217-row CSE-CIC-IDS2018 test split (2026-08-25).

**Mean abs error 2.1e-4, max abs error 0.5465** (vs. static's/memory's
~1e-6-2e-6) -- three orders of magnitude looser, and confirmed as a known,
documented limitation of the ONNX TreeEnsemble spec itself, not a bug in
this pipeline: LightGBM stores learned split thresholds as float64
internally, but ONNX's tree operator only supports float32, and the
official sklearn-onnx documentation states there is no fix short of a
custom double-precision runtime (matches what was found independently
here: onnxmltools' LightGBM converter hard-rejects `DoubleTensorType`
input outright). This is exactly why static/memory (small-magnitude
features) never showed this and network does: Flow Duration/IAT features
run up to ~1.15e8, wide enough for a meaningful number of learned
thresholds to land within float32's precision-loss range at that
magnitude, while static's/memory's feature scales don't.

**Practical impact, measured precisely -- accepted as final given how
small it is**: only **3 of 213,217 predictions (0.0014%) flip** at the
operating threshold. Two of the three sit *exactly* on the threshold value
itself (`0.944186` in both outputs, at 6-decimal display precision) -- a
float32-vs-float64 tie-breaking artifact, not a meaningful disagreement.
The third (`py_proba=0.947, onnx_proba=0.820`, true label benign) is a
real divergence, but the ONNX path is *more* correct than Python's there
(Python would have false-positived; ONNX correctly says benign). Aggregate
metrics shift only in the 4th decimal place: FPR 0.0980% (Python) vs.
0.0976% (ONNX), detection 97.52% (Python) vs. 97.51% (ONNX).
`data/models/cortex_network.onnx` is the final artifact.

**Theoretical mitigation, for the record -- not an action item now**: since
the failure mode is specifically about learned thresholds landing at a
magnitude where float32 loses precision, log-transforming or clipping the
largest-magnitude duration/IAT features before training would keep split
thresholds within float32's precise range, likely eliminating this
divergence at the source. Flagged as a future option if this ever becomes
operationally significant (e.g. a production threshold that happens to sit
exactly where a boundary flip changes the outcome more often than measured
here), not worth doing given the current, already-negligible impact.

## Cortex-Emulation
Fifth signal: 1D-CNN + multi-head self-attention over Speakeasy-emulator
API-name sequences (Quo Vadis, Trizna et al., HF `dtrizna/quovadis-speakeasy`,
Apache-2.0). Same architecture family as Cortex-Behavioral
(`models/emulation_cnn.py`, `models/train_emulation.py`,
`scripts/train_emulation.py`, `tokenizer/emulation_tokenizer.py`), tuned for
this signal's data: vocab 3,154 (train-only), `MAX_SEQ_LEN=500`,
`embed_dim=64`, `dropout=0.4`. Modeling population is `module_entry` rows
only, with exact-duplicate API-name sequences collapsed to one representative
per unique sequence (`duplicate_count` retained but inert in training) —
`thread`/`tls_callback_*` rows and the raw event detail stay in the canonical
parquet but are excluded here. Train/val are the authors' Jan-2022 partition
(val carved from train); test is their **Apr-2022** partition, a deliberate
temporal concept-drift holdout, preserved rather than re-randomized.

**Deployment status: report-only / additive — NOT in the policy decision.**
Cortex-Emulation has no branch in `inference/policy_engine.py::decide()`.
`EMULATION_MALICIOUS_MIN` is set (0.999358594, the 1%-FPR sweep point) **for
logging/telemetry only** — `emulation_verdict_from_score()` produces an
`EmulationVerdict` a caller may record alongside a scan, but nothing routes
it into the final decision. This mirrors malware-ml's own precedent for its
Behavioral v2 category signal (`deployment_status:
additive_report_only_not_in_policy_decision`). It is a **stricter**
disposition than memory's or network's ALERT cap, for two evidence-backed
reasons below.

**Training run (2026-08-27, seed=42, CPU).** Early stop is on val ROC-AUC
(deliberate deviation from `train_behavioral.py`'s loss-based stop — only the
patience / best-state-restore mechanism is reused), plus an explicit
train/val AUC-gap overfitting monitor (stop if `train_auc - val_auc > 0.05`
for 3 consecutive epochs). Output is a raw sigmoid — no calibration layer,
matching Cortex-Behavioral; the Platt-calibration bug fixed during
Cortex-Static's ONNX export was specific to LightGBM's `predict_proba`
applying a fitted calibrator on top of the booster, and has no analogue on
the torch path (checked, not skipped).

The run hit the **overfitting stop at epoch 22** (best epoch 19, val
AUC 0.9495). Val AUC plateaued around 0.94 by epoch 8 while train AUC ran to
0.995 — the model memorizes the ~6k collapsed training sequences trivially.
Held-out test AUC-ROC 0.8737.

**Threshold sweep — val+test combined (2,386 benign; pools Jan + Apr eras,
flagged explicitly):**

| target FPR | threshold | actual FPR | benign FPs | detection rate |
|---|---|---|---|---|
| 0.10% | 0.999996 | 0.084% | 2 / 2,386 | 4.2% |
| 0.50% | 0.999887 | 0.46% | 11 / 2,386 | 44.3% |
| **1.00%** | **0.999359** | **0.96%** | **23 / 2,386** | **49.9%** |
| 2.00% | 0.960831 | 1.97% | 47 / 2,386 | 55.0% |
| 5.00% | 0.676327 | 4.82% | 115 / 2,386 | 67.1% |

1% is the primary candidate (~24 expected FP, matching the ~50% pooled
detection). 0.1% (~2 expected FP) is too thin to trust as a precise FPR
claim — same caution applied to Cortex-Network's tightest targets. The score
distribution is strongly bimodal: the 1%-FPR threshold sits at **0.999359**,
and detection is only ~50% there.

### Known limitation 1 — ablation-confirmed Jan→Apr temporal concept-drift collapse

The dataset authors' Jan-2022 / Apr-2022 split exists to measure temporal
non-stationarity, and it does. At the 1%-FPR threshold, **malicious recall
drops from 70.7% on the Jan-era slice to 41.5% on the Apr-era slice**
(AUC-ROC 0.949 → 0.874); benign FPR stays ~0.7–1% in both eras, so the loss
is entirely on the malicious side.

This was checked against the pre-committed capacity fallback rather than
assumed. Re-running at **`embed_dim=32`** (the overfitting stop's own first
recommendation): the overfitting stop fired *earlier* (epoch 18, best epoch
10, val AUC 0.9396 — worse on every axis), and the drift gap **did not
move** — recall 68.0% (Jan) → 38.3% (Apr), a 29.7 pp gap versus 29.2 pp at
`embed_dim=64`. **Capacity reduction fixed nothing.** Detection at 1% FPR
fell across the board (pooled 49.9% → 46.8%). This rules out overfitting /
model size as the relevant lever — the drift is a genuine train/test
distribution shift, not a capacity artifact.

**Per-family drift diagnostic (descriptive, no retraining).** For the two
best-supported malicious families with very different drop sizes —
`ransomware` (test detection 48.3%, mild) and `coinminer` (17.1%, sharp;
8.0% at `embed_dim=32`):

- *Train-side duplication (Jan 2022):* `coinminer` has **185 unique training
  sequences** (97 singletons) across 5,862 pre-collapse rows; its
  `duplicate_count` decays gradually (top-5 = 58% of family volume, max
  931). `ransomware` has **1,065 unique sequences** (509 singletons) across
  9,059 rows — one dominant builder-kit sequence alone is 48% of family
  volume (`duplicate_count` 4,312), then a cliff. So raw-volume
  concentration is *higher* for ransomware; the real difference is
  **distinct-sequence variety** — ransomware gave the model ~5× more unique
  sequences to learn from.
- *Test-side recurrence (Apr 2022):* both families recur against train at
  about the same rate — `coinminer` 21.6% exact / 53.4% near-neighbour
  (token-set Jaccard ≥ 0.90) / 6.8% no match; `ransomware` 17.2% / 53.4% /
  4.3%. **The hypothesis that Apr `coinminer` simply doesn't recur in Jan
  training is not supported.**
- *e64 detection by match status:* `ransomware` tracks match quality —
  exact-match Apr rows 70% detected (mean score 0.86), near-only 29% (mean
  0.48): a real memorization-vs-generalization gap, but a gentle one.
  `coinminer` is **uniformly ~15–25% across every bucket**, including
  exact matches (15.8%, mean score 0.675 — below the 0.9994 threshold). The
  model never built a firing region for `coinminer` at all; its Apr
  sequences' nearest training neighbours are benign (`clean`) more often
  than `coinminer`, and a smaller model (`embed_dim=32`) collapses it
  further into the benign manifold (8.0%).

Mechanism, as additional evidence for the drift diagnosis: the model
**memorizes token sequences rather than learning generalizable behaviour**.
It fires near what it memorized and goes quiet on modest drift
(ransomware's "near" bucket at 29%); for a family with too thin a
distinct-sequence base it never builds a firing region, exact matches
included (coinminer). Capacity tuning cannot address either — a smaller
model memorizes just as completely and generalizes no better.

### Known limitation 2 — thin margin over a trivial baseline

At the 1%-FPR threshold, test accuracy is **86.4%** (`embed_dim=64`) /
**85.6%** (`embed_dim=32`), against:
- majority-class (predict all benign): **77.6%** (test is 77.6% benign)
- exact train-sequence duplicate-lookup (571 / 2,495 test sequences appear
  verbatim in train; misses fall back to benign): **82.2%**

The model beats the duplicate-lookup by only **+4.1 pp** (e64) / **+3.3 pp**
(e32). Combined with Limitation 1, a signal that recovers ~40% of malware on
the next collection era and barely outperforms a lookup table cannot carry
autonomous *or* ALERT authority — hence report-only.

### Per-family test recall (embed_dim=64, 1%-FPR threshold)

| family | n_train | n_test | detection |
|---|---|---|---|
| trojan | 262 | 79 | 64.6% |
| backdoor | 149 | 99 | 51.5% |
| ransomware | 1,065 | 116 | 48.3% |
| dropper | 193 | 44 | 47.7% |
| keylogger | 112 | 64 | 29.7% |
| rat | 60 | 69 | 27.5% |
| coinminer | 185 | 88 | 17.1% |
| clean *(benign FPR)* | 4,003 | 1,924 | 0.68% |
| windows_syswow64 *(benign FPR)* | 41 | 12 | 0.00% |

`rat` (27.5%) and `keylogger` (29.7%) are the thinnest-support families
(60 / 112 train) — reads as under-trained, consistent with the caution
flagged before training. `coinminer` (17.1%) has more train support than
several better-detected families and is diagnosed above as a distinct-
sequence-variety problem, not raw support.

### Future work (not committed to now)

A genuine fix for the drift collapse would most likely need **one of**:
- **Engineered behavioural-category features in place of raw API-token
  sequences.** Hypothesis: coarser behavioural categories (file / registry /
  network / process-injection activity classes) may be more temporally
  stable across collection dates than exact API-call sequences the current
  model memorizes. Worth testing — not guaranteed to hold.
- **Training data spanning more collection dates**, so the model sees
  temporal variation during training rather than only at test.

It does **not** need further hyperparameter tuning — the `embed_dim=64` vs
`embed_dim=32` ablation already ruled out model capacity as the relevant
lever.

### Not done for this signal

No ONNX export (the other four tree/PyTorch signals have one; Emulation's is
deferred until the model itself is worth shipping). No pipeline wiring in
`inference/pipeline.py`. `CortexEmulationNet` architecture,
`models/emulation_cnn.py`'s pre-committed second fallback (drop a conv
block) — not pursued, since the larger capacity cut (`embed_dim` halving)
already made things worse, not better.

## Running a scan end-to-end
```python
import torch

from models.behavioral_cnn import CortexBehavioralNet, SEQUENCE_LENGTH
from models.static_lgbm import LGBMModel
from tokenizer.api_tokenizer import ApiTokenizer
from inference.pipeline import CortexPipeline

static_model = LGBMModel.load("data/models/cortex_static")

tokenizer = ApiTokenizer.load("data/models/api_vocab.json")
# checkpoint is a raw state_dict (see scripts/train_behavioral.py), not a
# pickled full model -- construct the architecture first, then load into it.
behavioral_model = CortexBehavioralNet(vocab_size=tokenizer.vocab_size, sequence_length=SEQUENCE_LENGTH, embed_dim=128)
behavioral_model.load_state_dict(torch.load("data/models/cortex_behavioral_best.pt", map_location="cpu"))
behavioral_model.eval()

pipeline = CortexPipeline(static_model, behavioral_model, tokenizer)
result = pipeline.scan(r"C:\Samples\application.exe", api_calls_json_path=r"C:\Telemetry\api_calls.json")
print(pipeline.to_security_event(result))
```

## What's intentionally not copied
`features/pe_features.py` matches EMBER2024's public 2568-dim *group budget*
(verified against `FutureComputing4AI/EMBER2024`, Apache-2.0) but every field
inside each group was chosen independently — this is not a line-for-line
port. Same for the model classes: architecture shape (1D-CNN stack +
multi-head self-attention) follows the brief, layer-by-layer code is fresh.

## Open items to confirm as you go
1. Exact column names in the `joyce8/EMBER2024` parquet schema — adjust
   `_split_xy` once you've downloaded and inspected a split.
2. Source dataset for the behavioral `api_calls`/`label` table.
3. `embed_dim` for the behavioral CNN is set to 128 by default in
   `scripts/train_behavioral.py` — drop to 64 there if you want the smaller
   variant.
