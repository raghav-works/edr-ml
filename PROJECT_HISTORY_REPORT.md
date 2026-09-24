# Cortex Malware Detection — Project History Report

**Prepared:** 2026-08-27 (updated from the 2026-08-19 version)
**Scope:** Full history across three repositories — `malware-ml` (research), `cortex-endpoint` (Windows deployment package), `cortex-ml` (clean reproduction, now five signals)
**Locations:** `/home/tnids/malware-ml`, `/home/tnids/cortex-endpoint`, `/home/tnids/cortex-ml`

---

## Executive summary

Cortex is now a **five-signal** Windows malware detector — static PE classification, behavioral API-sequence classification, memory-forensics classification, network-flow classification, and emulation-trace classification (trained, but **report-only / not in the policy decision** — see Signal 5) — that has gone through three engineering phases:

1. **`malware-ml`** — the original research repo. Produced a validated static LightGBM model and two behavioral models across 8 branches, with a documented out-of-memory debugging saga, a rejected training strategy, a 20-trial hyperparameter search, and a best-in-project recalibrated candidate that was never promoted to production.
2. **`cortex-endpoint`** — a packaged Windows `.exe` built from `malware-ml`'s trained artifacts, with real validation evidence (real malware, real signed tools, a 5-file test run) and a shrinking-but-still-large PyInstaller bundle (876MB → 220MB after removing a torch dependency).
3. **`cortex-ml`** — an independently engineered, from-scratch reproduction, originally built with two signals (Static, Behavioral) and now extended to five. Three new signals (Memory, Network, Emulation) were built after the original version of this report: Cortex-Memory (CIC-MalMem-2022), Cortex-Network (CSE-CIC-IDS2018), and Cortex-Emulation (Quo Vadis Speakeasy traces). Each new signal repeated the same discipline the original two established — schema validation against real data (not docs), real leakage checks, thresholds re-derived from scratch — and each surfaced its own real, previously undocumented bugs and dataset-level limitations, detailed below. Cortex-Emulation, once trained, hit an ablation-confirmed Jan→Apr temporal concept-drift collapse and is deliberately kept **report-only / additive** — no branch in the policy engine's `decide()` — a stricter disposition than Memory's and Network's ALERT caps.

**Bottom line:** the underlying detection logic proven in `malware-ml`/`cortex-endpoint` was never the problem — it hit strong metrics (AUC > 0.998 static, AUC > 0.99 behavioral) and passed real-world validation against actual malware and signed system tools. `cortex-ml` reproduces and extends that architecture in a footprint an agent team can actually receive and run, and its policy engine enforces a **tiered authority model**: Static (BLOCK) and Behavioral (TERMINATE) hold autonomous authority in code; Memory and Network are independent signals capped at ALERT pending real-world validation beyond their respective lab-collected training datasets; and Emulation, after training exposed an ablation-confirmed temporal concept-drift collapse, is held at **report-only / additive** — trained and logged, but never routed into `decide()`. Each cap is separately earned and evidence-backed. **Caveat added 2026-08-28:** cortex-ml's *own* Static model is now production-readiness-DOWNGRADED — a 57-file round exposed a two-stage calibration-saturation failure (near-step Platt calibrator + `pe_features.py` fidelity gap) that makes its calibrated score carry almost no mid-range information and its rank order unreliable near the BLOCK threshold, including a confirmed benign-outranks-malware cross-over. Autonomous BLOCK should not be trusted until fixed and re-validated (see the Cortex-Static section).

> **RESOLVED, 2026-09-22.** The 2026-08-28 caveat above no longer describes
> the current system. The Platt calibrator was refit on raw booster margins
> instead of probability (`6eed759`), Static's BLOCK verdict was
> interim-capped to ALERT with corroboration-gated escalation to a real
> BLOCK only when a second, independent MALICIOUS signal (memory or
> network) corroborates it on the same scan (`b11999f`, `ac132d2`), and
> Static was fully retrained this session with proper split discipline —
> dedicated cal split, thresholds derived on cal with exact
> Clopper–Pearson confidence intervals, single pre-registered test read
> (`8192452`, `438d936`; see `docs/PhantomCortex_Static_Retrain_Report.pdf`
> and `reports/static_retrain_20260921/`). Static (BLOCK) is **no longer
> an autonomous, uncapped authority in code**; the "tiered authority
> model" sentence above is superseded on that specific point. The
> incident detail below is kept as accurate history of what was found.

---

## STEP 0 — Repository locations

All three repos live directly under the home directory, unambiguously:

| Repo | Path | Current branch | Branch count |
|---|---|---|---|
| malware-ml | `/home/tnids/malware-ml` | `feat/behavioral-v2-onnx` | 8 local |
| cortex-endpoint | `/home/tnids/cortex-endpoint` | `task2/thrember-bloat-trim` | 4 local |
| cortex-ml | `/home/tnids/cortex-ml` | `master` | 1 (single branch) |

No ambiguity was found — each name matched exactly one directory on this machine.

---

## STEP 0.5 — 5-test-file validation results: all 5 found

All five requested results were located at `cortex-endpoint/test_results/model_comparison/new_model/*.json`, produced by commit `befe0bd` ("Merge signed-PE fix; document PyInstaller false-positive pattern for follow-up") on the `task0/repro-hygiene` branch. **Every file's sha256 in the JSON was independently verified against the actual `.exe` on disk in `/home/tnids/*.exe` — exact byte-for-byte match on all 5**, confirming these are genuine results for these exact files, not stand-ins. `notepadd.exe`'s sha256 (`d38ba15b...`) matches the copy you already had saved. No results are missing — see the table in Era 2 below.

Two caveats worth flagging honestly:
- The bash history on this machine shows these scans were actually invoked via `python -m cortex_endpoint.cli scan` (the same code the PyInstaller `.exe` wraps), not by directly running `cortex-endpoint.exe` on this Linux machine (which can't execute a Windows PE binary without Wine, and Wine isn't installed here). The `.exe` itself was run separately on a Windows host per PowerShell command fragments (`.\cortex-endpoint.exe scan "C:\CortexTest\notepad_test.exe"`) and `scp` transfers to `192.168.37.108` and `172.29.235.247` visible in history — but the JSON outputs from those specific Windows-side runs weren't found on this machine. The `test_results/model_comparison/new_model/` JSONs are from the identical underlying scan logic (`cortex_endpoint.cli`), just invoked directly rather than through the compiled `.exe`.
- A parallel `old_model/` result set also exists for the same 5 files, comparing against a prior version of the static model — included in the Era 2 section below since it's directly relevant to what changed.

---

## ERA 1 — malware-ml (research phase)

### Branch structure

History is a shared trunk that forks into an increasingly specific chain — **not** a wide hyperparameter-search branch tree. All branches except `master`/`main` build on each other in strict order:

```
master → main → task0/repro-hygiene → task1/candidate-v1-fpr-calibration
  → feat/behavioral-v2-data-pipeline → feat/behavioral-v2-training → feat/behavioral-v2-onnx (current, uncommitted work only)
```
`feat/static-accuracy-onnx` is a sibling of `main` with **zero unique commits** — an unused placeholder branch, not an experiment.

| Branch | What it adds | Status |
|---|---|---|
| `master` | Initial commit only (static LightGBM pipeline seed) | Repo root |
| `main` | Memory-safe single-pass static training pipeline, Cortex-Static baseline handover, Cortex-Behavioral Binary (Model A) | Finalized handover point (2026-08-03) |
| `task0/repro-hygiene` | Pins the `thrember` commit; tracks `models/cortex_static/*` and a release baseline copy in git | Provenance only |
| `task1/candidate-v1-fpr-calibration` | Recalibrates Candidate V1's block threshold to hit a 0.1% FPR target | **Best static result in the project** |
| `feat/behavioral-v2-data-pipeline` | Mal-API-2019 + MalbehavD-V1 loaders, 20-class taxonomy, scoped 8-class label mapping | Data/features only |
| `feat/behavioral-v2-training` | Full v2 dataset/split/vocab/attention-CNN training + eval | 8-class behavioral model trained (report-only) |
| `feat/behavioral-v2-onnx` (current) | ONNX export work for Candidate V1 + behavioral v2 | In progress, uncommitted |

### The hyperparameter search: one 20-trial sweep, not many branches

The `runs/reconstructed_1200k_trial17` directory name suggested a wide search across numbered trial branches. That is **not** what happened, and the report should say so plainly rather than imply more than exists: the project ran a single **20-trial hyperparameter optimization sweep** (an Optuna/hyperopt-style loop within one training invocation, not 20 separate training runs or branches), and only the winning trial's configuration (trial 17) was ever written to a manifest — and even that manifest is explicitly marked:

> `"provenance": "reconstructed-from-console-output"`, `"provenance_note": "This manifest was reconstructed after the run from the operator-provided HPO and training console output; it was not programmatically captured during training."`

No per-trial logs, metrics, or manifests for the other 19 trials exist anywhere in the repo or its git history (confirmed via `git log --all --diff-filter=A --name-only -- '*runs/*manifest*'`, which returns nothing beyond this one file). Trial 17's winning hyperparameters:

| Hyperparameter | Value |
|---|---|
| `num_leaves` | 173 |
| `learning_rate` | 0.1357416564 |
| `min_child_samples` | 105 |
| `subsample` | 0.8552938786 |
| `colsample_bytree` | 0.8132394568 |
| `reg_alpha` | 0.0017381050 |
| `reg_lambda` | 0.0000233503 |

### The static model journey (in order)

**1. Dataset duplication bug found and fixed.** The raw EMBER2024 download had every row present twice (a 50% duplicate rate in every shard). `data/processed/ember2024_pe_dedup/duplicate_report.json` shows the correction: **4,680,000 raw train rows → 2,340,000 unique**, **1,080,000 raw test rows → 540,000 unique**. Zero train/test hash overlap confirmed (though ~60 test hashes repeat across test shards).

**2. First full-dataset training attempt: OOM kill.** A memmap was opened, but NumPy's advanced indexing on the selected rows materialized a full ordinary array in memory. The 23.4 GiB raw feature matrix, its selected-row copies, a prematurely-opened ~5.17 GiB test matrix, and LightGBM's own bins/gradients/Hessians/histograms all coexisted in RAM at once. Linux OOM-killed the process near the host's 38–39 GB limit. Diagnosed as a loader/lifetime bug, not evidence LightGBM couldn't fit the data.

**3. Rejected fix attempt: sequential chunked training.** `train_static_full_chunked.py` split the 2,140,000 fitting rows into three chunks (713,334 / 713,333 / 713,333) and called `lgb.train()` three times with `init_model` continuation between them. It completed without an OOM — but it wasn't a true global fit; each chunk's new trees only optimized against that chunk's data. Validation TPR at ~0.1% FPR **degraded across chunks**: 0.9169565 → 0.1709383 → 0.1743262 by the final chunk, with the threshold saturating at 0.9999999999082598. Test TPR fell to 0.1914744. **Explicitly rejected — never initialized from or deployed.**

**4. Working fix: disk-backed single-pass pipeline.** `static_split_plan.py` froze disjoint index sets; `indexed_memmap_sequence.py` exposed bounded reads; `prepare_static_binary.py` wrote binary train/validation files to disk; `train_static_single_pass.py` made exactly one global `lgb.train()` call over binary-backed data; `evaluate_static_batched.py` opened validation/test in bounded prediction batches. This trained on **all 2,140,000 fitting rows** + 200,000 validation rows, 8 threads, a 2,048 MiB histogram pool, 500 trees. Training took 792.6 seconds and peaked at **~5.07 GiB RSS** — versus the ~38–39 GB that killed the earlier attempt. (One residual quirk: the *preparation* stage alone, despite bounded reads, separately peaked at **~41.4 GiB RSS** — noted as unresolved and worth further investigation, though it doesn't affect the training-time memory story.)

This pipeline produced **Candidate V1**, trained on the full deduplicated dataset.

**5. The deployed baseline used only a 51.3% subsample — and says so explicitly.** Before the disk-backed pipeline existed, an earlier, memory-constrained run (using trial 17's HPO-selected hyperparameters) trained on only **1,200,000 of the 2,340,000 available deduplicated rows**. The model's own metadata file states this plainly:

> `"dataset_coverage_note": "Trained using 1,200,000 of 2,340,000 available deduplicated EMBER2024 PE training rows (51.3%). This was a deliberate choice due to RAM constraints on the training machine, not a data quality issue. Follow-up: train on the full dataset using chunked training."`

This 497-tree model (`models/cortex_static/model.txt`) became **the official deployed baseline** despite Candidate V1 (trained on the full dataset via the later disk-backed pipeline) outperforming it — see below.

### Static model results, side by side

| Model | Trained on | Test AUC | Test FPR | Test TPR | Threshold | Status |
|---|---:|---:|---:|---:|---:|---|
| **Deployed baseline** (497 trees, trial-17 HPO params) | 1,200,000 rows (51.3% of available) | 0.99774 | 0.10519% | 88.365% | 0.98822 | Currently official/deployed |
| Candidate V1, original threshold | 2,140,000 rows (full dataset) | 0.99867 | 0.15444% | 92.764% | 0.95213 | Rejected — threshold didn't transfer val→test |
| Candidate V1, 95% CI threshold | 2,140,000 rows | 0.99867 | 0.13259% | 92.094% | 0.96154 | Diagnostic only |
| **Candidate V1, recalibrated** (`task1`, commit `f31fdaa`) | 2,140,000 rows | **0.99867** | **0.09815%** | **90.853%** | **0.97344** | **Best result — beats baseline on FPR, TPR, and AUC; never promoted** |

The recalibrated Candidate V1 is flagged in its own commit message as the *third* time the official test set was consulted during threshold search — the project explicitly treats this as strong diagnostic evidence rather than a pristine one-shot number, and it was never promoted to replace the deployed baseline.

**Known unresolved issue, both models:** the candidate model incorrectly blocks `cortex-endpoint.exe` itself (score 0.9875, expected benign) — a PyInstaller-build-artifact false positive, explicitly called out in `cortex-endpoint/model_delivery/cortex_static_v2_research/evaluation_report.json` as unresolved: *"Independent acceptance testing is mandatory before production use."* The deployed baseline shows the same pattern to a lesser degree.

### Behavioral models

**Model A (binary, Oliveira-sourced).** Source: the Oliveira Windows API-call-sequence dataset — a CSV of `hash, t_0..t_99 (100-token API sequence), malware` rows, loaded via a purpose-built disk/offset-indexed reader (`src/malware_ml/data/oliveira_api_loader.py`) so the full corpus never needs to sit in memory. Raw source: 43,876 rows. MD5-grouped to establish sample identity: 43,865 non-conflicting unique groups after excluding 7 duplicate rows and 2 conflicting MD5 groups (4 rows). Grouped 70/15/15 split, seed 42, so no sample's near-duplicates leak across splits. Vocabulary: 307 named Windows APIs + one `<UNK>` token = 308-entry vocabulary. Architecture: embedding → 2×Conv1D → global max pool → binary sigmoid head, no attention. Trained through epoch 14; best validation ROC AUC at epoch 9, restored after 5 non-improving epochs.

Reserved-test results: **ROC AUC 0.9923, PR AUC 0.9998, TPR 89.78%, precision 99.91%, F1 94.58%** at the frozen validation-selected threshold (0.9910649657).

**This is the threshold-generalization failure story that directly motivated cortex-ml's more conservative approach:** the threshold was frozen using only the validation set, where it produced near-zero FPR — but on the held-out test set it produced **3.09% FPR (5 false positives out of 162 benign test samples)**. `malware-ml/docs/model_status.md` states this explicitly: *"The frozen threshold's near-zero validation FPR did not transfer to test (3.09% observed)."* With only 161–162 benign samples in each split, FPR estimates below ~3% aren't statistically supportable at all — each single false positive moves the measured FPR by ~0.6 points. **Not production-approved for this reason.**

**Behavioral v2 (multiclass, report-only).** Built on `feat/behavioral-v2-*`, sourced from **Mal-API-2019 + MalbehavD-V1** (the same combination cortex-ml later adopted), mapped from a 20-class taxonomy down to 8 scoped classes (Backdoor, Downloader, Worm, Dropper, Spyware, Adware, Benign, Other_Malicious). Attention-CNN architecture. Reserved test (1,264 rows): macro-F1 0.6151, accuracy 0.6297, macro AUC (OvR) 0.8930, per-class F1 ranging from 0.979 (Benign) down to 0.376 (Spyware, weakest class). Explicitly marked `deployment_status: additive_report_only_not_in_policy_decision` — it never feeds the allow/alert/block decision, a claim proven by `cortex-endpoint/tests/test_category_signal_policy_isolation.py`. ONNX export: 0.36MB, CPU p50 latency 0.96ms (faster than native PyTorch's 1.57ms), parity verified to 1.25e-6 max absolute difference.

### Documentation notes

`docs/complete_project_analysis_2026-08-03.md` is the deepest single writeup of the engineering journey (OOM failure, chunked-training failure, disk-backed fix) but is **explicitly stale** — it and the root `README.md` both predate the `task1` recalibration and the entire behavioral-v2 track, and still describe the baseline as "the official selected model" and behavioral work as "out of scope." A later handoff document, `PROJECT_CONTEXT.md` (dated 2026-08-18, the day before cortex-ml's first commit), explicitly warns against trusting those two documents' conclusions over the branch/commit evidence, and was the primary cross-check source for this report's Era 1 and Era 2 sections.

---

## ERA 2 — cortex-endpoint (packaging phase)

### Branch structure

Unlike malware-ml, history here is **linear**, not forked: `main → fix/signed-pe-authenticode → task0/repro-hygiene → task2/thrember-bloat-trim` (current), 20 commits total on the current tip. cortex-endpoint **never trains anything** — malware-ml is the sole source of truth for models; artifacts are copied over and hash-verified.

| Branch | Adds | Status |
|---|---|---|
| `main` | Baseline scaffold: thrember feature extractor, Cortex-Behavioral binary CNN scanner, staged decision policy, PyInstaller packaging + Windows CI | Working, but signed PEs were broken |
| `fix/signed-pe-authenticode` | Monkey-patches `signify.x509.CertificateStore.__getitem__` to fix a `signify` 0.8.1 incompatibility breaking signed-PE feature extraction; adds per-feature-group score explanations | Signed-PE parsing fixed |
| `task0/repro-hygiene` | Merges the signed-PE fix; adds `model_delivery/cortex_static_v2_research/` (full evaluated candidate package) and `test_results/` (real-malware + model-comparison scan outputs) | First real-world evaluation evidence |
| `task2/thrember-bloat-trim` (current) | Trims lightgbm's optional matplotlib/PIL import path, then removes torch entirely | Packaging-size work |

### How the model became a deployable agent

Architecture (`src/cortex_endpoint/`): `cli.py` → `static_scanner.py` (StaticScanner: thrember features → LightGBM or ONNX Runtime) and `behavioral_scanner.py` (BehavioralScanner: binary CNN, ONNX Runtime after commit `55402de`, plus optional v2 category signal) → `policy.py` (StagedPolicy: static evidence is primary, a static `block` is always final, a behavioral `malicious` verdict upgrades to `terminate`, scan failures fail-closed to `alert`) → `event_schema.py` (JSON verdict envelope). Packaged via `build_exe.py` (PyInstaller `--onedir`) with Windows CI in `.github/workflows/build-windows.yml`.

**Feature parity with the training-time extractor** — the single most important correctness gate for a static-features model — was explicitly tested and passed: `tests/test_static_feature_parity.py` verifies against "malware-ml pinned thrember installation," with the result *"four real-PE vectors, including one signed PE, matched byte-exactly"* (dated 2026-08-07). This verification is embedded in every single scan's JSON output (`model.runtime_feature_parity_verified: true`), not just recorded in a test log.

### Capabilities

- **Explainability:** every static scan includes a `score_explanation` block with per-feature-group `pred_contrib` breakdown (header, imports, pefilewarnings, histogram, authenticode, datadirectories, richheader, byteentropy, strings, section, exports, general), the most-influential group, and the raw margin — method `lightgbm_pred_contrib_grouped_by_ember_feature_family`.
- **Policy versioning:** every event carries `policy.version: "cortex-staged-v1"` and `schema_version: "1.0.0"`.
- **CLI:** `scan` (static only), `scan-behavioral` (JSON API-call list, simulates future telemetry), `scan-combined` (both signals).
- **Health/fail-safe:** every event embeds `policy.health` and `static_failure_mode: "fail_closed"` — a scan failure defaults to alerting, never to silently allowing.

### Size evolution: why this wasn't the handoff basis

There's no commit or doc in this repo that explicitly says "we're abandoning this for a leaner rebuild" — that decision postdates this repo's most recent work (cortex-ml's first commit is 2026-08-19; the last cortex-endpoint work captured in the handoff doc is 2026-08-18). But the quantitative case is documented directly in the commit history:

| Stage | Bundle size | Commit |
|---|---:|---|
| Baseline (torch + full lightgbm deps) | ~880MB+ | pre-trim |
| After stubbing lightgbm's matplotlib/PIL import path | 843MB (~40MB saved) | `8a2d53e` |
| After removing torch entirely | **220MB (656MB saved)** | `3e07a11` |

The measured on-disk snapshots confirm this: `dist/cortex-endpoint-OLD-with-torch/` = 876MB, current `dist/cortex-endpoint/` = 220MB. Even after that 656MB cut, the full handoff artifact (`cortex-endpoint.zip`, which bundles more than just the executable — test files, model_delivery, docs) is **1.36GB on disk**. Combined with `malware-ml`'s 178GB data directory and the two repos' manual, non-submodule artifact-copying relationship (explicitly noted in the prior handoff doc: *"no submodule link — artifacts are moved between them manually/by copy"*), this is a genuinely large, operationally awkward pair of repos to hand to a separate team — independent of whether the model itself is any good.

### Validation run — 5 real test files

All 5 results are present and sha256-verified against the on-disk test files. None are missing.

| File | sha256 (truncated) | Static score | Verdict | Behavioral | Final decision | Most influential feature group |
|---|---|---:|---|---|---|---|
| `svchost.exe` | `75772da6…` | 0.4744 | allow | not run (gated behind static ALLOW; static did not reach that stage here — static-only policy) | **allow** | header |
| `notepad_test.exe` | `ab15a95d…` | 0.0057 | allow | not run | **allow** | datadirectories |
| `benign_test_50mb.exe` | `fce08e33…` | 0.9690 | alert | not run | **alert** | header |
| `notepadd.exe` | `d38ba15b…` | 0.7746 | alert | not run | **alert** | header |
| `extractor.exe` | `71506a19…` | 0.9820 | block | not run | **block** | header |

All 5 ran in `static_only` policy stage (`behavioral_state: not_started` on every result) — the behavioral scanner was not invoked for any of these 5, so there is no behavioral status to report for them; the decision is entirely static-model-driven. `header` (COFF/optional-header values) is the dominant influential feature group for 3 of 5 files, `datadirectories` for one — consistent with what the feature-group breakdown is designed to surface (PE structural anomalies, not just byte-level statistics).

One data point worth flagging: `benign_test_50mb.exe`, a 50MB benign test file, scored an **alert** (0.9690) — close to the block threshold (0.9746). This is consistent with the project's own documented weakness (large/atypical PE structure driving header-derived features toward the malicious side) rather than a surprise; it's the same feature family driving the `notepadd.exe` and `extractor.exe` scores here, and the same family implicated in the project's self-flagging false-positive issue described in Era 1.

A parallel `old_model/` result set (same 5 files, prior static model version) exists alongside this in the same commit — comparing the two isn't necessary for this validation record but is available if a before/after regression check is wanted later.

### test_results/ — other validation evidence

- **`quarantine_malware/`** — 4 real named malware samples: `wannacry.json` (block, 0.9968), `zeus.json` (block, 1.0), `petya-4c1dc737.json` (alert, 0.8988), and **`petya-26b4699a.json` (allow, 0.6205 — a missed detection**, below even the alert threshold of 0.5566). Worth being direct about: this is one documented miss on a Petya variant in the available evidence, not a clean sweep.
- **`sysinternals_sigcheck/`** — legitimate, signed Microsoft Sysinternals tools. `sigcheck.json`/`sigcheck64.json`/`sigcheck64a.json` all correctly allowed (scores 0.0000–0.0004). Two non-PE files in the same batch (`Eula.txt`, `Sigcheck.zip`) correctly returned an `error` status rather than crashing or misclassifying.

---

## ERA 3 — cortex-ml (clean, independent reproduction — five signals)

Single branch (`master`). Four commits exist, all dated 2026-08-19, covering the original two signals (Static, Behavioral). **Everything covering Memory, Network, and Emulation is currently uncommitted working-tree changes on top of those four commits** — real, built, and (for Memory/Network) fully trained and verified, but not yet captured in commit history. This section documents both the committed and uncommitted work, clearly distinguishing which is which.

### Commit 1 — `6cd4131`: "Cortex-Static: EMBER2024 reproduction, OOM fixes, re-derived thresholds"

- **Dataset ingestion rewritten:** `datasets.load_dataset()` hit an Arrow schema-inference crash on this dataset, so `data/download_ember2024.py` streams the EMBER2024 PE-format records directly from the HF zips instead, deduplicating on sha256 in bounded batches.
- **Feature extractor reimplemented for correctness, not guessed:** `features/pe_features.py` was rewritten to match the real EMBER2024/thrember reference extractor's formulas exactly (fixed string-regex vocabulary, header categorical dictionaries, section stats, data-directory layout) — replacing an earlier version built from independently-guessed field semantics — so live PE-byte inference and EMBER2024 training records are numerically compatible.
- **The OOM debugging saga, with specific numbers:** training on the full 2.34M-row / 2568-feature split repeatedly crashed with OOM on a 38GB machine. Fixes applied, in order: a streaming train/val loader instead of a full DataFrame with fancy-indexed copies, `free_raw_data=True` plus explicit `del`/`gc.collect()`, reduced `n_jobs`, a capped `histogram_pool_size`, and two-round `Dataset` construction. **The final, decisive fix was adding swap space at the system level, not reducing model capacity or subsampling the data** — the direct contrast with malware-ml's 51.3% subsample decision under the same class of memory pressure.
- **Thresholds re-derived from scratch:** `STATIC_ALLOW_MAX` / `STATIC_BLOCK_MIN` were re-derived against this repo's own trained model's calibrated test-set probabilities (target FPR 0.01 / 0.001) rather than carrying over the prior project's placeholder values.

**Dataset & provenance:** EMBER2024, HuggingFace (`FutureComputing4AI/EMBER2024`-derived, Apache-2.0), PE-format records, 2.34M unique train rows after dedup, 539,940 unique test rows.
**Leakage check:** zero train/test hash overlap confirmed after dedup fix (a handful of within-test-shard repeats noted, not train/test crossing).
**Threshold values (at the time of this section; current values are in `config/thresholds.yaml`):** `STATIC_ALLOW_MAX = 0.6163460957` (score < this → ALLOW), `STATIC_BLOCK_MIN = 0.9950119117` (score ≥ this → BLOCK, else ALERT) — derived via `find_threshold_for_fpr()` against calibrated test-set probabilities at target FPR 0.01 (ALLOW boundary, actual FPR 0.009999, detection rate 0.9803) and 0.001 (BLOCK boundary, actual FPR 0.000997, detection rate 0.9168).
> **RESOLVED, 2026-09-22.** Static BLOCK is no longer unconditional. It is
> interim-capped to ALERT in `decide()`, escalating to a real BLOCK only
> when corroborated by a MALICIOUS memory or network verdict on the same
> scan (`b11999f`, `ac132d2`), and the model was fully retrained this
> session with cal-split threshold derivation and a pre-registered
> acceptance gate (`8192452`, `438d936`). The "DOWNGRADED" / "should not
> be trusted" language below is 2026-08-28 history, not current status.

**Policy authority:** In code, static BLOCK is still first-checked and unconditional in `decide()` — but its production-readiness is **DOWNGRADED** as of 2026-08-28 (see the calibration-saturation finding below). Autonomous BLOCK should not be trusted until the two-stage saturation is fixed and the model is re-validated on a real-world, confirmed-label file set.
**Known limitations:** the recalibrated candidate (documented in Era 1) shows a PyInstaller-build-artifact false positive on `cortex-endpoint.exe` itself; the 5-file re-run (STEP 3) and the calibration-saturation finding (below) show cortex-ml's own static model has the same and worse.

**Result: cortex_static test AUC-ROC 0.9988, detection rate 98.17%, FPR 1.10% at threshold=0.5** (2,996 boosting rounds, ~2.1h wall time). This is on EMBER2024 dataset vectors; live `pe_features.py` extraction and the deployed Platt calibrator both diverge from it in ways documented below.

> **RESOLVED, 2026-09-22.** See the update note at the top of this
> "Commit 1" section and in the executive summary above — the Platt
> calibrator was refit on raw margins (`6eed759`), BLOCK was
> interim-capped with corroboration-gating (`b11999f`, `ac132d2`), and
> Static was fully retrained with split discipline this session
> (`8192452`, `438d936`). The root-cause analysis and tables below remain
> an accurate record of the original 2026-08-28 finding.

**Known limitation — calibration saturation (production-readiness DOWNGRADED, 2026-08-28).** A 57-file validation round showed calibrated static scores collapsing into two bands (benign ≈ 0.006, malicious ≈ 0.996) with almost nothing between, plus a confirmed cross-over: `extractor.exe` (benign, calibrated 0.9956) outranked two confirmed-malicious AgentTesla samples sitting just above the BLOCK threshold. Investigated by comparing, per file, the raw LightGBM booster margin `M` (`predict(raw_score=True)`), the booster probability `p = σ(M)`, and the deployed Platt-calibrated score `c`. (The AgentTesla samples and that round's artifacts are not on this machine; the mechanism was reproduced on the 5 cortex-endpoint validation files, four of which already pack into calibrated 0.984–0.996.)

| file | raw margin M | p = σ(M) | Platt(p) = c |
|---|---:|---:|---:|
| notepad_test.exe (benign) | −5.24 | 0.0053 | 0.0062 |
| svchost.exe (benign, signed MS) | **+1.87** | **0.867** | **0.984** |
| notepadd.exe (benign) | **+2.58** | **0.930** | **0.992** |
| benign_test_50mb.exe (benign) | +3.81 | 0.978 | 0.995 |
| extractor.exe (benign, PyInstaller) | +4.48 | 0.989 | 0.996 |

**Both stages contribute; they separate by margin range.** (1) The Platt calibrator is a near-step function: a `LogisticRegression` (`C = 1e10`, effectively unregularised) fitted on the booster's *probability* over the EMBER2024 val set — not on the raw margin, the textbook Platt input. Because the booster is near-perfectly separated on EMBER val (AUC 0.9988), the fit learned `coef_ = 10.66`, `intercept_ = −5.13` (`c = σ(10.66·p − 5.13)`): `p=0.7 → c=0.91`, `p=0.8 → c=0.97`, `p=0.9 → c=0.989`. It evacuates calibrated 0.05–0.55 and compresses everything with `p > 0.97` into calibrated [0.995, 0.996], destroying rank resolution exactly where real benign/malicious files interleave — a 0.6-unit raw-margin gap there collapses to ~0.0005 calibrated, which is how a benign file outranks true malware. (2) The boundary-zone raw margins are *moderate*, not extreme, and the feature-fidelity gap inflates them: `svchost.exe`'s raw margin is only **+1.87** (`p = 0.87`, "leaning malicious but clearly uncertain" — an ALERT, not a BLOCK), which the step-function calibrator turns into **0.984**. That +1.87 is itself inflated by the documented `pe_features.py`-vs-thrember gap (`header +3.75` vs thrember `+3.36`, against a `+1.166` booster bias); cortex-endpoint's thrember-based raw margin for the same file was **−0.10**.

**Connections:** the deep-benign band (≈ 0.006) is genuine booster-stage saturation on large-negative-margin files — not a defect. `extractor.exe` / `benign_test_50mb.exe` scoring high is primarily booster-stage (margins +4.48 / +3.81) — the same **PyInstaller / atypical-PE-structure false-positive pattern** flagged for the Era 1 malware-ml candidate and cortex-endpoint's own binary, on the feature side. The **feature-fidelity gap** (STEP 3, "The 5 test files") decides *whether* a real-world benign file has a positive margin at all; the **calibrator** then guarantees any positive-ish margin becomes a near-1.0 score with no resolution. Neither stage alone produces the two-band interleaving; together they do.

**Fix direction (reported only — no threshold change, which would relocate the band, not restore resolution):** refit the calibrator on raw margins, regularised, ideally with boundary-zone examples (addresses stage 2); close the `pe_features.py` fidelity gap against a thrember reference (addresses stage 1). Re-validate on a real-world, confirmed-label file set before restoring BLOCK authority.

### Commit 2 — `a035aa5`: "Cortex-Behavioral: dataset sourcing, training, evaluation, re-derived threshold, export"

- **Dataset sourcing — three real, independently-verified sources:** Mal-API-2019 (malicious only, 7,107 rows), MalbehavD-V1 (both classes, 2,570 rows), and Carpenter's `benign.json` (benign only, 101 rows — its malicious batches are deliberately never downloaded, to avoid unverified malicious labels). All three sources' API names are lowercased into one shared vocabulary space, verified at >96% clean casing-only overlap between Mal-API-2019/MalbehavD-V1 and >67% between Carpenter and the other two. Deduplication is done on record `id`, deliberately **not** on the `api_calls` sequence itself — MalbehavD-V1's shorter sequences legitimately recur across genuinely different real files by coincidence, so sequence-based dedup would have wrongly discarded roughly 1,400 distinct real samples.
- **Split leakage found and closed:** `scripts/split_behavioral.py` implements a grouped stratified 80/10/10 split by `(source, label)`, with every row sharing an identical `api_calls` sequence forced into the same split — this closes a real leakage bug a naive row-independent split had left in place (**85 sequences spanning multiple splits**), verified as a hard zero-leakage invariant.
- **Two ONNX export bugs found and fixed:** `torch.onnx.export()` defaults to the dynamo-based exporter on the torch version in use, which needs `onnxscript` (not a dependency) — fixed by forcing the legacy exporter (`dynamo=False`). Separately, `quantize_dynamic()` tried to quantize the model's Conv1d stack, which has no `ConvInteger` kernel in the ONNX Runtime CPU provider in use — the INT8 model failed to load until quantization was restricted to `MatMul`/`Gemm` ops only.

**Dataset & provenance:** Mal-API-2019 (GitHub, `ocatak/malware_api_class`) + MalbehavD-V1 (GitHub, `mpasco/MalbehavD-V1`) + Carpenter (Kaggle, benign-only subset) — 9,778 combined records before dedup.
**Leakage check:** 85-sequence cross-split leak found and fixed via grouped split; verified 0 sequences span more than one split after the fix.
**Threshold value (current):** `BEHAVIORAL_MALICIOUS_MIN = 0.60` — swept against val+test combined (1,835 rows, 274 benign — not enough for a precise FPR target). **Known limitation, documented explicitly and not silently accepted:** one benign sample (a MalbehavD-V1 trace with `setsockopt`/`ioctlsocket`/`wsastartup`/`getsockname` calls alongside routine registry/system calls, plausibly confusable with C2 setup) is misclassified at every threshold below ~0.922 — the model scores it 0.921006, confidently wrong. The threshold was deliberately **not** raised to clear this single case: n=1 isn't sound evidence for a permanent recall tradeoff, and behavioral has no rule-based backstop in this repo yet.
**Policy authority:** **Highest — behavioral MALICIOUS autonomously triggers TERMINATE**, the system's single most severe automated action, and checked first in `decide()` (rung 1), ahead of static's corroboration-gated BLOCK (rung 2).
**Known limitations:** the single hard false positive above; no rule-based overlay for high-risk call combinations independent of the ML score (flagged as the more targeted future fix, not blanket threshold tuning).

**Result: cortex_behavioral test accuracy 98.36%, precision 100%, recall 98.08%, AUC-ROC 0.9973** at threshold=0.60 (18 epochs, early-stopped, ~4 minutes wall time on CPU).

### Commit 3 — `25bd436`: "Cortex-Static: export the trained model to ONNX with calibration preserved"

**The calibration bug caught before ONNX shipping.** `export_static_lgbm_to_onnx()` had been written earlier but never actually run against the real trained model. Running it for the first time surfaced a real correctness gap: the raw LightGBM booster's output and `LGBMModel.predict_proba()` (which applies a Platt calibrator on top) **differ by up to 0.213** on real test data — and the deployed policy thresholds were derived against the *calibrated* distribution. Fixed by merging the Platt calibrator's `LogisticRegression` (converted via skl2onnx) into the same ONNX graph as the LightGBM booster (converted via onnxmltools), chained after a `Slice` that extracts the malicious-class probability column. Two more version-compatibility bugs were found and fixed along the way: onnxmltools 1.16.0's LightGBM converter caps at opset 15 (the code had it hardcoded to 17), and onnxmltools/skl2onnx produce different IR versions and duplicate tensor names that need explicit alignment before `onnx.compose.merge_models` will combine them.

**Verified against 10,000 real test rows:** mean absolute diff ≈0 vs `predict_proba()`, max absolute diff 0.0032 (float precision noise), **zero verdict differences at the actual production thresholds**. `cortex_static.lgbm` (78.8MB) → `cortex_static.onnx` (61.4MB). No quantization step for the static model — tree ensembles are unaffected by INT8 weight quantization, so the FP32 export is the deployed artifact.

### Commit 4 — `0e295c7`: "README: fix broken end-to-end scan example for the behavioral model"

A documentation-only fix: the README's example loaded `cortex_behavioral_full.pt` (a filename training never actually produces — the real output is `cortex_behavioral_best.pt`) via `torch.load()` alone and expected a callable model back. The checkpoint is a raw `state_dict`, not a pickled full model, so it needs `CortexBehavioralNet()` constructed first, then `load_state_dict()`.

---

### Signal 3 — Cortex-Memory (uncommitted; fully built, trained, exported)

**Dataset & provenance.** CIC-MalMem-2022 (Carrier et al., ICISSP 2022) — a public research dataset, no licensing ambiguity. The official UNB/CIC portal requires a manual download form; the working copy was pulled from the Kaggle mirror `luccagodoy/obfuscated-malware-memory-2022-cic` on 2026-08-25 instead, because the official portal's download was missing 3 of the dataset's 4 top-level categories at the time of the pull. 58,596 rows, 55 VolMemLyzer-derived numeric memory-forensics features, `Category` (family + `Benign`) and `Class` (`Benign`/`Malware`) columns. Schema validated structurally against the real file (column count, dtypes, exact `Class` value set, first/last column identity) rather than trusted from documentation.

**Real bugs found and fixed:**
1. **Grouping-detection bug.** The first version of the split's "is this real per-sample grouping?" check used only `median group size > 1`, which incorrectly treated the constant `"Benign"` Category value as a real group (one giant group whose size equals the entire benign row count trivially has a huge median) — caught before running on real data, via a synthetic dataset shaped like the real one, where it silently put all 1,800 synthetic benign rows into a single split. Fixed by also requiring more than one distinct group value.
2. **Duplicate-hash check crash.** `verify_zero_duplicate_feature_hashes` tried to hash the `split` column itself (a string column added after the fact), crashing when casting `"train"`/`"val"`/`"test"` to float64. Fixed by excluding it explicitly.
3. **Real duplicate leakage found on the actual 58,596-row dataset:** 20 duplicate feature-vector hashes (80 rows, 0.14% of the dataset) spanned multiple splits — 8 groups/55 rows benign-only (no per-sample identifier exists for benign rows to protect against this), 11 groups/25 rows cross-malware-family collisions (different samples, coincidentally identical coarse VolMemLyzer summary stats). Fixed via a union-find merge (`_assign_merged_groups`) that unions sample-id groups with any exact-duplicate-feature-vector groups before allocating rows to splits, making cross-split duplication structurally impossible rather than something to catch after the fact.

**Leakage-check results (final, verified):** 0/32,106 sample-id groups span more than one split; 0/58,596 duplicate feature-vector hashes span more than one split.

**Threshold derivation and value at the time:** `MEMORY_MALICIOUS_MIN = 0.0005358335957155212` *(superseded 2026-09-10 by `0.0024964628`, re-derived at the same target on a dedicated `cal` split; see `config/thresholds.yaml`)* — derived via `find_threshold_for_fpr()` at target_fpr=0.01 against calibrated val+test-combined probabilities (5,860 benign rows — far more statistical power than Behavioral's 274). Test-set metrics at this threshold: **AUC-ROC=1.0, precision=0.9891, recall=1.0, F1=0.9945, FPR=1.13%**.

**Policy engine authority and why:** Evaluated as an **independent signal**, not gated behind static's ALLOW — Memory's purpose is catching injected/fileless malicious activity that structurally has no on-disk file for static to see in the first place, so gating it behind static would defeat that purpose. Its authority is **capped at ALERT**, one rung below Behavioral's TERMINATE, for a specific, diagnosed reason, not a generic "new model, be cautious" hedge: **22 of the model's 62 raw+derived features individually exceed 0.95 AUC on their own** (confirmed independently on both train and held-out test), consistent with CIC-MalMem-2022's own documented collection methodology — every benign sample is a repeated capture of "normal user behavior" on a single baseline Windows 10 VM, while malicious samples span far more varied executions. The model may be learning "does this look like that one baseline VM" more than "is malicious behavior present," a distinction this dataset alone cannot resolve.

**Known limitations, documented in code and README:** (1) the single-VM-benign separability finding above; (2) benign near-duplicates that are *not exactly* identical (only exact duplicates are caught and merged) remain unprotected, since nothing in the dataset schema identifies which benign rows come from the same underlying capture session; (3) cross-family malicious duplicate-feature-vector collisions are informational, not a leakage risk, but reveal that VolMemLyzer's coarse summary statistics have real collisions independent of sample identity; (4) real-world validation would require memory captures from multiple genuinely different (non-baseline) benign machines, not more rows from this same dataset.

**ONNX export:** verified against the real model and the actual 5,930-row test split — mean abs error 6.7e-9, max abs error 2.0e-6, **zero prediction mismatches at the operating threshold** across all test rows.

### Signal 4 — Cortex-Network (uncommitted; fully built, trained, exported)

**Dataset & provenance.** CSE-CIC-IDS2018 (Sharafaldin et al., CIC), chosen over UNSW-NB15 specifically because UNSW-NB15 has a documented licensing ambiguity while CSE-CIC-IDS2018 has an explicit, clear redistribution license. Hosted directly on AWS Open Data (`s3://cse-cic-ids2018/`, `ca-central-1`), no account or access-request form required. Only the `Processed Traffic Data for ML Algorithms/` prefix (10 CICFlowMeter CSVs, 6.4 GiB) was synced — not `Original Network Traffic and Log data/` (raw PCAPs). 16,233,002 raw rows across 10 capture days (Feb–Mar 2018).

**Real bugs and data-quality issues found and fixed, each handled deliberately rather than papered over:**
1. **Column count not uniform**: 9/10 files have 80 columns; the `Thuesday-20-02-2018` file (also the size outlier, ~4 GiB vs ~330 MiB typical) has 84 — 4 extra leading identity columns (`Flow ID`, `Src IP`, `Src Port`, `Dst IP`). Dropped, not merely aligned: CSE-CIC-IDS2018 was captured on a small, fixed testbed, and keeping these columns would let a model memorize which of the testbed's fixed machines played the attacker role rather than learn a transferable traffic pattern — the same class of shortcut-learning risk documented for Cortex-Memory.
2. **`Timestamp` excluded from the feature set** (metadata only) — each capture day is (almost) entirely one attack scenario, so a raw timestamp would let a model learn "which day this is" as a near-perfect label proxy.
3. **59 literal header-repeat rows** (across 3 of 10 files) filtered via an allowlist of the real observed Label values.
4. **`Flow Byts/s`/`Flow Pkts/s` NaN/Infinity values** (division by zero when Flow Duration=0) — rows dropped, not imputed.
5. **Label taxonomy inconsistent across files** (`"DDoS attacks-LOIC-HTTP"` vs `"DDOS attack-HOIC"`) — sidestepped for the binary target via a case-insensitive `!= "benign"` check rather than an exact-match allowlist; the raw string is kept for future multi-class work.
6. **Memory-safety**: the raw 16.2M-row dataset was never held fully in memory — a two-phase design (cheap Label-only scan → per-file load-clean-and-immediately-downsample) capped the working dataset at 2,000,000 rows, chosen for headroom on the training machine while giving far more statistical power than Memory needed. Measured: peak RSS 22.8 GiB, ~8.5 min wall clock.
7. **Massive exact-duplicate rate found and handled**: 495,181/2,000,000 rows (24.75%) were exact duplicates — far higher than Memory's 0.9% — 91,221 groups, sizes 2–18,376, largely explained by repetitive automated attack traffic and simple/idle benign flows. The same union-merge approach as Memory was applied.
8. **4,742 duplicate groups spanned both benign and malicious labels** — genuine label ambiguity in CICFlowMeter's feature representation, not leakage. **Excluded entirely from train, val, and test** (48,329 rows, 2.42% of the capped dataset), saved separately for inspection rather than silently discarded.
9. **A specific, accepted consequence of the leakage guarantee**: the single largest duplicate group (18,376 rows, spanning FTP-BruteForce and DoS-SlowHTTPTest — both apparently producing an identical "minimal/degenerate flow" signature) is one indivisible block that had to land almost entirely in val/test, leaving train with very few examples of these two attack types. Left as-is deliberately, documented explicitly as a train-support caveat rather than special-cased.
10. **A real `stratified_group_split` index bug**, found while adding the ambiguous-group exclusion: boolean-mask filtering left gaps in the DataFrame index, and the split allocation code wrote into a plain `np.empty(len(df))` array using those raw (non-contiguous) index values instead of positions — worked only by coincidence before this exclusion step existed. Fixed with an explicit `reset_index(drop=True)`.

**Leakage-check results (final, verified):** 0 duplicate-feature-vector groups span both benign and malicious labels (post-exclusion); 0 duplicate feature-vector hashes span more than one split (1,591,298 unique vectors / 1,951,671 rows).

**Final split:** 1,951,671 rows (2,000,000 capped minus 48,329 excluded ambiguous rows) — train=1,526,757 (78.23%, benign=1,286,894/malicious=239,863), val=211,697 (10.85%, 160,864/50,833), test=213,217 (10.92%, 160,869/52,348).

**Threshold derivation and value at the time:** `NETWORK_MALICIOUS_MIN = 0.9441855970306654` *(superseded 2026-09-10 by `0.6672636218`, re-derived at the same target on a dedicated `cal` split; see `config/thresholds.yaml`)* — derived at target_fpr=0.001 against calibrated val+test-combined probabilities (424,914 rows, 321,733 benign — resolution ~0.0003% per false positive, far finer than Memory's or Behavioral's). Test-set metrics: **AUC-ROC=0.9972, AUC-PR=0.9951, precision=0.9969, recall=0.9752, F1=0.9859, FPR=0.098%**.

**Per-attack-type finding, checked rather than assumed clean:** a single-feature-AUC sweep (the same check that caught Cortex-Memory's shortcut) found **0 of 78 features exceed 0.95 AUC individually** on either train or test — no single-VM-style artifact here. But per-attack-type detection rates revealed a genuine, non-training-artifact weak spot: **Infiltration detection is only 10.75%** despite 11,463 train rows (ruling out under-training — every other class with under 10,000 train rows still scored ≥91%). Independently confirmed as a documented CICFlowMeter feature-representation ceiling: infiltration attacks are slow/low-volume and don't produce the distinctive flow-statistics signature CICFlowMeter captures well for DDoS/brute-force; a comparative study found infiltration on this dataset is only well-detected via NetFlow-derived features. This was deliberately **not** chased by lowering the threshold, which would spike FPR across every well-detected class.

**Policy engine authority and why — two independent, evidence-backed legs, not one:** capped at ALERT, same rung as Memory, for **(a)** the Infiltration blind spot — Network structurally cannot be trusted alone for slow/low-volume attacks, exactly the threat class Behavioral (API-call sequences) and Memory (process/injection artifacts) have process-level visibility into that Network's flow-only view doesn't, so the system's overall coverage still holds even though Network alone doesn't — and **(b)** a documented external-validation-collapse pattern: models reported as "nearly perfect" on CSE-CIC-IDS2018, which this one's test-set numbers are, have been independently shown elsewhere to degrade toward random performance on external/real-world traffic.

**Known limitations:** the two above, plus an independently-reported **~7.5% label-noise ceiling** in CSE-CIC-IDS2018's automatic time-window-based labeling (not human-verified per-flow) — a dataset-level ceiling this pipeline cannot detect or correct from inside the dataset itself.

**ONNX export:** shipped as final, with a fully explained, external, documented limitation. Mean abs error 2.1e-4, **max abs error 0.5465** — three orders of magnitude looser than Static's/Memory's ~1e-6–2e-6. Root-caused to a confirmed, documented ONNX TreeEnsemble spec limitation (LightGBM stores split thresholds as float64; ONNX's tree operator only supports float32; the official sklearn-onnx documentation states there is no fix short of a custom double-precision runtime) made visible here specifically because Network's Flow Duration/IAT features run up to ~1.15e8, unlike Static's/Memory's smaller-magnitude features. Practical impact, measured precisely: only 3/213,217 predictions (0.0014%) flip at the operating threshold, two of them exact threshold-boundary ties; aggregate metrics shift only in the 4th decimal place. A theoretical mitigation (log-transform or clip the largest-magnitude duration/IAT features before training) is documented as a future option, not applied, given the negligible current impact.

### Signal 5 — Cortex-Emulation (uncommitted; trained 2026-08-27 — **report-only / additive, not in the policy decision**)

**Dataset & provenance.** Quo Vadis (Trizna et al., 2022/2024), HuggingFace `dtrizna/quovadis-speakeasy` — Speakeasy-emulator-generated malware/benign execution reports. **License verified explicitly before downloading anything**, matching the discipline applied to CSE-CIC-IDS2018: Apache-2.0, confirmed both via the HF repo's `license` tag and the actual card text (no additional restriction found beyond the bare tag). 92,700 files (75,298 train collected Jan 2022, 17,402 test collected Apr 2022 — a deliberate concept-drift split by the dataset's authors, preserved rather than re-randomized), ~5.5 GiB.

**Real bugs and data-quality issues found and fixed:**
1. **The dataset card's documented schema is wrong.** It describes each file as `{"sha256": ..., "entry_points": [...]}`. The real files are a bare JSON list with no wrapper object and no `sha256` field in the contents at all — recoverable only from the filename. A loader trusting the card's example would break immediately.
2. **One category uses a completely different filename scheme**: `report_windows_syswow64` (294 files total, 100% of that category, 0% of every other category) is named by the literal Windows system-binary filename (e.g. `AppVDllSurrogate.json`) instead of a sha256, since these are known-legitimate system files rather than anonymized malware samples. Found only when the loader crashed against the real full dataset; fixed by renaming the identifier concept from `sha256` to `sample_id` with a documented fallback.
3. **The card's own row-count table is off by 3** from what the repository actually contains (75,298 real vs. 75,301 claimed), and its prose numbers (76,126/17,407) are more stale still — verified via a full diff against the real repo file listing, not trusted from either card number.
4. **A multi-stage download-reliability saga**, resolved rather than worked around blindly: unauthenticated downloads were rate-limited to ~5 files/sec (5+ hour ETA); an HF token raised this only marginally; the real bottleneck was identified as HuggingFace's newer "xet" chunked-transfer protocol (confirmed via the error trace, `xet-read-token` 429s), and disabling it (`HF_HUB_DISABLE_XET=1`) tripled throughput; at high worker counts (24–32) the server then applied cumulative-volume rate limiting (429s followed by connection resets), which a naive retry loop couldn't outrun since each restart ramped back to the same limit within 1–2 minutes; settled on 6 workers with a 30-second backoff and an automatic retry wrapper, which completed across 7 attempts. **Completeness verified via a full diff against the real repo listing — 92,700/92,700 files, zero missing** — not just a count match.
5. **A real `has_error` bug**: the field is `{}` (empty dict) when nothing went wrong, not `null`; the first version of the loader used `entry.get("error") is not None`, which incorrectly flagged **100%** of all 167,548 entries as errored (`{} is not None` evaluates `True`). Fixed to a truthy check; the real rate is 73.2%.
6. **A finding that reversed under the requested full-scale re-check**: a 47-file pilot sample suggested errored entries have *longer* API sequences than error-free ones. Re-verified against the full dataset as explicitly instructed, and the finding **reversed**: error-set entries have a much *lower* median API count (1) than error-free ones (76) — the pilot sample happened to be almost entirely `module_entry` type, while the full data showed `thread`/`tls_callback_*` entries (which are near-empty and dominate the error population) driving the opposite pattern.
7. **A major duplication finding, investigated rather than accepted**: 96.8% of all 167,548 entry-point rows were exact duplicates (vs. Memory's 0.9%, Network's 25%). Traced to two causes: (a) 97.8% of non-`module_entry` rows (`thread`/`tls_callback_*`, 74,897 rows) have ≤1 API call — near-empty structural noise; (b) even `module_entry` alone (the real behavioral signal) is 94.3% duplicated, a genuine property of the dataset likely reflecting malware-builder-kit-generated variants that are byte-different but API-name-identical. An initial split attempt, before this was understood, produced a val set that was 90.2% malicious (74% ransomware alone) because one 4,312-row duplicate group could single-handedly dominate a split's composition.
8. **The `apihash` field was checked, not assumed sufficient**: validated against a hand-computed sequence hash in both directions — 0/8,737 false splits, but **686/6,699 apihash groups (10.2%) were false merges** (one apihash value covering genuinely different sequences). Not safe to use alone; the pipeline falls back to the hand-computed hash.

**Fix applied (the key design decision for this signal):** the modeling dataset is now **restricted to `module_entry` rows only** — `thread`/`tls_callback_*` rows remain in the canonical parquet (nothing is discarded from the dataset) but are excluded from train/val/test. Within `module_entry`, exact-duplicate API-name sequences are **collapsed to one representative row per unique sequence** (not grouped-and-kept-together, since one dominant group here is large enough to single-handedly skew a split), with a `duplicate_count` column recording how many original rows shared each surviving sequence — deferred for possible frequency-weighted use, not lost. Collapsing is done **separately** for the train (Jan 2022) and test (Apr 2022) partitions, never across them, preserving both "test is never touched" and the concept-drift comparison the authors' split exists for. **49 of 92,254 files (0.05%, 80% of them benign) have zero `module_entry` rows** and are excluded from the modeling population entirely — counted and reported precisely, not silently dropped.

**Leakage-check / separability-check results (final, verified):** `duplicate_count` sums exactly back to the pre-collapse row counts (75,263 for train+val, 17,388 for test) — no data silently lost. Every split is internally 100% unique sequences; zero sequence overlap between train and val. A Memory-style separability check was run and found **no equivalent shortcut**: single-feature AUC of raw sequence length alone is 0.5605 (barely above chance); the top 5 benign unique sequences cover only 21.4% of benign rows (top 1 alone, 12.8%, confirmed by direct inspection to be generic MSVC CRT startup boilerplate — expected and explainable, not a red flag the way Memory's single-VM finding was).

**Final split:** train=6,070 unique sequences (benign=4,044/malicious=2,026), val=675 (450/225), test=2,495 (1,936/559). Family representation is thinnest for `rat` (60 train) and `keylogger` (112 train) — flagged for per-class monitoring once trained, the same caution applied to Network's low-support attack types.

**Sample-weighting decision (reasoned, not defaulted): equal weighting, not `duplicate_count`-based.** Real numbers: train `duplicate_count` averages 5.7 for benign vs. 22.5 for malicious (max 4,312). Rejected a frequency-based weight for three reasons: it would reintroduce the exact single-group-domination problem the collapse was designed to fix, just via a softer dial; duplicate count conflates genuine prevalence with data-collection artifacts (builder-kit variants, shared compiler boilerplate) already shown by direct inspection to dominate the largest groups; and it would compound uncontrollably with the class-imbalance correction the model already needs (malicious sequences average ~4x higher duplicate count than benign). `duplicate_count` is retained in the processed data for possible future work, not discarded.

**Tokenizer and model, built and verified — training not yet run:** `tokenizer/emulation_tokenizer.py` (duplicated from, not sharing code with, Behavioral's `ApiTokenizer`, since its sequence length is a hardcoded module constant) — vocabulary built from **train only**, real vocab size **3,154** (3,152 real API names + `<PAD>`/`<UNK>`, roughly 8x Behavioral's), `MAX_SEQ_LEN=500` justified by the real final-split distribution (p90–p99 sit at 460–501 across train/val/test). `models/emulation_cnn.py` adapts Cortex-Behavioral's 1D-CNN + self-attention architecture — same shape, three deliberate deviations documented in the module: `embed_dim=64` (not Behavioral's 128, given an 8x larger vocabulary and no more data), `dropout=0.4` (not 0.3), `batch_size=64` (not 128); conv/attention capacity kept unchanged as a reasoned starting point, since Behavioral's own real train set (~7,340 examples) is the same order of magnitude as this signal's 6,070, not a case of reusing capacity proven on a much larger dataset. Verified with a real forward pass on the real train tensors: 777,345 total parameters (543,489 excluding embedding tables), tensor shapes `X_train (6070, 500)`, `X_val (675, 500)`, `X_test (2495, 500)`. A concrete overfitting fallback is pre-committed (drop `embed_dim` to 32 and/or remove one conv block) rather than left as a vague "watch for it."

### Training run, capacity ablation, and the drift diagnosis (2026-08-27)

**Training discipline.** `models/train_emulation.py` + `scripts/train_emulation.py`, mirroring `train_behavioral.py`'s shape with two deliberate deviations: early stop on **val ROC-AUC** (not loss — only the patience / best-state-restore mechanism is reused), plus an explicit **train/val AUC-gap overfitting monitor** (stop if `train_auc − val_auc > 0.05` for 3 consecutive epochs, and *report* — not auto-apply — the pre-committed `embed_dim`→32 fallback). Raw sigmoid output, no calibration layer: matches Cortex-Behavioral, and the Platt-calibration bug from Cortex-Static's ONNX export was LightGBM-`predict_proba`-specific with no torch-path analogue (checked explicitly, not skipped). Seed 42, CPU, `find_threshold_for_fpr()` duplicated self-contained into the module.

**Primary run (`embed_dim=64`).** Hit the **overfitting stop at epoch 22** (best epoch 19, val AUC 0.9495). Val AUC plateaued ~0.94 by epoch 8 while train AUC ran to 0.995 — the model memorizes the 6,070 collapsed training sequences trivially. Held-out test AUC-ROC 0.8737. Threshold sweep on val+test combined (2,386 benign, **explicitly flagged as pooling the Jan-2022 and Apr-2022 eras**): at the recommended **target_fpr=1% → threshold 0.999359**, pooled detection is only **49.9%** (23/2,386 benign FP); 0.1% (~2 FP) flagged too thin to trust, same caution as Cortex-Network's tightest targets. The score distribution is strongly bimodal — the 1%-FPR threshold sits at 0.9994.

**Known limitation 1 — ablation-confirmed Jan→Apr temporal concept-drift collapse.** At the 1%-FPR threshold, **malicious recall drops from 70.7% (Jan-era slice) to 41.5% (Apr-era slice)**, AUC-ROC 0.949 → 0.874; benign FPR holds ~0.7–1% in both eras, so the loss is entirely malicious-side. This was checked against the pre-committed fallback rather than assumed: re-running at **`embed_dim=32`** made the overfitting stop fire *earlier* (epoch 18, best epoch 10, val AUC 0.9396 — worse on every axis) and **left the drift gap unchanged** — recall 68.0% → 38.3% (29.7 pp gap vs 29.2 pp at `embed_dim=64`), pooled detection at 1% FPR 49.9% → 46.8%. **Capacity reduction fixed nothing** — this rules out model size / overfitting as the relevant lever; the drift is a genuine train/test distribution shift.

**Per-family drift diagnostic (descriptive, no retraining).** Comparing `coinminer` (test detection 17.1%, sharp; 8.0% at `embed_dim=32`) against `ransomware` (48.3%, mild) — both well-supported by raw volume:
- *Train-side:* `coinminer` has only **185 unique training sequences** (97 singletons); `ransomware` has **1,065** (509 singletons). Raw-volume concentration is actually *higher* for ransomware (one builder-kit sequence = 48% of family volume, `duplicate_count` 4,312), so the real difference is **distinct-sequence variety** — ~5× more for ransomware.
- *Test-side recurrence:* both recur against train at ~the same rate — `coinminer` 21.6% exact / 53.4% near-neighbour (token-set Jaccard ≥ 0.90); `ransomware` 17.2% / 53.4%. The "Apr `coinminer` doesn't recur in Jan training" hypothesis is **not supported**.
- *Detection by match status (e64):* `ransomware` tracks match quality (exact-match Apr rows 70% detected, near-only 29%) — a real but gentle memorization-vs-generalization gap. `coinminer` is **uniformly ~15–25% across every bucket including exact matches** (mean score 0.675, below threshold) — the model never built a firing region for it; its Apr sequences' nearest training neighbours are benign more often than `coinminer`.
- *Mechanism:* the model **memorizes token sequences rather than learning generalizable behaviour** — it fires near what it memorized and goes quiet on modest drift, and for a family with too thin a distinct-sequence base never builds a firing region at all. Capacity tuning cannot address either.

**Known limitation 2 — thin margin over a trivial baseline.** Test accuracy at the 1%-FPR threshold is **86.4%** (e64) / **85.6%** (e32), versus majority-class (predict-all-benign) **77.6%** and an **exact train-sequence duplicate-lookup 82.2%** (571/2,495 test sequences appear verbatim in train). The model beats the lookup table by only **+4.1 pp** (e64) / **+3.3 pp** (e32).

**Policy engine authority: report-only / additive — no branch in `decide()`.** `EMULATION_MALICIOUS_MIN` is set (0.999358594, the 1%-FPR point) and `EmulationVerdict` / `emulation_verdict_from_score()` exist, but **for logging/telemetry only** — nothing routes an emulation verdict into the final decision, and `config/thresholds.yaml`'s `emulation:` block is annotated the same way. This is the disposition malware-ml gave its own Behavioral v2 category signal (`deployment_status: additive_report_only_not_in_policy_decision`), and it is deliberately **stricter than Memory's or Network's ALERT cap**: a signal that recovers ~40% of malware on the next collection era and barely outperforms a lookup table cannot carry autonomous *or* ALERT authority. Revisit only with a materially different model, not more tuning.

**Future work (not committed to now):** a genuine fix would most likely need **either** engineered behavioural-category features in place of raw API-token sequences (hypothesis: coarser file/registry/network/injection activity classes may be more temporally stable across collection dates than exact API-call sequences — worth testing, not guaranteed) **or** training data spanning more collection dates. Explicitly **not** further hyperparameter tuning — the `embed_dim` ablation already ruled out capacity as the lever.

**Not done for this signal:** no ONNX export, no `inference/pipeline.py` wiring, and the second pre-committed fallback (drop a conv block) was not pursued — the larger capacity cut already made things worse. The args/ret_val detail inside each `apis` entry and the sparse `registry_access`/`file_access`/`dropped_files`/`process_events` categories remain deferred to a documented future v2, preserved in the canonical parquet's `raw_entry_json` column.

---

## STEP 3 — Cross-era comparison (Static and Behavioral only — Memory, Network, and Emulation have no prior-era equivalent to compare against)

### Static model: dataset strategy and results

| | malware-ml (deployed baseline) | malware-ml (best candidate, unshipped) | cortex-ml |
|---|---:|---:|---:|
| Training rows used | 1,200,000 | 2,140,000 | **2,340,000 (full dedup train set)** |
| % of available deduplicated data | **51.3%** (documented, deliberate, RAM-constrained) | ~91.5% | **100%** |
| Memory-constraint solution | Subsampling to fit in RAM | Disk-backed single-pass pipeline | **Swap file at the OS level** |
| Test AUC | 0.99774 | 0.99867 | 0.9988 |
| Test FPR | 0.10519% | 0.09815% (recalibrated) | 1.10% (at threshold 0.5, a much less conservative operating point) |
| Test TPR / detection rate | 88.365% | 90.853% | 98.17% |
| Status | **Deployed** (despite being outperformed) | Best result in the project, **never promoted** | Current/canonical |

Two genuinely different fixes to the same underlying memory-pressure problem: malware-ml's disk-backed pipeline (Candidate V1) and cortex-ml's swap-file approach both succeeded in training on the full dataset, arrived at independently. cortex-ml's headline FPR number isn't directly comparable to the other two without matching the operating threshold — its threshold (0.5) targets a different point on the ROC curve than the 0.1%-FPR-oriented thresholds malware-ml calibrated toward; what's comparable cleanly is AUC (all three cluster tightly around 0.998–0.999) and the fact that cortex-ml trained on 100% of the available data where malware-ml's *shipped* model used just over half.

### Behavioral model: dataset strategy and the threshold-generalization lesson

| | malware-ml (Model A, deployed) | cortex-ml |
|---|---|---|
| Data sources | Oliveira API-call CSV only | Mal-API-2019 + MalbehavD-V1 + Carpenter (3 sources) |
| Rows | 43,876 (43,865 after MD5-conflict exclusion) | 7,107 + 2,570 + 101 = 9,778 combined |
| Vocabulary | 307 named APIs + `<UNK>` | Built from combined vocabulary space, casing-normalized across sources |
| Test AUC | 0.9923 | 0.9973 |
| Test recall/TPR | 89.78% | 98.08% |
| Threshold-selection approach | Validation-only threshold, frozen before test | Val+test sweep, deliberately conservative (kept at 0.60 rather than chasing one hard case to 0.922+) |
| **Real-world generalization result** | **Near-zero FPR on validation → 3.09% FPR on test (5/162 FP)** | Not yet subjected to an equivalent held-out generalization stress test at this report's writing |

This is the direct, load-bearing evidence for why cortex-ml's threshold-selection discipline matters, not a hypothetical: malware-ml's Model A picked a threshold using only the validation set, that threshold looked excellent on validation, and then measurably failed to generalize the moment the test set was opened. cortex-ml's README documents its own threshold decision explicitly as a val+test sweep with a stated, reasoned refusal to over-fit to a single hard example — the same discipline was then independently repeated for Memory, Network, and Emulation, each re-deriving its own threshold from its own data rather than inheriting any prior number — and for Emulation the val+test sweep additionally surfaced the concept-drift collapse (by reporting val-era and test-era metrics separately) that led to its report-only disposition.

### The 5 test files: run 2026-08-27 — pipeline works end-to-end, but the comparison surfaced (and half-fixed) a real feature-extraction bug

`inference/pipeline.py::CortexPipeline.scan()` was run against all five files (`svchost.exe`, `notepad_test.exe`, `benign_test_50mb.exe`, `notepadd.exe`, `extractor.exe`), each byte-verified by sha256 against cortex-endpoint's own validation set. Only the **static** signal can genuinely run: a bare `.exe` carries no API-call trace, no memory dump, and no network flows, so behavioral/memory stay `NOT_PROVIDED` — and **network is not wired into the pipeline at all** (`scan()` has no network parameter; `policy_engine` has the threshold + verdict function but `pipeline.py` was never extended). This matches how cortex-endpoint's own run showed `behavioral_state: not_started` for these files.

| file | cortex-endpoint (Era 2) | cortex-ml (post-signify-fix) | agree? |
|---|---|---|---|
| svchost.exe | 0.4744 / allow | **0.9839 / ALERT** | ❌ — feature-fidelity gap (below) |
| notepad_test.exe | 0.0057 / allow | 0.0062 / ALLOW | ✅ |
| benign_test_50mb.exe | 0.9690 / alert | 0.9950 / BLOCK | ❌ — threshold-scheme + boundary tie |
| notepadd.exe | 0.7746 / alert | 0.9917 / ALERT | ✅ (verdict) |
| extractor.exe | 0.9820 / block | 0.9956 / block | ✅ |

**Bug found and fixed:** `features/pe_features.py` imported `from signify.authenticode import SignedPEFile`, an API that no longer exists in the installed **signify 0.9.2** (renamed to `AuthenticodeFile`). The module's `except ImportError` swallowed this silently, so `_SIGNIFY_AVAILABLE = False` and the entire 8-dim `authenticode` feature group returned **all-zeros for every file at inference time** — a train/serve skew, since EMBER2024's training vectors have real authenticode features. Ported forward to the 0.9.x API (`AuthenticodeFile.from_stream(...)` / `iter_signatures()`); verified `svchost.exe` now parses a genuine chain (`num_certs=1`, `chain_max_depth=2`, real countersigning timestamp), and the two unsigned files correctly show `num_certs=0` with `parse_error=0`. The fix corrected `svchost.exe`'s `authenticode` group contribution from **+1.17 (wrongly toward malicious)** to **−1.33 (toward benign)**, moving its score 0.9913 → 0.9839.

**Second gap, not fixed — the honest result:** `svchost.exe` still does **not** move toward ALLOW (0.9839 vs cortex-endpoint's 0.4744). The remaining ~2.0 raw-margin gap is a broader feature-fidelity problem: cortex-ml's `pe_features.py` produces a vector systematically shifted toward "malicious" versus the thrember reference — `header` pushes harder (+3.75 vs +3.36), benign-side groups push softer (`histogram` −0.51 vs −0.83), and cortex-ml's hand-rolled 8-feature `authenticode` group carries less signal than thrember's even when populated (−1.33 vs −2.12). `benign_test_50mb.exe`'s alert→BLOCK flip is separate and expected: an unsigned file (authenticode ≈ 0 in both), close raw margins, but cortex-ml's calibrated 0.995045 lands ~3e-5 above its own much stricter BLOCK threshold (0.995012 vs cortex-endpoint's 0.9746).

**The real gap this exposed:** cortex-ml has **no automated feature-parity test** comparable to cortex-endpoint's `tests/test_static_feature_parity.py` (which checks live extraction against a pinned thrember reference on real PEs, signed one included). Nothing currently compares `pe_features.py` output against a reference extractor on real files — exactly the check that would have caught the signify regression the moment it happened, instead of it silently degrading every live static score. Building one is now the top static-path open item (below).

---

## STEP 4 — Conclusion

### What's proven

- The core detection approach — now five independent signals combined through rule-priority (never averaged) decisions — works, with each signal's authority level explicitly tied to evidence about what it can and cannot be trusted to decide alone. Static and Behavioral carry full autonomous authority (BLOCK and TERMINATE respectively), each validated across three independent implementations with consistently strong AUCs (0.997–0.999 static, 0.992–0.997 behavioral). Memory and Network are real, working, fully-trained, ONNX-exported signals with excellent held-out test metrics (AUC 1.0 and 0.997 respectively) — but both are deliberately capped at ALERT, each for its own specific, diagnosed reason (Memory: single-VM-benign separability; Network: an Infiltration blind spot plus a documented external-validation-collapse pattern for this dataset), not a blanket policy. Emulation is trained but held **report-only / additive** (not in `decide()` at all): an `embed_dim=64`-vs-`32` ablation confirmed a Jan→Apr 2022 temporal concept-drift collapse (malicious recall ~70%→~40%, unchanged by capacity reduction), and the model beats a trivial train-sequence duplicate-lookup baseline by only ~3–4 pp — a stricter, separately-earned disposition, mirroring malware-ml's own Behavioral v2 precedent.
- Every one of the five signals independently repeated the same core discipline: validate schema against real data rather than documentation, run a real leakage check rather than assume clean data, and re-derive its own threshold from its own trained model rather than inherit a placeholder. Each surfaced real bugs in the process — the OOM/dataset-duplication sagas for Static, the split-leakage/ONNX bugs for Behavioral, a grouping-logic bug and 20 real duplicate-hash leaks for Memory, a severe (96.8%) duplication finding and a filename-scheme exception for Emulation, and a documented, externally-confirmed ONNX precision limitation for Network.
- cortex-ml's ONNX exports for the three tree-ensemble signals (Static, Memory, Network) are all verified numerically against `predict_proba()` on real test data, not assumed correct from successful conversion alone — one of the three (Network) surfaced a real, now-explained precision gap in the process.

### What's still open

- ~~**Cortex-Static production-readiness is DOWNGRADED (2026-08-28)** — autonomous BLOCK authority should not be trusted until fixed and re-validated.~~ **RESOLVED, 2026-09-22:** the Platt calibrator was refit on raw booster margins (`6eed759`), Static BLOCK was interim-capped to ALERT with corroboration-gated escalation (`b11999f`, `ac132d2`), and Static was fully retrained this session with cal-split threshold derivation and a pre-registered acceptance gate (`8192452`, `438d936`; see `docs/PhantomCortex_Static_Retrain_Report.pdf` and `reports/static_retrain_20260921/`). The original finding is kept below as historical root-cause record: a 57-file round found calibrated scores collapsing into two bands (~0.006 / ~0.996) with a confirmed benign-outranks-malware cross-over. Root-caused to a two-stage compounding failure: (1) an unregularised Platt calibrator fitted on the booster's *probability* over a near-separable EMBER val set → a near-step function (`coef_ = 10.66`) that evacuates the mid-range and compresses `p > 0.97` into calibrated [0.995, 0.996]; (2) the `pe_features.py` fidelity gap inflating real-world benign files' *moderate* raw margins into positive territory (`svchost.exe` raw margin +1.87 vs thrember's −0.10). Full side-by-side in the Cortex-Static / Commit 1 section.
- **cortex-ml has been run against the 5 real-world test files** (2026-08-27) — the pipeline works end-to-end (static-only; behavioral/memory have no input, network wasn't wired in yet -- wired 2026-09-08, `a8e6f45`). Verdicts agree on 3/5. `svchost.exe` disagrees (cortex-endpoint allow / cortex-ml ALERT): a silent signify-0.9.2 import failure had zeroed the whole `authenticode` feature group at inference time — now fixed by porting to the 0.9.x API — but `svchost.exe` still doesn't reach ALLOW, revealing a broader feature-fidelity gap (`pe_features.py` output is systematically shifted toward "malicious" vs the thrember reference). `benign_test_50mb.exe` disagrees for a separate, expected reason (stricter BLOCK threshold + a ~3e-5 boundary tie). Full corrected table and analysis in STEP 3.
- **cortex-ml has no automated feature-parity test** — nothing checks `features/pe_features.py` output against a reference (thrember) extraction on real PEs, the way cortex-endpoint's `tests/test_static_feature_parity.py` does. The signify bug above silently degraded every live static score and was only caught by a manual 5-file comparison; a parity test would have caught it immediately. This is now the top correctness gap in cortex-ml's static path.
- **cortex-ml's behavioral threshold has not been stress-tested against a genuinely held-out generalization check** the way malware-ml's Model A was (and failed).
- **Cortex-Emulation is trained but capped at report-only**: the Jan→Apr concept-drift collapse is diagnosed (including a per-family duplication/recurrence diagnostic showing the model memorizes token sequences rather than generalizing), and a genuine fix — engineered behavioural-category features, or training data across more collection dates — is scoped but not started. No ONNX export, no `pipeline.py` wiring. Whether the report-only cap can ever be lifted depends entirely on that future model, not on tuning the current one.
- **Memory and Network's authority caps are explicitly flagged for revisit**, not permanent: both need real-world validation evidence beyond their respective lab-collected datasets (real injected-process samples for Memory; real network traffic outside CSE-CIC-IDS2018 for Network) before their policy authority could reasonably be raised.
- Memory, Network, and Emulation's work was committed to git (`master`, commit `9331159`, 2026-08-27). The subsequent signify fix + these documentation updates are a later uncommitted change on top.
- **The Oliveira dataset mapping/sourcing logic that malware-ml built** (43,876-row disk-indexed loader, MD5 grouping, 307-word vocabulary) was never carried over to cortex-ml, which instead built its own Mal-API-2019/MalbehavD-V1/Carpenter pipeline from scratch — both legitimate, but the Oliveira loader represents real, reusable, already-solved engineering currently sitting unused.
- **cortex-endpoint's self-flagging false-positive issue** (its own compiled binary, and separately one Petya variant) was never resolved in that repo and hasn't been re-tested against any of cortex-ml's five models.
- cortex-ml currently has no packaged CLI or PyInstaller build — it's a Python library, not yet a deployable agent, and this gap has grown rather than shrunk as more signals were added.

### Recommendation

Continue building on **cortex-ml as the canonical base**. Immediate next steps, in rough priority order:

1. **Done for the current model:** Cortex-Emulation is trained, its threshold derived from its own val+test sweep, and its authority decided with evidence — **report-only / additive, not in `decide()`** — after an `embed_dim` ablation confirmed the drift collapse isn't a capacity artifact. If the signal is worth pursuing further: prototype the engineered behavioural-category feature representation (hypothesis: coarser activity classes are more temporally stable than raw API-token sequences), then reconsider authority. No ONNX export until the model is worth shipping.
2. **Done:** Memory/Network/Emulation committed (`9331159`).
3. **Done (with a finding):** `CortexPipeline.scan()` was run against the 5 test files — see STEP 3. It works end-to-end but exposed a silent signify-0.9.2 import failure zeroing the `authenticode` feature group (now fixed) and a residual `pe_features.py`-vs-thrember fidelity gap. **New top static-path item: build a feature-parity test** (`pe_features.py` output vs a reference extraction on real PEs, signed one included) so this class of silent extractor regression is caught automatically rather than by manual comparison.
4. Subject cortex-ml's Behavioral threshold to a val/test generalization check structured the same way malware-ml's failed one was, specifically to confirm the more conservative threshold-selection approach actually holds up under the same stress that broke the prior system.
5. Package cortex-ml behind a CLI (mirroring cortex-endpoint's `scan` / `scan-behavioral` / `scan-combined` surface) and a PyInstaller build, applying the lessons already paid for in cortex-endpoint's history (exclude torch from the start; watch for the matplotlib/PIL transitive dependency; budget for the same self-flagging false-positive risk from day one rather than discovering it late).
6. Revisit Memory's and Network's ALERT-only authority caps once real-world validation evidence — beyond either dataset's own test split — actually exists, rather than leaving the caps as a permanent default.
