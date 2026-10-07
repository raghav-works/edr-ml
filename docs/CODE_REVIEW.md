# Cortex (edr-ml) — Senior Code Review

Reviewer: Claude (AI), on request of the repository owner. Date: 2026-10-06.
Branch reviewed: `review/senior-code-review`, created from `github/main` at `f1d691a`. The `git fetch github` failed (no credentials on this machine), so this is the last-fetched `github/main`, identical to local `main`.
Scope: read-only review. No code, model, config, split or threshold was changed. Every number marked **MEASURED** below was reproduced on this machine from the local data and models. Nothing was taken from the repo's docs or earlier reports without re-running it. The analysis scripts are throwaway files in the session scratchpad; the method for each result is described in the Appendix.

Product goal used as the yardstick: lowest FP and FN on files a real Windows endpoint sees, at about 1-in-10,000 malware prevalence, including unseen families and newer samples.

---

## 1. Executive summary

1. **Partly.** For a file on disk, the only thing the deployed system can decide on its own is **Static plus the allowlist**. Behavioral, Memory and Network score vectors the caller must supply, and nothing in this repo produces them. Emulation is not wired in at all. So the "five-signal" system is, in practice, a one-signal file classifier. Without caller inputs it can return only ALLOW / ALERT / NEEDS_REVIEW, never BLOCK or TERMINATE.
2. Static is a competent EMBER2024 LightGBM (test AUC 0.9988, MEASURED), but its headline FPR is averaged over a benign set that is 75% DLLs and 56% signed. For **unsigned EXEs**, the population that matters most on an endpoint, test FPR is **7.4% at the ALERT boundary and 0.69% at BLOCK** (MEASURED), against the headline 1.2% / 0.1%.
3. At 1-in-10,000 prevalence the static ALERT stream has a **precision of 0.80%: about 123 false alerts per true detection** (MEASURED from test FPR/TPR). On 54 local unsigned benign binaries, 16 (30%) got ALERT or BLOCK.
4. **Biggest FN risk:** the static model's authenticode features record only that a signature is present. **Splicing a Microsoft certificate blob onto two real malware samples, without making the signature valid, dropped their scores from 0.95/0.98 to 0.036/0.047 (ALLOW)** (MEASURED). Signed malware is also a known blind spot: BLOCK-level TPR is 64% on signed malware and 52% on signed malicious DLLs.
5. **Biggest FP risk:** Behavioral is the only signal that can TERMINATE on its own. As shipped, **the current model code plus the deployed checkpoint gives 19 FPs out of 259 benign traces (FPR 7.3%)**, against 1 FP recorded in `EVAL_ALL_MODELS_RESULTS.txt` (MEASURED). Padding masking was added to the code on 2026-09-22 without retraining. On top of that, inference does not lowercase API names but training does, so real CamelCase traces become all-`<UNK>`.
6. The Behavioral, Memory and Network datasets each have a shortcut that explains most of their reported performance:
   - **Behavioral:** label follows dataset source (Mal-API-2019 is 100% malicious), and about 30% of val/test model inputs also appear in train.
   - **Memory:** a single-threshold stump on `svcscan.nservices` gives 99.5% accuracy.
   - **Network:** a CIC-IDS2018 testbed dataset with no link to any file.
7. The engineering hygiene is better than average: split discipline on cal, pre-registration, fail-loud config, 168 tests passing. But the safety work went into calibration and threshold bookkeeping, not into the data, the train/serve contracts or the runtime inputs, and those are what decide FP and FN.

---

## 2. Per-signal verdict table

| Signal | Helps file detection on a real endpoint? | Main evidence | Recommendation |
|---|---|---|---|
| **Static** (EMBER2024 LightGBM, 2568 features) | **Partly**: the only signal with a real runtime input | Test AUC 0.9988, reproduced exactly. Unsigned-EXE FPR 7.4% (ALERT) / 0.69% (BLOCK). Precision at 1e-4 is 0.80%. A transplanted cert blob turns malware into ALLOW. Signed-malware BLOCK TPR 64%. Val/cal are random slices of the train period, so cal FPR 0.05% became 0.10% on later test. 30% of local unsigned benign binaries got ALERT or BLOCK | **Fix**: remove or neutralise the presence-only authenticode features, set per-subgroup thresholds, run a time-split validation, collect real benign telemetry |
| **Behavioral** (1D-CNN + attention, first 100 API calls) | **No, not today** | No trace producer in this repo (caller supplies the JSON). Inference does not lowercase while training does, so CamelCase traces are 100% `<UNK>`. Masking was added after training, raising FPR from 0.4% to 7.3%. Label is confounded with source (Mal-API-2019 is all malicious; predicting the source from APIs gives AUC 0.986). 29–33% of val/test inputs equal a train input. Threshold was chosen on val+test. A bag-of-APIs logistic regression matches it (AUC 0.9988). Yet it holds sole TERMINATE authority | **Demote** (no TERMINATE) **and rebuild** on real sandbox/EDR traces from one collection pipeline |
| **Memory** (CIC-MalMem-2022 LightGBM) | **No** for file detection | Input is a whole-host Volatility summary (process counts, service counts), not a property of the scanned file. Caller supplied. All benign rows come from one VM: a one-split stump `svcscan.nservices >= 390` gets 99.5% accuracy and FPR 0.51%, better than the model's 0.96%. All test families are also in train | **Drop** from the file-scan path. Revisit only with multi-host benign data and a per-process feature design |
| **Network** (CSE-CIC-IDS2018 LightGBM) | **No** for file detection | Input is a single CICFlowMeter flow vector with no link to a file or process. Caller supplied. Testbed dataset with documented external-validity collapse. Infiltration detection 10% | **Drop** from the file-scan path (it belongs in an NDR product, if anywhere) |
| **Emulation** (Speakeasy traces CNN) | **No** | Not called by `pipeline.py` or `decide()`. No emulator runs at runtime. Threshold from a val+test sweep. Recall collapses from 70.7% to 41.5% on the Apr-2022 slice | **Drop** (or keep as a research notebook), not in the product tree |

---

## 3. Findings table

Severity reflects impact on the product goal: FP/FN on real endpoint files at 1e-4 prevalence. "Effort": S = under 1 day, M = 1–5 days, L = more than 1 week.

| ID | Area | Severity | Evidence | Impact on FP / FN | Recommended fix | Effort |
|---|---|---|---|---|---|---|
| F1 | C, G | **Critical** | `features/pe_features.py:654-695`: the authenticode group enumerates certificates via `iter_signatures()` with no verification. **MEASURED:** a Microsoft cert blob from `sigcheck64.exe` spliced into the security directory of 2 real malware PEs moved their scores 0.952→0.036 and 0.978→0.047 (ALERT→ALLOW). `verify_trusted_chain` correctly returned `chain_untrusted`, but static scoring ignored that | **FN**: a trivial, public evasion turns any unsigned malware into a static ALLOW | Feed the static model the *verified* trust result (from `authenticode_trust.py`) instead of the presence counts, or zero the 8 authenticode dims in training and serving and retrain. Add a regression test using a transplanted signature | M |
| F2 | C, D | **Critical** | Commit `76c3534` (2026-09-22) added PAD masking in `models/behavioral_cnn.py:118-136`. The checkpoint `cortex_behavioral_best.pt` dates from 2026-08-19. **MEASURED** on val+test (259 scored benign / 1,554 malicious) at 0.60: old code TP 1523 / FP 1; current code TP 1534 / **FP 19 (FPR 7.3%)**. ONNX/INT8 artifacts match the *old* code (max \|Δp\| 3.6e-7), so the ONNX and PyTorch paths now disagree on 29 of 600 short traces | **FP**: 7.3% of benign traces would get TERMINATE through `decide()` rung 1. README's own example loads the `.pt` with the current code | Retrain with masking (or revert it). Add a test that loads the shipped checkpoint, runs val, and asserts the recorded confusion matrix. Store a code-version hash in the checkpoint | S (revert) / M (retrain) |
| F3 | C | **Critical** | Training lowercases every API name (`data/download_behavioral.py:126,150,173`). Inference does not (`tokenizer/api_tokenizer.py:101`, `load_api_calls_json` at `:114`). **MEASURED:** `NtClose, NtOpenKey, …` encodes to 100% `<UNK>` (id 1). Capitalising each first letter of the val+test traces gives benign FPR **47.5%** and malicious recall **13.8%** | **FP and FN**: the model is effectively random on traces in normal Windows casing (Cuckoo, ETW and Sysmon all emit CamelCase) | Canonicalise inside `ApiTokenizer.encode`: lowercase, strip `A/W/Ex` suffixes consistently, map `Zw`→`Nt`. Alarm when the `<UNK>` rate exceeds a threshold (return PENDING) | S |
| F4 | A, F | **Critical** | `inference/policy_engine.py:662`: behavioral MALICIOUS → TERMINATE, unconditionally, even over an allowlisted static ALLOW. The threshold (`config/thresholds.yaml:92-117`) was picked by a sweep on **val+test combined** (274 benign). The long-trace (100+ call) band has only 35 benign rows across val+test (19 MalbehavD, 16 Carpenter) | **FP**: the least-validated model carries the most destructive authority. TERMINATE precision at 1e-4 prevalence cannot be estimated from 274 benign samples. The 95% upper bound on FPR with 0 FP out of 126 is already 2.9% | Cap behavioral at ALERT until it has thousands of real benign traces from the deployment's own sandbox. Require corroboration for TERMINATE. Re-derive the threshold on a cal split | S |
| F5 | B | **High** | Behavioral split groups by the **full** sequence (`scripts/split_behavioral.py:57`), but the model sees only the first 100 calls. **MEASURED:** 299/918 val and 266/917 test rows have a first-100 input identical to a train row (almost all malicious, from Mal-API-2019) | Inflates reported recall. The number that matters, recall on novel inputs, is 0.9798 at FPR 7.4% with current code | Group the split on the truncated model input (and on near-duplicates such as Jaccard over API bigrams). Report novel-only metrics | S |
| F6 | B | **High** | Behavioral label is confounded with source: Mal-API-2019 is 100% malicious (5,214 train rows), Carpenter is 100% benign. **MEASURED:** a bag-of-APIs LR predicts source = Mal-API-2019 with test AUC 0.9855. Only MalbehavD has both labels. A bag-of-APIs LR baseline gets test AUC **0.9988** (CNN 0.9967–0.9987). Raw trace *length* alone reaches AUC 0.855 | The model may be learning "which sandbox produced this trace". Real-world FP/FN is unknown | Train and evaluate on one collection pipeline with both classes. Report within-source metrics. Keep the bag-of-APIs baseline as the bar to beat | L |
| F7 | B, E | **High** | Static benign/malicious composition in test: DLL 75.0% / 6.3%, signed 55.6% / 7.2% (**MEASURED**). Subgroup FPR at ALERT boundary: unsigned EXE **7.42%** [7.12, 7.73], .NET 2.49%, >10 MB 3.46%, EXE 3.68%, against signed DLL 0.03%. BLOCK: unsigned EXE **0.69%** [0.60, 0.79]. TPR at BLOCK: signed malware **64%**, signed malicious DLL **52%**, malicious DLL 75% | **FP** concentrated on unsigned EXEs, the population users actually run. **FN** concentrated on signed or DLL malware. The headline FPR understates both | Report and threshold per subgroup (at least EXE/DLL × signed/unsigned). Reweight benign so it looks like endpoint telemetry. Add subgroup gates to the promotion criteria | M |
| F8 | E | **High** | **MEASURED** from test: at 1e-4 prevalence, ALERT (not-ALLOW) precision is **0.80%** (CI 0.78–0.83%), i.e. 123 FP per TP and about 12,100 FP per 1M benign files. BLOCK precision is 8.3%, or 1,011 FP per 1M benign files. At 1e-5, ALERT precision is 0.08% | **FP**: an endpoint scanning about 50k PE files would raise about 600 static ALERTs from benign files alone | Treat static ALERT as telemetry, not an alert. Target an operating point by FP per 1M files. Use allowlist/reputation and prevalence before ML, not after | S (policy) / M |
| F9 | B, D | **High** | Static val and cal are random rows of the *train* period (`scripts/train_static.py:81`). The parquet keeps only `sha256, label, features` (`data/download_ember2024.py:105-110`), so first-seen dates, family and file type were discarded. **MEASURED:** 15.5% of val/cal rows share a structural hash with train, against 6.1% of test. Cal-derived BLOCK FPR 0.05% became **0.101%** on the later test period; ALLOW 1.0% became 1.21% | Thresholds are systematically optimistic. FP drift over time is unmonitored | Keep EMBER2024 metadata (week, family, file_type). Carve cal from the *last* train weeks. Group by structural hash. Report a family-disjoint split | M |
| F10 | B | **High** | Memory benign data comes from one VM. **MEASURED:** 21 of 55 raw features have single-feature test AUC above 0.95. A depth-1 stump (`svcscan.nservices ≤ 389.5`) gets test accuracy 99.51%, FPR 0.51%, TPR 99.53%, which beats the deployed model's FPR 0.96%. All 16 test families also appear in train | Reported memory performance says nothing about production | Drop from the file pipeline (F11). If kept for research, require multi-host benign data and a leave-family-out evaluation | — |
| F11 | A | **High** | `inference/pipeline.py:159-164, 354-385`: memory and network features are caller-supplied host or flow vectors. Nothing in the repo produces them. They are attached to a *file's* ScanResult and can raise it to ALERT or unlock BLOCK (`policy_engine.py:673`) even though they describe a different object | **FP**: one infected-host snapshot or one noisy flow raises every file scanned in that call. **FN**: no help for file detection | Remove memory and network from `CortexPipeline.scan()` and `decide()`. If they are wanted, give them their own host-level and network-level decisions | S |
| F12 | F | **High** | NaN handling: `static_verdict_from_score(nan)` returns **BLOCK**, and `behavioral_verdict_from_score(nan)` and memory both return **BENIGN** (`policy_engine.py:291-303`, **MEASURED**). No test covers a NaN score | Fails *open* for behavioral and memory, and fails to the most severe verdict for static. A caller-supplied NaN feature vector, or a broken model, gives a silent wrong verdict | Check `math.isfinite(score)` in every `*_verdict_from_score` and route to ERROR. Add tests | S |
| F13 | F | **Medium** | Behavioral PENDING (fewer than 10 calls), NOT_PROVIDED and an ERROR in an unconfigured model are all treated as "no evidence", so a static ALLOW becomes final ALLOW (`policy_engine.py:698-703`) | **FN**: sandbox-aware malware that exits early, or makes few hooked calls, gets an unqualified ALLOW | Return NEEDS_REVIEW or a distinct `ALLOW_UNVERIFIED` when behavioral was requested but PENDING | S |
| F14 | F, G | **Medium** | Authenticode chain-verified files skip static entirely (`pipeline.py:225-228`). 246 of 304 local PEs (81%) took this path (**MEASURED**). There is no revocation (CRL/OCSP) check, no publisher reputation and no timestamp policy, so stolen or abused certificates are fully trusted | **FN**: signed malware (stolen certs, signed LOLBins, signed loaders) is never scored, and static is already weak there (F7) | Still score allowlisted files and log the score. Require revocation plus a publisher allowlist, not "any trusted root". Never let the allowlist suppress a high-confidence static score without review | M |
| F15 | G | **Medium** | Files over 100 MiB return `file_too_large` → NEEDS_REVIEW (`pipeline.py:74,139`). Padding malware with zeros or random overlay is free. Separately, **MEASURED:** the strings group regex loop (`pe_features.py:298-316`, 77 regexes per string) took **21–23 s** on a 30 MB overlay. 50–66 MB real installers took 8–16 s per scan | **FN** (pad to evade, or to hit the timeout). DoS and latency on endpoints | Score a bounded prefix (header, first and last N MB, sections) for oversized files rather than skipping them. Put a budget on string extraction. Add a per-scan timeout that fails closed | M |
| F16 | G | **Medium** | Overlay and string padding move the score unpredictably. **MEASURED:** malware sample 1 went 0.952 → 0.998 with a 1 MB zero overlay, and sample 2 went 0.978 → 0.924 with a 5 MB random overlay (the direction depends on the overlay content). Benign `sigcheck64` + 30 MB printable overlay went ALLOW → ALERT. Byte histogram and entropy groups are computed over the whole file including overlay | FP and FN instability. Installers, self-extractors and signed setups with large overlays look suspicious | Compute histogram/entropy/strings over mapped sections plus a capped overlay. Add overlay features rather than letting overlay content dominate. Add an adversarial-padding test set | M |
| F17 | G | **Medium** | `is_valid_pe` (`pe_features.py:913`) accepts a 1 KB truncated header. **MEASURED:** the scan ran with no degraded groups and scored ALLOW (0.07 s). Archives are not unpacked (a `.zip` gives NEEDS_REVIEW) | **FN**: truncated or partially downloaded droppers get a clean ALLOW | Treat "sections extend beyond file size" and other truncation as a `pefile` warning that becomes NEEDS_REVIEW | S |
| F18 | C | **Medium** | No live-vs-reference (thrember) feature parity has ever been measured. `tests/test_static_feature_parity.py` checks only internal consistency on 4 mingw fixtures. `README.md` §12 admits this. `email_addr` regex is a copy of the MAC regex (`pe_features.py:221`). This is probably faithful to thrember, but it shows the vocabulary was copied rather than checked | Unknown skew, in an unknown direction, on every live scan | Build a separate venv with thrember plus signify<0.9, and compare per group on 1,000+ real PEs (benign and malware). Fail CI on any per-group difference | M |
| F19 | C | **Medium** | Static ONNX full-test parity, **MEASURED** on 539,940 rows: max \|Δp\| **0.0305**, 4 verdict flips at ALLOW and 4 at BLOCK. The repo's test samples 4,000 rows and allows max error 0.05 (`tests/test_onnx_parity.py:52-53`). The cause is margin reconstruction via float32 `logit(clip(p, 1e-7))` (`export/export_onnx.py:98`) | A small number of FP/FN flips between the Python and ONNX deployments | Export the raw margin directly (`raw_score` output) instead of logit(sigmoid). Test parity on the full test set near thresholds | S |
| F20 | C | **Low** | Behavioral INT8 vs FP32 ONNX, **MEASURED** on val+test: max \|Δp\| 0.0066, **0 verdict flips**. But `export/quantize.py:53-58` validates on uniform random token IDs up to `vocab_size=4096` when the real vocab has 366 tokens | No measured harm today. The repo's own INT8 check is meaningless | Validate quantization on real held-out traces | S |
| F21 | F | **Low** | The pipeline never calls `.eval()` on the behavioral model (`pipeline.py:416-418`). **MEASURED:** a model left in train mode flips 259/1,813 verdicts. The README example does call `.eval()` | FP/FN if an integrator forgets | Call `model.eval()` in `CortexPipeline.__init__`, or wrap the model in a loader | S |
| F22 | D | **Medium** | Static `num_iterations` = **3000** = `DEFAULT_N_ESTIMATORS` (`models/static_lgbm.py:59`). Early stopping on AUC never fired. Early stopping uses whole-curve AUC, not the low-FPR region that is deployed. `is_unbalance=True` on 50/50 data does nothing | The model may be under-fit at the operating point. Tuning is blind to FPR at 1e-3 | Monitor partial AUC / TPR at FPR ≤ 1e-3 on a time-later val. Raise the cap and record best_iteration | S |
| F23 | D | **Medium** | Behavioral and Emulation thresholds were chosen on **val+test** (`config/thresholds.yaml:92-97, 185-196`). Behavioral has no calibrator and no cal split. Behavioral `scripts/train_behavioral.py:222` defaults `--embed-dim 128` but `TrainConfig` defaults 64, so the two training entrypoints build different models | Reported behavioral and emulation test metrics are not held out | Use the same 4-way discipline as static. One config source for model hyperparameters | S |
| F24 | H, G | **Medium** | `.meta` files are `pickle.load`ed (`models/static_lgbm.py:129`, `memory_lgbm.py:137`, `network_lgbm.py:135`, `features/memory_features.py:215`). A tampered model directory gives code execution in the scanner, which may run privileged | Security of the product itself | Store calibrator coefficients as JSON. Verify model hashes (already listed in `thresholds.yaml`) at load | S |
| F25 | I | **Medium** | **MEASURED** latency per file (static path, this 20-core Linux box): p50 0.08 s for files under 1 MB, 0.99 s for 1–10 MB, 7.7 s for 10–50 MB, about 15 s for 50–66 MB. Prediction itself is about 3 ms; feature extraction dominates. The file is parsed by `pefile` twice (`is_valid_pe` then `raw_features`) and by `signify` twice (authenticode feature plus trust check). Peak RSS 1.08 GB over 304 files; model load 340 MB | Too slow for on-access scanning of large binaries. Memory-heavy for an endpoint agent | Parse once and share the `pe` object. Cap byteentropy/strings work. Run ONNX in a separate low-priority process with a timeout | M |
| F26 | H | **Medium** | `PlattCalibrator`, `MetricsReport`, `find_threshold_for_fpr` and `_model_hash` are copy-pasted in 3 model files. `thresholds.yaml` holds about 170 lines of narrative comments. `inference/policy_engine.py` is 703 lines, of which `decide()` is about 40 lines of logic and the rest docstring. `file.*`/`sequence.*` in YAML are "informational" while the code holds its own constants (`pipeline.py:74`, `api_tokenizer.py:24`) | Drift risk: a threshold or size limit changed in one place but not the other | One `lgbm_common.py`. Read file and sequence limits from YAML. Move rationale into docs and keep the code short | M |
| F27 | H | **Medium** | Tests: 168 passed, 0 failed, 0 errors (**MEASURED**). Critical paths without tests: NaN scores; real checkpoint + vocab + pipeline end-to-end (would have caught F2 and F3); signature transplant (F1); subgroup or time-split regression gates; ONNX parity on the full set; behavioral ONNX/PyTorch parity | Regressions like F2 shipped silently | Add the tests listed in §4 | M |
| F28 | H | **Low** | Docs disagree with code or data. README §6 lists behavioral data as Mal-API-2019 + MalbehavD-V1 (Carpenter is missing). README §12 says "trained on 274 benign samples" (train has 1,103; 274 is val+test). README §5 says behavioral is INT8-quantized while the INT8 ONNX predates the current code. `requirements.txt` lists `datasets` although the downloader deliberately avoids it, and uses `>=` floors for torch/lightgbm/onnxruntime (not reproducible) | Misleading to reviewers and integrators | Fix the docs. Pin exact versions (lock file) | S |
| F29 | E | **Medium** | No unseen-family or time-split result exists for Static, Behavioral or Memory. EMBER2024's test is a later period, but family/week metadata was dropped (F9), so no family-disjoint number can be computed. Emulation's time-split result (recall 70.7% → 41.5%) shows what likely happens to the others | **FN** on new families and newer samples is unmeasured for every signal that matters | Keep metadata. Report leave-family-out and per-week test metrics | M |
| F30 | B | **Low** | Static labels are EMBER2024's VirusTotal-derived labels. Benign means "no detections", which leans towards widely distributed software and away from in-house or rare benign binaries. **MEASURED:** 30% of local unsigned benign binaries (dev builds, setuptools launchers, test executables) got ALERT/BLOCK | **FP** on rare or in-house software | Collect an endpoint benign corpus (the organisation's own software, dev tools, installers) as a standing FP benchmark | M |

### Headline numbers that are misleading

| Headline (where) | Why misleading |
|---|---|
| Static test FPR 1.21% / 0.10%, AUC 0.9988 (`thresholds.yaml`, `reports/…/eval_static_retrain_test.txt`) | Averaged over a benign set that is 75% DLL and 56% signed. Unsigned EXE FPR is 7.4% / 0.69%. Says nothing about the transplanted-signature evasion or the 64% signed-malware BLOCK TPR |
| Static "derived at 0.05% FPR on cal" (README §9) | Cal comes from the train period and has 2.5× more near-duplicates of train than test. Test FPR doubled to 0.10% |
| Behavioral test FP = 0, recall 0.986, AUC 0.9987 (`EVAL_ALL_MODELS_RESULTS.txt` §4) | Threshold chosen on val+test. About 30% of test inputs are in train. Label is confounded with source. Measured with the old model code (current code gives FPR 7.3%). Real-casing traces are all `<UNK>` |
| Behavioral "0 FP on 118 benign short rows" (`pipeline.py:392-394`, `api_tokenizer.py:12-13`) | Same issues; with current code there are 19 FPs in val+test |
| Memory test AUC 0.99999, recall 0.9997 | A single-feature stump does as well. Single benign VM. No unseen family in test |
| Network test recall 0.975 at FPR 0.09% | Testbed flows. Per-class (as recorded in `EVAL_ALL_MODELS_RESULTS.txt`, not re-run here) recall of 100% on classes with 2–3 training rows (FTP-BruteForce n_train=2, SlowHTTPTest n_train=3) indicates trivially separable or fingerprinted traffic, not generalisation |
| "Five-signal detection" (README title) | For a file on disk, one signal runs. Three need caller-supplied vectors that no component produces, and the fifth is not wired in |
| Static ONNX "0 verdict flips" (`tests/test_onnx_parity.py` docstring) | Measured on 4,000 rows. On the full 540k test set there are 4 + 4 flips |

---

## 4. Top 10 improvements (ordered by expected FP/FN reduction per unit effort)

1. **Take TERMINATE away from Behavioral now** (F4). Cap it at ALERT, or require static ≥ ALERT corroboration. One line in `decide()`. It removes the largest FP blast radius. (S)
2. **Fix the behavioral train/serve contract** (F2, F3, F21): revert or retrain the masking change, canonicalise API names in `encode()`, call `.eval()`, and add an end-to-end test on the shipped checkpoint asserting the recorded confusion matrix. (S)
3. **Neutralise the presence-only authenticode features** (F1): replace them with the verified trust result, or drop them and retrain. Add a transplanted-signature regression test. Closes a zero-cost evasion. (M)
4. **Stop skipping static for chain-verified files** (F14): always score, add revocation checks and a publisher allowlist, and route "trusted signer + high static score" to review. (M)
5. **Fail closed on NaN and give PENDING an explicit state** (F12, F13). (S)
6. **Remove memory and network from the file-scan decision** (F11). They cannot help file detection and can only add FPs. (S)
7. **Subgroup-aware thresholds and reporting for static** (F7, F8): EXE/DLL × signed/unsigned, with an operating point stated as FP per 1M files at the prevalence you actually see. Make static ALERT a telemetry tier. (M)
8. **Time-ordered cal and family-aware splits** (F9, F29): regenerate the EMBER2024 parquet keeping week, family and file type. Carve cal from the latest train weeks. Report leave-family-out results. (M)
9. **Bound the work on hostile or large files** (F15, F16, F17, F25): single parse, capped overlay/strings, a per-scan timeout that fails closed, a prefix-scan for files over 100 MB, and truncation detection. (M)
10. **Build an endpoint benign benchmark and measure thrember parity** (F18, F30): a few thousand of the organisation's own unsigned binaries, installers and dev tools, plus a separate venv parity run. This becomes the gate for any threshold or model promotion. (M–L)

---

## 5. Quick wins (under 1 day each) vs longer-term work

**Quick wins**
- Cap behavioral at ALERT, or require corroboration for TERMINATE (F4).
- Revert commit `76c3534`'s masking, *or* mark the `.pt` checkpoint as incompatible until retrained. Add a checkpoint/code version check (F2).
- Lowercase and canonicalise API names in `ApiTokenizer.encode`. Return PENDING when more than X% of tokens are `<UNK>` (F3).
- `isfinite` guards in all `*_verdict_from_score` functions (F12).
- Explicit NEEDS_REVIEW (or `ALLOW_UNVERIFIED`) when behavioral was requested but PENDING (F13).
- `model.eval()` inside the pipeline (F21).
- Detach memory and network from `scan()`/`decide()` (F11).
- Truncation → NEEDS_REVIEW (F17).
- Export the raw margin in static ONNX. Run parity on the full test set (F19).
- Replace pickle `.meta` files with JSON. Verify model sha256 at load (F24).
- Fix the README/doc inaccuracies. Pin dependency versions (F28).
- Report static subgroup metrics and FP per 1M files in `evaluate_all_models.py` (F7, F8, measurement only).

**Longer-term**
- Retrain static without presence-only signature features. Rethink allowlist trust (revocation, publisher reputation) (F1, F14).
- Regenerate EMBER2024 with metadata. Time-ordered cal. Family-disjoint evaluation (F9, F29).
- thrember parity harness in a separate venv. Per-group CI gate (F18).
- Endpoint benign corpus and an adversarial set (signature transplant, overlay padding, string stuffing, packers) as standing promotion gates (F16, F30).
- Rebuild behavioral on traces from the product's own sandbox or EDR sensor, with both classes from the same pipeline, and beat the bag-of-APIs baseline (F5, F6).
- Performance: single-parse extraction, capped feature work, out-of-process ONNX scoring with timeouts (F15, F25).
- Consolidate the duplicated LGBM code. Move rationale prose out of code and YAML into docs (F26).

---

## 6. Open questions for the manager / agent team

1. **What is the deployment surface?** On-access scanning (latency budget in ms), on-demand scans, or back-end triage? This decides whether 8–16 s on 50 MB installers is acceptable and whether static ALERT should notify a person at all.
2. **What produces API traces in production?** If there is no sandbox or EDR sensor feeding traces, Behavioral should be removed from the product rather than fixed. If there is one, which tool, what casing, which hook set? Its output format defines the tokenizer contract.
3. **Who owns Memory and Network?** Is a host-memory or network-flow product actually planned? If not, should they leave this repo?
4. **What does TERMINATE mean operationally?** Kill a process, quarantine a file, or both? What is the acceptable false-TERMINATE rate (per endpoint per month)?
5. **What prevalence and alert budget should thresholds target?** For example, "≤ 1 false ALERT per 1,000 endpoints per day". Today's thresholds target a FPR on a balanced research set.
6. **Can we get real endpoint benign telemetry** (hashes plus feature vectors of files actually present on company machines)? Without it no FP number in this repo is predictive.
7. **Allowlist policy:** is "any certificate chaining to a Microsoft-trusted root" acceptable as a full static bypass? Who maintains a publisher allowlist and revocation checks? Is the NSRL artifact actually deployed? (It wasn't configured in any run here.)
8. **Is EMBER2024 licensed and appropriate as the only static training source,** or can we add vendor or in-house labelled data (especially signed malware, unsigned benign EXEs and .NET)?
9. **Which branch is authoritative for deployment** (`gitlab` office squash vs `main`)? This review covers `github/main` `f1d691a`. The office branch shares no history with it, so the same findings must be checked there before anything ships.

---

## 7. Commit hash and push result

Filled in by the reviewer's hand-off message rather than in this file. A commit cannot contain its own hash. The push is to be run by the repository owner: `git push github review/senior-code-review`.

---

## Appendix — how each measurement was made

All runs used the repo's `.venv` (Python 3.10, lightgbm 4.7.0, torch 2.14.0, onnxruntime 1.23.2, pefile 2024.8.26, signify 0.9.2) on a 20-core / 38 GB Linux host. Nothing was written under the repo except this file.

- **Static rows, scores and duplicates:** streamed `ember2024_{train,test}.parquet` (2,340,000 + 539,940 rows). Rebuilt the exact train/val/cal carve with `scripts.train_static._permutation_bounds(seed=42, 0.1, 0.1)`. Scored val/cal/test with `LGBMModel.load('data/models/cortex_static')` (sha256 `acfa5a75…`, matching `thresholds.yaml`). Test metrics reproduce the recorded ones exactly (FPR 0.01213 / 0.00101, TPR 0.9821 / 0.9149). "Structural hash" = BLAKE2 of header (timestamp and checksum zeroed) + sections (rounded) + imports + exports + data directories. Subgroups use feature proxies: DLL = `IMAGE_FILE_DLL` header bit; signed = authenticode `num_certs > 0`; .NET = COM_DESCRIPTOR size > 0; high entropy = max section entropy > 7.2. CIs are Clopper–Pearson 95%.
- **Precision at prevalence:** PPV = TPR·π / (TPR·π + FPR·(1−π)) using test TPR/FPR, with the FPR CI propagated.
- **Static ONNX parity:** `cortex_static.onnx` vs LightGBM + Platt on all 539,940 test rows.
- **Real PE files:** 304 unique (by sha256) `.exe/.dll` files found on this host, scanned with `CortexPipeline(static_model=…)`. Files were never executed. Labels are assumed: the Sysinternals, Qt, OpenSSL, setuptools launchers and in-house build outputs are treated as benign; 2 files in `cortex-endpoint/test_artifacts/quarantine_malware/` are treated as malicious; `svchost.exe` and `notepadd.exe` in `Test_Files` are excluded as unknown. Results: 246 chain-verified (static skipped); of 54 remaining assumed-benign files, 16 got ALERT/BLOCK; both malware files got BLOCK (capped to ALERT). Small, convenience sample. It indicates direction, not a rate.
- **Evasion / robustness:** mutations applied in memory only (never written to disk) to the 2 malware PEs and to `sigcheck64.exe`. For the certificate transplant, the WIN_CERTIFICATE blob of `sigcheck64.exe` was appended (8-byte aligned) and `DATA_DIRECTORY[SECURITY]` was pointed at it.
- **Behavioral:** `cortex_behavioral_best.pt` + `api_vocab.json` (366 tokens, embed_dim 128) on `behavioral_{val,test}.parquet`. Ran under current `models/behavioral_cnn.py` and under `git show 76c3534^:models/behavioral_cnn.py`. Compared with `cortex_behavioral.onnx` and `_int8.onnx`. Baselines: sklearn `CountVectorizer(binary)` + `LogisticRegression` on the first 100 calls, trained on train, scored on test.
- **Memory:** single-feature AUCs and a depth-1 `DecisionTreeClassifier` trained on `memory_train`, scored on `memory_test`. Family = second `-` field of `Category`.
- **Tests:** `pytest -q` → 168 passed, 0 failed, 0 errors (34.7 s).

---

## Phase 1 status (2026-10-06)

Phase 1 is the safety hotfixes on branch `fix/p1-safety-hotfixes`, based on
`f1d691a`.
- No model was retrained.
- No threshold value changed. `config/thresholds.yaml` gained only policy
  switches and model hash pins.
- Final checks, all on the final code (`a4dcfa9`):
  - `pytest`: 403 passed, 0 failed, 0 errors.
  - `pytest -m slow`: 14 passed, 0 failed, 0 errors.
  - `scripts.evaluate_all_models`: exit 0, and **all five models match their
    records.** Static matches `reports/static_retrain_20260921/`; memory,
    network, behavioral and emulation (model-raw lines) match
    `EVAL_ALL_MODELS_RESULTS.txt`. The only extra lines are prevalence
    projections and the emulation PENDING lines. Output:
    `reports/phase1_final_eval_2026-10-06.txt`.
  - `scripts.verify_onnx_parity`: 0 verdict flips on static, memory and network.

| Finding | Status | Commit | Note |
|---|---|---|---|
| F1 | open | — | Phase 2 (needs a retrain or feature change) |
| F2 | fixed | `15a67f3` | Not retrained. A JSON sidecar records the forward pass the checkpoint was trained with (`use_padding_mask=false` for the shipped one). `load_behavioral_model()` is the only loader and checks the sha256 of checkpoint and vocab. Val reproduces TN 132 / FP 1 / FN 20 / TP 758. A masked retrain is still to come. The same fix was applied to emulation (`a4dcfa9`, follow-up 1) |
| F3 | fixed | `45039cb` | Lowercase only, exactly as training. A scored window with more than 10% `<UNK>` gives PENDING (`behavioral_unk_rate_high`). Input-format diagnostics are on ScanResult |
| F4 | partly fixed | `8987c4f` | TERMINATE needs static ALERT/BLOCK. Without it, ALERT `behavioral_malicious_uncorroborated`. The threshold is still from val+test (F23) |
| F5–F10 | open | — | |
| F11 | partly fixed (flag only) | `7b87e3e` | `file_scan.attach_memory_network`, default `true`, so behaviour is unchanged. With `false`, final BLOCK becomes unreachable (static BLOCK is corroborated only by memory/network). Waiting on the BLOCK-path decision |
| F12 | fixed | `d0153b7`, `45039cb` | A non-finite score or logit gives ERROR for every signal, emulation included |
| F13 | fixed | `d0153b7` | Behavioral PENDING with static ALLOW gives `ALLOW_UNVERIFIED` (`behavioral.pending_with_static_allow`) |
| F14–F16 | open | — | |
| F17 | fixed | `03831e5`, `ef91e08` | Truncation rules R1 (section raw data past EOF, zero tolerance), R2 (headers) and R3 (certificate table) give NEEDS_REVIEW. The PE is parsed once per scan |
| F18–F20 | open | — | |
| F21 | fixed | `45039cb` | The pipeline forces `eval()` and scores under `torch.inference_mode()` |
| F22, F23 | open | — | |
| F24 | fixed | `b6adfe8` | JSON `.meta.json` replaces pickle. Every load checks the `.lgbm` sha256 against the JSON. Deployed loads also check the `config/thresholds.yaml` `*.model_sha256` pin. Every `torch.load` uses `weights_only=True`. Scores are bit-identical before and after |
| F25 | open | — | `pefile` is now parsed once per scan (F17), but `signify` still runs twice and latency was not re-measured |
| F26–F30 | open | — | F27: Phase 1 added tests for every fixed finding (168 → 417 tests), but the gaps it lists for F1, subgroups and time splits remain |

Also on this branch: `fee8b13` makes slow tests opt-in. `pytest -m slow` must
pass before every commit and in every release check (README §10).

### Follow-ups found during Phase 1
1. **Emulation checkpoint vs code — fixed in `a4dcfa9`.** This was the same
   defect as F2.
   - **Cause.** Commit `b876d26` (2026-09-22) added padding masking to
     `models/emulation_cnn.py` after the shipped `cortex_emulation_best.pt`
     (2026-08-27) was trained. The current code gave val 442/8/71/154, and the
     one empty test trace scored NaN, which crashed `scripts.evaluate_all_models`.
   - **Fix.** The F2 pattern: a `use_padding_mask` switch, bit-identical to
     `b876d26^` when `False`; a JSON sidecar (back-filled `false`); and one loader,
     `load_emulation_model()`. The sidecar code is shared with behavioral in
     `models/sequence_artifacts.py`, and the behavioral format is unchanged.
   - **Result.** The recorded val (440 / 10 / 66 / 159, AUC 0.949472) and test
     (1923 / 13 / 327 / 232, AUC 0.873724) numbers reproduce exactly on the
     model-raw lines.
   - **Empty-trace guard, both sequence models.** An all-padding (empty-trace)
     row is never scored: it counts as PENDING (test has 1 benign row). The
     scoring helpers refuse such rows, and training drops them. Behavioral has
     no empty traces in any split, so its numbers are unchanged.
2. **F23 is unchanged.** The behavioral and emulation thresholds were still
   chosen on val+test, and behavioral has no calibrator or cal split.
3. **Re-check the F17 rules on the Phase 2 benign benchmark.** R1–R3 hit 0 of
   126 local benign PEs. That is a small sample, and R1 has zero tolerance.
4. **Model files must ship with their metadata.** These are gitignored, like
   the models; without them the loaders refuse to load.
   - each `*.lgbm` needs its `*.meta.json`;
   - `cortex_behavioral_best.pt` needs `cortex_behavioral_best.meta.json`;
   - `cortex_emulation_best.pt` needs `cortex_emulation_best.meta.json`.

   The old pickle `.meta` files were moved out of the repo (kept, not deleted).
5. **Emulation scores very short traces; behavioral does not.** Emulation
   scores traces of 1–9 API calls: 168 of 675 in val and 249 of 2,495 in test.
   Behavioral sends traces under 10 calls to PENDING. Emulation is report-only,
   so nothing depends on this today. If emulation is ever wired into decisions,
   it needs a rule for short traces based on measured evidence, not a copy of
   behavioral's cutoff.

### Decisions waiting on the manager
1. **The BLOCK path.** Today a final BLOCK needs static BLOCK plus a memory or
   network MALICIOUS on the same scan. F11 says those vectors do not describe
   the file. `file_scan.attach_memory_network` cannot be set to `false` until we
   decide what a BLOCK should require: none, static alone at a stricter
   threshold, or static plus behavioral.
2. **The new `ALLOW_UNVERIFIED` decision state (F13).** Consumers of the
   security event must handle it. We need to decide whether it should notify
   anyone, or whether `NEEDS_REVIEW` should be used instead (one config line).
3. **Alert budget and deployment surface** (§6 questions 1, 4 and 5). Every
   remaining threshold and FP decision depends on these: on-access or on-demand
   scanning, what TERMINATE means operationally, and an acceptable false-alert
   rate per endpoint.
