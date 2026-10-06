# Cortex (edr-ml) — Five-Signal Windows Malware Detection

## 1. Introduction

**edr-ml** is a machine-learning pipeline that decides whether a Windows file or activity is malicious. It scores up to five independent evidence sources (called *signals*) and combines their **verdicts** using fixed, auditable priority rules. It does not average scores.

| Signal | What it looks at | Role in the decision |
|---|---|---|
| **Static** | The PE file on disk (headers, sections, imports, strings, signature) | ALLOW / ALERT / BLOCK |
| **Behavioral** | The API-call trace recorded while the file runs | Can TERMINATE when Static is ALERT or BLOCK; otherwise ALERT |
| **Memory** | Memory-forensics features (injected / fileless activity) | Capped at ALERT |
| **Network** | Network-flow statistics (C2, DoS, brute force) | Capped at ALERT |
| **Emulation** | API sequence from the Speakeasy emulator | Logged only, not used for decisions |

Every model is trained offline on its own dataset, has its decision threshold derived from held-out calibration data, and (except Emulation) is exported to ONNX for deployment.

---

## 2. Architecture

```
                        ┌───────────────────────────────┐
                        │  Caller: file path + optional │
                        │  API trace / memory / network │
                        └──┬─────────┬────────┬─────────┘
                           │         │        │
      ┌────────────────────┘         │        └────────────────────┐
      ▼                              ▼                             ▼
┌──────────────┐           ┌──────────────────┐          ┌──────────────────┐
│ Path + PE    │           │  Cortex-Memory   │          │  Cortex-Network  │
│ validation   │           │   (LightGBM)     │          │   (LightGBM)     │
└──────┬───────┘           │ 62 mem features  │          │ 78 flow features │
       ▼                   └────────┬─────────┘          └────────┬─────────┘
┌──────────────┐                    │                             │
│ Known-file   │── NSRL / trusted ──┼──► Static ALLOW             │
│ allowlist    │   signature match  │                             │
└──────┬───────┘                    │                             │
       ▼                            │                             │
┌──────────────┐                    │                             │
│ Cortex-Static│                    │                             │
│ (LightGBM)   │                    │                             │
│ 2,568 PE     │                    │                             │
│ features     │                    │                             │
└──────┬───────┘                    │                             │
       ▼                            │                             │
┌──────────────┐                    │                             │
│ Cortex-      │                    │                             │
│ Behavioral   │                    │                             │
│ (1D-CNN +    │                    │                             │
│  Attention)  │                    │                             │
└──────┬───────┘                    │                             │
       ▼                            ▼                             ▼
     ┌──────────────────────────────────────────────────────────────┐
     │                       Policy Engine                          │
     │     Priority rules over verdicts (no score averaging)        │
     └──────────────────────────────┬───────────────────────────────┘
                                    ▼
                  ┌───────────────────────────────────┐
                  │           Final Decision          │
                  │ ALLOW / ALLOW_UNVERIFIED /        │
                  │ NEEDS_REVIEW / ALERT / BLOCK /    │
                  │ TERMINATE + JSON event            │
                  └───────────────────────────────────┘

   Cortex-Emulation (1D-CNN + Attention) runs separately and is logged only.
```

Memory and Network do not need a valid file, so fileless or network-only activity can still be reported. Behavioral runs after any Static verdict (ALLOW, ALERT or BLOCK). Only a Static error skips it.

A full Mermaid flowchart is in [ARCHITECTURE.md](ARCHITECTURE.md).

---

## 3. Project Structure

```
edr-ml/
│
├── config/
│   └── thresholds.yaml                 # Single source of truth for every decision threshold
│
├── data/                               # Dataset download, cleaning and loading
│   ├── download_ember2024.py           # EMBER2024 PE features (HuggingFace) + de-duplication
│   ├── download_behavioral.py          # API-trace corpus (Mal-API-2019, MalbehavD-V1, ...)
│   ├── download_memory.py              # CIC-MalMem-2022 loader + schema validation
│   ├── download_network.py             # CSE-CIC-IDS2018 loader, cleaning, stratified sampling
│   ├── download_emulation.py           # Quo Vadis Speakeasy emulation traces
│   ├── download_nsrl.py                # NSRL known-good hash set (allowlist)
│   ├── raw/                            # Raw downloads (gitignored)
│   ├── processed/                      # Train / val / cal / test parquet splits (gitignored)
│   └── models/                         # Trained models, vocabularies, ONNX files (gitignored)
│
├── features/                           # Feature extraction
│   ├── pe_features.py                  # 2,568-dim EMBER2024-compatible PE feature extractor
│   ├── ember2024_adapter.py            # Maps EMBER2024 records onto the same feature vector
│   ├── memory_features.py              # 7 derived memory-forensics features + scaler
│   ├── authenticode_trust.py           # Code-signing certificate chain verification
│   └── nsrl_allowlist.py               # Known-file hash allowlist lookup
│
├── tokenizer/                          # API-name → token-ID conversion
│   ├── api_tokenizer.py                # Behavioral: fixed length 100, <PAD>/<UNK>
│   └── emulation_tokenizer.py          # Emulation: fixed length 500
│
├── models/                             # Model definitions and training loops
│   ├── static_lgbm.py                  # Static LightGBM + Platt calibration
│   ├── memory_lgbm.py                  # Memory LightGBM + Platt calibration
│   ├── network_lgbm.py                 # Network LightGBM + Platt calibration
│   ├── behavioral_cnn.py               # 1D-CNN + multi-head self-attention
│   ├── train_behavioral.py             # Behavioral training loop (AdamW, early stopping)
│   ├── emulation_cnn.py                # Emulation 1D-CNN + attention
│   └── train_emulation.py              # Emulation training loop + threshold sweep
│
├── scripts/                            # Command-line entry points
│   ├── split_behavioral.py             # Leakage-checked dataset splits, one per signal
│   ├── split_memory.py
│   ├── split_network.py
│   ├── split_emulation.py
│   ├── train_static.py                 # Train each model
│   ├── train_behavioral.py
│   ├── train_memory.py
│   ├── train_network.py
│   ├── train_emulation.py
│   ├── derive_static_thresholds.py     # Derive Static thresholds on the cal split
│   ├── evaluate_all_models.py          # Val + test evaluation of every model
│   ├── evaluate_behavioral.py          # Behavioral-only evaluation
│   └── verify_onnx_parity.py           # Check ONNX output matches the Python models
│
├── export/                             # Deployment export
│   ├── export_onnx.py                  # LightGBM → ONNX, PyTorch → ONNX
│   └── quantize.py                     # INT8 quantization + accuracy/latency comparison
│
├── inference/                          # Runtime scanning
│   ├── pipeline.py                     # CortexPipeline.scan(): end-to-end scan
│   └── policy_engine.py                # Verdict thresholds + priority decision rules
│
├── tests/                              # pytest suite + signed/unsigned PE fixtures
├── reports/                            # Retrain delivery reports and checksums
├── docs/                               # Technical notes, flow walkthrough, PDF reports
│
├── ARCHITECTURE.md                     # Detailed architecture and decision flow
├── OPEN_ITEMS.md                       # Open work and plan of record
├── PROJECT_HISTORY_REPORT.md           # Development history
├── EVAL_ALL_MODELS_RESULTS.txt         # Latest evaluation output
├── requirements.txt
└── README.md
```

---

## 4. Project Overview

The pipeline has two stages.

**Model Training (offline)**
1. Download datasets and remove duplicates
2. Split into train / val / cal / test with leakage checks
3. Extract features (PE features, API tokens, memory / network features)
4. Train each model (early stopping on `val`)
5. Fit calibration on the separate `cal` split
6. Derive decision thresholds on `cal` at a target false-positive rate
7. Evaluate once on the untouched `test` split
8. Export to ONNX and verify the ONNX output matches the Python model

**Live Detection (runtime)**
1. Validate the file path (exists, readable, ≤ 100 MiB) and the PE format, and reject truncated PEs (parsed once; the same parse feeds feature extraction)
2. Check the known-file allowlist (NSRL hash or trusted Authenticode signature)
3. Extract PE features → Static verdict
4. Tokenize the API trace (API names lowercased, exactly as in training) → Behavioral verdict. A trace whose scored window is more than 10% unknown names (`behavioral.max_unk_rate`) is not scored: PENDING with reason `behavioral_unk_rate_high`. Input-format problems (empty, non-ASCII or whitespace-containing names) are counted in `behavioral_input_diagnostics` on the scan result
5. Score memory / network vectors if supplied → Memory / Network verdicts
6. Apply the policy engine rules → final decision
7. Emit a structured JSON security event (UTC timestamp + correlation UUID)

---

## 5. Models

| Model | Input | Architecture | Purpose |
|---|---|---|---|
| Cortex-Static | PE header, sections, imports, strings, signature (2,568 features) | LightGBM + Platt calibration | Fast static classification |
| Cortex-Behavioral | First 100 API calls | 1D-CNN (4 conv blocks) + multi-head attention | Runtime behavior analysis |
| Cortex-Memory | 55 VolMemLyzer features + 7 derived (62 total) | LightGBM + Platt calibration | Injected / fileless malware |
| Cortex-Network | 78 CICFlowMeter flow features | LightGBM + Platt calibration | C2, DoS, brute force |
| Cortex-Emulation | First 500 emulated API calls | 1D-CNN + attention (embed 64) | Packed / obfuscated payloads (logged only) |

Static, Memory and Network export to ONNX with the calibrator built into the graph. They are **not** INT8-quantized, because quantization has no benefit for tree ensembles. Behavioral exports to ONNX and is also INT8-quantized.

---

## 6. Dataset Strategy

Each model is trained on its own specialized dataset instead of one shared corpus.

| Model | Dataset | Notes |
|---|---|---|
| Cortex-Static | EMBER2024 (Apache-2.0) | De-duplicated on `sha256` right after download (raw download is ~2× duplicated) |
| Cortex-Behavioral | Mal-API-2019 + MalbehavD-V1 | Needs an `api_calls` (list of strings) + `label` (0/1) table |
| Cortex-Memory | CIC-MalMem-2022 | Group-aware split so dumps of the same sample never cross splits |
| Cortex-Network | CSE-CIC-IDS2018 (AWS Open Data) | IPs and timestamps dropped to prevent shortcut learning; 2M-row stratified sample |
| Cortex-Emulation | Quo Vadis Speakeasy (Apache-2.0) | Jan-2022 train, Apr-2022 test (deliberate time-drift holdout) |

---

## 7. Feature Engineering

**Static:** `features/pe_features.py` turns a PE file into a 2,568-value vector that follows EMBER2024's feature-group layout. It covers header fields, section entropy and sizes, imports and exports, byte histograms, strings and Authenticode signature data. If a critical feature group fails to extract, the scan returns `NEEDS_REVIEW` instead of scoring a silently zeroed vector.

**Behavioral / Emulation:** API names are mapped to integer IDs using a vocabulary built from the train split only. Traces are padded or truncated to a fixed length (100 for Behavioral, 500 for Emulation). Unknown APIs map to `<UNK>`.

**Memory:** 7 derived ratio features, each tied to a forensic indicator:
`callbacks_anonymous_ratio`, `svcscan_driver_ratio`, `handles_file_ratio`, `handles_mutant_ratio`, `psxview_hiding_score`, `ldrmodules_hidden_ratio`, `malfind_injection_rate`.

**Network:** 78 flow statistics. Identity columns (IPs, ports, flow ID) and timestamps are removed. Rows containing NaN or Infinity are dropped. Flows whose identical feature vectors carry both labels are excluded as ambiguous.

---

## 8. Model Training

### Cortex-Static (LightGBM)
| Parameter | Value |
|---|---|
| Num leaves | 255 |
| Max depth | 15 |
| Learning rate | 0.03 |
| Max boosting rounds | 3,000 (early stopping on val) |
| Row / column subsample | 0.8 / 0.7 |
| Class imbalance | `is_unbalance=True` |
| Calibration | Platt scaling on raw margins, fit on `cal` |

Memory and Network use the same LightGBM + Platt structure.

### Cortex-Behavioral (1D-CNN + Attention)
| Parameter | Value |
|---|---|
| Sequence length | 100 API calls |
| Embedding dimension | 128 (script default) |
| Conv blocks | 4 (kernels 3, 5, 7, 3) |
| Attention heads | 4 |
| Dropout | 0.3 |
| Optimizer | AdamW, LR 5e-4, weight decay 5e-4 |
| Batch size | 128 |
| Max epochs / patience | 150 / 15 |
| Loss | Binary cross-entropy with `pos_weight` |

---

## 9. Thresholds and Decision Policy

All thresholds are stored in [`config/thresholds.yaml`](config/thresholds.yaml). The policy engine loads the file at startup and fails if it is missing or incomplete.

| Signal | Rule | Derivation |
|---|---|---|
| Static | `< 0.479` → ALLOW, `< 0.981` → ALERT, else BLOCK | 1% / 0.05% FPR on `cal` |
| Behavioral | `≥ 0.60` → MALICIOUS | Sweep over val + test |
| Memory | `≥ 0.0025` → MALICIOUS | 1% FPR on `cal` |
| Network | `≥ 0.667` → MALICIOUS | 0.1% FPR on `cal` |
| Emulation | `≥ 0.9994` → MALICIOUS (logged only) | 1% FPR sweep |

The **policy engine** checks these rules in order and returns the first match:

| # | Condition | Decision |
|---|---|---|
| 1 | Behavioral is MALICIOUS and Static is ALERT or BLOCK | **TERMINATE** |
| 1b | Behavioral is MALICIOUS, Static is ALLOW or ERROR | **ALERT** (`behavioral_malicious_uncorroborated`; legacy TERMINATE if `behavioral.terminate_requires_corroboration: false`) |
| 2 | Static BLOCK **and** Memory or Network MALICIOUS | **BLOCK** |
| 3 | Memory or Network MALICIOUS | **ALERT** |
| 4 | Static ALERT, or Static BLOCK with no corroboration | **ALERT** |
| 5 | Any signal ERROR (file could not be analyzed) | **NEEDS_REVIEW** |
| 6 | Behavioral requested but PENDING (trace too short), Static ALLOW | **ALLOW_UNVERIFIED** (`behavioral_pending_unverified`; set by `behavioral.pending_with_static_allow`) |
| 7 | Otherwise | **ALLOW** |

**Truncated PE files** count as "could not be analyzed" (rule 5). Before the Static model scores a file, three structural checks run on the parsed PE. If any fires, Static is ERROR and the decision is **NEEDS_REVIEW**, with reason `static_pe_truncated` plus one detail code per rule:

| Detail code | Fires when |
|---|---|
| `pe_truncated:section_raw_beyond_eof:<bytes>` | a section's `PointerToRawData + SizeOfRawData` is past the end of the file (`<bytes>` = largest overrun; zero tolerance) |
| `pe_truncated:headers_beyond_eof` | `SizeOfHeaders` is larger than the file |
| `pe_truncated:certificate_table_beyond_eof` | the certificate table (a file offset) ends past the end of the file |

Measurement and rationale: [docs/TECHNICAL_NOTES.md](docs/TECHNICAL_NOTES.md#truncated-pe-files-f17).

> If you retrain a model, its score distribution changes and you must derive its thresholds again.

---

## 10. Quick Start

```bash
# 1. Install dependencies (Python >= 3.10)
pip install -r requirements.txt

# 2. Download and de-duplicate EMBER2024
python -m data.download_ember2024 --split train --out data/processed/ember2024_train.parquet
python -m data.download_ember2024 --split test  --out data/processed/ember2024_test.parquet

# 3. Train Cortex-Static (carves train/val/cal internally; never reads test)
python -m scripts.train_static \
  --train data/processed/ember2024_train.parquet \
  --out   data/models/cortex_static

# 4. Train Cortex-Behavioral
python -m scripts.train_behavioral \
  --train data/processed/behavioral_train.parquet \
  --val   data/processed/behavioral_val.parquet \
  --vocab-out      data/models/api_vocab.json \
  --checkpoint-out data/models/cortex_behavioral_best.pt

# 5. Evaluate every model at its deployed threshold
python -m scripts.evaluate_all_models

# 6. Check that ONNX exports match the Python models
python -m scripts.verify_onnx_parity

# 7. Run the tests
pytest
```

### Running a scan

```python
from models.behavioral_artifacts import load_behavioral_model
from models.static_lgbm import LGBMModel
from inference.pipeline import CortexPipeline
from inference.policy_engine import STATIC_MODEL_SHA256

static_model = LGBMModel.load("data/models/cortex_static",
                              expected_sha256=STATIC_MODEL_SHA256)  # JSON meta + sha256 pins (F24)
# Builds the model with the forward pass recorded in the checkpoint's sidecar
# (cortex_behavioral_best.meta.json) and verifies checkpoint + vocab sha256.
behavioral_model, tokenizer = load_behavioral_model(
    "data/models/cortex_behavioral_best.pt", "data/models/api_vocab.json")

pipeline = CortexPipeline(static_model, behavioral_model, tokenizer)
result = pipeline.scan(r"C:\Samples\application.exe",
                       api_calls_json_path=r"C:\Telemetry\api_calls.json")
print(pipeline.to_security_event(result))
```

---

## 11. Evaluation

`scripts/evaluate_all_models.py` prints the confusion matrix, AUC, FPR and detection rate for each model at its deployed threshold. The test splits are close to class-balanced, so the script also projects **precision and alert volume at realistic malware prevalence** (1 in 1,000 / 10,000 / 100,000 files):

```bash
python -m scripts.evaluate_all_models --prevalence 1e-4,1e-5 --target-ppv 0.5
```

The latest results are in [EVAL_ALL_MODELS_RESULTS.txt](EVAL_ALL_MODELS_RESULTS.txt) and [reports/static_retrain_20260921/](reports/static_retrain_20260921/).

---

## 12. Known Limitations

| Area | Limitation |
|---|---|
| Static | The feature-parity test against the reference `thrember` extractor is only partly done, because of a `signify` version conflict |
| Behavioral | Trained on 274 benign samples, so the threshold cannot promise a precise FPR |
| Memory | Near-perfect test scores come from single-VM benign data in the dataset, so Memory is capped at ALERT |
| Network | Detects infiltration attacks poorly (~11%), and the dataset has label noise, so Network is capped at ALERT |
| Emulation | Recall collapses on the Apr-2022 time slice (70.7% → 41.5%), so Emulation is logged only |

Full analysis of every item is in [docs/TECHNICAL_NOTES.md](docs/TECHNICAL_NOTES.md).

---

## 13. Tech Stack

| Component | Technology |
|---|---|
| Tree models | LightGBM |
| Deep learning | PyTorch |
| PE parsing | pefile, signify (Authenticode) |
| Data processing | Python, NumPy, pandas, PyArrow |
| ML utilities | scikit-learn |
| Datasets | HuggingFace `datasets`, AWS Open Data |
| Deployment format | ONNX, ONNX Runtime, onnxmltools, skl2onnx |
| Testing | pytest |

---

## 14. Further Reading

| Document | Contents |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Full runtime flow and the reasons behind each model choice |
| [docs/TECHNICAL_NOTES.md](docs/TECHNICAL_NOTES.md) | Per-signal derivations, dataset findings, and limitations in detail |
| [docs/Cortex_ML_Complete_Flow.txt](docs/Cortex_ML_Complete_Flow.txt) | Step-by-step plain-text walkthrough |
| [OPEN_ITEMS.md](OPEN_ITEMS.md) | Open work and plan of record |
| [PROJECT_HISTORY_REPORT.md](PROJECT_HISTORY_REPORT.md) | Development history |

---

## Future Improvements

- Close the remaining `pe_features.py` vs `thrember` feature-fidelity gap
- Rule-based overlay for high-risk API-call combinations (Behavioral)
- Validate Memory on real injected-process captures before lifting its ALERT cap
- Behavioral-category features and multi-period training data for Emulation drift
- ONNX export and pipeline wiring for Emulation once it is ready to ship
