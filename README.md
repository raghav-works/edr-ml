# cortex-ml — Static (LightGBM) + Behavioral (1D-CNN + Attention) reproduction

Independent reimplementation, built to the architecture you specified
(`Architecture.txt`): **sequential gating, no ensembling**. Static runs
first; only a static **ALLOW** verdict lets a file proceed to behavioral
analysis. Scores are never averaged — the policy engine combines *verdicts*
by fixed priority rules.

```
path validation → PE validation → 2568-dim feature extraction → LightGBM
        → static verdict (ALLOW / ALERT / BLOCK)
        → [only if ALLOW] → API-call tokenization → 1D-CNN+Attention
              → behavioral verdict (BENIGN / MALICIOUS / PENDING)
        → policy engine → final decision (ALLOW / ALERT / BLOCK / TERMINATE)
        → structured JSON security event
```

## Layout
```
cortex/
├── features/pe_features.py       # 2568-dim EMBER2024-compatible extractor (pefile-based)
├── models/
│   ├── static_lgbm.py            # LightGBM train/eval/calibrate/persist
│   ├── behavioral_cnn.py         # 1D-CNN + multi-head attention, single sigmoid score
│   └── train_behavioral.py       # training loop (AdamW, cosine restarts, early stop)
├── tokenizer/api_tokenizer.py    # API-name → token-ID, fixed length 100, <PAD>/<UNK>
├── data/download_ember2024.py    # HF download + de-dup (see note below)
├── export/
│   ├── export_onnx.py            # LightGBM→ONNX (onnxmltools), PyTorch→ONNX
│   └── quantize.py               # dynamic INT8 quantization + FP32/INT8 comparison + latency bench
├── inference/
│   ├── policy_engine.py          # exact threshold + priority-rule decision logic
│   └── pipeline.py               # end-to-end scan() matching the architecture flow
├── scripts/
│   ├── train_static.py
│   └── train_behavioral.py
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

export_static_lgbm_to_onnx("data/models/cortex_static.lgbm", "data/models/cortex_static.onnx")
# static model: NO INT8 quantization (tree ensemble — no effect on split thresholds)

export_behavioral_to_onnx(model, "data/models/cortex_behavioral.onnx")
quantize_behavioral_int8("data/models/cortex_behavioral.onnx", "data/models/cortex_behavioral_int8.onnx")
compare_accuracy("data/models/cortex_behavioral.onnx", "data/models/cortex_behavioral_int8.onnx")
benchmark_latency("data/models/cortex_behavioral_int8.onnx")
```

## Thresholds
Hard-coded in `inference/policy_engine.py` (mirrored in
`config/thresholds.yaml`) exactly as given:
- static score `< 0.6634794478` → ALLOW, `< 0.9882189978` → ALERT, else BLOCK
- behavioral score `>= 0.9910649657` → MALICIOUS, else BENIGN

These were supplied as-is from the prior project; they were presumably
calibrated against a specific validation set to hit target FPR/detection
rate. If you retrain from scratch, your raw score distribution will differ,
so re-derive them with `models.static_lgbm.find_threshold_for_fpr()` against
your own held-out set rather than assuming these exact numbers still hit the
same FPR/detection-rate targets.

## Running a scan end-to-end
```python
from models.static_lgbm import LGBMModel
from tokenizer.api_tokenizer import ApiTokenizer
from inference.pipeline import CortexPipeline
import torch

static_model = LGBMModel.load("data/models/cortex_static")
behavioral_model = torch.load("data/models/cortex_behavioral_full.pt")  # or reconstruct + load_state_dict
tokenizer = ApiTokenizer.load("data/models/api_vocab.json")

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
