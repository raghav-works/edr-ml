"""
Train Cortex-Memory on the CIC-MalMem-2022 train/val/test parquet splits
produced by scripts/split_memory.py.

Trains on --train only; --val drives early stopping and Platt calibration
(mirroring models/static_lgbm.py::train() exactly); --test stays untouched
until the final evaluate() call. The operating threshold is then derived
from --val and --test COMBINED (not train, and not inherited from anywhere
else) -- MEMORY_MALICIOUS_MIN in inference/policy_engine.py currently
raises NotImplementedError specifically pending this. See this script's
logged benign-count report for why val+test are combined rather than using
either split alone, and which FPR targets that combined count can actually
support -- same statistical-power discipline as
BEHAVIORAL_MALICIOUS_MIN's derivation.

Feature scaling note: features/memory_features.py's MemoryFeatureScaler is
deliberately NOT applied here. LightGBM (like any tree-ensemble model)
splits on per-feature thresholds, and its trees -- therefore its
predictions -- are provably invariant to any monotonic per-feature
transform, including standardization (mean-center + scale). Applying the
scaler would change every feature value but not a single split decision or
prediction, while adding a second artifact (fit on train) that would also
need to be baked into the ONNX export for zero actual effect. The derived
ratio features from add_derived_features() ARE applied here -- they add
real information a monotonic rescaling doesn't. MemoryFeatureScaler stays
available in features/memory_features.py for a future non-tree Cortex-
Memory variant, where scaling would matter.

Usage:
    python -m scripts.train_memory \\
        --train data/processed/memory_train.parquet \\
        --val   data/processed/memory_val.parquet \\
        --test  data/processed/memory_test.parquet \\
        --out   data/models/cortex_memory
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from features.memory_features import add_derived_features, feature_matrix_columns
from models.memory_lgbm import evaluate, find_threshold_for_fpr, train

logger = logging.getLogger("cortex.scripts.train_memory")


def _load_xy(path: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(path)
    df = add_derived_features(df)
    cols = feature_matrix_columns(df)
    X = df[cols].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int32)
    return X, y


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    X_train, y_train = _load_xy(args.train)
    X_val, y_val = _load_xy(args.val)
    X_test, y_test = _load_xy(args.test)
    logger.info("train=%d val=%d test=%d features=%d", len(y_train), len(y_val), len(y_test), X_train.shape[1])

    n_benign_val = int((y_val == 0).sum())
    n_benign_test = int((y_test == 0).sum())
    n_benign_combined = n_benign_val + n_benign_test
    logger.info(
        "Benign sample counts for threshold derivation -- val=%d test=%d combined=%d. "
        "Resolution: one false positive on the combined set moves the observed FPR by "
        "~%.4f%% (1 / %d).",
        n_benign_val, n_benign_test, n_benign_combined, 100.0 / n_benign_combined, n_benign_combined,
    )

    model = train(X_train, y_train, X_val, y_val)

    metrics = evaluate(model, X_test, y_test, threshold=0.5)
    logger.info("Test metrics @ threshold=0.5 (default, not yet the derived operating threshold): %s",
                metrics.to_dict())

    X_valtest = np.concatenate([X_val, X_test], axis=0)
    y_valtest = np.concatenate([y_val, y_test], axis=0)
    proba_valtest = model.predict_proba(X_valtest)
    benign_mask = y_valtest == 0
    malicious_mask = ~benign_mask

    logger.info("=== Threshold sweep on val+test combined (%d rows, %d benign) ===",
                len(y_valtest), int(benign_mask.sum()))
    for target_fpr in (0.001, 0.005, 0.01, 0.02, 0.05):
        t = find_threshold_for_fpr(y_valtest, proba_valtest, target_fpr)
        preds = (proba_valtest >= t).astype(np.int32)
        n_fp = int((preds[benign_mask] == 1).sum())
        actual_fpr = n_fp / int(benign_mask.sum())
        det_rate = float((preds[malicious_mask] == 1).mean())
        logger.info(
            "target_fpr=%.4f -> threshold=%.6f actual_fpr=%.6f (%d/%d benign FP) detection_rate=%.4f",
            target_fpr, t, actual_fpr, n_fp, int(benign_mask.sum()), det_rate,
        )

    model.save(args.out)
    logger.info("Saved model to %s.lgbm / %s.meta", args.out, args.out)


if __name__ == "__main__":
    main()
