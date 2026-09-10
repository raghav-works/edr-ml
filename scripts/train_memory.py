"""
Train Cortex-Memory on the CIC-MalMem-2022 train/val/cal parquet splits
produced by scripts/split_memory.py.

Split discipline (PDF review items 2 and 3):
  --train  booster fit only.
  --val    LightGBM early stopping (best_iteration) only.
  --cal    Platt calibrator fit AND the config/thresholds.yaml operating-
           point derivation -- one split covers both, since each is a "fit a
           monotone map / pick an operating point" task that is never scored
           against, and a 5th split would only fragment this thin dataset.
  test     NOT read by this script at all. The test split is consumed exactly
           once, at the very end, by scripts/evaluate_all_models.py, for
           reported numbers only. Keeping test out of this file makes "the
           operating point was not chosen on data it is later scored against"
           structural rather than a matter of discipline.

The calibrator is fit on --cal, never on --val: best_iteration is chosen to
maximize val separation, so a calibrator fit on val margins is optimistically
over-confident on genuinely held-out data (see models/memory_lgbm.py::train()).

--cal for memory is deliberately enlarged (scripts/split_memory.py
--cal-frac 0.2, ~11.8k rows / ~5.9k benign) so its benign count matches what
the pre-split-discipline val+test-combined derivation relied on -- see this
script's logged benign-count report for the FPR resolution that supports.

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
        --cal   data/processed/memory_cal.parquet \\
        --out   data/models/cortex_memory
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from features.memory_features import add_derived_features, feature_matrix_columns
from models.memory_lgbm import find_threshold_for_fpr, train

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
    ap.add_argument("--cal", required=True,
                    help="Calibration split -- Platt fit + threshold derivation. NOT the "
                         "early-stopping val split, and NOT test.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    X_train, y_train = _load_xy(args.train)
    X_val, y_val = _load_xy(args.val)
    X_cal, y_cal = _load_xy(args.cal)
    logger.info("train=%d val=%d cal=%d features=%d (test not read here -- see module docstring)",
                len(y_train), len(y_val), len(y_cal), X_train.shape[1])

    n_benign_cal = int((y_cal == 0).sum())
    logger.info(
        "Benign samples on the calibration split for threshold derivation: %d. "
        "Resolution: one false positive moves the observed FPR by ~%.4f%% (1 / %d).",
        n_benign_cal, 100.0 / n_benign_cal, n_benign_cal,
    )

    model = train(X_train, y_train, X_val, y_val, X_cal, y_cal)

    proba_cal = model.predict_proba(X_cal)
    benign_mask = y_cal == 0
    malicious_mask = ~benign_mask

    logger.info("=== Threshold sweep on the CALIBRATION split (%d rows, %d benign) ===",
                len(y_cal), int(benign_mask.sum()))
    logger.info("    (test is untouched; pick the config/thresholds.yaml value from this table)")
    for target_fpr in (0.001, 0.005, 0.01, 0.02, 0.05):
        t = find_threshold_for_fpr(y_cal, proba_cal, target_fpr)
        preds = (proba_cal >= t).astype(np.int32)
        n_fp = int((preds[benign_mask] == 1).sum())
        actual_fpr = n_fp / int(benign_mask.sum())
        det_rate = float((preds[malicious_mask] == 1).mean())
        logger.info(
            "target_fpr=%.4f -> threshold=%.10f actual_fpr=%.6f (%d/%d benign FP) detection_rate=%.4f",
            target_fpr, t, actual_fpr, n_fp, int(benign_mask.sum()), det_rate,
        )

    model.save(args.out)
    logger.info("Saved model to %s.lgbm / %s.meta", args.out, args.out)


if __name__ == "__main__":
    main()
