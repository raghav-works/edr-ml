"""
Train Cortex-Network on the CSE-CIC-IDS2018 train/val/cal parquet splits
produced by scripts/split_network.py.

Split discipline (PDF review items 2 and 3), identical to
scripts/train_memory.py:
  --train  booster fit only.
  --val    LightGBM early stopping (best_iteration) only.
  --cal    Platt calibrator fit AND the config/thresholds.yaml operating-
           point derivation.
  test     NOT read by this script at all. The test split -- including the
           per-attack-type detection breakdown this script used to compute --
           is consumed exactly once, at the end, by
           scripts/evaluate_all_models.py, for reported numbers only.

The calibrator is fit on --cal, never on --val: best_iteration maximizes val
separation, so a calibrator fit on val margins is optimistically
over-confident on genuinely held-out data (see models/network_lgbm.py::train()).

One check this script still runs on its own, because ingestion history
warrants it:

  Single-feature AUC sanity check, BEFORE trusting any aggregate metric.
  CSE-CIC-IDS2018 is a fixed-testbed dataset (a small number of specific
  victim/attacker machines), the same category that made Cortex-Memory's
  near-perfect metrics suspicious. Flow ID/Src IP/Src Port/Dst IP were
  already dropped at ingestion for exactly this reason, but that doesn't
  guarantee none of the remaining 78 features ended up an inadvertent
  shortcut. It is computed on train and on --cal (both untouched by the
  booster fit) -- consistent behaviour on cal rules out plain overfitting
  as the explanation. This is a feature-level leakage diagnostic, not a
  model metric, so it does not need the test split.

Usage:
    python -m scripts.train_network \\
        --train data/processed/network_train.parquet \\
        --val   data/processed/network_val.parquet \\
        --cal   data/processed/network_cal.parquet \\
        --out   data/models/cortex_network
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from data.download_network import FEATURE_COLUMNS
from models.network_lgbm import find_threshold_for_fpr, train

logger = logging.getLogger("cortex.scripts.train_network")


def _load(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def _xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int32)
    return X, y


def single_feature_auc_check(train_df: pd.DataFrame, cal_df: pd.DataFrame, top_n: int = 15) -> None:
    """Direction-agnostic single-feature AUC for every one of the 78
    features, computed independently on train and on the calibration split
    -- mirrors the check that caught Cortex-Memory's single-VM benign
    shortcut. A feature exceeding ~0.95 AUC on its own, on BOTH splits, is
    worth explaining before trusting the aggregate model metrics; cal is
    never seen by the booster fit or early stopping, so consistency there
    rules out plain overfitting as the explanation. (The test split is not
    used -- this is a feature-level diagnostic, and test stays untouched
    until scripts/evaluate_all_models.py.)"""
    logger.info("=== Single-feature AUC sanity check ===")
    for name, df in (("train", train_df), ("cal", cal_df)):
        y = df["label"].to_numpy()
        aucs = []
        for c in FEATURE_COLUMNS:
            x = df[c].to_numpy(dtype=np.float64)
            a = roc_auc_score(y, x)
            aucs.append((c, max(a, 1 - a)))
        aucs.sort(key=lambda t: -t[1])
        logger.info("Top %d single-feature AUCs on %s:", top_n, name)
        for c, a in aucs[:top_n]:
            logger.info("  %-25s AUC=%.6f", c, a)
        n_suspicious = sum(1 for _, a in aucs if a > 0.95)
        logger.info("%s: %d / %d features exceed 0.95 AUC individually", name, n_suspicious, len(aucs))


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

    train_df, val_df, cal_df = _load(args.train), _load(args.val), _load(args.cal)
    X_train, y_train = _xy(train_df)
    X_val, y_val = _xy(val_df)
    X_cal, y_cal = _xy(cal_df)
    logger.info("train=%d val=%d cal=%d features=%d (test not read here -- see module docstring)",
                len(y_train), len(y_val), len(y_cal), X_train.shape[1])

    n_benign_cal = int((y_cal == 0).sum())
    logger.info(
        "Benign samples on the calibration split for threshold derivation: %d. "
        "Resolution: one false positive moves the observed FPR by ~%.5f%% (1 / %d) -- "
        "far finer than memory's; FPR targets well below 0.1%% are defensible here.",
        n_benign_cal, 100.0 / n_benign_cal, n_benign_cal,
    )

    single_feature_auc_check(train_df, cal_df)

    t0 = time.monotonic()
    model = train(X_train, y_train, X_val, y_val, X_cal, y_cal)
    wall_clock_s = time.monotonic() - t0
    logger.info("=== Training wall-clock: %.1fs (%.1f min) ===", wall_clock_s, wall_clock_s / 60)

    proba_cal = model.predict_proba(X_cal)
    benign_mask = y_cal == 0
    malicious_mask = ~benign_mask

    logger.info("=== Threshold sweep on the CALIBRATION split (%d rows, %d benign) ===",
                len(y_cal), int(benign_mask.sum()))
    logger.info("    (test is untouched; pick the config/thresholds.yaml value from this table)")
    for target_fpr in (0.0001, 0.0005, 0.001, 0.005, 0.01):
        t = find_threshold_for_fpr(y_cal, proba_cal, target_fpr)
        preds = (proba_cal >= t).astype(np.int32)
        n_fp = int((preds[benign_mask] == 1).sum())
        actual_fpr = n_fp / int(benign_mask.sum())
        det_rate = float((preds[malicious_mask] == 1).mean())
        logger.info(
            "target_fpr=%.5f -> threshold=%.10f actual_fpr=%.6f (%d/%d benign FP) detection_rate=%.4f",
            target_fpr, t, actual_fpr, n_fp, int(benign_mask.sum()), det_rate,
        )

    model.save(args.out)
    logger.info("Saved model to %s.lgbm / %s.meta.json", args.out, args.out)


if __name__ == "__main__":
    main()
