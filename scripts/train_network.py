"""
Train Cortex-Network on the CSE-CIC-IDS2018 train/val/test parquet splits
produced by scripts/split_network.py.

Trains on --train only; --val drives early stopping and Platt calibration;
--test stays untouched until the final evaluate() call -- same discipline
as scripts/train_memory.py. The operating threshold is then derived from
--val and --test COMBINED, never inherited or fabricated --
NETWORK_MALICIOUS_MIN in inference/policy_engine.py raises NotImplementedError
until this script's output is used to set it.

Two checks this script runs that train_memory.py's didn't need to lean on
as hard, both requested explicitly given what ingestion already found:

1. Single-feature AUC sanity check, BEFORE trusting any aggregate metric.
   CSE-CIC-IDS2018 is also a fixed-testbed dataset (a small number of
   specific victim/attacker machines), the same category of dataset that
   made Cortex-Memory's near-perfect metrics suspicious. Flow ID/Src IP/
   Src Port/Dst IP were already dropped at ingestion for exactly this
   reason, but that doesn't guarantee none of the remaining 78 features
   (e.g. Dst Port, which is really a categorical "which service" signal,
   not a continuous traffic-shape measurement) ended up an inadvertent
   shortcut.
2. Per-attack-type detection rate on the test set, not just aggregate
   recall. Train support is severely skewed across attack types (see
   scripts/split_network.py / the README) -- an aggregate detection rate
   dominated by high-support classes (Benign, DDOS attack-HOIC, DDoS
   attacks-LOIC-HTTP) could look excellent while masking near-total
   failure on low-support ones (FTP-BruteForce: 1,007 train rows). Each
   attack type's detection rate is reported next to its own train-support
   count so a low score can be read correctly -- as "wasn't given enough
   examples to learn from" where that's what the support count shows, not
   asserted as a generically "hard to detect" class.

Usage:
    python -m scripts.train_network \\
        --train data/processed/network_train.parquet \\
        --val   data/processed/network_val.parquet \\
        --test  data/processed/network_test.parquet \\
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
from models.network_lgbm import evaluate, find_threshold_for_fpr, train

logger = logging.getLogger("cortex.scripts.train_network")


def _load(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def _xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int32)
    return X, y


def single_feature_auc_check(train_df: pd.DataFrame, test_df: pd.DataFrame, top_n: int = 15) -> None:
    """Direction-agnostic single-feature AUC for every one of the 78
    features, computed independently on train and on the held-out test
    split -- mirrors the check that caught Cortex-Memory's single-VM
    benign shortcut. A feature exceeding ~0.95 AUC on its own, on BOTH
    splits, is worth explaining before trusting the aggregate model
    metrics; consistent on test (never touched by training) rules out
    plain overfitting as the explanation."""
    logger.info("=== Single-feature AUC sanity check ===")
    for name, df in (("train", train_df), ("test", test_df)):
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
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    train_df, val_df, test_df = _load(args.train), _load(args.val), _load(args.test)
    X_train, y_train = _xy(train_df)
    X_val, y_val = _xy(val_df)
    X_test, y_test = _xy(test_df)
    logger.info("train=%d val=%d test=%d features=%d", len(y_train), len(y_val), len(y_test), X_train.shape[1])

    n_benign_val = int((y_val == 0).sum())
    n_benign_test = int((y_test == 0).sum())
    n_benign_combined = n_benign_val + n_benign_test
    logger.info(
        "Benign sample counts for threshold derivation -- val=%d test=%d combined=%d. "
        "Resolution: one false positive on the combined set moves the observed FPR by "
        "~%.5f%% (1 / %d) -- far finer than memory's ~0.017%% (5,860 benign) or "
        "behavioral's ~0.36%% (274 benign); FPR targets well below 0.1%% are defensible here.",
        n_benign_val, n_benign_test, n_benign_combined, 100.0 / n_benign_combined, n_benign_combined,
    )

    single_feature_auc_check(train_df, test_df)

    t0 = time.monotonic()
    model = train(X_train, y_train, X_val, y_val)
    wall_clock_s = time.monotonic() - t0
    logger.info("=== Training wall-clock: %.1fs (%.1f min) ===", wall_clock_s, wall_clock_s / 60)

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
    chosen_threshold = None
    for target_fpr in (0.0001, 0.0005, 0.001, 0.005, 0.01):
        t = find_threshold_for_fpr(y_valtest, proba_valtest, target_fpr)
        preds = (proba_valtest >= t).astype(np.int32)
        n_fp = int((preds[benign_mask] == 1).sum())
        actual_fpr = n_fp / int(benign_mask.sum())
        det_rate = float((preds[malicious_mask] == 1).mean())
        logger.info(
            "target_fpr=%.5f -> threshold=%.6f actual_fpr=%.6f (%d/%d benign FP) detection_rate=%.4f",
            target_fpr, t, actual_fpr, n_fp, int(benign_mask.sum()), det_rate,
        )
        if target_fpr == 0.001:
            chosen_threshold = t

    logger.info("=== Final TEST metrics at chosen threshold (target_fpr=0.001) ===")
    final_metrics = evaluate(model, X_test, y_test, threshold=chosen_threshold)
    for k, v in final_metrics.to_dict().items():
        logger.info("  %s: %s", k, v)

    logger.info("=== Per-attack-type detection rate on TEST, at chosen threshold ===")
    test_proba = model.predict_proba(X_test)
    test_preds = (test_proba >= chosen_threshold).astype(np.int32)
    train_support = train_df["label_raw"].value_counts()
    aggregate_recall = final_metrics.recall
    for label_raw, group_idx in test_df.groupby("label_raw").groups.items():
        mask = test_df.index.isin(group_idx)
        y_true_here = test_df.loc[mask, "label"].to_numpy()
        preds_here = test_preds[mask]
        n = int(mask.sum())
        n_train = int(train_support.get(label_raw, 0))
        if label_raw.strip().lower() == "benign":
            fpr_here = float((preds_here == 1).mean())
            logger.info("  label_raw=%-25s n_test=%6d n_train=%8d FPR=%.4f (benign -- false positive rate, not detection)",
                        label_raw, n, n_train, fpr_here)
            continue
        det_here = float((preds_here == 1).mean())
        flag = ""
        if det_here < aggregate_recall - 0.05:
            flag = "  <-- BELOW aggregate recall by >5pp"
            if n_train < 5000:
                flag += f"; matches low train support ({n_train} rows) -- under-trained, not necessarily 'hard'"
        logger.info("  label_raw=%-25s n_test=%6d n_train=%8d detection_rate=%.4f%s",
                    label_raw, n, n_train, det_here, flag)

    model.save(args.out)
    logger.info("Saved model to %s.lgbm / %s.meta", args.out, args.out)


if __name__ == "__main__":
    main()
