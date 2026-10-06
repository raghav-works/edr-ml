"""
Evaluate Cortex-Behavioral on the held-out test split -- never touched by
anything in Stage A (split), B (vocab), or C (training).

Usage:
    python -m scripts.evaluate_behavioral \\
        --test data/processed/behavioral_test.parquet \\
        --vocab data/models/api_vocab.json \\
        --checkpoint data/models/cortex_behavioral_best.pt
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from models.behavioral_artifacts import load_behavioral_model
from models.train_behavioral import evaluate_behavioral, predict_proba_behavioral

logger = logging.getLogger("cortex.scripts.evaluate_behavioral")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    test_df = pd.read_parquet(args.test)
    # architecture (embed_dim, use_padding_mask) comes from the sidecar
    model, tokenizer = load_behavioral_model(args.checkpoint, args.vocab)

    X_test, statuses = tokenizer.encode_batch(test_df["api_calls"].tolist())
    y_test = test_df["label"].to_numpy(dtype=np.float32)

    logger.info("test=%d (%d malicious / %d benign)", len(y_test), int(y_test.sum()), int((1 - y_test).sum()))

    metrics = evaluate_behavioral(model, X_test, y_test, threshold=args.threshold)
    logger.info("=== Overall + per-class metrics ===")
    for k, v in metrics.to_dict().items():
        logger.info("  %s: %s", k, v)

    # Per-source breakdown -- does the model perform differently on
    # malbehavd-v1-sourced benign vs carpenter-sourced benign?
    logger.info("=== Per-source breakdown ===")
    proba = predict_proba_behavioral(model, X_test)
    preds = (proba >= args.threshold).astype(int)
    for source in sorted(test_df["source"].unique()):
        mask = (test_df["source"] == source).to_numpy()
        for label in sorted(test_df.loc[mask, "label"].unique()):
            sub_mask = mask & (test_df["label"].to_numpy() == label)
            n = int(sub_mask.sum())
            if n == 0:
                continue
            correct = int((preds[sub_mask] == label).sum())
            mean_proba = float(proba[sub_mask].mean())
            logger.info(
                "  source=%-13s label=%d  n=%4d  accuracy=%.4f  mean_predicted_proba=%.4f",
                source, label, n, correct / n, mean_proba,
            )


if __name__ == "__main__":
    main()
