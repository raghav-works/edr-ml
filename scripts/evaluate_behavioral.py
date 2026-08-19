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
import torch

from models.behavioral_cnn import CortexBehavioralNet, SEQUENCE_LENGTH
from models.train_behavioral import evaluate_behavioral, predict_proba_behavioral
from tokenizer.api_tokenizer import ApiTokenizer

logger = logging.getLogger("cortex.scripts.evaluate_behavioral")


def load_model(checkpoint_path: str, vocab_size: int, embed_dim: int = 128) -> CortexBehavioralNet:
    model = CortexBehavioralNet(vocab_size=vocab_size, sequence_length=SEQUENCE_LENGTH, embed_dim=embed_dim)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    test_df = pd.read_parquet(args.test)
    tokenizer = ApiTokenizer.load(args.vocab)
    model = load_model(args.checkpoint, tokenizer.vocab_size, args.embed_dim)

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
