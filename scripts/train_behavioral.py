"""
Train Cortex-Behavioral on tokenized API-call sequences.

Expects a JSONL/parquet dataset where each row has:
    api_calls: list[str]   (ordered API call names)
    label:     int         (0 = benign, 1 = malicious)

Usage:
    python -m scripts.train_behavioral \\
        --data data/processed/behavioral_dataset.parquet \\
        --vocab-out data/models/api_vocab.json \\
        --checkpoint-out data/models/cortex_behavioral_best.pt
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from models.behavioral_cnn import TrainConfig
from models.train_behavioral import train
from tokenizer.api_tokenizer import ApiTokenizer

logger = logging.getLogger("cortex.scripts.train_behavioral")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--vocab-out", required=True)
    ap.add_argument("--checkpoint-out", required=True)
    ap.add_argument("--min-vocab-count", type=int, default=2)
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    tokenizer = ApiTokenizer.build_from_corpus(df["api_calls"].tolist(), min_count=args.min_vocab_count)
    tokenizer.save(args.vocab_out)
    logger.info("Built vocab: %d tokens (incl. <PAD>/<UNK>)", tokenizer.vocab_size)

    X, statuses = tokenizer.encode_batch(df["api_calls"].tolist())
    y = df["label"].to_numpy(dtype=np.float32)

    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    pos_weight = n_neg / max(n_pos, 1)
    logger.info("dataset: %d samples (%d malicious / %d benign), pos_weight=%.3f", len(y), n_pos, n_neg, pos_weight)

    cfg = TrainConfig(vocab_size=tokenizer.vocab_size, embed_dim=128)
    model = train(X, y, cfg, checkpoint_path=args.checkpoint_out, pos_weight=pos_weight)
    logger.info("Training complete, best checkpoint at %s", args.checkpoint_out)


if __name__ == "__main__":
    main()
