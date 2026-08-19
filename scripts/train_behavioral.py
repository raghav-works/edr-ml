"""
Train Cortex-Behavioral on tokenized API-call sequences.

Expects the train/val parquet files produced by scripts/split_behavioral.py
(each row: api_calls list[str], label int). Vocabulary and pos_weight are
computed from --train only -- val must stay unseen by anything derived from
the corpus, same leakage discipline as the split itself. --test is never
touched here; it's reserved for the held-out evaluation stage.

Usage:
    python -m scripts.train_behavioral \\
        --train data/processed/behavioral_train.parquet \\
        --val   data/processed/behavioral_val.parquet \\
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
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--vocab-out", required=True)
    ap.add_argument("--checkpoint-out", required=True)
    ap.add_argument("--min-vocab-count", type=int, default=1)
    ap.add_argument("--embed-dim", type=int, default=128)
    args = ap.parse_args()

    train_df = pd.read_parquet(args.train)
    val_df = pd.read_parquet(args.val)

    tokenizer = ApiTokenizer.build_from_corpus(train_df["api_calls"].tolist(), min_count=args.min_vocab_count)
    tokenizer.save(args.vocab_out)
    logger.info("Built vocab from train split only: %d tokens (incl. <PAD>/<UNK>)", tokenizer.vocab_size)

    X_train, train_statuses = tokenizer.encode_batch(train_df["api_calls"].tolist())
    y_train = train_df["label"].to_numpy(dtype=np.float32)
    X_val, val_statuses = tokenizer.encode_batch(val_df["api_calls"].tolist())
    y_val = val_df["label"].to_numpy(dtype=np.float32)

    n_pos, n_neg = int(y_train.sum()), int((1 - y_train).sum())
    pos_weight = n_neg / max(n_pos, 1)
    logger.info(
        "train=%d (%d malicious / %d benign, pos_weight=%.4f)  val=%d",
        len(y_train), n_pos, n_neg, pos_weight, len(y_val),
    )

    cfg = TrainConfig(vocab_size=tokenizer.vocab_size, embed_dim=args.embed_dim)
    model = train(X_train, y_train, X_val, y_val, cfg, checkpoint_path=args.checkpoint_out, pos_weight=pos_weight)
    logger.info("Training complete, best checkpoint at %s", args.checkpoint_out)


if __name__ == "__main__":
    main()
