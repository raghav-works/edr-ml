"""
Train Cortex-Static on deduplicated EMBER2024 parquet features.

Usage:
    python -m scripts.train_static \\
        --train data/processed/ember2024_train.parquet \\
        --test  data/processed/ember2024_test.parquet \\
        --out   data/models/cortex_static
"""

from __future__ import annotations

import argparse
import gc
import logging

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from models.static_lgbm import EMBER2024_FEATURE_COUNT, evaluate, train

logger = logging.getLogger("cortex.scripts.train_static")


def _feature_columns(columns) -> list[str]:
    feature_cols = [c for c in columns if c.startswith("feature_") or c.isdigit()]
    if len(feature_cols) != EMBER2024_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EMBER2024_FEATURE_COUNT} feature columns, found {len(feature_cols)}. "
            f"Adjust column-detection logic to match your parquet schema."
        )
    return feature_cols


def _split_xy(df: pd.DataFrame):
    label_col = "label" if "label" in df.columns else "y"
    feature_cols = _feature_columns(df.columns)
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[label_col].to_numpy(dtype=np.int32)
    # EMBER datasets use -1 for "unlabeled" — drop those for supervised training.
    mask = y != -1
    return X[mask], y[mask]


def _load_train_val_split(path: str, val_frac: float, seed: int):
    """Streams `path` row-group by row-group into preallocated train/val
    arrays instead of loading one full DataFrame, converting it to a second
    full numpy array, then slicing that twice via fancy indexing
    (X_all[tr_idx] / X_all[val_idx] each allocate a full copy while X_all is
    still referenced) -- that pattern holds ~4x a single split's raw array
    size in memory at peak, which OOM-kills training on the full 2.34M-row
    EMBER2024 train split on a 38GB machine. This streams one ~20K-row batch
    at a time and writes each row directly into its assigned train/val slot,
    so at most one batch plus the two output arrays are ever live.
    """
    pf = pq.ParquetFile(path)
    n_rows = pf.metadata.num_rows
    feature_cols = _feature_columns(pf.schema_arrow.names)
    label_col = "label" if "label" in pf.schema_arrow.names else "y"

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_rows)
    n_val = int(n_rows * val_frac)
    is_val = np.zeros(n_rows, dtype=bool)
    is_val[perm[:n_val]] = True
    dest_pos = np.empty(n_rows, dtype=np.int64)
    dest_pos[perm[:n_val]] = np.arange(n_val)
    dest_pos[perm[n_val:]] = np.arange(n_rows - n_val)

    X_train = np.empty((n_rows - n_val, len(feature_cols)), dtype=np.float32)
    X_val = np.empty((n_val, len(feature_cols)), dtype=np.float32)
    y_train = np.empty(n_rows - n_val, dtype=np.int32)
    y_val = np.empty(n_val, dtype=np.int32)

    offset = 0
    for rg in range(pf.metadata.num_row_groups):
        batch_df = pf.read_row_group(rg, columns=feature_cols + [label_col]).to_pandas()
        n = len(batch_df)
        Xb = batch_df[feature_cols].to_numpy(dtype=np.float32)
        yb = batch_df[label_col].to_numpy(dtype=np.int32)

        gi = slice(offset, offset + n)
        bval = is_val[gi]
        bpos = dest_pos[gi]

        X_val[bpos[bval]] = Xb[bval]
        y_val[bpos[bval]] = yb[bval]
        X_train[bpos[~bval]] = Xb[~bval]
        y_train[bpos[~bval]] = yb[~bval]

        offset += n
        del batch_df, Xb, yb

    # EMBER datasets use -1 for "unlabeled" — drop those for supervised training.
    tr_mask = y_train != -1
    val_mask = y_val != -1
    if not tr_mask.all():
        X_train, y_train = X_train[tr_mask], y_train[tr_mask]
    if not val_mask.all():
        X_val, y_val = X_val[val_mask], y_val[val_mask]
    return X_train, y_train, X_val, y_val


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args()

    X_train, y_train, X_val, y_val = _load_train_val_split(args.train, args.val_frac, seed=42)
    logger.info("train=%d val=%d features=%d", len(y_train), len(y_val), X_train.shape[1])

    model = train(X_train, y_train, X_val, y_val)
    # Calibration (inside train()) is the last thing that needed X_val; both
    # arrays are pure ~24GB dead weight from here on, so free main()'s own
    # references too -- train() deleting its local X_train parameter only
    # drops one of the two references to that array (this frame's `X_train`
    # name still points at it) and never touches X_val at all.
    del X_train, y_train, X_val, y_val
    gc.collect()

    test_df = pd.read_parquet(args.test)
    X_test, y_test = _split_xy(test_df)
    del test_df
    logger.info("test=%d", len(y_test))
    metrics = evaluate(model, X_test, y_test)
    logger.info("Test metrics: %s", metrics.to_dict())

    model.save(args.out)
    logger.info("Saved model to %s.lgbm / %s.meta", args.out, args.out)


if __name__ == "__main__":
    main()
