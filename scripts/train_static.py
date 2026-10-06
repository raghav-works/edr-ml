"""
Train Cortex-Static on deduplicated EMBER2024 parquet features.

Split discipline (PDF review items 2 and 3), applying the same mechanism
already proven on memory/network (scripts/split_memory.py / split_network.py;
see OPEN_ITEMS.md's "The retrain cluster" section, "Agreed carve"):
  train  booster fit only.
  val    LightGBM early stopping (best_iteration) only. Carved first from a
         seeded permutation of data/processed/ember2024_train.parquet's rows
         -- unaffected by the cal addition below, so it is the exact same
         rows as before the split-discipline fix (same seed, same val_frac,
         sliced off the permutation before cal is ever considered).
  cal    Platt calibrator fit AND the config/thresholds.yaml operating-point
         derivation (STATIC_ALLOW_MAX / STATIC_BLOCK_MIN) -- one split
         covers both, since each is a "fit a monotone map / pick an
         operating point" task that is never scored against. Carved from
         what would otherwise have been train: the next cal_frac slice of
         the same permutation, immediately after val.
  test   NOT read by this script at all. data/processed/ember2024_test.parquet
         is read exactly once, at the very end, by
         scripts/evaluate_all_models.py, for reported numbers only -- there
         is no --test flag here, so "test untouched during training" is
         structural, not disciplinary (same enforcement as train_memory.py /
         train_network.py).

Memory discipline -- cal is loaded in a SEPARATE pass, after training:
  A prior run on this machine crashed inside lgb.Dataset() construction
  (binning 2568 features) when train/val/cal were all loaded up front and
  held resident through that phase -- traced to lgb.Dataset() not
  inheriting the n_jobs thread cap (fixed in models/static_lgbm.py::train())
  plus cal's raw array (~2.2GB) sitting idle through the riskiest window
  for no reason, since it's not used until calibration. This script now
  loads train+val, trains (X_train/X_val freed inside train()), THEN loads
  cal fresh -- mirroring scripts/evaluate_all_models.py::_load_ember_val's
  stream-and-keep-only-matching-rows pattern -- so cal is never resident at
  the same time as the Dataset-construction/boosting peak.

The calibrator is fit on cal, never on val: best_iteration is chosen to
maximize val separation, so a calibrator fit on val margins is
optimistically over-confident on genuinely held-out data (PDF review item 2,
see models/static_lgbm.py::train()/calibrate()).

Usage:
    python -m scripts.train_static \\
        --train data/processed/ember2024_train.parquet \\
        --out   data/models/cortex_static
"""

from __future__ import annotations

import argparse
import gc
import logging

import numpy as np
import pyarrow.parquet as pq

from models.static_lgbm import EMBER2024_FEATURE_COUNT, calibrate, train_from_holder

logger = logging.getLogger("cortex.scripts.train_static")


def _feature_columns(columns) -> list[str]:
    feature_cols = [c for c in columns if c.startswith("feature_") or c.isdigit()]
    if len(feature_cols) != EMBER2024_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EMBER2024_FEATURE_COUNT} feature columns, found {len(feature_cols)}. "
            f"Adjust column-detection logic to match your parquet schema."
        )
    return feature_cols


def _permutation_bounds(n_rows: int, val_frac: float, cal_frac: float, seed: int):
    """The one seeded permutation everything else in this module derives
    from. n_val/n_cal boundaries only -- callers slice `perm` themselves.
    Kept as a single function so _load_train_val and _load_cal can never
    silently drift out of sync on the seed/fracs and carve overlapping or
    gapped row sets.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_rows)
    n_val = int(n_rows * val_frac)
    n_cal = int(n_rows * cal_frac)
    return perm, n_val, n_cal


def _load_train_val(path: str, val_frac: float, cal_frac: float, seed: int):
    """Streams `path` row-group by row-group into preallocated train/val
    arrays, skipping cal rows entirely (never allocated, never copied) --
    at most one batch plus the two output arrays are ever live. This
    streams instead of loading one full DataFrame, converting it to a
    second full numpy array, then slicing that via fancy indexing, which
    holds ~4x a single split's raw array size in memory at peak.

    perm[:n_val] is val, perm[n_val:n_val+n_cal] is cal (skipped here,
    loaded separately by _load_cal after training -- see module
    docstring), the remainder is train.
    """
    pf = pq.ParquetFile(path)
    n_rows = pf.metadata.num_rows
    feature_cols = _feature_columns(pf.schema_arrow.names)
    label_col = "label" if "label" in pf.schema_arrow.names else "y"

    perm, n_val, n_cal = _permutation_bounds(n_rows, val_frac, cal_frac, seed)
    n_train = n_rows - n_val - n_cal

    # dest: 0=train, 1=val, 2=cal (skipped), assigned by permutation position.
    dest = np.zeros(n_rows, dtype=np.int8)
    dest[perm[:n_val]] = 1
    dest[perm[n_val:n_val + n_cal]] = 2
    dest_pos = np.empty(n_rows, dtype=np.int64)
    dest_pos[perm[:n_val]] = np.arange(n_val)
    dest_pos[perm[n_val + n_cal:]] = np.arange(n_train)

    X_train = np.empty((n_train, len(feature_cols)), dtype=np.float32)
    X_val = np.empty((n_val, len(feature_cols)), dtype=np.float32)
    y_train = np.empty(n_train, dtype=np.int32)
    y_val = np.empty(n_val, dtype=np.int32)

    offset = 0
    for rg in range(pf.metadata.num_row_groups):
        batch_df = pf.read_row_group(rg, columns=feature_cols + [label_col]).to_pandas()
        n = len(batch_df)
        gi = slice(offset, offset + n)
        bdest = dest[gi]
        bval, btrain = bdest == 1, bdest == 0
        keep = bval | btrain
        if keep.any():
            Xb = batch_df.loc[keep, feature_cols].to_numpy(dtype=np.float32)
            yb = batch_df.loc[keep, label_col].to_numpy(dtype=np.int32)
            bpos = dest_pos[gi][keep]
            bval_k = bval[keep]

            X_val[bpos[bval_k]] = Xb[bval_k]
            y_val[bpos[bval_k]] = yb[bval_k]
            X_train[bpos[~bval_k]] = Xb[~bval_k]
            y_train[bpos[~bval_k]] = yb[~bval_k]
            del Xb, yb

        offset += n
        del batch_df

    # EMBER datasets use -1 for "unlabeled" -- drop those for supervised training.
    tr_mask = y_train != -1
    val_mask = y_val != -1
    if not tr_mask.all():
        X_train, y_train = X_train[tr_mask], y_train[tr_mask]
    if not val_mask.all():
        X_val, y_val = X_val[val_mask], y_val[val_mask]
    return X_train, y_train, X_val, y_val


def _load_cal(path: str, val_frac: float, cal_frac: float, seed: int):
    """Streams `path` a second time, keeping only cal rows (perm[n_val:
    n_val+n_cal]) -- the other ~90% is never materialised. Same
    stream-and-keep-only-matching-rows shape as
    scripts/evaluate_all_models.py::_load_ember_val. Called only after
    training, once X_train/X_val are already freed -- see module
    docstring for why.
    """
    pf = pq.ParquetFile(path)
    n_rows = pf.metadata.num_rows
    feature_cols = _feature_columns(pf.schema_arrow.names)
    label_col = "label" if "label" in pf.schema_arrow.names else "y"

    perm, n_val, n_cal = _permutation_bounds(n_rows, val_frac, cal_frac, seed)
    is_cal = np.zeros(n_rows, dtype=bool)
    is_cal[perm[n_val:n_val + n_cal]] = True

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    offset = 0
    for rg in range(pf.metadata.num_row_groups):
        batch_df = pf.read_row_group(rg, columns=feature_cols + [label_col]).to_pandas()
        n = len(batch_df)
        mask = is_cal[offset:offset + n]
        offset += n
        if mask.any():
            X_parts.append(batch_df.loc[mask, feature_cols].to_numpy(dtype=np.float32))
            y_parts.append(batch_df.loc[mask, label_col].to_numpy(dtype=np.int32))
        del batch_df

    X_cal = np.concatenate(X_parts)
    y_cal = np.concatenate(y_parts)
    keep = y_cal != -1
    return X_cal[keep], y_cal[keep]


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--cal-frac", type=float, default=0.1,
                    help="Calibration split -- Platt fit + threshold derivation. NOT the "
                         "early-stopping val split, and NOT test.")
    args = ap.parse_args()

    # Built as a dict whose values come directly from _load_train_val()'s
    # return tuple -- X_train/X_val are deliberately never bound to a
    # separate name of this function's own. Measured directly (see
    # models/static_lgbm.py::train_from_holder()'s docstring): a plain
    # bound name (or even an unpacked `train(*_load_train_val(...))` call)
    # keeps the array alive for train_from_holder()'s ENTIRE call, since
    # CPython's caller-side evaluation stack retains it regardless of
    # calling convention -- only popping it out of a container this frame
    # holds instead of naming it directly avoids that.
    holder = dict(zip(
        ("X_train", "y_train", "X_val", "y_val"),
        _load_train_val(args.train, args.val_frac, args.cal_frac, seed=42),
    ))
    logger.info("train=%d val=%d features=%d (cal loaded after training; test not read here "
                "-- see module docstring)", len(holder["y_train"]), len(holder["y_val"]),
                holder["X_train"].shape[1])

    model = train_from_holder(holder)
    # train_from_holder() already popped X_train/y_train/X_val/y_val out of
    # holder and deleted its own locals mid-call -- holder is empty now.
    gc.collect()

    X_cal, y_cal = _load_cal(args.train, args.val_frac, args.cal_frac, seed=42)
    logger.info("cal=%d", len(y_cal))
    model = calibrate(model, X_cal, y_cal)
    del X_cal, y_cal
    gc.collect()

    model.save(args.out)
    logger.info("Saved model to %s.lgbm / %s.meta.json", args.out, args.out)


if __name__ == "__main__":
    main()
