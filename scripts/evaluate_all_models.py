"""
Standalone evaluation of every trained Cortex model against BOTH its held-out
val and test splits, at the model's actual deployed operating threshold(s)
from config/thresholds.yaml.

Read-only: loads model files and processed parquet splits, computes metrics,
prints a report. Never retrains, never writes a model / threshold / parquet.
Re-run this after any future retrain or threshold re-derivation.

Thresholds are NOT hardcoded here. They are read from
inference.policy_engine's module-level constants, which that module loads
from config/thresholds.yaml at import time (via its own _load_thresholds()).
So this script sees exactly the values the live policy engine would use.

Models / splits covered
-----------------------
1. Cortex-Static      data/models/cortex_static.lgbm
     val  = the 10% val slice re-derived from data/processed/ember2024_train.parquet
            with numpy's default_rng(seed=42) permutation and val_frac=0.10 --
            the same construction scripts/train_static.py::_load_train_val_split
            uses (row order within the slice is irrelevant to any metric here).
     test = data/processed/ember2024_test.parquet
     Reported 3-way (ALLOW / ALERT / BLOCK) against the true label, since
     static has two operating thresholds (STATIC_ALLOW_MAX, STATIC_BLOCK_MIN),
     not one. Binary precision/recall/F1/FPR are then given at BOTH cut
     points: the ALLOW boundary (positive = "not ALLOW", i.e. escalated to
     ALERT or BLOCK -- this is the gate that matters operationally, since
     only a static ALLOW proceeds cleanly) and the BLOCK boundary
     (positive = BLOCK).

2. Cortex-Memory     data/models/cortex_memory.lgbm
     val  = data/processed/memory_val.parquet
     test = data/processed/memory_test.parquet
     Confusion matrix at MEMORY_MALICIOUS_MIN.

3. Cortex-Network    data/models/cortex_network.lgbm
     val  = data/processed/network_val.parquet
     test = data/processed/network_test.parquet
     Confusion matrix at NETWORK_MALICIOUS_MIN.

4. Cortex-Behavioral data/models/cortex_behavioral_best.pt
     val  = data/processed/behavioral_val.parquet
     test = data/processed/behavioral_test.parquet
     Confusion matrix at BEHAVIORAL_MALICIOUS_MIN, plus a breakdown by the
     raw-api_calls sequence-length bands the tokenizer / pipeline act on:
       too_short   : < MIN_SEQ_LEN (10) real calls  -- deployment floors
                     these to PENDING (never scored); the per-band matrix
                     shown is model-raw (what the CNN outputs if forced),
                     for diagnostic value only.
       short       : 10..99 real calls              -- scored on the padded
                     sequence, flagged "behavioral_short_trace" in the audit
                     trail.
       ok+truncated: >= 100 real calls              -- scored normally.
     The headline per-split total is the DEPLOYMENT total (short +
     ok+truncated only -- what the pipeline actually scores); an
     all-rows model-raw total is printed underneath it.

5. Cortex-Emulation  data/models/cortex_emulation_best.pt  (e64 checkpoint)
     val  = data/processed/emulation_val.parquet
     test = data/processed/emulation_test.parquet
     Confusion matrix at EMULATION_MALICIOUS_MIN. REPORT-ONLY: Cortex-
     Emulation is not in inference/policy_engine.py::decide() -- its verdict
     never reaches a live decision. Skipped cleanly if the checkpoint or
     splits are absent.

Usage:
    python -m scripts.evaluate_all_models                 # full run
    python -m scripts.evaluate_all_models --limit 3000    # sample each split (smoke)
    python -m scripts.evaluate_all_models --only static,memory
"""
from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

logger = logging.getLogger("cortex.evaluate_all_models")

# ---------------------------------------------------------------------------
# Artifact paths
# ---------------------------------------------------------------------------
STATIC_MODEL = "data/models/cortex_static"
MEMORY_MODEL = "data/models/cortex_memory"
NETWORK_MODEL = "data/models/cortex_network"
BEHAVIORAL_CKPT = "data/models/cortex_behavioral_best.pt"
BEHAVIORAL_VOCAB = "data/models/api_vocab.json"
EMULATION_CKPT = "data/models/cortex_emulation_best.pt"
EMULATION_VOCAB = "data/models/emulation_vocab.json"

EMBER_TRAIN = "data/processed/ember2024_train.parquet"
EMBER_TEST = "data/processed/ember2024_test.parquet"
MEMORY_VAL = "data/processed/memory_val.parquet"
MEMORY_TEST = "data/processed/memory_test.parquet"
NETWORK_VAL = "data/processed/network_val.parquet"
NETWORK_TEST = "data/processed/network_test.parquet"
BEHAVIORAL_VAL = "data/processed/behavioral_val.parquet"
BEHAVIORAL_TEST = "data/processed/behavioral_test.parquet"
EMULATION_VAL = "data/processed/emulation_val.parquet"
EMULATION_TEST = "data/processed/emulation_test.parquet"

EMBER_FEATURE_COUNT = 2568
STATIC_VAL_FRAC = 0.10
STATIC_SPLIT_SEED = 42

# Behavioral sequence-length bands (raw api_calls length, pre-truncation).
# MIN_SEQ_LEN / MAX_SEQ_LEN come from tokenizer.api_tokenizer, imported lazily.
BEH_BANDS = ("too_short", "short", "ok+truncated")


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
@dataclass
class BinaryEval:
    """A binary confusion matrix + the standard derived rates, for one
    (model, split, threshold) triple."""
    threshold: float
    n: int
    n_benign: int
    n_malicious: int
    tn: int
    fp: int
    fn: int
    tp: int
    precision: float
    recall: float
    f1: float
    fpr: float
    auc_roc: float

    def as_row(self) -> str:
        return (
            f"TN={self.tn} FP={self.fp} FN={self.fn} TP={self.tp} | "
            f"P={self.precision:.4f} R={self.recall:.4f} F1={self.f1:.4f} "
            f"FPR={self.fpr:.4f} AUC={_fmt_auc(self.auc_roc)} | "
            f"n={self.n} ({self.n_benign} benign / {self.n_malicious} malicious)"
        )


def _fmt_auc(a: float) -> str:
    return "n/a" if a != a else f"{a:.6f}"  # NaN check


def _auc(y: np.ndarray, proba: np.ndarray) -> float:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, proba))


def binary_eval(y_true: np.ndarray, proba: np.ndarray, threshold: float,
                *, positive_if_ge: bool = True) -> BinaryEval:
    """Confusion matrix + rates for a single score threshold.

    positive_if_ge=True  -> predict positive when proba >= threshold  (the
                            normal "MALICIOUS at or above" rule shared by
                            behavioral / memory / network / emulation and by
                            static's BLOCK boundary).
    positive_if_ge=False  is not used directly; static's ALLOW boundary is
                            handled by passing threshold=STATIC_ALLOW_MAX with
                            positive_if_ge=True (positive == "not ALLOW").
    """
    y = np.asarray(y_true).astype(np.int32)
    pred_pos = (proba >= threshold) if positive_if_ge else (proba < threshold)
    pred_pos = pred_pos.astype(np.int32)

    benign = y == 0
    malicious = y == 1
    tp = int((pred_pos[malicious] == 1).sum())
    fn = int((pred_pos[malicious] == 0).sum())
    fp = int((pred_pos[benign] == 1).sum())
    tn = int((pred_pos[benign] == 0).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return BinaryEval(
        threshold=float(threshold), n=int(len(y)),
        n_benign=int(benign.sum()), n_malicious=int(malicious.sum()),
        tn=tn, fp=fp, fn=fn, tp=tp,
        precision=precision, recall=recall, f1=f1, fpr=fpr,
        auc_roc=_auc(y, proba),
    )


def print_binary(title: str, ev: BinaryEval) -> None:
    logger.info("  %-28s %s", title, ev.as_row())


# ---------------------------------------------------------------------------
# 1. Cortex-Static
# ---------------------------------------------------------------------------
def _ember_feature_cols(names) -> list[str]:
    cols = [c for c in names if c.startswith("feature_") or c.isdigit()]
    if len(cols) != EMBER_FEATURE_COUNT:
        raise ValueError(f"expected {EMBER_FEATURE_COUNT} ember feature cols, found {len(cols)}")
    return cols


def _load_ember_val(path: str, limit: Optional[int]) -> tuple[np.ndarray, np.ndarray]:
    """Re-derive the 10% val slice used in training: default_rng(42) permutation
    over the full train-parquet row count, first val_frac rows flagged val.
    Streams row-group by row-group and keeps only the val rows (the other ~90%
    is never materialised). Drops EMBER's -1 'unlabeled' rows, same as training.
    """
    pf = pq.ParquetFile(path)
    n_rows = pf.metadata.num_rows
    feat = _ember_feature_cols(pf.schema_arrow.names)
    label_col = "label" if "label" in pf.schema_arrow.names else "y"

    rng = np.random.default_rng(STATIC_SPLIT_SEED)
    perm = rng.permutation(n_rows)
    n_val = int(n_rows * STATIC_VAL_FRAC)
    is_val = np.zeros(n_rows, dtype=bool)
    is_val[perm[:n_val]] = True

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    kept = 0
    offset = 0
    for rg in range(pf.metadata.num_row_groups):
        batch = pf.read_row_group(rg, columns=feat + [label_col]).to_pandas()
        n = len(batch)
        mask = is_val[offset:offset + n]
        offset += n
        if mask.any():
            X_parts.append(batch[feat].to_numpy(dtype=np.float32)[mask])
            y_parts.append(batch[label_col].to_numpy(dtype=np.int32)[mask])
            kept += int(mask.sum())
        del batch
        if limit and kept >= limit:
            break

    X = np.concatenate(X_parts)
    y = np.concatenate(y_parts)
    keep = y != -1
    X, y = X[keep], y[keep]
    if limit:
        X, y = X[:limit], y[:limit]
    return X, y


def _load_ember_test(path: str, limit: Optional[int]) -> tuple[np.ndarray, np.ndarray]:
    pf = pq.ParquetFile(path)
    feat = _ember_feature_cols(pf.schema_arrow.names)
    label_col = "label" if "label" in pf.schema_arrow.names else "y"
    X_parts, y_parts, got = [], [], 0
    for rg in range(pf.metadata.num_row_groups):
        batch = pf.read_row_group(rg, columns=feat + [label_col]).to_pandas()
        X_parts.append(batch[feat].to_numpy(dtype=np.float32))
        y_parts.append(batch[label_col].to_numpy(dtype=np.int32))
        got += len(batch)
        del batch
        if limit and got >= limit:
            break
    X = np.concatenate(X_parts)
    y = np.concatenate(y_parts)
    keep = y != -1
    X, y = X[keep], y[keep]
    if limit:
        X, y = X[:limit], y[:limit]
    return X, y


def _static_three_way(y: np.ndarray, proba: np.ndarray, allow_max: float, block_min: float) -> dict:
    """3-way ALLOW/ALERT/BLOCK counts vs true label."""
    y = np.asarray(y).astype(np.int32)
    allow = proba < allow_max
    block = proba >= block_min
    alert = (~allow) & (~block)
    out = {}
    for name, m in (("benign", y == 0), ("malicious", y == 1)):
        out[name] = {
            "ALLOW": int(allow[m].sum()),
            "ALERT": int(alert[m].sum()),
            "BLOCK": int(block[m].sum()),
            "n": int(m.sum()),
        }
    return out


def evaluate_static(limit: Optional[int] = None) -> dict:
    from inference import policy_engine as pe
    from models.static_lgbm import LGBMModel

    allow_max = pe.STATIC_ALLOW_MAX
    block_min = pe.STATIC_BLOCK_MIN
    logger.info("=" * 78)
    logger.info("1. CORTEX-STATIC  (%s.lgbm)", STATIC_MODEL)
    logger.info("   STATIC_ALLOW_MAX = %.16f", allow_max)
    logger.info("   STATIC_BLOCK_MIN = %.16f", block_min)

    model = LGBMModel.load(STATIC_MODEL)
    results = {"thresholds": {"STATIC_ALLOW_MAX": allow_max, "STATIC_BLOCK_MIN": block_min}, "splits": {}}

    for split, loader in (("val", _load_ember_val), ("test", _load_ember_test)):
        path = EMBER_TRAIN if split == "val" else EMBER_TEST
        logger.info("-" * 78)
        logger.info(" split=%s  (source: %s)", split, path)
        X, y = loader(path, limit)
        proba = model.predict_proba(X)

        tw = _static_three_way(y, proba, allow_max, block_min)
        logger.info("   3-way confusion (rows = true label, cols = static verdict):")
        logger.info("     %-12s %8s %8s %8s   %8s", "true\\pred", "ALLOW", "ALERT", "BLOCK", "n")
        for name in ("benign", "malicious"):
            r = tw[name]
            logger.info("     %-12s %8d %8d %8d   %8d", name, r["ALLOW"], r["ALERT"], r["BLOCK"], r["n"])

        # Binary metrics at each cut point.
        # ALLOW boundary: positive == "escalated" (ALERT or BLOCK) == score >= STATIC_ALLOW_MAX
        allow_ev = binary_eval(y, proba, allow_max)
        # BLOCK boundary: positive == BLOCK == score >= STATIC_BLOCK_MIN
        block_ev = binary_eval(y, proba, block_min)
        logger.info("   binary metrics:")
        print_binary("@ ALLOW boundary (not-ALLOW=+)", allow_ev)
        print_binary("@ BLOCK boundary (BLOCK=+)", block_ev)
        logger.info("   AUC-ROC (threshold-independent): %s", _fmt_auc(allow_ev.auc_roc))

        results["splits"][split] = {
            "three_way": tw,
            "allow_boundary": allow_ev,
            "block_boundary": block_ev,
        }
    return results


# ---------------------------------------------------------------------------
# 2. Cortex-Memory
# ---------------------------------------------------------------------------
def _load_memory_xy(path: str, limit: Optional[int]) -> tuple[np.ndarray, np.ndarray]:
    from features.memory_features import add_derived_features, feature_matrix_columns

    df = pd.read_parquet(path)
    if limit:
        df = df.iloc[:limit]
    df = add_derived_features(df)
    cols = feature_matrix_columns(df)
    X = df[cols].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int32)
    return X, y


def evaluate_memory(limit: Optional[int] = None) -> dict:
    from inference import policy_engine as pe
    from models.memory_lgbm import MemoryLGBMModel

    thr = pe.MEMORY_MALICIOUS_MIN
    logger.info("=" * 78)
    logger.info("2. CORTEX-MEMORY  (%s.lgbm)", MEMORY_MODEL)
    logger.info("   MEMORY_MALICIOUS_MIN = %.16f  (verdict capped at ALERT in decide())", thr)

    model = MemoryLGBMModel.load(MEMORY_MODEL)
    out = {"threshold": thr, "splits": {}}
    for split, path in (("val", MEMORY_VAL), ("test", MEMORY_TEST)):
        X, y = _load_memory_xy(path, limit)
        proba = model.predict_proba(X)
        ev = binary_eval(y, proba, thr)
        logger.info("-" * 78)
        logger.info(" split=%s  (%s)", split, path)
        print_binary("@ MEMORY_MALICIOUS_MIN", ev)
        out["splits"][split] = ev
    return out


# ---------------------------------------------------------------------------
# 3. Cortex-Network
# ---------------------------------------------------------------------------
def _load_network_xy(path: str, limit: Optional[int]) -> tuple[np.ndarray, np.ndarray]:
    from data.download_network import FEATURE_COLUMNS

    df = pd.read_parquet(path)
    if limit:
        df = df.iloc[:limit]
    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int32)
    return X, y


def evaluate_network(limit: Optional[int] = None) -> dict:
    from inference import policy_engine as pe
    from models.network_lgbm import NetworkLGBMModel

    thr = pe.NETWORK_MALICIOUS_MIN
    logger.info("=" * 78)
    logger.info("3. CORTEX-NETWORK  (%s.lgbm)", NETWORK_MODEL)
    logger.info("   NETWORK_MALICIOUS_MIN = %.16f  (verdict capped at ALERT in decide())", thr)

    model = NetworkLGBMModel.load(NETWORK_MODEL)
    out = {"threshold": thr, "splits": {}}
    for split, path in (("val", NETWORK_VAL), ("test", NETWORK_TEST)):
        X, y = _load_network_xy(path, limit)
        proba = model.predict_proba(X)
        ev = binary_eval(y, proba, thr)
        logger.info("-" * 78)
        logger.info(" split=%s  (%s)", split, path)
        print_binary("@ NETWORK_MALICIOUS_MIN", ev)
        out["splits"][split] = ev
    return out


# ---------------------------------------------------------------------------
# 4. Cortex-Behavioral
# ---------------------------------------------------------------------------
def _behavioral_bands(lengths: np.ndarray, min_seq: int, max_seq: int) -> dict[str, np.ndarray]:
    lengths = np.asarray(lengths)
    return {
        "too_short": lengths < min_seq,
        "short": (lengths >= min_seq) & (lengths < max_seq),
        "ok+truncated": lengths >= max_seq,
    }


def evaluate_behavioral(limit: Optional[int] = None) -> dict:
    import torch

    from inference import policy_engine as pe
    from models.behavioral_cnn import CortexBehavioralNet, SEQUENCE_LENGTH
    from models.train_behavioral import predict_proba_behavioral
    from tokenizer.api_tokenizer import MAX_SEQ_LEN, MIN_SEQ_LEN, ApiTokenizer

    thr = pe.BEHAVIORAL_MALICIOUS_MIN
    logger.info("=" * 78)
    logger.info("4. CORTEX-BEHAVIORAL  (%s)", BEHAVIORAL_CKPT)
    logger.info("   BEHAVIORAL_MALICIOUS_MIN = %.16f", thr)
    logger.info("   sequence-length bands (raw api_calls length): too_short <%d | short %d-%d | ok+truncated >=%d",
                MIN_SEQ_LEN, MIN_SEQ_LEN, MAX_SEQ_LEN - 1, MAX_SEQ_LEN)

    tokenizer = ApiTokenizer.load(BEHAVIORAL_VOCAB)
    # Deployed checkpoint is embed_dim=128 (scripts/train_behavioral.py default).
    model = CortexBehavioralNet(vocab_size=tokenizer.vocab_size,
                                sequence_length=SEQUENCE_LENGTH, embed_dim=128)
    model.load_state_dict(torch.load(BEHAVIORAL_CKPT, map_location="cpu"))
    model.eval()

    out = {"threshold": thr, "splits": {}}
    for split, path in (("val", BEHAVIORAL_VAL), ("test", BEHAVIORAL_TEST)):
        df = pd.read_parquet(path)
        if limit:
            df = df.iloc[:limit]
        seqs = [list(s) for s in df["api_calls"].tolist()]
        lengths = np.array([len(s) for s in seqs])
        X, _statuses = tokenizer.encode_batch(seqs)
        y = df["label"].to_numpy(dtype=np.int32)
        proba = predict_proba_behavioral(model, X)

        bands = _behavioral_bands(lengths, MIN_SEQ_LEN, MAX_SEQ_LEN)
        logger.info("-" * 78)
        logger.info(" split=%s  (%s)   n=%d", split, path, len(y))
        band_evs = {}
        for band in BEH_BANDS:
            m = bands[band]
            n = int(m.sum())
            if n == 0:
                logger.info("   band=%-13s (n=0, skipped)", band)
                band_evs[band] = None
                continue
            ev = binary_eval(y[m], proba[m], thr)
            tag = "  [model-raw; deployment floors this band to PENDING]" if band == "too_short" else ""
            print_binary(f"band={band}{tag}", ev)
            band_evs[band] = ev

        # Deployment total = short + ok+truncated (what the pipeline scores).
        dep_mask = bands["short"] | bands["ok+truncated"]
        dep_ev = binary_eval(y[dep_mask], proba[dep_mask], thr)
        raw_ev = binary_eval(y, proba, thr)
        logger.info("   ---")
        print_binary("TOTAL (deployment: short+ok)", dep_ev)
        print_binary("TOTAL (all rows, model-raw)", raw_ev)

        out["splits"][split] = {
            "bands": band_evs,
            "deployment_total": dep_ev,
            "model_raw_total": raw_ev,
        }
    return out


# ---------------------------------------------------------------------------
# 5. Cortex-Emulation  (report-only; not in decide())
# ---------------------------------------------------------------------------
def _emulation_artifacts_present() -> bool:
    return all(os.path.exists(p) for p in
               (EMULATION_CKPT, EMULATION_VOCAB, EMULATION_VAL, EMULATION_TEST))


def evaluate_emulation(limit: Optional[int] = None) -> Optional[dict]:
    if not _emulation_artifacts_present():
        logger.info("=" * 78)
        logger.info("5. CORTEX-EMULATION  -- SKIPPED (checkpoint or splits absent)")
        return None

    import torch

    from inference import policy_engine as pe
    from models.emulation_cnn import CortexEmulationNet, SEQUENCE_LENGTH
    from models.train_emulation import predict_proba_emulation
    from tokenizer.emulation_tokenizer import EmulationTokenizer

    thr = pe.EMULATION_MALICIOUS_MIN
    logger.info("=" * 78)
    logger.info("5. CORTEX-EMULATION  (%s)   *** REPORT-ONLY / NOT IN ANY LIVE DECISION PATH ***", EMULATION_CKPT)
    logger.info("   EMULATION_MALICIOUS_MIN = %.16f  (telemetry EmulationVerdict only; never reaches decide())", thr)

    tokenizer = EmulationTokenizer.load(EMULATION_VOCAB)
    # e64 checkpoint (config/thresholds.yaml references cortex_emulation_best.pt).
    model = CortexEmulationNet(vocab_size=tokenizer.vocab_size,
                               sequence_length=SEQUENCE_LENGTH, embed_dim=64)
    model.load_state_dict(torch.load(EMULATION_CKPT, map_location="cpu"))
    model.eval()

    out = {"threshold": thr, "splits": {}}
    for split, path in (("val", EMULATION_VAL), ("test", EMULATION_TEST)):
        df = pd.read_parquet(path)
        if limit:
            df = df.iloc[:limit]
        seqs = [list(s) for s in df["api_names"].tolist()]
        X, _statuses = tokenizer.encode_batch(seqs)
        y = df["label"].to_numpy(dtype=np.int32)
        proba = predict_proba_emulation(model, X)
        ev = binary_eval(y, proba, thr)
        logger.info("-" * 78)
        logger.info(" split=%s  (%s)", split, path)
        print_binary("@ EMULATION_MALICIOUS_MIN", ev)
        out["splits"][split] = ev
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
_EVALUATORS = {
    "static": evaluate_static,
    "memory": evaluate_memory,
    "network": evaluate_network,
    "behavioral": evaluate_behavioral,
    "emulation": evaluate_emulation,
}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap rows per split (sampling; for a fast smoke run). Default: full splits.")
    ap.add_argument("--only", default="",
                    help="comma list of {static,memory,network,behavioral,emulation} to run (default: all)")
    args = ap.parse_args()

    which = [s.strip() for s in args.only.split(",") if s.strip()] or list(_EVALUATORS)
    unknown = [w for w in which if w not in _EVALUATORS]
    if unknown:
        ap.error(f"unknown model(s): {unknown}; valid: {list(_EVALUATORS)}")

    if args.limit:
        logger.info(">>> --limit=%d : sampling the first %d rows of each split (NOT a full evaluation) <<<",
                    args.limit, args.limit)

    for name in which:
        _EVALUATORS[name](args.limit)

    logger.info("=" * 78)
    logger.info("done. read-only run -- no model / threshold / parquet was written.")


if __name__ == "__main__":
    main()
