"""
Train Cortex-Emulation on tokenized Speakeasy-emulation API-name sequences.

Expects the train/val/test parquet files produced by
scripts/split_emulation.py (each row: api_names list[str], label int, family
str, ...). Vocabulary and pos_weight are computed from --train only. --test
is never read until the final held-out evaluation + the val+test threshold
sweep at the end of this script.

This script's job ends at "trained model + derived-threshold sweep + full
report". It does NOT export ONNX, does NOT set EMULATION_MALICIOUS_MIN in
inference/policy_engine.py (that stays None / NotImplementedError), and does
NOT write config/thresholds.yaml -- the real threshold wiring and the policy
authority decision are a dedicated follow-up once these results are reviewed,
the same sequence used for Cortex-Memory and Cortex-Network.

Usage:
    python -m scripts.train_emulation \\
        --train data/processed/emulation_train.parquet \\
        --val   data/processed/emulation_val.parquet \\
        --test  data/processed/emulation_test.parquet \\
        --vocab-out      data/models/emulation_vocab.json \\
        --checkpoint-out data/models/cortex_emulation_best.pt
"""

from __future__ import annotations

import argparse
import logging
import random
import time

import numpy as np
import pandas as pd
import torch

from models.emulation_cnn import TrainConfig
from models.emulation_artifacts import sidecar_path
from models.sequence_artifacts import scorable_rows
from models.train_emulation import (
    evaluate_emulation, predict_proba_emulation, threshold_sweep, train,
)
from tokenizer.emulation_tokenizer import EmulationTokenizer

logger = logging.getLogger("cortex.scripts.train_emulation")

# clean = real benign traces; windows_syswow64 = known-legitimate Windows
# system binaries (named by filename, not sha256 -- see download_emulation.py).
# Everything else is a malware family. Used only for the per-family report.
BENIGN_FAMILIES = {"clean", "windows_syswow64"}
PRIMARY_TARGET_FPR = 0.01  # 1% -- the recommended operating-point candidate


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info("seed=%d (random / numpy / torch / torch.cuda all seeded)", seed)


def _xy(df: pd.DataFrame, tokenizer: EmulationTokenizer) -> tuple[np.ndarray, np.ndarray, list[str]]:
    seqs = [list(s) for s in df["api_names"].tolist()]
    X, statuses = tokenizer.encode_batch(seqs)
    y = df["label"].to_numpy(dtype=np.float32)
    return X, y, statuses


def _log_status_counts(name: str, statuses: list[str]) -> None:
    from collections import Counter
    c = Counter(statuses)
    logger.info("  %s encode status: %s", name, dict(sorted(c.items())))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/processed/emulation_train.parquet")
    ap.add_argument("--val", default="data/processed/emulation_val.parquet")
    ap.add_argument("--test", default="data/processed/emulation_test.parquet")
    ap.add_argument("--vocab-out", default="data/models/emulation_vocab.json")
    ap.add_argument("--checkpoint-out", default="data/models/cortex_emulation_best.pt")
    ap.add_argument("--min-vocab-count", type=int, default=1)
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)

    train_df = pd.read_parquet(args.train)
    val_df = pd.read_parquet(args.val)
    test_df = pd.read_parquet(args.test)
    logger.info(
        "rows: train=%d (%d benign / %d malicious)  val=%d (%d / %d)  test=%d (%d / %d)",
        len(train_df), int((train_df.label == 0).sum()), int((train_df.label == 1).sum()),
        len(val_df), int((val_df.label == 0).sum()), int((val_df.label == 1).sum()),
        len(test_df), int((test_df.label == 0).sum()), int((test_df.label == 1).sum()),
    )

    # --- vocab from train only ---
    tokenizer = EmulationTokenizer.build_from_corpus(
        [list(s) for s in train_df["api_names"].tolist()], min_count=args.min_vocab_count
    )
    tokenizer.save(args.vocab_out)
    logger.info("vocab built from train only: %d tokens (incl. <PAD>/<UNK>); saved to %s",
                tokenizer.vocab_size, args.vocab_out)

    X_train, y_train, s_train = _xy(train_df, tokenizer)
    X_val, y_val, s_val = _xy(val_df, tokenizer)
    X_test, y_test, s_test = _xy(test_df, tokenizer)
    _log_status_counts("train", s_train)
    _log_status_counts("val", s_val)
    _log_status_counts("test", s_test)

    # --- class-imbalance pos_weight (NOT duplicate_count; NOT sample weighting) ---
    n_pos, n_neg = int(y_train.sum()), int((1 - y_train).sum())
    pos_weight = n_neg / max(n_pos, 1)
    logger.info(
        "pos_weight = n_benign / n_malicious = %d / %d = %.4f  "
        "(BCEWithLogitsLoss class-imbalance term only; the split's `duplicate_count` "
        "column is inert here -- not a feature, not a sample weight)",
        n_neg, n_pos, pos_weight,
    )

    logger.info(
        "calibration: RAW sigmoid output, no Platt/temperature layer -- matches "
        "CortexEmulationNet.predict_proba() and Cortex-Behavioral. The commit-25bd436 "
        "Platt bug was LightGBM-predict_proba-specific and has no analogue on the torch path."
    )

    # --- train ---
    cfg = TrainConfig(vocab_size=tokenizer.vocab_size, embed_dim=args.embed_dim, seed=args.seed)
    t0 = time.monotonic()
    model, outcome = train(
        X_train, y_train, X_val, y_val, cfg,
        checkpoint_path=args.checkpoint_out, pos_weight=pos_weight, vocab_path=args.vocab_out,
    )
    wall = time.monotonic() - t0
    logger.info("best checkpoint at %s (sidecar %s, use_padding_mask=%s)",
                args.checkpoint_out, sidecar_path(args.checkpoint_out), cfg.use_padding_mask)

    # All-padding (empty-trace) rows have no defined score: PENDING, not scored.
    for name, Xs, ys in (("val", X_val, y_val), ("test", X_test, y_test)):
        n_pend = int((~scorable_rows(Xs)).sum())
        logger.info("%s: %d all-padding (empty-trace) row(s) -> PENDING, excluded from scoring", name, n_pend)
    keep_val, keep_test = scorable_rows(X_val), scorable_rows(X_test)
    X_val, y_val = X_val[keep_val], y_val[keep_val]
    X_test, y_test = X_test[keep_test], y_test[keep_test]

    # --- training-curve summary ---
    logger.info("\n=== Training curve (per epoch) ===")
    logger.info("epoch | train_loss val_loss | train_auc val_auc |    gap")
    for r in outcome.history:
        logger.info("%5d | %10.4f %8.4f | %9.4f %7.4f | %+7.4f",
                    r.epoch, r.train_loss, r.val_loss, r.train_auc, r.val_auc, r.gap)
    logger.info(
        "stop_reason=%s  best_epoch=%d  best_val_auc=%.4f  epochs_run=%d  wall=%.1fs (%.1f min)",
        outcome.stop_reason, outcome.best_epoch, outcome.best_val_auc, len(outcome.history),
        wall, wall / 60,
    )

    # --- overfitting-check outcome ---
    logger.info("\n=== Overfitting check (train_auc - val_auc gap) ===")
    if outcome.overfit_triggered:
        logger.info(
            "TRIGGERED at epoch %d with gap=%.4f (> 0.05 for 3 consecutive epochs). "
            "Training was stopped early. RECOMMENDED next-run config (not auto-applied): "
            "1) set embed_dim=32; 2) if the gap persists, drop one conv block "
            "(128->256->128 instead of 128->256->128->128), in that order.",
            outcome.overfit_epoch, outcome.overfit_gap,
        )
    else:
        max_gap = max((r.gap for r in outcome.history), default=float("nan"))
        logger.info("NOT triggered. Max train/val AUC gap over the run: %.4f (threshold 0.05). "
                    "No capacity fallback recommended.", max_gap)

    # --- held-out test @ 0.5 (reference only, not the operating point) ---
    test_proba = predict_proba_emulation(model, X_test)
    ref = evaluate_emulation(model, X_test, y_test, threshold=0.5, proba=test_proba)
    logger.info("\n=== Held-out TEST metrics @ threshold=0.5 (reference, not the derived operating point) ===")
    for k, v in ref.to_dict().items():
        logger.info("  %s: %s", k, v)

    # --- threshold derivation on val+test COMBINED ---
    val_proba = predict_proba_emulation(model, X_val)
    y_valtest = np.concatenate([y_val, y_test])
    proba_valtest = np.concatenate([val_proba, test_proba])
    n_benign_vt = int((y_valtest == 0).sum())

    logger.info("\n=== Threshold sweep on val+test COMBINED ===")
    logger.info(
        "NOTE: this pools two collection eras -- val = Jan-2022 partition (carved from train), "
        "test = Apr-2022 partition (the dataset authors' deliberate concept-drift holdout). "
        "The pooled threshold is a compromise across both; the era-separated metrics below "
        "show how it lands on each."
    )
    logger.info("combined benign count = %d  (val=%d + test=%d)",
                n_benign_vt, int((y_val == 0).sum()), int((y_test == 0).sum()))
    logger.info("target_fpr | threshold | expected_FP (target*benign) | actual_FP | actual_fpr | detection_rate")
    rows = threshold_sweep(y_valtest, proba_valtest)
    for r in rows:
        logger.info("  %.4f   | %9.6f | %26.1f | %9d | %10.6f | %.4f",
                    r.target_fpr, r.threshold, r.expected_fp, r.actual_fp, r.actual_fpr, r.detection_rate)
    logger.info(
        "RECOMMENDATION: 1%% (target_fpr=0.01, ~%.0f expected FP on %d benign) is the primary "
        "operating-point candidate. 0.1%% (target_fpr=0.001, ~%.0f expected FP) rests on a "
        "handful of benign samples and is too thin to trust as a precise FPR claim -- the same "
        "caution applied to Cortex-Network's tightest sweep targets. Final pick is deferred to "
        "the policy-authority follow-up.",
        0.01 * n_benign_vt, n_benign_vt, 0.001 * n_benign_vt,
    )

    chosen = next(r.threshold for r in rows if abs(r.target_fpr - PRIMARY_TARGET_FPR) < 1e-9)
    logger.info("\n=== Metrics at the recommended threshold (target_fpr=1%%, thr=%.6f) ===", chosen)

    pooled = evaluate_emulation(None, None, y_valtest, threshold=chosen, proba=proba_valtest)
    val_era = evaluate_emulation(None, None, y_val, threshold=chosen, proba=val_proba)
    test_era = evaluate_emulation(None, None, y_test, threshold=chosen, proba=test_proba)
    for label, rep in (("POOLED val+test", pooled), ("VAL-era (Jan 2022)", val_era), ("TEST-era (Apr 2022)", test_era)):
        logger.info("--- %s ---", label)
        for k, v in rep.to_dict().items():
            logger.info("    %s: %s", k, v)

    # --- per-family TEST recall at the recommended threshold ---
    logger.info("\n=== Per-family TEST results @ recommended threshold ===")
    test_preds = (test_proba >= chosen).astype(np.int32)
    train_fam_counts = train_df["family"].value_counts().to_dict()
    fam_series = test_df["family"].to_numpy()
    agg_recall = test_era.recall
    for fam in sorted(set(fam_series)):
        mask = fam_series == fam
        n = int(mask.sum())
        n_train = int(train_fam_counts.get(fam, 0))
        if fam in BENIGN_FAMILIES:
            fpr = float((test_preds[mask] == 1).mean())
            logger.info("  %-18s n_test=%4d n_train=%5d  FPR=%.4f  (benign family)", fam, n, n_train, fpr)
            continue
        det = float((test_preds[mask] == 1).mean())
        note = ""
        if det < agg_recall - 0.05:
            note = "  <-- below aggregate malicious recall by >5pp;"
            note += (f" train support is only {n_train} rows -- reads as UNDER-TRAINED, not "
                     f"intrinsically hard" if n_train < 300 else
                     f" train support is {n_train} rows -- not a support problem, reads as HARDER to detect")
        logger.info("  %-18s n_test=%4d n_train=%5d  detection_rate=%.4f%s", fam, n, n_train, det, note)
    logger.info(
        "Explicit call-outs: 'rat' (n_test=%d, n_train=%d) and 'keylogger' (n_test=%d, n_train=%d) "
        "are the thinnest-support malicious families -- read their detection_rate above against "
        "their train support before calling them 'hard'.",
        int((fam_series == "rat").sum()), int(train_fam_counts.get("rat", 0)),
        int((fam_series == "keylogger").sum()), int(train_fam_counts.get("keylogger", 0)),
    )

    # --- trivial-baseline sanity checks ---
    logger.info("\n=== Trivial-baseline comparison (TEST) ===")
    maj_acc = float((y_test == 0).mean())  # predict-all-benign
    logger.info("majority-class (predict all benign): accuracy=%.4f", maj_acc)

    # exact duplicate-sequence lookup: does the test sequence appear verbatim in train?
    def _key(seq) -> tuple:
        return tuple(seq)
    train_lookup: dict[tuple, list[int]] = {}
    for seq, lab in zip(train_df["api_names"].tolist(), train_df["label"].tolist()):
        train_lookup.setdefault(_key(seq), []).append(int(lab))
    hits, lookup_preds = 0, np.zeros(len(test_df), dtype=np.int32)
    for i, seq in enumerate(test_df["api_names"].tolist()):
        labs = train_lookup.get(_key(seq))
        if labs:
            hits += 1
            lookup_preds[i] = int(round(sum(labs) / len(labs)))
        else:
            lookup_preds[i] = 0  # fall back to majority class (benign)
    lookup_acc = float((lookup_preds == y_test.astype(np.int32)).mean())
    logger.info(
        "exact-duplicate lookup vs train (%d/%d test sequences found verbatim in train; "
        "misses fall back to benign): accuracy=%.4f", hits, len(test_df), lookup_acc,
    )
    logger.info(
        "model @ recommended threshold: test accuracy=%.4f  -->  beats majority by %+.4f, "
        "beats duplicate-lookup by %+.4f", test_era.accuracy,
        test_era.accuracy - maj_acc, test_era.accuracy - lookup_acc,
    )

    logger.info("\n=== Artifacts ===")
    logger.info("  model state_dict : %s", args.checkpoint_out)
    logger.info("  vocab            : %s", args.vocab_out)
    logger.info("  (no ONNX export, no thresholds.yaml write, no policy_engine threshold set -- by design)")


if __name__ == "__main__":
    main()
