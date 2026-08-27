"""
Training loop, evaluation, and threshold-sweep helpers for Cortex-Emulation
(binary malicious-score head over Speakeasy-emulation API-name sequences).

Structurally mirrors models/train_behavioral.py -- same make_loaders /
train / evaluate / predict_proba shape -- with three deliberate differences,
each driven by this signal's data rather than copied by default:

1. Early stopping is on val ROC-AUC (maximised), NOT val loss. This is a
   deliberate deviation from train_behavioral.py, which stops on val loss and
   never computes AUC during training. Only the patience counter + best-state
   clone/restore mechanism is reused. Rationale: the operating point for this
   signal is chosen later from a val+test FPR sweep, so ranking quality (AUC)
   is the thing worth preserving across epochs, not calibrated loss at a fixed
   0.5 cut. This also matches malware-ml's own behavioral model, which
   selected on best validation ROC-AUC.

2. An explicit train/val AUC-gap overfitting monitor. If (train_auc - val_auc)
   exceeds 0.05 for 3 consecutive epochs, training stops early and the caller
   is handed the triggering epoch + gap so it can *recommend* (not auto-apply)
   the pre-committed capacity fallback documented in models/emulation_cnn.py
   (embed_dim -> 32, then drop a conv block, in that order).

3. find_threshold_for_fpr() is duplicated here, self-contained, rather than
   imported from a LightGBM-specific module (models/network_lgbm.py etc.) --
   matching the project convention of keeping per-model modules independent
   (see tokenizer/emulation_tokenizer.py's and models/memory_lgbm.py's own
   notes on the same choice). The implementation is byte-for-byte the same
   semantics as models/static_lgbm.py / memory_lgbm.py / network_lgbm.py so
   the derived numbers stay methodologically comparable across all signals.

Calibration: this model's score is a RAW sigmoid of the logit
(CortexEmulationNet.predict_proba), with no Platt/temperature layer -- exactly
like Cortex-Behavioral. The Platt-calibration bug fixed during Cortex-Static's
ONNX export (commit 25bd436: raw booster vs LGBMModel.predict_proba differing
by up to 0.213) was specific to LightGBM's predict_proba applying a fitted
calibrator on top of the booster. There is no analogous hidden calibrator on
the torch path, so there is nothing to reconcile here -- but the check was
made, not skipped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, precision_recall_fscore_support,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset

from models.emulation_cnn import CortexEmulationNet, TrainConfig, build_model

logger = logging.getLogger("cortex.emulation.train")

# Gap between per-epoch train AUC and val AUC above which the model is judged
# to be overfitting; must persist this many consecutive epochs to trigger an
# early stop. Kept as module constants so the values reported in the run log
# are traceable to a single definition.
OVERFIT_GAP_THRESHOLD = 0.05
OVERFIT_GAP_PATIENCE = 3


# ---------------------------------------------------------------------------
# Threshold helper -- self-contained duplicate (see module docstring, point 3)
# ---------------------------------------------------------------------------
def find_threshold_for_fpr(y_true: NDArray, proba: NDArray, target_fpr: float) -> float:
    """Pick the lowest score threshold that keeps benign FPR <= target_fpr.
    Identical semantics to models/static_lgbm.py / memory_lgbm.py /
    network_lgbm.py -- duplicated here to keep this module self-contained."""
    fpr_arr, tpr_arr, thresh_arr = roc_curve(y_true, proba)
    idx = np.searchsorted(fpr_arr, target_fpr, side="right") - 1
    idx = max(idx, 0)
    return float(thresh_arr[idx])


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def make_loaders(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    batch_size: int,
):
    """Loaders straight from the pre-split arrays -- no internal re-split.
    The train/val/test partition is fixed upstream (scripts/split_emulation.py:
    test = the authors' Apr-2022 partition, val carved from the Jan-2022 train
    partition, exact-duplicate sequences already collapsed), and val has to
    stay exactly that split for per-epoch AUC to mean what the held-out test
    evaluation assumes.

    Returns (train_loader shuffled, train_eval_loader unshuffled, val_loader).
    The unshuffled train_eval_loader exists so the overfitting monitor can
    measure train AUC from a clean eval-mode pass (dropout off, BN in eval)
    rather than from noisy in-training-pass logits."""
    def _loader(X, y, shuffle):
        ds = TensorDataset(torch.from_numpy(X).long(), torch.from_numpy(y).float())
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    return (
        _loader(X_train, y_train, True),
        _loader(X_train, y_train, False),
        _loader(X_val, y_val, False),
    )


@torch.no_grad()
def _evaluate_split(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device):
    """One eval-mode pass. Returns (mean_loss, accuracy@0.5, roc_auc)."""
    model.eval()
    total_loss, total = 0.0, 0
    all_proba, all_y = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        total_loss += loss.item() * xb.size(0)
        total += xb.size(0)
        all_proba.append(torch.sigmoid(logits).cpu().numpy())
        all_y.append(yb.cpu().numpy())
    proba = np.concatenate(all_proba)
    y = np.concatenate(all_y)
    acc = float(((proba >= 0.5).astype(np.int32) == y.astype(np.int32)).mean())
    # roc_auc_score needs both classes present; val/train always have both here.
    auc = float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else float("nan")
    return total_loss / max(total, 1), acc, auc


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    train_acc: float
    train_auc: float
    val_loss: float
    val_acc: float
    val_auc: float
    lr: float
    gap: float  # train_auc - val_auc


@dataclass
class TrainOutcome:
    history: list[EpochRecord]
    stop_reason: str            # "early_stop_val_auc" | "overfit_gap" | "max_epochs"
    best_epoch: int
    best_val_auc: float
    overfit_triggered: bool
    overfit_epoch: Optional[int]
    overfit_gap: Optional[float]


def train(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    cfg: TrainConfig,
    checkpoint_path: Optional[str] = None,
    pos_weight: Optional[float] = None,
) -> tuple[CortexEmulationNet, TrainOutcome]:
    """Train on train only; early-stop on val ROC-AUC; restore best weights.

    pos_weight is the BCEWithLogitsLoss positive-class weight for class
    imbalance (n_benign / n_malicious ~= 1.996 for this split). It is entirely
    separate from the dataset's `duplicate_count` column and the
    equal-sample-weighting decision made in scripts/split_emulation.py --
    duplicate_count plays no role in training (not a feature, not a weight).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, device=str(device))

    train_loader, train_eval_loader, val_loader = make_loaders(
        X_train, y_train, X_val, y_val, cfg.batch_size
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    pw = torch.tensor(pos_weight, device=device) if pos_weight is not None else None
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    history: list[EpochRecord] = []
    best_val_auc = float("-inf")
    best_epoch = 0
    best_state = None
    patience_counter = 0
    consec_overfit = 0
    stop_reason = "max_epochs"
    overfit_triggered = False
    overfit_epoch: Optional[int] = None
    overfit_gap: Optional[float] = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        run_loss, seen = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            run_loss += loss.item() * xb.size(0)
            seen += xb.size(0)
        scheduler.step()

        # Clean eval-mode passes for the metrics that drive stopping decisions.
        train_loss, train_acc, train_auc = _evaluate_split(model, train_eval_loader, loss_fn, device)
        val_loss, val_acc, val_auc = _evaluate_split(model, val_loader, loss_fn, device)
        gap = train_auc - val_auc
        lr = optimizer.param_groups[0]["lr"]

        history.append(EpochRecord(
            epoch=epoch, train_loss=train_loss, train_acc=train_acc, train_auc=train_auc,
            val_loss=val_loss, val_acc=val_acc, val_auc=val_auc, lr=lr, gap=gap,
        ))
        logger.info(
            "epoch %3d/%d  train_loss=%.4f train_auc=%.4f  val_loss=%.4f val_auc=%.4f  "
            "gap=%+.4f  lr=%.2e",
            epoch, cfg.epochs, train_loss, train_auc, val_loss, val_auc, gap, lr,
        )

        # --- best-state tracking on val AUC (maximise) ---
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            if checkpoint_path:
                torch.save(best_state, checkpoint_path)
        else:
            patience_counter += 1

        # --- overfitting monitor (independent of the patience counter) ---
        if gap > OVERFIT_GAP_THRESHOLD:
            consec_overfit += 1
        else:
            consec_overfit = 0
        if consec_overfit >= OVERFIT_GAP_PATIENCE:
            overfit_triggered = True
            overfit_epoch = epoch
            overfit_gap = gap
            stop_reason = "overfit_gap"
            logger.warning(
                "OVERFITTING STOP: train/val AUC gap > %.2f for %d consecutive epochs "
                "(epoch %d, gap=%.4f). Recommended next-run config: embed_dim -> 32, "
                "and if that is not enough, drop one conv block (see models/emulation_cnn.py).",
                OVERFIT_GAP_THRESHOLD, OVERFIT_GAP_PATIENCE, epoch, gap,
            )
            break

        if patience_counter >= cfg.patience:
            stop_reason = "early_stop_val_auc"
            logger.info("early stopping on val AUC at epoch %d (best epoch %d, val_auc=%.4f)",
                        epoch, best_epoch, best_val_auc)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    outcome = TrainOutcome(
        history=history, stop_reason=stop_reason, best_epoch=best_epoch,
        best_val_auc=best_val_auc, overfit_triggered=overfit_triggered,
        overfit_epoch=overfit_epoch, overfit_gap=overfit_gap,
    )
    return model.to(device), outcome


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EmulationMetricsReport:
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    auc_pr: float
    threshold: float
    benign_precision: float
    benign_recall: float
    benign_f1: float
    benign_support: int
    malicious_precision: float
    malicious_recall: float
    malicious_f1: float
    malicious_support: int

    def to_dict(self) -> dict[str, float]:
        return {k: v for k, v in self.__dict__.items()}


@torch.no_grad()
def predict_proba_emulation(
    model: nn.Module, X: np.ndarray, device: Optional[torch.device] = None, batch_size: int = 512,
) -> np.ndarray:
    model.eval()
    device = device or next(model.parameters()).device
    out = np.empty(len(X), dtype=np.float64)
    for start in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[start:start + batch_size]).long().to(device)
        out[start:start + batch_size] = torch.sigmoid(model(xb)).cpu().numpy()
    return out


def evaluate_emulation(
    model: nn.Module, X: np.ndarray, y: np.ndarray,
    threshold: float = 0.5, device: Optional[torch.device] = None,
    proba: Optional[np.ndarray] = None,
) -> EmulationMetricsReport:
    """Mirrors models/train_behavioral.py::evaluate_behavioral -- same metric
    set plus a benign-vs-malicious per-class breakdown, since overall accuracy
    alone is misleading when the split is class-imbalanced (test is 77.6%
    benign). Pass `proba` to score a pre-computed probability vector (used for
    the val-era / test-era slices of the pooled val+test sweep)."""
    if proba is None:
        proba = predict_proba_emulation(model, X, device=device)
    preds = (proba >= threshold).astype(np.int32)
    y = y.astype(np.int32)

    p_cls, r_cls, f_cls, support = precision_recall_fscore_support(y, preds, labels=[0, 1], zero_division=0)
    return EmulationMetricsReport(
        accuracy=float(accuracy_score(y, preds)),
        precision=float(precision_score(y, preds, zero_division=0)),
        recall=float(recall_score(y, preds, zero_division=0)),
        f1=float(f1_score(y, preds, zero_division=0)),
        auc_roc=float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else float("nan"),
        auc_pr=float(average_precision_score(y, proba)) if len(np.unique(y)) > 1 else float("nan"),
        threshold=float(threshold),
        benign_precision=float(p_cls[0]), benign_recall=float(r_cls[0]),
        benign_f1=float(f_cls[0]), benign_support=int(support[0]),
        malicious_precision=float(p_cls[1]), malicious_recall=float(r_cls[1]),
        malicious_f1=float(f_cls[1]), malicious_support=int(support[1]),
    )


@dataclass
class SweepRow:
    target_fpr: float
    threshold: float
    n_benign: int
    expected_fp: float     # target_fpr * n_benign -- what the target implies
    actual_fp: int         # benign samples at/above the threshold on this data
    actual_fpr: float
    detection_rate: float


def threshold_sweep(
    y: np.ndarray, proba: np.ndarray, targets: tuple[float, ...] = (0.001, 0.005, 0.01, 0.02, 0.05),
) -> list[SweepRow]:
    """Run find_threshold_for_fpr() at each target on a single (pooled) score
    vector and report both the target-implied expected FP count and the FP
    count actually observed at the resulting threshold."""
    y = y.astype(np.int32)
    benign_mask = y == 0
    malicious_mask = ~benign_mask
    n_benign = int(benign_mask.sum())
    rows: list[SweepRow] = []
    for t_fpr in targets:
        thr = find_threshold_for_fpr(y, proba, t_fpr)
        preds = (proba >= thr).astype(np.int32)
        n_fp = int((preds[benign_mask] == 1).sum())
        rows.append(SweepRow(
            target_fpr=t_fpr, threshold=thr, n_benign=n_benign,
            expected_fp=t_fpr * n_benign, actual_fp=n_fp,
            actual_fpr=n_fp / max(n_benign, 1),
            detection_rate=float((preds[malicious_mask] == 1).mean()),
        ))
    return rows
