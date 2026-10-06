"""Training loop for Cortex-Behavioral (binary malicious-score head)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, precision_recall_fscore_support,
    precision_score, recall_score, roc_auc_score,
)
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset

from models.behavioral_cnn import CortexBehavioralNet, TrainConfig, build_model

logger = logging.getLogger("cortex.behavioral.train")


def make_loaders(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, batch_size: int):
    """Builds loaders directly from pre-split train/val arrays -- no internal
    random re-split here. The behavioral dataset's train/val/test split is
    decided upstream (scripts/split_behavioral.py, grouped by exact
    api_calls sequence to avoid leakage), and val must stay exactly that
    split for per-epoch monitoring to mean what Stage D's held-out test
    evaluation assumes it means."""
    def _loader(X, y, shuffle):
        ds = TensorDataset(torch.from_numpy(X).long(), torch.from_numpy(y).float())
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    return _loader(X_train, y_train, True), _loader(X_val, y_val, False)


@torch.no_grad()
def _validate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        total_loss += loss.item() * xb.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == yb).sum().item()
        total += xb.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def train(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    cfg: TrainConfig,
    checkpoint_path: Optional[str] = None,
    pos_weight: Optional[float] = None,
    vocab_path: Optional[str] = None,
) -> CortexBehavioralNet:
    """`checkpoint_path` and `vocab_path` go together: every saved checkpoint
    gets a JSON sidecar (models/behavioral_artifacts.py, docs/CODE_REVIEW.md
    F2) recording the architecture switches and the sha256 of the checkpoint
    and of the vocabulary it was trained with."""
    if checkpoint_path and not vocab_path:
        raise ValueError("vocab_path is required with checkpoint_path: the checkpoint "
                         "sidecar records the vocabulary's sha256")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, device=str(device))

    train_loader, val_loader = make_loaders(X_train, y_train, X_val, y_val, cfg.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    pw = torch.tensor(pos_weight, device=device) if pos_weight is not None else None
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            train_loss += loss.item() * xb.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == yb).sum().item()
            total += xb.size(0)
        scheduler.step()

        val_loss, val_acc = _validate(model, val_loader, loss_fn, device)

        logger.info(
            "epoch %3d/%d train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f lr=%.2e",
            epoch, cfg.epochs, train_loss / max(total, 1), correct / max(total, 1),
            val_loss, val_acc, optimizer.param_groups[0]["lr"],
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            if checkpoint_path:
                torch.save(best_state, checkpoint_path)
                _write_sidecar(checkpoint_path, vocab_path, cfg)
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                logger.info("early stopping at epoch %d", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model.to(device)


def _write_sidecar(checkpoint_path: str, vocab_path: str, cfg: TrainConfig) -> None:
    from models.behavioral_artifacts import write_behavioral_sidecar
    write_behavioral_sidecar(
        checkpoint_path, vocab_path,
        use_padding_mask=cfg.use_padding_mask, embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads, vocab_size=cfg.vocab_size,
    )


@dataclass(frozen=True)
class BehavioralMetricsReport:
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    auc_pr: float
    threshold: float
    # Per-class breakdown -- overall accuracy alone is misleading given the
    # class imbalance (train is ~85% malicious), so benign vs malicious
    # precision/recall/F1 are reported separately, not just averaged away.
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
def predict_proba_behavioral(model: nn.Module, X: np.ndarray, device: Optional[torch.device] = None,
                              batch_size: int = 512) -> np.ndarray:
    model.eval()
    device = device or next(model.parameters()).device
    out = np.empty(len(X), dtype=np.float64)
    for start in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[start:start + batch_size]).long().to(device)
        out[start:start + batch_size] = torch.sigmoid(model(xb)).cpu().numpy()
    return out


def evaluate_behavioral(
    model: nn.Module, X_test: np.ndarray, y_test: np.ndarray,
    threshold: float = 0.5, device: Optional[torch.device] = None,
) -> BehavioralMetricsReport:
    """Mirrors models/static_lgbm.py::evaluate()'s metric set, plus a
    per-class (benign vs malicious) precision/recall/F1 breakdown -- overall
    accuracy alone is misleading under class imbalance."""
    proba = predict_proba_behavioral(model, X_test, device=device)
    preds = (proba >= threshold).astype(np.int32)
    y = y_test.astype(np.int32)

    p_cls, r_cls, f_cls, support = precision_recall_fscore_support(y, preds, labels=[0, 1], zero_division=0)

    return BehavioralMetricsReport(
        accuracy=float(accuracy_score(y, preds)),
        precision=float(precision_score(y, preds, zero_division=0)),
        recall=float(recall_score(y, preds, zero_division=0)),
        f1=float(f1_score(y, preds, zero_division=0)),
        auc_roc=float(roc_auc_score(y, proba)),
        auc_pr=float(average_precision_score(y, proba)),
        threshold=threshold,
        benign_precision=float(p_cls[0]), benign_recall=float(r_cls[0]),
        benign_f1=float(f_cls[0]), benign_support=int(support[0]),
        malicious_precision=float(p_cls[1]), malicious_recall=float(r_cls[1]),
        malicious_f1=float(f_cls[1]), malicious_support=int(support[1]),
    )
