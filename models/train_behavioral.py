"""Training loop for Cortex-Behavioral (binary malicious-score head)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset

from models.behavioral_cnn import CortexBehavioralNet, TrainConfig, build_model

logger = logging.getLogger("cortex.behavioral.train")


def make_loaders(X: np.ndarray, y: np.ndarray, batch_size: int, val_frac: float = 0.15, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_val = int(len(y) * val_frac)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    def _loader(ix, shuffle):
        ds = TensorDataset(torch.from_numpy(X[ix]).long(), torch.from_numpy(y[ix]).float())
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    return _loader(train_idx, True), _loader(val_idx, False)


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
    cfg: TrainConfig,
    checkpoint_path: Optional[str] = None,
    pos_weight: Optional[float] = None,
) -> CortexBehavioralNet:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, device=str(device))

    train_loader, val_loader = make_loaders(X_train, y_train, cfg.batch_size, seed=cfg.seed)

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

        if epoch % 5 == 0 or epoch == 1:
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
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                logger.info("early stopping at epoch %d", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model.to(device)
