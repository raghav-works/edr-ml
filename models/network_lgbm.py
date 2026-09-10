"""
Cortex-Network: LightGBM trainer for the 78-dim CSE-CIC-IDS2018 network-flow
feature vector (see data/download_network.py for the feature list and
dataset details -- Flow ID/Src IP/Src Port/Dst IP and Timestamp are
deliberately excluded from FEATURE_COLUMNS there, not just unused here).
"""

from __future__ import annotations

import hashlib
import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import lightgbm as lgb
import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, auc, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
)

logger = logging.getLogger("cortex.network.train")

NETWORK_FEATURE_COUNT = 78  # data.download_network.FEATURE_COLUMNS
DEFAULT_SEED = 42

# Hyperparameters -- starting point mirrors models/memory_lgbm.py's (also a
# LightGBM binary classifier, same order of magnitude in feature count: 78
# vs memory's 62), validated empirically against the real ~1.53M-row train
# split before trusting it (see scripts/train_network.py's reported
# wall-clock time and peak memory) rather than assumed safe by analogy
# alone. One deliberate deviation from memory's config: is_unbalance=True.
# Memory's train split was ~50/50 (23,438 benign / 23,298 malware) so
# is_unbalance was correctly omitted there; network's train split is
# 1,286,894 benign / 239,863 malicious (~84.3% / 15.7%), a real ~5.4:1
# imbalance much closer to static's EMBER2024 situation than to memory's --
# copying memory's omission here would be blindly reusing a choice that
# was right for a different class balance, not a validated default.
DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "max_depth": -1,
    "learning_rate": 0.05,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "seed": DEFAULT_SEED,
    "verbose": -1,
    "is_unbalance": True,
}
DEFAULT_N_ESTIMATORS = 1500
DEFAULT_EARLY_STOPPING = 50


class PlattCalibrator:
    """Platt scaling: logistic regression on raw booster margins.

    Identical implementation to models/static_lgbm.py's and
    models/memory_lgbm.py's PlattCalibrator, duplicated rather than
    imported -- keeps this module self-contained, same rationale as
    memory_lgbm.py's copy of this class."""

    def __init__(self) -> None:
        self._lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=5000)

    def fit(self, scores: NDArray, labels: NDArray) -> "PlattCalibrator":
        self._lr.fit(scores.reshape(-1, 1), labels)
        return self

    def predict_proba(self, scores: NDArray) -> NDArray:
        return self._lr.predict_proba(scores.reshape(-1, 1))[:, 1]


@dataclass(frozen=True)
class MetricsReport:
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    auc_pr: float
    fpr_at_threshold: float
    detection_rate: float
    threshold: float

    def to_dict(self) -> dict[str, float]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class NetworkLGBMModel:
    booster: lgb.Booster
    calibrator: Optional[PlattCalibrator]
    feature_count: int
    num_iterations: int
    model_hash: str

    def predict_proba(self, X: NDArray[np.float32]) -> NDArray[np.float64]:
        if X.shape[1] != self.feature_count:
            raise ValueError(f"Expected {self.feature_count} features, got {X.shape[1]}")
        if self.calibrator is not None:
            # Platt calibrator is fit on raw booster margins (logits), not on
            # the booster's sigmoid probability -- inference must feed it the
            # same raw_score=True margins used at fit time (see train()).
            margins = self.booster.predict(X, raw_score=True, num_iteration=self.num_iterations)
            return self.calibrator.predict_proba(margins)
        return self.booster.predict(X, num_iteration=self.num_iterations)

    def predict(self, X: NDArray[np.float32], threshold: float = 0.5) -> NDArray[np.int32]:
        return (self.predict_proba(X) >= threshold).astype(np.int32)

    def save(self, path: Union[str, Path]) -> None:
        base = Path(path)
        base.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(base.with_suffix(".lgbm")), num_iteration=self.num_iterations)
        meta = {"calibrator": self.calibrator, "feature_count": self.feature_count,
                "num_iterations": self.num_iterations, "model_hash": self.model_hash}
        with open(base.with_suffix(".meta"), "wb") as fh:
            pickle.dump(meta, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "NetworkLGBMModel":
        base = Path(path)
        booster = lgb.Booster(model_file=str(base.with_suffix(".lgbm")))
        with open(base.with_suffix(".meta"), "rb") as fh:
            meta = pickle.load(fh)
        return cls(booster=booster, calibrator=meta["calibrator"], feature_count=meta["feature_count"],
                    num_iterations=meta["num_iterations"], model_hash=meta["model_hash"])


def _model_hash(booster: lgb.Booster) -> str:
    return hashlib.sha256(booster.model_to_string().encode("utf-8")).hexdigest()[:16]


def train(
    X_train: NDArray[np.float32], y_train: NDArray[np.int32],
    X_val: NDArray[np.float32], y_val: NDArray[np.int32],
    X_cal: NDArray[np.float32], y_cal: NDArray[np.int32],
    *, params: Optional[dict[str, Any]] = None,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    early_stopping_rounds: int = DEFAULT_EARLY_STOPPING,
    calibrate: bool = True,
) -> NetworkLGBMModel:
    hp = {**DEFAULT_PARAMS, **(params or {})}
    feature_count = X_train.shape[1]
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

    t0 = time.monotonic()
    booster = lgb.train(
        hp, train_set, num_boost_round=n_estimators,
        valid_sets=[train_set, val_set], valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(50)],
    )
    logger.info("Training done in %.1fs, best_iteration=%d", time.monotonic() - t0, booster.best_iteration)

    calibrator = None
    if calibrate:
        # Textbook Platt scaling fits the logistic regression on the model's
        # raw margins, NOT on probabilities -- fitting on the sigmoid output
        # of a near-separable booster collapses the calibrator into a
        # near-step function (see models/static_lgbm.py::train()).
        #
        # The margins come from X_cal, a split held out of BOTH the booster
        # fit and early stopping -- not X_val. best_iteration is chosen to
        # maximize separation on X_val, so val margins are optimistically
        # separated and a calibrator fit on them is over-confident on
        # genuinely unseen data (PDF review item 2). X_cal informs no fitting
        # or model-selection decision, so its margins are an honest basis for
        # the monotone probability map. Predicted at best_iteration, matching
        # inference.
        raw_cal_margins = booster.predict(X_cal, raw_score=True, num_iteration=booster.best_iteration)
        calibrator = PlattCalibrator().fit(raw_cal_margins, y_cal)

    return NetworkLGBMModel(booster=booster, calibrator=calibrator, feature_count=feature_count,
                             num_iterations=booster.best_iteration, model_hash=_model_hash(booster))


def evaluate(model: NetworkLGBMModel, X_test: NDArray[np.float32], y_test: NDArray[np.int32],
             threshold: float = 0.5) -> MetricsReport:
    proba = model.predict_proba(X_test)
    preds = (proba >= threshold).astype(np.int32)

    fpr_arr, tpr_arr, _ = roc_curve(y_test, proba)
    auc_roc = auc(fpr_arr, tpr_arr)
    prec_arr, rec_arr, _ = precision_recall_curve(y_test, proba)
    auc_pr = auc(rec_arr, prec_arr)

    benign = y_test == 0
    malicious = y_test == 1
    fpr_at_t = float((preds[benign] == 1).mean()) if benign.sum() else 0.0
    det_rate = float((preds[malicious] == 1).mean()) if malicious.sum() else 0.0

    return MetricsReport(
        accuracy=float(accuracy_score(y_test, preds)),
        precision=float(precision_score(y_test, preds, zero_division=0)),
        recall=float(recall_score(y_test, preds, zero_division=0)),
        f1=float(f1_score(y_test, preds, zero_division=0)),
        auc_roc=float(auc_roc), auc_pr=float(auc_pr),
        fpr_at_threshold=fpr_at_t, detection_rate=det_rate, threshold=threshold,
    )


def find_threshold_for_fpr(y_true: NDArray, proba: NDArray, target_fpr: float) -> float:
    """Pick the lowest score threshold that keeps benign FPR <= target_fpr.
    Identical to models/static_lgbm.py's and models/memory_lgbm.py's
    version of this function."""
    fpr_arr, tpr_arr, thresh_arr = roc_curve(y_true, proba)
    idx = np.searchsorted(fpr_arr, target_fpr, side="right") - 1
    idx = max(idx, 0)
    return float(thresh_arr[idx])
