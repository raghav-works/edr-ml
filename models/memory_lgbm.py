"""
Cortex-Memory: LightGBM trainer for the 62-dim (55 raw VolMemLyzer + 7
derived, see features/memory_features.py) CIC-MalMem-2022 feature vector.
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

logger = logging.getLogger("cortex.memory.train")

MEMORY_FEATURE_COUNT = 62  # 55 raw VolMemLyzer + 7 derived (features/memory_features.py)
DEFAULT_SEED = 42

# Hyperparameters -- matched to the 62-feature / ~47K-row regime, which is
# two to three orders of magnitude smaller than static's 2568-feature /
# 2.34M-row EMBER2024 problem. None of static_lgbm.py's memory-management
# tricks (n_jobs cap, histogram_pool_size, two_round Dataset construction,
# manual free_raw_data bookkeeping) are needed here -- this dataset doesn't
# come close to the machine's memory ceiling, so copying those in would be
# needless complexity, not caution.
DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    # LightGBM's own default. static's num_leaves=255 is for 2568 features;
    # blindly reusing that here on 62 features / ~47K rows would build trees
    # far more complex than the feature space or dataset size warrant --
    # inviting overfitting, not matching a proven config.
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
    # is_unbalance deliberately omitted (unlike static's EMBER2024 config):
    # the memory train split is 23,438 benign / 23,298 malware (~50.1% /
    # 49.9%), already effectively balanced, so class reweighting isn't
    # needed and would just distort calibration for no benefit.
}
DEFAULT_N_ESTIMATORS = 1000
DEFAULT_EARLY_STOPPING = 50


class PlattCalibrator:
    """Platt scaling: logistic regression on raw booster margins.

    Identical implementation to models/static_lgbm.py::PlattCalibrator,
    duplicated rather than imported -- keeps this module self-contained the
    same way models/train_behavioral.py keeps its own
    BehavioralMetricsReport instead of importing static's MetricsReport,
    rather than coupling two otherwise-independent model modules together.
    """

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
class MemoryLGBMModel:
    booster: lgb.Booster
    calibrator: Optional[PlattCalibrator]
    feature_count: int
    num_iterations: int
    model_hash: str

    def predict_proba(self, X: NDArray[np.float32]) -> NDArray[np.float64]:
        if X.shape[1] != self.feature_count:
            raise ValueError(f"Expected {self.feature_count} features, got {X.shape[1]}")
        raw = self.booster.predict(X, num_iteration=self.num_iterations)
        return self.calibrator.predict_proba(raw) if self.calibrator else raw

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
    def load(cls, path: Union[str, Path]) -> "MemoryLGBMModel":
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
    *, params: Optional[dict[str, Any]] = None,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    early_stopping_rounds: int = DEFAULT_EARLY_STOPPING,
    calibrate: bool = True,
) -> MemoryLGBMModel:
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
        raw_val_scores = booster.predict(X_val, num_iteration=booster.best_iteration)
        calibrator = PlattCalibrator().fit(raw_val_scores, y_val)

    return MemoryLGBMModel(booster=booster, calibrator=calibrator, feature_count=feature_count,
                            num_iterations=booster.best_iteration, model_hash=_model_hash(booster))


def evaluate(model: MemoryLGBMModel, X_test: NDArray[np.float32], y_test: NDArray[np.int32],
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
    Identical to models/static_lgbm.py's version of this function."""
    fpr_arr, tpr_arr, thresh_arr = roc_curve(y_true, proba)
    idx = np.searchsorted(fpr_arr, target_fpr, side="right") - 1
    idx = max(idx, 0)
    return float(thresh_arr[idx])
