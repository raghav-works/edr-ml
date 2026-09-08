"""
Cortex-Static: LightGBM trainer for the 2568-dim EMBER2024 feature vector.
"""

from __future__ import annotations

import gc
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

logger = logging.getLogger("cortex.static.train")

EMBER2024_FEATURE_COUNT = 2568
DEFAULT_SEED = 42

# Hyperparameters — matched to the 2568-feature / large-dataset regime
DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 255,
    "max_depth": 15,
    "learning_rate": 0.03,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 5.0,
    "reg_lambda": 0.01,
    "min_split_gain": 0.01,
    "is_unbalance": True,
    "seed": DEFAULT_SEED,
    # -1 (all 20 cores) OOM-killed histogram construction for num_leaves=255
    # on the full 2.34M-row / 2568-feature EMBER2024 train split on a 38GB
    # machine -- each thread builds its own per-feature histogram buffers,
    # and that overhead multiplies with thread count. Capping to 6 is a pure
    # resource/performance tradeoff (slower wall-clock, same model), not a
    # change to what gets learned -- num_leaves/max_depth/etc. are untouched.
    "n_jobs": 6,
    "verbose": -1,
    # Bounds LightGBM's internal histogram cache to 4GB -- purely a
    # cache-vs-recompute tradeoff during tree growth, does not affect split
    # decisions, tree structure, or any result the model produces.
    "histogram_pool_size": 4096,
}
DEFAULT_N_ESTIMATORS = 3000
DEFAULT_EARLY_STOPPING = 50


class PlattCalibrator:
    """Platt scaling: logistic regression on raw booster margins."""

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
class LGBMModel:
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
    def load(cls, path: Union[str, Path]) -> "LGBMModel":
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
) -> LGBMModel:
    hp = {**DEFAULT_PARAMS, **(params or {})}
    feature_count = X_train.shape[1]
    # free_raw_data=True lets LightGBM drop its own copy of the raw float32
    # arrays once each Dataset is constructed/binned, instead of holding
    # them alive (on top of Python's own references and the internal binned
    # histograms) for the entire boosting run -- on the full 2.34M-row
    # EMBER2024 train split, free_raw_data=False was enough to OOM-kill
    # training on a 38GB machine even after fixing the data-loading path.
    # X_val's raw array is still needed after training for calibration, but
    # that reads the `X_val` parameter directly, not anything from val_set,
    # so val_set's own internal copy can be dropped too.
    # two_round=True makes Dataset construction itself use a lower-peak-memory
    # loading strategy rather than holding the full unbinned array alongside
    # the binned representation during construction.
    train_set = lgb.Dataset(X_train, label=y_train, free_raw_data=True, params={"two_round": True})
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set, free_raw_data=True, params={"two_round": True})
    # X_val is NOT deleted here -- the calibration step below calls
    # booster.predict(X_val, ...) directly on this raw array (not on
    # val_set), so it has to survive past lgb.train(). Only X_train is
    # safe to drop: nothing references it again after Dataset construction.
    del X_train
    gc.collect()

    t0 = time.monotonic()
    booster = lgb.train(
        hp, train_set, num_boost_round=n_estimators,
        valid_sets=[train_set, val_set], valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(100)],
    )
    logger.info("Training done in %.1fs, best_iteration=%d", time.monotonic() - t0, booster.best_iteration)

    calibrator = None
    if calibrate:
        # Textbook Platt scaling fits the logistic regression on the model's
        # raw margins, NOT on probabilities. Fitting on booster.predict()'s
        # sigmoid output -- an already-[0,1], near-separable distribution --
        # collapses the calibrator into a near-step function and destroys
        # rank resolution in the decision-boundary zone (the Cortex-Static
        # calibration-saturation bug).
        raw_val_margins = booster.predict(X_val, raw_score=True, num_iteration=booster.best_iteration)
        calibrator = PlattCalibrator().fit(raw_val_margins, y_val)

    return LGBMModel(booster=booster, calibrator=calibrator, feature_count=feature_count,
                      num_iterations=booster.best_iteration, model_hash=_model_hash(booster))


def evaluate(model: LGBMModel, X_test: NDArray[np.float32], y_test: NDArray[np.int32],
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
    """Pick the lowest score threshold that keeps benign FPR <= target_fpr."""
    fpr_arr, tpr_arr, thresh_arr = roc_curve(y_true, proba)
    idx = np.searchsorted(fpr_arr, target_fpr, side="right") - 1
    idx = max(idx, 0)
    return float(thresh_arr[idx])
