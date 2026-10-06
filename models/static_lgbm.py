"""
Cortex-Static: LightGBM trainer for the 2568-dim EMBER2024 feature vector.
"""

from __future__ import annotations

import gc
import hashlib
import logging
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

from models.lgbm_artifacts import load_verified_booster, restore_calibrator, write_meta_json

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
        # JSON, not pickle (docs/CODE_REVIEW.md F24); records the .lgbm sha256
        write_meta_json(base, model_class="LGBMModel", calibrator=self.calibrator,
                        feature_count=self.feature_count, num_iterations=self.num_iterations,
                        model_hash=self.model_hash)

    @classmethod
    def load(cls, path: Union[str, Path], expected_sha256: Optional[str] = None) -> "LGBMModel":
        """Reads <path>.meta.json (never a pickle) and <path>.lgbm once,
        verifies the .lgbm sha256 against the JSON and, if given, against
        `expected_sha256` (the deployed model's pin in config/thresholds.yaml),
        and builds the booster from the verified bytes. Raises
        ModelArtifactError on any mismatch."""
        base = Path(path)
        meta, booster = load_verified_booster(base, model_class="LGBMModel", expected_sha256=expected_sha256)
        return cls(booster=booster, calibrator=restore_calibrator(PlattCalibrator, meta["calibrator"]),
                   feature_count=meta["feature_count"], num_iterations=meta["num_iterations"],
                   model_hash=meta["model_hash"])


def _model_hash(booster: lgb.Booster) -> str:
    return hashlib.sha256(booster.model_to_string().encode("utf-8")).hexdigest()[:16]


def train_from_holder(
    holder: dict,
    *, params: Optional[dict[str, Any]] = None,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    early_stopping_rounds: int = DEFAULT_EARLY_STOPPING,
) -> LGBMModel:
    """Boosting only -- returns an LGBMModel with calibrator=None. Fit the
    Platt calibrator afterward with calibrate() on a THIRD split (cal), once
    X_train/X_val are already freed. Keeping calibration out of this
    function means cal's raw array is never resident at the same time as
    Dataset construction + boosting -- the single riskiest memory window
    for a 2568-feature / multi-million-row Dataset (see the OOM note on
    lgb.Dataset() below).

    Takes a plain dict (holder), not (X_train, y_train, X_val, y_val)
    directly, and pops each array out of it rather than receiving them as
    positional arguments. This is not a style choice -- measured directly
    on this interpreter (Python 3.10.12, via weakref + RssAnon, not
    assumed) that a caller's own reference to a large array survives for
    the ENTIRE duration of any call where that array is a positional
    argument, whether the caller binds it to a name first or unpacks a
    producer's return value inline (`f(*g())` measured no different from
    `X = g(); f(X)`) -- CPython's call mechanism keeps the caller's
    evaluation-stack slot alive until the call returns either way. A
    version of this function that took (X_train, X_val, ...) directly
    would have its own internal `del X_train` free only ITS OWN local
    reference; scripts/train_static.py::main(), which also holds X_train,
    would keep the full ~18GB array resident for this function's ENTIRE
    body -- both Dataset() constructs AND the full boosting call --
    defeating the whole point of freeing it early (this is exactly what
    happened: Attempt 3 crashed at ~36.6GB RSS with this bug still present).

    The fix: the caller builds a plain dict whose values are the arrays --
    never binding them to a separate name of its own (see
    scripts/train_static.py::main()) -- and this function pops each array
    out of the dict (removing the dict's reference) before using and
    deleting it. The caller's persistent reference is to the dict object,
    not to the arrays inside it; once popped, nothing the caller holds
    still points to the array, so this function's `del` genuinely drops
    the last reference. Verified with weakref.ref() + RssAnon before this
    pattern was adopted, not assumed to work.

    train() below is a thin backward-compatible wrapper for callers that
    already hold plain X_train/X_val arrays (small-scale use, future
    tests) -- it does NOT get the same guaranteed mid-call freeing, since
    ITS caller still holds its own reference for the wrapper's call
    duration, same as before this fix.
    """
    X_train = holder.pop("X_train")
    y_train = holder.pop("y_train")
    X_val = holder.pop("X_val")
    y_val = holder.pop("y_val")

    hp = {**DEFAULT_PARAMS, **(params or {})}
    feature_count = X_train.shape[1]
    # free_raw_data=True lets LightGBM drop its own copy of the raw float32
    # arrays once each Dataset is constructed/binned, instead of holding
    # them alive (on top of Python's own references and the internal binned
    # histograms) for the entire boosting run -- on the full 2.34M-row
    # EMBER2024 train split, free_raw_data=False was enough to OOM-kill
    # training on a 38GB machine even after fixing the data-loading path.
    # two_round=True makes Dataset construction itself use a lower-peak-memory
    # loading strategy rather than holding the full unbinned array alongside
    # the binned representation during construction.
    #
    # num_threads is set here explicitly -- Dataset construction (binning
    # 2568 features) otherwise runs with LightGBM's own default thread
    # count (effectively every core on the machine), NOT the n_jobs cap
    # in DEFAULT_PARAMS below, since that dict is only ever passed to
    # lgb.train(), never to lgb.Dataset(). A prior crash on this machine
    # showed ~7GB of available memory disappear in a single 21-second
    # window immediately after the loader finished and inside these two
    # Dataset() calls -- far faster than the loader's own steady per-batch
    # pace, consistent with uncapped parallel binning across 2568 features
    # rather than a gradual leak. Matching DEFAULT_PARAMS["n_jobs"] here
    # closes that gap instead of guessing at a new number.
    #
    # lgb.Dataset(...) itself is LAZY -- it just stores a reference to the
    # raw array (ds.data is X_train right after this call, confirmed via
    # sys.getrefcount()); the actual binning, and free_raw_data=True's
    # ds.data = None, only happen inside .construct() (called implicitly
    # by lgb.train() otherwise). A prior fix's `del X_train, X_val` placed
    # here, before .construct() ever ran, was a no-op -- train_set.data
    # was still a live reference to the same array, so nothing was
    # actually freed, and a follow-up real run showed RSS climb without
    # plateauing (27GB -> 36.6GB, still rising when killed) because both
    # raw arrays stayed resident simultaneously all the way through
    # lgb.train()'s own internal construction of both Datasets. Calling
    # .construct() explicitly and sequentially -- del'ing X_train only
    # after train_set is actually constructed, before X_val/val_set are
    # even touched -- makes free_raw_data's reference-drop happen when the
    # code actually expects it to, not whenever lgb.train() gets around to
    # it internally.
    dataset_params = {"two_round": True, "num_threads": DEFAULT_PARAMS["n_jobs"]}
    train_set = lgb.Dataset(X_train, label=y_train, free_raw_data=True, params=dataset_params)
    train_set.construct()
    del X_train
    gc.collect()

    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set, free_raw_data=True, params=dataset_params)
    val_set.construct()
    del X_val
    gc.collect()

    t0 = time.monotonic()
    booster = lgb.train(
        hp, train_set, num_boost_round=n_estimators,
        valid_sets=[train_set, val_set], valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(100)],
    )
    logger.info("Training done in %.1fs, best_iteration=%d", time.monotonic() - t0, booster.best_iteration)

    return LGBMModel(booster=booster, calibrator=None, feature_count=feature_count,
                      num_iterations=booster.best_iteration, model_hash=_model_hash(booster))


def train(
    X_train: NDArray[np.float32], y_train: NDArray[np.int32],
    X_val: NDArray[np.float32], y_val: NDArray[np.int32],
    *, params: Optional[dict[str, Any]] = None,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    early_stopping_rounds: int = DEFAULT_EARLY_STOPPING,
) -> LGBMModel:
    """Backward-compatible wrapper around train_from_holder() for callers
    that already hold X_train/X_val as plain arrays (small-scale use,
    future tests).

    NOT identical to the committed-HEAD train(): that version took a
    `calibrate: bool = True` kwarg and returned a Platt-calibrated model,
    fit inline on X_val margins. This session's earlier calibrate-later
    redesign (train() / calibrate() split, so cal's raw array is never
    resident during Dataset construction + boosting) already removed that
    -- this function always returns calibrator=None, has no `calibrate`
    kwarg, and calibration only happens via a separate calibrate(model,
    X_cal, y_cal) call, on a different split, after this returns. That
    part is unchanged by today's fix.

    What IS unchanged by today's fix, relative to the calibrate-later
    version that immediately preceded it: same params, same Dataset
    construction, same boosting call. Today's change is a
    calling-convention change only (delegates to train_from_holder() via
    a holder dict instead of doing the work inline) -- Step C and Step D
    measured model_hash/best_iteration identical before and after this
    specific change, on both a 20,000-row and a 500,000-row real slice.

    Does NOT get train_from_holder()'s guaranteed mid-call freeing: THIS
    function's own caller still holds its own reference to X_train/X_val
    for the duration of this call, exactly as before this fix (see
    train_from_holder()'s docstring for why that matters and when it
    doesn't). scripts/train_static.py calls train_from_holder() directly
    with a holder it builds and never separately names, precisely because
    it needs that guarantee at real data scale.
    """
    return train_from_holder(
        {"X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val},
        params=params, n_estimators=n_estimators, early_stopping_rounds=early_stopping_rounds,
    )


def calibrate(model: LGBMModel, X_cal: NDArray[np.float32], y_cal: NDArray[np.int32]) -> LGBMModel:
    """Fit the Platt calibrator on a held-out cal split and return a new
    LGBMModel with it attached. Deliberately separate from train(): cal's
    raw array should only be loaded/resident AFTER boosting finishes and
    X_train/X_val are already freed (see train()'s docstring) -- calling
    this from a fresh cal load, not from arrays held since before training,
    is what actually keeps cal out of the Dataset-construction/boosting
    memory peak.

    Textbook Platt scaling fits the logistic regression on the model's raw
    margins, NOT on probabilities. Fitting on booster.predict()'s sigmoid
    output -- an already-[0,1], near-separable distribution -- collapses
    the calibrator into a near-step function and destroys rank resolution
    in the decision-boundary zone (the Cortex-Static calibration-saturation
    bug).

    The margins come from X_cal, not X_val: best_iteration is chosen to
    maximize separation on X_val, so val margins are optimistically
    separated and a calibrator fit on them is over-confident on genuinely
    unseen data (PDF review item 2). X_cal informs no fitting or
    model-selection decision, so its margins are an honest basis for the
    monotone probability map. Predicted at best_iteration, matching
    inference.
    """
    raw_cal_margins = model.booster.predict(X_cal, raw_score=True, num_iteration=model.num_iterations)
    calibrator = PlattCalibrator().fit(raw_cal_margins, y_cal)
    return LGBMModel(booster=model.booster, calibrator=calibrator, feature_count=model.feature_count,
                      num_iterations=model.num_iterations, model_hash=model.model_hash)


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
