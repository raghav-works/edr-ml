"""
LightGBM model metadata as JSON, with an integrity check on every load
(docs/CODE_REVIEW.md F24).

A model is two files next to each other:

    <base>.lgbm        LightGBM's own text format (booster.save_model)
    <base>.meta.json   this module: feature_count, num_iterations,
                       model_hash, the Platt calibrator's coefficient /
                       intercept / classes, and the .lgbm file's sha256

The .meta file used to be a pickle; unpickling runs arbitrary code, so a
replaced .meta was code execution in the scanner. Loaders now read JSON
only. An existing pickle .meta is converted once with
scripts/convert_model_meta_to_json.py (our own trusted files only) and
then never read again.

Every load recomputes the .lgbm sha256 and compares it with the JSON's
`lgbm_sha256`; callers that load a deployed model also pass
`expected_sha256` from config/thresholds.yaml (<kind>.model_sha256), which
is version-controlled and pairs the model with its thresholds.

The three PlattCalibrator classes (static/memory/network) stay separate;
this module only reads and restores the fitted LogisticRegression inside
them. Restoring sets the same attributes sklearn's fit() sets on an
estimator built with the same constructor arguments, so predict_proba runs
the same code on the same float64 values; JSON floats round-trip exactly.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np

if TYPE_CHECKING:
    import lightgbm as lgb

META_FORMAT_VERSION = 1
CONVERTER = "scripts/convert_model_meta_to_json.py"

PathLike = Union[str, Path]


class ModelArtifactError(RuntimeError):
    """A model file or its metadata is missing, malformed, or does not match."""


def lgbm_path(base: PathLike) -> Path:
    return Path(base).with_suffix(".lgbm")


def meta_json_path(base: PathLike) -> Path:
    return Path(base).with_suffix(".meta.json")


def legacy_pickle_meta_path(base: PathLike) -> Path:
    return Path(base).with_suffix(".meta")


def file_sha256(path: PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------- calibrator
def calibrator_to_dict(calibrator: Any) -> Optional[dict]:
    """{'type': 'platt', 'coef': float, 'intercept': float, 'classes': [0, 1]}
    from a fitted PlattCalibrator, or None."""
    if calibrator is None:
        return None
    lr = calibrator._lr
    coef = np.asarray(lr.coef_, dtype=np.float64)
    intercept = np.asarray(lr.intercept_, dtype=np.float64)
    if coef.shape != (1, 1) or intercept.shape != (1,):
        raise ModelArtifactError(f"Platt calibrator must be 1-feature binary, got coef {coef.shape}")
    return {"type": "platt", "coef": float(coef[0, 0]), "intercept": float(intercept[0]),
            "classes": [int(c) for c in lr.classes_]}


def restore_calibrator(calibrator_cls: type, spec: Optional[dict]) -> Any:
    """Inverse of calibrator_to_dict() for one of the PlattCalibrator classes."""
    if spec is None:
        return None
    cal = calibrator_cls()
    lr = cal._lr
    lr.coef_ = np.array([[spec["coef"]]], dtype=np.float64)
    lr.intercept_ = np.array([spec["intercept"]], dtype=np.float64)
    lr.classes_ = np.array(spec["classes"])
    lr.n_features_in_ = 1
    lr.n_iter_ = np.array([0], dtype=np.int32)
    return cal


def _check_calibrator_spec(spec: Any, where: Path) -> None:
    if spec is None:
        return
    if not isinstance(spec, dict) or spec.get("type") != "platt":
        raise ModelArtifactError(f"{where}: 'calibrator' must be null or a platt object, got {spec!r}")
    for key in ("coef", "intercept"):
        v = spec.get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ModelArtifactError(f"{where}: calibrator {key!r} must be a finite number, got {v!r}")
    if spec.get("classes") != [0, 1]:
        raise ModelArtifactError(f"{where}: calibrator 'classes' must be [0, 1], got {spec.get('classes')!r}")


# ------------------------------------------------------------------ meta
def write_meta_json(base: PathLike, *, model_class: str, calibrator: Any, feature_count: int,
                    num_iterations: int, model_hash: str) -> Path:
    """Write <base>.meta.json for the <base>.lgbm currently on disk."""
    meta = {
        "format_version": META_FORMAT_VERSION,
        "model_class": model_class,
        "feature_count": int(feature_count),
        "num_iterations": int(num_iterations),
        "model_hash": str(model_hash),
        "lgbm_sha256": file_sha256(lgbm_path(base)),
        "calibrator": calibrator_to_dict(calibrator),
    }
    out = meta_json_path(base)
    out.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return out


_REQUIRED = {"format_version": int, "model_class": str, "feature_count": int,
             "num_iterations": int, "model_hash": str, "lgbm_sha256": str}


def load_verified_booster(base: PathLike, *, model_class: str,
                          expected_sha256: Optional[str] = None) -> tuple[dict, "lgb.Booster"]:
    """Read and validate <base>.meta.json, read <base>.lgbm ONCE, verify
    those bytes against the JSON's lgbm_sha256 (and `expected_sha256`, if
    given), and build the booster from the same bytes -- so the file cannot
    change between the check and the load. Never reads a pickle."""
    path = meta_json_path(base)
    if not path.is_file():
        legacy = legacy_pickle_meta_path(base)
        if legacy.is_file():
            raise ModelArtifactError(
                f"{legacy} is a pickle; model metadata is read from JSON only. Convert it once "
                f"with `python -m {CONVERTER[:-3].replace('/', '.')} --model {base}` (only for "
                f"model files we produced ourselves), which writes {path.name}."
            )
        raise ModelArtifactError(f"no model metadata {path} for {lgbm_path(base)}")
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelArtifactError(f"unreadable model metadata {path}: {exc}") from exc
    if not isinstance(meta, dict):
        raise ModelArtifactError(f"{path} is not a JSON object")
    for key, typ in _REQUIRED.items():
        if key not in meta:
            raise ModelArtifactError(f"{path} is missing {key!r}")
        if not isinstance(meta[key], typ) or (typ is int and isinstance(meta[key], bool)):
            raise ModelArtifactError(f"{path}: {key!r} must be {typ.__name__}, got {meta[key]!r}")
    if meta["format_version"] != META_FORMAT_VERSION:
        raise ModelArtifactError(f"{path}: unsupported format_version {meta['format_version']}")
    if meta["model_class"] != model_class:
        raise ModelArtifactError(f"{path} is for {meta['model_class']}, not {model_class}")
    if "calibrator" not in meta:
        raise ModelArtifactError(f"{path} is missing 'calibrator' (null if uncalibrated)")
    _check_calibrator_spec(meta["calibrator"], path)

    model_file = lgbm_path(base)
    try:
        model_bytes = model_file.read_bytes()
    except OSError as exc:
        raise ModelArtifactError(f"cannot read model file {model_file}: {exc}") from exc
    actual = hashlib.sha256(model_bytes).hexdigest()
    if actual != meta["lgbm_sha256"]:
        raise ModelArtifactError(
            f"{model_file} sha256 {actual} does not match {path} lgbm_sha256 "
            f"{meta['lgbm_sha256']} -- the model file was replaced or modified")
    if expected_sha256 is not None and actual != expected_sha256:
        raise ModelArtifactError(
            f"{model_file} sha256 {actual} does not match the pinned {expected_sha256} "
            "(config/thresholds.yaml) -- these thresholds were not derived for this model")
    import lightgbm as lgb
    return meta, lgb.Booster(model_str=model_bytes.decode("utf-8"))
