"""
One-time migration of pickled model metadata to JSON (docs/CODE_REVIEW.md F24).

    python -m scripts.convert_model_meta_to_json \\
        --model data/models/cortex_static --model data/models/cortex_memory \\
        --model data/models/cortex_network --model data/models/cortex_static_retrain

ONLY RUN THIS ON FILES WE PRODUCED OURSELVES. It unpickles each <base>.meta
exactly once, and unpickling an untrusted file executes arbitrary code --
the very risk this migration removes. Check each .lgbm sha256 against its
independent record first (config/thresholds.yaml <kind>.model_sha256,
reports/, EVAL_ALL_MODELS_RESULTS.txt).

For each --model base it:
  * unpickles <base>.meta once,
  * checks that its model_hash is the first 16 hex digits of the .lgbm
    sha256 (the pickle belongs to this booster),
  * writes <base>.meta.json (refuses to overwrite unless --force),
  * reloads the model through the JSON-only loader and checks the restored
    calibrator coefficient/intercept are bit-identical to the pickle's.
The pickle is left in place but nothing reads it any more; delete it once
the JSON is verified.

--memory-scaler SRC DST converts a pickled MemoryFeatureScaler the same way.
"""
from __future__ import annotations

import argparse
import importlib
import pickle
import sys
from pathlib import Path

from models.lgbm_artifacts import (
    calibrator_to_dict, file_sha256, legacy_pickle_meta_path, lgbm_path, meta_json_path,
    write_meta_json,
)

_MODEL_CLASSES = {
    "models.static_lgbm": "LGBMModel",
    "models.memory_lgbm": "MemoryLGBMModel",
    "models.network_lgbm": "NetworkLGBMModel",
}


def convert_model(base: str, force: bool = False) -> Path:
    src = legacy_pickle_meta_path(base)
    out = meta_json_path(base)
    if out.exists() and not force:
        raise SystemExit(f"refusing to overwrite existing {out} (use --force)")
    with open(src, "rb") as fh:
        meta = pickle.load(fh)  # trusted, self-produced file only -- see module docstring

    cal = meta["calibrator"]
    module = type(cal).__module__ if cal is not None else None
    if cal is not None and module not in _MODEL_CLASSES:
        raise SystemExit(f"{src}: unexpected calibrator class {type(cal).__module__}.{type(cal).__name__}")
    sha = file_sha256(lgbm_path(base))
    if not sha.startswith(meta["model_hash"]):
        raise SystemExit(f"{src}: model_hash {meta['model_hash']} is not the prefix of "
                         f"{lgbm_path(base)} sha256 {sha} -- pickle and booster do not belong together")
    model_class = _MODEL_CLASSES[module]
    write_meta_json(base, model_class=model_class, calibrator=cal, feature_count=meta["feature_count"],
                    num_iterations=meta["num_iterations"], model_hash=meta["model_hash"])

    loaded = getattr(importlib.import_module(module), model_class).load(base)
    if calibrator_to_dict(loaded.calibrator) != calibrator_to_dict(cal):
        raise SystemExit(f"{out}: restored calibrator differs from the pickle")
    if (loaded.feature_count, loaded.num_iterations, loaded.model_hash) != (
            meta["feature_count"], meta["num_iterations"], meta["model_hash"]):
        raise SystemExit(f"{out}: restored metadata differs from the pickle")
    print(f"{src} -> {out}  ({model_class}, lgbm sha256 {sha})")
    return out


def convert_memory_scaler(src: str, dst: str, force: bool = False) -> Path:
    from features.memory_features import MemoryFeatureScaler
    out = Path(dst)
    if out.exists() and not force:
        raise SystemExit(f"refusing to overwrite existing {out} (use --force)")
    with open(src, "rb") as fh:
        scaler = pickle.load(fh)  # trusted, self-produced file only
    if not isinstance(scaler, MemoryFeatureScaler):
        raise SystemExit(f"{src}: not a MemoryFeatureScaler")
    scaler.save(out)
    back = MemoryFeatureScaler.load(out)
    if back.columns != scaler.columns or back.mean_.tobytes() != scaler.mean_.tobytes() \
            or back.std_.tobytes() != scaler.std_.tobytes():
        raise SystemExit(f"{out}: restored scaler differs from the pickle")
    print(f"{src} -> {out}  (MemoryFeatureScaler, {len(scaler.columns)} columns)")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", default=[], help="model base path (no suffix)")
    ap.add_argument("--memory-scaler", nargs=2, metavar=("SRC", "DST"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if not args.model and not args.memory_scaler:
        ap.error("nothing to convert")
    for base in args.model:
        convert_model(base, force=args.force)
    if args.memory_scaler:
        convert_memory_scaler(*args.memory_scaler, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
