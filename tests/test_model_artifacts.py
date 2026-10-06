"""
docs/CODE_REVIEW.md F24 -- safe model loading.

  * LightGBM metadata is JSON (<base>.meta.json); loaders never unpickle.
  * every load verifies the .lgbm sha256 against the JSON, and the deployed
    models against their config/thresholds.yaml pins.
  * a pickle-only model fails with a message naming the converter, which
    converts it once into a bit-identical JSON model.
  * MemoryFeatureScaler is JSON too.
  * every torch.load in the repo passes weights_only=True.
"""
from __future__ import annotations

import ast
import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from features.memory_features import MemoryFeatureScaler
from inference import policy_engine as pe
from models import memory_lgbm, network_lgbm, static_lgbm
from models.lgbm_artifacts import ModelArtifactError, file_sha256, lgbm_path, meta_json_path
from scripts.convert_model_meta_to_json import convert_memory_scaler, convert_model

_ROOT = Path(__file__).resolve().parents[1]
_KINDS = [(static_lgbm, "LGBMModel"), (memory_lgbm, "MemoryLGBMModel"), (network_lgbm, "NetworkLGBMModel")]


def _fit(module, cls_name, n_features=6, calibrated=True):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, n_features)).astype(np.float32)
    y = (X[:, 0] + 0.5 * rng.normal(size=400) > 0).astype(np.int32)
    booster = lgb.train({"objective": "binary", "verbose": -1, "num_leaves": 7, "seed": 1},
                        lgb.Dataset(X, label=y), num_boost_round=20)
    cal = module.PlattCalibrator().fit(booster.predict(X, raw_score=True), y) if calibrated else None
    model = getattr(module, cls_name)(booster=booster, calibrator=cal, feature_count=n_features,
                                      num_iterations=20, model_hash=module._model_hash(booster))
    return model, X


def _saved(tmp_path, module=static_lgbm, cls_name="LGBMModel"):
    model, X = _fit(module, cls_name)
    base = tmp_path / "m"
    model.save(base)
    return base, model, X


# ---------------------------------------------------------- round trip
@pytest.mark.parametrize("module,cls_name", _KINDS)
@pytest.mark.parametrize("calibrated", [True, False])
def test_json_round_trip_gives_identical_predictions(tmp_path, module, cls_name, calibrated):
    model, X = _fit(module, cls_name, calibrated=calibrated)
    base = tmp_path / "m"
    model.save(base)
    assert meta_json_path(base).is_file() and not base.with_suffix(".meta").exists()
    loaded = getattr(module, cls_name).load(base)
    assert loaded.predict_proba(X).tobytes() == model.predict_proba(X).tobytes()
    meta = json.loads(meta_json_path(base).read_text())
    assert meta["lgbm_sha256"] == file_sha256(lgbm_path(base))
    assert (meta["calibrator"] is None) is (not calibrated)


def test_meta_json_is_plain_data(tmp_path):
    base, _, _ = _saved(tmp_path)
    meta = json.loads(meta_json_path(base).read_text())
    assert set(meta) == {"format_version", "model_class", "feature_count", "num_iterations",
                         "model_hash", "lgbm_sha256", "calibrator"}
    assert set(meta["calibrator"]) == {"type", "coef", "intercept", "classes"}


# ---------------------------------------------------------- tampering
def test_tampered_model_file_fails_loudly(tmp_path):
    base, _, _ = _saved(tmp_path)
    text = lgbm_path(base).read_text()
    lgbm_path(base).write_text(text.replace("leaf_value=", "leaf_value=9", 1))
    with pytest.raises(ModelArtifactError, match="replaced or modified"):
        static_lgbm.LGBMModel.load(base)


def test_swapped_model_file_fails_loudly(tmp_path):
    base, _, _ = _saved(tmp_path)
    other, _ = _fit(static_lgbm, "LGBMModel", n_features=7)
    other.booster.save_model(str(lgbm_path(base)))
    with pytest.raises(ModelArtifactError, match="sha256"):
        static_lgbm.LGBMModel.load(base)


def test_pinned_hash_mismatch_fails_loudly(tmp_path):
    base, _, _ = _saved(tmp_path)
    static_lgbm.LGBMModel.load(base, expected_sha256=file_sha256(lgbm_path(base)))  # ok
    with pytest.raises(ModelArtifactError, match="config/thresholds.yaml"):
        static_lgbm.LGBMModel.load(base, expected_sha256="0" * 64)


def test_wrong_model_class_fails(tmp_path):
    base, _, _ = _saved(tmp_path)
    with pytest.raises(ModelArtifactError, match="is for LGBMModel"):
        memory_lgbm.MemoryLGBMModel.load(base)


# ------------------------------------------------------ pickle-only
def _legacy_pickle_model(tmp_path):
    """A model in the pre-F24 layout: .lgbm + pickled .meta, written by this
    test (a trusted, self-produced file)."""
    model, X = _fit(static_lgbm, "LGBMModel")
    base = tmp_path / "legacy"
    model.booster.save_model(str(lgbm_path(base)), num_iteration=model.num_iterations)
    with open(base.with_suffix(".meta"), "wb") as fh:
        pickle.dump({"calibrator": model.calibrator, "feature_count": model.feature_count,
                     "num_iterations": model.num_iterations, "model_hash": model.model_hash}, fh)
    return base, model, X


def test_pickle_only_model_gives_clear_error_naming_converter(tmp_path):
    base, _, _ = _legacy_pickle_model(tmp_path)
    with pytest.raises(ModelArtifactError, match="convert_model_meta_to_json") as exc:
        static_lgbm.LGBMModel.load(base)
    assert "pickle" in str(exc.value) and str(base) in str(exc.value)


def test_no_metadata_at_all_is_an_error(tmp_path):
    base, _, _ = _saved(tmp_path)
    meta_json_path(base).unlink()
    with pytest.raises(ModelArtifactError, match="no model metadata"):
        static_lgbm.LGBMModel.load(base)


def test_converter_produces_bit_identical_json_model(tmp_path):
    base, model, X = _legacy_pickle_model(tmp_path)
    convert_model(str(base))
    loaded = static_lgbm.LGBMModel.load(base)
    assert loaded.predict_proba(X).tobytes() == model.predict_proba(X).tobytes()
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        convert_model(str(base))


def test_converter_rejects_pickle_from_another_booster(tmp_path):
    base, _, _ = _legacy_pickle_model(tmp_path)
    other, _ = _fit(static_lgbm, "LGBMModel", n_features=7)
    other.booster.save_model(str(lgbm_path(base)))
    with pytest.raises(SystemExit, match="do not belong together"):
        convert_model(str(base))
    assert not meta_json_path(base).exists()


# ------------------------------------------------------ malformed JSON
@pytest.mark.parametrize("mutate", [
    lambda m: "{not json",
    lambda m: json.dumps([m]),
    lambda m: json.dumps({**m, "format_version": 2}),
    lambda m: json.dumps({k: v for k, v in m.items() if k != "lgbm_sha256"}),
    lambda m: json.dumps({**m, "feature_count": "6"}),
    lambda m: json.dumps({**m, "num_iterations": True}),
    lambda m: json.dumps({k: v for k, v in m.items() if k != "calibrator"}),
    lambda m: json.dumps({**m, "calibrator": {**m["calibrator"], "coef": "1.0"}}),
    lambda m: json.dumps({**m, "calibrator": {**m["calibrator"], "classes": [1, 0]}}),
    lambda m: json.dumps({**m, "calibrator": {**m["calibrator"], "type": "isotonic"}}),
], ids=["not-json", "not-object", "version", "no-sha", "str-int", "bool-int", "no-calibrator",
        "str-coef", "classes", "cal-type"])
def test_malformed_json_gives_clear_error(tmp_path, mutate):
    base, _, _ = _saved(tmp_path)
    meta = json.loads(meta_json_path(base).read_text())
    meta_json_path(base).write_text(mutate(meta))
    with pytest.raises(ModelArtifactError, match=str(meta_json_path(base).name)):
        static_lgbm.LGBMModel.load(base)


def test_non_finite_calibrator_coefficient_rejected(tmp_path):
    base, _, _ = _saved(tmp_path)
    text = meta_json_path(base).read_text()
    meta = json.loads(text)
    meta_json_path(base).write_text(text.replace(repr(meta["calibrator"]["coef"]), "NaN", 1))
    with pytest.raises(ModelArtifactError, match="finite"):
        static_lgbm.LGBMModel.load(base)


# --------------------------------------------------------- memory scaler
def _scaler():
    return MemoryFeatureScaler(columns=["a", "b", "c"], mean_=np.array([0.1, 1 / 3, -2.5e-17]),
                               std_=np.array([1.0, 7.123456789012345, 3e300]))


def test_memory_scaler_json_round_trip(tmp_path):
    s = _scaler()
    s.save(tmp_path / "scaler.json")
    back = MemoryFeatureScaler.load(tmp_path / "scaler.json")
    assert back.columns == s.columns
    assert back.mean_.tobytes() == s.mean_.tobytes() and back.std_.tobytes() == s.std_.tobytes()
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0]})
    assert back.transform(df).tobytes() == s.transform(df).tobytes()


def test_pickled_memory_scaler_gives_clear_error_and_converts(tmp_path):
    src = tmp_path / "scaler.pkl"
    with open(src, "wb") as fh:
        pickle.dump(_scaler(), fh, protocol=pickle.HIGHEST_PROTOCOL)  # self-produced
    with pytest.raises(ValueError, match="convert_model_meta_to_json"):
        MemoryFeatureScaler.load(src)
    convert_memory_scaler(str(src), str(tmp_path / "scaler.json"))
    assert MemoryFeatureScaler.load(tmp_path / "scaler.json").mean_.tobytes() == _scaler().mean_.tobytes()


@pytest.mark.parametrize("payload", ["{bad", json.dumps({"format_version": 1, "columns": ["a"], "mean": [], "std": [1]}),
                                     json.dumps({"columns": ["a"], "mean": [0], "std": [1]})])
def test_malformed_memory_scaler_json(tmp_path, payload):
    (tmp_path / "s.json").write_text(payload)
    with pytest.raises(ValueError):
        MemoryFeatureScaler.load(tmp_path / "s.json")


# ------------------------------------------------- no pickle, safe torch
def test_loaders_never_unpickle():
    """No pickle import in any model/feature loader module (the converter
    script is the one deliberate exception)."""
    for rel in ("models/static_lgbm.py", "models/memory_lgbm.py", "models/network_lgbm.py",
                "models/lgbm_artifacts.py", "features/memory_features.py",
                "models/behavioral_artifacts.py"):
        tree = ast.parse((_ROOT / rel).read_text(encoding="utf-8"))
        imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        imported |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        assert not imported & {"pickle", "cPickle", "dill", "joblib", "cloudpickle"}, rel


def _torch_load_calls():
    for path in _ROOT.rglob("*.py"):
        if any(part in {".git", ".venv", "venv", "site-packages"} for part in path.parts):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "load" and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "torch"):
                yield path.relative_to(_ROOT), node


def test_every_torch_load_is_weights_only():
    calls = list(_torch_load_calls())
    assert calls  # behavioral + emulation at least
    for rel, node in calls:
        kw = {k.arg: k.value for k in node.keywords}
        assert isinstance(kw.get("weights_only"), ast.Constant) and kw["weights_only"].value is True, \
            f"{rel}:{node.lineno} torch.load without weights_only=True"


# ----------------------------------------------------- yaml pins / real
def test_pins_are_loaded_from_yaml():
    for name in ("STATIC_MODEL_SHA256", "MEMORY_MODEL_SHA256", "NETWORK_MODEL_SHA256"):
        v = getattr(pe, name)
        assert len(v) == 64 and set(v) <= set("0123456789abcdef")


@pytest.mark.parametrize("bad", [None, "ABC", "g" * 64, 123, "a" * 63])
def test_sha256_pin_loader_is_strict(bad):
    with pytest.raises(RuntimeError, match="sha256|missing"):
        pe._sha256({"static": {} if bad is None else {"model_sha256": bad}}, "static", "model_sha256")


_DEPLOYED = [(static_lgbm.LGBMModel, "data/models/cortex_static", "STATIC_MODEL_SHA256"),
             (memory_lgbm.MemoryLGBMModel, "data/models/cortex_memory", "MEMORY_MODEL_SHA256"),
             (network_lgbm.NetworkLGBMModel, "data/models/cortex_network", "NETWORK_MODEL_SHA256")]


@pytest.mark.slow
@pytest.mark.parametrize("cls,base,pin", _DEPLOYED, ids=["static", "memory", "network"])
def test_deployed_model_matches_its_threshold_pin(cls, base, pin):
    base = _ROOT / base
    if not (lgbm_path(base).is_file() and meta_json_path(base).is_file()):
        pytest.skip("deployed model / converted meta.json absent")
    model = cls.load(base, expected_sha256=getattr(pe, pin))
    assert getattr(pe, pin).startswith(model.model_hash)
