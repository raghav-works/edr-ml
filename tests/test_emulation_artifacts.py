"""
Emulation checkpoint vs code (the emulation counterpart of docs/CODE_REVIEW.md
F2) and the all-padding (empty-trace) guard for both sequence models.

  * CortexEmulationNet(use_padding_mask=False) is the pre-b876d26 forward
    pass; True is the masked one (default for new checkpoints).
  * every emulation checkpoint has a JSON sidecar
    (models/emulation_artifacts.py); training writes it, the loader refuses
    to run without it and fails loudly on a checkpoint/vocab sha256 mismatch.
  * an all-padding row is never scored: the scoring helpers refuse it,
    training drops it, and no path returns NaN.
  * end to end: the shipped checkpoint through the loader reproduces the
    recorded val numbers (skipped if absent).
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from models import train_behavioral, train_emulation
from models.behavioral_artifacts import BehavioralArtifactError, load_behavioral_model
from models.behavioral_cnn import CortexBehavioralNet
from models.emulation_artifacts import (
    EmulationArtifactError, load_emulation_model, read_emulation_sidecar, write_emulation_sidecar,
)
from models.emulation_cnn import SEQUENCE_LENGTH, CortexEmulationNet, TrainConfig
from models.sequence_artifacts import SequenceArtifactError, file_sha256, scorable_rows, sidecar_path
from tokenizer.emulation_tokenizer import EmulationTokenizer

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VOCAB = ["ntclose", "ntopenkey", "ldrloaddll", "regopenkeyexw", "ntcreatefile", "ntreadfile"]


def _net(use_padding_mask: bool, seed: int = 0, vocab_size: int = 8) -> CortexEmulationNet:
    torch.manual_seed(seed)
    return CortexEmulationNet(vocab_size=vocab_size, embed_dim=16, use_padding_mask=use_padding_mask).eval()


def _padded_batch(vocab_size: int = 8) -> torch.Tensor:
    g = torch.Generator().manual_seed(1)
    x = torch.randint(2, vocab_size, (7, SEQUENCE_LENGTH), generator=g)
    for i, n in enumerate((1, 10, 50, 250, 499, 500, 500)):
        x[i, n:] = 0  # <PAD>
    return x


# --------------------------------------------------------- forward paths
def test_paths_agree_without_padding_and_differ_with_it():
    masked, legacy = _net(True), _net(False)
    legacy.load_state_dict(masked.state_dict())
    full = torch.randint(2, 8, (3, SEQUENCE_LENGTH), generator=torch.Generator().manual_seed(2))
    padded = _padded_batch()[:4]
    with torch.no_grad():
        assert torch.allclose(masked(full), legacy(full), atol=1e-6)
        assert not torch.allclose(masked(padded), legacy(padded), atol=1e-4)


def test_legacy_path_matches_pre_masking_code_from_git(tmp_path, monkeypatch):
    try:
        src = subprocess.run(["git", "-C", _ROOT, "show", "b876d26^:models/emulation_cnn.py"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git history with b876d26 not available")
    path = tmp_path / "old_emulation_cnn.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("old_emulation_cnn", path)
    old = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "old_emulation_cnn", old)  # @dataclass looks it up
    spec.loader.exec_module(old)

    torch.manual_seed(0)
    ref = old.CortexEmulationNet(vocab_size=8, embed_dim=16).eval()
    legacy = _net(False)
    legacy.load_state_dict(ref.state_dict())
    x = _padded_batch()
    x[0] = 0  # include an all-padding row: the legacy path defines a score for it
    with torch.no_grad():
        assert torch.equal(legacy(x), ref(x))


def test_new_checkpoints_default_to_masking():
    assert TrainConfig(vocab_size=8).use_padding_mask is True
    assert CortexEmulationNet(vocab_size=8, embed_dim=16).use_padding_mask is True


# ------------------------------------------------------------- sidecar
def _artifacts(tmp_path, use_padding_mask=True):
    vocab_path = tmp_path / "vocab.json"
    EmulationTokenizer(_VOCAB).save(vocab_path)
    ckpt = tmp_path / "emu.pt"
    net = _net(use_padding_mask)
    torch.save(net.state_dict(), ckpt)
    write_emulation_sidecar(ckpt, vocab_path, use_padding_mask=use_padding_mask, embed_dim=16,
                            num_heads=4, vocab_size=8, model_code_commit="test")
    return ckpt, vocab_path, net


@pytest.mark.parametrize("use_mask", [True, False])
def test_loader_builds_recorded_architecture_in_eval_mode(tmp_path, use_mask):
    ckpt, vocab_path, net = _artifacts(tmp_path, use_mask)
    model, tok = load_emulation_model(ckpt, vocab_path)
    assert model.use_padding_mask is use_mask and model.training is False
    assert tok.vocab_size == 8
    x = _padded_batch()
    with torch.no_grad():
        assert torch.equal(model(x), net(x))


def test_training_writes_sidecar_with_masking_and_drops_all_padding_rows(tmp_path, caplog):
    vocab_path = tmp_path / "vocab.json"
    tok = EmulationTokenizer(_VOCAB)
    tok.save(vocab_path)
    rng = np.random.default_rng(0)
    X = rng.integers(2, tok.vocab_size, (32, SEQUENCE_LENGTH)).astype(np.int64)
    X[:, 60:] = 0
    X[0] = 0   # an empty trace in train
    X[1] = 0   # and another (val uses the same arrays)
    y = (np.arange(32) % 2).astype(np.float32)
    ckpt = tmp_path / "best.pt"
    cfg = TrainConfig(vocab_size=tok.vocab_size, embed_dim=16, epochs=1, batch_size=16)

    with caplog.at_level(logging.INFO, logger="cortex.emulation.train"):
        model, outcome = train_emulation.train(X, y, X, y, cfg, checkpoint_path=str(ckpt),
                                               vocab_path=str(vocab_path))

    assert any("dropping 2 all-padding" in r.message and "train" in r.message for r in caplog.records)
    assert all(np.isfinite([h.train_loss for h in outcome.history]))
    meta = read_emulation_sidecar(ckpt)
    assert meta["use_padding_mask"] is True
    assert meta["model_class"] == "models.emulation_cnn.CortexEmulationNet"
    assert meta["checkpoint_sha256"] == file_sha256(ckpt)
    assert meta["vocab_sha256"] == file_sha256(vocab_path)
    loaded, _ = load_emulation_model(ckpt, vocab_path)
    assert loaded.use_padding_mask is True


def test_training_refuses_checkpoint_without_vocab_path(tmp_path):
    X = np.ones((4, SEQUENCE_LENGTH), dtype=np.int64)
    y = np.zeros(4, dtype=np.float32)
    with pytest.raises(ValueError, match="vocab_path is required"):
        train_emulation.train(X, y, X, y, TrainConfig(vocab_size=8, embed_dim=16, epochs=1),
                              checkpoint_path=str(tmp_path / "x.pt"))


def test_missing_sidecar_is_a_clear_error(tmp_path):
    ckpt, vocab_path, _ = _artifacts(tmp_path)
    sidecar_path(ckpt).unlink()
    with pytest.raises(EmulationArtifactError, match="no sidecar") as exc:
        load_emulation_model(ckpt, vocab_path)
    assert "scripts/write_emulation_sidecar.py" in str(exc.value)


def test_checkpoint_sha_mismatch_fails_loudly(tmp_path):
    ckpt, vocab_path, _ = _artifacts(tmp_path)
    torch.save(_net(True, seed=99).state_dict(), ckpt)
    with pytest.raises(EmulationArtifactError, match="checkpoint .* sha256"):
        load_emulation_model(ckpt, vocab_path)


def test_vocab_sha_mismatch_fails_loudly(tmp_path):
    ckpt, vocab_path, _ = _artifacts(tmp_path)
    EmulationTokenizer(list(reversed(_VOCAB))).save(vocab_path)
    with pytest.raises(EmulationArtifactError, match="vocabulary .* sha256"):
        load_emulation_model(ckpt, vocab_path)


def test_sidecar_architecture_must_fit_weights(tmp_path):
    ckpt, vocab_path, _ = _artifacts(tmp_path)
    meta = json.loads(sidecar_path(ckpt).read_text())
    meta["embed_dim"] = 32
    sidecar_path(ckpt).write_text(json.dumps(meta))
    with pytest.raises(EmulationArtifactError, match="does not fit"):
        load_emulation_model(ckpt, vocab_path)


def test_behavioral_loader_rejects_an_emulation_sidecar(tmp_path):
    ckpt, vocab_path, _ = _artifacts(tmp_path)
    with pytest.raises(BehavioralArtifactError, match="is for models.emulation_cnn.CortexEmulationNet"):
        load_behavioral_model(ckpt, vocab_path)


def test_error_classes_share_a_base():
    assert issubclass(EmulationArtifactError, SequenceArtifactError)
    assert issubclass(BehavioralArtifactError, SequenceArtifactError)


# ------------------------------------------------- all-padding / no NaN
def test_masked_model_is_nan_on_all_padding_row_which_is_why_the_guard_exists():
    with torch.no_grad():
        assert torch.isnan(_net(True)(torch.zeros(1, SEQUENCE_LENGTH, dtype=torch.long))).all()


def test_scorable_rows():
    X = np.zeros((3, 5), dtype=np.int64)
    X[1, 0] = 1   # <UNK> is a real token
    X[2, 4] = 7
    assert scorable_rows(X).tolist() == [False, True, True]


_BEH = (train_behavioral.predict_proba_behavioral,
        lambda m: CortexBehavioralNet(vocab_size=8, embed_dim=16, use_padding_mask=m).eval(), 100)
_EMU = (train_emulation.predict_proba_emulation, lambda m: _net(m), SEQUENCE_LENGTH)


@pytest.mark.parametrize("predict,make,seq", [_BEH, _EMU], ids=["behavioral", "emulation"])
@pytest.mark.parametrize("use_mask", [True, False])
def test_scoring_helper_refuses_all_padding_rows(predict, make, seq, use_mask):
    X = np.full((3, seq), 3, dtype=np.int64)
    X[1] = 0
    with pytest.raises(ValueError, match="all-padding"):
        predict(make(use_mask), X)
    ok = predict(make(use_mask), X[scorable_rows(X)])
    assert ok.shape == (2,) and np.isfinite(ok).all()


@pytest.mark.parametrize("predict,make,seq", [_BEH, _EMU], ids=["behavioral", "emulation"])
def test_include_all_padding_only_for_legacy_models(predict, make, seq):
    X = np.zeros((2, seq), dtype=np.int64)
    X[0, :5] = 3
    raw = predict(make(False), X, include_all_padding=True)
    assert np.isfinite(raw).all()
    with pytest.raises(ValueError, match="only allowed for a legacy unmasked model"):
        predict(make(True), X, include_all_padding=True)


def test_behavioral_training_drops_all_padding_rows(tmp_path, caplog):
    from models.behavioral_cnn import TrainConfig as BTrainConfig
    rng = np.random.default_rng(0)
    X = rng.integers(2, 8, (32, 100)).astype(np.int64)
    X[0] = 0
    y = (np.arange(32) % 2).astype(np.float32)
    with caplog.at_level(logging.INFO, logger="cortex.behavioral.train"):
        train_behavioral.train(X, y, X, y, BTrainConfig(vocab_size=8, embed_dim=16, epochs=1, batch_size=16))
    assert any("dropping 1 all-padding" in r.message for r in caplog.records)


# ------------------------------------------- shipped checkpoint, end to end
_CKPT = os.path.join(_ROOT, "data/models/cortex_emulation_best.pt")
_VOCAB_FILE = os.path.join(_ROOT, "data/models/emulation_vocab.json")
_VAL = os.path.join(_ROOT, "data/processed/emulation_val.parquet")


@pytest.mark.slow
@pytest.mark.skipif(not all(os.path.exists(p) for p in (_CKPT, _VOCAB_FILE, _VAL, sidecar_path(_CKPT))),
                    reason="shipped emulation checkpoint / sidecar / val split absent")
def test_shipped_checkpoint_reproduces_recorded_val_numbers():
    """EVAL_ALL_MODELS_RESULTS.txt section 5, val: TN=440 FP=10 FN=66 TP=159,
    AUC 0.949472 at EMULATION_MALICIOUS_MIN."""
    import pandas as pd

    from inference.policy_engine import EMULATION_MALICIOUS_MIN
    from scripts.evaluate_all_models import binary_eval

    model, tok = load_emulation_model(_CKPT, _VOCAB_FILE)
    assert model.use_padding_mask is False  # pre-b876d26 checkpoint
    df = pd.read_parquet(_VAL)
    X, _ = tok.encode_batch([list(s) for s in df["api_names"]])
    y = df["label"].to_numpy(np.int32)
    assert scorable_rows(X).all()  # val has no empty trace
    proba = train_emulation.predict_proba_emulation(model, X)
    assert np.isfinite(proba).all()
    ev = binary_eval(y, proba, EMULATION_MALICIOUS_MIN)
    assert (ev.tn, ev.fp, ev.fn, ev.tp) == (440, 10, 66, 159)
    assert round(ev.auc_roc, 6) == 0.949472
