"""
docs/CODE_REVIEW.md F2 -- a behavioral checkpoint always runs with the model
code it was trained with.

  * CortexBehavioralNet(use_padding_mask=False) is the pre-76c3534 forward
    pass; True is the masked one.
  * every checkpoint has a JSON sidecar (models/behavioral_artifacts.py);
    training writes it, the loader refuses to run without it and fails
    loudly on a checkpoint/vocab sha256 mismatch.
  * end to end: the shipped checkpoint through the loader + CortexPipeline
    reproduces the recorded val confusion matrix (skipped if absent).
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from models.behavioral_artifacts import (
    BehavioralArtifactError, file_sha256, load_behavioral_model, read_behavioral_sidecar,
    sidecar_path, write_behavioral_sidecar,
)
from models.behavioral_cnn import CortexBehavioralNet, TrainConfig
from models.train_behavioral import train
from tokenizer.api_tokenizer import ApiTokenizer

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VOCAB = ["ntclose", "ntopenkey", "ldrloaddll", "regopenkeyexw", "ntcreatefile", "ntreadfile"]


def _net(use_padding_mask: bool, seed: int = 0, vocab_size: int = 8) -> CortexBehavioralNet:
    torch.manual_seed(seed)
    return CortexBehavioralNet(vocab_size=vocab_size, embed_dim=16, use_padding_mask=use_padding_mask).eval()


def _padded_batch(vocab_size: int = 8) -> torch.Tensor:
    g = torch.Generator().manual_seed(1)
    x = torch.randint(2, vocab_size, (6, 100), generator=g)
    for i, n in enumerate((10, 25, 50, 75, 99, 100)):
        x[i, n:] = 0  # <PAD>
    return x


# --------------------------------------------------------- forward paths
def test_paths_agree_without_padding_and_differ_with_it():
    masked, legacy = _net(True), _net(False)
    legacy.load_state_dict(masked.state_dict())
    full = torch.randint(2, 8, (4, 100), generator=torch.Generator().manual_seed(2))
    padded = _padded_batch()
    with torch.no_grad():
        assert torch.allclose(masked(full), legacy(full), atol=1e-6)  # no <PAD>: masking is a no-op
        assert not torch.allclose(masked(padded[:5]), legacy(padded[:5]), atol=1e-4)


def test_legacy_path_matches_pre_masking_code_from_git(tmp_path, monkeypatch):
    try:
        src = subprocess.run(["git", "-C", _ROOT, "show", "76c3534^:models/behavioral_cnn.py"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git history with 76c3534 not available")
    path = tmp_path / "old_behavioral_cnn.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("old_behavioral_cnn", path)
    old = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "old_behavioral_cnn", old)  # @dataclass looks it up
    spec.loader.exec_module(old)

    torch.manual_seed(0)
    ref = old.CortexBehavioralNet(vocab_size=8, embed_dim=16).eval()
    legacy = _net(False)
    legacy.load_state_dict(ref.state_dict())
    x = _padded_batch()
    with torch.no_grad():
        assert torch.equal(legacy(x), ref(x))


def test_new_checkpoints_default_to_masking():
    assert TrainConfig(vocab_size=8).use_padding_mask is True
    assert CortexBehavioralNet(vocab_size=8, embed_dim=16).use_padding_mask is True


# ------------------------------------------------------------- sidecar
def _artifacts(tmp_path, use_padding_mask=True):
    vocab_path = tmp_path / "vocab.json"
    ApiTokenizer(_VOCAB).save(vocab_path)
    ckpt = tmp_path / "model.pt"
    net = _net(use_padding_mask)
    torch.save(net.state_dict(), ckpt)
    write_behavioral_sidecar(ckpt, vocab_path, use_padding_mask=use_padding_mask, embed_dim=16,
                             num_heads=4, vocab_size=8, model_code_commit="test")
    return ckpt, vocab_path, net


def test_sidecar_path_is_next_to_checkpoint(tmp_path):
    assert sidecar_path(tmp_path / "cortex_behavioral_best.pt") == tmp_path / "cortex_behavioral_best.meta.json"


@pytest.mark.parametrize("use_mask", [True, False])
def test_loader_builds_recorded_architecture_in_eval_mode(tmp_path, use_mask):
    ckpt, vocab_path, net = _artifacts(tmp_path, use_mask)
    model, tok = load_behavioral_model(ckpt, vocab_path)
    assert model.use_padding_mask is use_mask
    assert model.training is False
    assert tok.vocab_size == 8
    x = _padded_batch()
    with torch.no_grad():
        assert torch.equal(model(x), net(x))


def test_training_writes_sidecar_with_masking(tmp_path):
    vocab_path = tmp_path / "vocab.json"
    tok = ApiTokenizer(_VOCAB)
    tok.save(vocab_path)
    rng = np.random.default_rng(0)
    X = rng.integers(2, tok.vocab_size, (32, 100)).astype(np.int64)
    X[:, 60:] = 0
    y = (np.arange(32) % 2).astype(np.float32)
    ckpt = tmp_path / "best.pt"
    cfg = TrainConfig(vocab_size=tok.vocab_size, embed_dim=16, epochs=1, batch_size=16)

    train(X, y, X, y, cfg, checkpoint_path=str(ckpt), vocab_path=str(vocab_path))

    meta = read_behavioral_sidecar(ckpt)
    assert meta["use_padding_mask"] is True
    assert meta["embed_dim"] == 16 and meta["vocab_size"] == tok.vocab_size
    assert meta["checkpoint_sha256"] == file_sha256(ckpt)
    assert meta["vocab_sha256"] == file_sha256(vocab_path)
    assert meta["model_code_commit"]
    model, _ = load_behavioral_model(ckpt, vocab_path)  # round-trips
    assert model.use_padding_mask is True


def test_training_refuses_checkpoint_without_vocab_path(tmp_path):
    X = np.zeros((4, 100), dtype=np.int64)
    y = np.zeros(4, dtype=np.float32)
    with pytest.raises(ValueError, match="vocab_path is required"):
        train(X, y, X, y, TrainConfig(vocab_size=8, embed_dim=16, epochs=1),
              checkpoint_path=str(tmp_path / "x.pt"))


def test_missing_sidecar_is_a_clear_error(tmp_path):
    ckpt, vocab_path, _ = _artifacts(tmp_path)
    sidecar_path(ckpt).unlink()
    with pytest.raises(BehavioralArtifactError, match="no sidecar") as exc:
        load_behavioral_model(ckpt, vocab_path)
    assert "scripts/write_behavioral_sidecar.py" in str(exc.value)


def test_checkpoint_sha_mismatch_fails_loudly(tmp_path):
    ckpt, vocab_path, _ = _artifacts(tmp_path)
    torch.save(_net(True, seed=99).state_dict(), ckpt)  # different weights, same name
    with pytest.raises(BehavioralArtifactError, match="checkpoint .* sha256"):
        load_behavioral_model(ckpt, vocab_path)


def test_vocab_sha_mismatch_fails_loudly(tmp_path):
    ckpt, vocab_path, _ = _artifacts(tmp_path)
    ApiTokenizer(list(reversed(_VOCAB))).save(vocab_path)  # same size, different ids
    with pytest.raises(BehavioralArtifactError, match="vocabulary .* sha256"):
        load_behavioral_model(ckpt, vocab_path)


@pytest.mark.parametrize("key,value", [("use_padding_mask", "false"), ("embed_dim", True),
                                       ("format_version", 2)])
def test_malformed_sidecar_fails_loudly(tmp_path, key, value):
    ckpt, vocab_path, _ = _artifacts(tmp_path)
    meta = json.loads(sidecar_path(ckpt).read_text())
    meta[key] = value
    sidecar_path(ckpt).write_text(json.dumps(meta))
    with pytest.raises(BehavioralArtifactError):
        load_behavioral_model(ckpt, vocab_path)


def test_sidecar_architecture_must_fit_weights(tmp_path):
    ckpt, vocab_path, _ = _artifacts(tmp_path)
    meta = json.loads(sidecar_path(ckpt).read_text())
    meta["embed_dim"] = 32
    sidecar_path(ckpt).write_text(json.dumps(meta))
    with pytest.raises(BehavioralArtifactError, match="does not fit"):
        load_behavioral_model(ckpt, vocab_path)


# ------------------------------------------- shipped checkpoint, end to end
_CKPT = os.path.join(_ROOT, "data/models/cortex_behavioral_best.pt")
_VOCAB_FILE = os.path.join(_ROOT, "data/models/api_vocab.json")
_VAL = os.path.join(_ROOT, "data/processed/behavioral_val.parquet")
_PE = os.path.join(_ROOT, "tests/fixtures/pe_samples/sample_cli64.exe")


@pytest.mark.slow
@pytest.mark.skipif(not all(os.path.exists(p) for p in (_CKPT, _VOCAB_FILE, _VAL, sidecar_path(_CKPT))),
                    reason="shipped behavioral checkpoint / sidecar / val split absent")
def test_shipped_checkpoint_reproduces_recorded_val_confusion_matrix(tmp_path):
    """EVAL_ALL_MODELS_RESULTS.txt section 4, val, 'TOTAL (deployment:
    short+ok)': TN=132 FP=1 FN=20 TP=758 at BEHAVIORAL_MALICIOUS_MIN."""
    import pandas as pd

    from inference.pipeline import CortexPipeline
    from inference.policy_engine import BehavioralVerdict

    class _BenignStatic:
        def predict_proba(self, X, *_a, **_k):
            return np.full(len(X), 0.01)

    model, tok = load_behavioral_model(_CKPT, _VOCAB_FILE)
    assert model.use_padding_mask is False  # pre-76c3534 checkpoint
    pipe = CortexPipeline(static_model=_BenignStatic(), behavioral_model=model, tokenizer=tok, self_test=False)

    df = pd.read_parquet(_VAL)
    tn = fp = fn = tp = pending = 0
    trace = tmp_path / "trace.json"
    for calls, label in zip(df["api_calls"], df["label"]):
        trace.write_text(json.dumps(list(calls)))
        v = pipe.scan(_PE, api_calls_json_path=str(trace)).behavioral_verdict
        if v == BehavioralVerdict.PENDING:
            pending += 1
            continue
        mal = v == BehavioralVerdict.MALICIOUS
        tp += mal and label == 1
        fn += (not mal) and label == 1
        fp += mal and label == 0
        tn += (not mal) and label == 0
    assert (tn, fp, fn, tp) == (132, 1, 20, 758)
    assert pending == 7  # the too_short band
