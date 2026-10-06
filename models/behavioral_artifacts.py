"""
Cortex-Behavioral checkpoint metadata and the one supported loader
(docs/CODE_REVIEW.md F2).

A checkpoint is a raw state_dict. It does not say which forward pass it was
trained with, and commit 76c3534 changed the forward pass (padding masking)
without changing any parameter shape -- so an old checkpoint loaded silently
into the new code and scored with a different function (benign FP 1 -> 19 on
val+test). Every checkpoint therefore has a JSON sidecar next to it:

    cortex_behavioral_best.pt  ->  cortex_behavioral_best.meta.json

recording the architecture switches it needs and the sha256 of both the
checkpoint and the vocabulary it was trained with. load_behavioral_model()
is the only supported way to build a scoring model: it refuses to load
without a sidecar and fails loudly on any hash or shape mismatch. No pickle.
The sidecar format and checks are shared with Cortex-Emulation
(models/sequence_artifacts.py).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from models import sequence_artifacts as _sa
from models.behavioral_cnn import SEQUENCE_LENGTH, CortexBehavioralNet
from models.sequence_artifacts import (  # noqa: F401  (re-exported)
    SIDECAR_FORMAT_VERSION, PathLike, file_sha256, sidecar_path,
)
from tokenizer.api_tokenizer import ApiTokenizer

MODEL_CLASS = "models.behavioral_cnn.CortexBehavioralNet"
_MODEL_CODE_PATH = Path(__file__).resolve().with_name("behavioral_cnn.py")
_HOW_TO_CREATE = (
    "New checkpoints get it from scripts/train_behavioral.py; for an existing checkpoint "
    "create it once with scripts/write_behavioral_sidecar.py (see docs/TECHNICAL_NOTES.md, "
    "'Behavioral checkpoint sidecar')."
)


class BehavioralArtifactError(_sa.SequenceArtifactError):
    """The checkpoint, its sidecar, or its vocabulary cannot be trusted."""


def current_model_code_commit() -> str:
    """git HEAD of this repo, suffixed '+dirty' if models/behavioral_cnn.py
    has uncommitted changes; 'unknown' if git is unavailable."""
    return _sa.current_model_code_commit(_MODEL_CODE_PATH)


def write_behavioral_sidecar(
    checkpoint_path: PathLike, vocab_path: PathLike, *,
    use_padding_mask: bool, embed_dim: int, num_heads: int, vocab_size: int,
    sequence_length: int = SEQUENCE_LENGTH,
    model_code_commit: Optional[str] = None,
    model_code_sha256: Optional[str] = None,
    notes: Optional[str] = None,
) -> Path:
    """Write <checkpoint>.meta.json for the checkpoint as it is on disk now.
    Defaults record the CURRENT model code (training); pass explicit values
    only when back-filling a sidecar for an older checkpoint."""
    return _sa.write_sidecar(
        checkpoint_path, vocab_path, model_class=MODEL_CLASS, model_code_path=_MODEL_CODE_PATH,
        use_padding_mask=use_padding_mask, embed_dim=embed_dim, num_heads=num_heads,
        vocab_size=vocab_size, sequence_length=sequence_length,
        model_code_commit=model_code_commit, model_code_sha256=model_code_sha256, notes=notes,
    )


def read_behavioral_sidecar(checkpoint_path: PathLike) -> dict:
    return _sa.read_sidecar(checkpoint_path, model_class=MODEL_CLASS,
                            error_cls=BehavioralArtifactError, how_to_create=_HOW_TO_CREATE)


def load_behavioral_model(checkpoint_path: PathLike, vocab_path: PathLike,
                          device: str = "cpu") -> tuple[CortexBehavioralNet, ApiTokenizer]:
    """The only supported way to build a scoring Cortex-Behavioral model.

    Reads the sidecar, verifies the checkpoint and vocabulary sha256 against
    it, builds CortexBehavioralNet with the recorded settings (including
    use_padding_mask), loads the weights strictly, and returns
    (model in eval mode, tokenizer for that vocabulary)."""
    meta = read_behavioral_sidecar(checkpoint_path)
    _sa.verify_hashes(meta, checkpoint_path, vocab_path, BehavioralArtifactError)
    tokenizer = ApiTokenizer.load(vocab_path)
    if tokenizer.vocab_size != meta["vocab_size"]:
        raise BehavioralArtifactError(
            f"vocabulary {vocab_path} has {tokenizer.vocab_size} tokens, sidecar says {meta['vocab_size']}")
    model = CortexBehavioralNet(
        vocab_size=meta["vocab_size"], sequence_length=meta["sequence_length"],
        embed_dim=meta["embed_dim"], num_heads=meta["num_heads"],
        use_padding_mask=meta["use_padding_mask"],
    )
    _sa.load_state_strict(model, checkpoint_path, BehavioralArtifactError)
    return model.to(device).eval(), tokenizer
