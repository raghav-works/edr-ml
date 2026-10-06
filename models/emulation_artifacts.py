"""
Cortex-Emulation checkpoint metadata and the one supported loader -- the
same defect and fix as Cortex-Behavioral (docs/CODE_REVIEW.md F2;
models/behavioral_artifacts.py).

Commit b876d26 (2026-09-22) added padding masking to models/emulation_cnn.py
without changing any parameter shape. The shipped checkpoint
(cortex_emulation_best.pt, 2026-08-27) was trained with the earlier,
unmasked forward pass, so the current code silently scored it with a
different function (val TN/FP/FN/TP 440/10/66/159 -> 442/8/71/154) and
returned NaN for an empty trace. Every checkpoint therefore has a JSON
sidecar next to it:

    cortex_emulation_best.pt  ->  cortex_emulation_best.meta.json

in the shared format of models/sequence_artifacts.py. load_emulation_model()
is the only supported way to build a scoring model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from models import sequence_artifacts as _sa
from models.emulation_cnn import SEQUENCE_LENGTH, CortexEmulationNet
from models.sequence_artifacts import PathLike, sidecar_path  # noqa: F401  (re-exported)
from tokenizer.emulation_tokenizer import EmulationTokenizer

MODEL_CLASS = "models.emulation_cnn.CortexEmulationNet"
_MODEL_CODE_PATH = Path(__file__).resolve().with_name("emulation_cnn.py")
_HOW_TO_CREATE = (
    "New checkpoints get it from scripts/train_emulation.py; for an existing checkpoint "
    "create it once with scripts/write_emulation_sidecar.py (see docs/TECHNICAL_NOTES.md, "
    "'Emulation checkpoint sidecar')."
)


class EmulationArtifactError(_sa.SequenceArtifactError):
    """The checkpoint, its sidecar, or its vocabulary cannot be trusted."""


def write_emulation_sidecar(
    checkpoint_path: PathLike, vocab_path: PathLike, *,
    use_padding_mask: bool, embed_dim: int, num_heads: int, vocab_size: int,
    sequence_length: int = SEQUENCE_LENGTH,
    model_code_commit: Optional[str] = None,
    model_code_sha256: Optional[str] = None,
    notes: Optional[str] = None,
) -> Path:
    """Write <checkpoint>.meta.json. Defaults record the CURRENT model code
    (training); pass explicit values only when back-filling."""
    return _sa.write_sidecar(
        checkpoint_path, vocab_path, model_class=MODEL_CLASS, model_code_path=_MODEL_CODE_PATH,
        use_padding_mask=use_padding_mask, embed_dim=embed_dim, num_heads=num_heads,
        vocab_size=vocab_size, sequence_length=sequence_length,
        model_code_commit=model_code_commit, model_code_sha256=model_code_sha256, notes=notes,
    )


def read_emulation_sidecar(checkpoint_path: PathLike) -> dict:
    return _sa.read_sidecar(checkpoint_path, model_class=MODEL_CLASS,
                            error_cls=EmulationArtifactError, how_to_create=_HOW_TO_CREATE)


def load_emulation_model(checkpoint_path: PathLike, vocab_path: PathLike,
                         device: str = "cpu") -> tuple[CortexEmulationNet, EmulationTokenizer]:
    """The only supported way to build a scoring Cortex-Emulation model:
    sidecar required, checkpoint + vocabulary sha256 verified, architecture
    (incl. use_padding_mask) from the sidecar, strict weights_only load,
    returned in eval mode with its tokenizer."""
    meta = read_emulation_sidecar(checkpoint_path)
    _sa.verify_hashes(meta, checkpoint_path, vocab_path, EmulationArtifactError)
    tokenizer = EmulationTokenizer.load(vocab_path)
    if tokenizer.vocab_size != meta["vocab_size"]:
        raise EmulationArtifactError(
            f"vocabulary {vocab_path} has {tokenizer.vocab_size} tokens, sidecar says {meta['vocab_size']}")
    model = CortexEmulationNet(
        vocab_size=meta["vocab_size"], sequence_length=meta["sequence_length"],
        embed_dim=meta["embed_dim"], num_heads=meta["num_heads"],
        use_padding_mask=meta["use_padding_mask"],
    )
    _sa.load_state_strict(model, checkpoint_path, EmulationArtifactError)
    return model.to(device).eval(), tokenizer
