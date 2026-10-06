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
"""
from __future__ import annotations

import datetime
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Optional, Union

import torch

from models.behavioral_cnn import SEQUENCE_LENGTH, CortexBehavioralNet
from tokenizer.api_tokenizer import ApiTokenizer

SIDECAR_FORMAT_VERSION = 1
_MODEL_CODE_PATH = Path(__file__).resolve().with_name("behavioral_cnn.py")
_REQUIRED = {
    "format_version": int, "use_padding_mask": bool, "embed_dim": int, "num_heads": int,
    "vocab_size": int, "sequence_length": int, "checkpoint_sha256": str, "vocab_sha256": str,
    "model_code_commit": str,
}

PathLike = Union[str, Path]


class BehavioralArtifactError(RuntimeError):
    """The checkpoint, its sidecar, or its vocabulary cannot be trusted."""


def sidecar_path(checkpoint_path: PathLike) -> Path:
    return Path(checkpoint_path).with_suffix(".meta.json")


def file_sha256(path: PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def current_model_code_commit() -> str:
    """git HEAD of this repo, suffixed '+dirty' if models/behavioral_cnn.py
    has uncommitted changes; 'unknown' if git is unavailable."""
    repo = _MODEL_CODE_PATH.parents[1]
    try:
        head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--", str(_MODEL_CODE_PATH)],
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return head + ("+dirty" if dirty else "")


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
    meta = {
        "format_version": SIDECAR_FORMAT_VERSION,
        "model_class": "models.behavioral_cnn.CortexBehavioralNet",
        "use_padding_mask": bool(use_padding_mask),
        "embed_dim": int(embed_dim),
        "num_heads": int(num_heads),
        "vocab_size": int(vocab_size),
        "sequence_length": int(sequence_length),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "vocab_sha256": file_sha256(vocab_path),
        "model_code_commit": model_code_commit or current_model_code_commit(),
        "model_code_sha256": model_code_sha256 or file_sha256(_MODEL_CODE_PATH),
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "notes": notes,
    }
    out = sidecar_path(checkpoint_path)
    out.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return out


def read_behavioral_sidecar(checkpoint_path: PathLike) -> dict:
    path = sidecar_path(checkpoint_path)
    if not path.is_file():
        raise BehavioralArtifactError(
            f"no sidecar {path} for behavioral checkpoint {checkpoint_path}. A checkpoint "
            "does not record which forward pass it was trained with, so it will not be "
            "loaded without one. New checkpoints get it from scripts/train_behavioral.py; "
            "for an existing checkpoint create it once with scripts/write_behavioral_sidecar.py "
            "(see docs/TECHNICAL_NOTES.md, 'Behavioral checkpoint sidecar')."
        )
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BehavioralArtifactError(f"unreadable sidecar {path}: {exc}") from exc
    if not isinstance(meta, dict):
        raise BehavioralArtifactError(f"sidecar {path} is not a JSON object")
    for key, typ in _REQUIRED.items():
        if key not in meta:
            raise BehavioralArtifactError(f"sidecar {path} is missing {key!r}")
        if not isinstance(meta[key], typ) or (typ is int and isinstance(meta[key], bool)):
            raise BehavioralArtifactError(f"sidecar {path}: {key!r} must be {typ.__name__}, got {meta[key]!r}")
    if meta["format_version"] != SIDECAR_FORMAT_VERSION:
        raise BehavioralArtifactError(f"sidecar {path}: unsupported format_version {meta['format_version']}")
    return meta


def load_behavioral_model(checkpoint_path: PathLike, vocab_path: PathLike,
                          device: str = "cpu") -> tuple[CortexBehavioralNet, ApiTokenizer]:
    """The only supported way to build a scoring Cortex-Behavioral model.

    Reads the sidecar, verifies the checkpoint and vocabulary sha256 against
    it, builds CortexBehavioralNet with the recorded settings (including
    use_padding_mask), loads the weights strictly, and returns
    (model in eval mode, tokenizer for that vocabulary)."""
    meta = read_behavioral_sidecar(checkpoint_path)
    for label, path, key in (("checkpoint", checkpoint_path, "checkpoint_sha256"),
                             ("vocabulary", vocab_path, "vocab_sha256")):
        actual = file_sha256(path)
        if actual != meta[key]:
            raise BehavioralArtifactError(
                f"{label} {path} sha256 {actual} does not match {sidecar_path(checkpoint_path)} "
                f"{key} {meta[key]} -- wrong file, or the file changed after the sidecar was written"
            )
    tokenizer = ApiTokenizer.load(vocab_path)
    if tokenizer.vocab_size != meta["vocab_size"]:
        raise BehavioralArtifactError(
            f"vocabulary {vocab_path} has {tokenizer.vocab_size} tokens, sidecar says {meta['vocab_size']}")
    model = CortexBehavioralNet(
        vocab_size=meta["vocab_size"], sequence_length=meta["sequence_length"],
        embed_dim=meta["embed_dim"], num_heads=meta["num_heads"],
        use_padding_mask=meta["use_padding_mask"],
    )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise BehavioralArtifactError(
            f"checkpoint {checkpoint_path} does not fit the architecture in its sidecar: {exc}") from exc
    return model.to(device).eval(), tokenizer
