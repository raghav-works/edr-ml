"""
Shared checkpoint-sidecar code for the two token-sequence models,
Cortex-Behavioral and Cortex-Emulation (docs/CODE_REVIEW.md F2 and its
emulation follow-up).

Both models are raw state_dicts whose forward pass changed (padding masking:
behavioral 76c3534, emulation b876d26) without any parameter shape changing,
so a checkpoint loads silently into the wrong forward pass. Each checkpoint
therefore has a JSON sidecar next to it (<name>.pt -> <name>.meta.json)
recording use_padding_mask, the architecture, and the sha256 of the
checkpoint and of its vocabulary. The per-model modules
(models/behavioral_artifacts.py, models/emulation_artifacts.py) supply the
model class and tokenizer; the format and the checks live here.

Also here: the all-padding guard. A row with no real token (an empty trace)
has no defined score under a masked checkpoint -- every attention key is
masked, the softmax row is all -inf, and the output is NaN. Scoring helpers
refuse such rows; callers report them as PENDING instead.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

SIDECAR_FORMAT_VERSION = 1
PAD_ID = 0
_REPO = Path(__file__).resolve().parents[1]
_REQUIRED = {
    "format_version": int, "use_padding_mask": bool, "embed_dim": int, "num_heads": int,
    "vocab_size": int, "sequence_length": int, "checkpoint_sha256": str, "vocab_sha256": str,
    "model_code_commit": str,
}

PathLike = Union[str, Path]


class SequenceArtifactError(RuntimeError):
    """A sequence-model checkpoint, its sidecar, or its vocabulary cannot be trusted."""


def sidecar_path(checkpoint_path: PathLike) -> Path:
    return Path(checkpoint_path).with_suffix(".meta.json")


def file_sha256(path: PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def current_model_code_commit(model_code_path: Path) -> str:
    """git HEAD of this repo, suffixed '+dirty' if `model_code_path` has
    uncommitted changes; 'unknown' if git is unavailable."""
    try:
        head = subprocess.run(["git", "-C", str(_REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(_REPO), "status", "--porcelain", "--", str(model_code_path)],
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return head + ("+dirty" if dirty else "")


def model_code_at(commit: str, rel_path: str) -> bytes:
    """The bytes of `rel_path` at `commit` (for back-filling a sidecar)."""
    return subprocess.run(["git", "-C", str(_REPO), "show", f"{commit}:{rel_path}"],
                          capture_output=True, check=True).stdout


def write_sidecar(
    checkpoint_path: PathLike, vocab_path: PathLike, *,
    model_class: str, model_code_path: Path,
    use_padding_mask: bool, embed_dim: int, num_heads: int, vocab_size: int, sequence_length: int,
    model_code_commit: Optional[str] = None,
    model_code_sha256: Optional[str] = None,
    notes: Optional[str] = None,
) -> Path:
    """Write <checkpoint>.meta.json for the checkpoint as it is on disk now.
    Defaults record the CURRENT model code (training); pass explicit values
    only when back-filling a sidecar for an older checkpoint."""
    meta = {
        "format_version": SIDECAR_FORMAT_VERSION,
        "model_class": model_class,
        "use_padding_mask": bool(use_padding_mask),
        "embed_dim": int(embed_dim),
        "num_heads": int(num_heads),
        "vocab_size": int(vocab_size),
        "sequence_length": int(sequence_length),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "vocab_sha256": file_sha256(vocab_path),
        "model_code_commit": model_code_commit or current_model_code_commit(model_code_path),
        "model_code_sha256": model_code_sha256 or file_sha256(model_code_path),
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "notes": notes,
    }
    out = sidecar_path(checkpoint_path)
    out.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return out


def read_sidecar(checkpoint_path: PathLike, *, model_class: str, error_cls: type,
                 how_to_create: str) -> dict:
    path = sidecar_path(checkpoint_path)
    if not path.is_file():
        raise error_cls(
            f"no sidecar {path} for checkpoint {checkpoint_path}. A checkpoint does not "
            "record which forward pass it was trained with, so it will not be loaded "
            f"without one. {how_to_create}"
        )
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise error_cls(f"unreadable sidecar {path}: {exc}") from exc
    if not isinstance(meta, dict):
        raise error_cls(f"sidecar {path} is not a JSON object")
    for key, typ in _REQUIRED.items():
        if key not in meta:
            raise error_cls(f"sidecar {path} is missing {key!r}")
        if not isinstance(meta[key], typ) or (typ is int and isinstance(meta[key], bool)):
            raise error_cls(f"sidecar {path}: {key!r} must be {typ.__name__}, got {meta[key]!r}")
    if meta["format_version"] != SIDECAR_FORMAT_VERSION:
        raise error_cls(f"sidecar {path}: unsupported format_version {meta['format_version']}")
    if meta.get("model_class", model_class) != model_class:
        raise error_cls(f"sidecar {path} is for {meta['model_class']}, not {model_class}")
    return meta


def verify_hashes(meta: dict, checkpoint_path: PathLike, vocab_path: PathLike, error_cls: type) -> None:
    for label, path, key in (("checkpoint", checkpoint_path, "checkpoint_sha256"),
                             ("vocabulary", vocab_path, "vocab_sha256")):
        actual = file_sha256(path)
        if actual != meta[key]:
            raise error_cls(
                f"{label} {path} sha256 {actual} does not match {sidecar_path(checkpoint_path)} "
                f"{key} {meta[key]} -- wrong file, or the file changed after the sidecar was written"
            )


def load_state_strict(model: torch.nn.Module, checkpoint_path: PathLike, error_cls: type) -> None:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise error_cls(
            f"checkpoint {checkpoint_path} does not fit the architecture in its sidecar: {exc}") from exc


# ------------------------------------------------------------ all-padding
def scorable_rows(X: np.ndarray, pad_id: int = PAD_ID) -> np.ndarray:
    """Boolean mask: rows with at least one non-<PAD> token. An all-padding
    row is an empty trace -- never scored, reported as PENDING."""
    X = np.asarray(X)
    return (X != pad_id).any(axis=1)


def check_scorable(X: np.ndarray, model: torch.nn.Module, include_all_padding: bool = False) -> None:
    """Raise ValueError if X has an all-padding row, unless
    include_all_padding=True AND the model is a legacy unmasked one (which
    gives such a row a finite, if meaningless, score -- used only for the
    'model-raw' reporting line that reproduces pre-masking records)."""
    n_bad = int((~scorable_rows(X)).sum())
    if n_bad == 0:
        return
    if include_all_padding:
        if getattr(model, "use_padding_mask", True):
            raise ValueError("include_all_padding=True is only allowed for a legacy unmasked model "
                             "(use_padding_mask=False); a masked model returns NaN for an all-padding row")
        return
    raise ValueError(f"{n_bad} all-padding (empty-trace) row(s) passed for scoring; they have no "
                     "defined score -- filter with scorable_rows() and report them as PENDING")
