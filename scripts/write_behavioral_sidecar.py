"""
One-time back-fill of the JSON sidecar for a Cortex-Behavioral checkpoint
that was trained before checkpoints carried one (docs/CODE_REVIEW.md F2).
New checkpoints get their sidecar from scripts/train_behavioral.py; do not
use this for them.

You must state the model code the checkpoint was trained with -- it cannot
be inferred from the weights (masking changed the forward pass, not the
parameter shapes). The script records that commit and the sha256 of
models/behavioral_cnn.py AT that commit (via `git show`), and refuses to
overwrite an existing sidecar unless --force is given.

The shipped checkpoint (data/models/cortex_behavioral_best.pt, mtime
2026-08-19 05:25 UTC) predates padding masking (76c3534, 2026-09-22); its
model code is blob 1fe9c28 of models/behavioral_cnn.py, committed in
a035aa5 (2026-08-19 05:56 UTC). Back-filled with:

    python -m scripts.write_behavioral_sidecar \\
        --checkpoint data/models/cortex_behavioral_best.pt \\
        --vocab data/models/api_vocab.json \\
        --legacy-unmasked --embed-dim 128 \\
        --model-code-commit a035aa52cb9b755c323c0a915e72bddd6e068533
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

from models.behavioral_artifacts import sidecar_path, write_behavioral_sidecar
from tokenizer.api_tokenizer import ApiTokenizer


def _model_code_sha256_at(commit: str) -> str:
    blob = subprocess.run(["git", "show", f"{commit}:models/behavioral_cnn.py"],
                          capture_output=True, check=True).stdout
    return hashlib.sha256(blob).hexdigest()


def _has_padding_mask(commit: str) -> bool:
    src = subprocess.run(["git", "show", f"{commit}:models/behavioral_cnn.py"],
                         capture_output=True, text=True, check=True).stdout
    return "key_padding_mask" in src


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--embed-dim", type=int, required=True)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--model-code-commit", required=True,
                    help="full git commit of models/behavioral_cnn.py the checkpoint was trained with")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--legacy-unmasked", action="store_true",
                      help="trained before padding masking (76c3534)")
    mode.add_argument("--masked", action="store_true", help="trained with padding masking")
    ap.add_argument("--notes", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    out = sidecar_path(args.checkpoint)
    if out.exists() and not args.force:
        print(f"refusing to overwrite existing {out} (use --force)", file=sys.stderr)
        return 2
    use_mask = bool(args.masked)
    if _has_padding_mask(args.model_code_commit) != use_mask:
        print(f"commit {args.model_code_commit}'s models/behavioral_cnn.py "
              f"{'has' if not use_mask else 'lacks'} padding masking, contradicting "
              f"{'--legacy-unmasked' if not use_mask else '--masked'}", file=sys.stderr)
        return 2
    path = write_behavioral_sidecar(
        args.checkpoint, args.vocab,
        use_padding_mask=use_mask, embed_dim=args.embed_dim, num_heads=args.num_heads,
        vocab_size=ApiTokenizer.load(args.vocab).vocab_size,
        model_code_commit=args.model_code_commit,
        model_code_sha256=_model_code_sha256_at(args.model_code_commit),
        notes=args.notes,
    )
    print(Path(path).read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
