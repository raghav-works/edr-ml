"""
One-time back-fill of the JSON sidecar for a Cortex-Emulation checkpoint
that was trained before checkpoints carried one (the emulation counterpart
of docs/CODE_REVIEW.md F2; see scripts/write_behavioral_sidecar.py). New
checkpoints get their sidecar from scripts/train_emulation.py.

You must state the model code the checkpoint was trained with -- it cannot
be inferred from the weights (masking changed the forward pass, not the
parameter shapes). The script records that commit and the sha256 of
models/emulation_cnn.py AT that commit, checks that commit's code agrees
with --legacy-unmasked / --masked, and refuses to overwrite an existing
sidecar unless --force is given.

The shipped checkpoint (data/models/cortex_emulation_best.pt, mtime
2026-08-27 05:37 UTC) predates padding masking (b876d26, 2026-09-22). The
only earlier version of models/emulation_cnn.py is the one added in 9331159
(2026-08-27 10:13 UTC; the file's only other commit is b876d26), i.e. the
unmasked forward pass. Back-filled with:

    python -m scripts.write_emulation_sidecar \\
        --checkpoint data/models/cortex_emulation_best.pt \\
        --vocab data/models/emulation_vocab.json \\
        --legacy-unmasked --embed-dim 64 \\
        --model-code-commit 9331159d3dcec714618ee65adf2139e9d734c887
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from models.emulation_artifacts import sidecar_path, write_emulation_sidecar
from models.sequence_artifacts import model_code_at
from tokenizer.emulation_tokenizer import EmulationTokenizer

_REL = "models/emulation_cnn.py"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--embed-dim", type=int, required=True)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--model-code-commit", required=True,
                    help=f"full git commit of {_REL} the checkpoint was trained with")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--legacy-unmasked", action="store_true", help="trained before padding masking (b876d26)")
    mode.add_argument("--masked", action="store_true", help="trained with padding masking")
    ap.add_argument("--notes", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    out = sidecar_path(args.checkpoint)
    if out.exists() and not args.force:
        print(f"refusing to overwrite existing {out} (use --force)", file=sys.stderr)
        return 2
    use_mask = bool(args.masked)
    code = model_code_at(args.model_code_commit, _REL)
    if (b"key_padding_mask" in code) != use_mask:
        print(f"commit {args.model_code_commit}'s {_REL} {'has' if not use_mask else 'lacks'} padding "
              f"masking, contradicting {'--legacy-unmasked' if not use_mask else '--masked'}", file=sys.stderr)
        return 2
    path = write_emulation_sidecar(
        args.checkpoint, args.vocab,
        use_padding_mask=use_mask, embed_dim=args.embed_dim, num_heads=args.num_heads,
        vocab_size=EmulationTokenizer.load(args.vocab).vocab_size,
        model_code_commit=args.model_code_commit,
        model_code_sha256=hashlib.sha256(code).hexdigest(),
        notes=args.notes,
    )
    print(Path(path).read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
