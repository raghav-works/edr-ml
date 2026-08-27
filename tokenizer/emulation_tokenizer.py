"""
API-call tokenizer for Cortex-Emulation.

Structurally identical to tokenizer/api_tokenizer.py (Cortex-Behavioral's
tokenizer) -- same fixed-vocabulary, fixed-length, <PAD>/<UNK> design --
but NOT reused directly: ApiTokenizer hardcodes MAX_SEQ_LEN=100 as a module
constant referenced inside encode()/encode_batch(), not a per-instance
parameter, so a longer sequence length can't be configured by just
instantiating it differently. Duplicated into its own module instead,
matching the project's existing convention of keeping per-model modules
self-contained (e.g. models/memory_lgbm.py duplicates rather than imports
static_lgbm.py's PlattCalibrator).

MAX_SEQ_LEN=500, not Behavioral's 100 -- checked against the real
module_entry-only, post-collapse distribution (scripts/split_emulation.py's
output), not assumed: p90-p99 all sit at 460-501 across train/val/test, so
500 covers the great majority of sequences without truncation. Truncation
strategy matches Behavioral's exactly (first N calls kept, not last-N or
sampled) for consistency, even though a small number of sequences here are
far more extreme than anything Behavioral saw -- e.g. one real train
sequence has 36,906 raw API calls (a tight repetitive loop; see
data/download_emulation.py's module docstring), which truncation reduces
to its first 500 calls, losing the tail. Accepted as a known limitation of
the fixed-length approach, not solved here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence

import numpy as np

MAX_SEQ_LEN = 500
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


class EmulationTokenizer:
    """Simple fixed vocabulary tokenizer over Windows API-call names,
    sized for Cortex-Emulation's much longer typical sequences and larger
    vocabulary (3,152 distinct raw API names in the real train split, vs.
    Behavioral's ~300-400) -- see the module docstring."""

    def __init__(self, vocab: Sequence[str] | None = None) -> None:
        base = [PAD_TOKEN, UNK_TOKEN] + list(vocab or [])
        seen = set()
        ordered = []
        for tok in base:
            if tok not in seen:
                seen.add(tok)
                ordered.append(tok)
        self.token_to_id = {tok: i for i, tok in enumerate(ordered)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @classmethod
    def build_from_corpus(cls, api_name_lists: Sequence[Sequence[str]], min_count: int = 1) -> "EmulationTokenizer":
        counts: dict[str, int] = {}
        for calls in api_name_lists:
            for name in calls:
                counts[name] = counts.get(name, 0) + 1
        vocab = sorted([name for name, c in counts.items() if c >= min_count],
                        key=lambda n: (-counts[n], n))
        return cls(vocab)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.id_to_token, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "EmulationTokenizer":
        id_to_token = {int(k): v for k, v in json.loads(Path(path).read_text()).items()}
        vocab = [id_to_token[i] for i in sorted(id_to_token) if id_to_token[i] not in (PAD_TOKEN, UNK_TOKEN)]
        return cls(vocab)

    # ------------------------------------------------------------------
    def encode(self, api_calls: Sequence[str]) -> tuple[np.ndarray, str]:
        """
        Returns (token_id_array of shape (500,) int64, sequence_status).
        sequence_status in {"empty", "insufficient", "ok", "truncated"}.
        """
        n = len(api_calls)
        if n == 0:
            status = "empty"
        elif n < MAX_SEQ_LEN:
            status = "insufficient"
        elif n == MAX_SEQ_LEN:
            status = "ok"
        else:
            status = "truncated"

        used = api_calls[:MAX_SEQ_LEN]
        ids = np.full(MAX_SEQ_LEN, self.token_to_id[PAD_TOKEN], dtype=np.int64)
        for i, name in enumerate(used):
            ids[i] = self.token_to_id.get(name, self.token_to_id[UNK_TOKEN])
        return ids, status

    def encode_batch(self, batch: Sequence[Sequence[str]]) -> tuple[np.ndarray, List[str]]:
        out = np.zeros((len(batch), MAX_SEQ_LEN), dtype=np.int64)
        statuses = []
        for i, calls in enumerate(batch):
            ids, status = self.encode(calls)
            out[i] = ids
            statuses.append(status)
        return out, statuses
