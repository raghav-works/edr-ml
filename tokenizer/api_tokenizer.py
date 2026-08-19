"""
API-call tokenizer for Cortex-Behavioral.

Matches the architecture contract exactly:
    - input: ordered list of Windows API-call names (e.g. from a JSON trace)
    - only the first MAX_SEQ_LEN (100) calls are used; extra calls are ignored
    - unknown API names map to <UNK>
    - 0 calls -> PENDING (caller decides), 1..99 -> PENDING, 100+ -> score
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence

import numpy as np

MAX_SEQ_LEN = 100
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


class ApiTokenizer:
    """Simple fixed vocabulary tokenizer over Windows API-call names."""

    def __init__(self, vocab: Sequence[str] | None = None) -> None:
        base = [PAD_TOKEN, UNK_TOKEN] + list(vocab or [])
        # de-dup while preserving order
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
    def build_from_corpus(cls, api_name_lists: Sequence[Sequence[str]], min_count: int = 1) -> "ApiTokenizer":
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
    def load(cls, path: str | Path) -> "ApiTokenizer":
        id_to_token = {int(k): v for k, v in json.loads(Path(path).read_text()).items()}
        vocab = [id_to_token[i] for i in sorted(id_to_token) if id_to_token[i] not in (PAD_TOKEN, UNK_TOKEN)]
        return cls(vocab)

    # ------------------------------------------------------------------
    def encode(self, api_calls: Sequence[str]) -> tuple[np.ndarray, str]:
        """
        Returns (token_id_array of shape (100,) int64, sequence_status).
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


def load_api_calls_json(path: str | Path) -> List[str]:
    """Load and validate an API-call JSON file: must be a flat list of strings."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise ValueError(f"{path} must contain a JSON list of strings (API call names)")
    return data
