"""
Adapts an EMBER2024 dataset record (as downloaded by data/download_ember2024.py)
into the 2568-dim float32 vector `scripts/train_static.py` trains on.

EMBER2024's raw export and features/pe_features.py's raw_features() output
are, group for group, the same shape (see the alignment work in
pe_features.py's module docstring: EMBER2024's "general"/"header"/"section"/
"imports"/"exports"/"datadirectories"/"richheader"/"authenticode"/"strings"/
"pefilewarnings" fields were confirmed field-for-field against the real
EMBER2024/thrember reference extractor). So this adapter does not
re-derive features -- it just hands each group's already-computed dict/list
straight to PEFeatureExtractor's *existing* process_raw_features() group
logic, keeping exactly one implementation of the vectorization formulas
(pe_features.py) for both the "parse live PE bytes" and "read an EMBER2024
training record" paths.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from features.pe_features import EMBER2024_FEATURE_COUNT, PEFeatureExtractor

_EXTRACTOR = PEFeatureExtractor()

# EMBER2024 record fields -> pe_features.py group name (identical for every
# group except that EMBER2024's "histogram"/"byteentropy" need no renaming
# either -- this map exists so a future field rename on either side is a
# one-line fix instead of a silent KeyError).
_GROUP_FIELDS = [
    "general", "histogram", "byteentropy", "strings", "header", "section",
    "imports", "exports", "datadirectories", "richheader", "authenticode", "pefilewarnings",
]


def record_to_vector(record: Dict[str, Any]) -> np.ndarray:
    """Vectorize one EMBER2024 record into the 2568-dim float32 feature vector."""
    raw = {field: record[field] for field in _GROUP_FIELDS}
    vec = _EXTRACTOR.process_raw_features(raw)
    assert vec.shape == (EMBER2024_FEATURE_COUNT,), f"unexpected vector shape {vec.shape}"
    return vec
