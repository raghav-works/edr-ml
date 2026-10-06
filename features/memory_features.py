"""
Cortex-Memory: feature engineering over the 55 raw VolMemLyzer columns from
CIC-MalMem-2022 (see data/download_memory.py for the raw schema).

Two separate, deliberately independent steps:

1. `add_derived_features()` -- a small set of hand-picked ratio/aggregate
   features computed from the 55 raw columns (see DERIVED_FEATURES below
   for the full list with justifications). Purely a deterministic function
   of a single row's raw values -- no fitting, so it's safe to apply to
   train/val/test identically without any leakage concern.
2. `MemoryFeatureScaler` -- standardizes (zero mean, unit variance) the
   full raw+derived feature matrix. This DOES need fitting, and is fit on
   the train split's mean/std ONLY (`MemoryFeatureScaler.fit(train_df)`),
   then applied unchanged to val/test via `.transform()`. Fitting on
   val/test (e.g. via a global fit-then-split, or fitting on the full
   dataset before splitting) would leak their distribution -- their mean
   and spread -- into what the model implicitly sees at train time, the
   same leakage discipline scripts/split_memory.py already applies to
   vocabulary/pos_weight in the behavioral pipeline (computed from --train
   only, never from data the split hasn't assigned to train).

Deliberately NOT reimplemented: handles-per-process and DLL-per-process
ratios. Both were the natural first candidates for derived features, but
VolMemLyzer already computes them directly as raw columns
(`handles.avg_handles_per_proc`, `dlllist.avg_dlls_per_proc`) -- adding a
second, differently-named copy of numbers already in the raw 55 would be
padding the feature count, not adding information, so they're skipped here.
The same is true of `ldrmodules.*_avg` and `psxview.*_false_avg`, which are
already VolMemLyzer's own per-process-normalized versions of their sibling
raw counts.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from data.download_memory import feature_columns

logger = logging.getLogger("cortex.features.memory")

RAW_FEATURE_COUNT = 55

# Each derived feature is a ratio or aggregate that VolMemLyzer does NOT
# already provide as a raw column, chosen because it maps to a specific,
# named memory-forensics indicator -- not picked to hit a target count.
# name -> (numerator column(s), denominator column, one-line justification)
DERIVED_FEATURES: list[tuple[str, str]] = [
    (
        "callbacks_anonymous_ratio",
        "callbacks.nanonymous / callbacks.ncallbacks -- unresolvable-to-module "
        "('anonymous') kernel callbacks are a known rootkit/hooking indicator; "
        "the ratio is scale-invariant across snapshots with different overall "
        "callback volume, unlike the raw nanonymous count.",
    ),
    (
        "svcscan_driver_ratio",
        "(svcscan.kernel_drivers + svcscan.fs_drivers) / svcscan.nservices -- "
        "driver-type service registrations are disproportionately used for "
        "persistence/rootkit techniques versus ordinary process services; "
        "normalizing by total service count controls for baseline service "
        "count varying by machine.",
    ),
    (
        "handles_file_ratio",
        "handles.nfile / handles.nhandles -- an elevated share of file handles "
        "relative to total handles is consistent with mass file I/O (e.g. "
        "ransomware's bulk file encryption), a pattern the raw handle counts "
        "alone conflate with general process activity level.",
    ),
    (
        "handles_mutant_ratio",
        "handles.nmutant / handles.nhandles -- malware frequently creates "
        "named mutex/mutant objects for single-instance enforcement or "
        "inter-component coordination; elevated mutant share relative to "
        "total handles is a lightweight coordination-pattern indicator "
        "distinct from the file-access pattern above.",
    ),
    (
        "psxview_hiding_score",
        "sum of the 7 raw psxview.not_in_* process-hiding-detection counts, "
        "divided by pslist.nproc -- combines evidence from multiple "
        "independent process-enumeration methods into one normalized hiding "
        "severity score (a rootkit hidden from one API view but not another "
        "produces a partial signal in any single column); dividing by nproc "
        "controls for total process count varying across snapshots. Uses the "
        "raw not_in_* counts only, not the *_false_avg columns, which are "
        "already VolMemLyzer's own per-process average of these same counts.",
    ),
    (
        "ldrmodules_hidden_ratio",
        "(ldrmodules.not_in_load + not_in_init + not_in_mem) / dlllist.ndlls "
        "-- normalizes the raw counts of DLLs missing from PEB load/init/mem "
        "lists (a classic DLL-hiding/injection indicator) by total loaded-DLL "
        "count, so the signal reflects the *proportion* of hidden DLLs "
        "rather than being inflated on systems that simply load more DLLs.",
    ),
    (
        "malfind_injection_rate",
        "malfind.ninjections / pslist.nproc -- normalizes injected-code "
        "region count by total process count, so the signal reflects "
        "injection density rather than being confounded by snapshots that "
        "simply have more processes running.",
    ),
]
DERIVED_FEATURE_COLUMNS = [name for name, _ in DERIVED_FEATURES]


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """numerator / denominator, defined as 0.0 wherever denominator is 0.

    0 is the deliberate choice, not NaN or the raw division's inf/nan: a
    zero denominator here means "no callbacks/handles/services/DLLs at all
    on this snapshot" -- there is no concentration to measure, so "no
    signal" (0.0) is the correct neutral value, and it keeps every derived
    column finite and directly usable by MemoryFeatureScaler without a
    separate NaN-handling step.
    """
    safe_denom = denominator.replace(0, np.nan)
    return (numerator / safe_denom).fillna(0.0)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the DERIVED_FEATURE_COLUMNS to a copy of `df`. Purely a
    deterministic function of each row's existing raw columns -- safe to
    apply identically to train/val/test (no fitting happens here; see
    MemoryFeatureScaler for the one step that does)."""
    out = df.copy()
    out["callbacks_anonymous_ratio"] = _safe_ratio(out["callbacks.nanonymous"], out["callbacks.ncallbacks"])
    out["svcscan_driver_ratio"] = _safe_ratio(
        out["svcscan.kernel_drivers"] + out["svcscan.fs_drivers"], out["svcscan.nservices"],
    )
    out["handles_file_ratio"] = _safe_ratio(out["handles.nfile"], out["handles.nhandles"])
    out["handles_mutant_ratio"] = _safe_ratio(out["handles.nmutant"], out["handles.nhandles"])
    out["psxview_hiding_score"] = _safe_ratio(
        out["psxview.not_in_pslist"] + out["psxview.not_in_eprocess_pool"] + out["psxview.not_in_ethread_pool"]
        + out["psxview.not_in_pspcid_list"] + out["psxview.not_in_csrss_handles"] + out["psxview.not_in_session"]
        + out["psxview.not_in_deskthrd"],
        out["pslist.nproc"],
    )
    out["ldrmodules_hidden_ratio"] = _safe_ratio(
        out["ldrmodules.not_in_load"] + out["ldrmodules.not_in_init"] + out["ldrmodules.not_in_mem"],
        out["dlllist.ndlls"],
    )
    out["malfind_injection_rate"] = _safe_ratio(out["malfind.ninjections"], out["pslist.nproc"])
    return out


def feature_matrix_columns(df: pd.DataFrame) -> list[str]:
    """Raw feature columns (via data.download_memory.feature_columns) plus
    the derived columns, in a fixed order -- this order is what
    MemoryFeatureScaler's mean_/std_ arrays are indexed against, so it must
    stay consistent between fit and transform.

    Idempotent regardless of whether `df` already has the derived columns
    added (i.e. safe to call on either the raw loader/split output or the
    add_derived_features() output): feature_columns() only knows to exclude
    Category/Class/label, so on an already-derived `df` it would otherwise
    return the derived columns as if they were raw, and this function would
    then append them a second time. Filtering them out of the "raw" half
    first avoids that duplication either way."""
    raw = [c for c in feature_columns(df) if c not in DERIVED_FEATURE_COLUMNS]
    return raw + DERIVED_FEATURE_COLUMNS


@dataclass
class MemoryFeatureScaler:
    """Per-column (x - mean) / std standardization. `mean_`/`std_` are
    fit on one split's data only (see `fit`'s docstring) and then applied
    unchanged by `transform` -- this class holds no other state."""

    columns: list[str]
    mean_: np.ndarray
    std_: np.ndarray

    @classmethod
    def fit(cls, df: pd.DataFrame) -> "MemoryFeatureScaler":
        """Fits mean_/std_ from `df` only. Always call this with the TRAIN
        split's (already derived-feature-augmented) DataFrame -- fitting on
        val/test, or on the full dataset before splitting, leaks their
        distribution into what the model implicitly sees at train time."""
        columns = feature_matrix_columns(df)
        X = df[columns].to_numpy(dtype=np.float64)
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        # A handful of raw columns can be constant within a split (e.g.
        # pslist.nprocs64bit is 0 for every row in this 32-bit-only capture
        # environment) -- std=0 would divide by zero. Setting std=1 for
        # those columns leaves them at (x - mean), i.e. a constant 0 after
        # scaling, which is the correct behavior for a column carrying no
        # variance (and therefore no information) in this split.
        std[std == 0] = 1.0
        return cls(columns=columns, mean_=mean, std_=std)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.columns].to_numpy(dtype=np.float64)
        return (X - self.mean_) / self.std_

    def to_dict(self) -> dict:
        return {"format_version": 1, "columns": list(self.columns),
                "mean": [float(v) for v in self.mean_], "std": [float(v) for v in self.std_]}

    @classmethod
    def from_dict(cls, data: object, where: str = "<dict>") -> "MemoryFeatureScaler":
        if not isinstance(data, dict) or data.get("format_version") != 1:
            raise ValueError(f"{where}: not a format_version 1 MemoryFeatureScaler JSON object")
        columns, mean, std = data.get("columns"), data.get("mean"), data.get("std")
        if not (isinstance(columns, list) and all(isinstance(c, str) for c in columns)):
            raise ValueError(f"{where}: 'columns' must be a list of strings")
        for name, vals in (("mean", mean), ("std", std)):
            if not (isinstance(vals, list) and len(vals) == len(columns)
                    and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals)):
                raise ValueError(f"{where}: {name!r} must be {len(columns)} numbers")
        return cls(columns=columns, mean_=np.array(mean, dtype=np.float64),
                   std_=np.array(std, dtype=np.float64))

    def save(self, path: Union[str, Path]) -> None:
        """JSON, not pickle (docs/CODE_REVIEW.md F24)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "MemoryFeatureScaler":
        raw = Path(path).read_bytes()
        if raw[:1] == b"\x80":  # pickle protocol 2+ opcode
            raise ValueError(
                f"{path} is a pickled MemoryFeatureScaler; scalers are read from JSON only. "
                "Convert it once with `python -m scripts.convert_model_meta_to_json "
                f"--memory-scaler {path} <out.json>` (only for files we produced ourselves).")
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"unreadable MemoryFeatureScaler JSON {path}: {exc}") from exc
        return cls.from_dict(data, where=str(path))


def build_feature_matrix(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, MemoryFeatureScaler]:
    """End-to-end: derive features on all three splits, fit the scaler on
    train only, transform all three with it. Returns
    (X_train, X_val, X_test, y_train, y_val, y_test, scaler)."""
    train_aug = add_derived_features(train_df)
    val_aug = add_derived_features(val_df)
    test_aug = add_derived_features(test_df)

    scaler = MemoryFeatureScaler.fit(train_aug)
    X_train = scaler.transform(train_aug)
    X_val = scaler.transform(val_aug)
    X_test = scaler.transform(test_aug)

    y_train = train_df["label"].to_numpy(dtype=np.int32)
    y_val = val_df["label"].to_numpy(dtype=np.int32)
    y_test = test_df["label"].to_numpy(dtype=np.int32)

    logger.info(
        "Feature matrix: %d raw + %d derived = %d columns. train=%s val=%s test=%s",
        RAW_FEATURE_COUNT, len(DERIVED_FEATURE_COLUMNS), len(scaler.columns),
        X_train.shape, X_val.shape, X_test.shape,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler
