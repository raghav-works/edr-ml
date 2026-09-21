"""
Derive candidate STATIC_ALLOW_MAX / STATIC_BLOCK_MIN operating points on
the `cal` split ONLY -- never `test`, never `val`. Read-only and additive:
prints a menu of candidate thresholds per target FPR, with confidence
intervals; does not choose one, does not write config/thresholds.yaml or
anything under data/models/.

Why `cal` and not `test`: static's deployed thresholds (config/
thresholds.yaml, `allow_below`/`block_at_or_above`) were derived on
`ember2024_test` itself (see the header comment there) -- item 3's bug,
the exact thing the retrain-cluster split discipline exists to fix. This
script is the tool that was missing to do it properly: derive on `cal`,
leave `test` untouched for a single, final, reported-once read.

Reuses scripts.train_static._load_cal / _permutation_bounds and
models.static_lgbm.find_threshold_for_fpr / LGBMModel directly (imported,
never reimplemented), so the cal rows loaded here are GUARANTEED to be
exactly the same rows scripts/train_static.py::main()'s calibrate() step
used -- same seed (42), same val_frac/cal_frac carve, same source parquet.
If that carve ever changes, both call sites see it change together instead
of silently drifting apart.

This script never opens data/processed/ember2024_test.parquet and never
loads val rows -- grep the finished file to confirm (see the module's own
development notes / OPEN_ITEMS.md for the exact grep used).

Usage:
    python -m scripts.derive_static_thresholds \\
        --model data/models/cortex_static_retrain \\
        --out-report ~/cortex_static_retrain_run/threshold_report.txt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy import stats

from models.static_lgbm import LGBMModel, find_threshold_for_fpr
from scripts.train_static import _load_cal, _permutation_bounds

logger = logging.getLogger("cortex.scripts.derive_static_thresholds")

_REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TARGETS = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02]


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact (Clopper-Pearson) 95%-default CI for k successes out of n
    Bernoulli trials, via the beta distribution's inverse CDF. Chosen over
    a normal-approximation (Wald) interval because several of the target
    FPRs here rest on single-digit-to-low-double-digit false-positive
    counts (see module docstring), where a normal approximation is known
    to be unreliable; Clopper-Pearson is exact (conservative) at any n.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    lower = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2, k, n - k + 1))
    upper = 1.0 if k == n else float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return (lower, upper)


def sweep_thresholds(y_cal: np.ndarray, proba_cal: np.ndarray, targets: list[float]) -> list[dict]:
    """Pure function -- no model, no file I/O, so it's testable with a
    hand-made array. For each target FPR (in the given order), finds the
    threshold via find_threshold_for_fpr(), then reports the actually
    achieved FPR/detection rate plus their exact 95% Clopper-Pearson CIs
    and the count of cal rows exactly tied at that threshold (relevant
    since ties at the boundary land on whichever side the >= comparison
    puts them, so a large tie count is worth seeing before picking a
    threshold from real data with many repeated/quantized scores).
    """
    y_cal = np.asarray(y_cal)
    proba_cal = np.asarray(proba_cal)
    benign_mask = y_cal == 0
    malicious_mask = y_cal == 1
    n_benign = int(benign_mask.sum())
    n_malicious = int(malicious_mask.sum())

    rows = []
    for target in targets:
        t = find_threshold_for_fpr(y_cal, proba_cal, target)
        preds = proba_cal >= t
        n_fp = int((preds & benign_mask).sum())
        n_tp = int((preds & malicious_mask).sum())
        actual_fpr = n_fp / n_benign if n_benign else float("nan")
        detection_rate = n_tp / n_malicious if n_malicious else float("nan")
        rows.append({
            "target_fpr": target,
            "threshold": t,
            "actual_fpr": actual_fpr,
            "n_fp": n_fp,
            "n_benign": n_benign,
            "fpr_ci": clopper_pearson_ci(n_fp, n_benign),
            "detection_rate": detection_rate,
            "n_tp": n_tp,
            "n_malicious": n_malicious,
            "detection_ci": clopper_pearson_ci(n_tp, n_malicious),
            "n_tied": int((proba_cal == t).sum()),
        })
    return rows


def three_way_confusion(y_cal: np.ndarray, proba_cal: np.ndarray,
                         allow_max: float, block_min: float) -> dict:
    """ALLOW/ALERT/BLOCK 3-way counts vs true label -- information only,
    not used to pick anything. Mirrors the shape of scripts/
    evaluate_all_models.py::_static_three_way's output without importing
    or copying it (that function is scoped to the frozen test split)."""
    y_cal = np.asarray(y_cal).astype(np.int32)
    proba_cal = np.asarray(proba_cal)
    allow = proba_cal < allow_max
    block = proba_cal >= block_min
    alert = (~allow) & (~block)
    out = {}
    for name, mask in (("benign", y_cal == 0), ("malicious", y_cal == 1)):
        out[name] = {
            "ALLOW": int(allow[mask].sum()),
            "ALERT": int(alert[mask].sum()),
            "BLOCK": int(block[mask].sum()),
            "n": int(mask.sum()),
        }
    return out


def _predict_proba_chunked(model: LGBMModel, X: np.ndarray, chunk_size: int = 50_000) -> np.ndarray:
    """Calibrated probabilities in bounded-memory chunks."""
    out = np.empty(X.shape[0], dtype=np.float64)
    for start in range(0, X.shape[0], chunk_size):
        end = min(start + chunk_size, X.shape[0])
        out[start:end] = model.predict_proba(X[start:end])
    return out


def _fmt_ci(ci: tuple[float, float]) -> str:
    return f"[{ci[0]:.6f}, {ci[1]:.6f}]"


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path prefix (no extension) to a saved LGBMModel.")
    ap.add_argument("--train", default="data/processed/ember2024_train.parquet",
                     help="Source parquet cal is carved from -- NEVER the test parquet.")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--cal-frac", type=float, default=0.1)
    ap.add_argument("--targets", default=",".join(str(t) for t in DEFAULT_TARGETS),
                     help="Comma list of target FPRs to sweep.")
    ap.add_argument("--out-report", default=None,
                     help="Optional text file to also write the report to. Must be OUTSIDE the repo tree.")
    args = ap.parse_args()

    if args.out_report is not None:
        out_path = Path(args.out_report).expanduser().resolve()
        if out_path.is_relative_to(_REPO_ROOT):
            raise ValueError(
                f"--out-report ({out_path}) resolves inside the repo tree ({_REPO_ROOT}); "
                f"this script writes nothing inside the repo. Pick a path outside it."
            )
    else:
        out_path = None

    targets = [float(t) for t in args.targets.split(",") if t.strip()]

    report_lines: list[str] = []

    def emit(msg: str) -> None:
        logger.info(msg)
        report_lines.append(msg)

    model = LGBMModel.load(args.model)
    emit(f"model={args.model} num_iterations={model.num_iterations} feature_count={model.feature_count} "
         f"calibrator_present={model.calibrator is not None}")

    pf = pq.ParquetFile(args.train)
    n_rows = pf.metadata.num_rows
    perm, n_val, n_cal_expected_prefilter = _permutation_bounds(n_rows, args.val_frac, args.cal_frac, seed=42)
    emit(f"source={args.train} n_rows_total={n_rows} n_val={n_val} "
         f"n_cal_expected(pre -1-filter)={n_cal_expected_prefilter}")

    X_cal, y_cal = _load_cal(args.train, args.val_frac, args.cal_frac, seed=42)
    emit(f"cal rows loaded (post -1-filter): {len(y_cal)}")
    if len(y_cal) > n_cal_expected_prefilter:
        raise AssertionError(
            f"loaded cal row count {len(y_cal)} exceeds the pre-filter expectation "
            f"{n_cal_expected_prefilter} -- something is wrong with the carve."
        )
    n_dropped_unlabeled = n_cal_expected_prefilter - len(y_cal)
    emit(f"cal rows dropped as unlabeled (-1): {n_dropped_unlabeled}")

    n_benign = int((y_cal == 0).sum())
    n_malicious = int((y_cal == 1).sum())
    emit(f"cal benign={n_benign} malicious={n_malicious}")

    proba_cal = _predict_proba_chunked(model, X_cal, chunk_size=50_000)
    del X_cal

    n_distinct = int(np.unique(proba_cal).size)
    emit(f"distinct calibrated scores on cal: {n_distinct} (of {len(proba_cal)} rows)")

    benign_mask = y_cal == 0
    malicious_mask = y_cal == 1
    for name, mask in (("benign", benign_mask), ("malicious", malicious_mask)):
        n = int(mask.sum())
        above = int((proba_cal[mask] > 0.99).sum())
        below = int((proba_cal[mask] < 0.01).sum())
        emit(f"cal {name}: {above}/{n} ({100*above/n:.4f}%) scores > 0.99, "
             f"{below}/{n} ({100*below/n:.4f}%) scores < 0.01")

    emit("=== threshold sweep on CAL only (test untouched, val not read) ===")
    rows = sweep_thresholds(y_cal, proba_cal, targets)
    prev_threshold = None
    for r in rows:
        emit(
            f"target_fpr={r['target_fpr']:.4f} -> threshold={r['threshold']:.10f} "
            f"actual_fpr={r['actual_fpr']:.6f} (95% CI {_fmt_ci(r['fpr_ci'])}, {r['n_fp']}/{r['n_benign']} benign FP) "
            f"detection_rate={r['detection_rate']:.4f} (95% CI {_fmt_ci(r['detection_ci'])}, "
            f"{r['n_tp']}/{r['n_malicious']} malicious TP) n_tied_at_threshold={r['n_tied']}"
        )
        # Full-precision threshold, plus proof (not assumption) that this
        # exact repr() value reproduces the same FP/TP counts already
        # reported above -- guards against a rounding/formatting mismatch
        # ever being silently used to pick an operating point.
        repr_threshold = repr(r["threshold"])
        recomputed_preds = proba_cal >= float(repr_threshold)
        recomputed_fp = int((recomputed_preds & benign_mask).sum())
        recomputed_tp = int((recomputed_preds & malicious_mask).sum())
        if recomputed_fp != r["n_fp"] or recomputed_tp != r["n_tp"]:
            raise AssertionError(
                f"target_fpr={r['target_fpr']}: repr() threshold {repr_threshold} recomputed "
                f"FP={recomputed_fp} TP={recomputed_tp}, expected FP={r['n_fp']} TP={r['n_tp']}"
            )
        emit(f"  full-precision threshold: {repr_threshold} "
             f"(reproduces FP={recomputed_fp} TP={recomputed_tp} exactly, verified not assumed)")
        if prev_threshold is not None:
            jump = prev_threshold - r["threshold"]
            flag = "  <-- CLIFF (drop > 0.1)" if jump > 0.1 else ""
            emit(f"  threshold jump from previous target: {jump:.10f}{flag}")
        prev_threshold = r["threshold"]

    # Monotonicity check, reported not asserted (real data could in principle
    # tie two adjacent targets to the same threshold, which is fine).
    thresholds_seq = [r["threshold"] for r in rows]
    non_increasing = all(thresholds_seq[i] >= thresholds_seq[i + 1] for i in range(len(thresholds_seq) - 1))
    emit(f"thresholds non-increasing as target FPR grows: {non_increasing}")

    # Information-only 3-way confusion at (ALLOW target=0.01, BLOCK target=0.001),
    # if both targets were actually in the swept list.
    by_target = {r["target_fpr"]: r["threshold"] for r in rows}
    if 0.01 in by_target and 0.001 in by_target:
        allow_max = by_target[0.01]
        block_min = by_target[0.001]
        tw = three_way_confusion(y_cal, proba_cal, allow_max, block_min)
        emit(f"=== INFO ONLY: 3-way confusion on cal, ALLOW@target=0.01 ({allow_max:.10f}) / "
             f"BLOCK@target=0.001 ({block_min:.10f}) ===")
        for name in ("benign", "malicious"):
            r = tw[name]
            emit(f"  {name}: ALLOW={r['ALLOW']} ALERT={r['ALERT']} BLOCK={r['BLOCK']} n={r['n']}")
    else:
        emit("(skipping info-only 3-way confusion: targets 0.01 and/or 0.001 not in --targets)")

    emit("This is a MENU, not a decision -- no threshold was chosen, "
         "config/thresholds.yaml was not touched, nothing was written under data/models/.")

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(report_lines) + "\n")
        logger.info("Report also written to %s", out_path)


if __name__ == "__main__":
    main()
