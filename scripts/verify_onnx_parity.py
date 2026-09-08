"""
ONNX-vs-Python parity check for the three LightGBM tree signals
(Cortex-Static / Cortex-Memory / Cortex-Network).

For each model it runs the real held-out test split through both
    - the Python path: <Model>.load(...).predict_proba(X)
    - the exported ONNX graph via onnxruntime
and reports:
    - max / mean absolute error between the two calibrated-probability outputs
    - the number of predictions that flip across the deployed operating
      threshold(s) in inference/policy_engine.py

The ONNX graph now reconstructs the raw booster margin in-graph
(logit(p) = log(p) - log(1-p)) before the Platt-calibrator subgraph, because
the calibrators are fit on raw margins (not sigmoid probabilities). This
script is the acceptance gate for that change.

Usage:
    python -m scripts.verify_onnx_parity                 # all three, defaults
    python -m scripts.verify_onnx_parity --static-n 0    # full static test split
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger("cortex.verify_onnx_parity")

STATIC_ONNX = "data/models/cortex_static.onnx"
MEMORY_ONNX = "data/models/cortex_memory.onnx"
NETWORK_ONNX = "data/models/cortex_network.onnx"


def _onnx_run(onnx_path: str, X: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    out = []
    for start in range(0, len(X), 8192):
        chunk = X[start:start + 8192].astype(np.float32)
        out.append(sess.run([out_name], {in_name: chunk})[0].reshape(-1))
    return np.concatenate(out)


def _report(name: str, py: np.ndarray, ox: np.ndarray, y: np.ndarray,
            thresholds: dict[str, float]) -> dict:
    """Log the parity report and return it as a dict (so tests can assert on it)."""
    err = np.abs(py - ox)
    logger.info("[%s] n=%d  max_abs_err=%.3e  mean_abs_err=%.3e  median_abs_err=%.3e",
                name, len(py), err.max(), err.mean(), np.median(err))
    worst = int(np.argmax(err))
    logger.info("[%s]   worst row: py=%.8f onnx=%.8f (true label=%d)", name, py[worst], ox[worst], int(y[worst]))
    flips_out: dict[str, dict] = {}
    for tname, thr in thresholds.items():
        mask = (py >= thr) != (ox >= thr)
        flips = int(mask.sum())
        # of the flips, how many are within 1e-6 of the threshold (pure tie-break)
        near = int((np.abs(py - thr)[mask] < 1e-6).sum()) if flips else 0
        py_fpr = float((py[y == 0] >= thr).mean()) if (y == 0).any() else float("nan")
        ox_fpr = float((ox[y == 0] >= thr).mean()) if (y == 0).any() else float("nan")
        py_det = float((py[y == 1] >= thr).mean()) if (y == 1).any() else float("nan")
        ox_det = float((ox[y == 1] >= thr).mean()) if (y == 1).any() else float("nan")
        logger.info("[%s]   @ %-10s thr=%.10f : flips=%d (%d are threshold ties)  "
                    "FPR py=%.4f%% onnx=%.4f%%  det py=%.4f%% onnx=%.4f%%",
                    name, tname, thr, flips, near, py_fpr * 100, ox_fpr * 100, py_det * 100, ox_det * 100)
        flips_out[tname] = {"flips": flips, "ties": near, "threshold": float(thr)}
    return {
        "name": name,
        "n": int(len(py)),
        "max_abs_err": float(err.max()),
        "mean_abs_err": float(err.mean()),
        "median_abs_err": float(np.median(err)),
        "flips": flips_out,
    }


def check_static(static_n: int = 50000) -> dict:
    from inference.policy_engine import STATIC_ALLOW_MAX, STATIC_BLOCK_MIN
    from models.static_lgbm import LGBMModel

    model = LGBMModel.load("data/models/cortex_static")
    pf = pq.ParquetFile("data/processed/ember2024_test.parquet")
    feat = [f"feature_{i}" for i in range(2568)]
    Xs, ys = [], []
    got = 0
    for rg in range(pf.metadata.num_row_groups):
        b = pf.read_row_group(rg, columns=feat + ["label"]).to_pandas()
        Xs.append(b[feat].to_numpy(np.float32))
        ys.append(b["label"].to_numpy(np.int32))
        got += len(b)
        if static_n and got >= static_n:
            break
    X = np.concatenate(Xs); y = np.concatenate(ys)
    if static_n:
        X, y = X[:static_n], y[:static_n]
    keep = y != -1
    X, y = X[keep], y[keep]
    py = model.predict_proba(X)
    ox = _onnx_run(STATIC_ONNX, X)
    return _report("static", py, ox, y, {"ALLOW_MAX": STATIC_ALLOW_MAX, "BLOCK_MIN": STATIC_BLOCK_MIN})


def check_memory(limit: int | None = None) -> dict:
    from features.memory_features import add_derived_features, feature_matrix_columns
    from inference.policy_engine import MEMORY_MALICIOUS_MIN
    from models.memory_lgbm import MemoryLGBMModel

    model = MemoryLGBMModel.load("data/models/cortex_memory")
    df = add_derived_features(pd.read_parquet("data/processed/memory_test.parquet"))
    cols = feature_matrix_columns(df)
    X = df[cols].to_numpy(np.float32); y = df["label"].to_numpy(np.int32)
    if limit:
        X, y = X[:limit], y[:limit]
    py = model.predict_proba(X)
    ox = _onnx_run(MEMORY_ONNX, X)
    return _report("memory", py, ox, y, {"MALICIOUS_MIN": MEMORY_MALICIOUS_MIN})


def check_network(limit: int | None = None) -> dict:
    from data.download_network import FEATURE_COLUMNS
    from inference.policy_engine import NETWORK_MALICIOUS_MIN
    from models.network_lgbm import NetworkLGBMModel

    model = NetworkLGBMModel.load("data/models/cortex_network")
    df = pd.read_parquet("data/processed/network_test.parquet")
    X = df[FEATURE_COLUMNS].to_numpy(np.float32); y = df["label"].to_numpy(np.int32)
    if limit:
        X, y = X[:limit], y[:limit]
    py = model.predict_proba(X)
    ox = _onnx_run(NETWORK_ONNX, X)
    return _report("network", py, ox, y, {"MALICIOUS_MIN": NETWORK_MALICIOUS_MIN})


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--static-n", type=int, default=50000,
                    help="rows of the static test split to check (0 = all 539,940)")
    ap.add_argument("--skip", default="", help="comma list of static,memory,network to skip")
    args = ap.parse_args()
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    if "static" not in skip:
        check_static(args.static_n)
    if "memory" not in skip:
        check_memory()
    if "network" not in skip:
        check_network()


if __name__ == "__main__":
    main()
