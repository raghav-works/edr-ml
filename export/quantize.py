"""
INT8 dynamic quantization for Cortex-Behavioral.

Cortex-Static (LightGBM) is NOT quantized — tree-ensemble split thresholds
are unaffected by INT8 weight quantization, so the FP32 ONNX export is the
deployed static artifact.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

logger = logging.getLogger("cortex.export.quantize")


@dataclass
class ComparisonReport:
    mean_absolute_error: float
    max_absolute_error: float
    correlation: float
    samples_tested: int
    accuracy_preserved: bool
    tolerance: float


def quantize_behavioral_int8(fp32_path: str, int8_path: str) -> None:
    Path(int8_path).parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(model_input=fp32_path, model_output=int8_path, weight_type=QuantType.QInt8)
    fp32_mb = Path(fp32_path).stat().st_size / 1e6
    int8_mb = Path(int8_path).stat().st_size / 1e6
    logger.info("Quantized %s (%.2f MB) -> %s (%.2f MB, %.1f%% reduction)",
                fp32_path, fp32_mb, int8_path, int8_mb, 100 * (1 - int8_mb / fp32_mb))


def compare_accuracy(
    fp32_path: str, int8_path: str, sequence_length: int = 100,
    vocab_size: int = 4096, n_samples: int = 500, seed: int = 42, tolerance: float = 0.05,
) -> ComparisonReport:
    rng = np.random.default_rng(seed)
    X = rng.integers(0, vocab_size, size=(n_samples, sequence_length)).astype(np.int64)

    fp32_sess = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    int8_sess = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
    in_name = fp32_sess.get_inputs()[0].name

    fp32_out = fp32_sess.run(None, {in_name: X})[0].flatten()
    int8_out = int8_sess.run(None, {in_name: X})[0].flatten()

    abs_err = np.abs(fp32_out - int8_out)
    corr = float(np.corrcoef(fp32_out, int8_out)[0, 1]) if n_samples > 1 else 1.0
    report = ComparisonReport(
        mean_absolute_error=float(abs_err.mean()), max_absolute_error=float(abs_err.max()),
        correlation=corr, samples_tested=n_samples,
        accuracy_preserved=bool(abs_err.mean() <= tolerance), tolerance=tolerance,
    )
    logger.info("FP32 vs INT8: MAE=%.5f max=%.5f corr=%.5f preserved=%s",
                report.mean_absolute_error, report.max_absolute_error, report.correlation,
                report.accuracy_preserved)
    return report


def benchmark_latency(onnx_path: str, sequence_length: int = 100, vocab_size: int = 4096,
                       iterations: int = 1000, warmup: int = 50, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    sample = rng.integers(0, vocab_size, size=(1, sequence_length)).astype(np.int64)

    for _ in range(warmup):
        sess.run(None, {in_name: sample})

    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        sess.run(None, {in_name: sample})
        latencies.append((time.perf_counter() - t0) * 1000)

    arr = np.array(latencies)
    return {"mean_ms": float(arr.mean()), "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99))}
