"""
pytest wrapper around scripts/verify_onnx_parity.py -- runs the
ONNX-vs-Python parity check for the three LightGBM signals (Cortex-Static /
Cortex-Memory / Cortex-Network) as part of the suite.

Marked `slow` (loads the ~60 MB static ONNX graph + real held-out test
parquets; ~15 s total) and skipped in full if any model / data artifact is
absent -- a fresh checkout without data/models/ still gets a clean
`pytest -q` (these tests report as skipped, not errored).

Tolerances come from the item-1b full parity run:
  static  max_abs_err 9.997e-3  0 verdict flips
  memory  max_abs_err 7.50e-7   0 verdict flips
  network max_abs_err 2.11e-1   3/213k flips (2 are exact threshold ties) --
          the documented float32 ONNX-TreeEnsemble ceiling for network's
          large-magnitude IAT/duration features, accepted as final.
"""
from __future__ import annotations

import os

import pytest

_REQUIRED = [
    "data/models/cortex_static.onnx", "data/models/cortex_static.lgbm", "data/models/cortex_static.meta.json",
    "data/models/cortex_memory.onnx", "data/models/cortex_memory.lgbm", "data/models/cortex_memory.meta.json",
    "data/models/cortex_network.onnx", "data/models/cortex_network.lgbm", "data/models/cortex_network.meta.json",
    "data/processed/ember2024_test.parquet",
    "data/processed/memory_test.parquet",
    "data/processed/network_test.parquet",
]
_MISSING = [p for p in _REQUIRED if not os.path.exists(p)]

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(bool(_MISSING), reason=f"ONNX parity artifacts absent (e.g. {_MISSING[:2]})"),
]


def _assert_no_real_flips(result: dict, max_nontie: int) -> None:
    for tname, info in result["flips"].items():
        real = info["flips"] - info["ties"]
        assert real <= max_nontie, (
            f"{result['name']} @ {tname}: {info['flips']} flips ({info['ties']} ties) "
            f"-> {real} real divergences, limit {max_nontie}"
        )


def test_static_onnx_parity():
    from scripts.verify_onnx_parity import check_static

    r = check_static(static_n=4000)
    assert r["max_abs_err"] < 0.05
    assert r["mean_abs_err"] < 1e-3
    _assert_no_real_flips(r, max_nontie=0)


def test_memory_onnx_parity():
    from scripts.verify_onnx_parity import check_memory

    r = check_memory()
    assert r["max_abs_err"] < 1e-4
    _assert_no_real_flips(r, max_nontie=0)


def test_network_onnx_parity():
    from scripts.verify_onnx_parity import check_network

    r = check_network(limit=5000)
    # network rides the documented float32 TreeEnsemble ceiling, not exactness.
    assert r["max_abs_err"] < 0.6
    assert r["mean_abs_err"] < 5e-3
    _assert_no_real_flips(r, max_nontie=3)
