"""ONNX export for Cortex-Static (LightGBM) and Cortex-Behavioral (PyTorch)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger("cortex.export.onnx")


def export_static_lgbm_to_onnx(booster_path: str, output_path: str, num_features: int = 2568) -> None:
    """Convert a saved LightGBM booster (.lgbm text file) to ONNX via onnxmltools."""
    import lightgbm as lgb
    from onnxmltools import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType

    booster = lgb.Booster(model_file=booster_path)
    onnx_model = convert_lightgbm(
        booster, initial_types=[("input", FloatTensorType([None, num_features]))],
        target_opset=17,
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    logger.info("Static model exported to %s", output_path)


def export_behavioral_to_onnx(model: torch.nn.Module, output_path: str, sequence_length: int = 100,
                               opset: int = 17) -> None:
    model.eval().to("cpu")
    dummy = torch.randint(0, 2, (1, sequence_length), dtype=torch.long)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, output_path,
        input_names=["api_token_ids"], output_names=["malicious_logit"],
        dynamic_axes={"api_token_ids": {0: "batch"}, "malicious_logit": {0: "batch"}},
        opset_version=opset,
        dynamo=False,  # torch's newer dynamo-based exporter needs onnxscript,
                        # which isn't a project dependency; the legacy
                        # TorchScript-based exporter covers this model fine.
    )
    logger.info("Behavioral model exported to %s (%.2f MB)", output_path,
                Path(output_path).stat().st_size / 1e6)
