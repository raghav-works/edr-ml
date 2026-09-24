"""ONNX export for Cortex-Static, Cortex-Memory, and Cortex-Network (all
LightGBM), and Cortex-Behavioral (PyTorch)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger("cortex.export.onnx")

# onnxmltools' LightGBM converter (1.16.0 as pinned in requirements.txt)
# rejects target_opset above 15 outright ("higher than the converter
# support"). The exported graph is upgraded to this same opset afterward via
# onnx.version_converter so it can be merged with the skl2onnx-converted
# calibrator subgraph below, which onnxruntime/skl2onnx target at 15 too.
# Applies to any onnxmltools-converted LightGBM booster, not just static's
# -- shared by export_static_lgbm_to_onnx and export_memory_lgbm_to_onnx
# below, since both go through the identical onnxmltools/skl2onnx path.
#
# Verified against the real cortex_static model (2996 trees) and a
# realistic synthetic model (47 trees): raw and calibrated ONNX outputs
# both match the Python path to ~1e-7. One known degenerate edge case,
# not believed to affect any realistically-trained model: a pathological
# 1-tree booster (produced by early stopping firing at round 1 on
# near-random synthetic data during testing) showed a large mismatch
# (~0.48) between the ONNX TreeEnsembleClassifier's raw probability and
# booster.predict()'s. Root cause not tracked down -- suspected an
# onnxmltools quirk with extremely shallow/degenerate boosters -- but a
# 47-tree model showed no such issue, and no realistically-sized model
# should ever hit this.
_LGBM_ONNX_OPSET = 15


def export_static_lgbm_to_onnx(model_path: str, output_path: str, num_features: int = 2568) -> None:
    """Convert a saved Cortex-Static model (LGBMModel.save() output --
    `<model_path>.lgbm` + `<model_path>.meta`) to a single self-contained
    ONNX file.

    If the model has a Platt calibrator (the default -- see
    models/static_lgbm.py::train()), the calibrator's LogisticRegression is
    converted to ONNX too and merged into the SAME graph, chained after the
    booster's raw probability output. The exported model's one output,
    "malicious_probability", is therefore the *calibrated* probability --
    exactly what LGBMModel.predict_proba() returns, not the raw booster
    score. This matters: raw LightGBM probabilities and the Platt-calibrated
    ones differ by up to ~0.21 on real test data (the raw booster saturates
    hard at 0.0/1.0 for many samples; calibration compresses that back into
    [0.0059, 0.996]), and inference/policy_engine.py's thresholds were
    derived against the *calibrated* distribution. Exporting the raw booster
    alone would silently break threshold correctness for any consumer of
    this ONNX file. If there's no calibrator, the single output is just the
    booster's own probability (no merge needed).
    """
    import onnx
    import onnx.compose
    import onnx.numpy_helper
    import onnx.version_converter
    from onnxmltools import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType

    from models.static_lgbm import LGBMModel

    model = LGBMModel.load(model_path)

    lgbm_onnx = convert_lightgbm(
        model.booster, initial_types=[("input", FloatTensorType([None, num_features]))],
        target_opset=_LGBM_ONNX_OPSET, zipmap=False,
    )
    # "probabilities" here is [N, 2] = [P(benign), P(malicious)]. Slice out
    # column 1. The graph is still opset-9 for the default domain at this
    # point (onnxmltools' internal default, independent of target_opset), so
    # Slice must use the opset-9 attribute form (starts/ends/axes as node
    # attributes) rather than opset-10+'s tensor-input form -- the
    # version_converter call below upgrades this automatically once bumped.
    lgbm_onnx.graph.node.append(onnx.helper.make_node(
        "Slice", inputs=["probabilities"], outputs=["malicious_prob_col"],
        starts=[1], ends=[2], axes=[1],
    ))
    lgbm_onnx.graph.output.append(
        onnx.helper.make_tensor_value_info("malicious_prob_col", onnx.TensorProto.FLOAT, [None, 1])
    )
    lgbm_onnx = onnx.version_converter.convert_version(lgbm_onnx, _LGBM_ONNX_OPSET)

    if model.calibrator is not None:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType as SkFloatTensorType

        # The Platt calibrator is now fit on the booster's raw margins
        # (logits), not on its sigmoid probability. onnxmltools'
        # TreeEnsembleClassifier only exposes the post-sigmoid probability, so
        # reconstruct the margin in-graph before the calibrator subgraph:
        #   margin = logit(p) = log(p) - log(1 - p)
        # p is clipped off {0, 1} so log() stays finite when float32 rounds a
        # very confident score to exactly 0.0 / 1.0.
        _LOGIT_EPS = 1e-7
        lgbm_onnx.graph.initializer.extend([
            onnx.numpy_helper.from_array(np.array(_LOGIT_EPS, dtype=np.float32), name="logit_eps"),
            onnx.numpy_helper.from_array(np.array(1.0 - _LOGIT_EPS, dtype=np.float32), name="logit_1m_eps"),
            onnx.numpy_helper.from_array(np.array(1.0, dtype=np.float32), name="logit_one"),
        ])
        lgbm_onnx.graph.node.extend([
            onnx.helper.make_node("Clip", ["malicious_prob_col", "logit_eps", "logit_1m_eps"], ["logit_p"]),
            onnx.helper.make_node("Log", ["logit_p"], ["logit_log_p"]),
            onnx.helper.make_node("Sub", ["logit_one", "logit_p"], ["logit_1m_p"]),
            onnx.helper.make_node("Log", ["logit_1m_p"], ["logit_log_1m_p"]),
            onnx.helper.make_node("Sub", ["logit_log_p", "logit_log_1m_p"], ["malicious_margin_col"]),
        ])
        lgbm_onnx.graph.output.append(
            onnx.helper.make_tensor_value_info("malicious_margin_col", onnx.TensorProto.FLOAT, [None, 1])
        )

        calibrator_onnx = convert_sklearn(
            model.calibrator._lr, initial_types=[("float_input", SkFloatTensorType([None, 1]))],
            target_opset=_LGBM_ONNX_OPSET, options={id(model.calibrator._lr): {"zipmap": False}},
        )
        lgbm_onnx.ir_version = calibrator_onnx.ir_version  # onnxmltools/skl2onnx target different IR versions
        calibrator_onnx = onnx.compose.add_prefix(calibrator_onnx, prefix="calib_")  # both graphs use "label"/"probabilities"

        merged = onnx.compose.merge_models(
            lgbm_onnx, calibrator_onnx, io_map=[("malicious_margin_col", "calib_float_input")],
        )
        prob_source = "calib_probabilities"  # [N, 2] = [P(benign), P(malicious)] from the calibrator
    else:
        merged = lgbm_onnx
        prob_source = None

    # Trim to a single clean 1-D output: the calibrated (or raw, if no
    # calibrator) malicious-class probability -- not a 2-column array
    # consumers have to know to index into.
    if prob_source is not None:
        merged.graph.node.append(onnx.helper.make_node(
            "Slice", inputs=[prob_source, "final_slice_starts", "final_slice_ends", "final_slice_axes"],
            outputs=["malicious_probability_col"],
        ))
        merged.graph.initializer.extend([
            onnx.numpy_helper.from_array(np.array([1], dtype=np.int64), name="final_slice_starts"),
            onnx.numpy_helper.from_array(np.array([2], dtype=np.int64), name="final_slice_ends"),
            onnx.numpy_helper.from_array(np.array([1], dtype=np.int64), name="final_slice_axes"),
        ])
        merged.graph.node.append(onnx.helper.make_node(
            "Squeeze", inputs=["malicious_probability_col", "final_squeeze_axes"],
            outputs=["malicious_probability"],
        ))
        merged.graph.initializer.append(
            onnx.numpy_helper.from_array(np.array([1], dtype=np.int64), name="final_squeeze_axes")
        )
    else:
        merged.graph.node.append(onnx.helper.make_node(
            "Squeeze", inputs=["malicious_prob_col", "final_squeeze_axes"], outputs=["malicious_probability"],
        ))
        merged.graph.initializer.append(
            onnx.numpy_helper.from_array(np.array([1], dtype=np.int64), name="final_squeeze_axes")
        )
    del merged.graph.output[:]
    merged.graph.output.append(
        onnx.helper.make_tensor_value_info("malicious_probability", onnx.TensorProto.FLOAT, [None])
    )

    onnx.checker.check_model(merged)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(merged.SerializeToString())
    logger.info(
        "Static model exported to %s (calibrator %s, %.2f MB)",
        output_path, "chained into graph" if model.calibrator is not None else "absent",
        Path(output_path).stat().st_size / 1e6,
    )


def export_memory_lgbm_to_onnx(model_path: str, output_path: str, num_features: int = 62) -> None:
    """Convert a saved Cortex-Memory model (MemoryLGBMModel.save() output --
    `<model_path>.lgbm` + `<model_path>.meta`) to a single self-contained
    ONNX file. Structurally identical to export_static_lgbm_to_onnx() above
    (same onnxmltools -> skl2onnx -> graph-merge path, same opset
    constraints and degenerate-booster caveat -- see that function's
    docstring and _LGBM_ONNX_OPSET's comment for the full rationale, which
    applies here unchanged since both are LightGBM + optional Platt
    calibrator). Duplicated rather than parameterized into one shared
    function to keep static's and memory's export paths independent, the
    same way models/memory_lgbm.py duplicates rather than imports
    static_lgbm.py's PlattCalibrator.

    As with static, the calibrator (if present) is merged into the SAME
    graph, chained after the booster's raw probability output, so the
    exported model's one output, "malicious_probability", is the
    *calibrated* probability -- exactly what MemoryLGBMModel.predict_proba()
    returns, matching what inference/policy_engine.py's MEMORY_MALICIOUS_MIN
    was derived against. Verified against the real cortex_memory model and
    the actual CIC-MalMem-2022 test split -- see docs/TECHNICAL_NOTES.md's ONNX export
    section for the measured max/mean absolute error.
    """
    import onnx
    import onnx.compose
    import onnx.numpy_helper
    import onnx.version_converter
    from onnxmltools import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType

    from models.memory_lgbm import MemoryLGBMModel

    model = MemoryLGBMModel.load(model_path)

    lgbm_onnx = convert_lightgbm(
        model.booster, initial_types=[("input", FloatTensorType([None, num_features]))],
        target_opset=_LGBM_ONNX_OPSET, zipmap=False,
    )
    lgbm_onnx.graph.node.append(onnx.helper.make_node(
        "Slice", inputs=["probabilities"], outputs=["malicious_prob_col"],
        starts=[1], ends=[2], axes=[1],
    ))
    lgbm_onnx.graph.output.append(
        onnx.helper.make_tensor_value_info("malicious_prob_col", onnx.TensorProto.FLOAT, [None, 1])
    )
    lgbm_onnx = onnx.version_converter.convert_version(lgbm_onnx, _LGBM_ONNX_OPSET)

    if model.calibrator is not None:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType as SkFloatTensorType

        # The Platt calibrator is now fit on the booster's raw margins
        # (logits), not on its sigmoid probability. onnxmltools'
        # TreeEnsembleClassifier only exposes the post-sigmoid probability, so
        # reconstruct the margin in-graph before the calibrator subgraph:
        #   margin = logit(p) = log(p) - log(1 - p)
        # p is clipped off {0, 1} so log() stays finite when float32 rounds a
        # very confident score to exactly 0.0 / 1.0.
        _LOGIT_EPS = 1e-7
        lgbm_onnx.graph.initializer.extend([
            onnx.numpy_helper.from_array(np.array(_LOGIT_EPS, dtype=np.float32), name="logit_eps"),
            onnx.numpy_helper.from_array(np.array(1.0 - _LOGIT_EPS, dtype=np.float32), name="logit_1m_eps"),
            onnx.numpy_helper.from_array(np.array(1.0, dtype=np.float32), name="logit_one"),
        ])
        lgbm_onnx.graph.node.extend([
            onnx.helper.make_node("Clip", ["malicious_prob_col", "logit_eps", "logit_1m_eps"], ["logit_p"]),
            onnx.helper.make_node("Log", ["logit_p"], ["logit_log_p"]),
            onnx.helper.make_node("Sub", ["logit_one", "logit_p"], ["logit_1m_p"]),
            onnx.helper.make_node("Log", ["logit_1m_p"], ["logit_log_1m_p"]),
            onnx.helper.make_node("Sub", ["logit_log_p", "logit_log_1m_p"], ["malicious_margin_col"]),
        ])
        lgbm_onnx.graph.output.append(
            onnx.helper.make_tensor_value_info("malicious_margin_col", onnx.TensorProto.FLOAT, [None, 1])
        )

        calibrator_onnx = convert_sklearn(
            model.calibrator._lr, initial_types=[("float_input", SkFloatTensorType([None, 1]))],
            target_opset=_LGBM_ONNX_OPSET, options={id(model.calibrator._lr): {"zipmap": False}},
        )
        lgbm_onnx.ir_version = calibrator_onnx.ir_version
        calibrator_onnx = onnx.compose.add_prefix(calibrator_onnx, prefix="calib_")

        merged = onnx.compose.merge_models(
            lgbm_onnx, calibrator_onnx, io_map=[("malicious_margin_col", "calib_float_input")],
        )
        prob_source = "calib_probabilities"
    else:
        merged = lgbm_onnx
        prob_source = None

    if prob_source is not None:
        merged.graph.node.append(onnx.helper.make_node(
            "Slice", inputs=[prob_source, "final_slice_starts", "final_slice_ends", "final_slice_axes"],
            outputs=["malicious_probability_col"],
        ))
        merged.graph.initializer.extend([
            onnx.numpy_helper.from_array(np.array([1], dtype=np.int64), name="final_slice_starts"),
            onnx.numpy_helper.from_array(np.array([2], dtype=np.int64), name="final_slice_ends"),
            onnx.numpy_helper.from_array(np.array([1], dtype=np.int64), name="final_slice_axes"),
        ])
        merged.graph.node.append(onnx.helper.make_node(
            "Squeeze", inputs=["malicious_probability_col", "final_squeeze_axes"],
            outputs=["malicious_probability"],
        ))
        merged.graph.initializer.append(
            onnx.numpy_helper.from_array(np.array([1], dtype=np.int64), name="final_squeeze_axes")
        )
    else:
        merged.graph.node.append(onnx.helper.make_node(
            "Squeeze", inputs=["malicious_prob_col", "final_squeeze_axes"], outputs=["malicious_probability"],
        ))
        merged.graph.initializer.append(
            onnx.numpy_helper.from_array(np.array([1], dtype=np.int64), name="final_squeeze_axes")
        )
    del merged.graph.output[:]
    merged.graph.output.append(
        onnx.helper.make_tensor_value_info("malicious_probability", onnx.TensorProto.FLOAT, [None])
    )

    onnx.checker.check_model(merged)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(merged.SerializeToString())
    logger.info(
        "Memory model exported to %s (calibrator %s, %.2f MB)",
        output_path, "chained into graph" if model.calibrator is not None else "absent",
        Path(output_path).stat().st_size / 1e6,
    )


def export_network_lgbm_to_onnx(model_path: str, output_path: str, num_features: int = 78) -> None:
    """Convert a saved Cortex-Network model (NetworkLGBMModel.save() output --
    `<model_path>.lgbm` + `<model_path>.meta`) to a single self-contained
    ONNX file. Structurally identical to export_static_lgbm_to_onnx() and
    export_memory_lgbm_to_onnx() above (same onnxmltools -> skl2onnx ->
    graph-merge path, same opset constraints and degenerate-booster caveat
    -- see export_static_lgbm_to_onnx's docstring and _LGBM_ONNX_OPSET's
    comment for the full rationale). Duplicated rather than parameterized
    into one shared function, same rationale as memory's copy of this
    function relative to static's.

    As with static and memory, the calibrator (if present) is merged into
    the SAME graph, chained after the booster's raw probability output, so
    the exported model's one output, "malicious_probability", is the
    *calibrated* probability -- exactly what NetworkLGBMModel.predict_proba()
    returns, matching what inference/policy_engine.py's NETWORK_MALICIOUS_MIN
    was derived against. Verified against the real cortex_network model and
    the actual CSE-CIC-IDS2018 test split -- see docs/TECHNICAL_NOTES.md's ONNX export
    section for the measured max/mean absolute error.
    """
    import onnx
    import onnx.compose
    import onnx.numpy_helper
    import onnx.version_converter
    from onnxmltools import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType

    from models.network_lgbm import NetworkLGBMModel

    model = NetworkLGBMModel.load(model_path)

    lgbm_onnx = convert_lightgbm(
        model.booster, initial_types=[("input", FloatTensorType([None, num_features]))],
        target_opset=_LGBM_ONNX_OPSET, zipmap=False,
    )
    lgbm_onnx.graph.node.append(onnx.helper.make_node(
        "Slice", inputs=["probabilities"], outputs=["malicious_prob_col"],
        starts=[1], ends=[2], axes=[1],
    ))
    lgbm_onnx.graph.output.append(
        onnx.helper.make_tensor_value_info("malicious_prob_col", onnx.TensorProto.FLOAT, [None, 1])
    )
    lgbm_onnx = onnx.version_converter.convert_version(lgbm_onnx, _LGBM_ONNX_OPSET)

    if model.calibrator is not None:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType as SkFloatTensorType

        # The Platt calibrator is now fit on the booster's raw margins
        # (logits), not on its sigmoid probability. onnxmltools'
        # TreeEnsembleClassifier only exposes the post-sigmoid probability, so
        # reconstruct the margin in-graph before the calibrator subgraph:
        #   margin = logit(p) = log(p) - log(1 - p)
        # p is clipped off {0, 1} so log() stays finite when float32 rounds a
        # very confident score to exactly 0.0 / 1.0.
        _LOGIT_EPS = 1e-7
        lgbm_onnx.graph.initializer.extend([
            onnx.numpy_helper.from_array(np.array(_LOGIT_EPS, dtype=np.float32), name="logit_eps"),
            onnx.numpy_helper.from_array(np.array(1.0 - _LOGIT_EPS, dtype=np.float32), name="logit_1m_eps"),
            onnx.numpy_helper.from_array(np.array(1.0, dtype=np.float32), name="logit_one"),
        ])
        lgbm_onnx.graph.node.extend([
            onnx.helper.make_node("Clip", ["malicious_prob_col", "logit_eps", "logit_1m_eps"], ["logit_p"]),
            onnx.helper.make_node("Log", ["logit_p"], ["logit_log_p"]),
            onnx.helper.make_node("Sub", ["logit_one", "logit_p"], ["logit_1m_p"]),
            onnx.helper.make_node("Log", ["logit_1m_p"], ["logit_log_1m_p"]),
            onnx.helper.make_node("Sub", ["logit_log_p", "logit_log_1m_p"], ["malicious_margin_col"]),
        ])
        lgbm_onnx.graph.output.append(
            onnx.helper.make_tensor_value_info("malicious_margin_col", onnx.TensorProto.FLOAT, [None, 1])
        )

        calibrator_onnx = convert_sklearn(
            model.calibrator._lr, initial_types=[("float_input", SkFloatTensorType([None, 1]))],
            target_opset=_LGBM_ONNX_OPSET, options={id(model.calibrator._lr): {"zipmap": False}},
        )
        lgbm_onnx.ir_version = calibrator_onnx.ir_version
        calibrator_onnx = onnx.compose.add_prefix(calibrator_onnx, prefix="calib_")

        merged = onnx.compose.merge_models(
            lgbm_onnx, calibrator_onnx, io_map=[("malicious_margin_col", "calib_float_input")],
        )
        prob_source = "calib_probabilities"
    else:
        merged = lgbm_onnx
        prob_source = None

    if prob_source is not None:
        merged.graph.node.append(onnx.helper.make_node(
            "Slice", inputs=[prob_source, "final_slice_starts", "final_slice_ends", "final_slice_axes"],
            outputs=["malicious_probability_col"],
        ))
        merged.graph.initializer.extend([
            onnx.numpy_helper.from_array(np.array([1], dtype=np.int64), name="final_slice_starts"),
            onnx.numpy_helper.from_array(np.array([2], dtype=np.int64), name="final_slice_ends"),
            onnx.numpy_helper.from_array(np.array([1], dtype=np.int64), name="final_slice_axes"),
        ])
        merged.graph.node.append(onnx.helper.make_node(
            "Squeeze", inputs=["malicious_probability_col", "final_squeeze_axes"],
            outputs=["malicious_probability"],
        ))
        merged.graph.initializer.append(
            onnx.numpy_helper.from_array(np.array([1], dtype=np.int64), name="final_squeeze_axes")
        )
    else:
        merged.graph.node.append(onnx.helper.make_node(
            "Squeeze", inputs=["malicious_prob_col", "final_squeeze_axes"], outputs=["malicious_probability"],
        ))
        merged.graph.initializer.append(
            onnx.numpy_helper.from_array(np.array([1], dtype=np.int64), name="final_squeeze_axes")
        )
    del merged.graph.output[:]
    merged.graph.output.append(
        onnx.helper.make_tensor_value_info("malicious_probability", onnx.TensorProto.FLOAT, [None])
    )

    onnx.checker.check_model(merged)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(merged.SerializeToString())
    logger.info(
        "Network model exported to %s (calibrator %s, %.2f MB)",
        output_path, "chained into graph" if model.calibrator is not None else "absent",
        Path(output_path).stat().st_size / 1e6,
    )


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
