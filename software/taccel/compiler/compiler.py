"""Top-level compiler: PyTorch model → ProgramBinary."""
import struct
import numpy as np
from typing import Any, Dict, Optional, List, Tuple
from ..assembler.assembler import ProgramBinary
from ..isa.encoding import encode
from ..isa.opcodes import Opcode, OPCODE_SHIFT, OPCODE_MASK, A_IMM28_SHIFT, MASK_28BIT
from ..quantizer.quantize import quantize_weights, quantize_tensor
from ..quantizer.scales import ScalePropagator
from ..quantizer.calibrate import CalibrationResult, calibrate_model
from .ir import IRGraph
from .codegen import CodeGenerator
from .frontend import load_frontend
from .model_config import deit_tiny_config
from .tiler import pad_dim


class Compiler:
    """Compile a DeiT-tiny model to ProgramBinary."""

    def __init__(self):
        self.scale_prop = ScalePropagator()

    @staticmethod
    def _as_python_float_dict(scales: Dict[str, float]) -> Dict[str, float]:
        return {
            name: float(value)
            for name, value in sorted(scales.items())
        }

    @staticmethod
    def _weight_scale_kind(scales: Optional[np.ndarray]) -> str:
        if scales is None:
            return "none"
        scales_fp32 = scales.astype(np.float32)
        if len(scales_fp32) == 0:
            return "empty"
        if np.allclose(scales_fp32, scales_fp32[0]):
            return "per_tensor"
        return "per_channel"

    def _build_compiler_manifest(
        self,
        *,
        weight_data: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]],
        cal_scales: Dict[str, float],
        prescaled_biases: Dict[str, np.ndarray],
        codegen: CodeGenerator,
        bias_correction_biases: Optional[List[str]],
        weight_quantization_overrides: Optional[Dict[str, Dict[str, Any]]],
        gelu_from_accum: bool,
        gelu_from_accum_blocks: Optional[set],
        dequant_add_residual1_blocks: Optional[set],
        fused_softmax_attnv_blocks: Optional[set],
        fused_softmax_attnv_accum_out_proj: bool,
        requant_pc_qkv: bool,
        requant_pc_qkv_selection: Optional[set],
        requant_pc_fc1: bool,
        requant_pc_fc1_blocks: Optional[set],
        requant_pc_fc2: bool,
        requant_pc_fc2_blocks: Optional[set],
        requant_pc_out_proj: bool,
        requant_pc_out_proj_blocks: Optional[set],
        requant_pc_weight_names: set,
        requant_pc_scale_tables: Dict[str, np.ndarray],
        data_base: int,
        input_offset: int,
        pos_embed_patch_dram_offset: int,
        pos_embed_cls_dram_offset: int,
        cls_token_dram_offset: int,
    ) -> Dict[str, Any]:
        weight_manifest: Dict[str, Dict[str, Any]] = {}
        for name, (data, scales) in sorted(weight_data.items()):
            entry: Dict[str, Any] = {
                "stored_shape": list(data.shape),
                "dtype": str(data.dtype),
                "scale_kind": self._weight_scale_kind(scales),
                "uses_requant_pc": name in requant_pc_weight_names,
            }
            if scales is not None:
                scales_fp32 = scales.astype(np.float32)
                entry.update({
                    "scale_count": int(len(scales_fp32)),
                    "scale_min": float(np.min(scales_fp32)),
                    "scale_max": float(np.max(scales_fp32)),
                    "scale_mean": float(np.mean(scales_fp32)),
                })
            weight_manifest[name] = entry

        bias_manifest = {
            name: {
                "length": int(len(values)),
                "min": int(np.min(values)) if len(values) else 0,
                "max": int(np.max(values)) if len(values) else 0,
            }
            for name, values in sorted(prescaled_biases.items())
        }

        trace_nodes = sorted({
            event["node_name"]
            for events in codegen.trace_manifest.values()
            for event in events
        })

        return {
            "manifest_version": 1,
            "compiler": {
                "class": self.__class__.__name__,
                "options": {
                    "gelu_from_accum": bool(gelu_from_accum),
                    "bias_correction_biases": list(bias_correction_biases or []),
                    "weight_quantization_overrides": {
                        name: {
                            "mode": str(spec.get("mode", "custom")),
                            "per_channel": bool(spec.get("per_channel", True)),
                            "n_candidates": int(spec.get("n_candidates", 25)),
                            "alpha_min": float(spec.get("alpha_min", 0.5)),
                        }
                        for name, spec in sorted((weight_quantization_overrides or {}).items())
                    },
                    "gelu_from_accum_blocks": (
                        sorted(int(block_idx) for block_idx in gelu_from_accum_blocks)
                        if gelu_from_accum_blocks is not None else None
                    ),
                    "dequant_add_residual1_blocks": (
                        sorted(int(block_idx) for block_idx in dequant_add_residual1_blocks)
                        if dequant_add_residual1_blocks is not None else None
                    ),
                    "fused_softmax_attnv_blocks": (
                        sorted(int(block_idx) for block_idx in fused_softmax_attnv_blocks)
                        if fused_softmax_attnv_blocks is not None else None
                    ),
                    "fused_softmax_attnv_accum_out_proj": bool(fused_softmax_attnv_accum_out_proj),
                    "requant_pc_qkv": bool(requant_pc_qkv),
                    "requant_pc_qkv_selection": (
                        [
                            {"block": int(block), "projection": proj, "head": int(head)}
                            for block, proj, head in sorted(requant_pc_qkv_selection)
                        ]
                        if requant_pc_qkv_selection is not None else None
                    ),
                    "requant_pc_fc1": bool(requant_pc_fc1),
                    "requant_pc_fc1_blocks": (
                        sorted(int(block_idx) for block_idx in requant_pc_fc1_blocks)
                        if requant_pc_fc1_blocks is not None else None
                    ),
                    "requant_pc_fc2": bool(requant_pc_fc2),
                    "requant_pc_fc2_blocks": (
                        sorted(int(block_idx) for block_idx in requant_pc_fc2_blocks)
                        if requant_pc_fc2_blocks is not None else None
                    ),
                    "requant_pc_out_proj": bool(requant_pc_out_proj),
                    "requant_pc_out_proj_blocks": (
                        sorted(int(block_idx) for block_idx in requant_pc_out_proj_blocks)
                        if requant_pc_out_proj_blocks is not None else None
                    ),
                },
                "enabled_experiments": sorted(
                    name for name, enabled in (
                        ("gelu_from_accum", gelu_from_accum),
                        ("bias_correction", bool(bias_correction_biases)),
                        ("weight_quantization_override", bool(weight_quantization_overrides)),
                        ("dequant_add_residual1", dequant_add_residual1_blocks is not None),
                        ("fused_softmax_attnv", fused_softmax_attnv_blocks is not None),
                        ("fused_softmax_attnv_accum_out_proj", fused_softmax_attnv_accum_out_proj),
                        ("requant_pc_qkv", requant_pc_qkv),
                        ("requant_pc_fc1", requant_pc_fc1),
                        ("requant_pc_fc2", requant_pc_fc2),
                        ("requant_pc_out_proj", requant_pc_out_proj),
                    )
                    if enabled
                ),
            },
            "program_layout": {
                "data_base": int(data_base),
                "input_offset": int(input_offset),
                "pos_embed_patch_dram_offset": int(pos_embed_patch_dram_offset),
                "pos_embed_cls_dram_offset": int(pos_embed_cls_dram_offset),
                "cls_token_dram_offset": int(cls_token_dram_offset),
                "dram_layout": {
                    name: int(offset)
                    for name, offset in sorted(codegen.dram_layout.items())
                },
                "dram_temp_total": int(codegen.mem.dram_temp_total),
            },
            "calibration_scales": self._as_python_float_dict(cal_scales),
            "weights": weight_manifest,
            "biases": bias_manifest,
            "requant_pc_scale_tables": {
                name: [float(v) for v in table.astype(np.float32).tolist()]
                for name, table in sorted(requant_pc_scale_tables.items())
            },
            "trace_manifest_meta": {
                "pc_event_count": int(len(codegen.trace_manifest)),
                "event_count": int(sum(len(events) for events in codegen.trace_manifest.values())),
                "node_names": trace_nodes,
            },
        }

    def compile(self, state_dict: dict, calibration: Optional[CalibrationResult] = None,
                sample_inputs: Optional[list] = None,
                bias_corrections: Optional[Dict[str, np.ndarray]] = None,
                weight_quantization_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
                gelu_from_accum: bool = False,
                gelu_from_accum_blocks: Optional[set] = None,
                dequant_add_residual1_blocks: Optional[set] = None,
                fused_softmax_attnv_blocks: Optional[set] = None,
                fused_softmax_attnv_accum_out_proj: bool = False,
                requant_pc_qkv: bool = False,
                requant_pc_qkv_selection: Optional[set] = None,
                requant_pc_fc1: bool = False,
                requant_pc_fc1_blocks: Optional[set] = None,
                requant_pc_fc2: bool = False,
                requant_pc_fc2_blocks: Optional[set] = None,
                requant_pc_out_proj: bool = False,
                requant_pc_out_proj_blocks: Optional[set] = None) -> ProgramBinary:
        """Compile a DeiT-tiny model.

        Args:
            state_dict: PyTorch state dict with FP32 weights
            calibration: pre-computed calibration result, or None to use defaults
            sample_inputs: if calibration is None and sample_inputs provided, calibrate
        """
        frontend_result = load_frontend("deit_tiny")
        graph = frontend_result.graph
        config = frontend_result.config

        # Step 1: Quantize weights
        quant_weights = quantize_weights(
            state_dict,
            quantization_overrides=weight_quantization_overrides,
        )

        # Step 2: Get calibration scales
        if calibration is not None:
            cal_scales = calibration.scales
        else:
            # Use default scales based on weight ranges
            cal_scales = self._default_calibration_scales(state_dict)

        # Step 3: Pre-scale biases
        prescaled_biases = self._prescale_biases(
            state_dict,
            quant_weights,
            cal_scales,
            bias_corrections=bias_corrections,
        )

        if requant_pc_fc1 and gelu_from_accum:
            active_fc1_blocks = (
                set(range(config.n_layer)) if requant_pc_fc1_blocks is None else set(requant_pc_fc1_blocks)
            )
            active_gelu_blocks = (
                set(range(config.n_layer)) if gelu_from_accum_blocks is None else set(gelu_from_accum_blocks)
            )
            overlap = sorted(active_fc1_blocks.intersection(active_gelu_blocks))
            if overlap:
                raise ValueError(
                    "REQUANT_PC FC1 cannot overlap with GELU-from-ACCUM blocks because "
                    f"the ACCUM dequant scale is no longer uniform; overlapping blocks: {overlap}"
                )

        # Step 4: Prepare weight data for codegen
        weight_data = {}
        requant_pc_weight_names = set()
        requant_pc_scale_tables: Dict[str, np.ndarray] = {}
        for name, (data, scales) in quant_weights.items():
            if scales is not None:
                # Pad weight matrix to multiples of 16, then transpose to [K_in, N_out].
                # PyTorch stores weights as [N_out, K_in]; the systolic array reads src2 as
                # [K, N], so we must transpose before storing in DRAM.
                if data.ndim == 2:
                    data = np.pad(data,
                                  ((0, (16 - data.shape[0] % 16) % 16),
                                   (0, (16 - data.shape[1] % 16) % 16)),
                                  mode='constant')
                    data = np.ascontiguousarray(data.T)  # [N_pad, K_pad] → [K_pad, N_pad]
            weight_data[name] = (data, scales)

        # Add per-head Q/K/V weight slices using per-tensor quantization.
        # Using per-tensor (not per-channel) scale for each head ensures the single-scale
        # REQUANT in codegen is exact. Per-channel weights with mean(w_scales) REQUANT
        # introduces a per-channel weighting factor that distorts dot products, causing
        # wrong attention scores (e.g. 100% CLS-self instead of ~21%).
        for layer_idx in range(config.n_layer):
            prefix = f"vit.encoder.layer.{layer_idx}"
            ln1_scale = cal_scales.get(f"block{layer_idx}_ln1", 6.0 / 127.0)
            for proj in ["query", "key", "value"]:
                wname = f"{prefix}.attention.attention.{proj}.weight"
                if wname not in state_dict:
                    continue
                bname = f"{prefix}.attention.attention.{proj}.bias"
                bias_fp32_full = self._bias_fp32_with_correction(
                    state_dict,
                    bname,
                    bias_corrections,
                ) \
                    if bname in state_dict else None
                q_full_int8, q_full_scales = quant_weights[wname]
                for h in range(config.n_head):
                    head_weight_name = f"{wname}_h{h}"
                    use_requant_pc_qkv = requant_pc_qkv and (
                        requant_pc_qkv_selection is None
                        or (layer_idx, proj, h) in requant_pc_qkv_selection
                    )
                    if use_requant_pc_qkv:
                        w_h_int8 = q_full_int8[h * config.d_head:(h + 1) * config.d_head, :].astype(np.int8)
                        s_h = q_full_scales[h * config.d_head:(h + 1) * config.d_head].astype(np.float16)
                    else:
                        w_h_fp32 = state_dict[wname].numpy().astype(np.float32)[
                            h * config.d_head:(h + 1) * config.d_head, :
                        ]
                        max_abs_h = float(np.max(np.abs(w_h_fp32)))
                        w_scale_h = max(max_abs_h, 1e-8) / 127.0
                        w_h_int8 = np.clip(np.round(w_h_fp32 / w_scale_h), -128, 127).astype(np.int8)
                        s_h = np.full(w_h_int8.shape[0], w_scale_h, dtype=np.float16)
                    # Pad to multiples of 16
                    w_h_int8 = np.pad(w_h_int8,
                                      ((0, (16 - w_h_int8.shape[0] % 16) % 16),
                                       (0, (16 - w_h_int8.shape[1] % 16) % 16)),
                                      mode='constant')
                    if len(s_h) < w_h_int8.shape[0]:
                        s_h = np.pad(s_h, (0, w_h_int8.shape[0] - len(s_h)), constant_values=s_h[-1])
                    # Transpose [N_pad, K_pad] → [K_pad, N_pad] for systolic layout
                    w_h_int8 = np.ascontiguousarray(w_h_int8.T)
                    weight_data[head_weight_name] = (w_h_int8, s_h)

                    # Per-head bias uses the same quantized weight scales as the emitted matmul.
                    if bias_fp32_full is not None:
                        b_h = bias_fp32_full[h * config.d_head:(h + 1) * config.d_head]
                        s_h_arr = s_h[:len(b_h)].astype(np.float32)
                        bias_i32 = self.scale_prop.prescale_bias(
                            b_h, np.array([ln1_scale]), s_h_arr)
                        pad_len = (16 - len(bias_i32) % 16) % 16
                        if pad_len:
                            bias_i32 = np.pad(bias_i32, (0, pad_len), constant_values=0)
                        prescaled_biases[f"{bname}_h{h}"] = bias_i32
                    if use_requant_pc_qkv:
                        target_act_scale = cal_scales.get(
                            f"block{layer_idx}_head{h}_{proj}",
                            6.0 / 127.0,
                        )
                        requant_pc_weight_names.add(head_weight_name)
                        requant_pc_scale_tables[head_weight_name] = (
                            np.float32(ln1_scale) * s_h.astype(np.float32) / max(target_act_scale, 1e-12)
                        ).astype(np.float16)

        # Override out_proj, FC1, FC2 with per-tensor weight quantization.
        # quantize_weights() uses per-channel scales, but codegen's single-scale REQUANT
        # can only apply one scale to all output channels.  Using mean(per_ch_scales)
        # ≈ 0.5 × per_tensor_scale introduces a systematic ~2× amplitude error.
        # Per-tensor quantization makes REQUANT exact (mean(uniform) = the value).
        # Also fix bias prescaling: use calibrated layer-input scale (not hardcoded
        # 6/127) so the bias units match the accumulator units seen by REQUANT.
        for layer_idx in range(config.n_layer):
            prefix = f"vit.encoder.layer.{layer_idx}"
            b = f"block{layer_idx}"
            layer_specs = [
                # (dense_name_prefix, input_scale_key, output_scale_key, use_requant_pc)
                ("attention.output.dense", f"{b}_concat", f"{b}_out_proj", requant_pc_out_proj),
                (
                    "intermediate.dense",
                    f"{b}_ln2",
                    f"{b}_fc1",
                    requant_pc_fc1 and (
                        requant_pc_fc1_blocks is None or layer_idx in requant_pc_fc1_blocks
                    ),
                ),
                (
                    "output.dense",
                    f"{b}_gelu",
                    f"{b}_fc2",
                    requant_pc_fc2 and (
                        requant_pc_fc2_blocks is None or layer_idx in requant_pc_fc2_blocks
                    ),
                ),
            ]
            for dense_name, input_scale_key, output_scale_key, use_requant_pc in layer_specs:
                wname = f"{prefix}.{dense_name}.weight"
                bname = f"{prefix}.{dense_name}.bias"
                if wname not in state_dict:
                    continue
                act_scale_b = cal_scales.get(input_scale_key, 6.0 / 127.0)
                override_quant = (weight_quantization_overrides or {}).get(wname)
                if dense_name == "attention.output.dense" and (
                    not requant_pc_out_proj
                    or (
                        requant_pc_out_proj_blocks is not None
                        and layer_idx not in requant_pc_out_proj_blocks
                    )
                ):
                    use_requant_pc = False
                if use_requant_pc:
                    _, w_scales = quant_weights[wname]
                    if w_scales is None:
                        raise KeyError(f"Missing per-channel scales for REQUANT_PC weight '{wname}'")
                    w_scales_fp32 = w_scales.astype(np.float32)
                    target_act_scale = cal_scales.get(output_scale_key, 6.0 / 127.0)
                    requant_pc_weight_names.add(wname)
                    requant_pc_scale_tables[wname] = (
                        np.float32(act_scale_b) * w_scales_fp32 / max(target_act_scale, 1e-12)
                    ).astype(np.float16)
                    if bname in state_dict and hasattr(state_dict[bname], 'numpy'):
                        bias_fp32 = self._bias_fp32_with_correction(
                            state_dict,
                            bname,
                            bias_corrections,
                        )
                        bias_scales = w_scales_fp32[:len(bias_fp32)]
                        if len(bias_scales) < len(bias_fp32):
                            bias_scales = np.pad(
                                bias_scales,
                                (0, len(bias_fp32) - len(bias_scales)),
                                constant_values=bias_scales[-1],
                            )
                        bias_i32 = self.scale_prop.prescale_bias(
                            bias_fp32,
                            np.array([act_scale_b], dtype=np.float32),
                            bias_scales,
                        )
                        pad_len = (16 - len(bias_i32) % 16) % 16
                        if pad_len:
                            bias_i32 = np.pad(bias_i32, (0, pad_len), constant_values=0)
                        prescaled_biases[bname] = bias_i32
                    continue
                if override_quant is not None:
                    w_int8, override_scales = quant_weights[wname]
                    override_scales = override_scales.astype(np.float32)
                    if not np.allclose(override_scales, override_scales[0]):
                        raise ValueError(
                            f"Weight quantization override for '{wname}' must be per-tensor on scalar REQUANT paths"
                        )
                    w_scale = float(override_scales[0])
                else:
                    w_fp32 = state_dict[wname].numpy().astype(np.float32)
                    max_abs = float(np.max(np.abs(w_fp32)))
                    w_scale = max(max_abs, 1e-8) / 127.0
                    # Per-tensor INT8 quantization
                    w_int8 = np.clip(np.round(w_fp32 / w_scale), -128, 127).astype(np.int8)
                # Pad to multiples of 16
                w_int8 = np.pad(w_int8,
                                ((0, (16 - w_int8.shape[0] % 16) % 16),
                                 (0, (16 - w_int8.shape[1] % 16) % 16)),
                                mode='constant')
                # Uniform scale for all output channels (before transpose)
                n_out_pad = w_int8.shape[0]
                s = np.full(n_out_pad, w_scale, dtype=np.float16)
                # Transpose to [K_in_pad, N_out_pad] for systolic layout
                w_int8 = np.ascontiguousarray(w_int8.T)
                weight_data[wname] = (w_int8, s)

                # Override bias with calibrated act_scale × per-tensor w_scale so
                # that bias units match the REQUANT formula in codegen.
                if bname in state_dict and hasattr(state_dict[bname], 'numpy'):
                    bias_fp32 = self._bias_fp32_with_correction(
                        state_dict,
                        bname,
                        bias_corrections,
                    )
                    bias_i32 = self.scale_prop.prescale_bias(
                        bias_fp32,
                        np.array([act_scale_b], dtype=np.float32),
                        np.full(len(bias_fp32), w_scale, dtype=np.float32),
                    )
                    pad_len = (16 - len(bias_i32) % 16) % 16
                    if pad_len:
                        bias_i32 = np.pad(bias_i32, (0, pad_len), constant_values=0)
                    prescaled_biases[bname] = bias_i32

        # Override classifier with per-tensor weight quantization.
        # Classifier logits are read directly from ACCUM (INT32) before REQUANT.
        # Per-channel weights give acc[j] = dot[j] / (act_scale * w_scale[j]) where
        # w_scale[j] varies per class, distorting relative logit rankings.
        # Per-tensor makes acc[j] = dot[j] / (act_scale * w_scale_t) — all classes
        # at the same scale, so ranking is correct.
        if "classifier.weight" in state_dict:
            w_fp32 = state_dict["classifier.weight"].numpy().astype(np.float32)
            max_abs = float(np.max(np.abs(w_fp32)))
            w_scale = max(max_abs, 1e-8) / 127.0
            w_int8 = np.clip(np.round(w_fp32 / w_scale), -128, 127).astype(np.int8)
            w_int8 = np.pad(w_int8,
                            ((0, (16 - w_int8.shape[0] % 16) % 16),
                             (0, (16 - w_int8.shape[1] % 16) % 16)),
                            mode='constant')
            n_out_pad = w_int8.shape[0]
            s = np.full(n_out_pad, w_scale, dtype=np.float16)
            w_int8 = np.ascontiguousarray(w_int8.T)
            weight_data["classifier.weight"] = (w_int8, s)
            # Fix classifier bias: use final_ln_scale (= cls_extract scale) as act_scale
            if "classifier.bias" in state_dict and hasattr(state_dict["classifier.bias"], 'numpy'):
                bias_fp32 = self._bias_fp32_with_correction(
                    state_dict,
                    "classifier.bias",
                    bias_corrections,
                )
                act_scale_b = cal_scales.get("cls_extract", 6.0 / 127.0)
                bias_i32 = self.scale_prop.prescale_bias(
                    bias_fp32,
                    np.array([act_scale_b], dtype=np.float32),
                    np.full(len(bias_fp32), w_scale, dtype=np.float32),
                )
                pad_len = (16 - len(bias_i32) % 16) % 16
                if pad_len:
                    bias_i32 = np.pad(bias_i32, (0, pad_len), constant_values=0)
                prescaled_biases["classifier.bias"] = bias_i32

        # Quantize cls_token and pos_embed to INT8 using the embedding output scale.
        # These tensors are 3D in DeiT ([1,1,192] / [1,197,192]) so quantize_weights
        # silently skips them (ndim > 2 doesn't match any branch). We handle them
        # here by squeezing to 2D and quantizing at the pos_embed_add output scale so
        # codegen can load them as INT8 via DMA and use them in INT8 VADD.
        #
        # IMPORTANT: The scale must cover (cls/pos_embed + patches) after VADD, not
        # just cls/pos_embed alone. Using 6.0/127.0 causes heavy clipping because
        # pos_embed max_abs ≈ 6.9 and patch max_abs ≈ 6.0, so their sum ≈ 13.8.
        # cal_scales["pos_embed_add"] from actual calibration captures this full range.
        # Fallback: 14.0/127.0 safely covers the observed max of ~13.8.
        act_scale = cal_scales.get("pos_embed_add", 14.0 / 127.0)
        for k, v in state_dict.items():
            if not hasattr(v, 'numpy'):
                continue
            if 'cls_token' not in k and 'position_embeddings' not in k:
                continue
            t = v.numpy().astype(np.float32)
            while t.ndim > 2:
                t = t.squeeze(0)
            q = np.clip(np.round(t / act_scale), -128, 127).astype(np.int8)
            row_pad = (16 - q.shape[0] % 16) % 16
            col_pad = (16 - q.shape[1] % 16) % 16
            if row_pad or col_pad:
                q = np.pad(q, ((0, row_pad), (0, col_pad)), mode='constant')
            weight_data[k] = (q, None)

        # Step 6: Generate code
        codegen = CodeGenerator(
            weight_data,
            cal_scales,
            prescaled_biases,
            gelu_from_accum=gelu_from_accum,
            gelu_from_accum_blocks=gelu_from_accum_blocks,
            dequant_add_residual1_blocks=dequant_add_residual1_blocks,
            fused_softmax_attnv_blocks=fused_softmax_attnv_blocks,
            fused_softmax_attnv_accum_out_proj_blocks=(
                None if not fused_softmax_attnv_accum_out_proj else fused_softmax_attnv_blocks
            ),
            requant_pc_weight_names=requant_pc_weight_names,
            requant_pc_scale_tables=requant_pc_scale_tables,
            model_config=config,
        )
        instructions, dram_data = codegen.generate(graph)

        # Step 7: Extend dram_data to include DRAM temp region (strip-mine spill space)
        dram_temp_size = codegen.mem.dram_temp_total
        if dram_temp_size > 0:
            dram_data = dram_data + bytes(dram_temp_size)

        # Step 8: Assemble into ProgramBinary
        insn_bytes = bytearray()
        for insn in instructions:
            insn_bytes.extend(encode(insn))

        # Step 9: Build unified DRAM layout
        # data_base is the byte offset of the data section within the DRAM image,
        # aligned to 16 bytes so instructions don't bleed into parameter data.
        data_base = (len(insn_bytes) + 15) & ~15

        # Patch SET_ADDR_LO instructions: all DRAM addresses emitted by codegen are
        # data-relative (offset from start of dram_blob).  Add data_base to make them
        # DRAM-absolute (offset from start of the unified DRAM image).
        # SET_ADDR_HI needs no patching because all data fits within 24 bits, so the
        # high 28-bit half is always zero and adding data_base (<256 KB) still fits in
        # the low 28 bits with no carry.
        patched = bytearray(insn_bytes)
        for i in range(0, len(patched), 8):
            word = struct.unpack(">Q", patched[i:i + 8])[0]
            opcode_val = (word >> OPCODE_SHIFT) & OPCODE_MASK
            if opcode_val == Opcode.SET_ADDR_LO:
                old_imm28 = (word >> A_IMM28_SHIFT) & MASK_28BIT
                new_imm28 = (old_imm28 + data_base) & MASK_28BIT
                word = (word & ~(MASK_28BIT << A_IMM28_SHIFT)) | (new_imm28 << A_IMM28_SHIFT)
                patched[i:i + 8] = struct.pack(">Q", word)

        # input_offset: DRAM-absolute address where the host writes input patches
        input_patches_dram_off = codegen.dram_layout.get("__input_patches__", 0)
        input_offset = data_base + input_patches_dram_off

        # pos_embed_patch_dram_offset: DRAM-absolute start of patch rows (rows 1-196)
        # in the position_embeddings tensor.  Row 0 is the CLS position embedding and
        # must stay intact; rows 1-196 are the patch embeddings that the host folds
        # into the patch input (B3 preprocessing) to save one INT8 quantisation step.
        pos_emb_key = "vit.embeddings.position_embeddings"
        pos_emb_dram_start = codegen.dram_layout.get(pos_emb_key, None)
        if pos_emb_dram_start is not None:
            pos_embed_cls_dram_offset = data_base + pos_emb_dram_start
            # Row 0 (192 bytes) = CLS position embedding → skip it
            pos_embed_patch_dram_offset = data_base + pos_emb_dram_start + config.d_model
        else:
            pos_embed_cls_dram_offset = 0
            pos_embed_patch_dram_offset = 0
        cls_token_dram_start = codegen.dram_layout.get("vit.embeddings.cls_token", None)
        cls_token_dram_offset = data_base + cls_token_dram_start if cls_token_dram_start is not None else 0

        compiler_manifest = self._build_compiler_manifest(
            weight_data=weight_data,
            cal_scales=cal_scales,
            prescaled_biases=prescaled_biases,
            codegen=codegen,
            bias_correction_biases=sorted((bias_corrections or {}).keys()),
            weight_quantization_overrides=weight_quantization_overrides,
            gelu_from_accum=gelu_from_accum,
            gelu_from_accum_blocks=gelu_from_accum_blocks,
            dequant_add_residual1_blocks=dequant_add_residual1_blocks,
            fused_softmax_attnv_blocks=fused_softmax_attnv_blocks,
            fused_softmax_attnv_accum_out_proj=fused_softmax_attnv_accum_out_proj,
            requant_pc_qkv=requant_pc_qkv,
            requant_pc_qkv_selection=requant_pc_qkv_selection,
            requant_pc_fc1=requant_pc_fc1,
            requant_pc_fc1_blocks=requant_pc_fc1_blocks,
            requant_pc_fc2=requant_pc_fc2,
            requant_pc_fc2_blocks=requant_pc_fc2_blocks,
            requant_pc_out_proj=requant_pc_out_proj,
            requant_pc_out_proj_blocks=requant_pc_out_proj_blocks,
            requant_pc_weight_names=requant_pc_weight_names,
            requant_pc_scale_tables=requant_pc_scale_tables,
            data_base=data_base,
            input_offset=input_offset,
            pos_embed_patch_dram_offset=pos_embed_patch_dram_offset,
            pos_embed_cls_dram_offset=pos_embed_cls_dram_offset,
            cls_token_dram_offset=cls_token_dram_offset,
        )

        return ProgramBinary(
            instructions=bytes(patched),
            data=dram_data,
            entry_point=0,
            insn_count=len(instructions),
            data_base=data_base,
            input_offset=input_offset,
            pos_embed_patch_dram_offset=pos_embed_patch_dram_offset,
            pos_embed_cls_dram_offset=pos_embed_cls_dram_offset,
            cls_token_dram_offset=cls_token_dram_offset,
            trace_manifest=codegen.trace_manifest,
            compiler_manifest=compiler_manifest,
        )

    def _default_calibration_scales(self, state_dict: dict) -> Dict[str, float]:
        """Generate default calibration scales from weight statistics."""
        scales = {}
        # Default activation scale based on typical transformer activations
        default_scale = 6.0 / 127.0  # ~6 max abs is typical
        for name in state_dict:
            scales[name] = default_scale
            scales[f"{name}_input"] = default_scale
        return scales

    @staticmethod
    def _bias_fp32_with_correction(state_dict: dict,
                                   bias_name: str,
                                   bias_corrections: Optional[Dict[str, np.ndarray]]) -> np.ndarray:
        bias_fp32 = state_dict[bias_name].detach().cpu().numpy().astype(np.float32)
        if not bias_corrections or bias_name not in bias_corrections:
            return bias_fp32
        correction = np.asarray(bias_corrections[bias_name], dtype=np.float32)
        if correction.shape != bias_fp32.shape:
            raise ValueError(
                f"Bias correction for '{bias_name}' has shape {correction.shape}, "
                f"expected {bias_fp32.shape}"
            )
        return bias_fp32 + correction

    def _prescale_biases(self, state_dict: dict, quant_weights: dict,
                          cal_scales: dict,
                          bias_corrections: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
        """Pre-scale biases to INT32."""
        prescaled = {}
        default_act_scale = 6.0 / 127.0
        config = deit_tiny_config()

        for name, tensor in state_dict.items():
            if 'bias' not in name:
                continue
            if not hasattr(tensor, 'numpy'):
                continue

            bias_fp32 = self._bias_fp32_with_correction(state_dict, name, bias_corrections)

            # Find corresponding weight
            weight_name = name.replace('.bias', '.weight')
            if weight_name in quant_weights:
                _, w_scales = quant_weights[weight_name]
                if w_scales is not None:
                    act_scale = np.array([default_act_scale])
                    w_scales_fp32 = w_scales.astype(np.float32)
                    # Trim or pad w_scales to match bias length
                    if len(w_scales_fp32) > len(bias_fp32):
                        w_scales_fp32 = w_scales_fp32[:len(bias_fp32)]
                    elif len(w_scales_fp32) < len(bias_fp32):
                        w_scales_fp32 = np.pad(w_scales_fp32,
                                                (0, len(bias_fp32) - len(w_scales_fp32)),
                                                constant_values=w_scales_fp32[-1])
                    bias_int32 = self.scale_prop.prescale_bias(bias_fp32, act_scale, w_scales_fp32)
                    prescaled[name] = bias_int32

        # Add per-head bias slices for Q/K/V
        for layer_idx in range(config.n_layer):
            prefix = f"vit.encoder.layer.{layer_idx}"
            ln1_scale = cal_scales.get(f"block{layer_idx}_ln1", default_act_scale)
            for proj in ["query", "key", "value"]:
                bname = f"{prefix}.attention.attention.{proj}.bias"
                wname = f"{prefix}.attention.attention.{proj}.weight"
                if bname not in state_dict or wname not in quant_weights:
                    continue
                bias_fp32 = self._bias_fp32_with_correction(
                    state_dict,
                    bname,
                    bias_corrections,
                )
                _, w_scales_full = quant_weights[wname]
                if w_scales_full is None:
                    continue
                for h in range(config.n_head):
                    b_h = bias_fp32[h * config.d_head:(h + 1) * config.d_head]
                    s_h = w_scales_full[h * config.d_head:(h + 1) * config.d_head]
                    act_scale = np.array([ln1_scale], dtype=np.float32)
                    bias_i32 = self.scale_prop.prescale_bias(b_h, act_scale, s_h)
                    prescaled[f"{bname}_h{h}"] = bias_i32

        return prescaled
