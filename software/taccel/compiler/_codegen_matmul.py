"""INT8 matmul lowering methods for CodeGenerator (mixin).

`CodeGenerator` (codegen.py) inherits from `_CodegenMatmulMixin` to pick up
the INT8 path's matmul emission methods. The W8A16 path is handled
separately by `taccel.compiler.w8a16_emit`; this mixin handles the
weight-INT8 / activation-INT8 lowering used by the legacy fake-quant /
DeiT pipelines.

Methods on this mixin
---------------------

  - `_emit_matmul`                   — dispatcher: picks simple / strip-mined /
                                       large-weight-tiled / fused-out-proj-accum
                                       based on input/weight sizing.
  - `_emit_matmul_simple`            — fits ABUF/ACCUM/WBUF in one shot
                                       (most Q/K/V/out_proj at typical scale).
  - `_emit_matmul_strip_mined`       — strip-mined over M when output > ABUF.
  - `_emit_matmul_large_weight_tiled`— tiled over (K, N) when weight > WBUF
                                       (fc1 at GPT-2 scale).
  - `_emit_fused_out_proj_accum`     — fused out_proj + residual1 (avoids
                                       intermediate DRAM round-trip).
  - `_materialize_activation_k_tile` — DMA-load an activation K-slice into ABUF
                                       (helper for large_weight_tiled).
  - `_emit_bias_add` / `_emit_bias_add_tile` — INT32 bias VADD into ACCUM.
  - `_spill_abuf_tile_rows_to_dram`  — spill ABUF strip rows back to DRAM-temp.
  - `_large_weight_tile_symbol` / `_large_weight_tile_plan` /
    `_large_weight_tiles_for_n`     — large-weight tile-plan helpers
                                       (also consumed by `_layout_weights`
                                       in codegen.py).

All methods access state through `self` — the mixin is composed into
`CodeGenerator` via standard Python MRO, so calls to `self._emit(...)`,
`self.mem.abuf.*`, `self._record_trace_event(...)` etc. resolve back into
the main class.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

import numpy as np

from ..isa.instructions import (
    BufCopyInsn,
    ConfigTileInsn,
    DequantAddInsn,
    GeluInsn,
    Instruction,
    MatmulInsn,
    RequantInsn,
    RequantPcInsn,
    ScaleMulInsn,
    SetScaleInsn,
    SyncInsn,
    VaddInsn,
)
from ..isa.opcodes import (
    ABUF_SIZE,
    ACCUM_SIZE,
    BUF_ABUF,
    BUF_ACCUM,
    BUF_WBUF,
    WBUF_SIZE,
)
from .ir import IRNode
from .tiler import pad_dim, tile_matmul, tile_strip_mine, TILE


UNIT = 16
STAGE4_M_TILE = TILE
STAGE4_MAX_N_TILE = 512


def _fp16_to_uint16(val: float) -> int:
    """Convert FP32 value to FP16 bit pattern as uint16 (little-endian)."""
    fp16 = np.float16(val)
    return int(np.frombuffer(fp16.tobytes(), dtype=np.uint16)[0])


class _CodegenMatmulMixin:
    """INT8 matmul lowering methods, composed into `CodeGenerator`."""

    @staticmethod
    def _large_weight_tile_symbol(weight_name: str, k_start: int, k_len: int,
                                  n_start: int, n_len: int) -> str:
        return f"{weight_name}__stage4_tile_k{k_start}_{k_len}_n{n_start}_{n_len}"

    @staticmethod
    def _large_weight_tile_plan(K_pad: int, N_pad: int) -> List[Tuple[int, int, int, int]]:
        """Return deterministic (k_start, k_len, n_start, n_len) tiles.

        Tiles are sized for a 16-row activation strip.  `N_tile` is capped at
        512 so the `d=384` FC1 case uses the intended 384x512 WBUF tile.  If the
        full K dimension still does not fit, split K and use MATMUL accumulate.
        """
        if K_pad <= 0 or N_pad <= 0:
            raise ValueError("large weight tile dimensions must be positive")

        max_n_by_accum = ACCUM_SIZE // (STAGE4_M_TILE * 4)
        max_n_by_abuf = ABUF_SIZE // STAGE4_M_TILE
        n_tile = min(N_pad, STAGE4_MAX_N_TILE, max_n_by_accum, max_n_by_abuf)
        n_tile = max(TILE, (n_tile // TILE) * TILE)

        while n_tile >= TILE:
            k_tile = (WBUF_SIZE // n_tile) // TILE * TILE
            if k_tile >= TILE:
                break
            n_tile //= 2
            n_tile = (n_tile // TILE) * TILE
        if n_tile < TILE:
            raise MemoryError("Unable to choose a Stage 4 N tile that fits WBUF")

        k_tile = max(TILE, (WBUF_SIZE // n_tile) // TILE * TILE)
        k_tile = min(K_pad, k_tile)
        k_tile = max(TILE, (k_tile // TILE) * TILE)

        tiles: List[Tuple[int, int, int, int]] = []
        for n_start in range(0, N_pad, n_tile):
            n_len = min(n_tile, N_pad - n_start)
            for k_start in range(0, K_pad, k_tile):
                k_len = min(k_tile, K_pad - k_start)
                tiles.append((k_start, k_len, n_start, n_len))
        return tiles

    def _large_weight_tiles_for_n(self, K_pad: int, N_pad: int,
                                  n_start: int, n_len: int) -> List[Tuple[int, int, int, int]]:
        return [
            tile for tile in self._large_weight_tile_plan(K_pad, N_pad)
            if tile[2] == n_start and tile[3] == n_len
        ]

    def _emit_matmul(self, node: IRNode):
        """Emit a standard linear matmul with optional bias."""
        # Phase 3 (c.1) M2.5-B: in W8A32 mode the simple-matmul path is
        # rewritten to use MAX_ABS_REDUCE_FP32 + QUANT_FP32_INT8 + MATMUL +
        # DEQUANT_ACCUM_FP32_SCALED. Sizing guardrails inside the helper
        # raise NotImplementedError for the strip-mined / large-weight
        # cases (those are M3). Other paths (qkt, attn_v) are still
        # blocked by the `generate()` guardrail.
        if self.use_fp16_activations:
            from .w8a16_emit import emit_matmul_w8a16
            emit_matmul_w8a16(self, node)
            return

        M, N = node.output_shape
        weight_name = node.inputs[1]
        weight_data = self.weight_data.get(weight_name)
        if weight_data is None:
            return

        w_q, w_scales = weight_data
        # Weights are stored transposed as [K_in_pad, N_out_pad]; K is dim 0.
        K = w_q.shape[0] if w_q.ndim == 2 else w_q.shape[0]

        # Check if strip-mining is needed:
        # - INT8 output exceeds ABUF (128 KB), OR
        # - INT32 intermediate exceeds ACCUM (64 KB), OR
        # - full weight matrix exceeds WBUF (Stage 4 d=384 FC1/FC2)
        strip_mine = node.attrs.get("strip_mine", False)
        output_bytes = pad_dim(M) * pad_dim(N)
        accum_bytes = pad_dim(M) * pad_dim(N) * 4  # INT32 intermediate
        weight_bytes = int(w_q.size)
        if output_bytes > ABUF_SIZE or accum_bytes > ACCUM_SIZE:
            strip_mine = True

        if weight_bytes > WBUF_SIZE or bool(node.attrs.get("stage4_weight_tiled", False)):
            self._emit_matmul_large_weight_tiled(node, M, N, K, w_q, w_scales)
            return

        if strip_mine:
            if (
                node.name.endswith("_out_proj")
                and self._fused_softmax_attnv_accum_out_proj_enabled_for(node.name)
            ):
                self._emit_fused_out_proj_accum(node, M, N, K, w_q, w_scales)
                return
            self._emit_matmul_strip_mined(node, M, N, K, w_q, w_scales)
        else:
            self._emit_matmul_simple(node, M, N, K, w_q, w_scales)

    def _emit_matmul_simple(self, node: IRNode, M: int, N: int, K: int,
                            w_q: np.ndarray, w_scales: np.ndarray):
        """Emit a simple (non-strip-mined) matmul."""
        weight_name = node.inputs[1]
        M_pad = pad_dim(M)
        N_pad = pad_dim(N)
        # Weights are stored transposed as [K_in_pad, N_out_pad]; K is dim 0.
        K = w_q.shape[0] if w_q.ndim == 2 else w_q.shape[0]
        K_pad = pad_dim(K)

        # Load weights to WBUF via allocator so live attn@V outputs are not clobbered.
        # (Previously hardcoded to offset 0, which overwrote head N-1's attn@V output
        # when loading head N's Q/K/V weights, destroying per-image information.)
        dram_off = self._dram_offset_required(weight_name, f"loading weight '{weight_name}'")
        weight_bytes = w_q.size
        w_alloc = self.mem.wbuf.alloc(f"_w_{weight_name}", weight_bytes)
        self._emit_dma_load(BUF_WBUF, w_alloc.offset_units, weight_bytes, 0, dram_off)
        self._emit(SyncInsn(resource_mask=0b001))  # wait DMA

        # CONFIG_TILE
        m_tiles = M_pad // TILE
        n_tiles = N_pad // TILE
        k_tiles = K_pad // TILE
        self._emit(ConfigTileInsn(M=m_tiles - 1, N=n_tiles - 1, K=k_tiles - 1))

        # Allocate ABUF regions
        act_alloc = self.mem.abuf.get(node.inputs[0]) or \
                    self.mem.abuf.alloc(node.inputs[0], M_pad * K_pad)
        act_off = act_alloc.offset_units
        input_act_scale = self.calibration_scales.get(node.inputs[0], 6.0 / 127.0)
        target_act_scale = self.calibration_scales.get(node.name, 6.0 / 127.0)
        mean_w_scale = float(np.mean(w_scales.astype(np.float32)))
        accum_real_scale = input_act_scale * mean_w_scale
        trace_projection_inputs = self._should_trace_attention_projection_debug(node.name)

        if trace_projection_inputs:
            # Snapshot the exact activation and weight tiles consumed by MATMUL.
            # If the first divergence moves to one of these traces, we know the
            # bug is upstream of the systolic datapath.
            self._record_trace_event(
                f"{node.name}__act_input",
                BUF_ABUF,
                act_off,
                M_pad,
                K_pad,
                M,
                K,
                "int8",
                input_act_scale,
            )
            self._record_trace_event(
                f"{node.name}__act_input_padded",
                BUF_ABUF,
                act_off,
                M_pad,
                K_pad,
                M_pad,
                K_pad,
                "int8",
                input_act_scale,
            )
            self._record_trace_event(
                f"{node.name}__weight_input",
                BUF_WBUF,
                w_alloc.offset_units,
                K_pad,
                N_pad,
                K,
                N,
                "int8",
                mean_w_scale,
            )

        # MATMUL
        self._emit(MatmulInsn(
            src1_buf=BUF_ABUF, src1_off=act_off,
            src2_buf=BUF_WBUF, src2_off=w_alloc.offset_units,
            dst_buf=BUF_ACCUM, dst_off=0,
            flags=0,
        ))
        self._emit(SyncInsn(resource_mask=0b010))  # wait systolic
        if trace_projection_inputs:
            self._record_trace_event(
                f"{node.name}__accum_pre_bias",
                BUF_ACCUM,
                0,
                M_pad,
                N_pad,
                M,
                N,
                "int32",
                accum_real_scale,
            )
            self._record_trace_event(
                f"{node.name}__accum_pre_bias_padded",
                BUF_ACCUM,
                0,
                M_pad,
                N_pad,
                M_pad,
                N_pad,
                "int32",
                accum_real_scale,
            )

        # Free weight allocation (no longer needed after MATMUL)
        self.mem.wbuf.free(f"_w_{weight_name}")

        # Bias add if present
        bias_name = node.attrs.get("bias")
        if bias_name:
            if bias_name not in self.prescaled_biases:
                raise KeyError(f"Missing prescaled bias '{bias_name}' for node '{node.name}'")
            self._emit_bias_add(
                bias_name,
                N_pad,
                m_tiles,
                trace_node_name=f"{node.name}__bias_input" if trace_projection_inputs else None,
                trace_scale=accum_real_scale,
                logical_cols=N,
            )

        if trace_projection_inputs:
            # Capture the post-bias accumulator state so the first-divergence
            # harness can distinguish MATMUL/bias errors from requantization errors.
            self._record_trace_event(
                f"{node.name}__accum",
                BUF_ACCUM,
                0,
                M_pad,
                N_pad,
                M,
                N,
                "int32",
                accum_real_scale,
            )
            self._record_trace_event(
                f"{node.name}__accum_padded",
                BUF_ACCUM,
                0,
                M_pad,
                N_pad,
                M_pad,
                N_pad,
                "int32",
                accum_real_scale,
            )

        if self._dequant_add_enabled_for_output(node.name):
            if weight_name in self.requant_pc_weight_names:
                raise ValueError(
                    f"DEQUANT_ADD residual path currently requires scalar output scale, got REQUANT_PC weight '{weight_name}'"
                )
            self.pending_accum_outputs[node.name] = {
                "accum_real_scale": accum_real_scale,
                "shape": (M_pad, N_pad, M, N),
            }
            self._record_trace_event(
                node.name,
                BUF_ACCUM,
                0,
                M_pad,
                N_pad,
                M,
                N,
                "int32",
                accum_real_scale,
            )
            return

        # Allocate output in ABUF
        out_alloc = self.mem.abuf.alloc(node.name, M_pad * N_pad)
        if weight_name in self.requant_pc_weight_names:
            pc_scale_name = f"{weight_name}__requant_pc"
            pc_scale_dram = self._dram_offset_required(
                pc_scale_name,
                f"loading REQUANT_PC scales for '{weight_name}'",
            )
            pc_scale_bytes = N_pad * 2
            pc_scale_alloc = self.mem.wbuf.alloc(f"_rqpc_{weight_name}", pc_scale_bytes)
            self._emit_dma_load(BUF_WBUF, pc_scale_alloc.offset_units, pc_scale_bytes, 0, pc_scale_dram)
            self._emit(SyncInsn(resource_mask=0b001))
            self._emit(RequantPcInsn(
                src1_buf=BUF_ACCUM, src1_off=0,
                src2_buf=BUF_WBUF, src2_off=pc_scale_alloc.offset_units,
                dst_buf=BUF_ABUF, dst_off=out_alloc.offset_units,
            ))
            self.mem.wbuf.free(f"_rqpc_{weight_name}")
        else:
            requant_scale_f = input_act_scale * mean_w_scale / max(target_act_scale, 1e-12)
            sreg = self._alloc_sreg()
            self._emit(SetScaleInsn(sreg=sreg, src_mode=0, imm16=_fp16_to_uint16(requant_scale_f)))
            self._emit(RequantInsn(
                src1_buf=BUF_ACCUM, src1_off=0,
                dst_buf=BUF_ABUF, dst_off=out_alloc.offset_units,
                sreg=sreg,
            ))
        self._record_trace_event(
            node.name,
            BUF_ABUF,
            out_alloc.offset_units,
            M_pad,
            N_pad,
            M,
            N,
            "int8",
            target_act_scale,
        )
        if trace_projection_inputs:
            self._record_trace_event(
                f"{node.name}__output_padded",
                BUF_ABUF,
                out_alloc.offset_units,
                M_pad,
                N_pad,
                M_pad,
                N_pad,
                "int8",
                target_act_scale,
            )
        if node.name == "classifier":
            self._record_trace_event(
                node.name,
                BUF_ACCUM,
                0,
                M_pad,
                N_pad,
                M,
                N,
                "int32",
                accum_real_scale if accum_real_scale is not None else input_act_scale,
            )

    def _emit_fused_out_proj_accum(self, node: IRNode, M: int, N: int, K: int,
                                   w_q: np.ndarray, w_scales: np.ndarray):
        """Emit strip-mined out_proj that accumulates per-head fused outputs directly.

        This avoids materializing the concatenated INT8 tensor. Each head output keeps
        its own attn_v scale in WBUF, is rescaled to the concat scale strip-by-strip,
        and contributes via MATMUL accumulate into one shared ACCUM tile.
        """
        weight_name = node.inputs[1]
        match = re.match(r"block(\d+)_out_proj$", node.name)
        if match is None:
            raise ValueError(f"Cannot infer block index for fused out_proj node '{node.name}'")
        block_idx = int(match.group(1))

        M_pad = pad_dim(M)
        N_pad = pad_dim(N)
        K_pad = pad_dim(K)
        strip_rows = TILE
        num_strips = M_pad // strip_rows
        head_names = [f"block{block_idx}_head{head_idx}_attn_v" for head_idx in range(self.config.n_head)]
        num_heads = len(head_names)
        head_dim = K // max(num_heads, 1)
        head_dim_pad = pad_dim(head_dim)
        concat_scale = self.calibration_scales.get(node.inputs[0], 6.0 / 127.0)
        fuse_residual1 = self._dequant_add_residual1_enabled_for_output(node.name)
        residual1_name = node.name.replace("_out_proj", "_residual1") if fuse_residual1 else None

        dram_off = self._dram_offset_required(weight_name, f"loading weight '{weight_name}'")
        weight_bytes = w_q.size
        w_alloc = self.mem.wbuf.alloc(f"_w_{weight_name}", weight_bytes)
        self._emit_dma_load(BUF_WBUF, w_alloc.offset_units, weight_bytes, 0, dram_off)
        self._emit(SyncInsn(resource_mask=0b001))

        pc_scale_alloc = None
        if weight_name in self.requant_pc_weight_names:
            pc_scale_name = f"{weight_name}__requant_pc"
            pc_scale_dram = self._dram_offset_required(
                pc_scale_name,
                f"loading REQUANT_PC scales for '{weight_name}'",
            )
            pc_scale_bytes = N_pad * 2
            pc_scale_alloc = self.mem.wbuf.alloc(f"_rqpc_{weight_name}", pc_scale_bytes)
            self._emit_dma_load(BUF_WBUF, pc_scale_alloc.offset_units, pc_scale_bytes, 0, pc_scale_dram)
            self._emit(SyncInsn(resource_mask=0b001))
        if fuse_residual1 and pc_scale_alloc is not None:
            raise ValueError(
                f"DEQUANT_ADD residual1 path currently requires scalar out_proj scale, got REQUANT_PC weight '{weight_name}'"
            )

        head_allocs = []
        for head_name in head_names:
            head_alloc = self.mem.wbuf.get(head_name)
            if head_alloc is None:
                raise KeyError(
                    f"Missing fused attention output '{head_name}' while emitting direct out_proj accumulation"
                )
            head_allocs.append(head_alloc)

        dram_temp_off = self.dram_temp_start + self.mem.alloc_dram_temp(
            f"{node.name}_temp", M_pad * N_pad
        )

        skip_alloc = None
        if fuse_residual1:
            skip_name = self._residual1_skip_name(node.name)
            skip_alloc = self.mem.abuf.get(skip_name)
            if skip_alloc is None:
                skip_alloc = self.mem.abuf.alloc(skip_name, M_pad * N_pad)
            mean_w_scale = float(np.mean(w_scales.astype(np.float32)))
            output_scale = self.calibration_scales.get(residual1_name, 6.0 / 127.0)
            skip_scale = self.calibration_scales.get(skip_name, 6.0 / 127.0)
            accum_rescale = concat_scale * mean_w_scale / max(output_scale, 1e-12)
            skip_rescale = skip_scale / max(output_scale, 1e-12)

        for s in range(num_strips):
            row_start = s * strip_rows
            logical_rows = max(0, min(strip_rows, M - row_start))

            for head_idx, (head_name, head_alloc) in enumerate(zip(head_names, head_allocs)):
                head_strip_alloc = self.mem.abuf.alloc(
                    f"{node.name}_head{head_idx}_strip{s}", strip_rows * head_dim_pad
                )
                src_off = head_alloc.offset_units + (s * strip_rows * head_dim_pad) // UNIT
                self._emit(BufCopyInsn(
                    src_buf=BUF_WBUF, src_off=src_off,
                    dst_buf=BUF_ABUF, dst_off=head_strip_alloc.offset_units,
                    length=(strip_rows * head_dim_pad) // UNIT,
                ))
                self._emit(SyncInsn(resource_mask=0b001))

                head_scale = self.calibration_scales.get(head_name, concat_scale)
                scale_mul = head_scale / max(concat_scale, 1e-12)
                if not np.isclose(scale_mul, 1.0, atol=1e-4, rtol=1e-4):
                    self._emit(ConfigTileInsn(M=0, N=head_dim_pad // TILE - 1, K=0))
                    scale_sreg = self._alloc_sreg()
                    self._emit(SetScaleInsn(sreg=scale_sreg, src_mode=0, imm16=_fp16_to_uint16(scale_mul)))
                    self._emit(ScaleMulInsn(
                        src1_buf=BUF_ABUF, src1_off=head_strip_alloc.offset_units,
                        dst_buf=BUF_ABUF, dst_off=head_strip_alloc.offset_units,
                        sreg=scale_sreg,
                    ))
                    self._emit(SyncInsn(resource_mask=0b100))

                self._emit(ConfigTileInsn(M=0, N=N_pad // TILE - 1, K=head_dim_pad // TILE - 1))
                weight_slice_off = w_alloc.offset_units + (head_idx * head_dim_pad * N_pad) // UNIT
                self._emit(MatmulInsn(
                    src1_buf=BUF_ABUF, src1_off=head_strip_alloc.offset_units,
                    src2_buf=BUF_WBUF, src2_off=weight_slice_off,
                    dst_buf=BUF_ACCUM, dst_off=0,
                    flags=0 if head_idx == 0 else 1,
                ))
                self._emit(SyncInsn(resource_mask=0b010))
                self.mem.abuf.free(f"{node.name}_head{head_idx}_strip{s}")

            bias_name = node.attrs.get("bias")
            if bias_name:
                if bias_name not in self.prescaled_biases:
                    raise KeyError(f"Missing prescaled bias '{bias_name}' for node '{node.name}'")
                self._emit_bias_add(bias_name, N_pad, 1)

            if fuse_residual1:
                mean_w_scale = float(np.mean(w_scales.astype(np.float32)))
                self._record_trace_event(
                    node.name,
                    BUF_ACCUM,
                    0,
                    strip_rows,
                    N_pad,
                    logical_rows,
                    N,
                    "int32",
                    concat_scale * mean_w_scale,
                    row_start=row_start,
                    full_rows=M,
                    full_cols=N,
                )
                dequant_sreg = self._alloc_sreg_pair()
                self._emit(SetScaleInsn(sreg=dequant_sreg, src_mode=0, imm16=_fp16_to_uint16(accum_rescale)))
                self._emit(SetScaleInsn(sreg=dequant_sreg + 1, src_mode=0, imm16=_fp16_to_uint16(skip_rescale)))
                self._emit(ConfigTileInsn(M=0, N=N_pad // TILE - 1, K=0))
                skip_strip_off = skip_alloc.offset_units + (s * strip_rows * N_pad) // UNIT
                self._emit(DequantAddInsn(
                    src1_buf=BUF_ACCUM, src1_off=0,
                    src2_buf=BUF_ABUF, src2_off=skip_strip_off,
                    dst_buf=BUF_ABUF, dst_off=skip_strip_off,
                    sreg=dequant_sreg,
                ))
                self._record_trace_event(
                    residual1_name,
                    BUF_ABUF,
                    skip_strip_off,
                    strip_rows,
                    N_pad,
                    logical_rows,
                    N,
                    "int8",
                    output_scale,
                    row_start=row_start,
                    full_rows=M,
                    full_cols=N,
                )
            else:
                strip_out_off = self.mem.abuf.alloc(
                    f"{node.name}_strip{s}", strip_rows * N_pad
                ).offset_units
                target_act_scale = self.calibration_scales.get(node.name, 6.0 / 127.0)
                if pc_scale_alloc is not None:
                    self._emit(RequantPcInsn(
                        src1_buf=BUF_ACCUM, src1_off=0,
                        src2_buf=BUF_WBUF, src2_off=pc_scale_alloc.offset_units,
                        dst_buf=BUF_ABUF, dst_off=strip_out_off,
                    ))
                else:
                    mean_w_scale = float(np.mean(w_scales.astype(np.float32)))
                    requant_scale_f = concat_scale * mean_w_scale / max(target_act_scale, 1e-12)
                    sreg = self._alloc_sreg()
                    self._emit(SetScaleInsn(sreg=sreg, src_mode=0, imm16=_fp16_to_uint16(requant_scale_f)))
                    self._emit(RequantInsn(
                        src1_buf=BUF_ACCUM, src1_off=0,
                        dst_buf=BUF_ABUF, dst_off=strip_out_off,
                        sreg=sreg,
                    ))
                self._record_trace_event(
                    node.name,
                    BUF_ABUF,
                    strip_out_off,
                    strip_rows,
                    N_pad,
                    logical_rows,
                    N,
                    "int8",
                    target_act_scale,
                    row_start=row_start,
                    full_rows=M,
                    full_cols=N,
                )
                strip_dram_off = dram_temp_off + s * strip_rows * N_pad
                self._emit_dma_store(BUF_ABUF, strip_out_off, strip_rows * N_pad, 2, strip_dram_off)
                self._emit(SyncInsn(resource_mask=0b001))
                self.mem.abuf.free(f"{node.name}_strip{s}")

        for head_name in head_names:
            self.mem.wbuf.free(head_name)
        self.mem.wbuf.free(f"_w_{weight_name}")
        if pc_scale_alloc is not None:
            self.mem.wbuf.free(f"_rqpc_{weight_name}")

        if fuse_residual1:
            skip_name = self._residual1_skip_name(node.name)
            self.precomputed_nodes.add(residual1_name)
            alloc = self.mem.abuf.allocations.pop(skip_name, None)
            if alloc is not None:
                alloc.name = residual1_name
                self.mem.abuf.allocations[residual1_name] = alloc
            return

        self.dram_temp_outputs[node.name] = dram_temp_off
        placeholder_bytes = min(strip_rows * N_pad, ABUF_SIZE)
        try:
            out_alloc = self.mem.abuf.alloc(node.name, placeholder_bytes)
            out_alloc.size_bytes = M_pad * N_pad
        except MemoryError:
            # The real tensor is DRAM-resident; downstream nodes must consult
            # dram_temp_outputs first.  Large-vocab lm_head rows can exceed ABUF.
            pass

    def _emit_matmul_large_weight_tiled(self, node: IRNode, M: int, N: int, K: int,
                                        w_q: np.ndarray, w_scales: Optional[np.ndarray]):
        """Emit a matmul whose full weight matrix is too large for WBUF.

        The canonical weight blob is [K, N] row-major.  Stage 4 pre-packs
        contiguous [K_tile, N_tile] blobs in DRAM, then accumulates K chunks
        into ACCUM for each 16-row activation strip and output-column tile.
        Outputs are always spilled to DRAM temp in full row-major [M_pad, N_pad]
        layout so downstream ops can consume them through the existing spill
        mechanism.
        """
        weight_name = node.inputs[1]
        use_requant_pc = weight_name in self.requant_pc_weight_names
        gelu_name = node.attrs.get("inline_gelu")
        gelu_from_accum = self._gelu_from_accum_enabled_for(node, gelu_name)
        if gelu_name and not gelu_from_accum:
            raise ValueError(
                f"Stage 4 large-weight striping expects standalone GELU for {node.name!r}; "
                "remove inline_gelu or handle GELU from DRAM temp"
            )
        if gelu_from_accum and use_requant_pc:
            raise ValueError(
                f"GELU-from-ACCUM requires scalar FC1 accumulator scale, got REQUANT_PC weight '{weight_name}'"
            )

        M_pad = pad_dim(M)
        N_pad = pad_dim(N)
        K_pad = pad_dim(K)
        strip_rows = STAGE4_M_TILE
        input_name = node.inputs[0]
        input_dram_off = self.dram_temp_outputs.get(input_name)
        input_from_dram = input_dram_off is not None
        act_alloc = None
        if not input_from_dram:
            act_alloc = self.mem.abuf.get(input_name) or self.mem.abuf.alloc(input_name, M_pad * K_pad)
        else:
            self.mem.abuf.free(input_name)

        mean_w_scale = float(np.mean(w_scales.astype(np.float32))) if w_scales is not None else 1.0
        input_act_scale = self.calibration_scales.get(input_name, 6.0 / 127.0)
        target_act_scale = self.calibration_scales.get(node.name, 6.0 / 127.0)
        requant_scale_f = input_act_scale * mean_w_scale / max(target_act_scale, 1e-12)
        fuse_residual = self._dequant_add_enabled_for_output(node.name)
        if use_requant_pc and fuse_residual:
            raise ValueError(
                f"DEQUANT_ADD residual path currently requires scalar output scale, got REQUANT_PC weight '{weight_name}'"
            )
        pc_scale_dram = None
        if use_requant_pc:
            pc_scale_dram = self._dram_offset_required(
                f"{weight_name}__requant_pc",
                f"loading REQUANT_PC scales for '{weight_name}'",
            )
        residual_name = None
        skip_name = None
        skip_alloc = None
        residual_output_scale = None
        skip_rescale = None
        accum_rescale = None
        if fuse_residual:
            if self._dequant_add_residual1_enabled_for_output(node.name):
                residual_name = node.name.replace("_out_proj", "_residual1")
                skip_name = self._residual1_skip_name(node.name)
            elif self._dequant_add_residual2_enabled_for_output(node.name):
                residual_name = node.name.replace("_fc2", "_residual2")
                skip_name = self._residual2_skip_name(node.name)
            else:
                raise ValueError(f"Cannot infer fused residual output for {node.name!r}")
            if skip_name in self.dram_temp_outputs:
                skip_alloc = self._load_dram_to_abuf(skip_name, M_pad, N_pad)
            else:
                skip_alloc = self.mem.abuf.get(skip_name) or self.mem.abuf.alloc(skip_name, M_pad * N_pad)
            residual_output_scale = self.calibration_scales.get(residual_name, 6.0 / 127.0)
            skip_scale = self.calibration_scales.get(skip_name, 6.0 / 127.0)
            accum_rescale = input_act_scale * mean_w_scale / max(residual_output_scale, 1e-12)
            skip_rescale = skip_scale / max(residual_output_scale, 1e-12)
        dram_temp_off = self.dram_temp_start + self.mem.alloc_dram_temp(
            f"{node.name}_temp", M_pad * N_pad
        )

        n_tiles_seen = []
        for _k_start, _k_len, n_start, n_len in self._large_weight_tile_plan(K_pad, N_pad):
            if not n_tiles_seen or n_tiles_seen[-1] != (n_start, n_len):
                n_tiles_seen.append((n_start, n_len))

        for row_start in range(0, M_pad, strip_rows):
            logical_rows = max(0, min(strip_rows, M - row_start))
            for n_start, n_len in n_tiles_seen:
                tile_alloc = self.mem.abuf.alloc(
                    f"{node.name}_tile_r{row_start}_n{n_start}", strip_rows * n_len
                )
                first_k = True
                for k_start, k_len, _, _ in self._large_weight_tiles_for_n(K_pad, N_pad, n_start, n_len):
                    act_off, staging_name = self._materialize_activation_k_tile(
                        input_name=input_name,
                        input_from_dram=input_from_dram,
                        input_dram_off=input_dram_off,
                        act_alloc=act_alloc,
                        row_start=row_start,
                        k_start=k_start,
                        k_len=k_len,
                        K_pad=K_pad,
                        strip_rows=strip_rows,
                    )
                    weight_tile_name = self._large_weight_tile_symbol(
                        weight_name, k_start, k_len, n_start, n_len
                    )
                    weight_dram = self._dram_offset_required(
                        weight_tile_name,
                        f"loading Stage 4 packed weight tile for '{weight_name}'",
                    )
                    w_alloc = self.mem.wbuf.alloc(
                        f"_w_{node.name}_k{k_start}_n{n_start}", k_len * n_len
                    )
                    self._emit_dma_load(BUF_WBUF, w_alloc.offset_units, k_len * n_len, 0, weight_dram)
                    self._emit(SyncInsn(resource_mask=0b001))

                    self._emit(ConfigTileInsn(
                        M=strip_rows // TILE - 1,
                        N=n_len // TILE - 1,
                        K=k_len // TILE - 1,
                    ))
                    self._emit(MatmulInsn(
                        src1_buf=BUF_ABUF,
                        src1_off=act_off,
                        src2_buf=BUF_WBUF,
                        src2_off=w_alloc.offset_units,
                        dst_buf=BUF_ACCUM,
                        dst_off=0,
                        flags=0 if first_k else 1,
                    ))
                    self._emit(SyncInsn(resource_mask=0b010))
                    first_k = False
                    self.mem.wbuf.free(w_alloc.name)
                    if staging_name is not None:
                        self.mem.abuf.free(staging_name)

                bias_name = node.attrs.get("bias")
                if bias_name:
                    if bias_name not in self.prescaled_biases:
                        raise KeyError(f"Missing prescaled bias '{bias_name}' for node '{node.name}'")
                    self._emit_bias_add_tile(bias_name, n_start, n_len)

                self._emit(ConfigTileInsn(M=0, N=n_len // TILE - 1, K=0))
                if fuse_residual:
                    skip_stage = self.mem.abuf.alloc(
                        f"{node.name}_skip_stage_r{row_start}_n{n_start}",
                        strip_rows * n_len,
                    )
                    row_units = n_len // UNIT
                    for r in range(strip_rows):
                        self._emit(BufCopyInsn(
                            src_buf=BUF_ABUF,
                            src_off=skip_alloc.offset_units + ((row_start + r) * N_pad + n_start) // UNIT,
                            dst_buf=BUF_ABUF,
                            dst_off=skip_stage.offset_units + r * row_units,
                            length=row_units,
                        ))
                        self._emit(SyncInsn(resource_mask=0b001))
                    dequant_sreg = self._alloc_sreg_pair()
                    self._emit(SetScaleInsn(
                        sreg=dequant_sreg,
                        src_mode=0,
                        imm16=_fp16_to_uint16(accum_rescale),
                    ))
                    self._emit(SetScaleInsn(
                        sreg=dequant_sreg + 1,
                        src_mode=0,
                        imm16=_fp16_to_uint16(skip_rescale),
                    ))
                    skip_tile_off = skip_alloc.offset_units + (row_start * N_pad + n_start) // UNIT
                    self._emit(DequantAddInsn(
                        src1_buf=BUF_ACCUM,
                        src1_off=0,
                        src2_buf=BUF_ABUF,
                        src2_off=skip_stage.offset_units,
                        dst_buf=BUF_ABUF,
                        dst_off=tile_alloc.offset_units,
                        sreg=dequant_sreg,
                    ))
                    for r in range(strip_rows):
                        self._emit(BufCopyInsn(
                            src_buf=BUF_ABUF,
                            src_off=tile_alloc.offset_units + r * row_units,
                            dst_buf=BUF_ABUF,
                            dst_off=skip_alloc.offset_units + ((row_start + r) * N_pad + n_start) // UNIT,
                            length=row_units,
                        ))
                        self._emit(SyncInsn(resource_mask=0b001))
                    self.mem.abuf.free(skip_stage.name)
                else:
                    if gelu_from_accum:
                        gelu_sreg = self._alloc_sreg_pair()
                        gelu_in_scale = input_act_scale * mean_w_scale
                        gelu_out_scale = self.calibration_scales.get(gelu_name, 1.0 / 127.0)
                        self._emit(SetScaleInsn(
                            sreg=gelu_sreg,
                            src_mode=0,
                            imm16=_fp16_to_uint16(gelu_in_scale),
                        ))
                        self._emit(SetScaleInsn(
                            sreg=gelu_sreg + 1,
                            src_mode=0,
                            imm16=_fp16_to_uint16(gelu_out_scale),
                        ))
                        self._emit(GeluInsn(
                            src1_buf=BUF_ACCUM,
                            src1_off=0,
                            dst_buf=BUF_ABUF,
                            dst_off=tile_alloc.offset_units,
                            sreg=gelu_sreg,
                        ))
                        self._emit(SyncInsn(resource_mask=0b100))
                        if n_len == N_pad:
                            self._record_trace_event(
                                gelu_name,
                                BUF_ABUF,
                                tile_alloc.offset_units,
                                strip_rows,
                                N_pad,
                                logical_rows,
                                N,
                                "int8",
                                gelu_out_scale,
                                row_start=row_start,
                                full_rows=M,
                                full_cols=N,
                            )
                    elif use_requant_pc:
                        pc_scale_alloc = self.mem.wbuf.alloc(
                            f"_rqpc_{node.name}_n{n_start}",
                            n_len * 2,
                        )
                        self._emit_dma_load(
                            BUF_WBUF,
                            pc_scale_alloc.offset_units,
                            n_len * 2,
                            0,
                            pc_scale_dram + n_start * 2,
                        )
                        self._emit(SyncInsn(resource_mask=0b001))
                        self._emit(RequantPcInsn(
                            src1_buf=BUF_ACCUM,
                            src1_off=0,
                            src2_buf=BUF_WBUF,
                            src2_off=pc_scale_alloc.offset_units,
                            dst_buf=BUF_ABUF,
                            dst_off=tile_alloc.offset_units,
                        ))
                        self.mem.wbuf.free(pc_scale_alloc.name)
                    else:
                        sreg = self._alloc_sreg()
                        self._emit(SetScaleInsn(sreg=sreg, src_mode=0, imm16=_fp16_to_uint16(requant_scale_f)))
                        self._emit(RequantInsn(
                            src1_buf=BUF_ACCUM,
                            src1_off=0,
                            dst_buf=BUF_ABUF,
                            dst_off=tile_alloc.offset_units,
                            sreg=sreg,
                        ))
                    if n_len == N_pad and not gelu_from_accum:
                        self._record_trace_event(
                            node.name,
                            BUF_ABUF,
                            tile_alloc.offset_units,
                            strip_rows,
                            N_pad,
                            logical_rows,
                            N,
                            "int8",
                            target_act_scale,
                            row_start=row_start,
                            full_rows=M,
                            full_cols=N,
                        )
                    self._spill_abuf_tile_rows_to_dram(
                        tile_alloc.offset_units,
                        dram_temp_off,
                        row_start=row_start,
                        n_start=n_start,
                        rows=strip_rows,
                        cols=n_len,
                        full_cols=N_pad,
                        addr_reg=2,
                    )
                self.mem.abuf.free(tile_alloc.name)

        if fuse_residual:
            self.precomputed_nodes.add(residual_name)
            alloc = self.mem.abuf.allocations.pop(skip_name, None)
            if alloc is not None:
                alloc.name = residual_name
                self.mem.abuf.allocations[residual_name] = alloc
            return

        self.dram_temp_outputs[node.name] = dram_temp_off
        placeholder_bytes = min(strip_rows * N_pad, ABUF_SIZE)
        try:
            out_alloc = self.mem.abuf.alloc(node.name, placeholder_bytes)
            out_alloc.size_bytes = M_pad * N_pad
        except MemoryError:
            # The real tensor is DRAM-resident; large-vocab lm_head rows can
            # exceed ABUF and are consumed through dram_temp_outputs.
            pass

    def _materialize_activation_k_tile(
        self,
        *,
        input_name: str,
        input_from_dram: bool,
        input_dram_off: Optional[int],
        act_alloc: Optional[Allocation],
        row_start: int,
        k_start: int,
        k_len: int,
        K_pad: int,
        strip_rows: int,
    ) -> Tuple[int, Optional[str]]:
        """Return ABUF offset for a contiguous [strip_rows, k_len] activation tile."""
        if not input_from_dram and act_alloc is None:
            raise ValueError("act_alloc is required for ABUF-resident activation tiles")

        if not input_from_dram and k_start == 0 and k_len == K_pad:
            return act_alloc.offset_units + (row_start * K_pad) // UNIT, None

        staging_name = f"{input_name}_stage4_k{k_start}_r{row_start}"
        staging = self.mem.abuf.alloc(staging_name, strip_rows * k_len)
        row_units = k_len // UNIT
        if input_from_dram:
            if input_dram_off is None:
                raise ValueError("input_dram_off is required for DRAM-resident activation tiles")
            if k_start == 0 and k_len == K_pad:
                self._emit_dma_load(
                    BUF_ABUF,
                    staging.offset_units,
                    strip_rows * K_pad,
                    1,
                    input_dram_off + row_start * K_pad,
                )
                self._emit(SyncInsn(resource_mask=0b001))
            else:
                for r in range(strip_rows):
                    self._emit_dma_load(
                        BUF_ABUF,
                        staging.offset_units + r * row_units,
                        k_len,
                        1,
                        input_dram_off + (row_start + r) * K_pad + k_start,
                    )
                    self._emit(SyncInsn(resource_mask=0b001))
        else:
            src_row_units = K_pad // UNIT
            for r in range(strip_rows):
                self._emit(BufCopyInsn(
                    src_buf=BUF_ABUF,
                    src_off=act_alloc.offset_units + (row_start + r) * src_row_units + k_start // UNIT,
                    dst_buf=BUF_ABUF,
                    dst_off=staging.offset_units + r * row_units,
                    length=row_units,
                ))
                self._emit(SyncInsn(resource_mask=0b001))
        return staging.offset_units, staging_name

    def _emit_bias_add_tile(self, bias_name: str, n_start: int, n_len: int):
        bias_dram_off = self._dram_offset_required(bias_name, f"loading bias '{bias_name}'") + n_start * 4
        bias_bytes = n_len * 4
        bias_alloc = self.mem.wbuf.alloc(f"bias_{bias_name}_n{n_start}", bias_bytes)
        self._emit_dma_load(BUF_WBUF, bias_alloc.offset_units, bias_bytes, 1, bias_dram_off)
        self._emit(SyncInsn(resource_mask=0b001))
        self._emit(ConfigTileInsn(M=0, N=n_len // TILE - 1, K=0))
        self._emit(VaddInsn(
            src1_buf=BUF_ACCUM,
            src1_off=0,
            src2_buf=BUF_WBUF,
            src2_off=bias_alloc.offset_units,
            dst_buf=BUF_ACCUM,
            dst_off=0,
        ))
        self.mem.wbuf.free(bias_alloc.name)

    def _spill_abuf_tile_rows_to_dram(self, src_off_units: int, dram_base: int, *,
                                      row_start: int, n_start: int, rows: int,
                                      cols: int, full_cols: int, addr_reg: int):
        row_units = cols // UNIT
        for r in range(rows):
            self._emit_dma_store(
                BUF_ABUF,
                src_off_units + r * row_units,
                cols,
                addr_reg,
                dram_base + (row_start + r) * full_cols + n_start,
            )
            self._emit(SyncInsn(resource_mask=0b001))

    def _emit_matmul_strip_mined(self, node: IRNode, M: int, N: int, K: int,
                                  w_q: np.ndarray, w_scales: np.ndarray):
        """Emit strip-mined matmul for large outputs (e.g., FC1 768-wide).

        Handles two input modes:
          - Input in ABUF: read strips directly (FC1 case)
          - Input in DRAM temp: load each strip from DRAM to ABUF first (FC2 case)
        """
        weight_name = node.inputs[1]
        M_pad = pad_dim(M)
        N_pad = pad_dim(N)
        # Weights stored transposed as [K_in_pad, N_out_pad]; K is dim 0.
        K = w_q.shape[0] if w_q.ndim == 2 else w_q.shape[0]
        K_pad = pad_dim(K)
        strip_rows = TILE
        fuse_residual1 = self._dequant_add_residual1_enabled_for_output(node.name)
        residual1_name = node.name.replace("_out_proj", "_residual1") if fuse_residual1 else None

        # Load weights to WBUF via allocator so live WBUF data is not clobbered.
        dram_off = self._dram_offset_required(weight_name, f"loading weight '{weight_name}'")
        weight_bytes = w_q.size
        w_alloc = self.mem.wbuf.alloc(f"_w_{weight_name}", weight_bytes)
        self._emit_dma_load(BUF_WBUF, w_alloc.offset_units, weight_bytes, 0, dram_off)
        self._emit(SyncInsn(resource_mask=0b001))
        pc_scale_alloc = None
        if weight_name in self.requant_pc_weight_names:
            pc_scale_name = f"{weight_name}__requant_pc"
            pc_scale_dram = self._dram_offset_required(
                pc_scale_name,
                f"loading REQUANT_PC scales for '{weight_name}'",
            )
            pc_scale_bytes = N_pad * 2
            pc_scale_alloc = self.mem.wbuf.alloc(f"_rqpc_{weight_name}", pc_scale_bytes)
            self._emit_dma_load(BUF_WBUF, pc_scale_alloc.offset_units, pc_scale_bytes, 0, pc_scale_dram)
            self._emit(SyncInsn(resource_mask=0b001))
        if fuse_residual1 and pc_scale_alloc is not None:
            raise ValueError(
                f"DEQUANT_ADD residual1 path currently requires scalar out_proj scale, got REQUANT_PC weight '{weight_name}'"
            )

        # Allocate DRAM temp for output strips
        dram_temp_off = self.dram_temp_start + self.mem.alloc_dram_temp(
            f"{node.name}_temp", M_pad * N_pad)

        num_strips = M_pad // strip_rows

        # Determine if input is in DRAM temp (spilled by a previous strip-mined op)
        input_name = node.inputs[0]
        input_dram_off = self.dram_temp_outputs.get(input_name)
        input_from_dram = input_dram_off is not None

        if not input_from_dram:
            act_alloc = self.mem.abuf.get(input_name) or \
                        self.mem.abuf.alloc(input_name, M_pad * K_pad)

        skip_alloc = None
        if fuse_residual1:
            skip_name = self._residual1_skip_name(node.name)
            skip_alloc = self.mem.abuf.get(skip_name)
            if skip_alloc is None:
                skip_alloc = self.mem.abuf.alloc(skip_name, M_pad * N_pad)
            input_act_scale = self.calibration_scales.get(node.inputs[0], 6.0 / 127.0)
            mean_w_scale = float(np.mean(w_scales.astype(np.float32)))
            output_scale = self.calibration_scales.get(residual1_name, 6.0 / 127.0)
            skip_scale = self.calibration_scales.get(skip_name, 6.0 / 127.0)
            accum_rescale = input_act_scale * mean_w_scale / max(output_scale, 1e-12)
            skip_rescale = skip_scale / max(output_scale, 1e-12)

        for s in range(num_strips):
            row_start = s * strip_rows
            logical_rows = max(0, min(strip_rows, M - row_start))
            # If input is in DRAM, load this strip to a temp ABUF region
            if input_from_dram:
                strip_input_alloc = self.mem.abuf.alloc(
                    f"{node.name}_instrip{s}", strip_rows * K_pad)
                strip_src_dram = input_dram_off + s * strip_rows * K_pad
                self._emit_dma_load(BUF_ABUF, strip_input_alloc.offset_units,
                                    strip_rows * K_pad, 3, strip_src_dram)
                self._emit(SyncInsn(resource_mask=0b001))
                strip_act_off = strip_input_alloc.offset_units
            else:
                strip_act_off = act_alloc.offset_units + (s * strip_rows * K_pad) // UNIT

            # CONFIG_TILE for one strip
            n_tiles = N_pad // TILE
            k_tiles = K_pad // TILE
            self._emit(ConfigTileInsn(M=0, N=n_tiles - 1, K=k_tiles - 1))

            # MATMUL for strip
            self._emit(MatmulInsn(
                src1_buf=BUF_ABUF, src1_off=strip_act_off,
                src2_buf=BUF_WBUF, src2_off=w_alloc.offset_units,
                dst_buf=BUF_ACCUM, dst_off=0,
                flags=0,
            ))
            self._emit(SyncInsn(resource_mask=0b010))

            # Free input strip if loaded from DRAM
            if input_from_dram:
                self.mem.abuf.free(f"{node.name}_instrip{s}")

            # Bias add
            bias_name = node.attrs.get("bias")
            if bias_name:
                if bias_name not in self.prescaled_biases:
                    raise KeyError(f"Missing prescaled bias '{bias_name}' for node '{node.name}'")
                self._emit_bias_add(bias_name, N_pad, 1)

            input_act_scale = self.calibration_scales.get(node.inputs[0], 6.0 / 127.0)
            gelu_name = node.attrs.get("inline_gelu")
            strip_out_off = None if fuse_residual1 else self.mem.abuf.alloc(
                f"{node.name}_strip{s}", strip_rows * N_pad).offset_units

            if self._gelu_from_accum_enabled_for(node, gelu_name):
                gelu_sreg = self._alloc_sreg_pair()
                # FC1 uses per-tensor quantization, so mean_w_scale is the exact
                # accumulator-domain real-value scale for all output channels.
                mean_w_scale = float(np.mean(w_scales.astype(np.float32)))
                gelu_in_scale = input_act_scale * mean_w_scale
                gelu_out_scale = self.calibration_scales.get(gelu_name, 1.0 / 127.0)
                self._emit(SetScaleInsn(sreg=gelu_sreg, src_mode=0,
                                        imm16=_fp16_to_uint16(gelu_in_scale)))
                self._emit(SetScaleInsn(sreg=gelu_sreg + 1, src_mode=0,
                                        imm16=_fp16_to_uint16(gelu_out_scale)))
                self._emit(ConfigTileInsn(M=0, N=N_pad // TILE - 1, K=0))
                self._emit(GeluInsn(
                    src1_buf=BUF_ACCUM, src1_off=0,
                    dst_buf=BUF_ABUF, dst_off=strip_out_off,
                    sreg=gelu_sreg,
                ))
                self._emit(SyncInsn(resource_mask=0b100))
                self._record_trace_event(
                    gelu_name,
                    BUF_ABUF,
                    strip_out_off,
                    strip_rows,
                    N_pad,
                    logical_rows,
                    N,
                    "int8",
                    gelu_out_scale,
                    row_start=row_start,
                    full_rows=M,
                    full_cols=N,
                )
            elif fuse_residual1:
                self._record_trace_event(
                    node.name,
                    BUF_ACCUM,
                    0,
                    strip_rows,
                    N_pad,
                    logical_rows,
                    N,
                    "int32",
                    input_act_scale * mean_w_scale,
                    row_start=row_start,
                    full_rows=M,
                    full_cols=N,
                )
                dequant_sreg = self._alloc_sreg_pair()
                self._emit(SetScaleInsn(sreg=dequant_sreg, src_mode=0, imm16=_fp16_to_uint16(accum_rescale)))
                self._emit(SetScaleInsn(sreg=dequant_sreg + 1, src_mode=0, imm16=_fp16_to_uint16(skip_rescale)))
                self._emit(ConfigTileInsn(M=0, N=N_pad // TILE - 1, K=0))
                skip_strip_off = skip_alloc.offset_units + (s * strip_rows * N_pad) // UNIT
                self._emit(DequantAddInsn(
                    src1_buf=BUF_ACCUM, src1_off=0,
                    src2_buf=BUF_ABUF, src2_off=skip_strip_off,
                    dst_buf=BUF_ABUF, dst_off=skip_strip_off,
                    sreg=dequant_sreg,
                ))
                self._record_trace_event(
                    residual1_name,
                    BUF_ABUF,
                    skip_strip_off,
                    strip_rows,
                    N_pad,
                    logical_rows,
                    N,
                    "int8",
                    output_scale,
                    row_start=row_start,
                    full_rows=M,
                    full_cols=N,
                )
            else:
                target_act_scale = self.calibration_scales.get(node.name, 6.0 / 127.0)
                if pc_scale_alloc is not None:
                    self._emit(RequantPcInsn(
                        src1_buf=BUF_ACCUM, src1_off=0,
                        src2_buf=BUF_WBUF, src2_off=pc_scale_alloc.offset_units,
                        dst_buf=BUF_ABUF, dst_off=strip_out_off,
                    ))
                else:
                    # Requantize strip: scale = input_act_scale × mean(weight_scale) / target_act_scale
                    mean_w_scale = float(np.mean(w_scales.astype(np.float32)))
                    requant_scale_f = input_act_scale * mean_w_scale / max(target_act_scale, 1e-12)
                    sreg = self._alloc_sreg()
                    self._emit(SetScaleInsn(sreg=sreg, src_mode=0, imm16=_fp16_to_uint16(requant_scale_f)))

                    self._emit(RequantInsn(
                        src1_buf=BUF_ACCUM, src1_off=0,
                        dst_buf=BUF_ABUF, dst_off=strip_out_off,
                        sreg=sreg,
                    ))
                self._record_trace_event(
                    node.name,
                    BUF_ABUF,
                    strip_out_off,
                    strip_rows,
                    N_pad,
                    logical_rows,
                    N,
                    "int8",
                    target_act_scale,
                    row_start=row_start,
                    full_rows=M,
                    full_cols=N,
                )

                if gelu_name:
                    gelu_sreg = self._alloc_sreg_pair()
                    gelu_in_scale = self.calibration_scales.get(node.name, 1.0 / 127.0)
                    gelu_out_scale = self.calibration_scales.get(gelu_name, 1.0 / 127.0)
                    self._emit(SetScaleInsn(sreg=gelu_sreg, src_mode=0,
                                            imm16=_fp16_to_uint16(gelu_in_scale)))
                    self._emit(SetScaleInsn(sreg=gelu_sreg + 1, src_mode=0,
                                            imm16=_fp16_to_uint16(gelu_out_scale)))
                    self._emit(ConfigTileInsn(M=0, N=N_pad // TILE - 1, K=0))
                    self._emit(GeluInsn(
                        src1_buf=BUF_ABUF, src1_off=strip_out_off,
                        dst_buf=BUF_ABUF, dst_off=strip_out_off,
                        sreg=gelu_sreg,
                    ))
                    self._emit(SyncInsn(resource_mask=0b100))
                    self._record_trace_event(
                        gelu_name,
                        BUF_ABUF,
                        strip_out_off,
                        strip_rows,
                        N_pad,
                        logical_rows,
                        N,
                        "int8",
                        gelu_out_scale,
                        row_start=row_start,
                        full_rows=M,
                        full_cols=N,
                    )

            if not fuse_residual1:
                # Spill strip (post-GELU if inline) to DRAM
                strip_dram_off = dram_temp_off + s * strip_rows * N_pad
                self._emit_dma_store(BUF_ABUF, strip_out_off, strip_rows * N_pad, 2, strip_dram_off)
                self._emit(SyncInsn(resource_mask=0b001))
                self.mem.abuf.free(f"{node.name}_strip{s}")

        # Free weight allocation (no longer needed after all strips are processed)
        self.mem.wbuf.free(f"_w_{weight_name}")
        if pc_scale_alloc is not None:
            self.mem.wbuf.free(f"_rqpc_{weight_name}")

        if fuse_residual1:
            self.precomputed_nodes.add(residual1_name)
            alloc = self.mem.abuf.allocations.pop(skip_name, None)
            if alloc is not None:
                alloc.name = residual1_name
                self.mem.abuf.allocations[residual1_name] = alloc
            return

        # Register output as DRAM-temp resident
        self.dram_temp_outputs[node.name] = dram_temp_off

        # Record placeholder allocation for downstream nodes
        placeholder_bytes = min(strip_rows * N_pad, ABUF_SIZE)
        try:
            out_alloc = self.mem.abuf.alloc(node.name, placeholder_bytes)
            out_alloc.size_bytes = M_pad * N_pad  # real size is in DRAM
        except MemoryError:
            # The real tensor is DRAM-resident; this placeholder is only a
            # compatibility hint for legacy downstream code.
            pass

    def _emit_bias_add(self, bias_name: str, N_pad: int, m_tiles: int,
                       trace_node_name: Optional[str] = None,
                       trace_scale: float = 1.0,
                       logical_cols: Optional[int] = None):
        """Emit bias load + VADD to accumulator."""
        bias_dram_off = self._dram_offset_required(bias_name, f"loading bias '{bias_name}'")
        bias_bytes = N_pad * 4  # INT32

        # Load bias to WBUF (temporary location after weights)
        bias_wbuf_off = self.mem.wbuf.alloc(f"bias_{bias_name}", bias_bytes).offset_units
        self._emit_dma_load(BUF_WBUF, bias_wbuf_off, bias_bytes, 1, bias_dram_off)
        self._emit(SyncInsn(resource_mask=0b001))
        if trace_node_name is not None:
            self._record_trace_event(
                trace_node_name,
                BUF_WBUF,
                bias_wbuf_off,
                1,
                N_pad,
                1,
                logical_cols if logical_cols is not None else N_pad,
                "int32",
                trace_scale,
            )

        # VADD: ACCUM += WBUF[bias] (INT32 add with broadcast)
        self._emit(VaddInsn(
            src1_buf=BUF_ACCUM, src1_off=0,
            src2_buf=BUF_WBUF, src2_off=bias_wbuf_off,
            dst_buf=BUF_ACCUM, dst_off=0,
        ))

        self.mem.wbuf.free(f"bias_{bias_name}")

