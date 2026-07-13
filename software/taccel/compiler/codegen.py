"""IR → ISA instruction sequence code generator."""
import re
import struct
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from ..assembler.assembler import RelocationSite, RuntimeConfigAttnSite, RuntimePatchSite
from ..isa.opcodes import (
    ABUF_SIZE,
    ACCUM_SIZE,
    BUF_ABUF,
    BUF_WBUF,
    BUF_ACCUM,
    WBUF_SIZE,
)
from ..isa.instructions import (
    MatmulInsn, RequantInsn, RequantPcInsn, ScaleMulInsn, VaddInsn, SoftmaxInsn, LayernormInsn, GeluInsn,
    DequantAddInsn,
    SoftmaxAttnVInsn, ConfigAttnInsn, MaskedSoftmaxInsn, MaskedSoftmaxAttnVInsn,
    LoadInsn, StoreInsn, BufCopyInsn, SetAddrLoInsn, SetAddrHiInsn,
    ConfigTileInsn, SetScaleInsn, SyncInsn, NopInsn, HaltInsn, Instruction,
)
from .ir import IRNode, IRGraph
from .tiler import tile_matmul, tile_qkt, tile_strip_mine, pad_dim, TILE
from .memory_alloc import MemoryAllocator, Allocation
from .model_config import ModelConfig, deit_tiny_config
from .kv_cache import KVCacheLayout, normalize_kv_kind

UNIT = 16
STAGE4_M_TILE = TILE
STAGE4_MAX_N_TILE = 512


def stage4_forced_weights(graph, elem_bytes: int) -> set:
    """Weights whose consumer matmul will take the large-weight-TILED path.

    This MIRRORS the emitter's dispatch (`w8a16_emit/matmul`), which tiles when
    ANY of::

        output_bytes = M_pad*N_pad*elem_bytes  > ABUF_SIZE
        accum_bytes  = M_pad*N_pad*4           > ACCUM_SIZE
        weight_bytes                           > WBUF_SIZE   (weight-size test)

    The first two scale with **M_pad**, which is a property of the CONSUMER node,
    not of the weight. So a wide-M graph tiles weights a 1-row graph never does —
    e.g. lever I-b's dense multi-token prefill at M_pad=64 makes fc1's ACCUM tile
    64*512*4 = 128 KB > 64 KB and forces fc1 to tile, while the 1-row decode graph
    leaves it untiled. Staging keyed only on the weight's own size misses that and
    the emitter then KeyErrors on the missing tile symbol.

    `_layout_weights` also stages any weight with `data.size > WBUF_SIZE`, so this
    only has to cover the M_pad-driven cases.

    NOTE the two streams of a bundle must stage the SAME set, or their data blobs
    diverge and `decoder_bundle` rejects them. decoder_bundle therefore unions
    this across the prefill AND decode graphs and hands the union to both
    CodeGenerators. For every pre-I-b bundle the two graphs have the same matmul
    shapes, so the union is identical to each graph's own set and the staged blob
    is byte-unchanged.
    """
    out = set()
    for node in graph.nodes:
        if node.op != "matmul" or len(node.inputs) < 2:
            continue
        if node.attrs.get("stage4_weight_tiled", False):
            out.add(node.inputs[1])
            continue
        if len(node.output_shape) != 2:
            continue
        m_pad = pad_dim(int(node.output_shape[0]))
        n_pad = pad_dim(int(node.output_shape[1]))
        if (m_pad * n_pad * elem_bytes > ABUF_SIZE
                or m_pad * n_pad * 4 > ACCUM_SIZE):
            out.add(node.inputs[1])
    return out


def _fp16_to_uint16(val: float) -> int:
    """Convert FP32 value to FP16 bit pattern as uint16 (little-endian)."""
    fp16 = np.float16(val)
    # tobytes() on little-endian system gives LE bytes; interpret as uint16
    return int(np.frombuffer(fp16.tobytes(), dtype=np.uint16)[0])


def _set_addr(addr_reg: int, byte_addr: int) -> List[Instruction]:
    """Emit SET_ADDR_LO + SET_ADDR_HI to set a 56-bit DRAM address."""
    lo = byte_addr & 0xFFFFFFF
    hi = (byte_addr >> 28) & 0xFFFFFFF
    return [
        SetAddrLoInsn(addr_reg=addr_reg, imm28=lo),
        SetAddrHiInsn(addr_reg=addr_reg, imm28=hi),
    ]


class CodeGenerator:
    """Generate ISA instructions from IR graph.

    Matmul lowering dispatches to `taccel.compiler.w8a16_emit` (the only
    activation precision after the W8A8/DeiT path was retired). The
    large-weight tile-plan helpers (`_large_weight_tile_*`) are consumed
    by both `_layout_weights` (DRAM staging) and the W8A16 large-weight
    matmul emitter.
    """

    def __init__(self, weight_data: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]],
                 calibration_scales: Dict[str, float],
                 prescaled_biases: Dict[str, np.ndarray],
                 gelu_from_accum: bool = False,
                 gelu_from_accum_blocks: Optional[set] = None,
                 dequant_add_residual1_blocks: Optional[set] = None,
                 dequant_add_residual2_blocks: Optional[set] = None,
                 fused_softmax_attnv_blocks: Optional[set] = None,
                 fused_softmax_attnv_accum_out_proj_blocks: Optional[set] = None,
                 requant_pc_weight_names: Optional[set] = None,
                 requant_pc_scale_tables: Optional[Dict[str, np.ndarray]] = None,
                 model_config: Optional[ModelConfig] = None,
                 stream_name: str = "prefill",
                 kv_layout: Optional[KVCacheLayout] = None,
                 use_fp16_activations: bool = True,  # DEPRECATED: always True now; kept for back-compat
                 biases: Optional[Dict[str, np.ndarray]] = None,
                 weight_dtypes: Optional[Dict[str, str]] = None,
                 extra_stage4_weights: Optional[set] = None,
                 stage4_m_pad: int = STAGE4_M_TILE):
        """
        Args:
            weight_data: name → (quantized_data, per_channel_scales)
            calibration_scales: tensor_name → per-tensor activation scale
            prescaled_biases: name → INT32 pre-scaled bias array
            use_fp16_activations: DEPRECATED — kept for signature back-compat.
                Always treated as True. The W8A16 path is the only deployment
                surface; W8A8 INT8-round-trip optimizations (dequant_add residual,
                requant_pc, gelu_from_accum, fused_softmax_attnv) were retired
                with the DeiT tooling and stripped from RTL silicon 2026-05-23
                (see software/docs/isa_generation_freeze.md §3/§4).
            biases: Optional map of bias_name → FP32 1-D vector. The codegen
                stages each FP32 bias folded into the
                `__w8a16_pc_scale_and_bias` combined 2N FP16 blob
                consumed by DEQUANT_ACCUM_FP32_SCALED.
            weight_dtypes: Optional sidecar map `name → "int8" | "int4"`
                (default missing entries treated as "int8"). When a weight's
                dtype is "int4", `_layout_weights` packs the np.int8 storage
                via `decoder_bundle.pack_int4` before staging into the DRAM
                blob (2 nibbles per byte; layout per the `pack_int4`
                docstring — the RTL `weight_unpack.sv` and golden
                `memory.read_int4_tile` mirror this exact format).
                W4A16 plan Phase 2 (2026-05-24).

        Element size is FP16 (2 bytes/element) throughout the W8A16
        datapath; `self.elem_bytes = 2` and `self.fp_precision_flag = 1`
        are hardcoded.
        """
        self.weight_data = weight_data
        self.weight_dtypes = dict(weight_dtypes or {})
        self.config = model_config or deit_tiny_config()
        if stream_name not in ("prefill", "decode"):
            raise ValueError("stream_name must be 'prefill' or 'decode'")
        self.stream_name = stream_name
        self.kv_layout = kv_layout
        self.calibration_scales = calibration_scales
        self.prescaled_biases = prescaled_biases
        # FP biases for the W8A16 path. Empty dict when not provided so
        # emit_matmul_w8a16 raises a clear error if a bias is referenced
        # without a corresponding FP32 staging entry.
        self.biases: Dict[str, np.ndarray] = dict(biases or {})
        # W8A8 (INT8-activation) path was retired with the DeiT/RTL tooling.
        # W8A16 is the only activation precision now; field kept for back-compat
        # (always True) so any external code reading cg.use_fp16_activations
        # continues to work. Phase-A cleanup 2026-05-23 retired the W8A8 emission
        # branches; Phase-B will strip the corresponding gen-1 RTL.
        self.use_fp16_activations = True
        self.fp_precision = "fp16"
        self.elem_bytes: int = 2
        self.fp_precision_flag: int = 1
        # W8A16 force-disables the W8A8-only INT8-round-trip optimizations.
        # These optimizations targeted the now-retired gen-1 SFU path.
        if True:  # always — kept as a block for readability
            gelu_from_accum = False
            gelu_from_accum_blocks = set()
            dequant_add_residual1_blocks = set()
            dequant_add_residual2_blocks = set()
            fused_softmax_attnv_blocks = set()
            fused_softmax_attnv_accum_out_proj_blocks = set()
            requant_pc_weight_names = set()
            requant_pc_scale_tables = {}
        self.gelu_from_accum = gelu_from_accum
        self.gelu_from_accum_blocks = None if gelu_from_accum_blocks is None else set(gelu_from_accum_blocks)
        self.dequant_add_residual1_blocks = (
            None if dequant_add_residual1_blocks is None else set(dequant_add_residual1_blocks)
        )
        self.dequant_add_residual2_blocks = (
            None if dequant_add_residual2_blocks is None else set(dequant_add_residual2_blocks)
        )
        self.fused_softmax_attnv_blocks = None if fused_softmax_attnv_blocks is None else set(fused_softmax_attnv_blocks)
        self.fused_softmax_attnv_accum_out_proj_blocks = (
            None if fused_softmax_attnv_accum_out_proj_blocks is None
            else set(fused_softmax_attnv_accum_out_proj_blocks)
        )
        self.requant_pc_weight_names = set(requant_pc_weight_names or set())
        self.requant_pc_scale_tables = dict(requant_pc_scale_tables or {})
        self.mem = MemoryAllocator()
        self.instructions: List[Instruction] = []
        self.dram_layout: Dict[str, int] = {}  # name → dram byte offset
        self.dram_blob = bytearray()
        self.next_sreg_single = 0  # index into odd sreg pool (1,3,5,...)
        self.next_sreg_pair = 0    # index into even sreg pool (0,2,4,...)
        self.next_sreg_quad = 0    # index into quadruplet pool (0,4,8,12)
        # Track node outputs that live in DRAM temp (from strip-mined spills)
        # Maps output_name → DRAM byte offset of the spilled data
        self.dram_temp_outputs: Dict[str, int] = {}
        # M4-A: when an FP32 tile in `dram_temp_outputs` was spilled because
        # of ABUF pressure (not because the producer was strip-mined or
        # large-weight-tiled), we mark its byte size here so reload knows
        # the FP32 stride. INT8 reloads use M_pad * N_pad bytes; FP32
        # reloads use M_pad * N_pad * 4 bytes.
        self.dram_temp_fp32_outputs: Dict[str, int] = {}
        # Union of `stage4_forced_weights` across BOTH bundle streams, so the
        # prefill and decode codegens stage the SAME tiles and their data blobs
        # stay identical (decoder_bundle requires that). None => this graph's own
        # set only (byte-unchanged for every pre-I-b bundle).
        self.extra_stage4_weights: set = set(extra_stage4_weights or ())
        # Bundle-wide max M_pad, used to size the stage-4 N-tiles (their ACCUM
        # cost is M_pad*n_len*4). Staging and emit MUST agree on it or the tile
        # symbols diverge. Default 16 == the old hardcoded strip => byte-identical.
        self.stage4_m_pad: int = max(int(stage4_m_pad), TILE)
        # M4-A: exposed to per-emitter helpers (see `w8a16_emit.py`).
        # `last_uses` maps output_name → final node index that consumes it.
        # `current_node_idx` is the index of the node currently being emitted.
        # Set by `generate()` immediately before each `_emit_node` call.
        self.last_uses: Dict[str, int] = {}
        self.current_node_idx: int = -1
        # M4-A: spill an FP-precision residual tile to DRAM only when its
        # byte size exceeds this threshold. Tiny fixtures skip spill so
        # existing test instruction counts are preserved; real GPT-2
        # (d_model=768) triggers spill. 8 KB matches the d_model ≈ 256
        # FP16 tile boundary.
        self.fp_spill_threshold_bytes: int = 8192
        # Default to 0 so unit tests that bypass _layout_weights (and
        # would set dram_temp_start to a non-zero offset) don't crash
        # in _spill_fp32_tile_to_dram. _layout_weights overwrites this.
        self.dram_temp_start: int = 0
        # Optional trace metadata keyed by program counter. Each event tells the
        # simulator how to decode a node output back into FP32 for diagnostics.
        self.trace_manifest: Dict[int, List[Dict[str, Any]]] = {}
        self.pending_accum_outputs: Dict[str, Dict[str, Any]] = {}
        self.precomputed_nodes: set = set()
        self.relocation_sites: List[RelocationSite] = []
        self.runtime_patch_sites: List[RuntimePatchSite] = []
        self.runtime_config_attn_sites: List[RuntimeConfigAttnSite] = []

    def _dram_offset_required(self, name: str, context: str) -> int:
        """Return DRAM offset for a symbol or raise a clear error."""
        if name not in self.dram_layout:
            raise KeyError(f"Missing DRAM symbol '{name}' while {context}")
        return self.dram_layout[name]

    def generate(self, graph: IRGraph) -> Tuple[List[Instruction], bytes]:
        """Generate instructions for the entire IR graph.

        Returns (instructions, dram_data_blob).
        """
        # Phase 3 (c.1) M3-B: all W8A32 matmul ops are now lowered.
        # - `matmul`         → emit_matmul_w8a16       (M2.5-B)
        # - `matmul_qkt`     → emit_matmul_qkt_w8a16   (M3-A)
        # - `matmul_attn_v`  → emit_matmul_attn_v_w8a16 (M3-B)
        # The graph-level guardrail is empty. Per-emitter NotImplementedError
        # still fires for unsupported shapes (strip-mining, pad-row
        # zero-fill, masked-softmax CONFIG_ATTN — the last two are M3-C
        # scoped). Real GPT-2 graphs with masked softmax will hit the
        # ConfigError("CONFIG_ATTN not set") at simulator-execution time
        # until M3-C extends emit_softmax_fp32.
        # First pass: lay out weights in DRAM
        self._layout_weights(graph)

        # Compute last-use index for each node output so we can free ABUF
        last_uses = graph.compute_last_uses()
        # M4-A: expose to per-emitter helpers (`w8a16_emit.emit_layernorm_fp32`
        # spills FP32 residuals after reading them, and `emit_vadd_fp32`
        # reloads them from `dram_temp_outputs` and aliases the output
        # in-place into the sublayer input — both rely on this state).
        self.last_uses = last_uses

        # Second pass: emit instructions
        for idx, node in enumerate(graph.nodes):
            self.current_node_idx = idx
            # Compact ABUF before each layernorm to prevent fragmentation.
            # Strip-mined MLP ops leave holes; compacting before each LN gives
            # subsequent head matmuls a contiguous free region.
            if node.op == "layernorm":
                self._compact_abuf()
            self._emit_node(node)
            # Free ABUF allocations whose last use was this node
            for inp_name, last_idx in last_uses.items():
                if last_idx == idx:
                    alloc = self.mem.abuf.get(inp_name)
                    if alloc is not None:
                        self.mem.abuf.free(inp_name)
                    # Also free per-head sub-allocations (e.g. k_head0, q_head1)
                    for h in range(self.config.n_head):
                        self.mem.abuf.free(f"{inp_name}_head{h}")
                    # M4-A: if this tile was spilled to FP32 DRAM-temp and
                    # the consumer reloaded it, drop the dram_temp_outputs
                    # entry so we don't leak the symbol.
                    if inp_name in self.dram_temp_fp32_outputs:
                        del self.dram_temp_fp32_outputs[inp_name]
                        self.dram_temp_outputs.pop(inp_name, None)

        self.instructions.append(HaltInsn())
        return self.instructions, bytes(self.dram_blob)

    def _layout_weights(self, graph: IRGraph):
        """Pack all weights into DRAM data blob.

        W4A16 plan Phase 2: when `self.weight_dtypes.get(name) == "int4"`,
        pack the np.int8 storage (values in [-8, +7]) via
        `decoder_bundle.pack_int4` before staging — 2 nibbles per byte,
        last dim halved. The per-channel FP16 scales are NOT packed
        (they stay 2 bytes each). Default (missing entries) treats every
        weight as INT8, preserving byte-identical W8 bundle layout.
        """
        from .decoder_bundle import pack_int4

        # Staging MUST agree with the emitter's tiling decision or the emitter
        # KeyErrors on a tile symbol we never staged. `extra_stage4_weights` is
        # the UNION computed across BOTH streams by decoder_bundle — see
        # `stage4_forced_weights` for why a per-graph set is not enough.
        forced_stage4_weights = stage4_forced_weights(graph, self.elem_bytes)
        forced_stage4_weights |= set(self.extra_stage4_weights or ())
        offset = 0
        for name, (data, scales) in self.weight_data.items():
            if self.weight_dtypes.get(name) == "int4":
                if not (isinstance(data, np.ndarray) and data.dtype == np.int8
                        and data.ndim >= 1 and data.shape[-1] % 2 == 0):
                    raise ValueError(
                        f"weight {name!r} marked int4 but storage is "
                        f"dtype={getattr(data, 'dtype', '?')} shape="
                        f"{getattr(data, 'shape', '?')}; require np.int8 "
                        f"with even last-dim and values in [-8, +7]"
                    )
                blob = pack_int4(data).tobytes()
            else:
                blob = data.tobytes()
            self.dram_layout[name] = offset
            self.dram_blob.extend(blob)
            offset += len(blob)
            # Also store scales if present
            if scales is not None:
                scale_name = f"{name}__scales"
                self.dram_layout[scale_name] = offset
                scale_blob = scales.tobytes()
                self.dram_blob.extend(scale_blob)
                offset += len(scale_blob)
            if name in self.requant_pc_scale_tables:
                pc_scale_name = f"{name}__requant_pc"
                self.dram_layout[pc_scale_name] = offset
                pc_scale_blob = self.requant_pc_scale_tables[name].astype(np.float16).tobytes()
                self.dram_blob.extend(pc_scale_blob)
                offset += len(pc_scale_blob)

        # Stage 4 large-weight matmuls need output-column tiles, but the
        # canonical weight layout is row-major [K, N], so those tiles are
        # strided in the base blob. Pack deterministic contiguous tile blobs
        # before the temp region so runtime DMA can still use ordinary LOADs.
        for name, (data, _scales) in self.weight_data.items():
            if not (
                isinstance(data, np.ndarray)
                and data.ndim == 2
                and (data.size > WBUF_SIZE or name in forced_stage4_weights)
            ):
                continue
            K_pad, N_pad = int(data.shape[0]), int(data.shape[1])
            is_int4 = self.weight_dtypes.get(name) == "int4"
            for k_start, k_len, n_start, n_len in self._large_weight_tile_plan(
                    K_pad, N_pad, self.stage4_m_pad):
                tile_name = self._large_weight_tile_symbol(name, k_start, k_len, n_start, n_len)
                self.dram_layout[tile_name] = offset
                tile = np.ascontiguousarray(data[k_start:k_start + k_len, n_start:n_start + n_len])
                if is_int4:
                    # Pack along N (the last dim of the row-major tile).
                    # n_len is already guaranteed even by the tile plan; if
                    # ever odd we'd need padding (raise so we notice).
                    if tile.shape[-1] % 2 != 0:
                        raise ValueError(
                            f"int4 stage-4 tile {tile_name} has odd last-dim "
                            f"{tile.shape[-1]}; needs padding"
                        )
                    blob = pack_int4(tile.astype(np.int8, copy=False)).tobytes()
                else:
                    blob = tile.astype(np.int8, copy=False).tobytes()
                self.dram_blob.extend(blob)
                offset += len(blob)

        # Pre-scaled biases
        for name, bias_i32 in self.prescaled_biases.items():
            self.dram_layout[name] = offset
            blob = bias_i32.tobytes()
            self.dram_blob.extend(blob)
            offset += len(blob)

        # W8A32 (M2.5-B): stage raw FP32 biases under the symbol
        # `f"{name}__fp32"`. emit_matmul_w8a16 DMA-loads these to ABUF and
        # VADD_FP32s them onto the post-DEQUANT FP32 matmul output. The
        # INT32 prescaled version above is unusable in W8A32 because the
        # dynamic activation scale (max_abs/127) is only known at runtime.
        for name, bias_fp32 in self.biases.items():
            sym = f"{name}__fp32"
            self.dram_layout[sym] = offset
            blob = np.asarray(bias_fp32, dtype=np.float32).tobytes()
            self.dram_blob.extend(blob)
            offset += len(blob)

        if self.use_fp16_activations:
            # W8A16 (Phase 3 (c.2) M2): stage a combined PC-scale-plus-bias
            # FP16 blob per matmul weight. The DEQUANT_ACCUM_FP32_SCALED
            # epilogue under flags=1 reads `2N FP16` from src2 (N PC scales
            # then N bias values), folds bias in FP32 before casting to
            # FP16 once. This avoids FP16 double-rounding bias through a
            # separate VADD. Staging requires walking the matmul graph
            # nodes to pair each weight with its bias name (`node.attrs
            # ["bias"]`); matmuls without a bias get zero-padded bias half.
            #
            # We stage the same blob whether or not the codegen runs under
            # fp_precision="fp16" — under "fp32" it's dead DRAM (small
            # constant overhead), and pre-staging keeps the layout
            # comparison `prefill_data == decode_data` in the bundle
            # builder stable across precision flips.
            for node in graph.nodes:
                if node.op != "matmul":
                    continue
                weight_name = node.inputs[1] if len(node.inputs) > 1 else None
                if weight_name is None or weight_name not in self.weight_data:
                    continue
                data, scales = self.weight_data[weight_name]
                if not (isinstance(data, np.ndarray) and data.ndim == 2):
                    continue
                if scales is None:
                    continue
                N_pad = pad_dim(int(data.shape[1]))
                pc_fp16 = np.zeros(N_pad, dtype=np.float16)
                scales_fp16 = np.asarray(scales, dtype=np.float16).reshape(-1)
                n_take = min(len(scales_fp16), N_pad)
                pc_fp16[:n_take] = scales_fp16[:n_take]
                bias_name = node.attrs.get("bias")
                bias_fp16 = np.zeros(N_pad, dtype=np.float16)
                if bias_name is not None and bias_name in self.biases:
                    bias_src = np.asarray(self.biases[bias_name], dtype=np.float32).reshape(-1)
                    b_take = min(len(bias_src), N_pad)
                    bias_fp16[:b_take] = bias_src[:b_take].astype(np.float16)
                sym = f"{weight_name}__w8a16_pc_scale_and_bias"
                if sym in self.dram_layout:
                    # Multiple matmuls might share a weight (e.g., when a
                    # weight is reused across heads). Stage once; the bias
                    # half is derived from the first matmul that references
                    # the weight. If a subsequent matmul has a different
                    # bias, the codegen will raise a clear error at emit
                    # time when looking up the symbol. This is conservative
                    # and matches the W8A32 __fp32 staging convention.
                    continue
                self.dram_layout[sym] = offset
                blob = np.concatenate([pc_fp16, bias_fp16]).tobytes()
                self.dram_blob.extend(blob)
                offset += len(blob)

            # W8A32 (M3-A): stage a per-matmul_qkt FP16 PC scale vector.
            # emit_matmul_qkt_w8a16 uses DEQUANT_ACCUM_FP32 (the M1 variant —
            # no act_scale slot) to multiply the INT32 ACCUM by a single
            # composite factor `q_scale * k_scale * inv_sqrt_d_head` (static,
            # calibration-based). This folds the SCALE_MUL IR node into the
            # QKT epilogue — the same architecture the INT8 path uses (see
            # `qkt_in_scale` in _emit_qkt). Q and K themselves still benefit
            # from M2.5-A dynamic activation scaling in their parent Q/K
            # projection matmuls; only the QKT re-quant boundary is static.
            DEFAULT_ACT_SCALE = 6.0 / 127.0
            for node in graph.nodes:
                if node.op != "matmul_qkt":
                    continue
                q_input, k_input = node.inputs[0], node.inputs[1]
                q_scale = float(self.calibration_scales.get(q_input, DEFAULT_ACT_SCALE))
                k_scale = float(self.calibration_scales.get(k_input, DEFAULT_ACT_SCALE))
                inv_sqrt_d_head = float(
                    node.attrs.get("scale", self.config.d_head ** -0.5)
                )
                # Order of casts matters for FP16 bit-exactness — the
                # codegen and `test_emit_matmul_qkt_w8a16_pc_scale_bytes_
                # bit_exact_fp16_composite` test fixture both go FP32 ×
                # FP32 × FP32 → FP16. Don't refactor to a Python-float
                # (FP64) intermediate: it would shift the FP16 rounding
                # by 0-1 ULP for some values and silently break the test.
                composite_fp32 = (
                    np.float32(q_scale)
                    * np.float32(k_scale)
                    * np.float32(inv_sqrt_d_head)
                )
                # M4-G: stage at the MAX sequence length (model_config or
                # kv_layout) so prefill (key_len=1) and decode (key_len=seq)
                # share an identical DRAM blob. The DMA at runtime reads
                # only `valid_kv_len * 2` bytes anyway; the extra padding is
                # a small constant (~50 KB at max_seq_len=1024) per QKT.
                N_pad = pad_dim(
                    int(self.kv_layout.max_seq_len)
                    if self.kv_layout is not None
                    else int(self.config.max_seq_len)
                )
                pc_fp16 = np.full(N_pad, np.float16(composite_fp32), dtype=np.float16)
                sym = f"{node.name}__qkt_pc_scale"
                self.dram_layout[sym] = offset
                blob = pc_fp16.tobytes()
                self.dram_blob.extend(blob)
                offset += len(blob)

            # Lever B: stage the composite PC vector for each `qkt_dequant`
            # node (the split-out dequant of the packed QK^T). Identical
            # composite + cast order as the per-head `matmul_qkt` loop above —
            # the scale keys resolve to the same 6/127 defaults on the batched
            # path, so these bytes are byte-identical to the per-head bundle.
            for node in graph.nodes:
                if node.op != "qkt_dequant":
                    continue
                q_scale = float(self.calibration_scales.get(
                    node.attrs["q_scale_key"], DEFAULT_ACT_SCALE))
                k_scale = float(self.calibration_scales.get(
                    node.attrs["k_scale_key"], DEFAULT_ACT_SCALE))
                inv_sqrt_d_head = float(
                    node.attrs.get("scale", int(node.attrs["d_head"]) ** -0.5)
                )
                composite_fp32 = (
                    np.float32(q_scale)
                    * np.float32(k_scale)
                    * np.float32(inv_sqrt_d_head)
                )
                N_pad = pad_dim(
                    int(self.kv_layout.max_seq_len)
                    if self.kv_layout is not None
                    else int(self.config.max_seq_len)
                )
                pc_fp16 = np.full(N_pad, np.float16(composite_fp32), dtype=np.float16)
                sym = f"{node.name}__qkt_pc_scale"
                self.dram_layout[sym] = offset
                blob = pc_fp16.tobytes()
                self.dram_blob.extend(blob)
                offset += len(blob)

            # W8A32 (M3-B): stage a per-matmul_attn_v FP16 PC scale vector.
            # emit_matmul_attn_v_w8a16 uses DEQUANT_ACCUM_FP32 (M1, no
            # _SCALED) with a constant composite scale `sm_scale × v_scale`
            # — the softmax output's static calibration scale times V's
            # output scale from its parent emit_matmul_w8a16. No
            # 1/√d_head factor here (already applied by emit_matmul_qkt_w8a16
            # in its dequant epilogue, so the softmax input is already
            # the correctly-scaled attention scores).
            for node in graph.nodes:
                if node.op != "matmul_attn_v":
                    continue
                sm_input, v_input = node.inputs[0], node.inputs[1]
                sm_scale = float(self.calibration_scales.get(sm_input, DEFAULT_ACT_SCALE))
                v_scale = float(self.calibration_scales.get(v_input, DEFAULT_ACT_SCALE))
                # Cast order matches the test fixture: FP32 × FP32 → FP16.
                composite_fp32 = np.float32(sm_scale) * np.float32(v_scale)
                # N_pad = d_head_pad (the attention output column count).
                head_dim = int(node.output_shape[1])
                N_pad = pad_dim(head_dim)
                pc_fp16 = np.full(N_pad, np.float16(composite_fp32), dtype=np.float16)
                sym = f"{node.name}__attn_v_pc_scale"
                self.dram_layout[sym] = offset
                blob = pc_fp16.tobytes()
                self.dram_blob.extend(blob)
                offset += len(blob)

        # Zero-pad blob: used to mask attention padding rows (K and V) before QKT.
        # Padding rows 197-207 are zero in the input but LN(zero_row) = beta (non-zero),
        # which propagates through QKV projections. Zeroing K/V rows 197-207 eliminates
        # the beta-derived attention contribution from padding tokens.
        # Size: padding rows × row width. DeiT attention uses d_head; Stage 1
        # token embeddings use d_model for short fixed-sequence tests.
        #
        # W8A32 (M3-C): the same `__zero_pad__` symbol is used by the FP32
        # path's pad-row zero-fill (embedding lookup + K/V padding in the
        # QKT/attn_v emitters). Each FP32 element is 4 bytes/elem vs 1
        # byte/elem for INT8, so allocate 4× the size in W8A32 mode. The
        # INT8 path reads only the first N bytes it needs; the extra
        # padding is harmless. M3-prep skipped the embedding pad-row
        # zero-fill because no consumer existed; M3-C restores it for the
        # masked-softmax attention path.
        if self.config.embedding_kind == "patch_cls":
            _zero_pad_size = (pad_dim(self.config.max_seq_len) - self.config.max_seq_len) * self.config.d_head
        else:
            _zero_pad_size = (TILE - 1) * self.config.d_model
        if self.use_fp16_activations:
            # Scale by element size: W8A32 = 4 bytes/elem, W8A16 = 2.
            _zero_pad_size *= self.elem_bytes
            # M4-debug (padding-pollution fix): enlarge `__zero_pad__` to
            # cover the largest K_pad across all W8A32 matmuls so
            # emit_matmul_w8a16* can zero-fill input padding rows before
            # MAX_ABS_REDUCE_FP32. Padding rows accumulate non-zero values
            # (LN(zero_row)=beta, matmul outputs adding bias to all rows)
            # that inflate the dynamic activation scale and degrade INT8
            # precision catastrophically when M_pad >> M (decode/single-
            # token prefill).
            max_k_pad = self.config.d_model
            for _w_name, (data, _scales) in self.weight_data.items():
                if isinstance(data, np.ndarray) and data.ndim == 2:
                    max_k_pad = max(max_k_pad, pad_dim(int(data.shape[0])))
            _zero_pad_size = max(_zero_pad_size, (TILE - 1) * max_k_pad * self.elem_bytes)
        self.dram_layout["__zero_pad__"] = offset
        self.dram_blob.extend(bytes(_zero_pad_size))
        offset += _zero_pad_size

        # Input patches placeholder for patch+CLS encoders. Token/position
        # decoders do not use this Stage 1 ViT startup region.
        num_patches = self.config.max_seq_len - 1 if self.config.embedding_kind == "patch_cls" else 0
        _input_patches_size = num_patches * self.config.d_model
        self.dram_layout["__input_patches__"] = offset
        self.dram_blob.extend(bytes(_input_patches_size))
        offset += _input_patches_size

        # DRAM temp region for strip-mining starts after all weights
        self.dram_temp_start = offset
        # Pad DRAM to alignment
        while len(self.dram_blob) % UNIT != 0:
            self.dram_blob.append(0)

    def _alloc_sreg(self) -> int:
        """Allocate a single scale register from the odd pool (1,3,5,7,9,11,13).

        Singles and pairs use separate pools so they never overwrite each other.
        Scale registers are set immediately before use so wrapping is safe.
        """
        ODD_POOL = [1, 3, 5, 7, 9, 11, 13]
        reg = ODD_POOL[self.next_sreg_single % len(ODD_POOL)]
        self.next_sreg_single = (self.next_sreg_single + 1) % len(ODD_POOL)
        return reg

    def _alloc_sreg_pair(self) -> int:
        """Allocate a consecutive pair of scale registers from the even pool (0,2,4,...,12).

        Returns the lower (even) register; caller uses (reg, reg+1).
        Pairs and singles use separate pools so they never overwrite each other.
        """
        PAIR_POOL = [0, 2, 4, 6, 8, 10, 12]
        reg = PAIR_POOL[self.next_sreg_pair % len(PAIR_POOL)]
        self.next_sreg_pair = (self.next_sreg_pair + 1) % len(PAIR_POOL)
        return reg

    def _alloc_sreg_quad(self) -> int:
        """Allocate four consecutive scale registers."""
        QUAD_POOL = [0, 4, 8, 12]
        reg = QUAD_POOL[self.next_sreg_quad % len(QUAD_POOL)]
        self.next_sreg_quad = (self.next_sreg_quad + 1) % len(QUAD_POOL)
        return reg

    def _emit(self, insn: Instruction):
        self.instructions.append(insn)

    def _record_addr_site(self, lo_pc: int, hi_pc: int, addr_reg: int, *,
                          relocation_symbol: Optional[str] = None,
                          runtime_patch_kind: Optional[str] = None,
                          runtime_base_symbol: Optional[str] = None):
        if relocation_symbol is not None:
            self.relocation_sites.append(RelocationSite(
                stream=self.stream_name,
                local_lo_pc=lo_pc,
                local_hi_pc=hi_pc,
                addr_reg=addr_reg,
                symbol=relocation_symbol,
            ))
        if runtime_patch_kind is not None:
            if runtime_base_symbol is None:
                raise ValueError("runtime_base_symbol is required for runtime patch sites")
            self.runtime_patch_sites.append(RuntimePatchSite(
                stream=self.stream_name,
                kind=runtime_patch_kind,
                local_lo_pc=lo_pc,
                local_hi_pc=hi_pc,
                absolute_lo_pc=0,
                absolute_hi_pc=0,
                addr_reg=addr_reg,
                base_symbol=runtime_base_symbol,
            ))

    def _record_trace_event(self, node_name: str, buf_id: int, offset_units: int,
                            mem_rows: int, mem_cols: int,
                            logical_rows: int, logical_cols: int,
                            dtype: str, scale: float,
                            row_start: int = 0,
                            full_rows: Optional[int] = None,
                            full_cols: Optional[int] = None,
                            pc: Optional[int] = None,
                            capture_phase: str = "retire_cycle"):
        """Record how to snapshot a node tensor after an emitted instruction."""
        if logical_rows <= 0 or logical_cols <= 0:
            return
        if pc is None:
            pc = len(self.instructions) - 1
        event = {
            "node_name": node_name,
            "buf_id": int(buf_id),
            "offset_units": int(offset_units),
            "mem_rows": int(mem_rows),
            "mem_cols": int(mem_cols),
            "logical_rows": int(logical_rows),
            "logical_cols": int(logical_cols),
            "full_rows": int(full_rows if full_rows is not None else logical_rows),
            "full_cols": int(full_cols if full_cols is not None else logical_cols),
            "row_start": int(row_start),
            "dtype": dtype,
            "scale": float(scale),
            "when": "after",
            "capture_phase": capture_phase,
        }
        self.trace_manifest.setdefault(pc, []).append(event)

    def _gelu_from_accum_enabled_for(self, node: IRNode, gelu_name: Optional[str]) -> bool:
        """Return True when this FC1 -> GELU strip should consume ACCUM directly."""
        if not (gelu_name and self.gelu_from_accum):
            return False
        if self.gelu_from_accum_blocks is None:
            return True
        match = re.match(r"block(\d+)_", gelu_name or node.name)
        if match is None:
            return False
        return int(match.group(1)) in self.gelu_from_accum_blocks

    def _block_selected(self, name: str, selected_blocks: Optional[set]) -> bool:
        if selected_blocks is None:
            return False
        match = re.match(r"block(\d+)_", name)
        if match is None:
            return False
        return int(match.group(1)) in selected_blocks

    def _dequant_add_residual1_enabled_for_output(self, node_name: str) -> bool:
        return node_name.endswith("_out_proj") and self._block_selected(node_name, self.dequant_add_residual1_blocks)

    def _dequant_add_residual1_enabled_for_residual(self, node_name: str) -> bool:
        return node_name.endswith("_residual1") and self._block_selected(node_name, self.dequant_add_residual1_blocks)

    def _dequant_add_residual2_enabled_for_output(self, node_name: str) -> bool:
        return node_name.endswith("_fc2") and self._block_selected(node_name, self.dequant_add_residual2_blocks)

    def _dequant_add_residual2_enabled_for_residual(self, node_name: str) -> bool:
        return node_name.endswith("_residual2") and self._block_selected(node_name, self.dequant_add_residual2_blocks)

    def _dequant_add_enabled_for_output(self, node_name: str) -> bool:
        return (
            self._dequant_add_residual1_enabled_for_output(node_name)
            or self._dequant_add_residual2_enabled_for_output(node_name)
        )

    def _dequant_add_enabled_for_residual(self, node_name: str) -> bool:
        return (
            self._dequant_add_residual1_enabled_for_residual(node_name)
            or self._dequant_add_residual2_enabled_for_residual(node_name)
        )

    def _fused_softmax_attnv_accum_out_proj_enabled_for(self, node_name: str) -> bool:
        return self._block_selected(node_name, self.fused_softmax_attnv_accum_out_proj_blocks)

    def _should_trace_attention_projection_debug(self, node_name: str) -> bool:
        """Return True for per-head Q/K/V projections we may need to debug end-to-end."""
        return re.match(r"block\d+_head\d+_(query|key|value)$", node_name) is not None

    def _should_trace_ln1_padding_debug(self, node_name: str) -> bool:
        """Return True when a layernorm should emit padded input/output debug views."""
        return node_name == "block0_ln1"

    def _attention_mask_mode_for_qkt(self, node: IRNode, key_pad: int) -> Optional[int]:
        """Return CONFIG_ATTN mode for a masked QKT node, or None for legacy attention."""
        from .emit.attn import attention_mask_mode_for_qkt
        return attention_mask_mode_for_qkt(self, node, key_pad)

    def _emit_config_attn_for_qkt(self, node: IRNode, *, row_start: int,
                                  valid_kv_len: int, mode: int):
        from .emit.attn import emit_config_attn_for_qkt
        emit_config_attn_for_qkt(self, node, row_start=row_start,
                                 valid_kv_len=valid_kv_len, mode=mode)

    def _residual1_skip_name(self, out_proj_name: str) -> str:
        match = re.match(r"block(\d+)_out_proj$", out_proj_name)
        if match is None:
            raise ValueError(f"Cannot infer residual1 skip input from '{out_proj_name}'")
        block_idx = int(match.group(1))
        first_skip = "pos_embed_add" if self.config.embedding_kind == "patch_cls" else "tok_pos_add"
        return first_skip if block_idx == 0 else f"block{block_idx - 1}_residual2"

    def _residual2_skip_name(self, fc2_name: str) -> str:
        match = re.match(r"block(\d+)_fc2$", fc2_name)
        if match is None:
            raise ValueError(f"Cannot infer residual2 skip input from '{fc2_name}'")
        return f"block{int(match.group(1))}_residual1"

    @staticmethod
    def _large_weight_tile_symbol(weight_name: str, k_start: int, k_len: int,
                                  n_start: int, n_len: int) -> str:
        return f"{weight_name}__stage4_tile_k{k_start}_{k_len}_n{n_start}_{n_len}"

    @staticmethod
    def _large_weight_tile_plan(K_pad: int, N_pad: int,
                                m_pad: int = STAGE4_M_TILE) -> List[Tuple[int, int, int, int]]:
        """Return deterministic (k_start, k_len, n_start, n_len) tiles.

        Tiles are sized for an `m_pad`-row activation strip and capped at
        `STAGE4_MAX_N_TILE` / ACCUM / ABUF.

        **`m_pad` (lever I-b).** The emitter issues ALL M_pad rows in one MATMUL,
        so the ACCUM cost of an N-tile is `M_pad * n_len * 4` — it scales with the
        row count. This used to be hardcoded to a 16-row strip (`STAGE4_M_TILE`),
        which under-counts ACCUM for M_pad > 16: at M_pad=32 it fit only by luck
        (32*512*4 = 65536 = exactly ACCUM_SIZE) and at M_pad=64 it overflows 2x.
        Pass the real (bundle-wide max) M_pad and the N-cap shrinks to match.
        Splitting an N-tile is systolic-COST-NEUTRAL — the cost is
        `mt*nt*(64 + 130 + 17(kt-1))` and halving `n_len` halves `nt` while
        doubling the tile count — so this is nearly free.

        Both the DRAM staging (`_layout_weights`) and the emitter must pass the
        SAME `m_pad` or their tile symbols diverge; `CodeGenerator.stage4_m_pad`
        carries the bundle-wide value (default 16 => byte-identical to before).

        **Full-K preference (no MATMUL flags=1).** Whenever a single
        `[K_pad x n_tile]` INT8 weight tile fits WBUF for some
        `n_tile >= TILE`, this returns `k_tile == K_pad` so the emitter
        issues exactly one `MATMUL` (flags=0) per N-tile and never a
        K-split accumulate. The RTL systolic controller's flags=1 path is
        correct ONLY for a single 16x16 output tile (`clear_acc` is
        suppressed for the whole multi-(m,n)-tile walk and `pe_acc` has no
        per-tile preload) — see #115/#116. K-tiling is mathematically
        identical (integer matmul is tiling-invariant), so a full-K tile is
        a free, numerically-equivalent way to stay on the byte-exact-proven
        flags=0 per-N-tile path. For GPT-2 124M this turns every
        weight-tiled matmul (out_proj, fc1, lm_head; K_pad=768) into
        flags=0-only; fc2 (K_pad=3072) stays K-split but takes the
        large-input streaming path, which is flags=0-only by construction,
        so flags=1 is never reached anywhere in the frozen bundle.

        The K-split branch (flags=1 for a weight-tiled consumer) is
        retained only for the full-K-infeasible regime (huge K_pad);
        unreachable as flags=1 for GPT-2 124M, still #115-buggy until #116.

        Consumed by `_layout_weights` (DRAM staging) and the W8A16
        large-weight-tiled matmul emitter (`w8a16_emit.matmul`).
        """
        if K_pad <= 0 or N_pad <= 0:
            raise ValueError("large weight tile dimensions must be positive")

        max_n_by_accum = ACCUM_SIZE // (m_pad * 4)
        max_n_by_abuf = ABUF_SIZE // m_pad
        n_cap = min(N_pad, STAGE4_MAX_N_TILE, max_n_by_accum, max_n_by_abuf)
        n_cap = max(TILE, (n_cap // TILE) * TILE)

        # Full-K is viable only when (a) some full-K weight tile fits WBUF
        # and (b) one full-K activation strip is a modest ABUF transient.
        # When (b) fails the input is streamed K-tile by K-tile (the
        # fc2-style large-input path: flags=0 only, FP32 partial-sum
        # accumulate — NOT the #115-buggy flags=1 path) and MUST keep a
        # small k_tile, so we fall back to the K-split plan there. The plan
        # must stay a pure function of (K_pad, N_pad) for staging<->emit
        # symbol consistency, so the gate is purely K_pad-based.
        # Double-buffer budget: size weight tiles to at most HALF of WBUF so
        # TWO consecutive weight tiles fit simultaneously. This lets the
        # large-weight-tiled matmul emitter prefetch the next tile's DMA load
        # into the alternate WBUF region while the systolic array streams the
        # current tile through the dedicated WBUF read port (DMA‖Systolic
        # overlap; see [[dma_compute_overlap]]). pc_scale is never resident
        # alongside two weight tiles (it loads after the consumed tile frees),
        # so HALF is sufficient. Must stay a pure fn of (K_pad,N_pad) so the
        # DRAM staging (_layout_weights) and emit agree on tile symbols.
        wbuf_budget = WBUF_SIZE // 2
        full_k_fits_abuf = m_pad * K_pad * 4 <= ABUF_SIZE
        max_n_full_k = (wbuf_budget // K_pad) // TILE * TILE
        if full_k_fits_abuf and max_n_full_k >= TILE:
            n_tile = max(TILE, (min(n_cap, max_n_full_k) // TILE) * TILE)
            k_tile = K_pad
        else:
            # Full-K infeasible (huge K, or K_pad alone exceeds WBUF at
            # n_tile == TILE): split K. The weight-tiled consumer would use
            # MATMUL flags=1 here (still #115-buggy until #116); the
            # streaming consumer uses flags=0 and is correct. For GPT-2
            # 124M only K=3072 fc2 lands here, and it takes the streaming
            # (flags=0) path, so flags=1 is never reached.
            n_tile = n_cap
            while n_tile >= TILE:
                k_tile = (wbuf_budget // n_tile) // TILE * TILE
                if k_tile >= TILE:
                    break
                n_tile //= 2
                n_tile = (n_tile // TILE) * TILE
            if n_tile < TILE:
                raise MemoryError("Unable to choose a Stage 4 N tile that fits WBUF")
            k_tile = max(TILE, (wbuf_budget // n_tile) // TILE * TILE)
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
            tile for tile in self._large_weight_tile_plan(K_pad, N_pad, self.stage4_m_pad)
            if tile[2] == n_start and tile[3] == n_len
        ]

    def _emit_matmul(self, node: IRNode):
        """Emit a standard linear matmul through the W8A16 lowering.

        The legacy INT8 (W8A8) matmul path was retired with the DeiT/RTL
        tooling; `use_fp16_activations` is always True now, so this
        unconditionally dispatches to `emit_matmul_w8a16`, which handles
        the simple / large-weight-tiled / large-input-streaming cases
        internally based on tile sizing.
        """
        from .w8a16_emit import emit_matmul_w8a16
        emit_matmul_w8a16(self, node)

    def _emit_node(self, node: IRNode):
        """Emit instructions for a single IR node."""
        op = node.op
        if op == "matmul":
            self._emit_matmul(node)
        elif op == "matmul_qkt":
            self._emit_qkt(node)
        elif op == "packed_qkt_matmul":
            from .w8a16_emit import emit_packed_qkt_matmul
            emit_packed_qkt_matmul(self, node)
        elif op == "qkt_dequant":
            from .w8a16_emit import emit_qkt_dequant
            emit_qkt_dequant(self, node)
        elif op == "matmul_attn_v":
            self._emit_attn_v(node)
        elif op == "layernorm":
            self._emit_layernorm(node)
        elif op == "softmax":
            self._emit_softmax(node)
        elif op == "gelu":
            self._emit_gelu(node)
        elif op == "scale_mul":
            self._emit_scale_mul(node)
        elif op == "vadd":
            self._emit_vadd(node)
        elif op == "embed_lookup":
            self._emit_embedding_lookup(node, default_table="transformer.wte.weight")
        elif op == "pos_embed_lookup":
            self._emit_embedding_lookup(node, default_table="transformer.wpe.weight")
        elif op == "kv_store":
            self._emit_kv_store(node)
        elif op == "kv_load":
            self._emit_kv_load(node)
        elif op == "kv_quant":
            self._emit_kv_quant(node)
        elif op == "logits_store":
            self._emit_logits_store(node)
        elif op == "cls_prepend":
            self._emit_cls_prepend(node)
        elif op == "pos_embed_add":
            self._emit_pos_embed_add(node)
        elif op == "cls_extract":
            self._emit_cls_extract(node)
        elif op == "reshape_heads":
            pass  # No-op, handled by matmul_qkt
        elif op == "concat_heads":
            self._emit_concat_heads(node)
        elif op == "row_copy":
            self._emit_row_copy(node)
        elif op == "gather_rows":
            self._emit_gather_rows(node)

    def _emit_qkt(self, node: IRNode):
        from .emit.matmul import emit_qkt
        emit_qkt(self, node)

    def _emit_attn_v(self, node: IRNode):
        from .emit.matmul import emit_attn_v
        emit_attn_v(self, node)

    def _emit_concat_heads(self, node: IRNode):
        from .emit.matmul import emit_concat_heads
        emit_concat_heads(self, node)

    def _emit_row_copy(self, node: IRNode):
        from .emit.batch import emit_row_copy
        emit_row_copy(self, node)

    def _emit_gather_rows(self, node: IRNode):
        from .emit.batch import emit_gather_rows
        emit_gather_rows(self, node)

    def _emit_scale_mul(self, node: IRNode):
        from .emit.sfu import emit_scale_mul
        emit_scale_mul(self, node)

    def _emit_softmax(self, node: IRNode):
        from .emit.sfu import emit_softmax
        emit_softmax(self, node)

    def _emit_gelu(self, node: IRNode):
        from .emit.sfu import emit_gelu
        emit_gelu(self, node)

    def _emit_gelu_from_dram_temp(self, node: IRNode):
        from .emit.sfu import emit_gelu_from_dram_temp
        emit_gelu_from_dram_temp(self, node)

    def _emit_layernorm(self, node: IRNode):
        from .emit.sfu import emit_layernorm
        emit_layernorm(self, node)

    def _load_dram_to_abuf(self, input_name: str, M_pad: int, N_pad: int) -> Allocation:
        """Load a DRAM-temp-resident tensor into ABUF and return the allocation."""
        dram_off = self.dram_temp_outputs[input_name]
        # Free the small strip-mine placeholder before allocating the full tensor.
        # The placeholder (strip_rows * N_pad bytes) was created by _emit_matmul_strip_mined;
        # if we don't free it first, alloc() would overwrite it in the dict without
        # returning its bytes to the free list, causing a permanent memory leak.
        if self.mem.abuf.get(input_name) is not None:
            self.mem.abuf.free(input_name)
        alloc = self.mem.abuf.alloc(input_name, M_pad * N_pad)
        self._emit_dma_load(BUF_ABUF, alloc.offset_units, M_pad * N_pad, 3, dram_off)
        self._emit(SyncInsn(resource_mask=0b001))
        return alloc

    def _spill_fp32_tile_to_dram(self, name: str, alloc: Allocation,
                                  M_pad: int, N_pad: int) -> int:
        """M4-A: spill an FP32 ABUF tile to DRAM-temp, free its ABUF slot.

        Used by `emit_layernorm_fp32` to evict the LN input (residual)
        after LN has read it but before consumers (the next residual
        VADD) need it back. Without this, real GPT-2's 48 KB residual
        FP32 tile would pin ABUF across the entire per-head attention
        loop and force a peak occupancy > 128 KB.

        Returns the DRAM byte offset where the FP32 tile lives. The
        consumer (typically `emit_vadd_fp32`) reloads it via
        `_load_dram_to_abuf_fp` and the spilled `dram_temp_outputs`
        entry is cleaned up in the generate loop's post-emit free.
        """
        size_bytes = M_pad * N_pad * self.elem_bytes  # FP32=4, FP16=2
        dram_off = self.dram_temp_start + self.mem.alloc_dram_temp(
            f"fp32_spill_{name}", size_bytes
        )
        # DMA store ABUF → DRAM. addr_reg=2 matches the existing
        # strip-mined spill convention (avoids clashing with addr_reg=1
        # used by VADD producers and addr_reg=3 used by loads).
        self._emit_dma_store(BUF_ABUF, alloc.offset_units, size_bytes, 2, dram_off)
        self._emit(SyncInsn(resource_mask=0b010))
        self.dram_temp_outputs[name] = dram_off
        self.dram_temp_fp32_outputs[name] = size_bytes
        # Free the ABUF slot. The spilled name will re-allocate when
        # the consumer reloads.
        self.mem.abuf.free(name)
        return dram_off

    def _load_dram_to_abuf_fp(self, input_name: str,
                                 M_pad: int, N_pad: int) -> Allocation:
        """M4-A: FP32 counterpart to `_load_dram_to_abuf` (4 bytes/elem).

        Reads `M_pad * N_pad * 4` bytes from `dram_temp_outputs[input_name]`
        back into ABUF. Caller is responsible for marking the dram_temp
        slot consumed (the generate loop's post-emit free does this when
        `dram_temp_fp32_outputs[input_name]` is set).
        """
        dram_off = self.dram_temp_outputs[input_name]
        if self.mem.abuf.get(input_name) is not None:
            self.mem.abuf.free(input_name)
        size_bytes = M_pad * N_pad * self.elem_bytes
        alloc = self.mem.abuf.alloc(input_name, size_bytes)
        self._emit_dma_load(BUF_ABUF, alloc.offset_units, size_bytes, 3, dram_off)
        self._emit(SyncInsn(resource_mask=0b001))
        return alloc

    def _emit_vadd(self, node: IRNode):
        from .emit.sfu import emit_vadd
        emit_vadd(self, node)

    def _emit_embedding_lookup(self, node: IRNode, *, default_table: str):
        from .emit.embedding import emit_embedding_lookup
        emit_embedding_lookup(self, node, default_table=default_table)

    def _kv_entry_for_node(self, node: IRNode):
        from .emit.kv import kv_entry_for_node
        return kv_entry_for_node(self, node)

    def _kv_transfer_bytes(self, node: IRNode, *, decode_default: bool) -> int:
        from .emit.kv import kv_transfer_bytes
        return kv_transfer_bytes(self, node, decode_default=decode_default)

    def _kv_source_location(self, node: IRNode) -> Tuple[int, int]:
        from .emit.kv import kv_source_location
        return kv_source_location(self, node)

    def _emit_kv_store(self, node: IRNode):
        from .emit.kv import emit_kv_store
        emit_kv_store(self, node)

    def _emit_kv_load(self, node: IRNode):
        from .emit.kv import emit_kv_load
        emit_kv_load(self, node)

    def _emit_kv_quant(self, node: IRNode):
        from .emit.kv import emit_kv_quant
        emit_kv_quant(self, node)

    def _emit_logits_store(self, node: IRNode):
        from .emit.kv import emit_logits_store
        emit_logits_store(self, node)

    def _emit_cls_prepend(self, node: IRNode):
        from .emit.embedding import emit_cls_prepend
        emit_cls_prepend(self, node)

    def _emit_pos_embed_add(self, node: IRNode):
        from .emit.embedding import emit_pos_embed_add
        emit_pos_embed_add(self, node)

    def _emit_cls_extract(self, node: IRNode):
        from .emit.embedding import emit_cls_extract
        emit_cls_extract(self, node)

    def _compact_abuf(self):
        """Defragment ABUF by moving live allocations to lower addresses.

        Emits BUF_COPY instructions to slide live allocations leftward so all
        free space is consolidated into one contiguous block at the top.
        Updates allocation offsets so subsequent alloc() calls see the new layout.

        The simulator's execute_buf_copy reads source bytes before writing, so
        overlapping intra-ABUF copies (src > dst with partial overlap) are safe.
        """
        if not self.mem.abuf.allocations:
            return
        # Sort live allocations by current offset ascending (pack left to right)
        live = sorted(self.mem.abuf.allocations.values(), key=lambda a: a.offset_units)
        new_offset = 0
        any_moved = False
        for alloc in live:
            if alloc.offset_units != new_offset:
                self._emit(BufCopyInsn(
                    src_buf=BUF_ABUF, src_off=alloc.offset_units,
                    dst_buf=BUF_ABUF, dst_off=new_offset,
                    length=alloc.size_units,
                ))
                self._emit(SyncInsn(resource_mask=0b001))
                alloc.offset_units = new_offset
                any_moved = True
            new_offset += alloc.size_units
        if any_moved:
            # Rebuild free list as one contiguous block at the top
            self.mem.abuf._free = [(new_offset, self.mem.abuf.capacity_units - new_offset)]

    def _emit_dma_load(self, buf_id: int, sram_off_units: int, size_bytes: int,
                       addr_reg: int, dram_byte_offset: int, *,
                       dram_off_units: int = 0,
                       relocation_symbol: Optional[str] = None,
                       runtime_patch_kind: Optional[str] = None,
                       runtime_base_symbol: Optional[str] = None,
                       transpose: int = 0,
                       cols_log2: int = 0):
        """Emit SET_ADDR + LOAD sequence (transpose = lever-D K^T load)."""
        from .emit.dma import emit_dma_load
        emit_dma_load(self, buf_id, sram_off_units, size_bytes, addr_reg,
                      dram_byte_offset, dram_off_units=dram_off_units,
                      relocation_symbol=relocation_symbol,
                      runtime_patch_kind=runtime_patch_kind,
                      runtime_base_symbol=runtime_base_symbol,
                      transpose=transpose, cols_log2=cols_log2)

    def _emit_dma_store(self, buf_id: int, sram_off_units: int, size_bytes: int,
                        addr_reg: int, dram_byte_offset: int, *,
                        dram_off_units: int = 0,
                        relocation_symbol: Optional[str] = None,
                        runtime_patch_kind: Optional[str] = None,
                        runtime_base_symbol: Optional[str] = None):
        """Emit SET_ADDR + STORE sequence."""
        from .emit.dma import emit_dma_store
        emit_dma_store(self, buf_id, sram_off_units, size_bytes, addr_reg,
                       dram_byte_offset, dram_off_units=dram_off_units,
                       relocation_symbol=relocation_symbol,
                       runtime_patch_kind=runtime_patch_kind,
                       runtime_base_symbol=runtime_base_symbol)
