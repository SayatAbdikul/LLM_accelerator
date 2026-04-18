"""IR → ISA instruction sequence code generator."""
import re
import struct
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from ..isa.opcodes import BUF_ABUF, BUF_WBUF, BUF_ACCUM
from ..isa.instructions import (
    MatmulInsn, RequantInsn, RequantPcInsn, ScaleMulInsn, VaddInsn, SoftmaxInsn, LayernormInsn, GeluInsn,
    DequantAddInsn,
    SoftmaxAttnVInsn,
    LoadInsn, StoreInsn, BufCopyInsn, SetAddrLoInsn, SetAddrHiInsn,
    ConfigTileInsn, SetScaleInsn, SyncInsn, NopInsn, HaltInsn, Instruction,
)
from .ir import IRNode, IRGraph
from .tiler import tile_matmul, tile_qkt, tile_strip_mine, pad_dim, TILE
from .memory_alloc import MemoryAllocator, Allocation
from .model_config import ModelConfig, deit_tiny_config

UNIT = 16


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
    """Generate ISA instructions from IR graph."""

    def __init__(self, weight_data: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]],
                 calibration_scales: Dict[str, float],
                 prescaled_biases: Dict[str, np.ndarray],
                 gelu_from_accum: bool = False,
                 gelu_from_accum_blocks: Optional[set] = None,
                 dequant_add_residual1_blocks: Optional[set] = None,
                 fused_softmax_attnv_blocks: Optional[set] = None,
                 fused_softmax_attnv_accum_out_proj_blocks: Optional[set] = None,
                 requant_pc_weight_names: Optional[set] = None,
                 requant_pc_scale_tables: Optional[Dict[str, np.ndarray]] = None,
                 model_config: Optional[ModelConfig] = None):
        """
        Args:
            weight_data: name → (quantized_data, per_channel_scales)
            calibration_scales: tensor_name → per-tensor activation scale
            prescaled_biases: name → INT32 pre-scaled bias array
        """
        self.weight_data = weight_data
        self.config = model_config or deit_tiny_config()
        self.calibration_scales = calibration_scales
        self.prescaled_biases = prescaled_biases
        self.gelu_from_accum = gelu_from_accum
        self.gelu_from_accum_blocks = None if gelu_from_accum_blocks is None else set(gelu_from_accum_blocks)
        self.dequant_add_residual1_blocks = (
            None if dequant_add_residual1_blocks is None else set(dequant_add_residual1_blocks)
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
        # Optional trace metadata keyed by program counter. Each event tells the
        # simulator how to decode a node output back into FP32 for diagnostics.
        self.trace_manifest: Dict[int, List[Dict[str, Any]]] = {}
        self.pending_accum_outputs: Dict[str, Dict[str, Any]] = {}
        self.precomputed_nodes: set = set()

    def _dram_offset_required(self, name: str, context: str) -> int:
        """Return DRAM offset for a symbol or raise a clear error."""
        if name not in self.dram_layout:
            raise KeyError(f"Missing DRAM symbol '{name}' while {context}")
        return self.dram_layout[name]

    def generate(self, graph: IRGraph) -> Tuple[List[Instruction], bytes]:
        """Generate instructions for the entire IR graph.

        Returns (instructions, dram_data_blob).
        """
        # First pass: lay out weights in DRAM
        self._layout_weights(graph)

        # Compute last-use index for each node output so we can free ABUF
        last_uses = graph.compute_last_uses()

        # Second pass: emit instructions
        for idx, node in enumerate(graph.nodes):
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

        self.instructions.append(HaltInsn())
        return self.instructions, bytes(self.dram_blob)

    def _layout_weights(self, graph: IRGraph):
        """Pack all weights into DRAM data blob."""
        offset = 0
        for name, (data, scales) in self.weight_data.items():
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

        # Pre-scaled biases
        for name, bias_i32 in self.prescaled_biases.items():
            self.dram_layout[name] = offset
            blob = bias_i32.tobytes()
            self.dram_blob.extend(blob)
            offset += len(blob)

        # Zero-pad blob: used to mask attention padding rows (K and V) before QKT.
        # Padding rows 197-207 are zero in the input but LN(zero_row) = beta (non-zero),
        # which propagates through QKV projections. Zeroing K/V rows 197-207 eliminates
        # the beta-derived attention contribution from padding tokens.
        # Size: padding rows × row width. DeiT attention uses d_head; Stage 1
        # token embeddings use d_model for short fixed-sequence tests.
        if self.config.embedding_kind == "patch_cls":
            _zero_pad_size = (pad_dim(self.config.max_seq_len) - self.config.max_seq_len) * self.config.d_head
        else:
            _zero_pad_size = (TILE - 1) * self.config.d_model
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

    def _fused_softmax_attnv_accum_out_proj_enabled_for(self, node_name: str) -> bool:
        return self._block_selected(node_name, self.fused_softmax_attnv_accum_out_proj_blocks)

    def _should_trace_attention_projection_debug(self, node_name: str) -> bool:
        """Return True for per-head Q/K/V projections we may need to debug end-to-end."""
        return re.match(r"block\d+_head\d+_(query|key|value)$", node_name) is not None

    def _should_trace_ln1_padding_debug(self, node_name: str) -> bool:
        """Return True when a layernorm should emit padded input/output debug views."""
        return node_name == "block0_ln1"

    def _residual1_skip_name(self, out_proj_name: str) -> str:
        match = re.match(r"block(\d+)_out_proj$", out_proj_name)
        if match is None:
            raise ValueError(f"Cannot infer residual1 skip input from '{out_proj_name}'")
        block_idx = int(match.group(1))
        first_skip = "pos_embed_add" if self.config.embedding_kind == "patch_cls" else "tok_pos_add"
        return first_skip if block_idx == 0 else f"block{block_idx - 1}_residual2"

    def _emit_node(self, node: IRNode):
        """Emit instructions for a single IR node."""
        op = node.op
        if op == "matmul":
            self._emit_matmul(node)
        elif op == "matmul_qkt":
            self._emit_qkt(node)
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

    def _emit_matmul(self, node: IRNode):
        """Emit a standard linear matmul with optional bias."""
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
        # - INT32 intermediate exceeds ACCUM (64 KB)
        strip_mine = node.attrs.get("strip_mine", False)
        output_bytes = pad_dim(M) * pad_dim(N)
        accum_bytes = pad_dim(M) * pad_dim(N) * 4  # INT32 intermediate
        if output_bytes > 128 * 1024 or accum_bytes > 64 * 1024:
            strip_mine = True

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

        if self._dequant_add_residual1_enabled_for_output(node.name):
            if weight_name in self.requant_pc_weight_names:
                raise ValueError(
                    f"DEQUANT_ADD residual1 path currently requires scalar out_proj scale, got REQUANT_PC weight '{weight_name}'"
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
        out_alloc = self.mem.abuf.alloc(node.name, strip_rows * N_pad)
        out_alloc.size_bytes = M_pad * N_pad

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
        out_alloc = self.mem.abuf.alloc(node.name, strip_rows * N_pad)
        out_alloc.size_bytes = M_pad * N_pad  # real size is in DRAM

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

    def _emit_qkt(self, node: IRNode):
        """Emit Q@K^T attention matmul, strip-mined over Q's M dimension.

        Full [208,208] INT32 would need 173KB in ACCUM (only 64KB available).
        Instead process 16-row strips: each [16,208] INT32 = 13KB ≤ 64KB.
        SOFTMAX each strip from ACCUM directly to INT8 → WBUF immediately.
        After all strips, WBUF holds [208,208] INT8 = 43KB for downstream softmax.
        """
        head_idx = node.attrs["head_idx"]
        seq_len = node.output_shape[0]
        head_dim = self.config.d_head
        M_pad = pad_dim(seq_len)
        K_pad = pad_dim(head_dim)
        num_strips = M_pad // TILE  # 13 strips of 16 rows
        trace_qkt_debug = re.match(r"block\d+_head\d+_qkt$", node.name) is not None
        act_scale_q = self.calibration_scales.get(node.inputs[0], 6.0 / 127.0)
        act_scale_k = self.calibration_scales.get(node.inputs[1], 6.0 / 127.0)

        # BUF_COPY K_h → WBUF (transpose) to get K^T [64,208]
        k_alloc = self.mem.abuf.get(node.inputs[1])
        if k_alloc is None:
            k_alloc = self.mem.abuf.alloc(node.inputs[1], M_pad * head_dim)

        # Zero out K rows for padding positions (197-207).
        # LN(zero_row) = layernorm_beta (non-zero), so K[padding] = W_k @ beta + b_k.
        # Zeroing removes this contribution so padding columns don't steer attention.
        real_seq = node.output_shape[0]
        if M_pad > real_seq:
            pad_rows = M_pad - real_seq
            k_pad_units = k_alloc.offset_units + (real_seq * K_pad) // UNIT
            zero_pad_dram = self._dram_offset_required("__zero_pad__", "loading K padding mask")
            self._emit_dma_load(BUF_ABUF, k_pad_units, pad_rows * K_pad, 3,
                                zero_pad_dram)
            self._emit(SyncInsn(resource_mask=0b001))

        src_rows = M_pad // TILE
        length_units = (M_pad * K_pad) // UNIT
        kt_wbuf = self.mem.wbuf.alloc(f"kt_head{head_idx}", K_pad * M_pad)
        self._emit(BufCopyInsn(
            src_buf=BUF_ABUF, src_off=k_alloc.offset_units,
            dst_buf=BUF_WBUF, dst_off=kt_wbuf.offset_units,
            length=length_units,
            src_rows=src_rows,
            transpose=1,
        ))
        key_transpose_pc = len(self.instructions) - 1
        if trace_qkt_debug:
            # Snapshot the exact padded K tensor consumed by BUF_COPY and the
            # transposed WBUF tensor it produces. If the first divergence moves
            # to one of these traces we know whether the bug is in K
            # preparation or later in the Q x K^T path.
            self._record_trace_event(
                f"{node.name}__key_padded_input",
                BUF_ABUF,
                k_alloc.offset_units,
                M_pad,
                K_pad,
                M_pad,
                K_pad,
                "int8",
                act_scale_k,
                full_rows=M_pad,
                full_cols=K_pad,
                pc=key_transpose_pc,
            )
            self._record_trace_event(
                f"{node.name}__key_transposed",
                BUF_WBUF,
                kt_wbuf.offset_units,
                K_pad,
                M_pad,
                K_pad,
                M_pad,
                "int8",
                act_scale_k,
                full_rows=K_pad,
                full_cols=M_pad,
                pc=key_transpose_pc,
            )
        self._emit(SyncInsn(resource_mask=0b001))

        q_alloc = self.mem.abuf.get(node.inputs[0])
        if q_alloc is None:
            q_alloc = self.mem.abuf.alloc(node.inputs[0], M_pad * head_dim)

        fused_softmax_attnv = self._block_selected(node.name, self.fused_softmax_attnv_blocks)
        softmax_name = node.name.replace("_qkt", "_softmax")
        attn_v_name = node.name.replace("_qkt", "_attn_v")
        value_name = node.name.replace("_qkt", "_value")

        n_tiles = M_pad // TILE
        k_tiles = K_pad // TILE
        # C1: softmax consumes raw ACCUM values with this dequant scale.
        # qkt_in_scale = q_scale * k_scale * (1/sqrt(d_head)).
        qkt_in_scale = self.calibration_scales.get(
            node.name, act_scale_q * act_scale_k * node.attrs.get("scale", 0.125)
        )
        softmax_out_scale = self.calibration_scales.get(softmax_name, 1.0 / 127.0)
        if fused_softmax_attnv:
            v_alloc = self.mem.abuf.get(value_name)
            if v_alloc is None:
                v_alloc = self.mem.abuf.alloc(value_name, M_pad * K_pad)
            real_seq = node.output_shape[0]
            if M_pad > real_seq:
                pad_rows = M_pad - real_seq
                v_pad_units = v_alloc.offset_units + (real_seq * K_pad) // UNIT
                zero_pad_dram = self._dram_offset_required("__zero_pad__", "loading V padding mask")
                self._emit_dma_load(BUF_ABUF, v_pad_units, pad_rows * K_pad, 3, zero_pad_dram)
                self._emit(SyncInsn(resource_mask=0b001))
            target_act_scale = self.calibration_scales.get(attn_v_name, 6.0 / 127.0)
            attn_v_alloc = self.mem.wbuf.alloc(attn_v_name, M_pad * K_pad)
            v_scale = self.calibration_scales.get(value_name, 6.0 / 127.0)
        else:
            # Output: full [208,208] INT8 softmax probabilities in WBUF
            qkt_wbuf = self.mem.wbuf.alloc(node.name, M_pad * M_pad)

        for s in range(num_strips):
            row_start = s * TILE
            logical_rows = max(0, min(TILE, seq_len - row_start))
            # CONFIG_TILE: M=1 strip (16 rows), N=full, K=head_dim
            self._emit(ConfigTileInsn(M=0, N=n_tiles - 1, K=k_tiles - 1))
            qkt_config_pc = len(self.instructions) - 1
            if trace_qkt_debug:
                # Snapshot ACCUM immediately before the QK^T MATMUL. CONFIG_TILE
                # itself does not mutate SRAM, so tracing at this PC gives us the
                # architectural pre-state without adding a new "before" semantic
                # to the trace manifest.
                self._record_trace_event(
                    f"{node.name}__accum_pre_matmul",
                    BUF_ACCUM,
                    0,
                    TILE,
                    M_pad,
                    logical_rows,
                    seq_len,
                    "int32",
                    qkt_in_scale,
                    row_start=row_start,
                    full_rows=seq_len,
                    full_cols=seq_len,
                    pc=qkt_config_pc,
                )
                self._record_trace_event(
                    f"{node.name}__accum_pre_matmul_next",
                    BUF_ACCUM,
                    0,
                    TILE,
                    M_pad,
                    logical_rows,
                    seq_len,
                    "int32",
                    qkt_in_scale,
                    row_start=row_start,
                    full_rows=seq_len,
                    full_cols=seq_len,
                    pc=qkt_config_pc,
                    capture_phase="retire_plus_1",
                )

            # Q strip offset: s * 16 rows * K_pad cols
            q_strip_off = q_alloc.offset_units + (s * TILE * K_pad) // UNIT
            self._emit(MatmulInsn(
                src1_buf=BUF_ABUF, src1_off=q_strip_off,
                src2_buf=BUF_WBUF, src2_off=kt_wbuf.offset_units,
                dst_buf=BUF_ACCUM, dst_off=0,
                flags=0,
            ))
            qkt_matmul_pc = len(self.instructions) - 1
            if trace_qkt_debug:
                self._record_trace_event(
                    f"{node.name}__query_input",
                    BUF_ABUF,
                    q_strip_off,
                    TILE,
                    K_pad,
                    logical_rows,
                    head_dim,
                    "int8",
                    act_scale_q,
                    row_start=row_start,
                    full_rows=seq_len,
                    full_cols=head_dim,
                    pc=qkt_matmul_pc,
                )
            self._emit(SyncInsn(resource_mask=0b010))
            self._record_trace_event(
                node.name,
                BUF_ACCUM,
                0,
                TILE,
                M_pad,
                logical_rows,
                seq_len,
                "int32",
                qkt_in_scale,
                row_start=row_start,
                full_rows=seq_len,
                full_cols=seq_len,
            )

            if fused_softmax_attnv:
                self._emit(ConfigTileInsn(M=0, N=k_tiles - 1, K=n_tiles - 1))
                sreg = self._alloc_sreg_quad()
                self._emit(SetScaleInsn(sreg=sreg, src_mode=0, imm16=_fp16_to_uint16(qkt_in_scale)))
                self._emit(SetScaleInsn(sreg=sreg + 1, src_mode=0, imm16=_fp16_to_uint16(v_scale)))
                self._emit(SetScaleInsn(sreg=sreg + 2, src_mode=0, imm16=_fp16_to_uint16(target_act_scale)))
                self._emit(SetScaleInsn(sreg=sreg + 3, src_mode=0, imm16=_fp16_to_uint16(softmax_out_scale)))
                strip_out_off = attn_v_alloc.offset_units + (s * TILE * K_pad) // UNIT
                self._emit(SoftmaxAttnVInsn(
                    src1_buf=BUF_ACCUM, src1_off=0,
                    src2_buf=BUF_ABUF, src2_off=v_alloc.offset_units,
                    dst_buf=BUF_WBUF, dst_off=strip_out_off,
                    sreg=sreg,
                ))
                fused_pc = len(self.instructions) - 1
                self._emit(SyncInsn(resource_mask=0b100))
                self._record_trace_event(
                    softmax_name,
                    BUF_WBUF,
                    0,
                    TILE,
                    M_pad,
                    logical_rows,
                    seq_len,
                    "int8",
                    softmax_out_scale,
                    row_start=row_start,
                    full_rows=seq_len,
                    full_cols=seq_len,
                    pc=fused_pc,
                )
                self.trace_manifest.setdefault(fused_pc, [])[-1]["source"] = "virtual"
                self._record_trace_event(
                    attn_v_name,
                    BUF_WBUF,
                    strip_out_off,
                    TILE,
                    K_pad,
                    logical_rows,
                    head_dim,
                    "int8",
                    target_act_scale,
                    row_start=row_start,
                    full_rows=seq_len,
                    full_cols=head_dim,
                    pc=fused_pc,
                )
            else:
                # C1: SOFTMAX directly from ACCUM to avoid QKT INT8 bottleneck.
                # in_scale dequants INT32 accumulators; out_scale quantizes probabilities.
                sreg = self._alloc_sreg_pair()
                self._emit(SetScaleInsn(sreg=sreg, src_mode=0,
                                        imm16=_fp16_to_uint16(qkt_in_scale)))
                self._emit(SetScaleInsn(sreg=sreg + 1, src_mode=0,
                                        imm16=_fp16_to_uint16(softmax_out_scale)))
                strip_wbuf_off = qkt_wbuf.offset_units + (s * TILE * M_pad) // UNIT
                self._emit(SoftmaxInsn(
                    src1_buf=BUF_ACCUM, src1_off=0,
                    dst_buf=BUF_WBUF, dst_off=strip_wbuf_off,
                    sreg=sreg,
                ))
                softmax_pc = len(self.instructions) - 1
                if trace_qkt_debug:
                    self._record_trace_event(
                        f"{node.name}__accum_pre_softmax",
                        BUF_ACCUM,
                        0,
                        TILE,
                        M_pad,
                        logical_rows,
                        seq_len,
                        "int32",
                        qkt_in_scale,
                        row_start=row_start,
                        full_rows=seq_len,
                        full_cols=seq_len,
                        pc=softmax_pc,
                    )
                    self._record_trace_event(
                        f"{node.name}__accum_pre_softmax_next",
                        BUF_ACCUM,
                        0,
                        TILE,
                        M_pad,
                        logical_rows,
                        seq_len,
                        "int32",
                        qkt_in_scale,
                        row_start=row_start,
                        full_rows=seq_len,
                        full_cols=seq_len,
                        pc=softmax_pc,
                        capture_phase="retire_plus_1",
                    )
                self._emit(SyncInsn(resource_mask=0b100))
                self._record_trace_event(
                    softmax_name,
                    BUF_WBUF,
                    strip_wbuf_off,
                    TILE,
                    M_pad,
                    logical_rows,
                    seq_len,
                    "int8",
                    softmax_out_scale,
                    row_start=row_start,
                    full_rows=seq_len,
                    full_cols=seq_len,
                )

        self.mem.wbuf.free(f"kt_head{head_idx}")
        # Metadata now reflects softmax-quantized output in node.name allocation.
        self.calibration_scales[node.name] = softmax_out_scale

    def _emit_attn_v(self, node: IRNode):
        """Emit attention @ V matmul.

        attn scores are in WBUF (from softmax in-place).
        V_h is in ABUF. MATMUL src1=WBUF[attn], src2=ABUF[V].
        After matmul, free both attn (WBUF) and V (ABUF via last-use).
        """
        if self._block_selected(node.name, self.fused_softmax_attnv_blocks):
            return

        head_idx = node.attrs["head_idx"]
        seq_len = node.output_shape[0]
        head_dim = node.output_shape[1]
        M_pad = pad_dim(seq_len)
        N_pad = pad_dim(head_dim)

        # attn scores in WBUF under the softmax node name
        attn_alloc = self.mem.wbuf.get(node.inputs[0])
        if attn_alloc is None:
            attn_alloc = self.mem.wbuf.alloc(node.inputs[0], M_pad * M_pad)

        # V_h is the per-head ABUF allocation
        v_alloc = self.mem.abuf.get(node.inputs[1])
        if v_alloc is None:
            v_alloc = self.mem.abuf.alloc(node.inputs[1], M_pad * N_pad)

        # Zero out V rows for padding positions (197-207).
        # Same reason as K: LN(zero_row) = beta propagates non-zero values into V.
        # Zeroing V ensures padding positions contribute nothing to attn@V output.
        real_seq = node.output_shape[0]
        if M_pad > real_seq:
            pad_rows = M_pad - real_seq
            v_pad_units = v_alloc.offset_units + (real_seq * N_pad) // UNIT
            zero_pad_dram = self._dram_offset_required("__zero_pad__", "loading V padding mask")
            self._emit_dma_load(BUF_ABUF, v_pad_units, pad_rows * N_pad, 3,
                                zero_pad_dram)
            self._emit(SyncInsn(resource_mask=0b001))

        m_tiles = M_pad // TILE
        n_tiles = N_pad // TILE
        k_tiles = M_pad // TILE
        self._emit(ConfigTileInsn(M=m_tiles - 1, N=n_tiles - 1, K=k_tiles - 1))

        # MATMUL: attn(WBUF) @ V(ABUF) → ACCUM
        self._emit(MatmulInsn(
            src1_buf=BUF_WBUF, src1_off=attn_alloc.offset_units,
            src2_buf=BUF_ABUF, src2_off=v_alloc.offset_units,
            dst_buf=BUF_ACCUM, dst_off=0,
            flags=0,
        ))
        self._emit(SyncInsn(resource_mask=0b010))

        # Free attn scores from WBUF
        self.mem.wbuf.free(node.inputs[0])
        # Also free the scale_mul intermediate if still present
        for inp in node.inputs:
            self.mem.wbuf.free(inp)

        # Requantize: attn (INT8 softmax output) @ V (INT8 activation)  → INT32
        # requant_scale = attn_scale * v_scale / target_act_scale
        # attn_scale is the calibrated softmax output scale (max_prob/127 per head).
        # Using 1/127 would overestimate by 1/max_prob (up to 4×), causing heavy clipping.
        attn_scale = self.calibration_scales.get(node.inputs[0], 1.0 / 127.0)
        v_scale = self.calibration_scales.get(node.inputs[1], 6.0 / 127.0)
        target_act_scale = self.calibration_scales.get(node.name, 6.0 / 127.0)
        requant_scale_f = attn_scale * v_scale / max(target_act_scale, 1e-12)
        sreg = self._alloc_sreg()
        self._emit(SetScaleInsn(sreg=sreg, src_mode=0, imm16=_fp16_to_uint16(requant_scale_f)))
        out_alloc = self.mem.wbuf.alloc(node.name, M_pad * N_pad)
        self._emit(RequantInsn(
            src1_buf=BUF_ACCUM, src1_off=0,
            dst_buf=BUF_WBUF, dst_off=out_alloc.offset_units,
            sreg=sreg,
        ))
        self._record_trace_event(
            node.name,
            BUF_WBUF,
            out_alloc.offset_units,
            M_pad,
            N_pad,
            seq_len,
            head_dim,
            "int8",
            target_act_scale,
        )

    def _emit_concat_heads(self, node: IRNode):
        """BUF_COPY per-head outputs from WBUF into a contiguous ABUF region.

        Each head's attn_v output [M_pad, head_dim] is in WBUF. We interleave
        them into ABUF to form the correct [M_pad, num_heads*head_dim] layout
        required by the out_proj matmul.

        The matmul reads activations as row-major [M, K], so each output row t
        must contain all heads' data for that token:
            ABUF[out + t*K_total : out + t*K_total + K_total] = [h0[t], h1[t], ..., hH[t]]

        BufCopy only supports flat (non-strided) copies, so we emit one copy
        per token per head (row_units = head_dim / UNIT = 4 units per copy):
            src: WBUF[head_h + t * row_units]
            dst: ABUF[out  + t * out_row_units + h * row_units]
        """
        if self._fused_softmax_attnv_accum_out_proj_enabled_for(node.name):
            return

        head_dim = self.config.d_head
        seq_len = node.output_shape[0]
        M_pad = pad_dim(seq_len)
        N_pad = pad_dim(head_dim)
        num_heads = len(node.inputs)

        total_out_dim = num_heads * N_pad          # 192 bytes per token
        out_alloc = self.mem.abuf.alloc(node.name, M_pad * total_out_dim)

        row_units     = N_pad // UNIT              # units per token row per head (=4)
        out_row_units = total_out_dim // UNIT      # units per token row total   (=12)

        for h, inp_name in enumerate(node.inputs):
            src_alloc = self.mem.wbuf.get(inp_name)
            if src_alloc is None:
                continue
            for t in range(M_pad):                 # one copy per token
                src_off = src_alloc.offset_units + t * row_units
                dst_off = out_alloc.offset_units + t * out_row_units + h * row_units
                self._emit(BufCopyInsn(
                    src_buf=BUF_WBUF, src_off=src_off,
                    dst_buf=BUF_ABUF, dst_off=dst_off,
                    length=row_units,
                ))
                self._emit(SyncInsn(resource_mask=0b001))
            self.mem.wbuf.free(inp_name)
        self._record_trace_event(
            node.name,
            BUF_ABUF,
            out_alloc.offset_units,
            M_pad,
            total_out_dim,
            seq_len,
            node.output_shape[1],
            "int8",
            self.calibration_scales.get(node.name, 6.0 / 127.0),
        )

    def _emit_scale_mul(self, node: IRNode):
        """C1: scale_mul is metadata-only; scaling is folded into QKT softmax input scale."""
        in_alloc = self.mem.wbuf.get(node.inputs[0])
        if in_alloc is not None:
            # Rename in-place: pop the old key and re-insert under the new name
            # without touching the free list (free() would double-book the region).
            alloc = self.mem.wbuf.allocations.pop(node.inputs[0])
            self.mem.wbuf.allocations[node.name] = alloc

        # Propagate scale metadata for downstream rename nodes.
        self.calibration_scales[node.name] = self.calibration_scales.get(
            node.inputs[0], 6.0 / 127.0
        )

    def _emit_softmax(self, node: IRNode):
        """C1: softmax already emitted per-strip in _emit_qkt; this node is a rename."""
        in_alloc = self.mem.wbuf.get(node.inputs[0])
        if in_alloc is not None and node.inputs[0] in self.mem.wbuf.allocations:
            alloc = self.mem.wbuf.allocations.pop(node.inputs[0])
            self.mem.wbuf.allocations[node.name] = alloc
        self.calibration_scales[node.name] = self.calibration_scales.get(
            node.inputs[0], 1.0 / 127.0
        )

    def _emit_gelu(self, node: IRNode):
        """Emit GELU SFU instruction (no-op if inlined with a strip-mined matmul)."""
        if node.attrs.get("inline_with"):
            # GELU was applied inline in the strip-mined FC1 loop.
            # Propagate DRAM temp tracking and rename the ABUF placeholder.
            fc1_name = node.inputs[0]
            if fc1_name in self.dram_temp_outputs:
                self.dram_temp_outputs[node.name] = self.dram_temp_outputs[fc1_name]
            # Rename fc1's allocation to the gelu node name (transfer ownership).
            # Do NOT create a second Allocation pointing at the same bytes — that
            # would cause a double-free when the generate loop frees both fc1 and
            # gelu at their respective last-use indices.
            fc1_alloc = self.mem.abuf.allocations.pop(fc1_name, None)
            if fc1_alloc is not None:
                fc1_alloc.name = node.name
                self.mem.abuf.allocations[node.name] = fc1_alloc
            return
        M_pad = pad_dim(node.output_shape[0])
        N_pad = pad_dim(node.output_shape[1])
        m_tiles = M_pad // TILE
        n_tiles = N_pad // TILE
        self._emit(ConfigTileInsn(M=m_tiles - 1, N=n_tiles - 1, K=0))

        sreg = self._alloc_sreg_pair()
        in_scale = self.calibration_scales.get(node.inputs[0], 1.0 / 127.0)
        out_scale = self.calibration_scales.get(node.name, 1.0 / 127.0)
        self._emit(SetScaleInsn(sreg=sreg, src_mode=0, imm16=_fp16_to_uint16(in_scale)))
        self._emit(SetScaleInsn(sreg=sreg + 1, src_mode=0, imm16=_fp16_to_uint16(out_scale)))

        in_alloc = self.mem.abuf.get(node.inputs[0]) or \
                   self.mem.abuf.alloc(node.inputs[0], M_pad * N_pad)
        out_alloc = self.mem.abuf.alloc(node.name, M_pad * N_pad)
        self._emit(GeluInsn(
            src1_buf=BUF_ABUF, src1_off=in_alloc.offset_units,
            dst_buf=BUF_ABUF, dst_off=out_alloc.offset_units,
            sreg=sreg,
        ))
        self._emit(SyncInsn(resource_mask=0b100))
        self._record_trace_event(
            node.name,
            BUF_ABUF,
            out_alloc.offset_units,
            M_pad,
            N_pad,
            node.output_shape[0],
            node.output_shape[1],
            "int8",
            out_scale,
        )

    def _emit_layernorm(self, node: IRNode):
        """Emit LAYERNORM SFU instruction."""
        M_pad = pad_dim(node.output_shape[0])
        N_pad = pad_dim(node.output_shape[1])
        m_tiles = M_pad // TILE
        n_tiles = N_pad // TILE
        self._emit(ConfigTileInsn(M=m_tiles - 1, N=n_tiles - 1, K=0))

        sreg = self._alloc_sreg_pair()
        in_scale = self.calibration_scales.get(node.inputs[0], 1.0 / 127.0)
        out_scale = self.calibration_scales.get(node.name, 1.0 / 127.0)
        self._emit(SetScaleInsn(sreg=sreg, src_mode=0, imm16=_fp16_to_uint16(in_scale)))
        self._emit(SetScaleInsn(sreg=sreg + 1, src_mode=0, imm16=_fp16_to_uint16(out_scale)))

        # Load gamma/beta to WBUF
        gamma_name = node.inputs[1]
        beta_name = node.inputs[2]
        gamma_data = self.weight_data.get(gamma_name)
        beta_data = self.weight_data.get(beta_name)

        if gamma_data is not None and beta_data is not None:
            gamma_dram = self._dram_offset_required(gamma_name, f"loading layernorm gamma for '{node.name}'")
            beta_dram = self._dram_offset_required(beta_name, f"loading layernorm beta for '{node.name}'")
            # Pack gamma then beta in WBUF
            gb_bytes = N_pad * 4  # gamma[N] FP16 + beta[N] FP16 = N*4 bytes
            gb_alloc = self.mem.wbuf.alloc(f"gb_{node.name}", gb_bytes)
            self._emit_dma_load(BUF_WBUF, gb_alloc.offset_units, N_pad * 2, 1, gamma_dram)
            self._emit(SyncInsn(resource_mask=0b001))
            # Load beta right after gamma
            beta_off = gb_alloc.offset_units + (N_pad * 2) // UNIT
            self._emit_dma_load(BUF_WBUF, beta_off, N_pad * 2, 1, beta_dram)
            self._emit(SyncInsn(resource_mask=0b001))

        in_alloc = self.mem.abuf.get(node.inputs[0]) or \
                   self.mem.abuf.alloc(node.inputs[0], M_pad * N_pad)
        gb_alloc = self.mem.wbuf.get(f"gb_{node.name}")
        gb_off = gb_alloc.offset_units if gb_alloc else 0
        trace_ln1_padding = self._should_trace_ln1_padding_debug(node.name)

        if trace_ln1_padding:
            self._record_trace_event(
                f"{node.name}__input_padded",
                BUF_ABUF,
                in_alloc.offset_units,
                M_pad,
                N_pad,
                M_pad,
                N_pad,
                "int8",
                in_scale,
            )

        out_alloc = self.mem.abuf.alloc(node.name, M_pad * N_pad)
        self._emit(LayernormInsn(
            src1_buf=BUF_ABUF, src1_off=in_alloc.offset_units,
            src2_buf=BUF_WBUF, src2_off=gb_off,
            dst_buf=BUF_ABUF, dst_off=out_alloc.offset_units,
            sreg=sreg,
        ))
        self._emit(SyncInsn(resource_mask=0b100))
        self._record_trace_event(
            node.name,
            BUF_ABUF,
            out_alloc.offset_units,
            M_pad,
            N_pad,
            node.output_shape[0],
            node.output_shape[1],
            "int8",
            out_scale,
        )
        if trace_ln1_padding:
            self._record_trace_event(
                f"{node.name}__output_padded",
                BUF_ABUF,
                out_alloc.offset_units,
                M_pad,
                N_pad,
                M_pad,
                N_pad,
                "int8",
                out_scale,
            )

        if gb_alloc:
            self.mem.wbuf.free(f"gb_{node.name}")

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

    def _emit_vadd(self, node: IRNode):
        """Emit VADD for residual connection (INT8 saturating add).

        Handles the case where one input is DRAM-resident (strip-mined output)
        by loading it into ABUF first.
        """
        M_pad = pad_dim(node.output_shape[0])
        N_pad = pad_dim(node.output_shape[1])
        m_tiles = M_pad // TILE
        n_tiles = N_pad // TILE

        if node.name in self.precomputed_nodes:
            return

        if self._dequant_add_residual1_enabled_for_residual(node.name) and node.inputs[0] in self.pending_accum_outputs:
            skip_name = node.inputs[1]
            if skip_name in self.dram_temp_outputs:
                skip_alloc = self._load_dram_to_abuf(skip_name, M_pad, N_pad)
            else:
                skip_alloc = self.mem.abuf.get(skip_name) or \
                            self.mem.abuf.alloc(skip_name, M_pad * N_pad)

            pending = self.pending_accum_outputs.pop(node.inputs[0])
            output_scale = self.calibration_scales.get(node.name, 6.0 / 127.0)
            skip_scale = self.calibration_scales.get(skip_name, 6.0 / 127.0)
            accum_rescale = pending["accum_real_scale"] / max(output_scale, 1e-12)
            skip_rescale = skip_scale / max(output_scale, 1e-12)
            sreg = self._alloc_sreg_pair()
            self._emit(SetScaleInsn(sreg=sreg, src_mode=0, imm16=_fp16_to_uint16(accum_rescale)))
            self._emit(SetScaleInsn(sreg=sreg + 1, src_mode=0, imm16=_fp16_to_uint16(skip_rescale)))
            self._emit(ConfigTileInsn(M=m_tiles - 1, N=n_tiles - 1, K=0))
            self._emit(DequantAddInsn(
                src1_buf=BUF_ACCUM, src1_off=0,
                src2_buf=BUF_ABUF, src2_off=skip_alloc.offset_units,
                dst_buf=BUF_ABUF, dst_off=skip_alloc.offset_units,
                sreg=sreg,
            ))
            self._record_trace_event(
                node.name,
                BUF_ABUF,
                skip_alloc.offset_units,
                M_pad,
                N_pad,
                node.output_shape[0],
                node.output_shape[1],
                "int8",
                output_scale,
            )
            alloc = self.mem.abuf.allocations.pop(skip_name, None)
            if alloc is not None:
                alloc.name = node.name
                self.mem.abuf.allocations[node.name] = alloc
            return

        # Resolve src1 — load from DRAM if needed
        if node.inputs[0] in self.dram_temp_outputs:
            src1_alloc = self._load_dram_to_abuf(node.inputs[0], M_pad, N_pad)
            free_src1 = True
        else:
            src1_alloc = self.mem.abuf.get(node.inputs[0]) or \
                         self.mem.abuf.alloc(node.inputs[0], M_pad * N_pad)
            free_src1 = False

        # Resolve src2 — load from DRAM if needed
        if node.inputs[1] in self.dram_temp_outputs:
            src2_alloc = self._load_dram_to_abuf(node.inputs[1], M_pad, N_pad)
            free_src2 = True
        else:
            src2_alloc = self.mem.abuf.get(node.inputs[1]) or \
                         self.mem.abuf.alloc(node.inputs[1], M_pad * N_pad)
            free_src2 = False

        self._emit(ConfigTileInsn(M=m_tiles - 1, N=n_tiles - 1, K=0))

        # Write result in-place into src2's slot to avoid a third ABUF allocation.
        # (src2 is residual1 whose last use is this VADD, so overwriting is safe.)
        self._emit(VaddInsn(
            src1_buf=BUF_ABUF, src1_off=src1_alloc.offset_units,
            src2_buf=BUF_ABUF, src2_off=src2_alloc.offset_units,
            dst_buf=BUF_ABUF, dst_off=src2_alloc.offset_units,
        ))
        self._record_trace_event(
            node.name,
            BUF_ABUF,
            src2_alloc.offset_units,
            M_pad,
            N_pad,
            node.output_shape[0],
            node.output_shape[1],
            "int8",
            self.calibration_scales.get(node.name, 6.0 / 127.0),
        )

        # Free temporary ABUF slot used for the DRAM-loaded src1
        if free_src1:
            self.mem.abuf.free(node.inputs[0])

        # Rename src2's allocation to the output node name
        alloc = self.mem.abuf.allocations.pop(node.inputs[1], None)
        if alloc is not None:
            alloc.name = node.name
            self.mem.abuf.allocations[node.name] = alloc

    def _emit_embedding_lookup(self, node: IRNode, *, default_table: str):
        """Emit fixed-row token/position embedding loads for Stage 1 tests."""
        table_name = node.attrs.get("table", default_table)
        if table_name not in self.dram_layout:
            return
        seq_len, d_model = node.output_shape
        d_model_pad = pad_dim(d_model)
        if d_model_pad != d_model:
            raise ValueError("Stage 1 embedding lookup requires d_model to be 16-aligned")

        if "row_indices" in node.attrs:
            row_indices = list(node.attrs["row_indices"])
        elif "token_ids" in node.attrs:
            row_indices = list(node.attrs["token_ids"])
        elif "position_ids" in node.attrs:
            row_indices = list(node.attrs["position_ids"])
        else:
            row_indices = list(range(seq_len)) if node.op == "pos_embed_lookup" else [0] * seq_len
        if len(row_indices) != seq_len:
            raise ValueError(f"{node.name} expected {seq_len} row indices, got {len(row_indices)}")

        out_alloc = self.mem.abuf.alloc(node.name, pad_dim(seq_len) * d_model_pad)
        table_dram = self._dram_offset_required(table_name, f"loading embedding table '{table_name}'")
        row_units = d_model_pad // UNIT
        for row_idx, table_row in enumerate(row_indices):
            self._emit_dma_load(
                BUF_ABUF,
                out_alloc.offset_units + row_idx * row_units,
                d_model_pad,
                0,
                table_dram + int(table_row) * d_model_pad,
            )
            self._emit(SyncInsn(resource_mask=0b001))
        pad_rows = pad_dim(seq_len) - seq_len
        if pad_rows:
            zero_pad_dram = self._dram_offset_required("__zero_pad__", "zeroing embedding padding rows")
            self._emit_dma_load(
                BUF_ABUF,
                out_alloc.offset_units + seq_len * row_units,
                pad_rows * d_model_pad,
                3,
                zero_pad_dram,
            )
            self._emit(SyncInsn(resource_mask=0b001))
        self._record_trace_event(
            node.name,
            BUF_ABUF,
            out_alloc.offset_units,
            pad_dim(seq_len),
            d_model_pad,
            seq_len,
            d_model,
            "int8",
            self.calibration_scales.get(node.name, 6.0 / 127.0),
        )

    def _emit_cls_prepend(self, node: IRNode):
        """Emit CLS token prepend: load CLS to ABUF[0], then DMA patches to rows 1-196."""
        if self.config.embedding_kind != "patch_cls":
            raise ValueError("cls_prepend is only valid for patch_cls embeddings")
        cls_name = node.inputs[1]
        cls_dram = self._dram_offset_required(cls_name, "loading cls token")
        # Load CLS token [1, d_model] to ABUF row 0.
        self._emit_dma_load(BUF_ABUF, 0, self.config.d_model, 0, cls_dram)
        self._emit(SyncInsn(resource_mask=0b001))
        # DMA input patches from DRAM to ABUF rows 1-196.
        # Host writes INT8 patch embeddings [196, 192] to DRAM[input_offset] before run.
        # Row 1 starts at byte offset d_model in ABUF.
        patches_dram = self.dram_layout["__input_patches__"]
        patches_bytes = (self.config.max_seq_len - 1) * self.config.d_model
        self._emit_dma_load(BUF_ABUF, self.config.d_model // UNIT, patches_bytes, 1, patches_dram)
        self._emit(SyncInsn(resource_mask=0b001))
        # Mark allocation for the full [208, 192] padded sequence (rows 197-207 stay zero)
        self.mem.abuf.alloc(node.name, pad_dim(self.config.max_seq_len) * self.config.d_model, evictable=False)

    def _emit_pos_embed_add(self, node: IRNode):
        """Emit position embedding add."""
        if self.config.embedding_kind != "patch_cls":
            raise ValueError("pos_embed_add is only valid for patch_cls embeddings")
        pos_name = node.inputs[1]
        pos_dram = self._dram_offset_required(pos_name, "loading position embeddings")
        M_pad = pad_dim(self.config.max_seq_len)
        N = self.config.d_model
        N_pad = pad_dim(N)

        # Load pos_embed to WBUF [208, 192] (pre-padded at compile time)
        pos_bytes = M_pad * N_pad
        pos_alloc = self.mem.wbuf.alloc("pos_embed", pos_bytes)
        self._emit_dma_load(BUF_WBUF, pos_alloc.offset_units, pos_bytes, 0, pos_dram)
        self._emit(SyncInsn(resource_mask=0b001))

        # CONFIG_TILE for VADD
        m_tiles = M_pad // TILE
        n_tiles = N_pad // TILE
        self._emit(ConfigTileInsn(M=m_tiles - 1, N=n_tiles - 1, K=0))

        act_alloc = self.mem.abuf.get(node.inputs[0]) or \
                    self.mem.abuf.alloc(node.inputs[0], M_pad * N_pad)

        trace_scale = self.calibration_scales.get(node.name, 14.0 / 127.0)
        # Trace both inputs at the pre-VADD PC so the first-divergence harness
        # can tell whether the bug is in runtime placement or in the helper op.
        self._record_trace_event(
            f"{node.name}__act_input",
            BUF_ABUF,
            act_alloc.offset_units,
            M_pad,
            N_pad,
            node.output_shape[0],
            node.output_shape[1],
            "int8",
            trace_scale,
        )
        self._record_trace_event(
            f"{node.name}__pos_input",
            BUF_WBUF,
            pos_alloc.offset_units,
            M_pad,
            N_pad,
            node.output_shape[0],
            node.output_shape[1],
            "int8",
            trace_scale,
        )

        # VADD: activations + pos_embed (both INT8, same scale)
        out_alloc = self.mem.abuf.alloc(node.name, M_pad * N_pad)
        self._emit(VaddInsn(
            src1_buf=BUF_ABUF, src1_off=act_alloc.offset_units,
            src2_buf=BUF_WBUF, src2_off=pos_alloc.offset_units,
            dst_buf=BUF_ABUF, dst_off=out_alloc.offset_units,
        ))
        self._record_trace_event(
            node.name,
            BUF_ABUF,
            out_alloc.offset_units,
            M_pad,
            N_pad,
            node.output_shape[0],
            node.output_shape[1],
            "int8",
            trace_scale,
        )

        self.mem.wbuf.free("pos_embed")

    def _emit_cls_extract(self, node: IRNode):
        """Extract CLS token (row 0) via BUF_COPY."""
        if self.config.embedding_kind != "patch_cls":
            raise ValueError("cls_extract is only valid for patch_cls embeddings")
        N = self.config.d_model
        in_alloc = self.mem.abuf.get(node.inputs[0]) or \
                   self.mem.abuf.alloc(node.inputs[0], pad_dim(self.config.max_seq_len) * pad_dim(N))
        out_alloc = self.mem.abuf.alloc(node.name, pad_dim(N))
        # Copy 192 bytes = 12 × 16-byte units
        self._emit(BufCopyInsn(
            src_buf=BUF_ABUF, src_off=in_alloc.offset_units,
            dst_buf=BUF_ABUF, dst_off=out_alloc.offset_units,
            length=N // UNIT,
        ))
        self._record_trace_event(
            node.name,
            BUF_ABUF,
            out_alloc.offset_units,
            1,
            N,
            1,
            N,
            "int8",
            self.calibration_scales.get(node.name, 6.0 / 127.0),
        )

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
                       addr_reg: int, dram_byte_offset: int):
        """Emit SET_ADDR + LOAD sequence."""
        self.instructions.extend(_set_addr(addr_reg, dram_byte_offset))
        xfer_units = (size_bytes + UNIT - 1) // UNIT
        self._emit(LoadInsn(
            buf_id=buf_id,
            sram_off=sram_off_units,
            xfer_len=min(xfer_units, 0xFFFF),
            addr_reg=addr_reg,
            dram_off=0,
        ))

    def _emit_dma_store(self, buf_id: int, sram_off_units: int, size_bytes: int,
                        addr_reg: int, dram_byte_offset: int):
        """Emit SET_ADDR + STORE sequence."""
        self.instructions.extend(_set_addr(addr_reg, dram_byte_offset))
        xfer_units = (size_bytes + UNIT - 1) // UNIT
        self._emit(StoreInsn(
            buf_id=buf_id,
            sram_off=sram_off_units,
            xfer_len=min(xfer_units, 0xFFFF),
            addr_reg=addr_reg,
            dram_off=0,
        ))
