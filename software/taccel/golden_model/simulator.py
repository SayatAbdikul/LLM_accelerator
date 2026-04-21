"""Fetch-decode-execute loop for the golden model simulator.

Execution model
---------------
The golden model executes instructions **strictly sequentially** (no
pipelining, no out-of-order).  Real hardware has three independent execution
units (DMA, Systolic, SFU) that may operate in parallel.  The programmer
uses SYNC with a 3-bit resource mask to enforce ordering:

    SYNC 0b001   — wait for DMA   (LOAD / STORE)
    SYNC 0b010   — wait for Systolic (MATMUL)
    SYNC 0b100   — wait for SFU   (SOFTMAX / LAYERNORM / GELU)
    SYNC 0b111   — wait for all

In the golden model SYNC is a no-op since everything is already sequential.
In RTL SYNC is a pipeline barrier: the issue stage stalls until the selected
units drain.

Illegal opcodes
---------------
Decoding a reserved opcode (0x14–0x1F) or malformed instruction raises
IllegalOpcodeError.  RTL behaviour: the processor halts (equivalent to
executing HALT) and sets a fault status register.
"""
import numpy as np
from typing import Any, Dict, Optional, Set
from ..assembler.assembler import ProgramBundle
from ..isa.encoding import decode
from ..isa.opcodes import Opcode, BUF_ABUF, BUF_WBUF, BUF_ACCUM
from ..isa.instructions import (
    NopInsn, HaltInsn, SyncInsn, ConfigTileInsn, SetScaleInsn,
    SetAddrLoInsn, SetAddrHiInsn, LoadInsn, StoreInsn, BufCopyInsn,
    MatmulInsn, RequantInsn, RequantPcInsn, ScaleMulInsn, VaddInsn,
    SoftmaxInsn, LayernormInsn, GeluInsn, SoftmaxAttnVInsn,
    ConfigAttnInsn, MaskedSoftmaxInsn, MaskedSoftmaxAttnVInsn, DequantAddInsn,
)
from .state import MachineState
from . import memory as mem
from .systolic import execute_matmul
from .sfu import (
    execute_layernorm, execute_softmax, execute_gelu, execute_softmax_attnv,
    execute_masked_softmax, execute_masked_softmax_attnv,
)
from .dma import execute_load, execute_store, execute_buf_copy
from ..utils.int8_ops import clip_int8, clip_int32


class SimulatorError(Exception):
    pass

class IllegalOpcodeError(SimulatorError):
    def __init__(self, pc, raw_bytes):
        super().__init__(f"Illegal opcode at PC={pc}: {raw_bytes.hex()}")

class IllegalBufferError(SimulatorError):
    def __init__(self, buf_id):
        super().__init__(f"Illegal buffer ID: {buf_id}")

class ConfigError(SimulatorError):
    pass


class Simulator:
    """Golden model simulator with fetch-decode-execute loop."""

    def __init__(self, state: MachineState = None):
        self.state = state or MachineState()
        self.trace_manifest: Dict[int, list] = {}
        self.runtime_twin_specs: Dict[int, Dict[str, object]] = {}
        self.trace_enabled = False
        self.trace_node_names: Optional[Set[str]] = None
        self.trace_tensors: Dict[str, np.ndarray] = {}
        self.trace_raw_tensors: Dict[str, np.ndarray] = {}
        self.trace_saturation: Dict[str, Dict[str, int]] = {}
        self.trace_meta: Dict[str, Dict[str, object]] = {}
        self.trace_raw_events: list[Dict[str, Any]] = []
        self._virtual_trace_payloads: Dict[str, Dict[str, object]] = {}
        self.bundle: Optional[ProgramBundle] = None

    def load_program(self, program):
        """Load a ProgramBinary into the simulator."""
        self.program = program
        self.bundle = None
        self.state.pc = program.entry_point
        self.state.halted = False
        self.trace_manifest = getattr(program, "trace_manifest", {}) or {}
        self.runtime_twin_specs = {}
        compiler_manifest = getattr(program, "compiler_manifest", {}) or {}
        runtime_twin = compiler_manifest.get("runtime_twin_uniform", {}) or {}
        for kind in ("softmax", "gelu"):
            for pc_key, spec in (runtime_twin.get(kind, {}) or {}).items():
                spec_dict = dict(spec or {})
                spec_dict["kind"] = kind
                self.runtime_twin_specs[int(pc_key)] = spec_dict
        self.state.runtime_twin_specs = dict(self.runtime_twin_specs)
        self.trace_tensors = {}
        self.trace_raw_tensors = {}
        self.trace_saturation = {}
        self.trace_meta = {}
        self.trace_raw_events = []
        self._virtual_trace_payloads = {}
        if program.data_base > 0:
            # Unified DRAM layout: instructions at 0, data at data_base.
            # Build the full image so DRAM addresses are DRAM-absolute.
            insn_bytes = program.instructions
            padding = bytes(program.data_base - len(insn_bytes))
            image = insn_bytes + padding + program.data
            if len(image) > len(self.state.dram):
                self.state.dram = bytearray(len(image) + 1024 * 1024)
            self.state.dram[:len(image)] = image
        else:
            # Legacy: load data section at DRAM offset 0
            if program.data:
                if len(program.data) > len(self.state.dram):
                    self.state.dram = bytearray(len(program.data) + 1024 * 1024)
                self.state.dram[:len(program.data)] = program.data

    def _ensure_dram_capacity(self, required_bytes: int, *,
                              strict_dram_size: bool = False,
                              max_dram_bytes: int = 1 << 30) -> None:
        if required_bytes <= len(self.state.dram):
            return
        if strict_dram_size:
            raise SimulatorError(
                f"Program requires {required_bytes} DRAM bytes, "
                f"but strict DRAM size is {len(self.state.dram)} bytes"
            )
        if required_bytes > max_dram_bytes:
            raise SimulatorError(
                f"Program requires {required_bytes} DRAM bytes, exceeding cap {max_dram_bytes}"
            )
        self.state.dram = bytearray(required_bytes)

    def _reset_trace_state(self) -> None:
        self.trace_tensors = {}
        self.trace_raw_tensors = {}
        self.trace_saturation = {}
        self.trace_meta = {}
        self.trace_raw_events = []
        self._virtual_trace_payloads = {}

    def _reset_volatile_execution_state(self, entry_pc: int) -> None:
        self.state.pc = int(entry_pc)
        self.state.current_pc = int(entry_pc)
        self.state.halted = False
        self.state.tile_config = None
        self.state.attn_context = {
            "is_valid": False,
            "query_row_base": 0,
            "valid_kv_len": 0,
            "mode": 0,
        }
        self.state.scale_regs[:] = 0
        self.state.addr_regs[:] = 0
        self._virtual_trace_payloads = {}

    def load_bundle(self, bundle: ProgramBundle, *,
                    strict_dram_size: bool = False,
                    max_dram_bytes: int = 1 << 30) -> None:
        """Load a ProgramBundle into DRAM with idempotent fresh-image relocation."""
        image = bundle.materialize(reset_runtime=True)
        self._ensure_dram_capacity(
            bundle.required_dram_bytes,
            strict_dram_size=strict_dram_size,
            max_dram_bytes=max_dram_bytes,
        )
        self.state.dram[:bundle.required_dram_bytes] = image
        self.bundle = bundle
        self.program = bundle
        self.trace_manifest = {}
        self.runtime_twin_specs = {}
        self.state.runtime_twin_specs = {}
        self._reset_trace_state()
        self._reset_volatile_execution_state(bundle.prefill_pc)

    def run_program(self, bundle: ProgramBundle, name: str,
                    max_instructions: int = 10_000_000) -> int:
        """Run one ProgramBundle stream while preserving persistent memory."""
        if name not in ("prefill", "decode"):
            raise ValueError("ProgramBundle stream name must be 'prefill' or 'decode'")
        if getattr(self, "bundle", None) is not bundle:
            self.load_bundle(bundle)

        stream = bundle.stream_bytes(name)
        stream_offset = (
            bundle.prefill_instrs_offset if name == "prefill"
            else bundle.decode_instrs_offset
        )
        self.state.dram[stream_offset:stream_offset + len(stream)] = stream
        entry_pc = bundle.prefill_pc if name == "prefill" else bundle.decode_pc
        self.program = bundle
        self._reset_volatile_execution_state(entry_pc)
        return self.run(max_instructions=max_instructions)

    def run(self, max_instructions: int = 10_000_000):
        """Run until HALT or max instructions."""
        count = 0
        while not self.state.halted and count < max_instructions:
            self.step()
            count += 1
        return count

    def enable_trace(self, node_names=None):
        """Enable post-instruction tensor capture for selected node names."""
        self.trace_enabled = True
        self.trace_node_names = set(node_names) if node_names else None
        self.trace_tensors = {}
        self.trace_raw_tensors = {}
        self.trace_saturation = {}
        self.trace_meta = {}
        self.trace_raw_events = []

    def get_trace_payload(self):
        """Return traced tensors and saturation statistics."""
        trace_stats = {}
        for node_name, counts in self.trace_saturation.items():
            if "saturation_rate" in counts and "zero_fraction" in counts:
                trace_stats[node_name] = {
                    "saturation_rate": float(counts["saturation_rate"]),
                    "zero_fraction": float(counts["zero_fraction"]),
                }
                continue
            total = counts.get("total", 0)
            sat = counts.get("sat", 0)
            zero = counts.get("zero", 0)
            trace_stats[node_name] = {
                "saturation_rate": (sat / total) if total else 0.0,
                "zero_fraction": (zero / total) if total else 0.0,
            }
        return {
            "tensors": self.trace_tensors,
            "raw_tensors": self.trace_raw_tensors,
            "raw_events": self.trace_raw_events,
            "stats": trace_stats,
            "meta": self.trace_meta,
        }

    def _trace_node_logical_shape(self, node_name: str) -> tuple[int, int]:
        rows = 0
        cols = 0
        for events in self.trace_manifest.values():
            for event in events:
                if event["node_name"] != node_name:
                    continue
                row_start = int(event.get("row_start", 0))
                rows = max(rows, row_start + int(event["logical_rows"]))
                cols = max(cols, int(event["logical_cols"]))
        return rows, cols

    def _projection_padded_base_name(self, node_name: str) -> Optional[str]:
        if node_name.endswith("__act_input_padded"):
            base_name = node_name[:-7]
        elif node_name.endswith("__accum_pre_bias_padded"):
            base_name = node_name[:-7]
        elif node_name.endswith("__accum_padded"):
            base_name = node_name[:-7]
        elif node_name.endswith("__output_padded"):
            base_name = node_name[:-15]
        else:
            return None
        if base_name.split("__", 1)[0].endswith(("_query", "_key", "_value")):
            return base_name
        return None

    def _layernorm_padded_input_base_name(self, node_name: str) -> Optional[str]:
        if node_name.endswith("__input_padded") and node_name.startswith("block") and "_ln1__" in node_name:
            return node_name.removesuffix("__input_padded")
        return None

    def _zero_layernorm_input_padding(self, node_name: str, raw_view: np.ndarray) -> np.ndarray:
        base_name = self._layernorm_padded_input_base_name(node_name)
        if base_name is None:
            return raw_view
        logical_rows, logical_cols = self._trace_node_logical_shape(base_name)
        padded = raw_view.copy()
        if logical_rows < padded.shape[0]:
            padded[logical_rows:, :] = 0
        if logical_cols < padded.shape[1]:
            padded[:, logical_cols:] = 0
        return padded

    def _zero_projection_padding(self, node_name: str, raw_view: np.ndarray) -> np.ndarray:
        base_name = self._projection_padded_base_name(node_name)
        if base_name is None:
            return raw_view
        logical_rows, logical_cols = self._trace_node_logical_shape(base_name)
        padded = raw_view.copy()
        if logical_rows < padded.shape[0]:
            padded[logical_rows:, :] = 0
        if logical_cols < padded.shape[1]:
            padded[:, logical_cols:] = 0
        return padded

    def step(self):
        """Execute one instruction."""
        if self.state.halted:
            return

        pc = self.state.pc
        self.state.current_pc = pc
        self._virtual_trace_payloads = {}
        if pc >= self.program.insn_count:
            raise SimulatorError(f"PC={pc} past end of program ({self.program.insn_count} insns)")

        if self.program.data_base > 0:
            # Unified layout: fetch instruction bytes from DRAM
            raw = bytes(self.state.dram[pc * 8: pc * 8 + 8])
        else:
            raw = self.program.get_instruction_bytes(pc)
        try:
            insn = decode(raw)
        except ValueError:
            raise IllegalOpcodeError(pc, raw)

        self._execute(insn)
        self._capture_trace_events(pc)
        self.state.pc += 1

    def _capture_trace_events(self, pc: int):
        """Snapshot traced tensors after the instruction at pc has completed."""
        if not self.trace_enabled:
            return
        for event_index, event in enumerate(self.trace_manifest.get(pc, [])):
            node_name = event["node_name"]
            if self.trace_node_names is not None and node_name not in self.trace_node_names:
                continue

            buf_id = event["buf_id"]
            offset_units = event["offset_units"]
            mem_rows = event["mem_rows"]
            mem_cols = event["mem_cols"]
            logical_rows = event["logical_rows"]
            logical_cols = event["logical_cols"]
            full_rows = event["full_rows"]
            full_cols = event["full_cols"]
            row_start = event.get("row_start", 0)
            scale = np.float32(event["scale"])
            source = str(event.get("source", "architectural"))
            node_meta = self.trace_meta.setdefault(
                node_name,
                {
                    "scale": float(scale),
                    "dtype": event["dtype"],
                    "source": source,
                    "full_rows": int(full_rows),
                    "full_cols": int(full_cols),
                    "row_start": int(row_start),
                    "fragments": [],
                    "raw_available": source != "virtual",
                },
            )
            node_meta["scale"] = float(scale)
            node_meta["dtype"] = event["dtype"]
            node_meta["source"] = source
            node_meta["full_rows"] = int(full_rows)
            node_meta["full_cols"] = int(full_cols)
            node_meta["raw_available"] = bool(node_meta.get("raw_available", False) or source != "virtual")
            node_meta["fragments"].append(
                {
                    "pc": int(pc),
                    "event_index": int(event_index),
                    "row_start": int(row_start),
                    "logical_rows": int(logical_rows),
                    "logical_cols": int(logical_cols),
                }
            )

            raw_event: Dict[str, Any] = {
                "pc": int(pc),
                "event_index": int(event_index),
                "node_name": node_name,
                "dtype": event["dtype"],
                "scale": float(scale),
                "source": source,
                "capture_phase": str(event.get("capture_phase", "retire_cycle")),
                "row_start": int(row_start),
                "logical_rows": int(logical_rows),
                "logical_cols": int(logical_cols),
                "full_rows": int(full_rows),
                "full_cols": int(full_cols),
            }

            if source == "virtual":
                payload = self._virtual_trace_payloads.get(node_name)
                if payload is None:
                    raw_event["raw_available"] = False
                    self.trace_raw_events.append(raw_event)
                    continue
                raw_view = payload["raw"][:logical_rows, :logical_cols]
                if payload["dtype"] == "int8":
                    dequant = raw_view.astype(np.float32) * scale
                else:
                    dequant = raw_view.astype(np.float32)
                node_meta["dtype"] = payload["dtype"]
                node_meta["scale"] = float(payload.get("scale", scale))
                raw_event["dtype"] = payload["dtype"]
                raw_event["scale"] = float(payload.get("scale", scale))
                raw_event["raw_available"] = False
            elif event["dtype"] == "int32":
                if (node_name.endswith("__accum_pre_matmul") or
                        node_name.endswith("__accum_pre_matmul_next")):
                    # Debug traces use this checkpoint to compare the intended
                    # architectural pre-state for a fresh MATMUL strip. QK^T
                    # dispatches in this path are non-accumulating, so the
                    # golden reference should expose zeros here even if a prior
                    # op happened to leave bytes in ACCUM.
                    raw_view = np.zeros((logical_rows, logical_cols), dtype=np.int32)
                elif (node_name.endswith("__accum_pre_softmax") or
                      node_name.endswith("__accum_pre_softmax_next")):
                    # These checkpoints intentionally mirror the stable QK^T
                    # INT32 output at the softmax boundary.
                    raw_tile = mem.read_int32_tile(self.state, buf_id, offset_units, mem_rows, mem_cols)
                    raw_view = raw_tile[:logical_rows, :logical_cols]
                else:
                    raw_tile = mem.read_int32_tile(self.state, buf_id, offset_units, mem_rows, mem_cols)
                    raw_view = raw_tile[:logical_rows, :logical_cols]
                    raw_view = self._zero_layernorm_input_padding(node_name, raw_view)
                    raw_view = self._zero_projection_padding(node_name, raw_view)
                dequant = raw_view.astype(np.float32) * scale
                raw_event["raw_available"] = True
                raw_event["raw"] = raw_view.tolist()
            else:
                raw_tile = mem.read_int8_tile(self.state, buf_id, offset_units, mem_rows, mem_cols)
                raw_view = raw_tile[:logical_rows, :logical_cols]
                raw_view = self._zero_layernorm_input_padding(node_name, raw_view)
                raw_view = self._zero_projection_padding(node_name, raw_view)
                dequant = raw_view.astype(np.float32) * scale
                raw_event["raw_available"] = True
                raw_event["raw"] = raw_view.tolist()

            if node_name not in self.trace_tensors:
                self.trace_tensors[node_name] = np.zeros((full_rows, full_cols), dtype=np.float32)
            self.trace_tensors[node_name][row_start:row_start + logical_rows, :logical_cols] = dequant
            if source != "virtual":
                if node_name not in self.trace_raw_tensors:
                    raw_dtype = np.int32 if event["dtype"] == "int32" else np.int8
                    self.trace_raw_tensors[node_name] = np.zeros((full_rows, full_cols), dtype=raw_dtype)
                self.trace_raw_tensors[node_name][row_start:row_start + logical_rows, :logical_cols] = raw_view
            self.trace_raw_events.append(raw_event)

            if source == "virtual":
                if payload["dtype"] == "int8":
                    stats = self.trace_saturation.setdefault(node_name, {"sat": 0, "zero": 0, "total": 0})
                    stats["sat"] += int(payload.get("sat", 0))
                    stats["zero"] += int(payload.get("zero", 0))
                    stats["total"] += int(payload.get("total", raw_view.size))
            elif event["dtype"] == "int8":
                stats = self.trace_saturation.setdefault(node_name, {"sat": 0, "zero": 0, "total": 0})
                stats["sat"] += int(np.count_nonzero((raw_view == 127) | (raw_view == -128)))
                stats["zero"] += int(np.count_nonzero(raw_view == 0))
                stats["total"] += int(raw_view.size)

    def _execute(self, insn):
        """Dispatch instruction to handler."""
        op = insn.opcode

        if op == Opcode.NOP:
            self.state.cycle_count += 1
        elif op == Opcode.HALT:
            self.state.halted = True
        elif op == Opcode.SYNC:
            # In golden model, SYNC is a no-op (all ops are synchronous)
            self.state.cycle_count += 1
        elif op == Opcode.CONFIG_TILE:
            self.state.tile_config = (insn.M, insn.N, insn.K)
        elif op == Opcode.CONFIG_ATTN:
            self._exec_config_attn(insn)
        elif op == Opcode.SET_SCALE:
            self._exec_set_scale(insn)
        elif op == Opcode.SET_ADDR_LO:
            self._exec_set_addr_lo(insn)
        elif op == Opcode.SET_ADDR_HI:
            self._exec_set_addr_hi(insn)
        elif op == Opcode.LOAD:
            execute_load(self.state, insn)
        elif op == Opcode.STORE:
            execute_store(self.state, insn)
        elif op == Opcode.BUF_COPY:
            execute_buf_copy(self.state, insn)
        elif op == Opcode.MATMUL:
            execute_matmul(self.state, insn)
        elif op == Opcode.REQUANT:
            self._exec_requant(insn)
        elif op == Opcode.REQUANT_PC:
            self._exec_requant_pc(insn)
        elif op == Opcode.SCALE_MUL:
            self._exec_scale_mul(insn)
        elif op == Opcode.VADD:
            self._exec_vadd(insn)
        elif op == Opcode.DEQUANT_ADD:
            self._exec_dequant_add(insn)
        elif op == Opcode.SOFTMAX:
            execute_softmax(self.state, insn)
        elif op == Opcode.MASKED_SOFTMAX:
            if self.state.tile_config is None:
                raise ConfigError("CONFIG_TILE not set")
            self._validate_attn_context_for_key_cols((self.state.tile_config[1] + 1) * 16)
            execute_masked_softmax(self.state, insn)
        elif op == Opcode.SOFTMAX_ATTNV:
            virtual_payloads = execute_softmax_attnv(self.state, insn)
            self._virtual_trace_payloads = {
                node_name: dict(payload)
                for node_name, payload in (virtual_payloads or {}).items()
            }
        elif op == Opcode.MASKED_SOFTMAX_ATTNV:
            if self.state.tile_config is None:
                raise ConfigError("CONFIG_TILE not set")
            self._validate_attn_context_for_key_cols((self.state.tile_config[2] + 1) * 16)
            virtual_payloads = execute_masked_softmax_attnv(self.state, insn)
            self._virtual_trace_payloads = {
                node_name: dict(payload)
                for node_name, payload in (virtual_payloads or {}).items()
            }
        elif op == Opcode.LAYERNORM:
            execute_layernorm(self.state, insn)
        elif op == Opcode.GELU:
            execute_gelu(self.state, insn)
        else:
            raise IllegalOpcodeError(self.state.pc, b'\x00' * 8)

    def _exec_config_attn(self, insn: ConfigAttnInsn):
        """CONFIG_ATTN: set attention mask context for masked softmax ops."""
        if self.state.tile_config is None:
            raise ConfigError("CONFIG_TILE not set")
        if insn.mode == 0:
            raise ConfigError("CONFIG_ATTN mode 0b00 is reserved")
        if insn.valid_kv_len == 0:
            raise ConfigError("CONFIG_ATTN valid_kv_len must be > 0")
        self.state.attn_context = {
            "is_valid": True,
            "query_row_base": int(insn.query_row_base),
            "valid_kv_len": int(insn.valid_kv_len),
            "mode": int(insn.mode),
        }

    def _validate_attn_context_for_key_cols(self, softmax_key_cols: int):
        """Validate the active attention context against an opcode's key width."""
        if self.state.tile_config is None:
            raise ConfigError("CONFIG_TILE not set")
        ctx = getattr(self.state, "attn_context", None) or {}
        if not ctx.get("is_valid", False):
            raise ConfigError("CONFIG_ATTN not set")
        mode = int(ctx.get("mode", 0))
        valid_kv_len = int(ctx.get("valid_kv_len", 0))
        if mode == 0:
            raise ConfigError("CONFIG_ATTN mode 0b00 is reserved")
        if valid_kv_len == 0:
            raise ConfigError("CONFIG_ATTN valid_kv_len must be > 0")
        if mode in (0b01, 0b11) and softmax_key_cols < valid_kv_len:
            raise ConfigError(
                f"CONFIG_ATTN valid_kv_len={valid_kv_len} exceeds softmax key columns {softmax_key_cols}"
            )
        if mode == 0b10 and softmax_key_cols != valid_kv_len:
            raise ConfigError(
                f"CONFIG_ATTN pure-causal mode requires softmax key columns {softmax_key_cols} "
                f"to equal valid_kv_len={valid_kv_len}"
            )

    def _exec_set_scale(self, insn):
        """SET_SCALE: load FP16 into scale register."""
        if insn.src_mode == 0:
            # Immediate FP16 - store as little-endian for numpy
            fp16_bytes = insn.imm16.to_bytes(2, 'little')
            val = np.frombuffer(fp16_bytes, dtype=np.float16)[0]
            self.state.scale_regs[insn.sreg] = val
        else:
            # From buffer
            buf_id = {1: BUF_ABUF, 2: BUF_WBUF, 3: BUF_ACCUM}[insn.src_mode]
            data = mem.read_bytes(self.state, buf_id, insn.imm16, 2)
            val = np.frombuffer(data, dtype=np.float16)[0]
            self.state.scale_regs[insn.sreg] = val

    def _exec_set_addr_lo(self, insn):
        """SET_ADDR_LO: set bits [27:0] of DRAM address register."""
        reg = insn.addr_reg
        self.state.addr_regs[reg] = (int(self.state.addr_regs[reg]) & ~0xFFFFFFF) | insn.imm28

    def _exec_set_addr_hi(self, insn):
        """SET_ADDR_HI: set bits [55:28] of DRAM address register."""
        reg = insn.addr_reg
        self.state.addr_regs[reg] = (int(self.state.addr_regs[reg]) & 0xFFFFFFF) | (insn.imm28 << 28)

    def _exec_requant(self, insn):
        """REQUANT: INT32 → INT8 via scale register.

        dst = clip(round(src1 × S[sreg]), -128, 127)

        Scale register holds FP16; widened to FP32 for the multiply.
        Rounding: **round-half-to-even** (IEEE 754 default / banker's rounding).
        RTL must implement the same mode; round-half-away-from-zero would
        produce ±1 LSB differences on exact 0.5 boundaries.

        INT32 inputs are exact in FP32 for the accumulator range used here
        (max |acc| = 197 tokens × 127² ≈ 3.2M << 2^24 FP32 exact limit).
        """
        if self.state.tile_config is None:
            raise ConfigError("CONFIG_TILE not set")

        m_tiles = self.state.tile_config[0] + 1
        n_tiles = self.state.tile_config[1] + 1
        M = m_tiles * 16
        N = n_tiles * 16

        # FP16 scale register widened to FP32 — no extra precision introduced
        scale = np.float32(self.state.scale_regs[insn.sreg])

        src = mem.read_int32_tile(self.state, insn.src1_buf, insn.src1_off, M, N)

        # Multiply in FP32, round-half-to-even, clip to INT8
        result = np.clip(np.round(src.astype(np.float32) * scale), -128, 127).astype(np.int8)

        mem.write_int8_tile(self.state, insn.dst_buf, insn.dst_off, result)
        self.state.cycle_count += M * N

    def _exec_requant_pc(self, insn):
        """REQUANT_PC: INT32 → INT8 using one FP16 scale per output column.

        src1 must be ACCUM. src2 points at a packed FP16 scale table in ABUF/WBUF
        with N entries, one per output column. The same scale row is reused for all
        M rows in the configured tile.
        """
        if self.state.tile_config is None:
            raise ConfigError("CONFIG_TILE not set")
        if insn.src1_buf != BUF_ACCUM:
            raise IllegalBufferError(insn.src1_buf)
        if insn.src2_buf == BUF_ACCUM:
            raise IllegalBufferError(insn.src2_buf)
        if insn.dst_buf == BUF_ACCUM:
            raise IllegalBufferError(insn.dst_buf)

        m_tiles = self.state.tile_config[0] + 1
        n_tiles = self.state.tile_config[1] + 1
        M = m_tiles * 16
        N = n_tiles * 16

        src = mem.read_int32_tile(self.state, insn.src1_buf, insn.src1_off, M, N)
        scale_bytes = mem.read_bytes(self.state, insn.src2_buf, insn.src2_off, N * 2)
        scales = np.frombuffer(scale_bytes, dtype=np.float16).astype(np.float32).reshape(1, N)
        result = np.clip(np.round(src.astype(np.float32) * scales), -128, 127).astype(np.int8)

        mem.write_int8_tile(self.state, insn.dst_buf, insn.dst_off, result)
        self.state.cycle_count += M * N

    def _exec_scale_mul(self, insn):
        """SCALE_MUL: multiply tile by scale.

        When src is ACCUM (INT32): dst = clip(round(src × scale), INT32_MIN, INT32_MAX)
        When src is ABUF (INT8):   dst = clip(round(src × scale), -128, 127)

        Scale register holds FP16; widened to FP32 for the multiply.
        Rounding: round-half-to-even (numpy default).
        """
        if self.state.tile_config is None:
            raise ConfigError("CONFIG_TILE not set")

        m_tiles = self.state.tile_config[0] + 1
        n_tiles = self.state.tile_config[1] + 1
        M = m_tiles * 16
        N = n_tiles * 16

        # FP16 scale register widened to FP32 — no extra precision introduced
        scale = np.float32(self.state.scale_regs[insn.sreg])

        if insn.src1_buf == BUF_ACCUM:
            # INT32 path: multiply in FP32, clip to INT32
            src = mem.read_int32_tile(self.state, insn.src1_buf, insn.src1_off, M, N)
            scaled = np.round(src.astype(np.float32) * scale)
            result = np.clip(scaled, np.iinfo(np.int32).min, np.iinfo(np.int32).max).astype(np.int32)
            mem.write_int32_tile(self.state, insn.dst_buf, insn.dst_off, result)
        else:
            # INT8 path: multiply in FP32, clip to INT8
            src = mem.read_int8_tile(self.state, insn.src1_buf, insn.src1_off, M, N)
            result = np.clip(np.round(src.astype(np.float32) * scale), -128, 127).astype(np.int8)
            mem.write_int8_tile(self.state, insn.dst_buf, insn.dst_off, result)

        self.state.cycle_count += M * N

    def _exec_vadd(self, insn):
        """VADD: elementwise add.

        Dispatch based on src1_buf:
        - ABUF (00): saturating INT8 add
        - ACCUM (10): INT32 add with broadcast (src2 row broadcast when M > 1 row in src2)
        """
        if self.state.tile_config is None:
            raise ConfigError("CONFIG_TILE not set")

        m_tiles = self.state.tile_config[0] + 1
        n_tiles = self.state.tile_config[1] + 1
        M = m_tiles * 16
        N = n_tiles * 16

        if insn.src1_buf == BUF_ABUF:
            # Saturating INT8 add
            src1 = mem.read_int8_tile(self.state, insn.src1_buf, insn.src1_off, M, N)
            src2 = mem.read_int8_tile(self.state, insn.src2_buf, insn.src2_off, M, N)
            result = np.zeros((M, N), dtype=np.int8)
            for i in range(M):
                for j in range(N):
                    result[i, j] = clip_int8(int(src1[i, j]) + int(src2[i, j]))
            mem.write_int8_tile(self.state, insn.dst_buf, insn.dst_off, result)

        elif insn.src1_buf == BUF_ACCUM:
            # INT32 add with broadcast: src2 is a bias vector (1 row),
            # broadcast across all M rows of the accumulator.
            src1 = mem.read_int32_tile(self.state, insn.src1_buf, insn.src1_off, M, N)
            src2_row = mem.read_int32_tile(self.state, insn.src2_buf, insn.src2_off, 1, N)
            src2 = np.tile(src2_row, (M, 1))
            result = src1 + src2
            mem.write_int32_tile(self.state, insn.dst_buf, insn.dst_off, result)

        else:
            raise IllegalBufferError(insn.src1_buf)

        self.state.cycle_count += M * N

    def _exec_dequant_add(self, insn):
        """DEQUANT_ADD: FP32 add of ACCUM and INT8 skip path, requanted to INT8."""
        if self.state.tile_config is None:
            raise ConfigError("CONFIG_TILE not set")
        if insn.src1_buf != BUF_ACCUM:
            raise IllegalBufferError(insn.src1_buf)
        if insn.src2_buf == BUF_ACCUM:
            raise IllegalBufferError(insn.src2_buf)
        if insn.dst_buf == BUF_ACCUM:
            raise IllegalBufferError(insn.dst_buf)
        if insn.sreg >= 15:
            raise ConfigError("DEQUANT_ADD sreg+1 out of range")

        m_tiles = self.state.tile_config[0] + 1
        n_tiles = self.state.tile_config[1] + 1
        M = m_tiles * 16
        N = n_tiles * 16

        accum_rescale = np.float32(self.state.scale_regs[insn.sreg])
        skip_rescale = np.float32(self.state.scale_regs[insn.sreg + 1])
        accum = mem.read_int32_tile(self.state, insn.src1_buf, insn.src1_off, M, N).astype(np.float32)
        skip = mem.read_int8_tile(self.state, insn.src2_buf, insn.src2_off, M, N).astype(np.float32)
        result = np.clip(np.round(accum * accum_rescale + skip * skip_rescale), -128, 127).astype(np.int8)

        mem.write_int8_tile(self.state, insn.dst_buf, insn.dst_off, result)
        self.state.cycle_count += M * N
