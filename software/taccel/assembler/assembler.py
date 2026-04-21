"""Two-pass assembler: text assembly → ProgramBinary."""
import json
import struct
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Union
from .syntax import parse_line
from ..isa.encoding import decode, encode
from ..isa.instructions import ConfigAttnInsn, Instruction, SetAddrHiInsn, SetAddrLoInsn


MAGIC = 0x54414343  # "TACC", needed fro checking the binary file format and version compatibility.
LEGACY_VERSION = 0x0001
RUNTIME_METADATA_VERSION = 0x0002
VERSION = 0x0003
LEGACY_HEADER_FMT = ">IHHIIIIQ"
HEADER_FMT = ">IHH" + "I" * 14
LEGACY_HEADER_SIZE = struct.calcsize(LEGACY_HEADER_FMT)
HEADER_SIZE = struct.calcsize(HEADER_FMT)
MASK_28BIT = 0x0FFFFFFF
MASK_56BIT = (1 << 56) - 1
VALID_RUNTIME_PATCH_KINDS = {"token_embed", "pos_embed", "kv_base"}


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def _instruction_at(instructions: Union[bytes, bytearray], pc: int) -> Instruction:
    offset = pc * 8
    if pc < 0 or offset + 8 > len(instructions):
        raise ValueError(f"Instruction PC {pc} is out of range for {len(instructions) // 8} instructions")
    return decode(bytes(instructions[offset:offset + 8]))


def patch_set_addr_pair(instructions: bytearray, local_lo_pc: int, local_hi_pc: int,
                        addr_reg: int, byte_addr: int) -> None:
    """Patch a SET_ADDR_LO/SET_ADDR_HI pair to a 56-bit absolute byte address."""
    if not (0 <= byte_addr <= MASK_56BIT):
        raise ValueError(f"byte_addr must fit in 56 bits, got {byte_addr:#x}")
    lo_insn = _instruction_at(instructions, local_lo_pc)
    hi_insn = _instruction_at(instructions, local_hi_pc)
    if not isinstance(lo_insn, SetAddrLoInsn):
        raise ValueError(f"PC {local_lo_pc} is not SET_ADDR_LO")
    if not isinstance(hi_insn, SetAddrHiInsn):
        raise ValueError(f"PC {local_hi_pc} is not SET_ADDR_HI")
    if lo_insn.addr_reg != addr_reg:
        raise ValueError(f"SET_ADDR_LO at PC {local_lo_pc} uses R{lo_insn.addr_reg}, expected R{addr_reg}")
    if hi_insn.addr_reg != addr_reg:
        raise ValueError(f"SET_ADDR_HI at PC {local_hi_pc} uses R{hi_insn.addr_reg}, expected R{addr_reg}")

    lo = byte_addr & MASK_28BIT
    hi = (byte_addr >> 28) & MASK_28BIT
    instructions[local_lo_pc * 8:local_lo_pc * 8 + 8] = encode(SetAddrLoInsn(addr_reg=addr_reg, imm28=lo))
    instructions[local_hi_pc * 8:local_hi_pc * 8 + 8] = encode(SetAddrHiInsn(addr_reg=addr_reg, imm28=hi))


def read_set_addr_pair(instructions: Union[bytes, bytearray], local_lo_pc: int, local_hi_pc: int,
                       addr_reg: int) -> int:
    """Read and validate a SET_ADDR_LO/HI pair as a 56-bit byte address."""
    lo_insn = _instruction_at(instructions, local_lo_pc)
    hi_insn = _instruction_at(instructions, local_hi_pc)
    if not isinstance(lo_insn, SetAddrLoInsn):
        raise ValueError(f"PC {local_lo_pc} is not SET_ADDR_LO")
    if not isinstance(hi_insn, SetAddrHiInsn):
        raise ValueError(f"PC {local_hi_pc} is not SET_ADDR_HI")
    if lo_insn.addr_reg != addr_reg:
        raise ValueError(f"SET_ADDR_LO at PC {local_lo_pc} uses R{lo_insn.addr_reg}, expected R{addr_reg}")
    if hi_insn.addr_reg != addr_reg:
        raise ValueError(f"SET_ADDR_HI at PC {local_hi_pc} uses R{hi_insn.addr_reg}, expected R{addr_reg}")
    return (int(hi_insn.imm28) << 28) | int(lo_insn.imm28)


def patch_config_attn(instructions: bytearray, local_pc: int, *,
                      query_row_base: int, valid_kv_len: int, mode: int) -> None:
    """Patch a CONFIG_ATTN instruction payload while preserving its opcode."""
    insn = _instruction_at(instructions, local_pc)
    if not isinstance(insn, ConfigAttnInsn):
        raise ValueError(f"PC {local_pc} is not CONFIG_ATTN")
    patched = ConfigAttnInsn(
        query_row_base=int(query_row_base),
        valid_kv_len=int(valid_kv_len),
        mode=int(mode),
    )
    instructions[local_pc * 8:local_pc * 8 + 8] = encode(patched)


def relocate_set_addr_pairs(instructions: Union[bytes, bytearray], delta: int) -> bytearray:
    """Return a copy with every adjacent SET_ADDR_LO/HI pair increased by delta."""
    patched = bytearray(instructions)
    pc = 0
    insn_count = len(patched) // 8
    while pc < insn_count:
        insn = _instruction_at(patched, pc)
        if not isinstance(insn, SetAddrLoInsn):
            pc += 1
            continue
        if pc + 1 >= insn_count:
            raise ValueError(f"SET_ADDR_LO at PC {pc} is missing paired SET_ADDR_HI")
        hi_insn = _instruction_at(patched, pc + 1)
        if not isinstance(hi_insn, SetAddrHiInsn):
            raise ValueError(f"SET_ADDR_LO at PC {pc} is not followed by SET_ADDR_HI")
        if hi_insn.addr_reg != insn.addr_reg:
            raise ValueError(
                f"SET_ADDR_LO/HI register mismatch at PCs {pc}/{pc + 1}: "
                f"R{insn.addr_reg} vs R{hi_insn.addr_reg}"
            )
        old_addr = (int(hi_insn.imm28) << 28) | int(insn.imm28)
        patch_set_addr_pair(patched, pc, pc + 1, insn.addr_reg, old_addr + delta)
        pc += 2
    return patched


@dataclass(frozen=True)
class RelocationSite:
    """Static relocation against a symbol in a ProgramBundle layout."""
    stream: str
    local_lo_pc: int
    local_hi_pc: int
    addr_reg: int
    symbol: str

    def __post_init__(self):
        if self.stream not in ("prefill", "decode"):
            raise ValueError("RelocationSite.stream must be 'prefill' or 'decode'")


@dataclass(frozen=True)
class RuntimePatchSite:
    """Runtime-patched SET_ADDR pair in a ProgramBundle stream."""
    stream: str
    kind: str
    local_lo_pc: int
    local_hi_pc: int
    absolute_lo_pc: int
    absolute_hi_pc: int
    addr_reg: int
    base_symbol: str

    def __post_init__(self):
        if self.stream not in ("prefill", "decode"):
            raise ValueError("RuntimePatchSite.stream must be 'prefill' or 'decode'")
        if self.kind not in VALID_RUNTIME_PATCH_KINDS:
            raise ValueError(
                f"RuntimePatchSite.kind must be one of {sorted(VALID_RUNTIME_PATCH_KINDS)}, got {self.kind!r}"
            )


@dataclass(frozen=True)
class RuntimeConfigAttnSite:
    """Runtime-patched CONFIG_ATTN in a ProgramBundle stream."""
    stream: str
    local_pc: int
    absolute_pc: int
    mode: int

    def __post_init__(self):
        if self.stream not in ("prefill", "decode"):
            raise ValueError("RuntimeConfigAttnSite.stream must be 'prefill' or 'decode'")
        if not (0 <= self.mode <= 0x3):
            raise ValueError(f"RuntimeConfigAttnSite.mode must be 0-3, got {self.mode}")


@dataclass
class ProgramBundle:
    """Two instruction streams sharing one decoder DRAM image."""
    prefill_instrs: bytes = b""
    decode_instrs: bytes = b""
    shared_data: bytes = b""
    temp_size: int = 0
    logits_size: int = 0
    kv_cache_size: int = 0
    input_offset: int = 0
    prefill_logits_offset: int = 0
    decode_logits_offset: int = 0
    symbol_offsets: Dict[str, int] = field(default_factory=dict)
    symbol_regions: Dict[str, str] = field(default_factory=dict)
    relocation_sites: List[RelocationSite] = field(default_factory=list)
    runtime_patch_sites: List[RuntimePatchSite] = field(default_factory=list)
    runtime_config_attn_sites: List[RuntimeConfigAttnSite] = field(default_factory=list)
    embedding_row_bytes: int = 16
    kv_step_bytes: int = 16

    prefill_instrs_offset: int = field(init=False)
    decode_instrs_offset: int = field(init=False)
    data_base: int = field(init=False)
    temp_base: int = field(init=False)
    logits_base: int = field(init=False)
    kv_cache_base: int = field(init=False)
    kv_cache_size_bytes: int = field(init=False)
    required_dram_bytes: int = field(init=False)
    prefill_pc: int = field(init=False)
    decode_pc: int = field(init=False)
    insn_count: int = field(init=False)

    def __post_init__(self):
        self.prefill_instrs = bytes(self.prefill_instrs)
        self.decode_instrs = bytes(self.decode_instrs)
        self.shared_data = bytes(self.shared_data)
        if len(self.prefill_instrs) % 8 or len(self.decode_instrs) % 8:
            raise ValueError("ProgramBundle instruction streams must be 8-byte aligned")
        if min(self.temp_size, self.logits_size, self.kv_cache_size) < 0:
            raise ValueError("ProgramBundle region sizes must be non-negative")
        if self.embedding_row_bytes <= 0:
            raise ValueError("ProgramBundle.embedding_row_bytes must be positive")
        if self.kv_step_bytes <= 0:
            raise ValueError("ProgramBundle.kv_step_bytes must be positive")

        self.prefill_instrs_offset = 0
        self.decode_instrs_offset = _align(len(self.prefill_instrs), 8)
        decode_end = self.decode_instrs_offset + len(self.decode_instrs)
        self.data_base = _align(decode_end, 16)
        self.temp_base = _align(self.data_base + len(self.shared_data), 16)
        self.logits_base = _align(self.temp_base + self.temp_size, 16)
        self.kv_cache_base = _align(self.logits_base + self.logits_size, 16)
        self.kv_cache_size_bytes = self.kv_cache_size
        self.required_dram_bytes = self.kv_cache_base + self.kv_cache_size
        self.prefill_pc = self.prefill_instrs_offset // 8
        self.decode_pc = self.decode_instrs_offset // 8
        self.insn_count = self.data_base // 8
        if self.input_offset == 0:
            self.input_offset = self.data_base
        if self.prefill_logits_offset == 0:
            self.prefill_logits_offset = self.logits_base
        if self.decode_logits_offset == 0:
            self.decode_logits_offset = self.logits_base

        self.symbol_offsets = dict(self.symbol_offsets)
        self.symbol_regions = dict(self.symbol_regions)
        self.relocation_sites = list(self.relocation_sites)
        self.runtime_patch_sites = [
            replace(
                site,
                absolute_lo_pc=self._stream_base_pc(site.stream) + site.local_lo_pc,
                absolute_hi_pc=self._stream_base_pc(site.stream) + site.local_hi_pc,
            )
            for site in self.runtime_patch_sites
        ]
        self.runtime_config_attn_sites = [
            replace(
                site,
                absolute_pc=self._stream_base_pc(site.stream) + site.local_pc,
            )
            for site in self.runtime_config_attn_sites
        ]
        self._prefill_runtime_instrs: Optional[bytearray] = None
        self._decode_runtime_instrs: Optional[bytearray] = None

    def _stream_base_pc(self, stream: str) -> int:
        if stream == "prefill":
            return self.prefill_pc
        if stream == "decode":
            return self.decode_pc
        raise ValueError("stream must be 'prefill' or 'decode'")

    def _stream_offset(self, stream: str) -> int:
        if stream == "prefill":
            return self.prefill_instrs_offset
        if stream == "decode":
            return self.decode_instrs_offset
        raise ValueError("stream must be 'prefill' or 'decode'")

    def _runtime_stream(self, stream: str) -> bytearray:
        if self._prefill_runtime_instrs is None or self._decode_runtime_instrs is None:
            self.reset_runtime_images()
        if stream == "prefill":
            return self._prefill_runtime_instrs
        if stream == "decode":
            return self._decode_runtime_instrs
        raise ValueError("stream must be 'prefill' or 'decode'")

    def symbol_address(self, symbol: str) -> int:
        builtins = {
            "data_base": self.data_base,
            "shared_data": self.data_base,
            "temp_base": self.temp_base,
            "logits_base": self.logits_base,
            "kv_cache_base": self.kv_cache_base,
            "input_offset": self.input_offset,
            "prefill_logits_offset": self.prefill_logits_offset,
            "decode_logits_offset": self.decode_logits_offset,
        }
        if symbol in builtins:
            return builtins[symbol]
        if symbol in self.symbol_offsets:
            region = self.symbol_regions.get(symbol, "data")
            region_bases = {
                "data": self.data_base,
                "shared_data": self.data_base,
                "temp": self.temp_base,
                "logits": self.logits_base,
                "kv_cache": self.kv_cache_base,
            }
            if region not in region_bases:
                raise ValueError(f"Unknown ProgramBundle symbol region {region!r} for {symbol!r}")
            return region_bases[region] + int(self.symbol_offsets[symbol])
        raise KeyError(f"Unknown ProgramBundle symbol: {symbol}")

    def reset_runtime_images(self) -> None:
        streams = {
            "prefill": bytearray(self.prefill_instrs),
            "decode": bytearray(self.decode_instrs),
        }
        for site in self.relocation_sites:
            addend = read_set_addr_pair(
                streams[site.stream],
                site.local_lo_pc,
                site.local_hi_pc,
                site.addr_reg,
            )
            patch_set_addr_pair(
                streams[site.stream],
                site.local_lo_pc,
                site.local_hi_pc,
                site.addr_reg,
                self.symbol_address(site.symbol) + addend,
            )
        self._prefill_runtime_instrs = streams["prefill"]
        self._decode_runtime_instrs = streams["decode"]

    def patch_runtime_site(self, site_or_kind: Union[RuntimePatchSite, str],
                           offset: int = 0, *, stream: Optional[str] = None) -> RuntimePatchSite:
        if isinstance(site_or_kind, RuntimePatchSite):
            site = site_or_kind
        else:
            matches = [
                candidate for candidate in self.runtime_patch_sites
                if candidate.kind == site_or_kind and (stream is None or candidate.stream == stream)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected exactly one runtime patch site for kind={site_or_kind!r}, "
                    f"stream={stream!r}; found {len(matches)}"
                )
            site = matches[0]
        patch_set_addr_pair(
            self._runtime_stream(site.stream),
            site.local_lo_pc,
            site.local_hi_pc,
            site.addr_reg,
            self.symbol_address(site.base_symbol) + int(offset),
        )
        return site

    def patch_config_attn_site(self, site: RuntimeConfigAttnSite, *,
                               query_row_base: int, valid_kv_len: int) -> RuntimeConfigAttnSite:
        patch_config_attn(
            self._runtime_stream(site.stream),
            site.local_pc,
            query_row_base=query_row_base,
            valid_kv_len=valid_kv_len,
            mode=site.mode,
        )
        return site

    def stream_bytes(self, stream: str) -> bytes:
        return bytes(self._runtime_stream(stream))

    def materialize(self, *, reset_runtime: bool = True) -> bytes:
        if reset_runtime:
            self.reset_runtime_images()
        image = bytearray(self.required_dram_bytes)
        prefill = self._runtime_stream("prefill")
        decode_stream = self._runtime_stream("decode")
        image[self.prefill_instrs_offset:self.prefill_instrs_offset + len(prefill)] = prefill
        image[self.decode_instrs_offset:self.decode_instrs_offset + len(decode_stream)] = decode_stream
        image[self.data_base:self.data_base + len(self.shared_data)] = self.shared_data
        return bytes(image)

    def get_instruction_bytes(self, pc: int) -> bytes:
        image = self.materialize(reset_runtime=False)
        offset = pc * 8
        return image[offset:offset + 8]


@dataclass
class ProgramBinary:
    """Binary program format with header, instructions, and data."""
    instructions: bytes = b""
    data: bytes = b""
    entry_point: int = 0
    insn_count: int = 0
    data_base: int = 0    # byte offset of data section in unified DRAM image (0 = legacy)
    input_offset: int = 0  # byte offset of input patches region in unified DRAM image
    pos_embed_patch_dram_offset: int = 0  # byte offset of patch rows of pos_embed (rows 1-196)
    pos_embed_cls_dram_offset: int = 0  # byte offset of CLS row of pos_embed (row 0)
    cls_token_dram_offset: int = 0  # byte offset of cls_token parameter
    trace_manifest: Dict[int, List[Dict[str, Any]]] = field(default_factory=dict)
    compiler_manifest: Dict[str, Any] = field(default_factory=dict)

    def to_bytes(self) -> bytes:
        data_offset = HEADER_SIZE + len(self.instructions)
        metadata_blob = b""
        metadata_offset = 0
        metadata_size = 0
        metadata_payload: Dict[str, Any] = {}
        if self.trace_manifest:
            metadata_payload["trace_manifest"] = self.trace_manifest
        if self.compiler_manifest:
            metadata_payload["compiler_manifest"] = self.compiler_manifest
        if metadata_payload:
            metadata_blob = json.dumps(
                metadata_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            metadata_offset = data_offset + len(self.data)
            metadata_size = len(metadata_blob)
        header = struct.pack(
            HEADER_FMT,
            MAGIC,
            VERSION,
            0,  # flags
            self.insn_count,
            data_offset,
            len(self.data),
            self.entry_point,
            self.data_base,
            self.input_offset,
            self.pos_embed_patch_dram_offset,
            self.pos_embed_cls_dram_offset,
            self.cls_token_dram_offset,
            metadata_offset,
            metadata_size,
            0,  # reserved2
            0,  # reserved3
            0,  # reserved4
        )
        return header + self.instructions + self.data + metadata_blob

    @classmethod
    def from_bytes(cls, data: bytes) -> "ProgramBinary":
        if len(data) < LEGACY_HEADER_SIZE:
            raise ValueError(f"Data too short for header: {len(data)} bytes")
        magic = struct.unpack(">I", data[:4])[0]
        version = struct.unpack(">H", data[4:6])[0]
        if magic != MAGIC:
            raise ValueError(f"Bad magic: {magic:#010x}, expected {MAGIC:#010x}")

        if version == LEGACY_VERSION:
            magic, version, flags, insn_count, data_offset, data_size, entry_point, _ = \
                struct.unpack(LEGACY_HEADER_FMT, data[:LEGACY_HEADER_SIZE])
            header_size = LEGACY_HEADER_SIZE
            metadata = {
                "data_base": 0,
                "input_offset": 0,
                "pos_embed_patch_dram_offset": 0,
                "pos_embed_cls_dram_offset": 0,
                "cls_token_dram_offset": 0,
            }
            aux_metadata = {}
        elif version == RUNTIME_METADATA_VERSION:
            if len(data) < HEADER_SIZE:
                raise ValueError(f"Data too short for v{RUNTIME_METADATA_VERSION} header: {len(data)} bytes")
            unpacked = struct.unpack(HEADER_FMT, data[:HEADER_SIZE])
            (
                magic,
                version,
                flags,
                insn_count,
                data_offset,
                data_size,
                entry_point,
                data_base,
                input_offset,
                pos_embed_patch_dram_offset,
                pos_embed_cls_dram_offset,
                cls_token_dram_offset,
                _reserved0,
                _reserved1,
                _reserved2,
                _reserved3,
                _reserved4,
            ) = unpacked
            header_size = HEADER_SIZE
            metadata = {
                "data_base": data_base,
                "input_offset": input_offset,
                "pos_embed_patch_dram_offset": pos_embed_patch_dram_offset,
                "pos_embed_cls_dram_offset": pos_embed_cls_dram_offset,
                "cls_token_dram_offset": cls_token_dram_offset,
            }
            aux_metadata = {}
        elif version == VERSION:
            if len(data) < HEADER_SIZE:
                raise ValueError(f"Data too short for v{VERSION} header: {len(data)} bytes")
            unpacked = struct.unpack(HEADER_FMT, data[:HEADER_SIZE])
            (
                magic,
                version,
                flags,
                insn_count,
                data_offset,
                data_size,
                entry_point,
                data_base,
                input_offset,
                pos_embed_patch_dram_offset,
                pos_embed_cls_dram_offset,
                cls_token_dram_offset,
                metadata_offset,
                metadata_size,
                _reserved2,
                _reserved3,
                _reserved4,
            ) = unpacked
            header_size = HEADER_SIZE
            metadata = {
                "data_base": data_base,
                "input_offset": input_offset,
                "pos_embed_patch_dram_offset": pos_embed_patch_dram_offset,
                "pos_embed_cls_dram_offset": pos_embed_cls_dram_offset,
                "cls_token_dram_offset": cls_token_dram_offset,
            }
            aux_metadata = {}
            if metadata_offset or metadata_size:
                if metadata_offset < data_offset + data_size:
                    raise ValueError(
                        f"Bad metadata offset {metadata_offset}, expected >= {data_offset + data_size}"
                    )
                metadata_end = metadata_offset + metadata_size
                if metadata_end > len(data):
                    raise ValueError(
                        f"Bad metadata range [{metadata_offset}, {metadata_end}) for blob of {len(data)} bytes"
                    )
                aux_metadata = json.loads(data[metadata_offset:metadata_end].decode("utf-8"))
        else:
            raise ValueError(f"Unsupported program version: {version:#06x}")

        if data_offset < header_size:
            raise ValueError(f"Bad data offset {data_offset}, header size is {header_size}")
        instructions = data[header_size:data_offset]
        prog_data = data[data_offset:data_offset + data_size]
        trace_manifest = {}
        if aux_metadata.get("trace_manifest"):
            trace_manifest = {
                int(pc): events
                for pc, events in aux_metadata["trace_manifest"].items()
            }
        compiler_manifest = aux_metadata.get("compiler_manifest", {})
        return cls(
            instructions=instructions,
            data=prog_data,
            entry_point=entry_point,
            insn_count=insn_count,
            trace_manifest=trace_manifest,
            compiler_manifest=compiler_manifest,
            **metadata,
        )

    def to_dram_image(self) -> bytes:
        """Return unified DRAM image: instructions + alignment padding + data.

        This is the single contiguous blob the host loads into DRAM.
        Instructions start at offset 0; data starts at data_base (16-byte aligned).
        """
        insn_bytes = self.instructions
        if self.data_base > 0:
            padding_size = self.data_base - len(insn_bytes)
        else:
            aligned = (len(insn_bytes) + 15) & ~15
            padding_size = aligned - len(insn_bytes)
        return insn_bytes + bytes(padding_size) + self.data

    def get_instruction_bytes(self, pc: int) -> bytes:
        """Get the 8 bytes for instruction at given PC."""
        offset = pc * 8
        return self.instructions[offset:offset + 8]


class Assembler:
    """Two-pass assembler."""

    def assemble(self, source: str, data: bytes = b"") -> ProgramBinary:
        lines = source.strip().split('\n')

        # Pass 1: collect labels
        labels = {}
        pc = 0
        for line in lines:
            label, insn = parse_line(line)
            if label is not None:
                labels[label] = pc
            if insn is not None:
                pc += 1

        # Pass 2: emit instructions
        insn_bytes = bytearray()
        for line in lines:
            _, insn = parse_line(line)
            if insn is not None: 
                insn_bytes.extend(encode(insn)) # encding the instruction class

        insn_count = len(insn_bytes) // 8
        return ProgramBinary(
            instructions=bytes(insn_bytes),
            data=data,
            entry_point=0,
            insn_count=insn_count,
        )
