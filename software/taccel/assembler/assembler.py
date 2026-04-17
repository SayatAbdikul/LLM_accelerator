"""Two-pass assembler: text assembly → ProgramBinary."""
import json
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from .syntax import parse_line
from ..isa.encoding import encode
from ..isa.instructions import Instruction


MAGIC = 0x54414343  # "TACC", needed fro checking the binary file format and version compatibility.
LEGACY_VERSION = 0x0001
RUNTIME_METADATA_VERSION = 0x0002
VERSION = 0x0003
LEGACY_HEADER_FMT = ">IHHIIIIQ"
HEADER_FMT = ">IHH" + "I" * 14
LEGACY_HEADER_SIZE = struct.calcsize(LEGACY_HEADER_FMT)
HEADER_SIZE = struct.calcsize(HEADER_FMT)


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
