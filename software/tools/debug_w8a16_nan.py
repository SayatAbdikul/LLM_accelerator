#!/usr/bin/env python3
"""Find why QUANT_FP32_INT8 reads NaN in the W8A16 path.

Hypothesis: an upstream INT8 write left bytes in ABUF that, when later
re-read as FP16, decode as NaN (FP16 NaN/inf bit patterns include any
exponent=0x1F encoding, common in INT8 byte pairs like 0xFF, 0xFE...).

This tool patches ALL ABUF/WBUF write helpers to log
(pc, kind, buf_id, byte_start, byte_end, size_bytes). On the first
QUANT_FP32_INT8 with NaN in its source, it dumps every overlapping write
to that source region, ordered most-recent first.
"""
from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Deque

import numpy as np
import torch

from taccel.golden_model import memory as mem_helpers
from taccel.golden_model.simulator import Simulator
from taccel.isa.opcodes import Opcode, BUF_ABUF, BUF_WBUF
from taccel.runtime.host_runner import HostRunner
from taccel.runtime.tiny_fixture import build_stage3_tiny_decoder_bundle


WRITE_LOG: Deque[dict] = deque(maxlen=20000)
INSN_LOG: Deque[dict] = deque(maxlen=64)
RESULT: dict | None = None
CUR_SIM: Simulator | None = None


def _safe_int(v, default: int = -1) -> int:
    """Coerce optional int-like values without treating 0 as falsy."""
    if v is None:
        return int(default)
    return int(v)


def _patch_writers():
    """Patch every write helper to record (pc, kind, buf_id, byte_range)."""
    targets = {
        "write_int8_tile": ("int8", 1),
        "write_int32_tile": ("int32", 4),
        "write_fp16_tile": ("fp16", 2),
        "write_fp16_vector": ("fp16_vec", 2),
        "write_bytes": ("bytes", 1),
    }
    for name, (kind, _) in targets.items():
        real = getattr(mem_helpers, name)

        def make(real_fn, kind_label):
            def wrapped(state, buf_id, offset_units, data):
                pc = (
                    int(getattr(CUR_SIM.state, "current_pc", -1))
                    if CUR_SIM is not None
                    else -1
                )
                if kind_label == "bytes":
                    size = len(data)
                else:
                    arr = np.asarray(data)
                    if kind_label == "int8":
                        size = arr.size
                    elif kind_label == "int32":
                        size = arr.size * 4
                    elif kind_label == "fp32":
                        size = arr.size * 4
                    elif kind_label in ("fp16", "fp16_vec"):
                        size = arr.size * 2
                    else:
                        size = arr.nbytes
                byte_start = int(offset_units) * mem_helpers.UNIT
                WRITE_LOG.append(
                    {
                        "pc": pc,
                        "kind": kind_label,
                        "buf_id": int(buf_id),
                        "byte_start": byte_start,
                        "byte_end": byte_start + int(size),
                        "size_bytes": int(size),
                        "shape": (tuple(arr.shape) if kind_label != "bytes" else None),
                    }
                )
                return real_fn(state, buf_id, offset_units, data)

            return wrapped

        setattr(mem_helpers, name, make(real, kind))


class TracingSim(Simulator):
    def step(self):
        from taccel.isa.encoding import decode

        prev_pc = int(self.state.pc)
        self.state.current_pc = prev_pc
        if self.program.data_base > 0:
            raw = bytes(self.state.dram[prev_pc * 8: prev_pc * 8 + 8])
        else:
            raw = self.program.get_instruction_bytes(prev_pc)
        try:
            cur_insn = decode(raw)
        except ValueError:
            cur_insn = None

        # PRE-check QUANT_FP32_INT8 source for NaN/inf.
        global RESULT
        if (
            RESULT is None
            and cur_insn is not None
            and cur_insn.opcode == Opcode.QUANT_FP32_INT8
            and self.state.tile_config is not None
        ):
            m_tiles = self.state.tile_config[0] + 1
            n_tiles = self.state.tile_config[1] + 1
            M = m_tiles * 16
            N = n_tiles * 16
            fp_precision = int(cur_insn.flags) & 0x1
            try:
                src = mem_helpers.read_fp16_tile(
                    self.state, cur_insn.src1_buf, cur_insn.src1_off, M, N
                )
                src_f32 = src.astype(np.float32, copy=False)
            except Exception:
                src_f32 = None
            scale = float(self.state.scale_regs[cur_insn.sreg])
            if src_f32 is not None and not np.isfinite(src_f32).all():
                bpe = 2 if fp_precision == 1 else 4
                src_byte_start = int(cur_insn.src1_off) * mem_helpers.UNIT
                src_byte_end = src_byte_start + M * N * bpe
                # Pull raw bytes that the QUANT just saw.
                buf = self.state.get_buffer(cur_insn.src1_buf)
                raw_bytes = bytes(buf[src_byte_start:src_byte_end])
                first_idx = np.argwhere(~np.isfinite(src_f32))[0]
                nan_row, nan_col = int(first_idx[0]), int(first_idx[1])
                nan_elem_offset = nan_row * N + nan_col
                nan_byte_offset = src_byte_start + nan_elem_offset * bpe
                nan_raw = raw_bytes[
                    nan_elem_offset * bpe: (nan_elem_offset + 1) * bpe
                ]
                # Find writes that overlap with this source region.
                overlaps = []
                for w in WRITE_LOG:
                    if w["buf_id"] != int(cur_insn.src1_buf):
                        continue
                    if w["byte_end"] <= src_byte_start:
                        continue
                    if w["byte_start"] >= src_byte_end:
                        continue
                    overlaps.append(w)
                # Find writes that contain the specific NaN-byte.
                nan_writes = [
                    w
                    for w in overlaps
                    if w["byte_start"] <= nan_byte_offset < w["byte_end"]
                ]
                RESULT = {
                    "quant_pc": prev_pc,
                    "src_buf": int(cur_insn.src1_buf),
                    "src_off_units": int(cur_insn.src1_off),
                    "src_byte_range": (src_byte_start, src_byte_end),
                    "tile_shape": src.shape,
                    "fp_precision_flag": fp_precision,
                    "sreg_value": scale,
                    "src_n_nan": int(np.isnan(src_f32).sum()),
                    "src_n_inf": int(np.isinf(src_f32).sum()),
                    "first_nan_idx": (nan_row, nan_col),
                    "first_nan_byte_offset": nan_byte_offset,
                    "first_nan_raw_bytes": " ".join(f"0x{b:02x}" for b in nan_raw),
                    "n_writes_in_region": len(overlaps),
                    "n_writes_at_nan_byte": len(nan_writes),
                    "overlapping_writes": overlaps,
                    "writes_at_nan_byte": nan_writes,
                }
                self.state.halted = True
                return

        super().step()

        if cur_insn is not None:
            INSN_LOG.append(
                {
                    "pc": prev_pc,
                    "op": cur_insn.opcode.name,
                    "dst_buf": _safe_int(getattr(cur_insn, "dst_buf", None)),
                    "dst_off": _safe_int(getattr(cur_insn, "dst_off", None)),
                    "src1_buf": _safe_int(getattr(cur_insn, "src1_buf", None)),
                    "src1_off": _safe_int(getattr(cur_insn, "src1_off", None)),
                    "src2_buf": _safe_int(getattr(cur_insn, "src2_buf", None)),
                    "src2_off": _safe_int(getattr(cur_insn, "src2_off", None)),
                    "flags": _safe_int(getattr(cur_insn, "flags", None), default=0),
                    "sreg": _safe_int(getattr(cur_insn, "sreg", None)),
                }
            )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint",
        type=Path,
        nargs="?",
        default=Path("software/tests/fixtures/generated/gpt2_converted_nanogpt.pt"),
    )
    parser.add_argument("--prompt-id", type=int, default=796)
    parser.add_argument("--max-new-tokens", type=int, default=5)
    parser.add_argument("--decode-steps", type=int, default=5)
    parser.add_argument("--ptq-preset", default="weight_only_int8")
    args = parser.parse_args(argv)

    payload = torch.load(args.checkpoint, map_location="cpu")
    tiny = build_stage3_tiny_decoder_bundle(
        payload,
        smoke_decode_steps=args.max_new_tokens,
        ptq_preset=args.ptq_preset,
    )

    global CUR_SIM
    sim = TracingSim()
    CUR_SIM = sim
    _patch_writers()

    runner = HostRunner(
        tiny.build.bundle,
        simulator=sim,
        logits_dtype=np.float16,
    )

    print(
        f"Running W8A16 bundle (prompt_id={args.prompt_id}, "
        f"decode_steps={args.decode_steps})..."
    )
    vocab_size = int(payload["model_args"]["vocab_size"])
    logits = runner.run_prefill([int(args.prompt_id)])
    if RESULT is None and args.decode_steps > 0:
        active = np.asarray(logits, dtype=np.float32)[: int(vocab_size)]
        next_token = int(np.argmax(active))
        for step in range(int(args.decode_steps)):
            position = 1 + step
            print(
                f"  decode step {step}: input token = {next_token}, position = {position}"
            )
            logits = runner.run_decode_step(int(next_token), int(position))
            if RESULT is not None:
                break
            active = np.asarray(logits, dtype=np.float32)[: int(vocab_size)]
            next_token = int(np.argmax(active))

    if RESULT is None:
        print("No QUANT_FP32_INT8 saw NaN/inf in its source.")
        return 0

    print()
    print("=" * 76)
    print(
        f"QUANT_FP32_INT8 at PC={RESULT['quant_pc']} sees {RESULT['src_n_nan']} NaN(s) "
        f"+ {RESULT['src_n_inf']} inf(s)"
    )
    print("=" * 76)
    print(f"  source region: buf={RESULT['src_buf']}, "
          f"off_units={RESULT['src_off_units']}, "
          f"byte_range={RESULT['src_byte_range']}, "
          f"shape={RESULT['tile_shape']}, "
          f"fp_precision_flag={RESULT['fp_precision_flag']}")
    print(f"  sreg_value: {RESULT['sreg_value']}")
    print(f"  first NaN at (row={RESULT['first_nan_idx'][0]}, "
          f"col={RESULT['first_nan_idx'][1]}) "
          f"byte_offset={RESULT['first_nan_byte_offset']}")
    print(f"  raw bytes at first NaN: {RESULT['first_nan_raw_bytes']}")
    print()
    print(
        f"  {RESULT['n_writes_in_region']} writes overlap the source region; "
        f"{RESULT['n_writes_at_nan_byte']} cover the exact NaN byte:"
    )
    print()
    print("  Writes covering the NaN byte (oldest first):")
    for w in RESULT["writes_at_nan_byte"]:
        print(
            f"    PC={w['pc']:5d}  kind={w['kind']:8s}  "
            f"byte_range=({w['byte_start']}, {w['byte_end']})  "
            f"size={w['size_bytes']}  shape={w['shape']}"
        )
    print()
    print("  All region-overlapping writes (oldest first, last 16):")
    for w in RESULT["overlapping_writes"][-16:]:
        print(
            f"    PC={w['pc']:5d}  kind={w['kind']:8s}  "
            f"byte_range=({w['byte_start']}, {w['byte_end']})  "
            f"size={w['size_bytes']}  shape={w['shape']}"
        )
    print()
    print(f"  Last {len(INSN_LOG)} instructions (oldest first):")
    for entry in INSN_LOG:
        marker = "  >>>" if entry["pc"] == RESULT["quant_pc"] else "     "
        print(
            f"{marker} PC={entry['pc']:5d}  op={entry['op']:32s}  "
            f"dst=(b{entry['dst_buf']},{entry['dst_off']})  "
            f"src1=(b{entry['src1_buf']},{entry['src1_off']})  "
            f"src2=(b{entry['src2_buf']},{entry['src2_off']})  "
            f"flags={entry['flags']}  sreg={entry['sreg']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
