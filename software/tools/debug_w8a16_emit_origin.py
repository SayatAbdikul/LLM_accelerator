#!/usr/bin/env python3
"""Find which codegen emit function produces a given QUANT_FP32_INT8 PC.

Wraps CodeGenerator._emit during build so we can capture the call stack
for every QuantFp32Int8Insn emission, tagged with the codegen instance
id and the in-stream insn index at emit time.
"""
from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import torch

from taccel.compiler import codegen as codegen_mod
from taccel.isa.instructions import QuantFp32Int8Insn
from taccel.runtime.tiny_fixture import build_stage3_tiny_decoder_bundle


CAPTURED: list = []


def _patch_codegen():
    real_emit = codegen_mod.CodeGenerator._emit

    def wrapped(self, insn):
        if isinstance(insn, QuantFp32Int8Insn):
            stack = traceback.extract_stack(limit=24)
            interesting = [
                f for f in stack
                if "taccel" in f.filename and "_emit" not in f.name
                and "wrapped" not in f.name
            ]
            CAPTURED.append({
                "cg_id": id(self),
                "insn_index": len(self.instructions),
                "current_node_idx": getattr(self, "current_node_idx", None),
                "src1_off": insn.src1_off,
                "dst_off": insn.dst_off,
                "flags": insn.flags,
                "sreg": insn.sreg,
                "stack": [(f.filename.split("/")[-1], f.lineno, f.name) for f in interesting[-8:]],
            })
        return real_emit(self, insn)

    codegen_mod.CodeGenerator._emit = wrapped


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint",
        type=Path,
        nargs="?",
        default=Path("software/tests/fixtures/generated/gpt2_converted_nanogpt.pt"),
    )
    parser.add_argument("--target-pc", type=int, action="append", default=[62219, 62223])
    args = parser.parse_args(argv)

    _patch_codegen()
    payload = torch.load(args.checkpoint, map_location="cpu")
    tiny = build_stage3_tiny_decoder_bundle(
        payload, smoke_decode_steps=5, ptq_preset="weight_only_int8", fp_precision="fp16"
    )

    decode_pc_base = tiny.build.bundle.decode_pc
    prefill_pc_base = tiny.build.bundle.prefill_pc
    decode_cg_id = id(tiny.build.decode_codegen)
    prefill_cg_id = id(tiny.build.prefill_codegen)

    target_pcs = set(int(pc) for pc in args.target_pc)
    print(f"Looking for emit origins of PCs: {sorted(target_pcs)}")
    print(f"prefill_pc={prefill_pc_base}, decode_pc={decode_pc_base}")
    print(f"prefill_cg_id={prefill_cg_id}, decode_cg_id={decode_cg_id}")
    print(f"Captured {len(CAPTURED)} QuantFp32Int8 emissions total")
    print()

    matches = []
    for entry in CAPTURED:
        if entry["cg_id"] == decode_cg_id:
            abs_pc = decode_pc_base + entry["insn_index"]
            stream = "decode"
        elif entry["cg_id"] == prefill_cg_id:
            abs_pc = prefill_pc_base + entry["insn_index"]
            stream = "prefill"
        else:
            continue
        if abs_pc in target_pcs:
            matches.append((abs_pc, stream, entry))

    if not matches:
        # Show a few nearby decode-stream QUANTs for sanity.
        decode_only = [
            (decode_pc_base + e["insn_index"], "decode", e)
            for e in CAPTURED if e["cg_id"] == decode_cg_id
        ]
        decode_only.sort(key=lambda x: x[0])
        # Find PCs nearest to the target.
        for tgt in target_pcs:
            nearby = sorted(decode_only, key=lambda x: abs(x[0] - tgt))[:3]
            print(f"PC={tgt}: no exact match; nearest decode PCs:")
            for pc, _, e in nearby:
                print(f"  PC={pc} insn_idx={e['insn_index']} dst={e['dst_off']} src1={e['src1_off']}")
            print()
    else:
        for abs_pc, stream, entry in matches:
            print(f"--- PC={abs_pc} (stream={stream}) ---")
            print(f"  src1_off={entry['src1_off']}, dst_off={entry['dst_off']}, "
                  f"flags={entry['flags']}, sreg={entry['sreg']}")
            print(f"  current_node_idx: {entry['current_node_idx']}")
            for fn, ln, name in entry["stack"]:
                print(f"    {fn}:{ln} in {name}()")
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
