"""Empirical ISA coverage probe — ground truth for the ISA generation freeze.

Compiles GPT-2 124M through the current toolchain and histograms the opcodes
that actually land in the bundle (prefill + decode), for both W8A16 presets.
Static grep over codegen lies (it shows emit *call sites* that are
unreachable for GPT-2 W8A16); this measures what is really emitted.

Run from repo root:
    PYTHONPATH=software python3 software/tools/isa_coverage_probe.py

Referenced by `software/docs/isa_generation_freeze.md` §2 as the
re-verifiable basis for the normative opcode set. If the histogram changes,
the freeze contract must be revised.
"""
import collections
import traceback

import torch

from taccel.isa.opcodes import Opcode
from taccel.runtime.tiny_fixture import build_stage3_tiny_decoder_bundle

FIX = "software/tests/fixtures/generated/gpt2_converted_nanogpt.pt"


def main() -> int:
    payload = torch.load(FIX, map_location="cpu")
    all_ops = list(Opcode)
    emitted_union = set()

    for preset in ("weight_only_int8", "weight_only_int8_quarot"):
        try:
            tiny = build_stage3_tiny_decoder_bundle(
                payload, ptq_preset=preset, smoke_decode_steps=2
            )
            pf = tiny.build.prefill_codegen.instructions
            dc = tiny.build.decode_codegen.instructions
            cpf = collections.Counter(getattr(i, "opcode", None) for i in pf)
            cdc = collections.Counter(getattr(i, "opcode", None) for i in dc)
            emitted = {op for op in (cpf + cdc) if isinstance(op, Opcode)}
            emitted_union |= emitted
            print(f"\n=== preset={preset} "
                  f"({len(pf)} prefill + {len(dc)} decode insns) ===")
            for op in all_ops:
                mark = "" if op in emitted else "   <-- NOT EMITTED"
                print(f"  0x{op.value:02X} {op.name:26} "
                      f"prefill={cpf.get(op,0):5d} "
                      f"decode={cdc.get(op,0):5d}{mark}")
        except Exception:
            print(f"\n=== preset={preset}: BUILD FAILED ===")
            traceback.print_exc()

    print("\n=== UNION across W8A16 presets ===")
    not_emitted = [op for op in all_ops if op not in emitted_union]
    print(f"emitted {len(emitted_union)}/{len(all_ops)} opcodes")
    print("NEVER EMITTED by current GPT-2 W8A16 toolchain (non-normative):")
    for op in not_emitted:
        print(f"  0x{op.value:02X} {op.name}")
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
