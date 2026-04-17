#!/usr/bin/env python3
"""Compile a named compare_golden preset and save its assembly listing."""
import argparse
import os
import sys
from pathlib import Path


TOOL_DIR = Path(__file__).resolve().parent
SOFTWARE_DIR = TOOL_DIR.parent
sys.path.insert(0, str(SOFTWARE_DIR))

from transformers import AutoImageProcessor

from compare_golden import (
    MODEL_NAME,
    compile_model,
    get_diagnostic_preset,
    load_flat_local_images,
    load_local_images,
    load_model,
    preset_compile_kwargs,
)
from taccel.assembler import Disassembler


def _strip_offsets(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if "] " in line:
            lines.append(line.split("] ", 1)[1])
        else:
            lines.append(line)
    return "\n".join(lines)


def _load_calibration_images(preset: dict):
    benchmark = preset["benchmark"]
    calibration_ids = benchmark["calibration_image_ids"]
    dataset = benchmark["benchmark_dataset"]
    image_root = benchmark.get("local_benchmark_image_dir", "")

    if dataset == "frozen_coco":
        loaded = load_local_images(
            calibration_ids,
            "calibration",
            image_root=image_root,
        )
        return [img for _, img in loaded]

    if dataset in {"cats_dogs_local", "local_flat"}:
        loaded = load_flat_local_images(
            calibration_ids,
            "calibration",
            image_root=image_root,
        )
        return [sample["image"] for sample in loaded]

    raise ValueError(
        f"Unsupported benchmark dataset for assembly export: {dataset!r}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compile a compare_golden preset and export its assembly listing"
    )
    parser.add_argument(
        "--diagnostic-preset",
        default="current_best_sq_ln2_fc1_b0_8_10",
        help="compare_golden diagnostic preset to compile",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output .asm path (default: software/out/<preset>_inference.asm)",
    )
    parser.add_argument(
        "--program-bin",
        help="Optional ProgramBinary .bin path to save alongside the assembly",
    )
    parser.add_argument(
        "--no-offsets",
        action="store_true",
        help="Strip [0xNNNN] PC offsets from the assembly listing",
    )
    args = parser.parse_args()

    os.chdir(SOFTWARE_DIR)
    preset = get_diagnostic_preset(args.diagnostic_preset)
    if preset is None:
        raise ValueError(
            f"Unknown diagnostic preset {args.diagnostic_preset!r}"
        )

    output_path = Path(args.output) if args.output else (
        SOFTWARE_DIR / "out" / f"{args.diagnostic_preset}_inference.asm"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bin_path = Path(args.program_bin) if args.program_bin else None
    if bin_path is not None:
        bin_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading model for preset: {args.diagnostic_preset}")
    model, state_dict = load_model()
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    print("Loading calibration images...")
    calibration_samples = _load_calibration_images(preset)
    print(f"  Loaded {len(calibration_samples)} calibration images")

    print("Compiling accelerator program...")
    program, _ = compile_model(
        model,
        state_dict,
        calibration_samples,
        processor,
        **preset_compile_kwargs(preset),
    )

    if bin_path is not None:
        bin_path.write_bytes(program.to_bytes())
        print(f"Saved ProgramBinary: {bin_path}")

    print("Disassembling program...")
    text = Disassembler().disassemble(program)
    if args.no_offsets:
        text = _strip_offsets(text)
    output_path.write_text(text + "\n")

    print(f"Saved assembly: {output_path}")
    print(f"Instruction count: {program.insn_count}")
    print(f"Instruction bytes: {len(program.instructions):,}")
    print(f"Data bytes: {len(program.data):,}")


if __name__ == "__main__":
    main()
