#!/usr/bin/env python3
"""CLI: pytorch_model.bin → program.bin"""
import argparse
import sys
import os
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_NAME = "facebook/deit-tiny-patch16-224"


def build_calibration(model_name: str, state_dict: dict, image_paths: List[str]):
    """Load a DeiT model and calibrate it on local image files."""
    from PIL import Image
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification
    from taccel.quantizer.calibrate import calibrate_model

    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_config(config)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    processor = AutoImageProcessor.from_pretrained(model_name)

    sample_inputs = []
    for path in image_paths:
        with Image.open(path) as img:
            sample_inputs.append(processor(images=img.convert("RGB"), return_tensors="pt"))
    return calibrate_model(model, sample_inputs)


def main():
    parser = argparse.ArgumentParser(description="TACCEL compiler: PyTorch model → ProgramBinary")
    parser.add_argument("--weights", required=True, help="PyTorch weights file (pytorch_model.bin)")
    parser.add_argument("--model", default="deit-tiny",
                        choices=["deit-tiny"], help="Model architecture")
    parser.add_argument("-o", "--output", default="program.bin", help="Output .bin file")
    parser.add_argument("--calibration-images", nargs="*",
                        help="Image files for calibration (optional)")
    parser.add_argument(
        "--gelu-from-accum",
        action="store_true",
        help="Enable the experimental GELU-from-ACCUM codegen path",
    )
    parser.add_argument(
        "--requant-pc-qkv",
        action="store_true",
        help="Enable the experimental REQUANT_PC path for per-head Q/K/V projections",
    )
    parser.add_argument(
        "--requant-pc-out-proj",
        action="store_true",
        help="Enable the experimental REQUANT_PC path for out_proj matmuls",
    )
    args = parser.parse_args()

    print(f"Loading weights from {args.weights}...")
    import torch
    state_dict = torch.load(args.weights, map_location="cpu", weights_only=False)
    if hasattr(state_dict, 'items') is False:
        state_dict = state_dict.state_dict()

    calibration = None
    if args.calibration_images:
        print(f"Calibrating on {len(args.calibration_images)} image(s)...")
        calibration = build_calibration(MODEL_NAME, state_dict, args.calibration_images)
        print(f"  Collected {len(calibration.scales)} activation scales")

    print(f"Compiling {args.model}...")
    from taccel.compiler import Compiler
    compiler = Compiler()
    prog = compiler.compile(
        state_dict,
        calibration=calibration,
        gelu_from_accum=args.gelu_from_accum,
        requant_pc_qkv=args.requant_pc_qkv,
        requant_pc_out_proj=args.requant_pc_out_proj,
    )

    with open(args.output, 'wb') as f:
        f.write(prog.to_bytes())

    print(f"\nCompilation complete:")
    print(f"  Instructions: {prog.insn_count}")
    print(f"  Instruction bytes: {len(prog.instructions):,}")
    print(f"  Data (weights): {len(prog.data):,} bytes")
    if calibration is not None:
        print(f"  Input offset: {prog.input_offset}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
