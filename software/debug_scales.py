"""Check what scales the codegen actually uses for attn_v in prefill/decode."""
import sys
import numpy as np
sys.path.insert(0, "/Users/sayat/Documents/GitHub/LLM_accelerator/software")

import torch
from taccel.runtime.calibration import build_calibration_scales
from taccel.runtime.tiny_fixture import build_stage3_tiny_decoder_bundle

FIXTURE = "/Users/sayat/Documents/GitHub/LLM_accelerator/software/tests/fixtures/generated/nanogpt_shakespeare_char_d128_l2.pt"

payload = torch.load(FIXTURE, map_location="cpu")
scales = build_calibration_scales(payload)

print("=== Calibration scales (before codegen) ===")
for k in sorted(scales.keys()):
    if "softmax" in k or "attn_v" in k or "value" in k or "key" in k:
        print(f"  {k}: {scales[k]:.7f}")

tiny = build_stage3_tiny_decoder_bundle(payload, smoke_decode_steps=4, calibration_scales=scales)

print("\n=== After prefill codegen ===")
pcg = tiny.build.prefill_codegen
for k in sorted(pcg.calibration_scales.keys()):
    if "softmax" in k or "attn_v" in k:
        print(f"  {k}: {pcg.calibration_scales[k]:.7f}")

print("\n=== After decode codegen ===")
dcg = tiny.build.decode_codegen
for k in sorted(dcg.calibration_scales.keys()):
    if "softmax" in k or "attn_v" in k:
        print(f"  {k}: {dcg.calibration_scales[k]:.7f}")

# Also check if scale_mul is in calibration
print("\n=== scale (scale_mul) entries after codegen ===")
for k in sorted(pcg.calibration_scales.keys()):
    if "_scale" in k and "head" in k:
        print(f"  prefill/{k}: {pcg.calibration_scales[k]:.7f}")

print("Reference softmax scale (original):", scales.get("block0_head0_softmax", "NOT IN SCALES"))
