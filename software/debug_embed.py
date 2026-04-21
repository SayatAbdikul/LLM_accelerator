"""Debug the embedding path divergence."""
import sys
import numpy as np
sys.path.insert(0, "/Users/sayat/Documents/GitHub/LLM_accelerator/software")

import torch
from taccel.runtime.calibration import build_calibration_scales
from taccel.runtime.tiny_fixture import build_stage3_tiny_decoder_bundle
from taccel.runtime.host_runner import HostRunner
from taccel.runtime.fake_quant_reference import _to_f32
from taccel.quantizer.quantize import quantize_tensor

FIXTURE = "/Users/sayat/Documents/GitHub/LLM_accelerator/software/tests/fixtures/generated/nanogpt_shakespeare_char_d128_l2.pt"

payload = torch.load(FIXTURE, map_location="cpu")
scales = build_calibration_scales(payload)
sd = payload["state_dict"]
model_args = payload["model_args"]

TOK = 0

# === What the golden model does ===
# 1. Quantize wte with global per-tensor scale
wte_fp32 = _to_f32(sd["transformer.wte.weight"])
wpe_fp32 = _to_f32(sd["transformer.wpe.weight"])
wte_int8, wte_sc = quantize_tensor(wte_fp32, per_channel=False)
wpe_int8, wpe_sc = quantize_tensor(wpe_fp32, per_channel=False)

print(f"wte global scale: {float(wte_sc[0]):.6f}")
print(f"wpe global scale: {float(wpe_sc[0]):.6f}")
print(f"calibration tok_embed scale: {scales['tok_embed']:.6f}")
print(f"calibration pos_embed scale: {scales['pos_embed']:.6f}")
print(f"calibration tok_pos_add scale: {scales['tok_pos_add']:.6f}")

# 2. VADD (INT8 saturating add)
tok_int8 = wte_int8[TOK]  # [d_model]
pos_int8 = wpe_int8[0]    # position 0, [d_model]
vadd_int8 = np.clip(tok_int8.astype(np.int32) + pos_int8.astype(np.int32), -128, 127).astype(np.int8)

# 3. Scale by tok_pos_add_scale (what layernorm uses)
x_golden = vadd_int8.astype(np.float32) * scales["tok_pos_add"]
print(f"\nGolden x (first 8): {x_golden[:8]}")
print(f"Golden x range: [{x_golden.min():.5f}, {x_golden.max():.5f}]")

# === What the reference does ===
from taccel.quantizer.quantize import dequantize_tensor
wte_fq = dequantize_tensor(wte_int8, wte_sc).astype(np.float32)
wpe_fq = dequantize_tensor(wpe_int8, wpe_sc).astype(np.float32)

def qdq(x, scale):
    s = np.float32(scale)
    q = np.clip(np.round(x.astype(np.float32) / s), -128, 127).astype(np.int8)
    return q.astype(np.float32) * s

tok_e = qdq(wte_fq[TOK], scales["tok_embed"])
pos_e = qdq(wpe_fq[0], scales["pos_embed"])
x_ref = qdq(tok_e + pos_e, scales["tok_pos_add"])

print(f"\nRef x (first 8): {x_ref[:8]}")
print(f"Ref x range: [{x_ref.min():.5f}, {x_ref.max():.5f}]")

# Compare
diff = (x_golden - x_ref)
print(f"\nDiff x (first 8): {diff[:8]}")
print(f"Max |diff|: {np.abs(diff).max():.6f}")
print(f"Mean |diff|: {np.abs(diff).mean():.6f}")

# Count how many elements differ
in_quant_units_golden = (x_golden / scales["tok_pos_add"]).round().astype(np.int32)
in_quant_units_ref = (x_ref / scales["tok_pos_add"]).round().astype(np.int32)
print(f"\nDiff in INT8 units: max={np.abs(in_quant_units_golden - in_quant_units_ref).max()}")
print(f"Num elements differ: {np.sum(in_quant_units_golden != in_quant_units_ref)}")

# === Correct reference approach ===
x_correct = vadd_int8.astype(np.float32) * scales["tok_pos_add"]
print(f"\n=== Correct x matches golden: {np.array_equal(x_golden, x_correct)}")
