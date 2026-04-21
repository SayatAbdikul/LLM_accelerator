"""Compare KV cache contents to what the reference computes."""
import sys
import numpy as np
sys.path.insert(0, "/Users/sayat/Documents/GitHub/LLM_accelerator/software")

import torch
from taccel.runtime.calibration import build_calibration_scales
from taccel.runtime.tiny_fixture import build_stage3_tiny_decoder_bundle
from taccel.runtime.host_runner import HostRunner
from taccel.runtime.fake_quant_reference import _qdq, _layernorm_np, _scale, _int8_saturating_add, _fp32_to_int8, _to_f32, _fq_linear
from taccel.quantizer.quantize import quantize_tensor

FIXTURE = "/Users/sayat/Documents/GitHub/LLM_accelerator/software/tests/fixtures/generated/nanogpt_shakespeare_char_d128_l2.pt"

payload = torch.load(FIXTURE, map_location="cpu")
scales = build_calibration_scales(payload)
sd = payload["state_dict"]
model_args = payload["model_args"]
vocab = int(model_args["vocab_size"])
n_head = int(model_args["n_head"])
n_layer = int(model_args["n_layer"])
d_model = int(model_args["n_embd"])
d_head = d_model // n_head

SMOKE = 4
tiny = build_stage3_tiny_decoder_bundle(payload, smoke_decode_steps=SMOKE, calibration_scales=scales)
bundle = tiny.build.bundle
runner = HostRunner(bundle, logits_dtype=np.int8)

# Prefill
runner.run_prefill([0])

kv_step_bytes = int(bundle.kv_step_bytes)
kv_cache_base = int(bundle.kv_cache_base)
print(f"d_head={d_head}, n_head={n_head}, kv_step_bytes={kv_step_bytes}")
print(f"kv_cache_base DRAM offset: {kv_cache_base}")

# Read K0 for layer 0, head 0 from KV cache
# Layout: from kv_cache.py - find the entry for layer 0, key, head 0
kv_layout = tiny.build.kv_layout
entry_L0_K_H0 = kv_layout.entry(0, "key", 0)
print(f"entry_L0_K_H0: step_offset={entry_L0_K_H0.step_offset}, base_symbol={entry_L0_K_H0.base_symbol}")

# K0 is at: kv_cache_base + entry.step_offset + 0 * kv_step_bytes
k0_offset = kv_cache_base + entry_L0_K_H0.step_offset + 0 * kv_step_bytes
k0_from_cache = np.frombuffer(
    bytes(runner.simulator.state.dram[k0_offset : k0_offset + d_head]),
    dtype=np.int8,
)
print(f"\nK0 layer0 head0 from KV cache (first 8): {k0_from_cache[:8]}")

# Compute reference K0
wte_q, _ = quantize_tensor(_to_f32(sd["transformer.wte.weight"]), per_channel=False)
wpe_q, _ = quantize_tensor(_to_f32(sd["transformer.wpe.weight"]), per_channel=False)
x_int8 = _int8_saturating_add(wte_q[[0]], wpe_q[[0]])
x = x_int8.astype(np.float32) * np.float32(_scale(scales, "tok_pos_add"))
eps = np.float32(1e-6)
ln1_w = _to_f32(sd["transformer.h.0.ln_1.weight"])
ln1_b = _to_f32(sd["transformer.h.0.ln_1.bias"])
ln1 = _qdq(_layernorm_np(x, ln1_w, ln1_b, eps), _scale(scales, "block0_ln1"))
k_w = _fq_linear(sd["transformer.h.0.attn.c_attn.weight_h0_key"])
k = _qdq(ln1 @ k_w.T, _scale(scales, "block0_head0_key"))  # [1, d_head]
k_scale = _scale(scales, "block0_head0_key")
k0_ref_int8 = _fp32_to_int8(k, k_scale)
print(f"K0 from reference (first 8): {k0_ref_int8[0, :8]}")
print(f"K0 match: {np.array_equal(k0_from_cache, k0_ref_int8[0])}")

# Exact REQUANT formula
ln1_int8 = _fp32_to_int8(ln1, _scale(scales, "block0_ln1"))
k_weight = _to_f32(sd["transformer.h.0.attn.c_attn.weight_h0_key"])
k_w_int8, k_w_scales = quantize_tensor(k_weight, per_channel=True)
mean_k_scale = float(np.mean(k_w_scales.astype(np.float32)))
ACCUM = ln1_int8.astype(np.int32) @ k_w_int8.astype(np.int32).T

ln1_scale_val = _scale(scales, "block0_ln1")
k_scale_val = _scale(scales, "block0_head0_key")
requant_fp64 = ln1_scale_val * mean_k_scale / k_scale_val
requant_fp16 = float(np.float16(requant_fp64))
k0_int8_exact = np.clip(np.round(ACCUM[0].astype(np.float32) * np.float32(requant_fp16)), -128, 127).astype(np.int8)
print(f"\nrequant_fp64={requant_fp64:.7f}, requant_fp16={requant_fp16:.7f}")
print(f"K0 exact REQUANT (first 8): {k0_int8_exact[:8]}")
print(f"KV cache matches exact REQUANT: {np.array_equal(k0_from_cache, k0_int8_exact)}")
print(f"Reference matches exact REQUANT: {np.array_equal(k0_ref_int8[0], k0_int8_exact)}")

if not np.array_equal(k0_from_cache, k0_ref_int8[0]):
    diff = k0_from_cache.astype(np.int32) - k0_ref_int8[0].astype(np.int32)
    print(f"\nDiff: max={np.abs(diff).max()}, mean={np.abs(diff).mean():.2f}")
    mismatch = np.where(diff != 0)[0]
    print(f"Mismatched indices: {mismatch[:10]}")
    print(f"cache vals: {k0_from_cache[mismatch[:10]]}")
    print(f"ref vals:   {k0_ref_int8[0, mismatch[:10]]}")
    print(f"exact vals: {k0_int8_exact[mismatch[:10]]}")
