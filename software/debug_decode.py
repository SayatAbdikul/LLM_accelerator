"""Debug the decode step divergence: compare golden vs reference for step 1."""
import sys
import numpy as np
sys.path.insert(0, "/Users/sayat/Documents/GitHub/LLM_accelerator/software")

import torch
from taccel.runtime.calibration import build_calibration_scales
from taccel.runtime.tiny_fixture import build_stage3_tiny_decoder_bundle
from taccel.runtime.fake_quant_reference import NanoGPTFQReference, _qdq, _layernorm_np, _scale, _int8_saturating_add, _fp32_to_int8, _to_f32, _fq_linear, _causal_softmax, _gelu_np
from taccel.runtime.host_runner import HostRunner
from taccel.runtime.fake_quant import cosine_similarity
from taccel.quantizer.quantize import quantize_tensor

FIXTURE = "/Users/sayat/Documents/GitHub/LLM_accelerator/software/tests/fixtures/generated/nanogpt_shakespeare_char_d128_l2.pt"

payload = torch.load(FIXTURE, map_location="cpu")
scales = build_calibration_scales(payload)
sd = payload["state_dict"]
model_args = payload["model_args"]
vocab = int(model_args["vocab_size"])
n_head = int(model_args["n_head"])
d_head = int(model_args["n_embd"]) // n_head

# Build golden model
tiny = build_stage3_tiny_decoder_bundle(payload, smoke_decode_steps=4, calibration_scales=scales)
runner = HostRunner(tiny.build.bundle, logits_dtype=np.int8)

# Prefill
prefill_logits = runner.run_prefill([0])
greedy_tok = int(np.argmax(prefill_logits[:vocab]))
print(f"Prefill greedy token: {greedy_tok}")
print(f"Prefill logits (first 10): {prefill_logits[:10]}")

# Decode step 1
decode_logits = runner.run_decode_step(greedy_tok, 1)
print(f"\nDecode step 1 logits (first 10): {decode_logits[:10]}")

# Reference: compute forward([0, greedy_tok]) and compare last row
ref = NanoGPTFQReference(
    state_dict=sd,
    model_args=model_args,
    scales=scales,
)
ref_logits_step1 = ref.forward([0, greedy_tok])
print(f"Ref.forward([0, {greedy_tok}]) logits (first 10): {ref_logits_step1[:10]}")

cos = cosine_similarity(decode_logits[:vocab].astype(np.float32), ref_logits_step1[:vocab].astype(np.float32))
print(f"Cosine: {cos:.4f}")
print(f"Max diff: {np.abs(decode_logits[:vocab].astype(np.int32) - ref_logits_step1[:vocab].astype(np.int32)).max()}")

# Now let's trace the reference step by step for seq=2
print("\n=== Tracing ref seq=2 step by step ===")
wte_q, _ = quantize_tensor(_to_f32(sd["transformer.wte.weight"]), per_channel=False)
wpe_q, _ = quantize_tensor(_to_f32(sd["transformer.wpe.weight"]), per_channel=False)

tids = [0, greedy_tok]
pids = [0, 1]
s = scales

x_int8 = _int8_saturating_add(wte_q[tids], wpe_q[pids])
x_scale = _scale(s, "tok_pos_add")
x = x_int8.astype(np.float32) * np.float32(x_scale)
print(f"x[0] (pos0): mean={x[0].mean():.4f}, range=[{x[0].min():.4f}, {x[0].max():.4f}]")
print(f"x[1] (pos1): mean={x[1].mean():.4f}, range=[{x[1].min():.4f}, {x[1].max():.4f}]")

# Check against prefill reference
x_int8_prefill = _int8_saturating_add(wte_q[[0]], wpe_q[[0]])
x_prefill = x_int8_prefill.astype(np.float32) * np.float32(x_scale)
print(f"x[0] matches prefill: {np.array_equal(x[0], x_prefill[0])}")

# Layer 0
L = 0
eps = np.float32(1e-6)
ln1_w = _to_f32(sd[f"transformer.h.{L}.ln_1.weight"])
ln1_b = _to_f32(sd[f"transformer.h.{L}.ln_1.bias"])
ln1 = _qdq(_layernorm_np(x, ln1_w, ln1_b, eps), _scale(s, f"block{L}_ln1"))
print(f"\nln1[1] (pos1): mean={ln1[1].mean():.4f}, std={ln1[1].std():.4f}")

# Check that each head's Q, K, V are correct for pos1
for H in range(n_head):
    q_w = _fq_linear(sd[f"transformer.h.{L}.attn.c_attn.weight_h{H}_query"])
    k_w = _fq_linear(sd[f"transformer.h.{L}.attn.c_attn.weight_h{H}_key"])
    v_w = _fq_linear(sd[f"transformer.h.{L}.attn.c_attn.weight_h{H}_value"])

    q = _qdq(ln1 @ q_w.T, _scale(s, f"block{L}_head{H}_query"))  # [2, d_head]
    k = _qdq(ln1 @ k_w.T, _scale(s, f"block{L}_head{H}_key"))    # [2, d_head]
    v = _qdq(ln1 @ v_w.T, _scale(s, f"block{L}_head{H}_value"))  # [2, d_head]

    # attn for position 1 (row 1)
    attn = (q @ k.T) * np.float32(0.125)  # [2, 2]
    probs_full = _causal_softmax(attn)  # [2, 2], row 1: [p01, p11]

    # Check that softmax is correct
    if H == 0:
        print(f"\n  H0 attn[1,0]={attn[1,0]:.4f}, attn[1,1]={attn[1,1]:.4f}")
        print(f"  H0 probs[1]={probs_full[1]}")

# Now try: what if reference uses seq=1 for pos1 independently?
print("\n=== Trying independent seq=1 reference for pos1 ===")
x1_int8 = _int8_saturating_add(wte_q[[greedy_tok]], wpe_q[[1]])
x1 = x1_int8.astype(np.float32) * np.float32(x_scale)
ln1_1 = _qdq(_layernorm_np(x1, ln1_w, ln1_b, eps), _scale(s, f"block{L}_ln1"))
print(f"Independent x1: mean={x1[0].mean():.4f}")
print(f"ln1 from seq=2 vs seq=1 match: {np.array_equal(ln1[1:2], ln1_1)}")
