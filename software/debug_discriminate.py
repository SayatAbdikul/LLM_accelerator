"""Discriminating test: seq=2 prefill vs decode step 1.

If golden_prefill([0,0])[-1] == golden_decode(step=1): both golden paths agree.
If they both disagree with reference at 0.82 cosine: reference has the bug.
If they don't match: decode codegen has a bug.
"""
import sys
import numpy as np
sys.path.insert(0, "/Users/sayat/Documents/GitHub/LLM_accelerator/software")

import torch
from taccel.runtime.calibration import build_calibration_scales
from taccel.runtime.tiny_fixture import build_stage3_tiny_decoder_bundle
from taccel.runtime.host_runner import HostRunner
from taccel.runtime.fake_quant_reference import (
    NanoGPTFQReference, _to_f32, _layernorm_np, _qdq, _scale,
    _int8_saturating_add, _fp32_to_int8, _fq_linear, _causal_softmax, _gelu_np
)
from taccel.quantizer.quantize import quantize_tensor
from taccel.runtime.fake_quant import cosine_similarity

FIXTURE = "/Users/sayat/Documents/GitHub/LLM_accelerator/software/tests/fixtures/generated/nanogpt_shakespeare_char_d128_l2.pt"

payload = torch.load(FIXTURE, map_location="cpu")
scales = build_calibration_scales(payload)
sd = payload["state_dict"]
model_args = payload["model_args"]
vocab = int(model_args["vocab_size"])
n_head = int(model_args["n_head"])
n_embd = int(model_args["n_embd"])
d_head = n_embd // n_head
n_layer = int(model_args["n_layer"])

# ============================================================
# Path A: HostRunner decode step 1 logits
# ============================================================
tiny = build_stage3_tiny_decoder_bundle(payload, smoke_decode_steps=4, calibration_scales=scales)
runner = HostRunner(tiny.build.bundle, logits_dtype=np.int8)
bundle = tiny.build.bundle
runner.run_prefill([0])
decode_logits = runner.run_decode_step(0, 1)  # greedy_tok=0, step=1
print(f"Decode step 1 logits[:10]: {decode_logits[:10]}")

# ============================================================
# Path B: seq=2 prefill-only bundle, last position logits
# ============================================================
# Build a separate prefill bundle with sequence [0, 0]
from taccel.runtime.tiny_fixture import build_stage3_tiny_decoder_bundle
tiny2 = build_stage3_tiny_decoder_bundle(payload, smoke_decode_steps=0, calibration_scales=scales)
runner2 = HostRunner(tiny2.build.bundle, logits_dtype=np.int8)
prefill2_logits = runner2.run_prefill([0, 0])  # seq=[0, 0], last position
print(f"Prefill seq=[0,0] last pos logits[:10]: {prefill2_logits[:10]}")

cos_AB = cosine_similarity(decode_logits[:vocab].astype(np.float32), prefill2_logits[:vocab].astype(np.float32))
print(f"\nCosine(decode_step1, prefill[0,0][-1]): {cos_AB:.4f}")
print(f"Max diff: {np.abs(decode_logits[:vocab].astype(np.int32) - prefill2_logits[:vocab].astype(np.int32)).max()}")

# ============================================================
# Path C: numpy fake-quant reference, seq=[0, 0]
# ============================================================
s = scales
eps = np.float32(1e-6)

wte_q, _ = quantize_tensor(_to_f32(sd["transformer.wte.weight"]), per_channel=False)
wpe_q, _ = quantize_tensor(_to_f32(sd["transformer.wpe.weight"]), per_channel=False)

tids = [0, 0]
pids = [0, 1]

x_int8 = _int8_saturating_add(wte_q[tids], wpe_q[pids])
x_scale = _scale(s, "tok_pos_add")
x = x_int8.astype(np.float32) * np.float32(x_scale)

for L in range(n_layer):
    ln1_w = _to_f32(sd[f"transformer.h.{L}.ln_1.weight"])
    ln1_b = _to_f32(sd[f"transformer.h.{L}.ln_1.bias"])
    ln1 = _qdq(_layernorm_np(x, ln1_w, ln1_b, eps), _scale(s, f"block{L}_ln1"))

    head_outs_int8 = []
    for H in range(n_head):
        q_w = _fq_linear(sd[f"transformer.h.{L}.attn.c_attn.weight_h{H}_query"])
        k_w = _fq_linear(sd[f"transformer.h.{L}.attn.c_attn.weight_h{H}_key"])
        v_w = _fq_linear(sd[f"transformer.h.{L}.attn.c_attn.weight_h{H}_value"])

        q = _qdq(ln1 @ q_w.T, _scale(s, f"block{L}_head{H}_query"))
        k = _qdq(ln1 @ k_w.T, _scale(s, f"block{L}_head{H}_key"))
        v = _qdq(ln1 @ v_w.T, _scale(s, f"block{L}_head{H}_value"))

        attn = (q @ k.T) * np.float32(0.125)
        probs = _causal_softmax(attn)
        probs_qdq = _qdq(probs, _scale(s, f"block{L}_head{H}_softmax"))
        head_out = _qdq(probs_qdq @ v, _scale(s, f"block{L}_head{H}_attn_v"))
        head_outs_int8.append(_fp32_to_int8(head_out, _scale(s, f"block{L}_head{H}_attn_v")))

    concat_int8 = np.concatenate(head_outs_int8, axis=-1)
    concat_scale = _scale(s, f"block{L}_concat")
    c_proj_w = _fq_linear(sd[f"transformer.h.{L}.attn.c_proj.weight"])
    c_proj_b = _to_f32(sd[f"transformer.h.{L}.attn.c_proj.bias"])
    concat = concat_int8.astype(np.float32) * np.float32(concat_scale)
    out_proj = _qdq(concat @ c_proj_w.T + c_proj_b, _scale(s, f"block{L}_out_proj"))
    out_proj_scale = _scale(s, f"block{L}_out_proj")

    x_int8 = _int8_saturating_add(x_int8, _fp32_to_int8(out_proj, out_proj_scale))
    x_scale = _scale(s, f"block{L}_residual1")
    x = x_int8.astype(np.float32) * np.float32(x_scale)

    ln2_w = _to_f32(sd[f"transformer.h.{L}.ln_2.weight"])
    ln2_b = _to_f32(sd[f"transformer.h.{L}.ln_2.bias"])
    ln2 = _qdq(_layernorm_np(x, ln2_w, ln2_b, eps), _scale(s, f"block{L}_ln2"))
    fc_w = _fq_linear(sd[f"transformer.h.{L}.mlp.c_fc.weight"])
    fc_b = _to_f32(sd[f"transformer.h.{L}.mlp.c_fc.bias"])
    fc1 = _qdq(ln2 @ fc_w.T + fc_b, _scale(s, f"block{L}_fc1"))
    gelu = _qdq(_gelu_np(fc1), _scale(s, f"block{L}_gelu", 1.0/127.0))
    proj_w = _fq_linear(sd[f"transformer.h.{L}.mlp.c_proj.weight"])
    proj_b = _to_f32(sd[f"transformer.h.{L}.mlp.c_proj.bias"])
    fc2 = _qdq(gelu @ proj_w.T + proj_b, _scale(s, f"block{L}_fc2"))
    fc2_scale = _scale(s, f"block{L}_fc2")

    x_int8 = _int8_saturating_add(x_int8, _fp32_to_int8(fc2, fc2_scale))
    x_scale = _scale(s, f"block{L}_residual2")
    x = x_int8.astype(np.float32) * np.float32(x_scale)

ln_f_w = _to_f32(sd["transformer.ln_f.weight"])
ln_f_b = _to_f32(sd["transformer.ln_f.bias"])
ln_f = _qdq(_layernorm_np(x, ln_f_w, ln_f_b, eps), _scale(s, "ln_f"))
lm_w = _fq_linear(sd["lm_head.weight"])
logits_fp32 = (ln_f[-1:] @ lm_w.T)[0]
lm_head_scale = _scale(s, "lm_head")
ref_logits = np.clip(np.round(logits_fp32 / np.float32(lm_head_scale)), -128, 127).astype(np.int8)

print(f"\nRef logits (seq=[0,0])[:10]: {ref_logits[:10]}")
cos_AC = cosine_similarity(decode_logits[:vocab].astype(np.float32), ref_logits[:vocab].astype(np.float32))
cos_BC = cosine_similarity(prefill2_logits[:vocab].astype(np.float32), ref_logits[:vocab].astype(np.float32))
print(f"Cosine(decode_step1, ref): {cos_AC:.4f}")
print(f"Cosine(prefill[0,0][-1], ref): {cos_BC:.4f}")

# ============================================================
# Also check layer 1 KV cache against reference
# ============================================================
print("\n=== Layer 1 KV cache check (head 0) ===")
sim = runner.simulator
dram = sim.state.dram
kv_layout = tiny.build.kv_layout
kv_cache_base = int(bundle.kv_cache_base)
kv_step_bytes = int(bundle.kv_step_bytes)

# Recompute reference for [0, greedy_tok=0] with layer 1 KV
tids2 = [0, 0]
pids2 = [0, 1]
x_int8_2 = _int8_saturating_add(wte_q[tids2], wpe_q[pids2])
x_scale_2 = _scale(s, "tok_pos_add")
x2 = x_int8_2.astype(np.float32) * np.float32(x_scale_2)

L = 0
ln1_w = _to_f32(sd[f"transformer.h.{L}.ln_1.weight"])
ln1_b = _to_f32(sd[f"transformer.h.{L}.ln_1.bias"])
ln1 = _qdq(_layernorm_np(x2, ln1_w, ln1_b, eps), _scale(s, f"block{L}_ln1"))
head_outs_int8 = []
for H in range(n_head):
    q_w = _fq_linear(sd[f"transformer.h.{L}.attn.c_attn.weight_h{H}_query"])
    k_w = _fq_linear(sd[f"transformer.h.{L}.attn.c_attn.weight_h{H}_key"])
    v_w = _fq_linear(sd[f"transformer.h.{L}.attn.c_attn.weight_h{H}_value"])
    q = _qdq(ln1 @ q_w.T, _scale(s, f"block{L}_head{H}_query"))
    k = _qdq(ln1 @ k_w.T, _scale(s, f"block{L}_head{H}_key"))
    v = _qdq(ln1 @ v_w.T, _scale(s, f"block{L}_head{H}_value"))
    attn = (q @ k.T) * np.float32(0.125)
    probs = _causal_softmax(attn)
    probs_qdq = _qdq(probs, _scale(s, f"block{L}_head{H}_softmax"))
    head_out = _qdq(probs_qdq @ v, _scale(s, f"block{L}_head{H}_attn_v"))
    head_outs_int8.append(_fp32_to_int8(head_out, _scale(s, f"block{L}_head{H}_attn_v")))

concat_int8 = np.concatenate(head_outs_int8, axis=-1)
concat_scale = _scale(s, f"block{L}_concat")
c_proj_w = _fq_linear(sd[f"transformer.h.{L}.attn.c_proj.weight"])
c_proj_b = _to_f32(sd[f"transformer.h.{L}.attn.c_proj.bias"])
concat = concat_int8.astype(np.float32) * np.float32(concat_scale)
out_proj = _qdq(concat @ c_proj_w.T + c_proj_b, _scale(s, f"block{L}_out_proj"))
out_proj_scale = _scale(s, f"block{L}_out_proj")
x_int8_2 = _int8_saturating_add(x_int8_2, _fp32_to_int8(out_proj, out_proj_scale))
x_scale_2 = _scale(s, f"block{L}_residual1")
x2 = x_int8_2.astype(np.float32) * np.float32(x_scale_2)
ln2_w = _to_f32(sd[f"transformer.h.{L}.ln_2.weight"])
ln2_b = _to_f32(sd[f"transformer.h.{L}.ln_2.bias"])
ln2 = _qdq(_layernorm_np(x2, ln2_w, ln2_b, eps), _scale(s, f"block{L}_ln2"))
fc_w = _fq_linear(sd[f"transformer.h.{L}.mlp.c_fc.weight"])
fc_b = _to_f32(sd[f"transformer.h.{L}.mlp.c_fc.bias"])
fc1 = _qdq(ln2 @ fc_w.T + fc_b, _scale(s, f"block{L}_fc1"))
gelu = _qdq(_gelu_np(fc1), _scale(s, f"block{L}_gelu", 1.0/127.0))
proj_w = _fq_linear(sd[f"transformer.h.{L}.mlp.c_proj.weight"])
proj_b = _to_f32(sd[f"transformer.h.{L}.mlp.c_proj.bias"])
fc2 = _qdq(gelu @ proj_w.T + proj_b, _scale(s, f"block{L}_fc2"))
fc2_scale = _scale(s, f"block{L}_fc2")
x_int8_2 = _int8_saturating_add(x_int8_2, _fp32_to_int8(fc2, fc2_scale))
x_scale_2 = _scale(s, f"block{L}_residual2")
x2 = x_int8_2.astype(np.float32) * np.float32(x_scale_2)

# Layer 1 KV
L = 1
ln1_w = _to_f32(sd[f"transformer.h.{L}.ln_1.weight"])
ln1_b = _to_f32(sd[f"transformer.h.{L}.ln_1.bias"])
ln1 = _qdq(_layernorm_np(x2, ln1_w, ln1_b, eps), _scale(s, f"block{L}_ln1"))

for H in range(1):  # just head 0
    k_w = _fq_linear(sd[f"transformer.h.{L}.attn.c_attn.weight_h{H}_key"])
    v_w = _fq_linear(sd[f"transformer.h.{L}.attn.c_attn.weight_h{H}_value"])
    k_ref = _fp32_to_int8(_qdq(ln1 @ k_w.T, _scale(s, f"block{L}_head{H}_key")), _scale(s, f"block{L}_head{H}_key"))
    v_ref = _fp32_to_int8(_qdq(ln1 @ v_w.T, _scale(s, f"block{L}_head{H}_value")), _scale(s, f"block{L}_head{H}_value"))

    entry_K1 = kv_layout.entry(L, "key", H)
    entry_V1 = kv_layout.entry(L, "value", H)

    # Position 0
    k1p0_offset = kv_cache_base + entry_K1.byte_offset + 0 * kv_step_bytes
    k1p0_cache = np.frombuffer(bytes(dram[k1p0_offset : k1p0_offset + d_head]), dtype=np.int8)
    v1p0_offset = kv_cache_base + entry_V1.byte_offset + 0 * kv_step_bytes
    v1p0_cache = np.frombuffer(bytes(dram[v1p0_offset : v1p0_offset + d_head]), dtype=np.int8)

    # Position 1
    k1p1_offset = kv_cache_base + entry_K1.byte_offset + 1 * kv_step_bytes
    k1p1_cache = np.frombuffer(bytes(dram[k1p1_offset : k1p1_offset + d_head]), dtype=np.int8)
    v1p1_offset = kv_cache_base + entry_V1.byte_offset + 1 * kv_step_bytes
    v1p1_cache = np.frombuffer(bytes(dram[v1p1_offset : v1p1_offset + d_head]), dtype=np.int8)

    print(f"L1H{H} K0: cache={k1p0_cache[:4]}, ref={k_ref[0,:4]}, match={np.array_equal(k1p0_cache, k_ref[0])}")
    print(f"L1H{H} K1: cache={k1p1_cache[:4]}, ref={k_ref[1,:4]}, match={np.array_equal(k1p1_cache, k_ref[1])}")
    print(f"L1H{H} V0: cache={v1p0_cache[:4]}, ref={v_ref[0,:4]}, match={np.array_equal(v1p0_cache, v_ref[0])}")
    print(f"L1H{H} V1: cache={v1p1_cache[:4]}, ref={v_ref[1,:4]}, match={np.array_equal(v1p1_cache, v_ref[1])}")
