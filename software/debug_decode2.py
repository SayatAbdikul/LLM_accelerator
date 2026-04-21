"""Detailed decode step 1 trace: find exact divergence point."""
import sys
import numpy as np
sys.path.insert(0, "/Users/sayat/Documents/GitHub/LLM_accelerator/software")

import torch
from taccel.runtime.calibration import build_calibration_scales
from taccel.runtime.tiny_fixture import build_stage3_tiny_decoder_bundle
from taccel.runtime.fake_quant_reference import _qdq, _layernorm_np, _scale, _int8_saturating_add, _fp32_to_int8, _to_f32, _fq_linear, _causal_softmax, _gelu_np
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
n_layer = int(model_args["n_layer"])

tiny = build_stage3_tiny_decoder_bundle(payload, smoke_decode_steps=4, calibration_scales=scales)
runner = HostRunner(tiny.build.bundle, logits_dtype=np.int8)
bundle = tiny.build.bundle

# Prefill
runner.run_prefill([0])
greedy_tok = 0  # Known from previous run

# Decode step 1
decode_logits = runner.run_decode_step(greedy_tok, 1)

# ---------------------------------------------------------------
# Now trace reference for decode step 1 (seq=[0, greedy_tok])
# ---------------------------------------------------------------
from taccel.quantizer.quantize import quantize_tensor

s = scales
eps = np.float32(1e-6)

# Print key/value calibration scales
for L in range(n_layer):
    for H in range(n_head):
        print(f"L{L}H{H}: key={s.get(f'block{L}_head{H}_key',None):.6f}  key_kv_load={s.get(f'block{L}_head{H}_key_kv_load',None):.6f}")
        print(f"       val={s.get(f'block{L}_head{H}_value',None):.6f}  val_kv_load={s.get(f'block{L}_head{H}_value_kv_load',None):.6f}")
        print(f"       attn_v={s.get(f'block{L}_head{H}_attn_v',None):.6f}  concat={s.get(f'block{L}_concat',None):.6f}")

wte_q, _ = quantize_tensor(_to_f32(sd["transformer.wte.weight"]), per_channel=False)
wpe_q, _ = quantize_tensor(_to_f32(sd["transformer.wpe.weight"]), per_channel=False)

tids = [0, greedy_tok]
pids = [0, 1]

x_int8 = _int8_saturating_add(wte_q[tids], wpe_q[pids])
x_scale = _scale(s, "tok_pos_add")
x = x_int8.astype(np.float32) * np.float32(x_scale)

L = 0
print(f"\n=== Layer {L} ===")
ln1_w = _to_f32(sd[f"transformer.h.{L}.ln_1.weight"])
ln1_b = _to_f32(sd[f"transformer.h.{L}.ln_1.bias"])
ln1 = _qdq(_layernorm_np(x, ln1_w, ln1_b, eps), _scale(s, f"block{L}_ln1"))
print(f"ln1[1] mean={ln1[1].mean():.5f} std={ln1[1].std():.5f}")

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
    print(f"  H{H}: attn[1]=[{attn[1,0]:.4f},{attn[1,1]:.4f}] probs[1]=[{probs[1,0]:.4f},{probs[1,1]:.4f}]")

    softmax_scale = _scale(s, f"block{L}_head{H}_softmax")
    probs_qdq = _qdq(probs, softmax_scale)
    head_out = _qdq(probs_qdq @ v, _scale(s, f"block{L}_head{H}_attn_v"))
    attn_v_scale = _scale(s, f"block{L}_head{H}_attn_v")
    ho_int8 = _fp32_to_int8(head_out, attn_v_scale)
    print(f"  H{H}: head_out[1]={head_out[1,:4]}  int8[1]={ho_int8[1,:4]}")
    head_outs_int8.append(ho_int8)

concat_int8 = np.concatenate(head_outs_int8, axis=-1)
concat_scale = _scale(s, f"block{L}_concat")
print(f"concat_int8[1,:8]={concat_int8[1,:8]}  concat_scale={concat_scale:.6f}")

c_proj_w = _fq_linear(sd[f"transformer.h.{L}.attn.c_proj.weight"])
c_proj_b = _to_f32(sd[f"transformer.h.{L}.attn.c_proj.bias"])
concat = concat_int8.astype(np.float32) * np.float32(concat_scale)
out_proj = _qdq(concat @ c_proj_w.T + c_proj_b, _scale(s, f"block{L}_out_proj"))
out_proj_scale = _scale(s, f"block{L}_out_proj")
out_proj_int8 = _fp32_to_int8(out_proj, out_proj_scale)
print(f"out_proj[1,:8]={out_proj[1,:8]}")
print(f"out_proj_int8[1,:8]={out_proj_int8[1,:8]}")

# Residual 1
x_int8 = _int8_saturating_add(x_int8, _fp32_to_int8(out_proj, out_proj_scale))
x_scale = _scale(s, f"block{L}_residual1")
x = x_int8.astype(np.float32) * np.float32(x_scale)
print(f"residual1[1,:8]={x_int8[1,:8]}  residual1_scale={x_scale:.6f}")

# MLP
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
fc2_int8 = _fp32_to_int8(fc2, fc2_scale)
print(f"fc2[1,:8]={fc2[1,:8]}  fc2_int8[1,:8]={fc2_int8[1,:8]}")

x_int8 = _int8_saturating_add(x_int8, fc2_int8)
x_scale = _scale(s, f"block{L}_residual2")
x = x_int8.astype(np.float32) * np.float32(x_scale)
print(f"residual2[1,:8]={x_int8[1,:8]}  residual2_scale={x_scale:.6f}")

print("\n=== Layer 1 input (residual2) is same as layer 1 ===")

L = 1
ln1_w = _to_f32(sd[f"transformer.h.{L}.ln_1.weight"])
ln1_b = _to_f32(sd[f"transformer.h.{L}.ln_1.bias"])
ln1 = _qdq(_layernorm_np(x, ln1_w, ln1_b, eps), _scale(s, f"block{L}_ln1"))
print(f"L1 ln1[1,:8]={_fp32_to_int8(ln1, _scale(s, f'block{L}_ln1'))[1,:8]}")

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

    softmax_scale = _scale(s, f"block{L}_head{H}_softmax")
    probs_qdq = _qdq(probs, softmax_scale)
    head_out = _qdq(probs_qdq @ v, _scale(s, f"block{L}_head{H}_attn_v"))
    attn_v_scale = _scale(s, f"block{L}_head{H}_attn_v")
    ho_int8 = _fp32_to_int8(head_out, attn_v_scale)
    head_outs_int8.append(ho_int8)

L = 1
concat_int8 = np.concatenate(head_outs_int8, axis=-1)
concat_scale = _scale(s, f"block{L}_concat")
c_proj_w = _fq_linear(sd[f"transformer.h.{L}.attn.c_proj.weight"])
c_proj_b = _to_f32(sd[f"transformer.h.{L}.attn.c_proj.bias"])
concat = concat_int8.astype(np.float32) * np.float32(concat_scale)
out_proj = _qdq(concat @ c_proj_w.T + c_proj_b, _scale(s, f"block{L}_out_proj"))
out_proj_scale = _scale(s, f"block{L}_out_proj")
out_proj_int8 = _fp32_to_int8(out_proj, out_proj_scale)

x_int8 = _int8_saturating_add(x_int8, out_proj_int8)
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

print(f"\ndecode_logits (first 10): {decode_logits[:10]}")
print(f"ref_logits    (first 10): {ref_logits[:10]}")
cos = cosine_similarity(decode_logits[:vocab].astype(np.float32), ref_logits[:vocab].astype(np.float32))
print(f"Cosine: {cos:.4f}")
print(f"Max diff: {np.abs(decode_logits[:vocab].astype(np.int32) - ref_logits[:vocab].astype(np.int32)).max()}")
