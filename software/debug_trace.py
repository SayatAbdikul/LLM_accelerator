"""Layer-by-layer trace comparison between golden model and fake-quant reference."""
import sys, numpy as np
sys.path.insert(0, "/Users/sayat/Documents/GitHub/LLM_accelerator/software")

import torch
from taccel.runtime.calibration import build_calibration_scales
from taccel.runtime.tiny_fixture import (
    build_stage3_tiny_decoder_bundle,
    run_tiny_decode_trace,
    quantize_fixture_payload,
)
from taccel.runtime.fake_quant_reference import NanoGPTFQReference, _fp32_forward, _qdq, _fq_linear, _scale, _layernorm_np, _to_f32
from taccel.runtime.host_runner import HostRunner

FIXTURE = "/Users/sayat/Documents/GitHub/LLM_accelerator/software/tests/fixtures/generated/nanogpt_shakespeare_char_d128_l2.pt"

payload = torch.load(FIXTURE, map_location="cpu")
scales = build_calibration_scales(payload)

print("=== Calibration scales ===")
for k, v in sorted(scales.items()):
    print(f"  {k}: {v:.6f}")

# Run golden model (single token step)
tiny = build_stage3_tiny_decoder_bundle(payload, smoke_decode_steps=4, calibration_scales=scales)
runner = HostRunner(tiny.build.bundle, logits_dtype=np.int8)
golden_logits = runner.run_prefill([0])
print("\n=== Golden model prefill logits (first 10) ===")
print(golden_logits[:10])

# Build reference and run forward
ref = NanoGPTFQReference(
    state_dict=payload["state_dict"],
    model_args=payload["model_args"],
    scales=scales,
)
ref_logits = ref.forward([0])
print("\n=== Reference logits (first 10) ===")
print(ref_logits[:10])

vocab = tiny.config.vocab_size
diff = golden_logits[:vocab].astype(np.int32) - ref_logits[:vocab].astype(np.int32)
print(f"\nMax |diff|: {np.abs(diff).max()}")
print(f"Mean |diff|: {np.abs(diff).mean():.2f}")

# Now trace internally in the reference
sd = payload["state_dict"]
model_args = payload["model_args"]
s = scales

n_layer = int(model_args["n_layer"])
n_head = int(model_args["n_head"])
d_model = int(model_args["n_embd"])
d_head = d_model // n_head
vocab_size = int(model_args["vocab_size"])
EPS = 1e-5  # what reference uses

from taccel.quantizer.quantize import quantize_tensor

def fq_embedding(tensor):
    arr = _to_f32(tensor)
    q, sc = quantize_tensor(arr, per_channel=False)
    from taccel.quantizer.quantize import dequantize_tensor
    try:
        return dequantize_tensor(q, sc).astype(np.float32)
    except:
        return q.astype(np.float32) * float(sc[0])

def fq_linear(tensor):
    arr = _to_f32(tensor)
    q, sc = quantize_tensor(arr, per_channel=True)
    mean_sc = np.float32(np.mean(sc.astype(np.float32)))
    return q.astype(np.float32) * mean_sc

def qdq(x, name, default=6.0/127.0):
    sc = _scale(s, name, default)
    return _qdq(x, sc)

# Forward
tids = [0]
pids = [0]
wte = fq_embedding(sd["transformer.wte.weight"])
wpe = fq_embedding(sd["transformer.wpe.weight"])
tok_e = qdq(wte[tids], "tok_embed")
pos_e = qdq(wpe[pids], "pos_embed")
x = qdq(tok_e + pos_e, "tok_pos_add")

print(f"\n=== x after embed add (block0 input) ===")
print(f"  mean={x.mean():.4f}, std={x.std():.4f}, range=[{x.min():.4f}, {x.max():.4f}]")

for L in range(n_layer):
    ln1_w = _to_f32(sd[f"transformer.h.{L}.ln_1.weight"])
    ln1_b = _to_f32(sd[f"transformer.h.{L}.ln_1.bias"])
    ln1_raw = _layernorm_np(x, ln1_w, ln1_b, EPS)
    ln1 = qdq(ln1_raw, f"block{L}_ln1")
    print(f"\n  Layer {L} ln1: mean={ln1.mean():.4f}, std={ln1.std():.4f}")

    head_outs = []
    for H in range(n_head):
        q_w = fq_linear(sd[f"transformer.h.{L}.attn.c_attn.weight_h{H}_query"])
        k_w = fq_linear(sd[f"transformer.h.{L}.attn.c_attn.weight_h{H}_key"])
        v_w = fq_linear(sd[f"transformer.h.{L}.attn.c_attn.weight_h{H}_value"])
        q_raw = ln1 @ q_w.T
        k_raw = ln1 @ k_w.T
        v_raw = ln1 @ v_w.T
        q = qdq(q_raw, f"block{L}_head{H}_query")
        k = qdq(k_raw, f"block{L}_head{H}_key")
        v = qdq(v_raw, f"block{L}_head{H}_value")
        # seq=1, attn is trivially [[1.0]]
        probs = np.ones((1, 1), dtype=np.float32)
        probs_q = qdq(probs, f"block{L}_head{H}_softmax", 1.0/127.0)
        head_out = qdq(probs_q @ v, f"block{L}_head{H}_attn_v")
        head_outs.append(head_out)
        if H == 0:
            print(f"    H0 q={q.ravel()[:4]}, v={v.ravel()[:4]}, head_out={head_out.ravel()[:4]}")

    concat = np.concatenate(head_outs, axis=-1)
    concat_q = qdq(concat, f"block{L}_concat")
    c_proj_w = fq_linear(sd[f"transformer.h.{L}.attn.c_proj.weight"])
    c_proj_b = _to_f32(sd[f"transformer.h.{L}.attn.c_proj.bias"])
    out_proj = qdq(concat_q @ c_proj_w.T + c_proj_b, f"block{L}_out_proj")
    print(f"  Layer {L} out_proj: mean={out_proj.mean():.4f}, std={out_proj.std():.4f}")

    x = qdq(x + out_proj, f"block{L}_residual1")

    ln2_w = _to_f32(sd[f"transformer.h.{L}.ln_2.weight"])
    ln2_b = _to_f32(sd[f"transformer.h.{L}.ln_2.bias"])
    ln2_raw = _layernorm_np(x, ln2_w, ln2_b, EPS)
    ln2 = qdq(ln2_raw, f"block{L}_ln2")

    fc_w = fq_linear(sd[f"transformer.h.{L}.mlp.c_fc.weight"])
    fc_b = _to_f32(sd[f"transformer.h.{L}.mlp.c_fc.bias"])
    fc1 = qdq(ln2 @ fc_w.T + fc_b, f"block{L}_fc1")

    from taccel.runtime.fake_quant_reference import _gelu_np
    gelu = qdq(_gelu_np(fc1), f"block{L}_gelu", 1.0/127.0)

    proj_w = fq_linear(sd[f"transformer.h.{L}.mlp.c_proj.weight"])
    proj_b = _to_f32(sd[f"transformer.h.{L}.mlp.c_proj.bias"])
    fc2 = qdq(gelu @ proj_w.T + proj_b, f"block{L}_fc2")
    x = qdq(x + fc2, f"block{L}_residual2")
    print(f"  Layer {L} residual2: mean={x.mean():.4f}, std={x.std():.4f}")

ln_f_w = _to_f32(sd["transformer.ln_f.weight"])
ln_f_b = _to_f32(sd["transformer.ln_f.bias"])
ln_f = qdq(_layernorm_np(x, ln_f_w, ln_f_b, EPS), "ln_f")
lm_head_w = fq_linear(sd["lm_head.weight"])
logits_raw = ln_f[-1:] @ lm_head_w.T
logits_trace = np.clip(np.round(logits_raw[0] / _scale(s, "lm_head")), -128, 127).astype(np.int8)

print(f"\n=== Traced logits (first 10) ===")
print(logits_trace[:10])
print(f"=== Reference logits (first 10) ===")
print(ref_logits[:10])
print(f"Same: {np.array_equal(logits_trace[:vocab], ref_logits[:vocab])}")

print(f"\n=== Golden vs traced diff (first 20) ===")
d = golden_logits[:20].astype(np.int32) - logits_trace[:20].astype(np.int32)
print(f"golden: {golden_logits[:20]}")
print(f"traced: {logits_trace[:20]}")
print(f"diff:   {d}")
