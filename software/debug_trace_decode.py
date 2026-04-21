"""Read actual intermediate values from decode step 1 via trace manifest."""
import sys
import numpy as np
sys.path.insert(0, "/Users/sayat/Documents/GitHub/LLM_accelerator/software")

import torch
from taccel.runtime.calibration import build_calibration_scales
from taccel.runtime.tiny_fixture import build_stage3_tiny_decoder_bundle
from taccel.runtime.host_runner import HostRunner
from taccel.runtime.fake_quant_reference import (
    _to_f32, _layernorm_np, _qdq, _scale, _int8_saturating_add,
    _fp32_to_int8, _fq_linear, _causal_softmax, _gelu_np
)
from taccel.quantizer.quantize import quantize_tensor

FIXTURE = "/Users/sayat/Documents/GitHub/LLM_accelerator/software/tests/fixtures/generated/nanogpt_shakespeare_char_d128_l2.pt"

payload = torch.load(FIXTURE, map_location="cpu")
scales = build_calibration_scales(payload)
sd = payload["state_dict"]
model_args = payload["model_args"]
n_head = int(model_args["n_head"])
d_head = int(model_args["n_embd"]) // n_head
n_layer = int(model_args["n_layer"])
eps = np.float32(1e-6)
s = scales

tiny = build_stage3_tiny_decoder_bundle(payload, smoke_decode_steps=4, calibration_scales=scales)
runner = HostRunner(tiny.build.bundle, logits_dtype=np.int8)
bundle = tiny.build.bundle

runner.run_prefill([0])
runner.run_decode_step(0, 1)

dcg = tiny.build.decode_codegen
dram = runner.simulator.state.dram

def read_trace(manifest, name, dram):
    for pc, events in manifest.items():
        for ev in events:
            if ev.get("name") == name:
                off = ev.get("offset_units", 0) * 16
                rows = ev.get("logical_rows", 1)
                cols = ev.get("logical_cols", 1)
                padded_rows = ev.get("rows", rows)
                padded_cols = ev.get("cols", cols)
                dtype = ev.get("dtype", "int8")
                total = padded_rows * padded_cols
                if dtype == "int8":
                    data = bytes(dram[off: off + total])
                    arr = np.frombuffer(data, dtype=np.int8).reshape(padded_rows, padded_cols)
                else:
                    data = bytes(dram[off: off + total * 4])
                    arr = np.frombuffer(data, dtype=np.int32).reshape(padded_rows, padded_cols)
                return arr[:rows, :cols], ev.get("scale", 1.0)
    return None, None

# ============================================================
# Reference: seq=[0,0]
# ============================================================
wte_q, _ = quantize_tensor(_to_f32(sd["transformer.wte.weight"]), per_channel=False)
wpe_q, _ = quantize_tensor(_to_f32(sd["transformer.wpe.weight"]), per_channel=False)
x_int8 = _int8_saturating_add(wte_q[[0, 0]], wpe_q[[0, 1]])
x = x_int8.astype(np.float32) * _scale(s, "tok_pos_add")

L = 0
ln1_w = _to_f32(sd[f"transformer.h.{L}.ln_1.weight"])
ln1_b = _to_f32(sd[f"transformer.h.{L}.ln_1.bias"])
ln1 = _qdq(_layernorm_np(x, ln1_w, ln1_b, eps), _scale(s, f"block{L}_ln1"))

H = 0
q_w = _fq_linear(sd[f"transformer.h.{L}.attn.c_attn.weight_h{H}_query"])
k_w = _fq_linear(sd[f"transformer.h.{L}.attn.c_attn.weight_h{H}_key"])
v_w = _fq_linear(sd[f"transformer.h.{L}.attn.c_attn.weight_h{H}_value"])
q = _qdq(ln1 @ q_w.T, _scale(s, f"block{L}_head{H}_query"))
k = _qdq(ln1 @ k_w.T, _scale(s, f"block{L}_head{H}_key"))
v = _qdq(ln1 @ v_w.T, _scale(s, f"block{L}_head{H}_value"))
q_int8 = _fp32_to_int8(q, _scale(s, f"block{L}_head{H}_query"))
k_int8 = _fp32_to_int8(k, _scale(s, f"block{L}_head{H}_key"))
v_int8 = _fp32_to_int8(v, _scale(s, f"block{L}_head{H}_value"))
attn = (q @ k.T) * 0.125
probs = _causal_softmax(attn)
head_out = _qdq(_qdq(probs, _scale(s, f"block{L}_head{H}_softmax")) @ v, _scale(s, f"block{L}_head{H}_attn_v"))
head_out_int8 = _fp32_to_int8(head_out, _scale(s, f"block{L}_head{H}_attn_v"))

print(f"=== Reference L{L}H{H} ===")
print(f"  q_int8[1,:4]: {q_int8[1,:4]}")
print(f"  k_int8[0,:4]: {k_int8[0,:4]}, k_int8[1,:4]: {k_int8[1,:4]}")
print(f"  v_int8[0,:4]: {v_int8[0,:4]}, v_int8[1,:4]: {v_int8[1,:4]}")
print(f"  attn[1]: {attn[1]}")
print(f"  probs[1]: {probs[1]}")
print(f"  head_out_int8[1,:4]: {head_out_int8[1,:4]}")

# Read trace values from decode
print(f"\n=== Golden model decode trace L{L}H{H} ===")

# QKT ACCUM (raw int32 from MATMUL)
qkt_arr, qkt_scale = read_trace(dcg.trace_manifest, f"block{L}_head{H}_qkt", dram)
if qkt_arr is not None:
    print(f"  QKT int32[0,:2]: {qkt_arr[0,:2]} (scale={qkt_scale:.3e})")
    print(f"  QKT fp32[0,:2]: {qkt_arr[0,:2].astype(np.float32) * qkt_scale}")

# Softmax output
sm_arr, sm_scale = read_trace(dcg.trace_manifest, f"block{L}_head{H}_softmax", dram)
if sm_arr is not None:
    print(f"  Softmax int8[0,:2]: {sm_arr[0,:2]} (scale={sm_scale:.3e})")
    print(f"  Softmax fp32[0,:2]: {sm_arr[0,:2].astype(np.float32) * sm_scale}")

# AttnV output
av_arr, av_scale = read_trace(dcg.trace_manifest, f"block{L}_head{H}_attn_v", dram)
if av_arr is not None:
    print(f"  AttnV int8[0,:4]: {av_arr[0,:4]} (scale={av_scale:.3e})")

# Concat
cc_arr, cc_scale = read_trace(dcg.trace_manifest, f"block{L}_concat", dram)
if cc_arr is not None:
    print(f"\nConcat int8[0,:8]: {cc_arr[0,:8]} (scale={cc_scale:.3e})")
    # Reference
    ho_list = []
    for H2 in range(n_head):
        q_w2 = _fq_linear(sd[f"transformer.h.{L}.attn.c_attn.weight_h{H2}_query"])
        k_w2 = _fq_linear(sd[f"transformer.h.{L}.attn.c_attn.weight_h{H2}_key"])
        v_w2 = _fq_linear(sd[f"transformer.h.{L}.attn.c_attn.weight_h{H2}_value"])
        q2 = _qdq(ln1 @ q_w2.T, _scale(s, f"block{L}_head{H2}_query"))
        k2 = _qdq(ln1 @ k_w2.T, _scale(s, f"block{L}_head{H2}_key"))
        v2 = _qdq(ln1 @ v_w2.T, _scale(s, f"block{L}_head{H2}_value"))
        probs2 = _causal_softmax((q2 @ k2.T) * 0.125)
        ho2 = _qdq(_qdq(probs2, _scale(s, f"block{L}_head{H2}_softmax")) @ v2, _scale(s, f"block{L}_head{H2}_attn_v"))
        ho_list.append(_fp32_to_int8(ho2, _scale(s, f"block{L}_head{H2}_attn_v")))
    ref_concat = np.concatenate(ho_list, axis=-1)
    print(f"Ref concat int8[1,:8]: {ref_concat[1,:8]}")
    print(f"Match: {np.array_equal(cc_arr[0], ref_concat[1])}")

# Out_proj
op_arr, op_scale = read_trace(dcg.trace_manifest, f"block{L}_out_proj", dram)
if op_arr is not None:
    print(f"\nOut_proj int8[0,:4]: {op_arr[0,:4]} (scale={op_scale:.3e})")
