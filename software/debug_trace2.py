"""Compare golden model trace events with reference for decode step 1."""
import sys
import numpy as np
sys.path.insert(0, "/Users/sayat/Documents/GitHub/LLM_accelerator/software")

import torch
from taccel.runtime.calibration import build_calibration_scales
from taccel.runtime.tiny_fixture import build_stage3_tiny_decoder_bundle
from taccel.runtime.host_runner import HostRunner
from taccel.runtime.fake_quant_reference import (
    _qdq, _layernorm_np, _scale, _int8_saturating_add, _fp32_to_int8,
    _to_f32, _fq_linear, _causal_softmax, _gelu_np
)
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

# ============================================================
# Run prefill to check K/V cache
# ============================================================
runner.run_prefill([0])
greedy_tok = 0

# ============================================================
# Run decode step to get intermediate trace events
# ============================================================
# The HostRunner may have trace events available
decode_logits = runner.run_decode_step(greedy_tok, 1)

# Access the decode codegen trace manifest
dcg = tiny.build.decode_codegen
print(f"Decode trace manifest has {len(dcg.trace_manifest)} entries")

# Find trace events by node name
trace_by_name = {}
for pc, events in dcg.trace_manifest.items():
    for ev in events:
        name = ev.get("name", "")
        if name not in trace_by_name:
            trace_by_name[name] = []
        trace_by_name[name].append((pc, ev))

print("Traced nodes:", [k for k in sorted(trace_by_name.keys()) if not k.endswith("__")])

# ============================================================
# Now read actual intermediate values from simulator DRAM
# ============================================================
# We need to re-run the decode with trace capture
# Let's use the simulator's trace_manifest to read values

# Actually, let's just use the simulator directly to read DRAM
# at the trace event memory locations after decode step
sim = runner.simulator

# Read DRAM state
dram = sim.state.dram

def read_trace_event_value(event, dram):
    """Read INT8 or INT32 tile from DRAM based on trace event."""
    import struct
    buf = event.get("buffer", "abuf")
    offset = event.get("offset_units", 0) * 16  # 16 bytes per unit
    rows = event.get("logical_rows", 1)
    cols = event.get("logical_cols", 1)
    dtype_str = event.get("dtype", "int8")

    if dtype_str == "int8":
        data = bytes(dram[offset : offset + rows * cols])
        return np.frombuffer(data, dtype=np.int8).reshape(rows, cols)
    else:
        data = bytes(dram[offset : offset + rows * cols * 4])
        return np.frombuffer(data, dtype=np.int32).reshape(rows, cols)

# ============================================================
# Reference computation for decode step 1
# ============================================================
s = scales
eps = np.float32(1e-6)

wte_q, _ = quantize_tensor(_to_f32(sd["transformer.wte.weight"]), per_channel=False)
wpe_q, _ = quantize_tensor(_to_f32(sd["transformer.wpe.weight"]), per_channel=False)

tids = [0, greedy_tok]
pids = [0, 1]

x_int8 = _int8_saturating_add(wte_q[tids], wpe_q[pids])
x_scale = _scale(s, "tok_pos_add")
x = x_int8.astype(np.float32) * np.float32(x_scale)

L = 0
ln1_w = _to_f32(sd[f"transformer.h.{L}.ln_1.weight"])
ln1_b = _to_f32(sd[f"transformer.h.{L}.ln_1.bias"])
ln1 = _qdq(_layernorm_np(x, ln1_w, ln1_b, eps), _scale(s, f"block{L}_ln1"))
ln1_int8 = _fp32_to_int8(ln1, _scale(s, f"block{L}_ln1"))

print(f"\n=== Layer 0 reference values (position 1) ===")
print(f"ln1[1] int8: {ln1_int8[1, :8]}")

# Q, K, V for each head
head_out_int8_ref = []
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
    probs_int8 = _fp32_to_int8(probs, _scale(s, f"block{L}_head{H}_softmax"))

    head_out = _qdq(probs_qdq @ v, _scale(s, f"block{L}_head{H}_attn_v"))
    head_out_int8 = _fp32_to_int8(head_out, _scale(s, f"block{L}_head{H}_attn_v"))
    head_out_int8_ref.append(head_out_int8)

    if H == 0:
        q_int8 = _fp32_to_int8(q, _scale(s, f"block{L}_head{H}_query"))
        v_int8 = _fp32_to_int8(v, _scale(s, f"block{L}_head{H}_value"))
        print(f"  H{H} Q[1]={q_int8[1,:4]}  probs_int8[1]={probs_int8[1]}  V[1]={v_int8[1,:4]}")
        print(f"  H{H} head_out[1]={head_out_int8[1,:4]}")
        print(f"  V[0]={v_int8[0,:4]}")

# Now compare with golden model via KV cache
kv_layout = tiny.build.kv_layout
print("\n=== KV cache check (layer 0, head 0) ===")
entry_V0 = kv_layout.entry(0, "value", 0)
kv_cache_base = int(bundle.kv_cache_base)
kv_step_bytes = int(bundle.kv_step_bytes)
v0_offset = kv_cache_base + entry_V0.byte_offset + 0 * kv_step_bytes
v0_from_cache = np.frombuffer(bytes(dram[v0_offset : v0_offset + d_head]), dtype=np.int8)
print(f"V0 from KV cache (first 4): {v0_from_cache[:4]}")
v_int8_ref = _fp32_to_int8(
    _qdq(ln1 @ _fq_linear(sd["transformer.h.0.attn.c_attn.weight_h0_value"]).T, _scale(s, "block0_head0_value")),
    _scale(s, "block0_head0_value")
)
print(f"V0 from reference (first 4): {v_int8_ref[0,:4]}")
print(f"V0 match: {np.array_equal(v0_from_cache, v_int8_ref[0])}")

# Also check V1 (stored during decode)
v1_offset = kv_cache_base + entry_V0.byte_offset + 1 * kv_step_bytes
v1_from_cache = np.frombuffer(bytes(dram[v1_offset : v1_offset + d_head]), dtype=np.int8)
print(f"V1 from KV cache (first 4): {v1_from_cache[:4]}")
print(f"V1 from reference (first 4): {v_int8_ref[1,:4]}")
print(f"V1 match: {np.array_equal(v1_from_cache, v_int8_ref[1])}")
