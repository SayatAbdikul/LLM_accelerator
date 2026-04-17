#!/usr/bin/env python3
"""Phase A diagnostics: identify where FP32 vs INT8 accuracy is lost.

Collects:
  A1. Per-block residual output ranges vs. constant block_input_scale (H1)
  A3. Per-head QKT ranges vs. hardcoded 6.0/127.0 (H5)
  VADD saturation % per residual add in the golden model (H4)

Usage:
    python3 diagnose_accuracy.py [--max-images 5]
"""
import argparse
import collections
import json
import math
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compare_golden import (
    DIAGNOSTIC_PRESETS,
    FROZEN_EVAL_IMAGE_IDS,
    LOCAL_FROZEN_IMAGE_DIR,
    MODEL_NAME,
    collect_images,
    compile_model,
    fp32_trace,
    get_diagnostic_preset,
    golden_inference,
    load_local_images,
    load_model,
    patch_embed_int8,
    preset_compile_kwargs,
    tensor_error_metrics,
)
from taccel.compiler.tiler import pad_dim
from taccel.golden_model import MachineState, Simulator
from taccel.golden_model import memory as mem_module
from taccel.isa.opcodes import BUF_ABUF
from transformers import AutoImageProcessor

DEPTH = 12
NUM_HEADS = 3
HEAD_DIM = 64


# ---------------------------------------------------------------------------
# A1 + A3: FP32 hooks
# ---------------------------------------------------------------------------

def _reshape_heads(tensor, module):
    """Reshape [B, S, D] -> [B, H, S, Dh] without relying on HF internals."""
    bsz, seq_len, _ = tensor.shape
    nh = module.num_attention_heads
    hd = module.attention_head_size
    return tensor.view(bsz, seq_len, nh, hd).permute(0, 2, 1, 3)


def collect_fp32_stats(model, processor, images):
    """Register hooks to measure per-block residual ranges and per-head QKT ranges."""
    residual1_maxabs = collections.defaultdict(list)  # block_idx -> [max_abs per image]
    residual2_maxabs = collections.defaultdict(list)  # block_idx -> [max_abs per image]
    qkt_maxabs = collections.defaultdict(list)        # (block, head) -> [max_abs per image]

    handles = []

    # --- A1: residual1 output range ---
    # The residual1 output is the input to layernorm_after (pre-MLP residual).
    for i in range(DEPTH):
        def _make_res1(block_idx):
            def hook(module, inputs):
                x = inputs[0]
                residual1_maxabs[block_idx].append(float(x.abs().max().item()))
            return hook
        h = model.vit.encoder.layer[i].layernorm_after.register_forward_pre_hook(_make_res1(i))
        handles.append(h)

    # --- A1: residual2 output range ---
    # The ViTLayer output is the residual2 output (after the MLP residual add).
    for i in range(DEPTH):
        def _make_res2(block_idx):
            def hook(module, inputs, output):
                x = output[0] if isinstance(output, tuple) else output
                residual2_maxabs[block_idx].append(float(x.abs().max().item()))
            return hook
        h = model.vit.encoder.layer[i].register_forward_hook(_make_res2(i))
        handles.append(h)

    # --- A3: QKT range per head ---
    # Hook ViTSelfAttention, recompute Q@K^T/sqrt(d) from the module's projections.
    # This runs Q/K projections twice per image (diagnostics only).
    for i in range(DEPTH):
        def _make_qkt(block_idx):
            def hook(module, inputs, output):
                hidden = inputs[0]  # post-LayerNorm activations fed into attention
                with torch.no_grad():
                    q = _reshape_heads(module.query(hidden), module)
                    k = _reshape_heads(module.key(hidden), module)
                    scale = math.sqrt(module.attention_head_size)
                    qkt = torch.matmul(q, k.transpose(-1, -2)) / scale  # [B, H, S, S]
                    for h in range(qkt.shape[1]):
                        max_abs = float(qkt[0, h].abs().max().item())
                        qkt_maxabs[(block_idx, h)].append(max_abs)
            return hook
        h = model.vit.encoder.layer[i].attention.attention.register_forward_hook(_make_qkt(i))
        handles.append(h)

    model.eval()
    with torch.no_grad():
        for _, img in images:
            inp = processor(images=img, return_tensors="pt")
            model(**inp)

    for h in handles:
        h.remove()

    return residual1_maxabs, residual2_maxabs, qkt_maxabs


# ---------------------------------------------------------------------------
# VADD saturation: golden model instrumentation
# ---------------------------------------------------------------------------

def run_golden_with_vadd_stats(program, patches_int8, cls_int8=None, num_classes=1000):
    """Run golden model and return (logits, vadd_stats).

    vadd_stats is a list of {'total': int, 'saturated': int} for each
    INT8 VADD instruction executed (pos_embed + residual adds).
    """
    state = MachineState()
    sim = Simulator(state)
    sim.load_program(program)

    M, N = patches_int8.shape
    N_pad = pad_dim(N)
    if N < N_pad:
        p = np.zeros((M, N_pad), dtype=np.int8)
        p[:M, :N] = patches_int8
        patches_int8 = p
    patch_bytes = patches_int8.tobytes()
    dram_off = program.input_offset
    state.dram[dram_off:dram_off + len(patch_bytes)] = patch_bytes
    if cls_int8 is not None and getattr(program, "cls_token_dram_offset", 0) > 0:
        cls_row = np.asarray(cls_int8, dtype=np.int8)
        if cls_row.ndim == 1:
            cls_row = cls_row.reshape(1, -1)
        cls_bytes = cls_row[0].tobytes()
        off = program.cls_token_dram_offset
        state.dram[off:off + len(cls_bytes)] = cls_bytes
    if cls_int8 is not None and getattr(program, "pos_embed_cls_dram_offset", 0) > 0:
        off = program.pos_embed_cls_dram_offset
        state.dram[off:off + 192] = bytes(192)
    if program.pos_embed_patch_dram_offset > 0:
        patch_pos_size = patches_int8.shape[0] * pad_dim(patches_int8.shape[1])
        off = program.pos_embed_patch_dram_offset
        state.dram[off:off + patch_pos_size] = bytes(patch_pos_size)

    vadd_stats = []
    _original = sim._exec_vadd  # bound method

    def _diagnostic_vadd(insn):
        if sim.state.tile_config is not None and insn.src1_buf == BUF_ABUF:
            m_tiles = sim.state.tile_config[0] + 1
            n_tiles = sim.state.tile_config[1] + 1
            M_t = m_tiles * 16
            N_t = n_tiles * 16
            src1 = mem_module.read_int8_tile(sim.state, insn.src1_buf, insn.src1_off, M_t, N_t)
            src2 = mem_module.read_int8_tile(sim.state, insn.src2_buf, insn.src2_off, M_t, N_t)
            sums = src1.astype(np.int16) + src2.astype(np.int16)
            saturated = int(np.sum((sums > 127) | (sums < -128)))
            vadd_stats.append({"total": M_t * N_t, "saturated": saturated})
        _original(insn)

    sim._exec_vadd = _diagnostic_vadd
    sim.run()

    logits = state.accum[:num_classes].copy().astype(np.float32)
    return logits, vadd_stats


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _vadd_label(idx):
    """Map VADD index to a human-readable label.

    Expected order: pos_embed_add(0), then per block: res1(2i+1), res2(2i+2).
    """
    if idx == 0:
        return "pos_embed_add"
    bidx = (idx - 1) // 2
    rtype = "res1" if (idx - 1) % 2 == 0 else "res2"
    return f"block{bidx:02d}_{rtype}"


def print_report(residual1_maxabs, residual2_maxabs, qkt_maxabs,
                 vadd_stats_per_image, block_input_scale):

    W = 90

    # --- A1: Per-block residual ranges ---
    print("\n" + "=" * W)
    print("  A1: Per-Block Residual Output Ranges vs. constant block_input_scale")
    print("=" * W)
    print(f"\n  Constant block_input_scale = {block_input_scale:.5f}  "
          f"(covers max_abs ≤ {block_input_scale * 127:.2f})\n")
    hdr = (f"  {'Blk':>4}  {'Res1 max (avg)':>15}  {'Ideal scale1':>13}  {'Ratio1':>7}  "
           f"{'Res2 max (avg)':>15}  {'Ideal scale2':>13}  {'Ratio2':>7}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    any_mismatch = False
    for i in range(DEPTH):
        r1 = residual1_maxabs[i]
        r2 = residual2_maxabs[i]
        if not r1 or not r2:
            continue
        r1m = np.mean(r1)
        r2m = np.mean(r2)
        id1 = r1m / 127.0
        id2 = r2m / 127.0
        rat1 = id1 / block_input_scale if block_input_scale > 1e-9 else 0.0
        rat2 = id2 / block_input_scale if block_input_scale > 1e-9 else 0.0
        flag1 = " !" if (rat1 > 2.0 or rat1 < 0.5) else "  "
        flag2 = " !" if (rat2 > 2.0 or rat2 < 0.5) else "  "
        if "!" in flag1 or "!" in flag2:
            any_mismatch = True
        print(f"  {i:>4}  {r1m:>15.3f}  {id1:>13.5f}  {rat1:>6.2f}x{flag1}  "
              f"{r2m:>15.3f}  {id2:>13.5f}  {rat2:>6.2f}x{flag2}")

    print()
    if any_mismatch:
        print("  [!] Ratio > 2x  → scale too coarse → residual add CLIPS values")
        print("  [!] Ratio < 0.5x → scale too fine → wastes dynamic range")
    else:
        print("  All ratios within 2x: constant block_input_scale is a reasonable fit.")

    # --- A3: Per-head QKT ranges ---
    print("\n" + "=" * W)
    print("  A3: Per-Head QKT Ranges vs. Hardcoded 6.0/127.0 = 0.04724")
    print("=" * W)
    HARDCODED = 6.0 / 127.0
    print(f"\n  Hardcoded QKT scale = {HARDCODED:.5f}  (covers max_abs ≤ 6.0)\n")
    hdr2 = f"  {'Blk':>4}  {'Hd':>3}  {'QKT max (avg)':>14}  {'Ideal scale':>12}  {'Ratio':>7}  Verdict"
    print(hdr2)
    print("  " + "-" * (len(hdr2) - 2))

    clipping_heads = []
    for i in range(DEPTH):
        for h in range(NUM_HEADS):
            vals = qkt_maxabs[(i, h)]
            if not vals:
                continue
            mean_max = np.mean(vals)
            ideal = mean_max / 127.0
            ratio = ideal / HARDCODED
            if ratio > 1.3:
                verdict = "CLIPS  <-- quantizes with loss"
                clipping_heads.append((i, h, mean_max, ratio))
            elif ratio < 0.5:
                verdict = "wastes range"
            else:
                verdict = "OK"
            print(f"  {i:>4}  {h:>3}  {mean_max:>14.3f}  {ideal:>12.5f}  {ratio:>6.2f}x  {verdict}")

    print()
    if clipping_heads:
        print(f"  [!] {len(clipping_heads)} head(s) clip the QKT INT8 range → calibrate per-head QKT scales (fix B1)")
    else:
        print("  All QKT ranges fit within 6.0 → hardcoded scale is fine.")

    # --- VADD Saturation ---
    print("\n" + "=" * W)
    print("  VADD Saturation (INT8 residual/pos_embed adds)")
    print("=" * W)

    if not vadd_stats_per_image:
        print("  No INT8 VADD stats collected.")
    else:
        n_vadds = max(len(s) for s in vadd_stats_per_image)
        print(f"\n  {'VADD #':>7}  {'Label':<20}  {'Avg sat %':>10}  {'Max sat %':>10}  Note")
        print("  " + "-" * 70)
        any_clip = False
        for idx in range(n_vadds):
            pcts = []
            for stats in vadd_stats_per_image:
                if idx < len(stats):
                    s = stats[idx]
                    pcts.append(100.0 * s["saturated"] / max(s["total"], 1))
            if not pcts:
                continue
            avg_p = np.mean(pcts)
            max_p = np.max(pcts)
            note = ""
            if avg_p > 5.0:
                note = "!!! heavy clipping"
                any_clip = True
            elif avg_p > 1.0:
                note = "! moderate clipping"
                any_clip = True
            label = _vadd_label(idx)
            print(f"  {idx:>7}  {label:<20}  {avg_p:>10.2f}%  {max_p:>10.2f}%  {note}")
        print()
        if any_clip:
            print("  [!] Saturation > 1% → residual adds are clipping → fix B2 (per-block scales)")
        else:
            print("  Saturation < 1% in all residual adds → VADD clipping is not a major issue.")

    # --- Summary ---
    print("\n" + "=" * W)
    print("  SUMMARY & RECOMMENDED NEXT STEPS")
    print("=" * W)
    print()

    issues = []
    # Check residual scale mismatch
    for i in range(DEPTH):
        r2 = residual2_maxabs[i]
        if r2:
            ratio = (np.mean(r2) / 127.0) / block_input_scale
            if ratio > 2.0 or ratio < 0.5:
                issues.append(f"  H1: Block {i} residual2 scale ratio = {ratio:.2f}x → implement B2 (per-block scales)")
                break

    # Check QKT clipping
    if clipping_heads:
        issues.append(f"  H5: {len(clipping_heads)} heads have QKT max_abs > 6.0 → implement B1 (calibrate QKT scales)")

    # Check VADD saturation
    if vadd_stats_per_image:
        for idx in range(min(25, max(len(s) for s in vadd_stats_per_image))):
            pcts = [100.0 * s[idx]["saturated"] / max(s[idx]["total"], 1)
                    for s in vadd_stats_per_image if idx < len(s)]
            if pcts and np.mean(pcts) > 1.0:
                issues.append(f"  H4: VADD #{idx} ({_vadd_label(idx)}) avg saturation {np.mean(pcts):.1f}% → fix B2")
                break

    if issues:
        for issue in issues:
            print(issue)
    else:
        print("  No major issues detected in these diagnostics.")
        print("  Consider investigating H3 (softmax double-quantization) via A4.")
    print()


def simulate_projection_quantization_modes(
    input_fp32: np.ndarray,
    input_scale: float,
    weight_fp32: np.ndarray,
    bias_fp32: np.ndarray,
    target_scale: float,
):
    """Simulate baseline per-tensor vs REQUANT_PC per-channel projection outputs."""
    input_scale = max(float(input_scale), 1e-12)
    target_scale = max(float(target_scale), 1e-12)

    inp_q = np.clip(np.round(input_fp32.astype(np.float32) / input_scale), -128, 127).astype(np.int8)
    inp_i32 = inp_q.astype(np.int32)

    pt_w_scale = max(float(np.max(np.abs(weight_fp32))), 1e-8) / 127.0
    pt_w_q = np.clip(np.round(weight_fp32 / pt_w_scale), -128, 127).astype(np.int8)
    pt_bias_i32 = np.round(bias_fp32.astype(np.float32) / (input_scale * pt_w_scale)).astype(np.int32)
    pt_acc = inp_i32 @ pt_w_q.astype(np.int32).T + pt_bias_i32.reshape(1, -1)
    pt_requant_scale = np.float32(input_scale * pt_w_scale / target_scale)
    pt_q = np.clip(np.round(pt_acc.astype(np.float32) * pt_requant_scale), -128, 127).astype(np.int8)
    pt_dq = pt_q.astype(np.float32) * np.float32(target_scale)

    pc_w_max = np.maximum(np.max(np.abs(weight_fp32), axis=1), 1e-8).astype(np.float32)
    pc_w_scales = pc_w_max / 127.0
    pc_w_q = np.clip(np.round(weight_fp32 / pc_w_scales.reshape(-1, 1)), -128, 127).astype(np.int8)
    pc_bias_i32 = np.round(bias_fp32.astype(np.float32) / (input_scale * pc_w_scales)).astype(np.int32)
    pc_acc = inp_i32 @ pc_w_q.astype(np.int32).T + pc_bias_i32.reshape(1, -1)
    pc_requant_scales = (np.float32(input_scale) * pc_w_scales / np.float32(target_scale)).reshape(1, -1)
    pc_q = np.clip(np.round(pc_acc.astype(np.float32) * pc_requant_scales), -128, 127).astype(np.int8)
    pc_dq = pc_q.astype(np.float32) * np.float32(target_scale)

    return {
        "baseline": pt_dq,
        "requant_pc": pc_dq,
        "input_zero_fraction": float(np.mean(inp_q == 0)),
        "input_saturation_rate": float(np.mean((inp_q == 127) | (inp_q == -128))),
        "pt_weight_scale": float(pt_w_scale),
        "pc_weight_scale_mean": float(np.mean(pc_w_scales)),
        "pc_weight_scale_min": float(np.min(pc_w_scales)),
        "pc_weight_scale_max": float(np.max(pc_w_scales)),
    }


def collect_qkv_requant_pc_stats(model, processor, images, cal_scales, blocks=None):
    """Compare local Q/K/V head outputs for baseline vs REQUANT_PC quantization."""
    selected_blocks = set(range(DEPTH)) if blocks is None else set(blocks)
    proj_buckets = {}
    qkt_buckets = {}

    for block_idx in selected_blocks:
        for head_idx in range(NUM_HEADS):
            for proj in ("query", "key", "value"):
                proj_buckets[(block_idx, head_idx, proj)] = {
                    "baseline_cos": [],
                    "pc_cos": [],
                    "baseline_mae": [],
                    "pc_mae": [],
                    "input_zero_fraction": [],
                    "input_saturation_rate": [],
                    "pt_weight_scale": [],
                    "pc_weight_scale_mean": [],
                    "pc_weight_scale_min": [],
                    "pc_weight_scale_max": [],
                }
            qkt_buckets[(block_idx, head_idx)] = {
                "baseline_cos": [],
                "pc_cos": [],
                "baseline_mae": [],
                "pc_mae": [],
            }

    model.eval()
    with torch.no_grad():
        for _, img in images:
            inp = processor(images=img, return_tensors="pt")
            prev = model.vit.embeddings(inp["pixel_values"])

            for block_idx, layer in enumerate(model.vit.encoder.layer):
                attn = layer.attention.attention
                ln1 = layer.layernorm_before(prev)

                if block_idx in selected_blocks:
                    ln1_np = ln1[0].detach().cpu().numpy().astype(np.float32)
                    ln1_scale = cal_scales.get(f"block{block_idx}_ln1", 6.0 / 127.0)
                    q_full = attn.query(ln1)[0].detach().cpu().numpy().astype(np.float32)
                    k_full = attn.key(ln1)[0].detach().cpu().numpy().astype(np.float32)
                    v_full = attn.value(ln1)[0].detach().cpu().numpy().astype(np.float32)
                    full_outputs = {
                        "query": q_full,
                        "key": k_full,
                        "value": v_full,
                    }
                    baseline_heads = {"query": {}, "key": {}}
                    pc_heads = {"query": {}, "key": {}}

                    for head_idx in range(NUM_HEADS):
                        sl = slice(head_idx * HEAD_DIM, (head_idx + 1) * HEAD_DIM)
                        for proj in ("query", "key", "value"):
                            target_scale = cal_scales.get(
                                f"block{block_idx}_head{head_idx}_{proj}",
                                6.0 / 127.0,
                            )
                            weight = getattr(attn, proj).weight.detach().cpu().numpy().astype(np.float32)[sl, :]
                            bias = getattr(attn, proj).bias.detach().cpu().numpy().astype(np.float32)[sl]
                            target = full_outputs[proj][:, sl]
                            simulated = simulate_projection_quantization_modes(
                                ln1_np,
                                ln1_scale,
                                weight,
                                bias,
                                target_scale,
                            )
                            bucket = proj_buckets[(block_idx, head_idx, proj)]
                            base_metrics = tensor_error_metrics(target, simulated["baseline"])
                            pc_metrics = tensor_error_metrics(target, simulated["requant_pc"])
                            bucket["baseline_cos"].append(base_metrics["cosine_sim"])
                            bucket["pc_cos"].append(pc_metrics["cosine_sim"])
                            bucket["baseline_mae"].append(base_metrics["mean_abs_error"])
                            bucket["pc_mae"].append(pc_metrics["mean_abs_error"])
                            bucket["input_zero_fraction"].append(simulated["input_zero_fraction"])
                            bucket["input_saturation_rate"].append(simulated["input_saturation_rate"])
                            bucket["pt_weight_scale"].append(simulated["pt_weight_scale"])
                            bucket["pc_weight_scale_mean"].append(simulated["pc_weight_scale_mean"])
                            bucket["pc_weight_scale_min"].append(simulated["pc_weight_scale_min"])
                            bucket["pc_weight_scale_max"].append(simulated["pc_weight_scale_max"])
                            if proj in ("query", "key"):
                                baseline_heads[proj][head_idx] = simulated["baseline"]
                                pc_heads[proj][head_idx] = simulated["requant_pc"]

                    for head_idx in range(NUM_HEADS):
                        sl = slice(head_idx * HEAD_DIM, (head_idx + 1) * HEAD_DIM)
                        q_target = q_full[:, sl]
                        k_target = k_full[:, sl]
                        qkt_target = (q_target @ k_target.T).astype(np.float32) * np.float32(0.125)
                        qkt_base = (baseline_heads["query"][head_idx] @ baseline_heads["key"][head_idx].T).astype(np.float32) * np.float32(0.125)
                        qkt_pc = (pc_heads["query"][head_idx] @ pc_heads["key"][head_idx].T).astype(np.float32) * np.float32(0.125)
                        bucket = qkt_buckets[(block_idx, head_idx)]
                        base_metrics = tensor_error_metrics(qkt_target, qkt_base)
                        pc_metrics = tensor_error_metrics(qkt_target, qkt_pc)
                        bucket["baseline_cos"].append(base_metrics["cosine_sim"])
                        bucket["pc_cos"].append(pc_metrics["cosine_sim"])
                        bucket["baseline_mae"].append(base_metrics["mean_abs_error"])
                        bucket["pc_mae"].append(pc_metrics["mean_abs_error"])

                block_out = layer(prev)
                prev = block_out[0] if isinstance(block_out, tuple) else block_out

    projection_report = []
    for (block_idx, head_idx, proj), bucket in sorted(proj_buckets.items()):
        projection_report.append({
            "block_idx": block_idx,
            "head_idx": head_idx,
            "proj": proj,
            "baseline_mean_cosine": float(np.mean(bucket["baseline_cos"])) if bucket["baseline_cos"] else 0.0,
            "requant_pc_mean_cosine": float(np.mean(bucket["pc_cos"])) if bucket["pc_cos"] else 0.0,
            "delta_cosine": float(np.mean(bucket["pc_cos"]) - np.mean(bucket["baseline_cos"])) if bucket["pc_cos"] else 0.0,
            "baseline_mean_mae": float(np.mean(bucket["baseline_mae"])) if bucket["baseline_mae"] else 0.0,
            "requant_pc_mean_mae": float(np.mean(bucket["pc_mae"])) if bucket["pc_mae"] else 0.0,
            "mean_input_zero_fraction": float(np.mean(bucket["input_zero_fraction"])) if bucket["input_zero_fraction"] else 0.0,
            "mean_input_saturation_rate": float(np.mean(bucket["input_saturation_rate"])) if bucket["input_saturation_rate"] else 0.0,
            "mean_per_tensor_weight_scale": float(np.mean(bucket["pt_weight_scale"])) if bucket["pt_weight_scale"] else 0.0,
            "mean_per_channel_weight_scale": float(np.mean(bucket["pc_weight_scale_mean"])) if bucket["pc_weight_scale_mean"] else 0.0,
            "mean_per_channel_weight_scale_min": float(np.mean(bucket["pc_weight_scale_min"])) if bucket["pc_weight_scale_min"] else 0.0,
            "mean_per_channel_weight_scale_max": float(np.mean(bucket["pc_weight_scale_max"])) if bucket["pc_weight_scale_max"] else 0.0,
        })

    qkt_report = []
    for (block_idx, head_idx), bucket in sorted(qkt_buckets.items()):
        qkt_report.append({
            "block_idx": block_idx,
            "head_idx": head_idx,
            "baseline_mean_cosine": float(np.mean(bucket["baseline_cos"])) if bucket["baseline_cos"] else 0.0,
            "requant_pc_mean_cosine": float(np.mean(bucket["pc_cos"])) if bucket["pc_cos"] else 0.0,
            "delta_cosine": float(np.mean(bucket["pc_cos"]) - np.mean(bucket["baseline_cos"])) if bucket["pc_cos"] else 0.0,
            "baseline_mean_mae": float(np.mean(bucket["baseline_mae"])) if bucket["baseline_mae"] else 0.0,
            "requant_pc_mean_mae": float(np.mean(bucket["pc_mae"])) if bucket["pc_mae"] else 0.0,
        })

    return {
        "n_images": len(images),
        "projection_report": projection_report,
        "qkt_report": qkt_report,
    }


def print_qkv_requant_pc_report(report):
    """Print a compact summary of local Q/K/V and QKT deltas."""
    proj_rows = report["projection_report"]
    qkt_rows = report["qkt_report"]

    print("\n" + "=" * 90)
    print("  Q/K/V REQUANT_PC Local Diagnostic")
    print("=" * 90)
    print(f"\n  Images analysed: {report['n_images']}")

    proj_sorted = sorted(proj_rows, key=lambda row: row["delta_cosine"])
    print("\n  Worst projection deltas (REQUANT_PC - baseline):")
    for row in proj_sorted[:8]:
        print(
            f"    block{row['block_idx']:02d} h{row['head_idx']} {row['proj']:<5}  "
            f"baseline={row['baseline_mean_cosine']:.4f}  pc={row['requant_pc_mean_cosine']:.4f}  "
            f"delta={row['delta_cosine']:+.4f}"
        )

    proj_best = sorted(proj_rows, key=lambda row: row["delta_cosine"], reverse=True)
    print("\n  Best projection deltas:")
    for row in proj_best[:8]:
        print(
            f"    block{row['block_idx']:02d} h{row['head_idx']} {row['proj']:<5}  "
            f"baseline={row['baseline_mean_cosine']:.4f}  pc={row['requant_pc_mean_cosine']:.4f}  "
            f"delta={row['delta_cosine']:+.4f}"
        )

    qkt_sorted = sorted(qkt_rows, key=lambda row: row["delta_cosine"])
    print("\n  Worst QKT deltas:")
    for row in qkt_sorted[:8]:
        print(
            f"    block{row['block_idx']:02d} h{row['head_idx']} qkt    "
            f"baseline={row['baseline_mean_cosine']:.4f}  pc={row['requant_pc_mean_cosine']:.4f}  "
            f"delta={row['delta_cosine']:+.4f}"
        )

    by_proj = collections.defaultdict(list)
    for row in proj_rows:
        by_proj[row["proj"]].append(row["delta_cosine"])
    print("\n  Mean projection delta by type:")
    for proj in ("query", "key", "value"):
        vals = by_proj.get(proj, [])
        if vals:
            print(f"    {proj:<5}: {float(np.mean(vals)):+.4f}")

    if qkt_rows:
        print(f"\n  Mean QKT delta: {float(np.mean([row['delta_cosine'] for row in qkt_rows])):+.4f}")


def _parse_int_set(spec: str):
    if not spec:
        return None
    return {int(part) for part in spec.split(",") if part.strip()}


def _parse_block_head_set(spec: str):
    if not spec:
        return None
    result = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        block_str, head_str = item.split(":", 1)
        result.add((int(block_str), int(head_str)))
    return result


def _node_block_idx(node_name: str):
    if not node_name.startswith("block"):
        return None
    digits = []
    for ch in node_name[5:]:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    return int("".join(digits)) if digits else None


def _node_stage(node_name: str):
    if node_name == "classifier":
        return "classifier"
    for suffix in (
        "query",
        "key",
        "value",
        "qkt",
        "softmax",
        "attn_v",
        "concat",
        "out_proj",
        "gelu",
        "fc2",
        "residual2",
    ):
        if node_name.endswith(f"_{suffix}"):
            return suffix
    return None


def _node_head_idx(node_name: str):
    marker = "_head"
    if marker not in node_name:
        return None
    tail = node_name.split(marker, 1)[1]
    digits = []
    for ch in tail:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    return int("".join(digits)) if digits else None


def compare_attention_runtime_path(
    fp32_value: np.ndarray,
    fp32_qkt: np.ndarray,
    fp32_softmax: np.ndarray,
    fp32_attn_v: np.ndarray,
    baseline_value: np.ndarray,
    baseline_qkt: np.ndarray,
    baseline_softmax: np.ndarray,
    baseline_attn_v: np.ndarray,
    variant_value: np.ndarray,
    variant_qkt: np.ndarray,
    variant_softmax: np.ndarray,
    variant_attn_v: np.ndarray,
):
    """Compare baseline vs variant attention-path tensors for one head."""
    baseline_replay = np.matmul(baseline_softmax.astype(np.float32), baseline_value.astype(np.float32))
    variant_replay = np.matmul(variant_softmax.astype(np.float32), variant_value.astype(np.float32))
    variant_softmax_only = np.matmul(variant_softmax.astype(np.float32), baseline_value.astype(np.float32))
    variant_value_only = np.matmul(baseline_softmax.astype(np.float32), variant_value.astype(np.float32))

    report = {
        "baseline": {
            "value_metrics": tensor_error_metrics(fp32_value, baseline_value),
            "qkt_metrics": tensor_error_metrics(fp32_qkt, baseline_qkt),
            "softmax_metrics": tensor_error_metrics(fp32_softmax, baseline_softmax),
            "attn_v_metrics": tensor_error_metrics(fp32_attn_v, baseline_attn_v),
            "replay_to_fp32_metrics": tensor_error_metrics(fp32_attn_v, baseline_replay),
            "runtime_vs_replay_metrics": tensor_error_metrics(baseline_attn_v, baseline_replay),
        },
        "variant": {
            "value_metrics": tensor_error_metrics(fp32_value, variant_value),
            "qkt_metrics": tensor_error_metrics(fp32_qkt, variant_qkt),
            "softmax_metrics": tensor_error_metrics(fp32_softmax, variant_softmax),
            "attn_v_metrics": tensor_error_metrics(fp32_attn_v, variant_attn_v),
            "replay_to_fp32_metrics": tensor_error_metrics(fp32_attn_v, variant_replay),
            "runtime_vs_replay_metrics": tensor_error_metrics(variant_attn_v, variant_replay),
        },
        "variant_replay_isolation": {
            "softmax_only_to_fp32_metrics": tensor_error_metrics(fp32_attn_v, variant_softmax_only),
            "value_only_to_fp32_metrics": tensor_error_metrics(fp32_attn_v, variant_value_only),
        },
    }

    stage_deltas = {}
    for stage in ("value", "qkt", "softmax", "attn_v"):
        stage_deltas[f"{stage}_delta_cosine"] = (
            report["variant"][f"{stage}_metrics"]["cosine_sim"]
            - report["baseline"][f"{stage}_metrics"]["cosine_sim"]
        )
    stage_deltas["replay_to_fp32_delta_cosine"] = (
        report["variant"]["replay_to_fp32_metrics"]["cosine_sim"]
        - report["baseline"]["replay_to_fp32_metrics"]["cosine_sim"]
    )
    stage_deltas["runtime_vs_replay_delta_cosine"] = (
        report["variant"]["runtime_vs_replay_metrics"]["cosine_sim"]
        - report["baseline"]["runtime_vs_replay_metrics"]["cosine_sim"]
    )
    report["deltas"] = {key: float(val) for key, val in stage_deltas.items()}
    return report


def summarize_trace_variant_delta(
    baseline_trace_report: dict,
    variant_trace_report: dict,
    blocks=None,
):
    """Compare two saved compare_golden trace reports on common traced images."""
    baseline_per_image = baseline_trace_report.get("per_image", [])
    variant_per_image = variant_trace_report.get("per_image", [])
    baseline_map = {item["img_id"]: item for item in baseline_per_image}
    variant_map = {item["img_id"]: item for item in variant_per_image}
    common_ids = sorted(set(baseline_map) & set(variant_map))

    buckets = {}
    first_drop_counts = collections.Counter()
    interesting_stages = {
        "qkt", "softmax", "attn_v", "concat", "out_proj", "gelu", "fc2", "residual2", "classifier"
    }

    for img_id in common_ids:
        baseline_item = baseline_map[img_id]
        variant_item = variant_map[img_id]
        baseline_nodes = {metric["node"]: metric for metric in baseline_item.get("node_metrics", [])}
        variant_nodes = {metric["node"]: metric for metric in variant_item.get("node_metrics", [])}

        variant_first = variant_item.get("first_major_drop")
        if variant_first is not None:
            node_name = variant_first.get("node")
            if node_name:
                stage = _node_stage(node_name)
                block_idx = _node_block_idx(node_name)
                if stage in interesting_stages and (blocks is None or block_idx in blocks or node_name == "classifier"):
                    first_drop_counts[node_name] += 1

        common_nodes = sorted(set(baseline_nodes) & set(variant_nodes))
        for node_name in common_nodes:
            stage = _node_stage(node_name)
            block_idx = _node_block_idx(node_name)
            if stage not in interesting_stages:
                continue
            if node_name != "classifier" and blocks is not None and block_idx not in blocks:
                continue

            base_metric = baseline_nodes[node_name]
            var_metric = variant_nodes[node_name]
            bucket = buckets.setdefault(node_name, {
                "node": node_name,
                "block_idx": block_idx,
                "stage": stage,
                "baseline_cos": [],
                "variant_cos": [],
                "baseline_qdq_cos": [],
                "variant_qdq_cos": [],
                "baseline_max_abs": [],
                "variant_max_abs": [],
                "image_ids": [],
            })
            bucket["baseline_cos"].append(float(base_metric["cosine_sim"]))
            bucket["variant_cos"].append(float(var_metric["cosine_sim"]))
            bucket["baseline_qdq_cos"].append(float(base_metric.get("qdq_cosine_sim", 0.0)))
            bucket["variant_qdq_cos"].append(float(var_metric.get("qdq_cosine_sim", 0.0)))
            bucket["baseline_max_abs"].append(float(base_metric["max_abs_error"]))
            bucket["variant_max_abs"].append(float(var_metric["max_abs_error"]))
            bucket["image_ids"].append(int(img_id))

    node_report = []
    for bucket in buckets.values():
        node_report.append({
            "node": bucket["node"],
            "block_idx": bucket["block_idx"],
            "stage": bucket["stage"],
            "n_images": len(bucket["image_ids"]),
            "image_ids": bucket["image_ids"],
            "baseline_mean_cosine": float(np.mean(bucket["baseline_cos"])) if bucket["baseline_cos"] else 0.0,
            "variant_mean_cosine": float(np.mean(bucket["variant_cos"])) if bucket["variant_cos"] else 0.0,
            "delta_cosine": (
                float(np.mean(bucket["variant_cos"]) - np.mean(bucket["baseline_cos"]))
                if bucket["variant_cos"] else 0.0
            ),
            "baseline_mean_qdq_cosine": float(np.mean(bucket["baseline_qdq_cos"])) if bucket["baseline_qdq_cos"] else 0.0,
            "variant_mean_qdq_cosine": float(np.mean(bucket["variant_qdq_cos"])) if bucket["variant_qdq_cos"] else 0.0,
            "delta_qdq_cosine": (
                float(np.mean(bucket["variant_qdq_cos"]) - np.mean(bucket["baseline_qdq_cos"]))
                if bucket["variant_qdq_cos"] else 0.0
            ),
            "baseline_mean_max_abs_error": float(np.mean(bucket["baseline_max_abs"])) if bucket["baseline_max_abs"] else 0.0,
            "variant_mean_max_abs_error": float(np.mean(bucket["variant_max_abs"])) if bucket["variant_max_abs"] else 0.0,
            "delta_max_abs_error": (
                float(np.mean(bucket["variant_max_abs"]) - np.mean(bucket["baseline_max_abs"]))
                if bucket["variant_max_abs"] else 0.0
            ),
        })

    node_report.sort(key=lambda row: row["delta_cosine"])

    stage_summary = []
    by_stage = collections.defaultdict(list)
    for row in node_report:
        by_stage[row["stage"]].append(row)
    for stage, rows in sorted(by_stage.items()):
        stage_summary.append({
            "stage": stage,
            "n_nodes": len(rows),
            "mean_delta_cosine": float(np.mean([row["delta_cosine"] for row in rows])),
            "worst_node": min(rows, key=lambda row: row["delta_cosine"])["node"],
            "worst_delta_cosine": float(min(row["delta_cosine"] for row in rows)),
        })

    return {
        "common_traced_image_ids": common_ids,
        "blocks": sorted(blocks) if blocks is not None else None,
        "node_report": node_report,
        "stage_summary": stage_summary,
        "variant_first_drop_counts": dict(first_drop_counts),
    }


def print_trace_variant_delta_report(report):
    """Print a compact late-path diff between baseline and a variant trace run."""
    print("\n" + "=" * 90)
    print("  Trace Variant Delta Report")
    print("=" * 90)
    print(f"\n  Common traced images: {report['common_traced_image_ids']}")
    if report["blocks"] is not None:
        print(f"  Analysed blocks    : {report['blocks']}")

    print("\n  Mean delta by stage (variant - baseline):")
    for row in report["stage_summary"]:
        print(
            f"    {row['stage']:<10} mean_delta={row['mean_delta_cosine']:+.4f}  "
            f"worst={row['worst_node']} ({row['worst_delta_cosine']:+.4f})"
        )

    print("\n  Worst node deltas:")
    for row in report["node_report"][:12]:
        label = row["node"]
        print(
            f"    {label:<24} delta_cos={row['delta_cosine']:+.4f}  "
            f"base={row['baseline_mean_cosine']:.4f}  var={row['variant_mean_cosine']:.4f}  "
            f"delta_qdq={row['delta_qdq_cosine']:+.4f}  "
            f"delta_max_abs={row['delta_max_abs_error']:+.4f}"
        )

    if report["variant_first_drop_counts"]:
        print("\n  Variant first-major-drop counts:")
        for node_name, count in sorted(
            report["variant_first_drop_counts"].items(),
            key=lambda item: (-item[1], item[0]),
        ):
            print(f"    {node_name:<24} {count}")


def summarize_late_attention_path_delta(
    baseline_trace_report: dict,
    variant_trace_report: dict,
    blocks=None,
):
    """Compare baseline vs variant attention-path nodes grouped by (block, head)."""
    baseline_per_image = baseline_trace_report.get("per_image", [])
    variant_per_image = variant_trace_report.get("per_image", [])
    baseline_map = {item["img_id"]: item for item in baseline_per_image}
    variant_map = {item["img_id"]: item for item in variant_per_image}
    common_ids = sorted(set(baseline_map) & set(variant_map))
    stage_order = ("value", "qkt", "softmax", "attn_v")

    per_head = {}
    for img_id in common_ids:
        baseline_item = baseline_map[img_id]
        variant_item = variant_map[img_id]
        baseline_nodes = {metric["node"]: metric for metric in baseline_item.get("node_metrics", [])}
        variant_nodes = {metric["node"]: metric for metric in variant_item.get("node_metrics", [])}

        for node_name in sorted(set(baseline_nodes) & set(variant_nodes)):
            stage = _node_stage(node_name)
            if stage not in stage_order:
                continue
            block_idx = _node_block_idx(node_name)
            head_idx = _node_head_idx(node_name)
            if block_idx is None or head_idx is None:
                continue
            if blocks is not None and block_idx not in blocks:
                continue

            bucket = per_head.setdefault((block_idx, head_idx), {
                "block_idx": block_idx,
                "head_idx": head_idx,
                "image_ids": set(),
                "stages": {
                    key: {
                        "baseline_cos": [],
                        "variant_cos": [],
                        "baseline_qdq_cos": [],
                        "variant_qdq_cos": [],
                        "baseline_max_abs": [],
                        "variant_max_abs": [],
                    }
                    for key in stage_order
                },
            })
            base_metric = baseline_nodes[node_name]
            var_metric = variant_nodes[node_name]
            sb = bucket["stages"][stage]
            sb["baseline_cos"].append(float(base_metric["cosine_sim"]))
            sb["variant_cos"].append(float(var_metric["cosine_sim"]))
            sb["baseline_qdq_cos"].append(float(base_metric.get("qdq_cosine_sim", 0.0)))
            sb["variant_qdq_cos"].append(float(var_metric.get("qdq_cosine_sim", 0.0)))
            sb["baseline_max_abs"].append(float(base_metric["max_abs_error"]))
            sb["variant_max_abs"].append(float(var_metric["max_abs_error"]))
            bucket["image_ids"].add(int(img_id))

    per_head_report = []
    stage_rows = collections.defaultdict(list)
    for (block_idx, head_idx), bucket in sorted(per_head.items()):
        row = {
            "block_idx": block_idx,
            "head_idx": head_idx,
            "n_images": len(bucket["image_ids"]),
            "image_ids": sorted(bucket["image_ids"]),
            "stages": {},
        }
        path_deltas = []
        for stage in stage_order:
            sb = bucket["stages"][stage]
            if not sb["baseline_cos"]:
                continue
            baseline_mean_cos = float(np.mean(sb["baseline_cos"]))
            variant_mean_cos = float(np.mean(sb["variant_cos"]))
            baseline_mean_qdq = float(np.mean(sb["baseline_qdq_cos"]))
            variant_mean_qdq = float(np.mean(sb["variant_qdq_cos"]))
            baseline_mean_max_abs = float(np.mean(sb["baseline_max_abs"]))
            variant_mean_max_abs = float(np.mean(sb["variant_max_abs"]))
            stage_row = {
                "baseline_mean_cosine": baseline_mean_cos,
                "variant_mean_cosine": variant_mean_cos,
                "delta_cosine": variant_mean_cos - baseline_mean_cos,
                "baseline_mean_qdq_cosine": baseline_mean_qdq,
                "variant_mean_qdq_cosine": variant_mean_qdq,
                "delta_qdq_cosine": variant_mean_qdq - baseline_mean_qdq,
                "baseline_mean_max_abs_error": baseline_mean_max_abs,
                "variant_mean_max_abs_error": variant_mean_max_abs,
                "delta_max_abs_error": variant_mean_max_abs - baseline_mean_max_abs,
            }
            row["stages"][stage] = stage_row
            stage_rows[stage].append({
                "block_idx": block_idx,
                "head_idx": head_idx,
                **stage_row,
            })
            path_deltas.append(stage_row["delta_cosine"])

        attn_v_stage = row["stages"].get("attn_v")
        row["mean_path_delta_cosine"] = float(np.mean(path_deltas)) if path_deltas else 0.0
        row["attn_v_delta_cosine"] = attn_v_stage["delta_cosine"] if attn_v_stage else 0.0
        per_head_report.append(row)

    per_head_report.sort(key=lambda row: (row["attn_v_delta_cosine"], row["mean_path_delta_cosine"]))

    stage_summary = []
    for stage in stage_order:
        rows = stage_rows.get(stage, [])
        if not rows:
            continue
        worst = min(rows, key=lambda row: row["delta_cosine"])
        stage_summary.append({
            "stage": stage,
            "n_heads": len(rows),
            "mean_delta_cosine": float(np.mean([row["delta_cosine"] for row in rows])),
            "worst_block_idx": int(worst["block_idx"]),
            "worst_head_idx": int(worst["head_idx"]),
            "worst_delta_cosine": float(worst["delta_cosine"]),
        })

    return {
        "common_traced_image_ids": common_ids,
        "blocks": sorted(blocks) if blocks is not None else None,
        "stage_order": list(stage_order),
        "stage_summary": stage_summary,
        "per_head_report": per_head_report,
    }


def print_late_attention_path_delta_report(report):
    """Print a focused late-attention path diff for selected blocks."""
    print("\n" + "=" * 90)
    print("  Late Attention Path Delta Report")
    print("=" * 90)
    print(f"\n  Common traced images: {report['common_traced_image_ids']}")
    if report["blocks"] is not None:
        print(f"  Analysed blocks    : {report['blocks']}")

    print("\n  Mean delta by stage (variant - baseline):")
    for row in report["stage_summary"]:
        print(
            f"    {row['stage']:<8} mean_delta={row['mean_delta_cosine']:+.4f}  "
            f"worst=block{row['worst_block_idx']:02d} h{row['worst_head_idx']} "
            f"({row['worst_delta_cosine']:+.4f})"
        )

    print("\n  Worst heads by attn_v / path delta:")
    for row in report["per_head_report"][:8]:
        stage_bits = []
        for stage in report["stage_order"]:
            stage_row = row["stages"].get(stage)
            if stage_row is None:
                continue
            stage_bits.append(f"{stage}={stage_row['delta_cosine']:+.4f}")
        print(
            f"    block{row['block_idx']:02d} h{row['head_idx']}  "
            f"attn_v={row['attn_v_delta_cosine']:+.4f}  "
            f"path={row['mean_path_delta_cosine']:+.4f}  "
            + "  ".join(stage_bits)
        )


def summarize_block_impact(compare_json: dict):
    """Summarize per-image and aggregate block contributions from one traced run."""
    trace_report = compare_json.get("trace_report", {})
    traced = trace_report.get("per_image", [])
    benchmark_items = {item["img_id"]: item for item in compare_json.get("per_image", [])}

    aggregate = {
        block_idx: {
            "input_cosine": [],
            "residual1_cosine": [],
            "residual2_cosine": [],
            "attention_delta": [],
            "mlp_delta": [],
            "total_delta": [],
            "loss_increase": [],
            "loss_recovery": [],
            "worsening_share": [],
            "recovery_share": [],
            "first_major_drop_count": 0,
            "worst_node_counts": collections.Counter(),
            "worst_delta_node_counts": collections.Counter(),
        }
        for block_idx in range(DEPTH)
    }

    per_image = []
    for traced_item in traced:
        img_id = int(traced_item["img_id"])
        benchmark = benchmark_items.get(img_id, {})
        node_metrics = traced_item.get("node_metrics", [])
        node_map = {metric["node"]: metric for metric in node_metrics}
        classifier_cosine = float(
            node_map.get("classifier", {}).get(
                "cosine_sim",
                benchmark.get("cosine_sim", 0.0),
            )
        )

        block_rows = []
        for block_idx in range(DEPTH):
            input_node = "pos_embed_add" if block_idx == 0 else f"block{block_idx - 1}_residual2"
            residual1_node = f"block{block_idx}_residual1"
            residual2_node = f"block{block_idx}_residual2"

            input_cosine = float(node_map.get(input_node, {}).get("cosine_sim", 1.0 if block_idx == 0 else 0.0))
            residual1_cosine = float(node_map.get(residual1_node, {}).get("cosine_sim", input_cosine))
            residual2_cosine = float(node_map.get(residual2_node, {}).get("cosine_sim", residual1_cosine))

            attention_delta = residual1_cosine - input_cosine
            mlp_delta = residual2_cosine - residual1_cosine
            total_delta = residual2_cosine - input_cosine
            loss_increase = max(-total_delta, 0.0)
            loss_recovery = max(total_delta, 0.0)

            block_node_metrics = [metric for metric in node_metrics if _node_block_idx(metric["node"]) == block_idx]
            worst_node = min(block_node_metrics, key=lambda metric: metric["cosine_sim"]) if block_node_metrics else None
            delta_candidates = [
                metric for metric in block_node_metrics
                if metric.get("delta_from_prev") is not None
            ]
            worst_delta_node = min(delta_candidates, key=lambda metric: metric["delta_from_prev"]) if delta_candidates else None

            block_rows.append({
                "block_idx": block_idx,
                "input_node": input_node,
                "input_cosine": input_cosine,
                "residual1_cosine": residual1_cosine,
                "residual2_cosine": residual2_cosine,
                "attention_delta": attention_delta,
                "mlp_delta": mlp_delta,
                "total_delta": total_delta,
                "loss_increase": loss_increase,
                "loss_recovery": loss_recovery,
                "worst_node": worst_node["node"] if worst_node is not None else None,
                "worst_node_cosine": float(worst_node["cosine_sim"]) if worst_node is not None else None,
                "worst_delta_node": worst_delta_node["node"] if worst_delta_node is not None else None,
                "worst_delta_from_prev": (
                    float(worst_delta_node["delta_from_prev"])
                    if worst_delta_node is not None else None
                ),
            })

        total_loss_increase = sum(row["loss_increase"] for row in block_rows)
        total_loss_recovery = sum(row["loss_recovery"] for row in block_rows)
        for row in block_rows:
            row["worsening_share"] = (
                row["loss_increase"] / total_loss_increase if total_loss_increase > 0 else 0.0
            )
            row["recovery_share"] = (
                row["loss_recovery"] / total_loss_recovery if total_loss_recovery > 0 else 0.0
            )

            bucket = aggregate[row["block_idx"]]
            bucket["input_cosine"].append(row["input_cosine"])
            bucket["residual1_cosine"].append(row["residual1_cosine"])
            bucket["residual2_cosine"].append(row["residual2_cosine"])
            bucket["attention_delta"].append(row["attention_delta"])
            bucket["mlp_delta"].append(row["mlp_delta"])
            bucket["total_delta"].append(row["total_delta"])
            bucket["loss_increase"].append(row["loss_increase"])
            bucket["loss_recovery"].append(row["loss_recovery"])
            bucket["worsening_share"].append(row["worsening_share"])
            bucket["recovery_share"].append(row["recovery_share"])
            if row["worst_node"] is not None:
                bucket["worst_node_counts"][row["worst_node"]] += 1
            if row["worst_delta_node"] is not None:
                bucket["worst_delta_node_counts"][row["worst_delta_node"]] += 1

        first_major_drop = traced_item.get("first_major_drop")
        first_major_drop_block = None
        if first_major_drop is not None:
            first_major_drop_block = _node_block_idx(first_major_drop.get("node", ""))
            if first_major_drop_block is not None:
                aggregate[first_major_drop_block]["first_major_drop_count"] += 1

        per_image.append({
            "img_id": img_id,
            "top1_match": bool(benchmark.get("top1_match", False)),
            "top5_overlap": float(benchmark.get("top5_overlap", 0.0)),
            "final_cosine": float(benchmark.get("cosine_sim", classifier_cosine)),
            "classifier_cosine": classifier_cosine,
            "fp32_top1": benchmark.get("fp32_top5", [None])[0],
            "golden_top1": benchmark.get("golden_top5", [None])[0],
            "first_major_drop": first_major_drop,
            "first_major_drop_block": first_major_drop_block,
            "total_loss_increase": total_loss_increase,
            "total_loss_recovery": total_loss_recovery,
            "blocks": block_rows,
        })

    aggregate_blocks = []
    for block_idx in range(DEPTH):
        bucket = aggregate[block_idx]
        worst_node = bucket["worst_node_counts"].most_common(1)
        worst_delta_node = bucket["worst_delta_node_counts"].most_common(1)
        aggregate_blocks.append({
            "block_idx": block_idx,
            "mean_input_cosine": float(np.mean(bucket["input_cosine"])) if bucket["input_cosine"] else 0.0,
            "mean_residual1_cosine": float(np.mean(bucket["residual1_cosine"])) if bucket["residual1_cosine"] else 0.0,
            "mean_residual2_cosine": float(np.mean(bucket["residual2_cosine"])) if bucket["residual2_cosine"] else 0.0,
            "mean_attention_delta": float(np.mean(bucket["attention_delta"])) if bucket["attention_delta"] else 0.0,
            "mean_mlp_delta": float(np.mean(bucket["mlp_delta"])) if bucket["mlp_delta"] else 0.0,
            "mean_total_delta": float(np.mean(bucket["total_delta"])) if bucket["total_delta"] else 0.0,
            "mean_loss_increase": float(np.mean(bucket["loss_increase"])) if bucket["loss_increase"] else 0.0,
            "mean_loss_recovery": float(np.mean(bucket["loss_recovery"])) if bucket["loss_recovery"] else 0.0,
            "mean_worsening_share": float(np.mean(bucket["worsening_share"])) if bucket["worsening_share"] else 0.0,
            "mean_recovery_share": float(np.mean(bucket["recovery_share"])) if bucket["recovery_share"] else 0.0,
            "first_major_drop_count": int(bucket["first_major_drop_count"]),
            "most_common_worst_node": worst_node[0][0] if worst_node else None,
            "most_common_worst_node_count": int(worst_node[0][1]) if worst_node else 0,
            "most_common_worst_delta_node": worst_delta_node[0][0] if worst_delta_node else None,
            "most_common_worst_delta_node_count": int(worst_delta_node[0][1]) if worst_delta_node else 0,
        })

    blocks_by_worsening = sorted(
        aggregate_blocks,
        key=lambda row: (row["mean_worsening_share"], -row["mean_total_delta"]),
        reverse=True,
    )
    blocks_by_recovery = sorted(
        aggregate_blocks,
        key=lambda row: (row["mean_recovery_share"], row["mean_total_delta"]),
        reverse=True,
    )

    return {
        "n_images": len(per_image),
        "aggregate_blocks": aggregate_blocks,
        "blocks_ranked_by_worsening": [
            {
                "block_idx": row["block_idx"],
                "mean_worsening_share": row["mean_worsening_share"],
                "mean_total_delta": row["mean_total_delta"],
                "first_major_drop_count": row["first_major_drop_count"],
            }
            for row in blocks_by_worsening
        ],
        "blocks_ranked_by_recovery": [
            {
                "block_idx": row["block_idx"],
                "mean_recovery_share": row["mean_recovery_share"],
                "mean_total_delta": row["mean_total_delta"],
            }
            for row in blocks_by_recovery
        ],
        "per_image": per_image,
    }


def print_block_impact_report(report):
    """Print a compact block-impact summary for one traced run."""
    print("\n" + "=" * 90)
    print("  Block Impact Report")
    print("=" * 90)
    print(f"\n  Analysed images: {report['n_images']}")

    print("\n  Blocks ranked by mean worsening share:")
    for row in report["blocks_ranked_by_worsening"][:6]:
        print(
            f"    block{row['block_idx']:02d}  "
            f"share={row['mean_worsening_share']:.3f}  "
            f"mean_delta={row['mean_total_delta']:+.4f}  "
            f"first_drop={row['first_major_drop_count']}"
        )

    print("\n  Aggregate block deltas:")
    for row in report["aggregate_blocks"]:
        print(
            f"    block{row['block_idx']:02d}  "
            f"attn={row['mean_attention_delta']:+.4f}  "
            f"mlp={row['mean_mlp_delta']:+.4f}  "
            f"total={row['mean_total_delta']:+.4f}  "
            f"worst={row['most_common_worst_node'] or 'n/a'}"
        )

    print("\n  Per-image dominant worsening blocks:")
    for item in report["per_image"]:
        dominant = sorted(
            item["blocks"],
            key=lambda row: (row["worsening_share"], row["loss_increase"]),
            reverse=True,
        )[:3]
        bits = [
            f"b{row['block_idx']} share={row['worsening_share']:.2f} "
            f"delta={row['total_delta']:+.4f}"
            for row in dominant
            if row["loss_increase"] > 0
        ]
        if not bits:
            bits = ["no net worsening blocks"]
        print(
            f"    id={item['img_id']:<6} final_cos={item['final_cosine']:.4f} "
            f"top1={'match' if item['top1_match'] else 'mismatch'}  "
            + " | ".join(bits)
        )


def collect_runtime_late_attention_report(
    model,
    state_dict,
    processor,
    calibration_images,
    calibration_image_ids,
    target_images,
    head_specs,
    compile_kwargs=None,
):
    """Compile baseline/variant programs and compare traced late attention tensors."""
    compile_kwargs = dict(compile_kwargs or {})
    baseline_program, baseline_scales = compile_model(
        model,
        state_dict,
        calibration_images,
        processor,
        **compile_kwargs,
    )
    variant_kwargs = dict(compile_kwargs)
    variant_kwargs["requant_pc_qkv"] = True
    variant_program, variant_scales = compile_model(
        model,
        state_dict,
        calibration_images,
        processor,
        **variant_kwargs,
    )
    baseline_embed_scale = baseline_scales.get("pos_embed_add", 14.0 / 127.0)
    variant_embed_scale = variant_scales.get("pos_embed_add", 14.0 / 127.0)

    trace_nodes = set()
    for block_idx, head_idx in head_specs:
        b = f"block{block_idx}_head{head_idx}"
        trace_nodes.update({
            f"{b}_value",
            f"{b}_qkt",
            f"{b}_softmax",
            f"{b}_attn_v",
        })

    per_image = []
    head_buckets = {}

    for img_id, img in target_images:
        _, fp32_traces = fp32_trace(model, processor, img)

        patches_int8_base, cls_int8_base, _ = patch_embed_int8(
            model,
            processor,
            img,
            baseline_embed_scale,
        )
        _, _, _, baseline_trace = golden_inference(
            baseline_program,
            patches_int8_base,
            cls_int8=cls_int8_base,
            trace_nodes=trace_nodes,
        )

        patches_int8_var, cls_int8_var, _ = patch_embed_int8(
            model,
            processor,
            img,
            variant_embed_scale,
        )
        _, _, _, variant_trace = golden_inference(
            variant_program,
            patches_int8_var,
            cls_int8=cls_int8_var,
            trace_nodes=trace_nodes,
        )

        base_tensors = (baseline_trace or {}).get("tensors", {})
        var_tensors = (variant_trace or {}).get("tensors", {})
        image_heads = []

        for block_idx, head_idx in sorted(head_specs):
            prefix = f"block{block_idx}_head{head_idx}"
            report = compare_attention_runtime_path(
                fp32_traces[f"{prefix}_value"],
                fp32_traces[f"{prefix}_qkt"],
                fp32_traces[f"{prefix}_softmax"],
                fp32_traces[f"{prefix}_attn_v"],
                base_tensors[f"{prefix}_value"],
                base_tensors[f"{prefix}_qkt"],
                base_tensors[f"{prefix}_softmax"],
                base_tensors[f"{prefix}_attn_v"],
                var_tensors[f"{prefix}_value"],
                var_tensors[f"{prefix}_qkt"],
                var_tensors[f"{prefix}_softmax"],
                var_tensors[f"{prefix}_attn_v"],
            )
            report["block_idx"] = block_idx
            report["head_idx"] = head_idx
            image_heads.append(report)

            bucket = head_buckets.setdefault((block_idx, head_idx), {
                "image_ids": [],
                "value_delta": [],
                "qkt_delta": [],
                "softmax_delta": [],
                "attn_v_delta": [],
                "replay_delta": [],
                "runtime_vs_replay_delta": [],
                "variant_softmax_only": [],
                "variant_value_only": [],
            })
            bucket["image_ids"].append(int(img_id))
            bucket["value_delta"].append(report["deltas"]["value_delta_cosine"])
            bucket["qkt_delta"].append(report["deltas"]["qkt_delta_cosine"])
            bucket["softmax_delta"].append(report["deltas"]["softmax_delta_cosine"])
            bucket["attn_v_delta"].append(report["deltas"]["attn_v_delta_cosine"])
            bucket["replay_delta"].append(report["deltas"]["replay_to_fp32_delta_cosine"])
            bucket["runtime_vs_replay_delta"].append(report["deltas"]["runtime_vs_replay_delta_cosine"])
            bucket["variant_softmax_only"].append(
                report["variant_replay_isolation"]["softmax_only_to_fp32_metrics"]["cosine_sim"]
            )
            bucket["variant_value_only"].append(
                report["variant_replay_isolation"]["value_only_to_fp32_metrics"]["cosine_sim"]
            )

        per_image.append({
            "img_id": int(img_id),
            "heads": image_heads,
        })

    aggregate = []
    for (block_idx, head_idx), bucket in sorted(head_buckets.items()):
        aggregate.append({
            "block_idx": block_idx,
            "head_idx": head_idx,
            "image_ids": bucket["image_ids"],
            "mean_value_delta_cosine": float(np.mean(bucket["value_delta"])),
            "mean_qkt_delta_cosine": float(np.mean(bucket["qkt_delta"])),
            "mean_softmax_delta_cosine": float(np.mean(bucket["softmax_delta"])),
            "mean_attn_v_delta_cosine": float(np.mean(bucket["attn_v_delta"])),
            "mean_replay_to_fp32_delta_cosine": float(np.mean(bucket["replay_delta"])),
            "mean_runtime_vs_replay_delta_cosine": float(np.mean(bucket["runtime_vs_replay_delta"])),
            "mean_variant_softmax_only_cosine": float(np.mean(bucket["variant_softmax_only"])),
            "mean_variant_value_only_cosine": float(np.mean(bucket["variant_value_only"])),
        })

    aggregate.sort(key=lambda row: row["mean_attn_v_delta_cosine"])
    return {
        "target_image_ids": [int(img_id) for img_id, _ in target_images],
        "calibration_image_ids": [int(img_id) for img_id in calibration_image_ids],
        "heads": aggregate,
        "per_image": per_image,
    }


def print_runtime_late_attention_report(report):
    """Print runtime baseline-vs-variant late attention head comparisons."""
    print("\n" + "=" * 90)
    print("  Runtime Late Attention Report")
    print("=" * 90)
    print(f"\n  Target images      : {report['target_image_ids']}")
    print(f"  Calibration images : {report['calibration_image_ids']}")
    print("\n  Worst heads by attn_v delta:")
    for row in report["heads"][:8]:
        print(
            f"    block{row['block_idx']:02d} h{row['head_idx']}  "
            f"value={row['mean_value_delta_cosine']:+.4f}  "
            f"qkt={row['mean_qkt_delta_cosine']:+.4f}  "
            f"softmax={row['mean_softmax_delta_cosine']:+.4f}  "
            f"attn_v={row['mean_attn_v_delta_cosine']:+.4f}  "
            f"replay={row['mean_replay_to_fp32_delta_cosine']:+.4f}  "
            f"runtime-vs-replay={row['mean_runtime_vs_replay_delta_cosine']:+.4f}"
        )
        print(
            f"      variant softmax-only={row['mean_variant_softmax_only_cosine']:.4f}  "
            f"variant value-only={row['mean_variant_value_only_cosine']:.4f}"
        )


def _load_diag_images(image_ids, label: str, image_root: str, source: str):
    if source == "local":
        return load_local_images(image_ids, label, image_root=image_root)
    return collect_images(image_ids, label)


def _resolve_diagnostic_preset(args):
    if not getattr(args, "diagnostic_preset", ""):
        return None
    return get_diagnostic_preset(args.diagnostic_preset)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase A diagnostics for INT8 accuracy")
    parser.add_argument(
        "--diagnostic-preset",
        choices=sorted(DIAGNOSTIC_PRESETS),
        default="",
        help="Apply a canonical diagnostics preset for local compile/run modes",
    )
    parser.add_argument(
        "--trace-json-a",
        default="",
        help="Baseline compare_golden JSON with trace_report (used by --trace-diff-report)",
    )
    parser.add_argument(
        "--trace-json-b",
        default="",
        help="Variant compare_golden JSON with trace_report (used by --trace-diff-report)",
    )
    parser.add_argument(
        "--trace-diff-report",
        action="store_true",
        help="Compare two saved trace JSONs on common traced images",
    )
    parser.add_argument(
        "--trace-diff-blocks",
        default="6,8,10,11",
        help="Comma-separated block indices for --trace-diff-report",
    )
    parser.add_argument(
        "--trace-diff-output",
        default="",
        help="Optional JSON output path for --trace-diff-report",
    )
    parser.add_argument(
        "--late-attn-path-report",
        action="store_true",
        help="Compare baseline vs variant late attention paths grouped by (block, head)",
    )
    parser.add_argument(
        "--late-attn-blocks",
        default="8,9,10,11",
        help="Comma-separated block indices for --late-attn-path-report",
    )
    parser.add_argument(
        "--late-attn-output",
        default="",
        help="Optional JSON output path for --late-attn-path-report",
    )
    parser.add_argument(
        "--block-impact-report",
        action="store_true",
        help="Summarize per-image block-level impact from one saved compare_golden trace JSON",
    )
    parser.add_argument(
        "--block-impact-output",
        default="",
        help="Optional JSON output path for --block-impact-report",
    )
    parser.add_argument(
        "--runtime-late-attn-report",
        action="store_true",
        help="Compile baseline and REQUANT_PC QKV variants and compare runtime late attention tensors",
    )
    parser.add_argument(
        "--runtime-heads",
        default="11:1,11:2",
        help="Comma-separated block:head pairs for --runtime-late-attn-report",
    )
    parser.add_argument(
        "--runtime-image-ids",
        default="2006,5037",
        help="Comma-separated COCO ids traced by --runtime-late-attn-report",
    )
    parser.add_argument(
        "--runtime-calibration-images",
        type=int,
        default=20,
        help="Number of frozen eval ids used for calibration in --runtime-late-attn-report",
    )
    parser.add_argument(
        "--runtime-output",
        default="",
        help="Optional JSON output path for --runtime-late-attn-report",
    )
    parser.add_argument("--max-images", type=int, default=5,
                        help="Number of images to use (default: 5 for speed)")
    parser.add_argument(
        "--qkv-requant-pc-report",
        action="store_true",
        help="Compare local Q/K/V head outputs for baseline vs REQUANT_PC quantization",
    )
    parser.add_argument(
        "--qkv-blocks",
        default="",
        help="Comma-separated block indices used by --qkv-requant-pc-report (default: all blocks)",
    )
    parser.add_argument(
        "--qkv-output",
        default="",
        help="Optional JSON output path for the local Q/K/V REQUANT_PC report",
    )
    args = parser.parse_args()
    preset = _resolve_diagnostic_preset(args)

    if args.trace_diff_report:
        if not args.trace_json_a or not args.trace_json_b:
            parser.error("--trace-diff-report requires --trace-json-a and --trace-json-b")
        with open(args.trace_json_a, "r") as f:
            baseline_json = json.load(f)
        with open(args.trace_json_b, "r") as f:
            variant_json = json.load(f)
        blocks = _parse_int_set(args.trace_diff_blocks)
        report = summarize_trace_variant_delta(
            baseline_json.get("trace_report", {}),
            variant_json.get("trace_report", {}),
            blocks=blocks,
        )
        print_trace_variant_delta_report(report)
        if args.trace_diff_output:
            with open(args.trace_diff_output, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\n  Saved trace diff report to {args.trace_diff_output}")
        return

    if args.late_attn_path_report:
        if not args.trace_json_a or not args.trace_json_b:
            parser.error("--late-attn-path-report requires --trace-json-a and --trace-json-b")
        with open(args.trace_json_a, "r") as f:
            baseline_json = json.load(f)
        with open(args.trace_json_b, "r") as f:
            variant_json = json.load(f)
        blocks = _parse_int_set(args.late_attn_blocks)
        report = summarize_late_attention_path_delta(
            baseline_json.get("trace_report", {}),
            variant_json.get("trace_report", {}),
            blocks=blocks,
        )
        print_late_attention_path_delta_report(report)
        if args.late_attn_output:
            with open(args.late_attn_output, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\n  Saved late attention path report to {args.late_attn_output}")
        return

    if args.block_impact_report:
        if not args.trace_json_a:
            parser.error("--block-impact-report requires --trace-json-a")
        with open(args.trace_json_a, "r") as f:
            compare_json = json.load(f)
        report = summarize_block_impact(compare_json)
        print_block_impact_report(report)
        if args.block_impact_output:
            with open(args.block_impact_output, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\n  Saved block impact report to {args.block_impact_output}")
        return

    if args.runtime_late_attn_report:
        head_specs = _parse_block_head_set(args.runtime_heads)
        if not head_specs:
            parser.error("--runtime-late-attn-report requires at least one --runtime-heads entry")
        image_ids = [int(part) for part in args.runtime_image_ids.split(",") if part.strip()]
        if preset is not None:
            calibration_ids = preset["benchmark"]["calibration_image_ids"]
            image_source = preset["benchmark"]["benchmark_image_source"]
            image_root = preset["benchmark"]["local_benchmark_image_dir"]
            runtime_compile_kwargs = preset_compile_kwargs(preset)
        else:
            calibration_ids = FROZEN_EVAL_IMAGE_IDS[:args.runtime_calibration_images]
            image_source = "download"
            image_root = LOCAL_FROZEN_IMAGE_DIR
            runtime_compile_kwargs = {}

        print("=" * 60)
        print("  TACCEL Accuracy Diagnostics — Runtime Late Attention")
        print("=" * 60)

        print("\n[1/4] Loading model...")
        model, state_dict = load_model()
        processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        print(f"  Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

        print(f"\n[2/4] Collecting {len(calibration_ids)} calibration images...")
        calibration_pairs = _load_diag_images(
            calibration_ids,
            "runtime calibration",
            image_root=image_root,
            source=image_source,
        )
        calibration_images = [img for _, img in calibration_pairs]
        print(f"  Got {len(calibration_images)} calibration images")

        print(f"\n[3/4] Collecting {len(image_ids)} target images...")
        target_images = _load_diag_images(
            image_ids,
            "runtime target",
            image_root=image_root,
            source=image_source,
        )
        print(f"  Got {len(target_images)} target images")

        print("\n[4/4] Comparing runtime attention paths...")
        report = collect_runtime_late_attention_report(
            model,
            state_dict,
            processor,
            calibration_images,
            calibration_ids,
            target_images,
            head_specs,
            compile_kwargs=runtime_compile_kwargs,
        )
        report["diagnostic_preset"] = args.diagnostic_preset or None
        print_runtime_late_attention_report(report)
        if args.runtime_output:
            with open(args.runtime_output, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\n  Saved runtime late attention report to {args.runtime_output}")
        return

    print("=" * 60)
    print("  TACCEL Accuracy Diagnostics — Phase A")
    print("=" * 60)

    print("\n[1/5] Loading model...")
    model, state_dict = load_model()
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    print(f"  Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

    if preset is not None:
        image_ids = preset["benchmark"]["eval_image_ids"][:args.max_images]
        image_source = preset["benchmark"]["benchmark_image_source"]
        image_root = preset["benchmark"]["local_benchmark_image_dir"]
        compile_kwargs = preset_compile_kwargs(preset)
    else:
        image_ids = FROZEN_EVAL_IMAGE_IDS[:args.max_images]
        image_source = "download"
        image_root = LOCAL_FROZEN_IMAGE_DIR
        compile_kwargs = {}
    print(f"\n[2/5] Collecting {len(image_ids)} frozen evaluation images...")
    images = _load_diag_images(image_ids, "diagnostic", image_root=image_root, source=image_source)
    if not images:
        print("  No images downloaded.")
        sys.exit(1)
    print(f"  Got {len(images)} images")

    print("\n[3/5] Compiling INT8 program (calibration from collected images)...")
    sample_imgs = [img for _, img in images]
    program, cal_scales = compile_model(model, state_dict, sample_imgs, processor, **compile_kwargs)
    block_input_scale = cal_scales.get("pos_embed_add", 14.0 / 127.0)
    embed_scale = block_input_scale
    print(f"  block_input_scale (pos_embed_add) = {block_input_scale:.5f}")

    print("\n[4/5] Collecting FP32 activation statistics (hooks)...")
    res1, res2, qkt = collect_fp32_stats(model, processor, images)
    print("  Done.")

    print("\n[5/5] Running golden model with VADD saturation instrumentation...")
    vadd_stats_per_image = []
    for i, (img_id, img) in enumerate(images, 1):
        patches_int8, cls_int8, _ = patch_embed_int8(model, processor, img, embed_scale)
        _, vstats = run_golden_with_vadd_stats(program, patches_int8, cls_int8=cls_int8)
        vadd_stats_per_image.append(vstats)
        print(f"  Image {i}/{len(images)}: {len(vstats)} INT8 VADD calls recorded")

    print_report(res1, res2, qkt, vadd_stats_per_image, block_input_scale)

    if args.qkv_requant_pc_report:
        blocks = (
            {int(part) for part in args.qkv_blocks.split(",") if part.strip()}
            if args.qkv_blocks else None
        )
        print("\n[6/6] Analysing local Q/K/V REQUANT_PC behavior...")
        qkv_report = collect_qkv_requant_pc_stats(model, processor, images, cal_scales, blocks=blocks)
        print_qkv_requant_pc_report(qkv_report)
        if args.qkv_output:
            with open(args.qkv_output, "w") as f:
                json.dump(qkv_report, f, indent=2)
            print(f"\n  Saved Q/K/V report to {args.qkv_output}")


if __name__ == "__main__":
    main()
