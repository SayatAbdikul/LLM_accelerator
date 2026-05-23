"""W8A8 like-for-like NumPy reference — Phase 0 PPL gate.

Sibling of `NanoGPTW8A16SimulatorReference` that additionally models the
**inter-op INT8 storage step** introduced by the W8A8 codegen path
(reserved opcode 0x1C, `DEQUANT_INT8_FP32`).

Semantic difference vs. W8A16
-----------------------------

W8A16 (production today):
  - Activations stored as FP16 between ops (ABUF holds 2 bytes/elem).
  - Each matmul *already* INT8-quantizes its input dynamically
    (`x_int8 = clip(round(x * inv_fp16), -128, 127)` — see W8A16 ref line 32).
  - SFU ops (LayerNorm, GELU, softmax, VADD) read FP16 from ABUF, compute
    in FP32, write FP16 to ABUF.

W8A8 (this plan, Phase 0 fake-quant model):
  - Activations stored as INT8 between ops (ABUF holds 1 byte/elem + per-tile scale).
  - Matmul still INT8-quantizes dynamically (same as W8A16).
  - SFU ops still compute FP32-internal (frozen contract); we model this
    by quantizing the SFU output to INT8 with the per-tensor calibrated
    scale, then immediately dequantizing back to FP32 for downstream
    consumers. This is exactly the round-trip the new `0x1C` opcode
    introduces in front of each SFU consumer.

The PPL delta vs W8A16 is therefore the cumulative noise of ~16 inter-op
INT8 round-trips per transformer block × N_layer blocks. With QuaRot
rotation (which Gaussianizes the activation distribution and removes
heavy tails), this noise is expected to be small (≤+5 PPL on GPT-2
124M, predicted; Phase 0 measures the actual number).

Implementation
--------------

Subclasses `NanoGPTW8A16SimulatorReference` and overrides only the two
methods that hold the forward pass (`_attention_head`, `run_decode_step`).
The constructor is unchanged. The W8A16 baseline therefore stays
byte-identical (no parent-class modification).

Inter-op storage points that get the INT8 round-trip (matching the
W8A8 codegen emission points in `compiler/emit/{sfu,kv,embedding}.py`):

  - tok_pos_add (embedding sum)
  - blockN_ln1, blockN_ln2 (LayerNorm outputs)
  - blockN_headH_query/key/value (matmul outputs feeding QKT/attn_v)
  - blockN_headH_softmax (softmax output feeding attn_v)
  - blockN_headH_attn_v (attn_v matmul output feeding concat)
  - blockN_concat (concat-heads output feeding c_proj)
  - blockN_out_proj (c_proj output feeding residual1)
  - blockN_residual1 (VADD output)
  - blockN_fc1 (matmul output feeding GELU)
  - blockN_gelu (GELU output feeding fc2)
  - blockN_fc2 (matmul output feeding residual2)
  - blockN_residual2 (VADD output)
  - ln_f (final LayerNorm output feeding lm_head)

  NOT quantized at this layer: lm_head logits (final output, no
  downstream consumer in inference; PPL is computed on FP32 logits).

Fallback for missing scales: `default_act_scale = 6.0 / 127.0`
(inherited from W8A16 ref).

NOT bit-exact against torch — same caveat as parent class.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from ._ref_ops import cast_fp16, gelu_tanh, layernorm, softmax_masked
from .w8a16_simulator_reference import (
    NanoGPTW8A16SimulatorReference,
    _w8a16_dynamic_matmul,
)


def _inter_op_int8_round_trip(x_fp32: np.ndarray, scale: float) -> np.ndarray:
    """INT8 quant-then-dequant with per-tensor static scale.

    Models the W8A8 inter-op storage step:
      x_int8 = clip(round(x_fp32 / scale), -128, 127).astype(int8)   # QUANT_FP32_INT8 emit
      x_back = x_int8.astype(fp32) * scale                            # DEQUANT_INT8_FP32 (0x1C) emit

    The intermediate INT8 storage is exactly what ABUF would hold in
    W8A8 mode. Return dtype is FP32 (next consumer's input type).
    """
    s = max(float(scale), 1e-12)
    x_q = np.clip(np.round(x_fp32 / s), -128, 127).astype(np.int8)
    return x_q.astype(np.float32) * np.float32(s)


class NanoGPTW8A8SimulatorReference(NanoGPTW8A16SimulatorReference):
    """W8A8 NumPy reference modeling inter-op INT8 storage.

    Inherits constructor, weight quantization, and KV-cache machinery
    from `NanoGPTW8A16SimulatorReference`. Overrides only the forward
    pass to inject the INT8 round-trip at each inter-op storage point.

    Usage::

        ref = NanoGPTW8A8SimulatorReference(payload, calibration_scales=scales)
        logits = ref.run_teacher_forced(token_ids)   # List[FP32 [vocab]]
    """

    def _q_inter(self, x_fp32: np.ndarray, name: str) -> np.ndarray:
        """Apply the inter-op INT8 round-trip with per-tensor calibrated scale."""
        scale = self._act_scale_for(name)
        return _inter_op_int8_round_trip(x_fp32, scale)

    def _attention_head(self, ln1: np.ndarray, layer_idx: int, head_idx: int,
                        position: int, record=None) -> np.ndarray:
        head = self.layers[layer_idx]["heads"][head_idx]
        # Per-head Q/K/V matmul outputs. W8A8: store as INT8, downstream
        # consumers (QKT, attn_v, KV cache) re-read via DEQUANT_INT8_FP32.
        q = _w8a16_dynamic_matmul(
            ln1, head["q_int8"], head["q_scales"], head["q_b"],
        )
        k = _w8a16_dynamic_matmul(
            ln1, head["k_int8"], head["k_scales"], head["k_b"],
        )
        v = _w8a16_dynamic_matmul(
            ln1, head["v_int8"], head["v_scales"], head["v_b"],
        )
        q = self._q_inter(q, f"block{layer_idx}_head{head_idx}_query")
        k = self._q_inter(k, f"block{layer_idx}_head{head_idx}_key")
        v = self._q_inter(v, f"block{layer_idx}_head{head_idx}_value")
        if record is not None:
            record(f"block{layer_idx}_head{head_idx}_query", q)
            record(f"block{layer_idx}_head{head_idx}_key", k)
            record(f"block{layer_idx}_head{head_idx}_value", v)
        # KV cache: store the inter-op INT8 round-trip (matches what
        # the W8A8 codegen would write to the INT8 KV ABUF tile).
        tq = self.kv_quant
        if tq is not None and tq.quant_k:
            k_store = tq.round_trip(k[0])
        else:
            k_store = cast_fp16(k[0])
        if tq is not None and tq.quant_v:
            v_store = tq.round_trip(v[0])
        else:
            v_store = cast_fp16(v[0])
        self._caches[layer_idx][head_idx]["k"].append(k_store)
        self._caches[layer_idx][head_idx]["v"].append(v_store)
        if record is not None and tq is not None:
            record(f"block{layer_idx}_head{head_idx}_k_cache_tq", k_store)
            record(f"block{layer_idx}_head{head_idx}_v_cache_tq", v_store)
        k_cache = np.stack(self._caches[layer_idx][head_idx]["k"], axis=0)
        v_cache = np.stack(self._caches[layer_idx][head_idx]["v"], axis=0)

        # QKT (static composite scale; INT8 quant is dynamic per-tile inside).
        q_scale = self._act_scale_for(f"block{layer_idx}_head{head_idx}_query")
        k_scale = self._act_scale_for(f"block{layer_idx}_head{head_idx}_key")
        composite_qkt = np.float16(
            np.float32(q_scale) * np.float32(k_scale) * np.float32(self.inv_sqrt_d_head)
        )
        q_int8 = self._qkt_attn_v_quantize(cast_fp16(q), q_scale)
        k_int8 = self._qkt_attn_v_quantize(cast_fp16(k_cache), k_scale)
        scores_int32 = q_int8.astype(np.int32) @ k_int8.astype(np.int32).T
        scores_fp32 = scores_int32.astype(np.float32) * np.float32(composite_qkt)
        # W8A8: QKT output (scores) feeds softmax — apply inter-op INT8 round-trip
        # at QKT output. Use a per-head 'qkt' scale; fall back to default if missing.
        scores_fp32 = cast_fp16(scores_fp32)
        scores_fp32 = self._q_inter(
            scores_fp32, f"block{layer_idx}_head{head_idx}_qkt",
        )

        # Masked softmax with valid_kv_len = position + 1.
        valid_kv_len = position + 1
        probs_fp32 = softmax_masked(
            scores_fp32, valid_kv_len, fp16_storage=True,
        )
        # W8A8: softmax output → attn_v input via inter-op INT8 round-trip.
        probs_fp32 = self._q_inter(
            probs_fp32, f"block{layer_idx}_head{head_idx}_softmax",
        )
        if record is not None:
            record(f"block{layer_idx}_head{head_idx}_softmax", probs_fp32)

        # attn_v.
        sm_scale = self._act_scale_for(f"block{layer_idx}_head{head_idx}_softmax")
        v_scale = self._act_scale_for(f"block{layer_idx}_head{head_idx}_value")
        composite_av = np.float16(np.float32(sm_scale) * np.float32(v_scale))
        sm_int8 = self._qkt_attn_v_quantize(probs_fp32, sm_scale)
        v_cache_int8 = self._qkt_attn_v_quantize(
            cast_fp16(v_cache), v_scale,
        )
        head_out_int32 = sm_int8.astype(np.int32) @ v_cache_int8.astype(np.int32)
        head_out_fp32 = head_out_int32.astype(np.float32) * np.float32(composite_av)
        head_out = cast_fp16(head_out_fp32)
        # W8A8: attn_v output feeds concat — inter-op INT8 round-trip per head.
        head_out = self._q_inter(
            head_out, f"block{layer_idx}_head{head_idx}_attn_v",
        )
        if record is not None:
            record(f"block{layer_idx}_head{head_idx}_attn_v", head_out)
        return head_out

    def run_decode_step(self, token_id: int, position: int,
                        trace: Optional[dict] = None) -> np.ndarray:
        """W8A8 decode step. Same node sequence as parent; injects INT8
        inter-op round-trip at every SFU-output storage point."""
        if position != self._next_position:
            raise ValueError(
                f"NanoGPTW8A8SimulatorReference: position {position} doesn't "
                f"match internal cursor {self._next_position}"
            )

        def record(name: str, value) -> None:
            if trace is None:
                return
            trace[name] = {"value": np.asarray(value, dtype=np.float32).copy()}

        # Embedding sum → stored as INT8 in W8A8 mode (tok+pos VADD output).
        x = self.wte[token_id:token_id + 1] + self.wpe[position:position + 1]
        x = self._q_inter(cast_fp16(x), "tok_pos_add")
        record("tok_pos_add", x)

        for layer_idx, layer in enumerate(self.layers):
            ln1 = layernorm(
                x, layer["ln1_w"], layer["ln1_b"],
                eps=self.layer_norm_epsilon, fp16_storage=True,
            )
            # LN1 output → INT8 storage (consumed by Q/K/V matmuls).
            ln1 = self._q_inter(ln1, f"block{layer_idx}_ln1")
            record(f"block{layer_idx}_ln1", ln1)

            head_outs = []
            for head_idx in range(self.n_head):
                head_outs.append(
                    self._attention_head(
                        ln1, layer_idx, head_idx, position, record=record
                    )
                )
            concat = np.concatenate(head_outs, axis=-1)
            # concat output → INT8 storage (consumed by c_proj matmul).
            concat = self._q_inter(concat, f"block{layer_idx}_concat")
            record(f"block{layer_idx}_concat", concat)
            out_proj = _w8a16_dynamic_matmul(
                concat, layer["c_proj_int8"], layer["c_proj_scales"], layer["c_proj_b"],
            )
            # c_proj output → INT8 storage (consumed by residual1 VADD).
            out_proj = self._q_inter(out_proj, f"block{layer_idx}_out_proj")
            record(f"block{layer_idx}_out_proj", out_proj)
            # residual1: VADD of two INT8 tensors via FP32 internal then INT8 quant.
            x = cast_fp16(x + out_proj)
            x = self._q_inter(x, f"block{layer_idx}_residual1")
            record(f"block{layer_idx}_residual1", x)

            ln2 = layernorm(
                x, layer["ln2_w"], layer["ln2_b"],
                eps=self.layer_norm_epsilon, fp16_storage=True,
            )
            ln2 = self._q_inter(ln2, f"block{layer_idx}_ln2")
            record(f"block{layer_idx}_ln2", ln2)
            fc1 = _w8a16_dynamic_matmul(
                ln2, layer["fc1_int8"], layer["fc1_scales"], layer["fc1_b"],
            )
            fc1 = self._q_inter(fc1, f"block{layer_idx}_fc1")
            record(f"block{layer_idx}_fc1", fc1)
            gelu = gelu_tanh(fc1, fp16_storage=True)
            gelu = self._q_inter(gelu, f"block{layer_idx}_gelu")
            record(f"block{layer_idx}_gelu", gelu)
            fc2 = _w8a16_dynamic_matmul(
                gelu, layer["fc2_int8"], layer["fc2_scales"], layer["fc2_b"],
            )
            fc2 = self._q_inter(fc2, f"block{layer_idx}_fc2")
            record(f"block{layer_idx}_fc2", fc2)
            x = cast_fp16(x + fc2)
            x = self._q_inter(x, f"block{layer_idx}_residual2")
            record(f"block{layer_idx}_residual2", x)

        ln_f = layernorm(
            x, self.ln_f_w, self.ln_f_b,
            eps=self.layer_norm_epsilon, fp16_storage=True,
        )
        # Final LN output → INT8 storage (consumed by lm_head matmul).
        ln_f = self._q_inter(ln_f, "ln_f")
        record("ln_f", ln_f)
        # lm_head: matmul; logits are FP32 (no downstream consumer).
        logits = _w8a16_dynamic_matmul(
            ln_f[-1:], self.lm_head_w_int8, self.lm_head_w_scales, self.lm_head_b,
        )
        record("lm_head", logits)
        self._next_position += 1
        return logits[0].astype(np.float32)
