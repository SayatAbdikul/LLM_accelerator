"""W8A32 like-for-like reference — M4-E.

Companion to `WeightOnlyHostRunner`, but with the *same* per-matmul
INT8-activation re-quantization semantics that the simulator-backed
W8A32 codegen produces (`emit_matmul_w8a32` / `emit_matmul_qkt_w8a32` /
`emit_matmul_attn_v_w8a32`). Where `WeightOnlyHostRunner` produces the
**FP32-with-INT8-weight-QDQ** ceiling (~53.42 PPL on real GPT-2),
`NanoGPTW8A32SimulatorReference` produces the **dynamic-INT8-activation
+ INT8 weight** number that should bit-match the simulator-backed
bundle (modulo runtime FP16 ULP).

Why this exists
---------------

Without a like-for-like reference, the simulator-backed perplexity
number is uninterpretable: any difference from `WeightOnlyHostRunner`
could be either a codegen bug or the expected M2.5-A dynamic-scaling
compounding cost. M4-E lets us split those: "M4-E vs simulator-backed
within ULP" is the codegen-correctness gate; "M4-E vs
WeightOnlyHostRunner" is the design-cost telemetry (probably a future
M5 calibration milestone).

Per-matmul semantics
--------------------

For node N with input `x` (FP32) and weight `w` (INT8, per-channel FP16
scales `w_scales`):

  max_abs = max(|x|)                       # over the full activation tile
  inv_fp16 = np.float16(127 / max_abs)     # MAX_ABS_REDUCE writes this to S[s]
  fwd_fp16 = np.float16(max_abs / 127)     # ...and this to S[s+1]
  x_int8  = clip(round(x * float(inv_fp16)), -128, 127).astype(int8)
  accum32 = x_int8.astype(int32) @ w.astype(int32)
  y_fp32  = accum32 * w_scales.astype(np.float16).astype(np.float32) * float(fwd_fp16)
          + bias_fp32

QKT path (matches `emit_matmul_qkt_w8a32`):

  q_scale, k_scale: static calibration (defaults 6/127 if absent)
  composite_fp16  = np.float16(q_scale * k_scale * inv_sqrt_d_head)
  q_int8 = clip(round(q * 127/q_max_abs)…) but with STATIC scale instead of
           dynamic: q_int8 = clip(round(q / q_scale), -128, 127).astype(int8)
  k_int8 = clip(round(k / k_scale), -128, 127).astype(int8)
  scores_int32 = q_int8 @ k_int8.T
  scores_fp32  = scores_int32 * float(composite_fp16)

attn_v path: similar with `composite_fp16 = sm_scale * v_scale`.

Pad-row zero-fill (M3-C): K and V pad rows (rows >= valid_kv_len) are
zeroed before quantization. Mirrors codegen behavior.

This reference uses NumPy (no torch) and stays minimal so M4-G can
iterate quickly. It is NOT bit-exact against torch (different floating-
point evaluation order). The simulator-backed bundle is the ground
truth that this reference targets within FP16 ULP.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


def _pad_to_multiple(arr: np.ndarray, axis: int, multiple: int = 16) -> np.ndarray:
    """Right-pad `arr` along `axis` to a multiple of `multiple` with zeros."""
    n = arr.shape[axis]
    pad = (-n) % multiple
    if pad == 0:
        return arr
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (0, pad)
    return np.pad(arr, pad_width, mode="constant", constant_values=0)


def _quant_w_per_channel_int8(w_fp32: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize FP32 weight `[K, N]` to INT8 with per-column FP16 scales.

    Mirrors `_torch_qdq_per_channel` math but returns the raw INT8 +
    scales (no QDQ back to FP32). The simulator-backed codegen consumes
    these directly via `weight_data`.
    """
    if w_fp32.ndim != 2:
        raise ValueError(f"expected 2-D weight, got shape {w_fp32.shape}")
    max_per_col = np.maximum(np.max(np.abs(w_fp32), axis=0), 1e-8)
    w_scales = (max_per_col / 127.0).astype(np.float16)
    w_int8 = np.clip(
        np.round(w_fp32 / w_scales.astype(np.float32).reshape(1, -1)),
        -128, 127,
    ).astype(np.int8)
    return w_int8, w_scales


def _w8a32_dynamic_matmul(
    x_fp32: np.ndarray,
    w_int8: np.ndarray,
    w_scales_fp16: np.ndarray,
    bias_fp32: Optional[np.ndarray] = None,
    fp_precision: str = "fp32",
) -> np.ndarray:
    """One INT8-activation × INT8-weight matmul with FP{32,16} dequant.

    Matches the codegen's `emit_matmul_w8a32` semantics exactly:
      - max_abs over the full activation tile (one MAX_ABS_REDUCE). Under
        fp_precision='fp16' the source is FP16-widened to FP32 first.
      - FP16 inv_scale, FP16 fwd_scale (DEQUANT_ACCUM_FP32_SCALED sreg+1).
      - INT8 round-half-to-even (numpy default → matches QUANT_FP32_INT8).
      - Per-column FP16 weight scales applied after the INT32 accum.
      - Bias added in FP32 BEFORE FP16 cast (bias-fold contract).
      - Output: FP32 (fp_precision='fp32') or FP16-then-widen
        (fp_precision='fp16', matches simulator's FP16-store path).
    """
    # Under fp_precision="fp16" the source tile in the simulator is FP16-
    # widened on read. Mirror that here so max_abs is computed over the
    # same precision-rounded values.
    if fp_precision == "fp16":
        x_fp32 = x_fp32.astype(np.float16).astype(np.float32)
    max_abs = float(np.max(np.abs(x_fp32))) if x_fp32.size else 0.0
    # Eps clamp: matches the simulator's 2**-9 floor (avoids inf/NaN
    # when activations are all zero; FP16 max is ~65504).
    max_abs_eps = max(max_abs, 2.0 ** -9)
    inv_fp16 = np.float16(127.0 / max_abs_eps)
    fwd_fp16 = np.float16(max_abs_eps / 127.0)
    x_int8 = np.clip(
        np.round(x_fp32 * np.float32(inv_fp16)),
        -128, 127,
    ).astype(np.int8)
    # INT32 matmul (cast to widen avoids overflow on K=3072 worst case
    # of K * 127 * 127 = 49.5 M, well under INT32 range).
    accum32 = x_int8.astype(np.int32) @ w_int8.astype(np.int32)
    # Dequant: per-column FP16 weight scale × FP32(fwd_fp16) × INT32 accum.
    pc_fp32 = w_scales_fp16.astype(np.float32).reshape(1, -1)
    y_fp32 = accum32.astype(np.float32) * pc_fp32 * np.float32(fwd_fp16)
    # Bias added BEFORE FP16 cast (matches the simulator's bias-fold
    # contract under flags=1; the W8A32 path's cast-then-add VADD has the
    # same numerical effect under flags=0 because the cast is a no-op).
    if bias_fp32 is not None:
        y_fp32 = y_fp32 + bias_fp32.astype(np.float32)
    if fp_precision == "fp16":
        return y_fp32.astype(np.float16).astype(np.float32)
    return y_fp32.astype(np.float32)


def _maybe_cast_fp16(x: np.ndarray, fp_precision: str) -> np.ndarray:
    """Cast to FP16 and back to FP32 if precision='fp16' (storage round-trip).

    Mirrors `mem.write_fp16_tile` followed by `mem.read_fp16_tile` in the
    simulator: the in-memory storage is FP16, but the data is widened to
    FP32 on next read so the downstream FP32-internal compute path stays
    unchanged. This is a no-op under fp_precision='fp32'.
    """
    if fp_precision == "fp16":
        return x.astype(np.float16).astype(np.float32)
    return x


def _layer_norm_fp32(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                     eps: float = 1e-5, *, fp_precision: str = "fp32") -> np.ndarray:
    """Row-wise FP32 LayerNorm (matches `_exec_layernorm_fp32`).

    Internal mean/variance reduction is always FP32; under fp_precision='fp16'
    the source is FP16-widened on read and the output cast to FP16 on store.
    """
    x = _maybe_cast_fp16(x, fp_precision)
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    out = ((x - mean) / np.sqrt(var + eps)) * gamma.reshape(1, -1) + beta.reshape(1, -1)
    return _maybe_cast_fp16(out, fp_precision)


def _gelu_fp32(x: np.ndarray, *, fp_precision: str = "fp32") -> np.ndarray:
    """gelu_new (tanh approximation, matches the FP32 sub-layer hardware)."""
    x = _maybe_cast_fp16(x, fp_precision)
    out = 0.5 * x * (1.0 + np.tanh(
        np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)
    ))
    return _maybe_cast_fp16(out, fp_precision)


def _softmax_masked_fp32(scores: np.ndarray, valid_kv_len: int,
                         *, fp_precision: str = "fp32") -> np.ndarray:
    """Causal masked softmax along the last axis. `valid_kv_len`
    columns are kept; the rest get -inf before the softmax."""
    scores = _maybe_cast_fp16(scores, fp_precision)
    masked = scores.astype(np.float32).copy()
    if valid_kv_len < masked.shape[-1]:
        masked[..., valid_kv_len:] = -np.inf
    masked -= np.max(masked, axis=-1, keepdims=True)
    exp = np.exp(masked)
    out = exp / np.sum(exp, axis=-1, keepdims=True)
    return _maybe_cast_fp16(out, fp_precision)


class NanoGPTW8A32SimulatorReference:
    """Like-for-like W8A32 reference matching the simulator-backed bundle.

    See module docstring. Constructs INT8 weights + FP16 scales from
    the FP32 state_dict and runs the forward pass with per-matmul
    dynamic activation quantization for non-attention matmuls and
    static composite scales for QKT / attn_v.
    """

    def __init__(self, payload: dict, *, default_act_scale: float = 6.0 / 127.0,
                 calibration_scales: Optional[Dict[str, float]] = None,
                 fp_precision: str = "fp32") -> None:
        if fp_precision not in ("fp32", "fp16"):
            raise ValueError(f"fp_precision must be 'fp32' or 'fp16', got {fp_precision!r}")
        self.fp_precision = fp_precision
        self.payload = payload
        cfg = payload["model_args"]
        sd = payload["state_dict"]
        self.n_layer = int(cfg["n_layer"])
        self.n_head = int(cfg["n_head"])
        self.d_model = int(cfg["n_embd"])
        self.d_head = self.d_model // self.n_head
        self.vocab_size = int(cfg["vocab_size"])
        self.block_size = int(cfg["block_size"])
        self.layer_norm_epsilon = float(cfg.get("layer_norm_epsilon", 1e-5))
        self.inv_sqrt_d_head = self.d_head ** -0.5
        self.default_act_scale = float(default_act_scale)
        self.calibration_scales = dict(calibration_scales or {})

        # Embeddings stay FP32 (the codegen DMAs FP32 token/pos embed
        # tiles in W8A32 mode — see _emit_embedding_lookup).
        self.wte = np.asarray(sd["transformer.wte.weight"], dtype=np.float32)
        self.wpe = np.asarray(sd["transformer.wpe.weight"], dtype=np.float32)

        # Final LN + lm_head.
        self.ln_f_w = np.asarray(sd["transformer.ln_f.weight"], dtype=np.float32)
        self.ln_f_b = np.asarray(sd["transformer.ln_f.bias"], dtype=np.float32)
        # lm_head: weight is [vocab, d_model] in checkpoint; W8A32 emits
        # `lm_head` as `x @ W.T` where W is [vocab, d_model]. So store
        # the transposed-shape INT8: w_int8 is [d_model, vocab].
        lm_head_fp32 = np.asarray(sd["lm_head.weight"], dtype=np.float32)
        self.lm_head_w_int8, self.lm_head_w_scales = _quant_w_per_channel_int8(
            lm_head_fp32.T
        )
        # Optional lm_head bias (created by the LN-fold rotation flow).
        lm_head_bias = sd.get("lm_head.bias")
        self.lm_head_b = (
            np.asarray(lm_head_bias, dtype=np.float32)
            if lm_head_bias is not None else None
        )

        self.layers = []
        for layer_idx in range(self.n_layer):
            prefix = f"transformer.h.{layer_idx}"
            layer = {
                "ln1_w": np.asarray(sd[f"{prefix}.ln_1.weight"], dtype=np.float32),
                "ln1_b": np.asarray(sd[f"{prefix}.ln_1.bias"], dtype=np.float32),
                "ln2_w": np.asarray(sd[f"{prefix}.ln_2.weight"], dtype=np.float32),
                "ln2_b": np.asarray(sd[f"{prefix}.ln_2.bias"], dtype=np.float32),
                "heads": [],
            }
            for head_idx in range(self.n_head):
                head_prefix = f"{prefix}.attn.c_attn"
                q_fp32 = np.asarray(
                    sd[f"{head_prefix}.weight_h{head_idx}_query"], dtype=np.float32
                )
                k_fp32 = np.asarray(
                    sd[f"{head_prefix}.weight_h{head_idx}_key"], dtype=np.float32
                )
                v_fp32 = np.asarray(
                    sd[f"{head_prefix}.weight_h{head_idx}_value"], dtype=np.float32
                )
                q_b = np.asarray(
                    sd[f"{head_prefix}.bias_h{head_idx}_query"], dtype=np.float32
                )
                k_b = np.asarray(
                    sd[f"{head_prefix}.bias_h{head_idx}_key"], dtype=np.float32
                )
                v_b = np.asarray(
                    sd[f"{head_prefix}.bias_h{head_idx}_value"], dtype=np.float32
                )
                # Store INT8 + FP16 scales (transposed to [K, N] = [d_model, d_head]).
                q_int8, q_scales = _quant_w_per_channel_int8(q_fp32.T)
                k_int8, k_scales = _quant_w_per_channel_int8(k_fp32.T)
                v_int8, v_scales = _quant_w_per_channel_int8(v_fp32.T)
                layer["heads"].append({
                    "q_int8": q_int8, "q_scales": q_scales, "q_b": q_b,
                    "k_int8": k_int8, "k_scales": k_scales, "k_b": k_b,
                    "v_int8": v_int8, "v_scales": v_scales, "v_b": v_b,
                })
            c_proj_fp32 = np.asarray(
                sd[f"{prefix}.attn.c_proj.weight"], dtype=np.float32
            )
            layer["c_proj_int8"], layer["c_proj_scales"] = _quant_w_per_channel_int8(
                c_proj_fp32.T
            )
            layer["c_proj_b"] = np.asarray(
                sd[f"{prefix}.attn.c_proj.bias"], dtype=np.float32
            )
            fc_fp32 = np.asarray(sd[f"{prefix}.mlp.c_fc.weight"], dtype=np.float32)
            layer["fc1_int8"], layer["fc1_scales"] = _quant_w_per_channel_int8(
                fc_fp32.T
            )
            layer["fc1_b"] = np.asarray(
                sd[f"{prefix}.mlp.c_fc.bias"], dtype=np.float32
            )
            proj_fp32 = np.asarray(sd[f"{prefix}.mlp.c_proj.weight"], dtype=np.float32)
            layer["fc2_int8"], layer["fc2_scales"] = _quant_w_per_channel_int8(
                proj_fp32.T
            )
            layer["fc2_b"] = np.asarray(
                sd[f"{prefix}.mlp.c_proj.bias"], dtype=np.float32
            )
            self.layers.append(layer)

        # KV cache state.
        self._caches: List[List[Dict[str, List[np.ndarray]]]] = []
        self._next_position = 0
        self._reset_caches()

    def _reset_caches(self) -> None:
        self._caches = [
            [{"k": [], "v": []} for _ in range(self.n_head)]
            for _ in range(self.n_layer)
        ]
        self._next_position = 0

    def _act_scale_for(self, name: str) -> float:
        return float(self.calibration_scales.get(name, self.default_act_scale))

    def _qkt_attn_v_quantize(self, x_fp32: np.ndarray, static_scale: float) -> np.ndarray:
        """Static-scale INT8 quantization for QKT / attn_v inputs.

        Matches the codegen path's `q_scale * k_scale * inv_sqrt_d_head`
        composite — Q and K each get quantized with their *own* static
        calibration scale (NOT a dynamic max_abs), then INT8-matmul,
        then dequant with the composite FP16 factor.
        """
        return np.clip(
            np.round(x_fp32 / max(static_scale, 1e-12)),
            -128, 127,
        ).astype(np.int8)

    def _attention_head(self, ln1: np.ndarray, layer_idx: int, head_idx: int,
                        position: int) -> np.ndarray:
        head = self.layers[layer_idx]["heads"][head_idx]
        # Per-head Q/K/V projections (dynamic activation scale).
        q = _w8a32_dynamic_matmul(
            ln1, head["q_int8"], head["q_scales"], head["q_b"],
            fp_precision=self.fp_precision,
        )
        k = _w8a32_dynamic_matmul(
            ln1, head["k_int8"], head["k_scales"], head["k_b"],
            fp_precision=self.fp_precision,
        )
        v = _w8a32_dynamic_matmul(
            ln1, head["v_int8"], head["v_scales"], head["v_b"],
            fp_precision=self.fp_precision,
        )
        # KV cache append. Under fp_precision='fp16' the K/V tiles are
        # FP16-stored in the simulator's KV cache; mirror that here.
        self._caches[layer_idx][head_idx]["k"].append(_maybe_cast_fp16(k[0], self.fp_precision))
        self._caches[layer_idx][head_idx]["v"].append(_maybe_cast_fp16(v[0], self.fp_precision))
        k_cache = np.stack(self._caches[layer_idx][head_idx]["k"], axis=0)
        v_cache = np.stack(self._caches[layer_idx][head_idx]["v"], axis=0)

        # QKT: static composite scale.
        q_scale = self._act_scale_for(f"block{layer_idx}_head{head_idx}_query")
        k_scale = self._act_scale_for(f"block{layer_idx}_head{head_idx}_key")
        composite_qkt = np.float16(
            np.float32(q_scale) * np.float32(k_scale) * np.float32(self.inv_sqrt_d_head)
        )
        # Q/K source widened from FP16 if applicable (matches QUANT input).
        q_int8 = self._qkt_attn_v_quantize(_maybe_cast_fp16(q, self.fp_precision), q_scale)
        k_int8 = self._qkt_attn_v_quantize(_maybe_cast_fp16(k_cache, self.fp_precision), k_scale)
        scores_int32 = q_int8.astype(np.int32) @ k_int8.astype(np.int32).T
        scores_fp32 = scores_int32.astype(np.float32) * np.float32(composite_qkt)
        # QKT output stored at FP precision.
        scores_fp32 = _maybe_cast_fp16(scores_fp32, self.fp_precision)

        # Masked softmax with valid_kv_len = position + 1.
        valid_kv_len = position + 1
        probs_fp32 = _softmax_masked_fp32(
            scores_fp32, valid_kv_len, fp_precision=self.fp_precision,
        )

        # attn_v: static composite scale.
        sm_scale = self._act_scale_for(f"block{layer_idx}_head{head_idx}_softmax")
        v_scale = self._act_scale_for(f"block{layer_idx}_head{head_idx}_value")
        composite_av = np.float16(np.float32(sm_scale) * np.float32(v_scale))
        sm_int8 = self._qkt_attn_v_quantize(probs_fp32, sm_scale)
        v_cache_int8 = self._qkt_attn_v_quantize(
            _maybe_cast_fp16(v_cache, self.fp_precision), v_scale,
        )
        head_out_int32 = sm_int8.astype(np.int32) @ v_cache_int8.astype(np.int32)
        head_out_fp32 = head_out_int32.astype(np.float32) * np.float32(composite_av)
        return _maybe_cast_fp16(head_out_fp32, self.fp_precision)

    def run_decode_step(self, token_id: int, position: int) -> np.ndarray:
        """Run one decode step and return its FP32 logits."""
        if position != self._next_position:
            raise ValueError(
                f"NanoGPTW8A32SimulatorReference: position {position} doesn't "
                f"match internal cursor {self._next_position}"
            )
        # Token + position embedding lookups. Storage is FP{32,16} matching
        # how the codegen stages embedding tables.
        x = self.wte[token_id:token_id + 1] + self.wpe[position:position + 1]
        x = _maybe_cast_fp16(x, self.fp_precision)

        for layer_idx, layer in enumerate(self.layers):
            ln1 = _layer_norm_fp32(
                x, layer["ln1_w"], layer["ln1_b"], self.layer_norm_epsilon,
                fp_precision=self.fp_precision,
            )
            head_outs = []
            for head_idx in range(self.n_head):
                head_outs.append(
                    self._attention_head(ln1, layer_idx, head_idx, position)
                )
            concat = np.concatenate(head_outs, axis=-1)
            out_proj = _w8a32_dynamic_matmul(
                concat, layer["c_proj_int8"], layer["c_proj_scales"], layer["c_proj_b"],
                fp_precision=self.fp_precision,
            )
            x = _maybe_cast_fp16(x + out_proj, self.fp_precision)  # residual1

            ln2 = _layer_norm_fp32(
                x, layer["ln2_w"], layer["ln2_b"], self.layer_norm_epsilon,
                fp_precision=self.fp_precision,
            )
            fc1 = _w8a32_dynamic_matmul(
                ln2, layer["fc1_int8"], layer["fc1_scales"], layer["fc1_b"],
                fp_precision=self.fp_precision,
            )
            gelu = _gelu_fp32(fc1, fp_precision=self.fp_precision)
            fc2 = _w8a32_dynamic_matmul(
                gelu, layer["fc2_int8"], layer["fc2_scales"], layer["fc2_b"],
                fp_precision=self.fp_precision,
            )
            x = _maybe_cast_fp16(x + fc2, self.fp_precision)  # residual2

        ln_f = _layer_norm_fp32(
            x, self.ln_f_w, self.ln_f_b, self.layer_norm_epsilon,
            fp_precision=self.fp_precision,
        )
        # lm_head: takes only the last row of ln_f (incremental decode).
        logits = _w8a32_dynamic_matmul(
            ln_f[-1:], self.lm_head_w_int8, self.lm_head_w_scales, self.lm_head_b,
            fp_precision=self.fp_precision,
        )
        self._next_position += 1
        return logits[0].astype(np.float32)

    def run_prefill(self, token_ids: Sequence[int]) -> np.ndarray:
        """Run prefill on a prompt (1-token-per-step) and return the
        last position's FP32 logits. Resets the KV cache."""
        tokens = [int(t) for t in token_ids]
        if not tokens:
            raise ValueError("run_prefill requires at least one token")
        self._reset_caches()
        last_logits: Optional[np.ndarray] = None
        for tok in tokens:
            last_logits = self.run_decode_step(int(tok), self._next_position)
        assert last_logits is not None
        return last_logits

    def run_teacher_forced(self, token_ids: Sequence[int]) -> List[np.ndarray]:
        """Convenience: produce logits for every teacher-forced position."""
        toks = [int(t) for t in token_ids]
        if not toks:
            return []
        self._reset_caches()
        out: List[np.ndarray] = []
        for tok in toks:
            out.append(self.run_decode_step(tok, self._next_position))
        return out
