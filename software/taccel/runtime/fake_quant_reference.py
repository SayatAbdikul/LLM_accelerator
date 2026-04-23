"""NumPy fake-quantised nanoGPT forward pass.

Mirrors the TACCEL compiler's quantisation scheme so the reference and the
golden-model simulator share the same per-tensor INT8 scales.  The reference
can run either a full causal forward or an incremental KV-cache decode path.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

try:
    from scipy.special import erf as _scipy_erf
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _to_f32(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def _qdq(x: np.ndarray, scale: float) -> np.ndarray:
    """INT8 quantise-dequantise matching sfu.py _requantize_int8.

    Rounds to nearest (round-half-to-even via NumPy default), clips to
    [-128, 127], then dequantises back to FP32.
    """
    if scale <= 0.0:
        return x.astype(np.float32)
    s = np.float32(scale)
    q = np.clip(np.round(x.astype(np.float32) / s), -128, 127).astype(np.int8)
    return q.astype(np.float32) * s


def _to_int8_logits(x: np.ndarray, scale: float) -> np.ndarray:
    """Quantise FP32 logits to INT8 — matches the golden-model STORE path."""
    if scale <= 0.0:
        return np.zeros_like(x, dtype=np.int8)
    return np.clip(np.round(x.astype(np.float32) / np.float32(scale)), -128, 127).astype(np.int8)


def _scale(scales: Dict[str, float], name: str, default: float = 6.0 / 127.0) -> float:
    return float(scales.get(name, default))


def _layernorm_np(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float) -> np.ndarray:
    """FP32 LayerNorm matching sfu.py Welford-based reduction."""
    mean = x.mean(axis=-1, keepdims=True).astype(np.float32)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True).astype(np.float32)
    return (x - mean) / np.sqrt(var + np.float32(eps)) * w + b


def _gelu_np(x: np.ndarray) -> np.ndarray:
    """GELU matching sfu.py: uses scipy.special.erf when available."""
    xf = x.astype(np.float32)
    if _HAS_SCIPY:
        return xf * np.float32(0.5) * (np.float32(1.0) + _scipy_erf(xf / np.sqrt(np.float32(2.0))))
    # Abramowitz & Stegun 7.1.26 polynomial fallback
    sgn = np.sign(xf)
    t = np.float32(1.0) / (np.float32(1.0) + np.float32(0.3275911) * np.abs(xf))
    poly = t * (
        np.float32(0.254829592) + t * (
            np.float32(-0.284496736) + t * (
                np.float32(1.421413741) + t * (
                    np.float32(-1.453152027) + t * np.float32(1.061405429)
                )
            )
        )
    )
    erf_approx = sgn * (np.float32(1.0) - poly * np.exp(-(xf ** 2)))
    return xf * np.float32(0.5) * (np.float32(1.0) + erf_approx)


def _causal_softmax(x: np.ndarray) -> np.ndarray:
    """Row-wise causal softmax for x[seq, seq], matching execute_masked_softmax.

    Columns j > i (upper triangle) are masked before the max-subtract step.
    """
    seq = x.shape[0]
    out = np.empty_like(x, dtype=np.float32)
    for i in range(seq):
        row = x[i].astype(np.float32).copy()
        row[i + 1:] = -np.inf
        valid = row[: i + 1]
        row_max = float(valid.max())
        exp_row = np.exp(row - row_max)
        exp_row[i + 1:] = 0.0
        out[i] = exp_row / float(exp_row.sum())
    return out


# ---------------------------------------------------------------------------
# Weight fake-quantisation at build time
# ---------------------------------------------------------------------------

def _linear_components(tensor) -> tuple[np.ndarray, np.float32, np.ndarray, np.ndarray]:
    """Return compiler-matched linear weight components.

    The codegen REQUANT uses a single scalar scale = mean(per_channel_scales),
    not per-channel dequant.  The compiler also pads output-channel scales to a
    16-wide boundary before taking that mean, so the reference mirrors that
    padding even when the logical vocab/output size is not a multiple of 16.
    """
    from ..quantizer.quantize import quantize_tensor
    arr = _to_f32(tensor)  # [out, in]
    q, scales = quantize_tensor(arr, per_channel=True)
    padded_out = q.shape[0] + ((16 - q.shape[0] % 16) % 16)
    scales_f = scales.astype(np.float32)
    if scales_f.size < padded_out:
        scales_f = np.pad(scales_f, (0, padded_out - scales_f.size), constant_values=scales_f[-1])
    mean_scale = np.float32(np.mean(scales_f))
    q_i8 = q.astype(np.int8)
    return q_i8, mean_scale, q_i8.astype(np.float32) * mean_scale, scales_f


def _fq_linear(tensor) -> np.ndarray:
    """Per-channel INT8 quantisation dequantised with compiler scalar scale."""
    return _linear_components(tensor)[2]


def _fq_embedding(tensor) -> np.ndarray:
    """Per-tensor INT8 QDQ of an embedding table, matching _quantize_embedding."""
    from ..quantizer.quantize import quantize_tensor, dequantize_tensor
    arr = _to_f32(tensor)
    q, scales = quantize_tensor(arr, per_channel=False)
    return dequantize_tensor(q, scales).astype(np.float32)


# ---------------------------------------------------------------------------
# Reference model
# ---------------------------------------------------------------------------

def _int8_saturating_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Saturating INT8 add matching the golden model's VADD instruction."""
    return np.clip(a.astype(np.int32) + b.astype(np.int32), -128, 127).astype(np.int8)


def _fp32_to_int8(x: np.ndarray, scale: float) -> np.ndarray:
    """Extract the INT8 integer representation from a QDQ FP32 tensor.

    After _qdq(result, scale), values are exact multiples of scale.
    Dividing by scale recovers the stored integers.
    """
    return np.clip(np.round(x.astype(np.float32) / np.float32(scale)), -128, 127).astype(np.int8)


def _requant_accum_int8(accum: np.ndarray, accum_scale: float, out_scale: float) -> np.ndarray:
    """Requantize INT32 accumulators through the architectural FP16 scale reg."""
    if out_scale <= 0.0:
        return np.zeros_like(accum, dtype=np.int8)
    requant = np.float32(np.float16(np.float32(accum_scale) / np.float32(out_scale)))
    return np.clip(np.round(accum.astype(np.float32) * requant), -128, 127).astype(np.int8)


def _dequant_add_accum_int8(
    accum: np.ndarray,
    accum_scale: float,
    skip_i8: np.ndarray,
    skip_scale: float,
    out_scale: float,
) -> np.ndarray:
    """Mirror DEQUANT_ADD: ACCUM and INT8 skip are rescaled via FP16 regs."""
    if out_scale <= 0.0:
        return np.zeros_like(skip_i8, dtype=np.int8)
    accum_rescale = np.float32(np.float16(np.float32(accum_scale) / np.float32(out_scale)))
    skip_rescale = np.float32(np.float16(np.float32(skip_scale) / np.float32(out_scale)))
    return np.clip(
        np.round(accum.astype(np.float32) * accum_rescale + skip_i8.astype(np.float32) * skip_rescale),
        -128,
        127,
    ).astype(np.int8)


def _bias_i32(state_dict: dict, name: str, act_scale: float, weight_scales: np.ndarray,
              output_dim: int) -> np.ndarray:
    if name not in state_dict:
        return np.zeros(int(output_dim), dtype=np.int32)
    bias = _to_f32(state_dict[name]).reshape(-1)
    if bias.size != int(output_dim):
        raise ValueError(f"{name!r} has {bias.size} values, expected {output_dim}")
    scales = np.asarray(weight_scales, dtype=np.float32)[:bias.size]
    denom = np.maximum(np.abs(np.float32(act_scale) * scales), 1e-10)
    return np.round(bias / denom).astype(np.int32)


class NanoGPTFQReference:
    """NumPy fake-quantised nanoGPT forward pass sharing scales with the compiler.

    Replicates the golden model's INT8 arithmetic faithfully:
    - Embeddings: INT8 per-tensor lookup + saturating INT8 VADD (matching the
      VADD instruction in the compiled program).
    - Residuals: saturating INT8 VADD matching the VADD instruction.
    - Concat: INT8 values from each head reinterpreted with concat_scale (no
      spurious extra REQUANT step).
    - LayerNorm: epsilon = 1e-6 matching sfu.py's hardcoded value.
    - Matmul → REQUANT: FP32 matmul with FQ weights gives the same INT32
      accumulator value as integer matmul (sum ≤ 2M < 2^24 FP32 exact range).

    Usage::

        ref = NanoGPTFQReference(state_dict, model_args, scales)
        logits_int8 = ref.forward([tok0, tok1, ...])   # INT8 [vocab_size]
    """

    def __init__(
        self,
        state_dict: dict,
        model_args: dict,
        scales: Dict[str, float],
    ) -> None:
        self.n_layer = int(model_args["n_layer"])
        self.n_head = int(model_args["n_head"])
        self.d_model = int(model_args["n_embd"])
        self.d_head = self.d_model // self.n_head
        self.vocab_size = int(model_args["vocab_size"])
        self.attn_scale = np.float32(self.d_head ** -0.5)
        # sfu.py uses 1e-6 hardcoded — not from model_args
        self.eps = np.float32(1e-6)
        self.scales = dict(scales)
        sd = state_dict
        self._has_nonzero_linear_bias = any(
            ("bias" in name and ("c_attn" in name or "c_proj" in name or "c_fc" in name))
            and np.any(np.abs(_to_f32(value)) > 0.0)
            for name, value in sd.items()
        )

        # INT8 embeddings are stored at the tok_pos_add scale because the
        # compiled program uses a raw INT8 VADD for token + position.  Separate
        # table scales would make q_token + q_pos numerically meaningless.
        self.wte_fp32 = _to_f32(sd["transformer.wte.weight"])
        self.wpe_fp32 = _to_f32(sd["transformer.wpe.weight"])
        embed_add_scale = _scale(self.scales, "tok_pos_add")
        wte_q = np.clip(np.round(self.wte_fp32 / np.float32(embed_add_scale)), -128, 127).astype(np.int8)
        wpe_q = np.clip(np.round(self.wpe_fp32 / np.float32(embed_add_scale)), -128, 127).astype(np.int8)
        self.wte_int8 = wte_q  # [V, d], INT8
        self.wpe_int8 = wpe_q  # [T, d], INT8

        self.ln_f_w = _to_f32(sd["transformer.ln_f.weight"])
        self.ln_f_b = _to_f32(sd["transformer.ln_f.bias"])
        self.lm_head_w_q, self.lm_head_w_scale, self.lm_head_w, self.lm_head_w_scales = _linear_components(sd["lm_head.weight"])
        self.lm_head_w_fp32 = _to_f32(sd["lm_head.weight"])

        self.layers = []
        for L in range(self.n_layer):
            heads = []
            for H in range(self.n_head):
                q_name = f"transformer.h.{L}.attn.c_attn.weight_h{H}_query"
                k_name = f"transformer.h.{L}.attn.c_attn.weight_h{H}_key"
                v_name = f"transformer.h.{L}.attn.c_attn.weight_h{H}_value"
                q_q, q_scale_w, q_w, q_scales = _linear_components(sd[q_name])
                k_q, k_scale_w, k_w, k_scales = _linear_components(sd[k_name])
                v_q, v_scale_w, v_w, v_scales = _linear_components(sd[v_name])
                ln1_scale = _scale(self.scales, f"block{L}_ln1")
                q_b_name = f"transformer.h.{L}.attn.c_attn.bias_h{H}_query"
                k_b_name = f"transformer.h.{L}.attn.c_attn.bias_h{H}_key"
                v_b_name = f"transformer.h.{L}.attn.c_attn.bias_h{H}_value"
                heads.append((
                    q_w,
                    k_w,
                    v_w,
                    _to_f32(sd[f"transformer.h.{L}.attn.c_attn.weight_h{H}_query"]),
                    _to_f32(sd[f"transformer.h.{L}.attn.c_attn.weight_h{H}_key"]),
                    _to_f32(sd[f"transformer.h.{L}.attn.c_attn.weight_h{H}_value"]),
                    q_q,
                    k_q,
                    v_q,
                    q_scale_w,
                    k_scale_w,
                    v_scale_w,
                    q_scales,
                    k_scales,
                    v_scales,
                    _bias_i32(sd, q_b_name, ln1_scale, q_scales, self.d_head),
                    _bias_i32(sd, k_b_name, ln1_scale, k_scales, self.d_head),
                    _bias_i32(sd, v_b_name, ln1_scale, v_scales, self.d_head),
                    _to_f32(sd[q_b_name]) if q_b_name in sd else np.zeros(self.d_head, dtype=np.float32),
                    _to_f32(sd[k_b_name]) if k_b_name in sd else np.zeros(self.d_head, dtype=np.float32),
                    _to_f32(sd[v_b_name]) if v_b_name in sd else np.zeros(self.d_head, dtype=np.float32),
                ))
            c_proj_q, c_proj_scale_w, c_proj_w, c_proj_scales = _linear_components(sd[f"transformer.h.{L}.attn.c_proj.weight"])
            fc_q, fc_scale_w, fc_w, fc_scales = _linear_components(sd[f"transformer.h.{L}.mlp.c_fc.weight"])
            proj_q, proj_scale_w, proj_w, proj_scales = _linear_components(sd[f"transformer.h.{L}.mlp.c_proj.weight"])
            self.layers.append({
                "ln1_w": _to_f32(sd[f"transformer.h.{L}.ln_1.weight"]),
                "ln1_b": _to_f32(sd[f"transformer.h.{L}.ln_1.bias"]),
                "ln2_w": _to_f32(sd[f"transformer.h.{L}.ln_2.weight"]),
                "ln2_b": _to_f32(sd[f"transformer.h.{L}.ln_2.bias"]),
                "heads": heads,
                "c_proj_w": c_proj_w,
                "c_proj_w_q": c_proj_q,
                "c_proj_w_scale": c_proj_scale_w,
                "c_proj_w_scales": c_proj_scales,
                "c_proj_w_fp32": _to_f32(sd[f"transformer.h.{L}.attn.c_proj.weight"]),
                "c_proj_b": _to_f32(sd[f"transformer.h.{L}.attn.c_proj.bias"]),
                "c_proj_b_i32": _bias_i32(
                    sd,
                    f"transformer.h.{L}.attn.c_proj.bias",
                    _scale(self.scales, f"block{L}_concat"),
                    c_proj_scales,
                    self.d_model,
                ),
                "fc_w": fc_w,
                "fc_w_q": fc_q,
                "fc_w_scale": fc_scale_w,
                "fc_w_scales": fc_scales,
                "fc_w_fp32": _to_f32(sd[f"transformer.h.{L}.mlp.c_fc.weight"]),
                "fc_b": _to_f32(sd[f"transformer.h.{L}.mlp.c_fc.bias"]),
                "fc_b_i32": _bias_i32(
                    sd,
                    f"transformer.h.{L}.mlp.c_fc.bias",
                    _scale(self.scales, f"block{L}_ln2"),
                    fc_scales,
                    4 * self.d_model,
                ),
                "proj_w": proj_w,
                "proj_w_q": proj_q,
                "proj_w_scale": proj_scale_w,
                "proj_w_scales": proj_scales,
                "proj_w_fp32": _to_f32(sd[f"transformer.h.{L}.mlp.c_proj.weight"]),
                "proj_b": _to_f32(sd[f"transformer.h.{L}.mlp.c_proj.bias"]),
                "proj_b_i32": _bias_i32(
                    sd,
                    f"transformer.h.{L}.mlp.c_proj.bias",
                    _scale(self.scales, f"block{L}_gelu", 1.0 / 127.0),
                    proj_scales,
                    self.d_model,
                ),
            })

    def _large_vocab_lm_head_reference(self) -> bool:
        """Use integer lm_head reference when codegen must tile the weight."""
        return int(self.lm_head_w_q.size) > 256 * 1024

    def _integer_linear_reference(self) -> bool:
        """Use bias-aware integer linear math for GPT-2-class converted models."""
        return self._large_vocab_lm_head_reference() or self._has_nonzero_linear_bias

    def forward(
        self,
        token_ids: Sequence[int],
        position_ids: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """Return INT8 logits for the last position. Shape [vocab_size]."""
        tids = list(token_ids)
        seq = len(tids)
        pids = list(position_ids) if position_ids is not None else list(range(seq))
        s = self.scales

        # Embedding: INT8 saturating VADD matching the compiled VADD instruction.
        # Both tables are stored INT8 (per-tensor quantized) and added as integers.
        x_int8 = _int8_saturating_add(self.wte_int8[tids], self.wpe_int8[pids])
        x_scale = _scale(s, "tok_pos_add")
        x = x_int8.astype(np.float32) * np.float32(x_scale)

        for L, layer in enumerate(self.layers):
            # --- Attention ---
            ln1_scale = _scale(s, f"block{L}_ln1")
            ln1 = _qdq(
                _layernorm_np(x, layer["ln1_w"], layer["ln1_b"], self.eps),
                ln1_scale,
            )
            ln1_i8 = _fp32_to_int8(ln1, ln1_scale)
            head_outs_int8 = []
            for H, head_weights in enumerate(layer["heads"]):
                q_scale = _scale(s, f"block{L}_head{H}_query")
                k_scale = _scale(s, f"block{L}_head{H}_key")
                v_scale = _scale(s, f"block{L}_head{H}_value")
                if self._integer_linear_reference():
                    q_q, k_q, v_q = head_weights[6:9]
                    q_scale_w, k_scale_w, v_scale_w = head_weights[9:12]
                    q_b_i32, k_b_i32, v_b_i32 = head_weights[15:18]
                    q_accum = ln1_i8.astype(np.int32) @ q_q.astype(np.int32).T + q_b_i32
                    k_accum = ln1_i8.astype(np.int32) @ k_q.astype(np.int32).T + k_b_i32
                    v_accum = ln1_i8.astype(np.int32) @ v_q.astype(np.int32).T + v_b_i32
                    q_i8 = _requant_accum_int8(q_accum, np.float32(ln1_scale) * np.float32(q_scale_w), q_scale)
                    k_i8 = _requant_accum_int8(k_accum, np.float32(ln1_scale) * np.float32(k_scale_w), k_scale)
                    v_i8 = _requant_accum_int8(v_accum, np.float32(ln1_scale) * np.float32(v_scale_w), v_scale)
                    q = q_i8.astype(np.float32) * np.float32(q_scale)
                    k = k_i8.astype(np.float32) * np.float32(k_scale)
                    v = v_i8.astype(np.float32) * np.float32(v_scale)
                else:
                    q_w, k_w, v_w = head_weights[:3]
                    q = _qdq(ln1 @ q_w.T, q_scale)
                    k = _qdq(ln1 @ k_w.T, k_scale)
                    v = _qdq(ln1 @ v_w.T, v_scale)

                attn = (q @ k.T) * self.attn_scale
                probs = np.ones((1, 1), dtype=np.float32) if seq == 1 else _causal_softmax(attn)
                probs = _qdq(probs, _scale(s, f"block{L}_head{H}_softmax", 1.0 / 127.0))
                head_out = _qdq(probs @ v, _scale(s, f"block{L}_head{H}_attn_v"))
                # Extract INT8: each head's output is stored in WBUF as INT8
                head_outs_int8.append(
                    _fp32_to_int8(head_out, _scale(s, f"block{L}_head{H}_attn_v"))
                )

            # Concat: INT8 values placed side-by-side in WBUF with no extra REQUANT.
            # The out_proj matmul then treats them with concat_scale as the input scale.
            concat_int8 = np.concatenate(head_outs_int8, axis=-1)
            concat_scale = _scale(s, f"block{L}_concat")
            concat = concat_int8.astype(np.float32) * np.float32(concat_scale)

            concat_scale = _scale(s, f"block{L}_concat")
            out_proj_scale = _scale(s, f"block{L}_out_proj")
            out_accum = (
                concat_int8.astype(np.int32) @ layer["c_proj_w_q"].astype(np.int32).T
                + layer["c_proj_b_i32"]
            )
            out_accum_scale = np.float32(concat_scale) * np.float32(layer["c_proj_w_scale"])
            out_proj_i8 = _requant_accum_int8(out_accum, out_accum_scale, out_proj_scale)
            out_proj = out_proj_i8.astype(np.float32) * np.float32(out_proj_scale)
            # Residual 1: decoder bundles use DEQUANT_ADD, which consumes the
            # branch accumulator directly through FP16 rescale registers.
            prev_x_int8 = x_int8
            prev_x_scale = x_scale
            x_scale = _scale(s, f"block{L}_residual1")
            x_int8 = _dequant_add_accum_int8(out_accum, out_accum_scale, prev_x_int8, prev_x_scale, x_scale)
            x = x_int8.astype(np.float32) * np.float32(x_scale)

            # --- MLP ---
            ln2 = _qdq(
                _layernorm_np(x, layer["ln2_w"], layer["ln2_b"], self.eps),
                _scale(s, f"block{L}_ln2"),
            )
            fc1_scale = _scale(s, f"block{L}_fc1")
            ln2_scale = _scale(s, f"block{L}_ln2")
            ln2_i8 = _fp32_to_int8(ln2, ln2_scale)
            if self._integer_linear_reference():
                fc1_accum = (
                    ln2_i8.astype(np.int32) @ layer["fc_w_q"].astype(np.int32).T
                    + layer["fc_b_i32"]
                )
                fc1_i8 = _requant_accum_int8(
                    fc1_accum,
                    np.float32(ln2_scale) * np.float32(layer["fc_w_scale"]),
                    fc1_scale,
                )
                fc1 = fc1_i8.astype(np.float32) * np.float32(fc1_scale)
            else:
                fc1 = _qdq(ln2 @ layer["fc_w"].T + layer["fc_b"], fc1_scale)
            gelu = _qdq(_gelu_np(fc1), _scale(s, f"block{L}_gelu", 1.0 / 127.0))
            gelu_scale = _scale(s, f"block{L}_gelu", 1.0 / 127.0)
            gelu_i8 = _fp32_to_int8(gelu, gelu_scale)
            fc2_scale = _scale(s, f"block{L}_fc2")
            fc2_accum = (
                gelu_i8.astype(np.int32) @ layer["proj_w_q"].astype(np.int32).T
                + layer["proj_b_i32"]
            )
            fc2_accum_scale = np.float32(gelu_scale) * np.float32(layer["proj_w_scale"])
            fc2_i8 = _requant_accum_int8(fc2_accum, fc2_accum_scale, fc2_scale)
            fc2 = fc2_i8.astype(np.float32) * np.float32(fc2_scale)
            prev_x_int8 = x_int8
            prev_x_scale = x_scale
            x_scale = _scale(s, f"block{L}_residual2")
            x_int8 = _dequant_add_accum_int8(fc2_accum, fc2_accum_scale, prev_x_int8, prev_x_scale, x_scale)
            x = x_int8.astype(np.float32) * np.float32(x_scale)

        ln_f_scale = _scale(s, "ln_f")
        ln_f = _qdq(_layernorm_np(x, self.ln_f_w, self.ln_f_b, self.eps), ln_f_scale)
        if not self._large_vocab_lm_head_reference():
            logits = ln_f[-1:] @ self.lm_head_w.T
            return _to_int8_logits(logits[0], _scale(s, "lm_head"))
        ln_f_i8 = _fp32_to_int8(ln_f[-1:], ln_f_scale)
        logits_accum = ln_f_i8.astype(np.int32) @ self.lm_head_w_q.astype(np.int32).T
        logits_i8 = _requant_accum_int8(
            logits_accum,
            np.float32(ln_f_scale) * np.float32(self.lm_head_w_scale),
            _scale(s, "lm_head"),
        )
        return logits_i8[0]

    def forward_incremental(
        self,
        token_ids: Sequence[int],
        position_ids: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """Return INT8 logits for the last token using explicit K/V caches."""
        trace = self.incremental_logits_trace(token_ids, position_ids=position_ids)
        if not trace:
            raise ValueError("token_ids must be non-empty")
        return trace[-1]

    def incremental_logits_trace(
        self,
        token_ids: Sequence[int],
        position_ids: Optional[Sequence[int]] = None,
        fp32_groups: Optional[set[str]] = None,
    ) -> list[np.ndarray]:
        """Run a cache-style decode and return logits after each token."""
        tids = [int(tok) for tok in token_ids]
        if not tids:
            return []
        pids = list(position_ids) if position_ids is not None else list(range(len(tids)))
        if len(pids) != len(tids):
            raise ValueError("position_ids length must match token_ids length")

        caches = [
            [
                {"k": [], "v": []}
                for _ in range(self.n_head)
            ]
            for _ in range(self.n_layer)
        ]
        return [
            self._decode_incremental_step(tok, pos, caches, fp32_groups=fp32_groups)
            for tok, pos in zip(tids, pids)
        ]

    def incremental_node_trace(
        self,
        token_ids: Sequence[int],
        position_ids: Optional[Sequence[int]] = None,
        fp32_groups: Optional[set[str]] = None,
    ) -> list[dict[str, dict[str, np.ndarray | float | None]]]:
        """Run incremental decode and return per-node fake-quant trace entries."""
        tids = [int(tok) for tok in token_ids]
        if not tids:
            return []
        pids = list(position_ids) if position_ids is not None else list(range(len(tids)))
        if len(pids) != len(tids):
            raise ValueError("position_ids length must match token_ids length")
        caches = [
            [
                {"k": [], "v": []}
                for _ in range(self.n_head)
            ]
            for _ in range(self.n_layer)
        ]
        traces = []
        for tok, pos in zip(tids, pids):
            step_trace: dict[str, dict[str, np.ndarray | float | None]] = {}
            self._decode_incremental_step(
                tok,
                pos,
                caches,
                trace=step_trace,
                fp32_groups=fp32_groups,
            )
            traces.append(step_trace)
        return traces

    def _decode_incremental_step(
        self,
        token_id: int,
        position_id: int,
        caches,
        trace: Optional[dict[str, dict[str, np.ndarray | float | None]]] = None,
        fp32_groups: Optional[set[str]] = None,
    ) -> np.ndarray:
        s = self.scales
        groups = set(fp32_groups or ())

        def record(name: str, value: np.ndarray, *, scale: Optional[float] = None,
                   int8: Optional[np.ndarray] = None) -> None:
            if trace is None:
                return
            trace[name] = {
                "value": np.asarray(value, dtype=np.float32).copy(),
                "scale": None if scale is None else float(scale),
                "int8": None if int8 is None else np.asarray(int8, dtype=np.int8).copy(),
            }

        x_scale = _scale(s, "tok_pos_add")
        if "embeddings" in groups:
            x = self.wte_fp32[[token_id]] + self.wpe_fp32[[position_id]]
            x_int8 = _fp32_to_int8(_qdq(x, x_scale), x_scale)
        else:
            x_int8 = _int8_saturating_add(
                self.wte_int8[[token_id]],
                self.wpe_int8[[position_id]],
            )
            x = x_int8.astype(np.float32) * np.float32(x_scale)
        record("tok_pos_add", x, scale=x_scale, int8=x_int8)

        for L, layer in enumerate(self.layers):
            ln1_scale = _scale(s, f"block{L}_ln1")
            ln1 = _qdq(
                _layernorm_np(x, layer["ln1_w"], layer["ln1_b"], self.eps),
                ln1_scale,
            )
            record(f"block{L}_ln1", ln1, scale=ln1_scale, int8=_fp32_to_int8(ln1, ln1_scale))
            ln1_i8 = _fp32_to_int8(ln1, ln1_scale)
            head_outs_int8 = []
            for H, head_weights in enumerate(layer["heads"]):
                q_w, k_w, v_w, q_w_fp32, k_w_fp32, v_w_fp32 = head_weights[:6]
                q_q, k_q, v_q = head_weights[6:9]
                q_scale_w, k_scale_w, v_scale_w = head_weights[9:12]
                q_b_i32, k_b_i32, v_b_i32 = head_weights[15:18]
                q_b_fp32, k_b_fp32, v_b_fp32 = head_weights[18:21]
                q_scale = _scale(s, f"block{L}_head{H}_query")
                k_scale = _scale(s, f"block{L}_head{H}_key")
                v_scale = _scale(s, f"block{L}_head{H}_value")
                attn_v_scale = _scale(s, f"block{L}_head{H}_attn_v")

                if "qkv" in groups:
                    q = ln1 @ q_w_fp32.T + q_b_fp32
                    k = ln1 @ k_w_fp32.T + k_b_fp32
                    v = ln1 @ v_w_fp32.T + v_b_fp32
                elif self._integer_linear_reference():
                    q_accum = ln1_i8.astype(np.int32) @ q_q.astype(np.int32).T + q_b_i32
                    k_accum = ln1_i8.astype(np.int32) @ k_q.astype(np.int32).T + k_b_i32
                    v_accum = ln1_i8.astype(np.int32) @ v_q.astype(np.int32).T + v_b_i32
                    q_i8 = _requant_accum_int8(q_accum, np.float32(ln1_scale) * np.float32(q_scale_w), q_scale)
                    k_i8 = _requant_accum_int8(k_accum, np.float32(ln1_scale) * np.float32(k_scale_w), k_scale)
                    v_i8 = _requant_accum_int8(v_accum, np.float32(ln1_scale) * np.float32(v_scale_w), v_scale)
                    q = q_i8.astype(np.float32) * np.float32(q_scale)
                    k = k_i8.astype(np.float32) * np.float32(k_scale)
                    v = v_i8.astype(np.float32) * np.float32(v_scale)
                else:
                    q = _qdq(ln1 @ q_w.T, q_scale)
                    k = _qdq(ln1 @ k_w.T, k_scale)
                    v = _qdq(ln1 @ v_w.T, v_scale)
                k_i8 = _fp32_to_int8(k, k_scale)
                v_i8 = _fp32_to_int8(v, v_scale)
                record(f"block{L}_head{H}_query", q, scale=q_scale, int8=_fp32_to_int8(q, q_scale))
                record(f"block{L}_head{H}_key", k, scale=k_scale, int8=k_i8)
                record(f"block{L}_head{H}_value", v, scale=v_scale, int8=v_i8)
                caches[L][H]["k"].append(k_i8[0].copy())
                caches[L][H]["v"].append(v_i8[0].copy())

                k_cache = np.stack(caches[L][H]["k"], axis=0).astype(np.float32) * np.float32(k_scale)
                v_cache = np.stack(caches[L][H]["v"], axis=0).astype(np.float32) * np.float32(v_scale)
                attn = (q @ k_cache.T) * self.attn_scale
                row = attn[0].astype(np.float32)
                row_max = float(row.max())
                exp_row = np.exp(row - row_max)
                probs = (exp_row / float(exp_row.sum()))[None, :]
                softmax_scale = _scale(s, f"block{L}_head{H}_softmax", 1.0 / 127.0)
                if "softmax" not in groups:
                    probs = _qdq(probs, softmax_scale)
                record(
                    f"block{L}_head{H}_softmax",
                    probs,
                    scale=softmax_scale,
                    int8=_fp32_to_int8(probs, softmax_scale),
                )
                head_out = probs @ v_cache
                if "attn_v" not in groups:
                    head_out = _qdq(head_out, attn_v_scale)
                record(
                    f"block{L}_head{H}_attn_v",
                    head_out,
                    scale=attn_v_scale,
                    int8=_fp32_to_int8(head_out, attn_v_scale),
                )
                head_outs_int8.append(_fp32_to_int8(head_out, attn_v_scale))

            concat_int8 = np.concatenate(head_outs_int8, axis=-1)
            concat_scale = _scale(s, f"block{L}_concat")
            concat = concat_int8.astype(np.float32) * np.float32(concat_scale)
            record(f"block{L}_concat", concat, scale=concat_scale, int8=concat_int8)

            out_proj_scale = _scale(s, f"block{L}_out_proj")
            if "out_proj" in groups:
                out_proj = concat @ layer["c_proj_w_fp32"].T + layer["c_proj_b"]
                out_proj_i8 = _fp32_to_int8(_qdq(out_proj, out_proj_scale), out_proj_scale)
                out_accum = None
                out_accum_scale = None
            else:
                out_accum = (
                    concat_int8.astype(np.int32) @ layer["c_proj_w_q"].astype(np.int32).T
                    + layer["c_proj_b_i32"]
                )
                out_accum_scale = np.float32(concat_scale) * np.float32(layer["c_proj_w_scale"])
                out_proj_i8 = _requant_accum_int8(out_accum, out_accum_scale, out_proj_scale)
                out_proj = out_proj_i8.astype(np.float32) * np.float32(out_proj_scale)
            record(f"block{L}_out_proj", out_proj, scale=out_proj_scale, int8=out_proj_i8)
            x_scale = _scale(s, f"block{L}_residual1")
            if "residual_vadd" in groups or out_accum is None:
                x = _qdq(x + out_proj, x_scale)
                x_int8 = _fp32_to_int8(x, x_scale)
            else:
                x_int8 = _dequant_add_accum_int8(out_accum, out_accum_scale, x_int8, _scale(s, "tok_pos_add") if L == 0 else _scale(s, f"block{L - 1}_residual2"), x_scale)
                x = x_int8.astype(np.float32) * np.float32(x_scale)
            record(f"block{L}_residual1", x, scale=x_scale, int8=x_int8)

            ln2_scale = _scale(s, f"block{L}_ln2")
            ln2 = _qdq(
                _layernorm_np(x, layer["ln2_w"], layer["ln2_b"], self.eps),
                ln2_scale,
            )
            record(f"block{L}_ln2", ln2, scale=ln2_scale, int8=_fp32_to_int8(ln2, ln2_scale))
            fc1_scale = _scale(s, f"block{L}_fc1")
            gelu_scale = _scale(s, f"block{L}_gelu", 1.0 / 127.0)
            fc2_scale = _scale(s, f"block{L}_fc2")
            if "mlp" in groups:
                fc1 = ln2 @ layer["fc_w_fp32"].T + layer["fc_b"]
                gelu = _gelu_np(fc1)
                fc2 = gelu @ layer["proj_w_fp32"].T + layer["proj_b"]
                fc2_i8 = _fp32_to_int8(_qdq(fc2, fc2_scale), fc2_scale)
                fc2_accum = None
                fc2_accum_scale = None
            elif self._integer_linear_reference():
                ln2_i8 = _fp32_to_int8(ln2, ln2_scale)
                fc1_accum = (
                    ln2_i8.astype(np.int32) @ layer["fc_w_q"].astype(np.int32).T
                    + layer["fc_b_i32"]
                )
                fc1_i8 = _requant_accum_int8(
                    fc1_accum,
                    np.float32(ln2_scale) * np.float32(layer["fc_w_scale"]),
                    fc1_scale,
                )
                fc1 = fc1_i8.astype(np.float32) * np.float32(fc1_scale)
                gelu = _qdq(_gelu_np(fc1), gelu_scale)
                gelu_i8 = _fp32_to_int8(gelu, gelu_scale)
                fc2_accum = (
                    gelu_i8.astype(np.int32) @ layer["proj_w_q"].astype(np.int32).T
                    + layer["proj_b_i32"]
                )
                fc2_accum_scale = np.float32(gelu_scale) * np.float32(layer["proj_w_scale"])
                fc2_i8 = _requant_accum_int8(fc2_accum, fc2_accum_scale, fc2_scale)
                fc2 = fc2_i8.astype(np.float32) * np.float32(fc2_scale)
            else:
                fc1 = _qdq(ln2 @ layer["fc_w"].T + layer["fc_b"], fc1_scale)
                gelu = _qdq(_gelu_np(fc1), gelu_scale)
                gelu_i8 = _fp32_to_int8(gelu, gelu_scale)
                fc2_accum = gelu_i8.astype(np.int32) @ layer["proj_w_q"].astype(np.int32).T
                fc2_accum_scale = np.float32(gelu_scale) * np.float32(layer["proj_w_scale"])
                fc2_i8 = _requant_accum_int8(fc2_accum, fc2_accum_scale, fc2_scale)
                fc2 = fc2_i8.astype(np.float32) * np.float32(fc2_scale)
            record(f"block{L}_fc1", fc1, scale=fc1_scale, int8=_fp32_to_int8(fc1, fc1_scale))
            record(f"block{L}_gelu", gelu, scale=gelu_scale, int8=_fp32_to_int8(gelu, gelu_scale))
            record(f"block{L}_fc2", fc2, scale=fc2_scale, int8=fc2_i8)
            residual1_int8 = x_int8
            residual1_scale = x_scale
            x_scale = _scale(s, f"block{L}_residual2")
            if "residual_vadd" in groups or fc2_accum is None:
                x = _qdq(x + fc2, x_scale)
                x_int8 = _fp32_to_int8(x, x_scale)
            else:
                x_int8 = _dequant_add_accum_int8(fc2_accum, fc2_accum_scale, residual1_int8, residual1_scale, x_scale)
                x = x_int8.astype(np.float32) * np.float32(x_scale)
            record(f"block{L}_residual2", x, scale=x_scale, int8=x_int8)

        ln_f_scale = _scale(s, "ln_f")
        ln_f_raw = _layernorm_np(x, self.ln_f_w, self.ln_f_b, self.eps)
        ln_f = ln_f_raw if "ln_f" in groups else _qdq(ln_f_raw, ln_f_scale)
        record("ln_f", ln_f, scale=ln_f_scale, int8=_fp32_to_int8(ln_f, ln_f_scale))
        if "lm_head" in groups:
            logits = ln_f @ self.lm_head_w_fp32.T
            record("lm_head", logits, scale=None, int8=None)
            return logits[0].astype(np.float32)
        if not self._large_vocab_lm_head_reference():
            logits = ln_f @ self.lm_head_w.T
            logits_i8 = _to_int8_logits(logits[0], _scale(s, "lm_head"))
            record(
                "lm_head",
                logits_i8.astype(np.float32) * np.float32(_scale(s, "lm_head")),
                scale=_scale(s, "lm_head"),
                int8=logits_i8,
            )
            return logits_i8
        ln_f_i8 = _fp32_to_int8(ln_f, ln_f_scale)
        logits_accum = ln_f_i8.astype(np.int32) @ self.lm_head_w_q.astype(np.int32).T
        logits_i8 = _requant_accum_int8(
            logits_accum,
            np.float32(ln_f_scale) * np.float32(self.lm_head_w_scale),
            _scale(s, "lm_head"),
        )
        logits = logits_i8.astype(np.float32) * np.float32(_scale(s, "lm_head"))
        record(
            "lm_head",
            logits,
            scale=_scale(s, "lm_head"),
            int8=logits_i8,
        )
        return logits_i8[0]


# ---------------------------------------------------------------------------
# FP32-only forward for scale calibration (no QDQ)
# ---------------------------------------------------------------------------

def _fp32_forward(
    state_dict: dict,
    model_args: dict,
    token_ids: Sequence[int],
) -> Dict[str, np.ndarray]:
    """Run FP32 forward and return per-node outputs keyed by IR node name."""
    n_layer = int(model_args["n_layer"])
    n_head = int(model_args["n_head"])
    d_model = int(model_args["n_embd"])
    d_head = d_model // n_head
    eps = float(model_args.get("layer_norm_epsilon", 1e-5))
    sd = state_dict

    tids = list(token_ids)
    pids = list(range(len(tids)))
    seq = len(tids)

    wte = _to_f32(sd["transformer.wte.weight"])
    wpe = _to_f32(sd["transformer.wpe.weight"])
    ln_f_w = _to_f32(sd["transformer.ln_f.weight"])
    ln_f_b = _to_f32(sd["transformer.ln_f.bias"])
    lm_head_w = _to_f32(sd["lm_head.weight"])

    out: Dict[str, np.ndarray] = {}
    out["tok_embed"] = wte[tids]
    out["pos_embed"] = wpe[pids]
    x = out["tok_pos_add"] = out["tok_embed"] + out["pos_embed"]

    for L in range(n_layer):
        ln1_w = _to_f32(sd[f"transformer.h.{L}.ln_1.weight"])
        ln1_b = _to_f32(sd[f"transformer.h.{L}.ln_1.bias"])
        ln1 = out[f"block{L}_ln1"] = _layernorm_np(x, ln1_w, ln1_b, eps)

        head_outs = []
        for H in range(n_head):
            q_w = _to_f32(sd[f"transformer.h.{L}.attn.c_attn.weight_h{H}_query"])
            k_w = _to_f32(sd[f"transformer.h.{L}.attn.c_attn.weight_h{H}_key"])
            v_w = _to_f32(sd[f"transformer.h.{L}.attn.c_attn.weight_h{H}_value"])
            q_b = _to_f32(sd.get(
                f"transformer.h.{L}.attn.c_attn.bias_h{H}_query",
                np.zeros(d_head, dtype=np.float32),
            ))
            k_b = _to_f32(sd.get(
                f"transformer.h.{L}.attn.c_attn.bias_h{H}_key",
                np.zeros(d_head, dtype=np.float32),
            ))
            v_b = _to_f32(sd.get(
                f"transformer.h.{L}.attn.c_attn.bias_h{H}_value",
                np.zeros(d_head, dtype=np.float32),
            ))
            q = out[f"block{L}_head{H}_query"] = ln1 @ q_w.T + q_b
            k = out[f"block{L}_head{H}_key"] = ln1 @ k_w.T + k_b
            v = out[f"block{L}_head{H}_value"] = ln1 @ v_w.T + v_b
            attn = (q @ k.T) * (d_head ** -0.5)
            probs = np.ones((1, 1), dtype=np.float32) if seq == 1 else _causal_softmax(attn)
            out[f"block{L}_head{H}_softmax"] = probs
            head_out = out[f"block{L}_head{H}_attn_v"] = probs @ v
            head_outs.append(head_out)

        concat = out[f"block{L}_concat"] = np.concatenate(head_outs, axis=-1)
        c_proj_w = _to_f32(sd[f"transformer.h.{L}.attn.c_proj.weight"])
        c_proj_b = _to_f32(sd[f"transformer.h.{L}.attn.c_proj.bias"])
        out_proj = out[f"block{L}_out_proj"] = concat @ c_proj_w.T + c_proj_b
        x = out[f"block{L}_residual1"] = x + out_proj

        ln2_w = _to_f32(sd[f"transformer.h.{L}.ln_2.weight"])
        ln2_b = _to_f32(sd[f"transformer.h.{L}.ln_2.bias"])
        ln2 = out[f"block{L}_ln2"] = _layernorm_np(x, ln2_w, ln2_b, eps)
        fc_w = _to_f32(sd[f"transformer.h.{L}.mlp.c_fc.weight"])
        fc_b = _to_f32(sd[f"transformer.h.{L}.mlp.c_fc.bias"])
        fc1 = out[f"block{L}_fc1"] = ln2 @ fc_w.T + fc_b
        gelu = out[f"block{L}_gelu"] = _gelu_np(fc1)
        proj_w = _to_f32(sd[f"transformer.h.{L}.mlp.c_proj.weight"])
        proj_b = _to_f32(sd[f"transformer.h.{L}.mlp.c_proj.bias"])
        fc2 = out[f"block{L}_fc2"] = gelu @ proj_w.T + proj_b
        x = out[f"block{L}_residual2"] = x + fc2

    ln_f = out["ln_f"] = _layernorm_np(x, ln_f_w, ln_f_b, eps)
    out["lm_head"] = ln_f[-1:] @ lm_head_w.T
    return out
