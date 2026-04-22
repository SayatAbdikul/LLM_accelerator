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

def _fq_linear(tensor) -> np.ndarray:
    """Per-channel INT8 quantisation dequantised with the MEAN channel scale.

    The codegen REQUANT uses a single scalar scale = mean(per_channel_scales),
    not per-channel dequant.  Using mean_scale here makes fake-quant arithmetic
    match the golden-model REQUANT output exactly (up to rounding).
    """
    from ..quantizer.quantize import quantize_tensor
    arr = _to_f32(tensor)  # [out, in]
    q, scales = quantize_tensor(arr, per_channel=True)
    mean_scale = np.float32(np.mean(scales.astype(np.float32)))
    return q.astype(np.float32) * mean_scale  # scalar dequant to match REQUANT


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

        from ..quantizer.quantize import quantize_tensor
        sd = state_dict
        # INT8 per-tensor embeddings (matching _quantize_embedding in tiny_fixture.py)
        wte_q, _ = quantize_tensor(_to_f32(sd["transformer.wte.weight"]), per_channel=False)
        wpe_q, _ = quantize_tensor(_to_f32(sd["transformer.wpe.weight"]), per_channel=False)
        self.wte_int8 = wte_q  # [V, d], INT8
        self.wpe_int8 = wpe_q  # [T, d], INT8

        self.ln_f_w = _to_f32(sd["transformer.ln_f.weight"])
        self.ln_f_b = _to_f32(sd["transformer.ln_f.bias"])
        self.lm_head_w = _fq_linear(sd["lm_head.weight"])        # [V, d]

        self.layers = []
        for L in range(self.n_layer):
            heads = []
            for H in range(self.n_head):
                heads.append((
                    _fq_linear(sd[f"transformer.h.{L}.attn.c_attn.weight_h{H}_query"]),
                    _fq_linear(sd[f"transformer.h.{L}.attn.c_attn.weight_h{H}_key"]),
                    _fq_linear(sd[f"transformer.h.{L}.attn.c_attn.weight_h{H}_value"]),
                ))
            self.layers.append({
                "ln1_w": _to_f32(sd[f"transformer.h.{L}.ln_1.weight"]),
                "ln1_b": _to_f32(sd[f"transformer.h.{L}.ln_1.bias"]),
                "ln2_w": _to_f32(sd[f"transformer.h.{L}.ln_2.weight"]),
                "ln2_b": _to_f32(sd[f"transformer.h.{L}.ln_2.bias"]),
                "heads": heads,
                "c_proj_w": _fq_linear(sd[f"transformer.h.{L}.attn.c_proj.weight"]),
                "c_proj_b": _to_f32(sd[f"transformer.h.{L}.attn.c_proj.bias"]),
                "fc_w": _fq_linear(sd[f"transformer.h.{L}.mlp.c_fc.weight"]),
                "fc_b": _to_f32(sd[f"transformer.h.{L}.mlp.c_fc.bias"]),
                "proj_w": _fq_linear(sd[f"transformer.h.{L}.mlp.c_proj.weight"]),
                "proj_b": _to_f32(sd[f"transformer.h.{L}.mlp.c_proj.bias"]),
            })

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
            ln1 = _qdq(
                _layernorm_np(x, layer["ln1_w"], layer["ln1_b"], self.eps),
                _scale(s, f"block{L}_ln1"),
            )
            head_outs_int8 = []
            for H, (q_w, k_w, v_w) in enumerate(layer["heads"]):
                q = _qdq(ln1 @ q_w.T, _scale(s, f"block{L}_head{H}_query"))
                k = _qdq(ln1 @ k_w.T, _scale(s, f"block{L}_head{H}_key"))
                v = _qdq(ln1 @ v_w.T, _scale(s, f"block{L}_head{H}_value"))

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

            out_proj = _qdq(
                concat @ layer["c_proj_w"].T + layer["c_proj_b"],
                _scale(s, f"block{L}_out_proj"),
            )
            # Residual 1: INT8 saturating VADD matching the compiled VADD instruction.
            out_proj_scale = _scale(s, f"block{L}_out_proj")
            x_int8 = _int8_saturating_add(x_int8, _fp32_to_int8(out_proj, out_proj_scale))
            x_scale = _scale(s, f"block{L}_residual1")
            x = x_int8.astype(np.float32) * np.float32(x_scale)

            # --- MLP ---
            ln2 = _qdq(
                _layernorm_np(x, layer["ln2_w"], layer["ln2_b"], self.eps),
                _scale(s, f"block{L}_ln2"),
            )
            fc1 = _qdq(ln2 @ layer["fc_w"].T + layer["fc_b"], _scale(s, f"block{L}_fc1"))
            gelu = _qdq(_gelu_np(fc1), _scale(s, f"block{L}_gelu", 1.0 / 127.0))
            fc2 = _qdq(
                gelu @ layer["proj_w"].T + layer["proj_b"],
                _scale(s, f"block{L}_fc2"),
            )
            # Residual 2: INT8 saturating VADD.
            fc2_scale = _scale(s, f"block{L}_fc2")
            x_int8 = _int8_saturating_add(x_int8, _fp32_to_int8(fc2, fc2_scale))
            x_scale = _scale(s, f"block{L}_residual2")
            x = x_int8.astype(np.float32) * np.float32(x_scale)

        ln_f = _qdq(_layernorm_np(x, self.ln_f_w, self.ln_f_b, self.eps), _scale(s, "ln_f"))
        logits = ln_f[-1:] @ self.lm_head_w.T  # [1, vocab_size]
        return _to_int8_logits(logits[0], _scale(s, "lm_head"))

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
            self._decode_incremental_step(tok, pos, caches)
            for tok, pos in zip(tids, pids)
        ]

    def _decode_incremental_step(self, token_id: int, position_id: int, caches) -> np.ndarray:
        s = self.scales
        x_int8 = _int8_saturating_add(
            self.wte_int8[[token_id]],
            self.wpe_int8[[position_id]],
        )
        x_scale = _scale(s, "tok_pos_add")
        x = x_int8.astype(np.float32) * np.float32(x_scale)

        for L, layer in enumerate(self.layers):
            ln1 = _qdq(
                _layernorm_np(x, layer["ln1_w"], layer["ln1_b"], self.eps),
                _scale(s, f"block{L}_ln1"),
            )
            head_outs_int8 = []
            for H, (q_w, k_w, v_w) in enumerate(layer["heads"]):
                q_scale = _scale(s, f"block{L}_head{H}_query")
                k_scale = _scale(s, f"block{L}_head{H}_key")
                v_scale = _scale(s, f"block{L}_head{H}_value")
                attn_v_scale = _scale(s, f"block{L}_head{H}_attn_v")

                q = _qdq(ln1 @ q_w.T, q_scale)
                k = _qdq(ln1 @ k_w.T, k_scale)
                v = _qdq(ln1 @ v_w.T, v_scale)
                k_i8 = _fp32_to_int8(k, k_scale)
                v_i8 = _fp32_to_int8(v, v_scale)
                caches[L][H]["k"].append(k_i8[0].copy())
                caches[L][H]["v"].append(v_i8[0].copy())

                k_cache = np.stack(caches[L][H]["k"], axis=0).astype(np.float32) * np.float32(k_scale)
                v_cache = np.stack(caches[L][H]["v"], axis=0).astype(np.float32) * np.float32(v_scale)
                attn = (q @ k_cache.T) * self.attn_scale
                row = attn[0].astype(np.float32)
                row_max = float(row.max())
                exp_row = np.exp(row - row_max)
                probs = (exp_row / float(exp_row.sum()))[None, :]
                probs = _qdq(probs, _scale(s, f"block{L}_head{H}_softmax", 1.0 / 127.0))
                head_out = _qdq(probs @ v_cache, attn_v_scale)
                head_outs_int8.append(_fp32_to_int8(head_out, attn_v_scale))

            concat_int8 = np.concatenate(head_outs_int8, axis=-1)
            concat_scale = _scale(s, f"block{L}_concat")
            concat = concat_int8.astype(np.float32) * np.float32(concat_scale)

            out_proj = _qdq(
                concat @ layer["c_proj_w"].T + layer["c_proj_b"],
                _scale(s, f"block{L}_out_proj"),
            )
            out_proj_scale = _scale(s, f"block{L}_out_proj")
            x_int8 = _int8_saturating_add(x_int8, _fp32_to_int8(out_proj, out_proj_scale))
            x_scale = _scale(s, f"block{L}_residual1")
            x = x_int8.astype(np.float32) * np.float32(x_scale)

            ln2 = _qdq(
                _layernorm_np(x, layer["ln2_w"], layer["ln2_b"], self.eps),
                _scale(s, f"block{L}_ln2"),
            )
            fc1 = _qdq(ln2 @ layer["fc_w"].T + layer["fc_b"], _scale(s, f"block{L}_fc1"))
            gelu = _qdq(_gelu_np(fc1), _scale(s, f"block{L}_gelu", 1.0 / 127.0))
            fc2 = _qdq(
                gelu @ layer["proj_w"].T + layer["proj_b"],
                _scale(s, f"block{L}_fc2"),
            )
            fc2_scale = _scale(s, f"block{L}_fc2")
            x_int8 = _int8_saturating_add(x_int8, _fp32_to_int8(fc2, fc2_scale))
            x_scale = _scale(s, f"block{L}_residual2")
            x = x_int8.astype(np.float32) * np.float32(x_scale)

        ln_f = _qdq(_layernorm_np(x, self.ln_f_w, self.ln_f_b, self.eps), _scale(s, "ln_f"))
        logits = ln_f @ self.lm_head_w.T
        return _to_int8_logits(logits[0], _scale(s, "lm_head"))


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
            q = out[f"block{L}_head{H}_query"] = ln1 @ q_w.T
            k = out[f"block{L}_head{H}_key"] = ln1 @ k_w.T
            v = out[f"block{L}_head{H}_value"] = ln1 @ v_w.T
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
