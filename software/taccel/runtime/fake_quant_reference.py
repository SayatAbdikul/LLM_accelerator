"""NumPy fake-quantised nanoGPT forward pass.

Mirrors the TACCEL compiler's quantisation scheme so the reference and the
golden-model simulator share the same per-tensor INT8 scales.  The reference
can run either a full causal forward or an incremental KV-cache decode path.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from ._ref_ops import cast_fp16, gelu_erf, gelu_tanh, layernorm, softmax_causal


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _to_f32(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def _to_fp16_widened_f32(x) -> np.ndarray:
    """Mirror golden-model parameter storage for FP16 gamma/beta vectors."""
    return cast_fp16(_to_f32(x))


def _arch_scale(scale: float) -> np.float32:
    """Scale-register value after FP16 storage and FP32 widening."""
    return np.float32(np.float16(np.float32(scale)))


def _qdq(x: np.ndarray, scale: float) -> np.ndarray:
    """INT8 quantise-dequantise matching sfu.py _requantize_int8.

    Rounds to nearest (round-half-to-even via NumPy default), clips to
    [-128, 127], then dequantises back to FP32.
    """
    if scale <= 0.0:
        return x.astype(np.float32)
    s = _arch_scale(scale)
    q = np.clip(np.round(x.astype(np.float32) / s), -128, 127).astype(np.int8)
    return q.astype(np.float32) * s


def _to_int8_logits(x: np.ndarray, scale: float) -> np.ndarray:
    """Quantise FP32 logits to INT8 — matches the golden-model STORE path."""
    if scale <= 0.0:
        return np.zeros_like(x, dtype=np.int8)
    return np.clip(np.round(x.astype(np.float32) / _arch_scale(scale)), -128, 127).astype(np.int8)


def _scale(scales: Dict[str, float], name: str, default: float = 6.0 / 127.0) -> float:
    return float(scales.get(name, default))


def _resolve_gelu_fn(model_args: dict):
    """Return the GELU function matching the model's activation_function."""
    name = str(model_args.get(
        "activation_function",
        "gelu_new" if bool(model_args.get("split_qkv_bias", False)) else "gelu",
    ))
    return gelu_tanh if name in {"gelu_new", "gelu_fast"} else gelu_erf


# ---------------------------------------------------------------------------
# Weight fake-quantisation at build time
# ---------------------------------------------------------------------------

def _linear_components(
    tensor,
    *,
    per_channel: bool = True,
) -> tuple[np.ndarray, np.float32, np.ndarray, np.ndarray]:
    """Return compiler-matched linear weight components.

    The codegen REQUANT uses a single scalar scale = mean(per_channel_scales),
    not per-channel dequant.  The compiler also pads output-channel scales to a
    16-wide boundary before taking that mean, so the reference mirrors that
    padding even when the logical vocab/output size is not a multiple of 16.
    """
    from ..quantizer.quantize import quantize_tensor
    arr = _to_f32(tensor)  # [out, in]
    q, scales = quantize_tensor(arr, per_channel=per_channel)
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
    return np.clip(np.round(x.astype(np.float32) / _arch_scale(scale)), -128, 127).astype(np.int8)


def _requant_accum_int8(accum: np.ndarray, accum_scale: float, out_scale: float) -> np.ndarray:
    """Requantize INT32 accumulators through the architectural FP16 scale reg."""
    if out_scale <= 0.0:
        return np.zeros_like(accum, dtype=np.int8)
    requant = np.float32(np.float16(np.float32(accum_scale) / np.float32(out_scale)))
    return np.clip(np.round(accum.astype(np.float32) * requant), -128, 127).astype(np.int8)


def _requant_accum_pc_int8(accum: np.ndarray, requant_pc: np.ndarray) -> np.ndarray:
    """Requantize INT32 accumulators with one architectural FP16 scale per column."""
    scales = np.asarray(requant_pc, dtype=np.float16).astype(np.float32).reshape(1, -1)
    return np.clip(np.round(accum.astype(np.float32) * scales[:, :accum.shape[-1]]), -128, 127).astype(np.int8)


def _dequant_accum_fp32(accum: np.ndarray, accum_scale: float) -> np.ndarray:
    """Dequantize INT32 accumulators through one architectural FP16 scale reg."""
    return accum.astype(np.float32) * _arch_scale(accum_scale)


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


def _bias_fp32(
    state_dict: dict,
    name: str,
    output_dim: int,
) -> np.ndarray:
    if name in state_dict:
        bias = _to_f32(state_dict[name]).reshape(-1)
    else:
        bias = np.zeros(int(output_dim), dtype=np.float32)
    if bias.size != int(output_dim):
        raise ValueError(f"{name!r} has {bias.size} values, expected {output_dim}")
    return bias.astype(np.float32)


def _bias_i32(state_dict: dict, name: str, act_scale: float, weight_scales: np.ndarray,
              output_dim: int) -> np.ndarray:
    bias = _bias_fp32(state_dict, name, output_dim)
    scales = np.asarray(weight_scales, dtype=np.float32)[:bias.size]
    denom = np.maximum(np.abs(np.float32(act_scale) * scales), 1e-10)
    return np.round(bias / denom).astype(np.int32)


_WEIGHT_COMPONENT_CACHE: dict[tuple[int, int, int, tuple[int, ...]], dict] = {}


def clear_weight_component_cache() -> None:
    """Clear cached fake-quant weight components used by NanoGPTFQReference."""
    _WEIGHT_COMPONENT_CACHE.clear()


def _cached_weight_components(state_dict: dict, model_args: dict, gelu_from_accum_blocks: set[int]) -> dict:
    key = (
        id(state_dict),
        int(model_args["n_layer"]),
        int(model_args["n_head"]),
        tuple(sorted(int(v) for v in gelu_from_accum_blocks)),
    )
    cached = _WEIGHT_COMPONENT_CACHE.get(key)
    if cached is not None:
        return cached

    sd = state_dict
    n_layer = int(model_args["n_layer"])
    n_head = int(model_args["n_head"])
    d_model = int(model_args["n_embd"])
    d_head = d_model // n_head
    components: dict[str, object] = {
        "wte_fp32": _to_f32(sd["transformer.wte.weight"]),
        "wpe_fp32": _to_f32(sd["transformer.wpe.weight"]),
        "ln_f_w": _to_fp16_widened_f32(sd["transformer.ln_f.weight"]),
        "ln_f_b": _to_fp16_widened_f32(sd["transformer.ln_f.bias"]),
        "lm_head": _linear_components(sd["lm_head.weight"]),
        "lm_head_w_fp32": _to_f32(sd["lm_head.weight"]),
        # Optional `lm_head.bias` exists when the model has been transformed
        # by `fold_layernorm_for_quarot` (β-fold of `ln_f` produces this).
        # Standard GPT-2 has no `lm_head.bias` — None signals "no bias to add".
        "lm_head_b_fp32": _to_f32(sd["lm_head.bias"]) if "lm_head.bias" in sd else None,
        "layers": [],
    }
    layers = components["layers"]
    for L in range(n_layer):
        heads = []
        for H in range(n_head):
            q_name = f"transformer.h.{L}.attn.c_attn.weight_h{H}_query"
            k_name = f"transformer.h.{L}.attn.c_attn.weight_h{H}_key"
            v_name = f"transformer.h.{L}.attn.c_attn.weight_h{H}_value"
            q_q, q_scale_w, q_w, q_scales = _linear_components(sd[q_name])
            k_q, k_scale_w, k_w, k_scales = _linear_components(sd[k_name])
            v_q, v_scale_w, v_w, v_scales = _linear_components(sd[v_name])
            q_b_name = f"transformer.h.{L}.attn.c_attn.bias_h{H}_query"
            k_b_name = f"transformer.h.{L}.attn.c_attn.bias_h{H}_key"
            v_b_name = f"transformer.h.{L}.attn.c_attn.bias_h{H}_value"
            heads.append((
                q_w,
                k_w,
                v_w,
                _to_f32(sd[q_name]),
                _to_f32(sd[k_name]),
                _to_f32(sd[v_name]),
                q_q,
                k_q,
                v_q,
                q_scale_w,
                k_scale_w,
                v_scale_w,
                q_scales,
                k_scales,
                v_scales,
                _bias_fp32(sd, q_b_name, d_head),
                _bias_fp32(sd, k_b_name, d_head),
                _bias_fp32(sd, v_b_name, d_head),
            ))
        c_proj_q, c_proj_scale_w, c_proj_w, c_proj_scales = _linear_components(sd[f"transformer.h.{L}.attn.c_proj.weight"])
        fc_q, fc_scale_w, fc_w, fc_scales = _linear_components(
            sd[f"transformer.h.{L}.mlp.c_fc.weight"],
            per_channel=L not in gelu_from_accum_blocks,
        )
        proj_q, proj_scale_w, proj_w, proj_scales = _linear_components(sd[f"transformer.h.{L}.mlp.c_proj.weight"])
        layers.append({
            "ln1_w": _to_fp16_widened_f32(sd[f"transformer.h.{L}.ln_1.weight"]),
            "ln1_b": _to_fp16_widened_f32(sd[f"transformer.h.{L}.ln_1.bias"]),
            "ln2_w": _to_fp16_widened_f32(sd[f"transformer.h.{L}.ln_2.weight"]),
            "ln2_b": _to_fp16_widened_f32(sd[f"transformer.h.{L}.ln_2.bias"]),
            "heads": heads,
            "c_proj_components": (c_proj_q, c_proj_scale_w, c_proj_w, c_proj_scales),
            "c_proj_w_fp32": _to_f32(sd[f"transformer.h.{L}.attn.c_proj.weight"]),
            "c_proj_b": _bias_fp32(sd, f"transformer.h.{L}.attn.c_proj.bias", d_model),
            "fc_components": (fc_q, fc_scale_w, fc_w, fc_scales),
            "fc_w_fp32": _to_f32(sd[f"transformer.h.{L}.mlp.c_fc.weight"]),
            "fc_b": _bias_fp32(sd, f"transformer.h.{L}.mlp.c_fc.bias", 4 * d_model),
            "proj_components": (proj_q, proj_scale_w, proj_w, proj_scales),
            "proj_w_fp32": _to_f32(sd[f"transformer.h.{L}.mlp.c_proj.weight"]),
            "proj_b": _bias_fp32(sd, f"transformer.h.{L}.mlp.c_proj.bias", d_model),
        })
    _WEIGHT_COMPONENT_CACHE[key] = components
    return components


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
        requant_pc_weight_names: Sequence[str] | None = None,
        raw_residual1_blocks: Sequence[int] | None = None,
        raw_residual2_blocks: Sequence[int] | None = None,
        gelu_from_accum_blocks: Sequence[int] | None = None,
        ln_eps: float | None = None,
    ) -> None:
        self.n_layer = int(model_args["n_layer"])
        self.n_head = int(model_args["n_head"])
        self.d_model = int(model_args["n_embd"])
        self.d_head = self.d_model // self.n_head
        self.vocab_size = int(model_args["vocab_size"])
        self.attn_scale = np.float32(self.d_head ** -0.5)
        # sfu.py uses 1e-6 hardcoded — not from model_args; ln_eps overrides for experiments
        self.eps = np.float32(ln_eps if ln_eps is not None else 1e-6)
        self.gelu_fn = _resolve_gelu_fn(model_args)
        self.scales = dict(scales)
        self.requant_pc_weight_names = set(str(v) for v in (requant_pc_weight_names or ()))
        self.raw_residual1_blocks = set(int(v) for v in (raw_residual1_blocks or ()))
        self.raw_residual2_blocks = set(int(v) for v in (raw_residual2_blocks or ()))
        self.gelu_from_accum_blocks = set(int(v) for v in (gelu_from_accum_blocks or ()))
        sd = state_dict
        self._has_nonzero_linear_bias = any(
            ("bias" in name and ("c_attn" in name or "c_proj" in name or "c_fc" in name))
            and np.any(np.abs(_to_f32(value)) > 0.0)
            for name, value in sd.items()
        )
        cached = _cached_weight_components(sd, model_args, self.gelu_from_accum_blocks)

        # INT8 embeddings are stored at the tok_pos_add scale because the
        # compiled program uses a raw INT8 VADD for token + position.  Separate
        # table scales would make q_token + q_pos numerically meaningless.
        self.wte_fp32 = cached["wte_fp32"]
        self.wpe_fp32 = cached["wpe_fp32"]
        embed_add_scale = _scale(self.scales, "tok_pos_add")
        wte_q = np.clip(np.round(self.wte_fp32 / np.float32(embed_add_scale)), -128, 127).astype(np.int8)
        wpe_q = np.clip(np.round(self.wpe_fp32 / np.float32(embed_add_scale)), -128, 127).astype(np.int8)
        self.wte_int8 = wte_q  # [V, d], INT8
        self.wpe_int8 = wpe_q  # [T, d], INT8

        self.ln_f_w = cached["ln_f_w"]
        self.ln_f_b = cached["ln_f_b"]
        self.lm_head_w_q, self.lm_head_w_scale, self.lm_head_w, self.lm_head_w_scales = cached["lm_head"]
        self.lm_head_w_fp32 = cached["lm_head_w_fp32"]
        self.lm_head_requant_pc = (
            np.float32(_scale(self.scales, "ln_f"))
            * self.lm_head_w_scales.astype(np.float32)
            / max(_scale(self.scales, "lm_head"), 1e-12)
        ).astype(np.float16)
        # Optional `lm_head.bias` (created by `fold_layernorm_for_quarot`'s
        # β-fold of `ln_f`). When present, the bias must be added in the
        # post-requant FP32 domain because INT8 requant is per-channel — we
        # store the bias in "logit units" (bias / lm_head_scale) so adding it
        # to the post-requant int values stays in INT8 logit space.
        self.lm_head_b_fp32 = cached.get("lm_head_b_fp32")
        if self.lm_head_b_fp32 is not None:
            lm_head_scale = max(_scale(self.scales, "lm_head"), 1e-12)
            self.lm_head_b_logit = (
                self.lm_head_b_fp32.astype(np.float32) / np.float32(lm_head_scale)
            )
        else:
            self.lm_head_b_logit = None

        self.layers = []
        for L, cached_layer in enumerate(cached["layers"]):
            heads = []
            for H, cached_head in enumerate(cached_layer["heads"]):
                (
                    q_w,
                    k_w,
                    v_w,
                    q_w_fp32,
                    k_w_fp32,
                    v_w_fp32,
                    q_q,
                    k_q,
                    v_q,
                    q_scale_w,
                    k_scale_w,
                    v_scale_w,
                    q_scales,
                    k_scales,
                    v_scales,
                    q_b,
                    k_b,
                    v_b,
                ) = cached_head
                ln1_scale = _scale(self.scales, f"block{L}_ln1")
                heads.append((
                    q_w,
                    k_w,
                    v_w,
                    q_w_fp32,
                    k_w_fp32,
                    v_w_fp32,
                    q_q,
                    k_q,
                    v_q,
                    q_scale_w,
                    k_scale_w,
                    v_scale_w,
                    q_scales,
                    k_scales,
                    v_scales,
                    np.round(q_b / np.maximum(np.abs(np.float32(ln1_scale) * q_scales.astype(np.float32)[:self.d_head]), 1e-10)).astype(np.int32),
                    np.round(k_b / np.maximum(np.abs(np.float32(ln1_scale) * k_scales.astype(np.float32)[:self.d_head]), 1e-10)).astype(np.int32),
                    np.round(v_b / np.maximum(np.abs(np.float32(ln1_scale) * v_scales.astype(np.float32)[:self.d_head]), 1e-10)).astype(np.int32),
                    q_b,
                    k_b,
                    v_b,
                ))
            c_proj_q, c_proj_scale_w, c_proj_w, c_proj_scales = cached_layer["c_proj_components"]
            fc_q, fc_scale_w, fc_w, fc_scales = cached_layer["fc_components"]
            proj_q, proj_scale_w, proj_w, proj_scales = cached_layer["proj_components"]
            c_proj_name = f"transformer.h.{L}.attn.c_proj.weight"
            fc_name = f"transformer.h.{L}.mlp.c_fc.weight"
            proj_name = f"transformer.h.{L}.mlp.c_proj.weight"
            self.layers.append({
                "ln1_w": cached_layer["ln1_w"],
                "ln1_b": cached_layer["ln1_b"],
                "ln2_w": cached_layer["ln2_w"],
                "ln2_b": cached_layer["ln2_b"],
                "heads": heads,
                "c_proj_w": c_proj_w,
                "c_proj_w_q": c_proj_q,
                "c_proj_w_scale": c_proj_scale_w,
                "c_proj_w_scales": c_proj_scales,
                "c_proj_requant_pc": (
                    (
                        np.float32(_scale(self.scales, f"block{L}_concat"))
                        * c_proj_scales.astype(np.float32)
                        / max(_scale(self.scales, f"block{L}_out_proj"), 1e-12)
                    ).astype(np.float16)
                    if c_proj_name in self.requant_pc_weight_names
                    else None
                ),
                "c_proj_w_fp32": cached_layer["c_proj_w_fp32"],
                "c_proj_b": cached_layer["c_proj_b"],
                "c_proj_b_i32": np.round(
                    cached_layer["c_proj_b"]
                    / np.maximum(np.abs(np.float32(_scale(self.scales, f"block{L}_concat")) * c_proj_scales.astype(np.float32)[:self.d_model]), 1e-10)
                ).astype(np.int32),
                "fc_w": fc_w,
                "fc_w_q": fc_q,
                "fc_w_scale": fc_scale_w,
                "fc_w_scales": fc_scales,
                "fc_requant_pc": (
                    (
                        np.float32(_scale(self.scales, f"block{L}_ln2"))
                        * fc_scales.astype(np.float32)
                        / max(_scale(self.scales, f"block{L}_fc1"), 1e-12)
                    ).astype(np.float16)
                    if fc_name in self.requant_pc_weight_names
                    else None
                ),
                "fc_w_fp32": cached_layer["fc_w_fp32"],
                "fc_b": cached_layer["fc_b"],
                "fc_b_i32": np.round(
                    cached_layer["fc_b"]
                    / np.maximum(np.abs(np.float32(_scale(self.scales, f"block{L}_ln2")) * fc_scales.astype(np.float32)[:4 * self.d_model]), 1e-10)
                ).astype(np.int32),
                "proj_w": proj_w,
                "proj_w_q": proj_q,
                "proj_w_scale": proj_scale_w,
                "proj_w_scales": proj_scales,
                "proj_requant_pc": (
                    (
                        np.float32(_scale(self.scales, f"block{L}_gelu", 1.0 / 127.0))
                        * proj_scales.astype(np.float32)
                        / max(_scale(self.scales, f"block{L}_fc2"), 1e-12)
                    ).astype(np.float16)
                    if proj_name in self.requant_pc_weight_names
                    else None
                ),
                "proj_w_fp32": cached_layer["proj_w_fp32"],
                "proj_b": cached_layer["proj_b"],
                "proj_b_i32": np.round(
                    cached_layer["proj_b"]
                    / np.maximum(np.abs(np.float32(_scale(self.scales, f"block{L}_gelu", 1.0 / 127.0)) * proj_scales.astype(np.float32)[:self.d_model]), 1e-10)
                ).astype(np.int32),
            })

    def _large_vocab_lm_head_reference(self) -> bool:
        """Use integer lm_head reference when codegen must tile the weight."""
        return int(self.lm_head_w_q.size) > 256 * 1024

    def _integer_linear_reference(self) -> bool:
        """Use bias-aware integer linear math for GPT-2-class converted models."""
        return self._large_vocab_lm_head_reference() or self._has_nonzero_linear_bias

    def _use_integer_linear(self, requant_pc: np.ndarray | None = None) -> bool:
        return self._integer_linear_reference() or requant_pc is not None

    def forward(
        self,
        token_ids: Sequence[int],
        position_ids: Optional[Sequence[int]] = None,
        *,
        return_all_logits: bool = False,
        capture: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """Return INT8 logits for the last position, or every causal position.

        If ``capture`` is supplied, intermediate fake-quant activations are
        written into it under the same names ``_fp32_forward`` uses (e.g.
        ``block{L}_ln2``, ``block{L}_gelu``, ``block{L}_concat``). This is a
        diagnostic hook for GPTQ Hessian capture — it copies the post-qdq
        FP32 view of the matmul input, matching the value the c_fc / c_proj
        / out_proj matmuls actually consume during fake-quant evaluation.
        """
        tids = list(token_ids)
        seq = len(tids)
        pids = list(position_ids) if position_ids is not None else list(range(seq))
        s = self.scales

        # Embedding: INT8 saturating VADD matching the compiled VADD instruction.
        # Both tables are stored INT8 (per-tensor quantized) and added as integers.
        x_int8 = _int8_saturating_add(self.wte_int8[tids], self.wpe_int8[pids])
        x_scale = _scale(s, "tok_pos_add")
        x = x_int8.astype(np.float32) * _arch_scale(x_scale)

        for L, layer in enumerate(self.layers):
            # --- Attention ---
            ln1_scale = _scale(s, f"block{L}_ln1")
            ln1 = _qdq(
                layernorm(x, layer["ln1_w"], layer["ln1_b"], self.eps),
                ln1_scale,
            )
            if capture is not None:
                capture[f"block{L}_ln1"] = np.asarray(ln1, dtype=np.float32).copy()
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
                    q = q_i8.astype(np.float32) * _arch_scale(q_scale)
                    k = k_i8.astype(np.float32) * _arch_scale(k_scale)
                    v = v_i8.astype(np.float32) * _arch_scale(v_scale)
                else:
                    q_w, k_w, v_w = head_weights[:3]
                    q = _qdq(ln1 @ q_w.T, q_scale)
                    k = _qdq(ln1 @ k_w.T, k_scale)
                    v = _qdq(ln1 @ v_w.T, v_scale)

                q_i8 = _fp32_to_int8(q, q_scale)
                k_i8 = _fp32_to_int8(k, k_scale)
                v_i8 = _fp32_to_int8(v, v_scale)
                if self._integer_linear_reference():
                    qkt_accum = q_i8.astype(np.int32) @ k_i8.astype(np.int32).T
                    attn = _dequant_accum_fp32(
                        qkt_accum,
                        np.float32(q_scale) * np.float32(k_scale) * np.float32(self.attn_scale),
                    )
                else:
                    attn = (q @ k.T) * self.attn_scale
                probs = np.ones((1, 1), dtype=np.float32) if seq == 1 else softmax_causal(attn)
                softmax_scale = _scale(s, f"block{L}_head{H}_softmax", 1.0 / 127.0)
                attn_v_scale = _scale(s, f"block{L}_head{H}_attn_v")
                probs = _qdq(probs, softmax_scale)
                probs_i8 = _fp32_to_int8(probs, softmax_scale)
                if self._integer_linear_reference():
                    head_accum = probs_i8.astype(np.int32) @ v_i8.astype(np.int32)
                    head_out_i8 = _requant_accum_int8(
                        head_accum,
                        np.float32(softmax_scale) * np.float32(v_scale),
                        attn_v_scale,
                    )
                else:
                    head_out = _qdq(probs @ v, attn_v_scale)
                    head_out_i8 = _fp32_to_int8(head_out, attn_v_scale)
                # Extract INT8: each head's output is stored in WBUF as INT8.
                head_outs_int8.append(head_out_i8)

            # Concat: INT8 values placed side-by-side in WBUF with no extra REQUANT.
            # The out_proj matmul then treats them with concat_scale as the input scale.
            concat_int8 = np.concatenate(head_outs_int8, axis=-1)
            concat_scale = _scale(s, f"block{L}_concat")
            concat = concat_int8.astype(np.float32) * _arch_scale(concat_scale)
            if capture is not None:
                capture[f"block{L}_concat"] = np.asarray(concat, dtype=np.float32).copy()

            concat_scale = _scale(s, f"block{L}_concat")
            out_proj_scale = _scale(s, f"block{L}_out_proj")
            out_accum = (
                concat_int8.astype(np.int32) @ layer["c_proj_w_q"].astype(np.int32).T
                + layer["c_proj_b_i32"]
            )
            out_accum_scale = np.float32(concat_scale) * np.float32(layer["c_proj_w_scale"])
            if layer["c_proj_requant_pc"] is not None:
                out_proj_i8 = _requant_accum_pc_int8(out_accum, layer["c_proj_requant_pc"])
            else:
                out_proj_i8 = _requant_accum_int8(out_accum, out_accum_scale, out_proj_scale)
            out_proj = out_proj_i8.astype(np.float32) * _arch_scale(out_proj_scale)
            # Residual 1: decoder bundles use DEQUANT_ADD, which consumes the
            # branch accumulator directly through FP16 rescale registers.
            prev_x_int8 = x_int8
            prev_x_scale = x_scale
            x_scale = _scale(s, f"block{L}_residual1")
            if L in self.raw_residual1_blocks:
                x_int8 = _int8_saturating_add(prev_x_int8, out_proj_i8)
            else:
                x_int8 = _dequant_add_accum_int8(out_accum, out_accum_scale, prev_x_int8, prev_x_scale, x_scale)
            x = x_int8.astype(np.float32) * _arch_scale(x_scale)

            # --- MLP ---
            ln2 = _qdq(
                layernorm(x, layer["ln2_w"], layer["ln2_b"], self.eps),
                _scale(s, f"block{L}_ln2"),
            )
            if capture is not None:
                capture[f"block{L}_ln2"] = np.asarray(ln2, dtype=np.float32).copy()
            fc1_scale = _scale(s, f"block{L}_fc1")
            ln2_scale = _scale(s, f"block{L}_ln2")
            ln2_i8 = _fp32_to_int8(ln2, ln2_scale)
            if self._use_integer_linear(layer["fc_requant_pc"]):
                fc1_accum = (
                    ln2_i8.astype(np.int32) @ layer["fc_w_q"].astype(np.int32).T
                    + layer["fc_b_i32"]
                )
                if L in self.gelu_from_accum_blocks:
                    fc1 = _dequant_accum_fp32(
                        fc1_accum,
                        np.float32(ln2_scale) * np.float32(layer["fc_w_scale"]),
                    )
                elif layer["fc_requant_pc"] is not None:
                    fc1_i8 = _requant_accum_pc_int8(fc1_accum, layer["fc_requant_pc"])
                    fc1 = fc1_i8.astype(np.float32) * _arch_scale(fc1_scale)
                else:
                    fc1_i8 = _requant_accum_int8(
                        fc1_accum,
                        np.float32(ln2_scale) * np.float32(layer["fc_w_scale"]),
                        fc1_scale,
                    )
                    fc1 = fc1_i8.astype(np.float32) * _arch_scale(fc1_scale)
            else:
                fc1 = _qdq(ln2 @ layer["fc_w"].T + layer["fc_b"], fc1_scale)
            gelu = _qdq(self.gelu_fn(fc1), _scale(s, f"block{L}_gelu", 1.0 / 127.0))
            if capture is not None:
                capture[f"block{L}_gelu"] = np.asarray(gelu, dtype=np.float32).copy()
            gelu_scale = _scale(s, f"block{L}_gelu", 1.0 / 127.0)
            gelu_i8 = _fp32_to_int8(gelu, gelu_scale)
            fc2_scale = _scale(s, f"block{L}_fc2")
            fc2_accum = (
                gelu_i8.astype(np.int32) @ layer["proj_w_q"].astype(np.int32).T
                + layer["proj_b_i32"]
            )
            fc2_accum_scale = np.float32(gelu_scale) * np.float32(layer["proj_w_scale"])
            if layer["proj_requant_pc"] is not None:
                fc2_i8 = _requant_accum_pc_int8(fc2_accum, layer["proj_requant_pc"])
            else:
                fc2_i8 = _requant_accum_int8(fc2_accum, fc2_accum_scale, fc2_scale)
            fc2 = fc2_i8.astype(np.float32) * _arch_scale(fc2_scale)
            prev_x_int8 = x_int8
            prev_x_scale = x_scale
            x_scale = _scale(s, f"block{L}_residual2")
            if L in self.raw_residual2_blocks:
                x_int8 = _int8_saturating_add(prev_x_int8, fc2_i8)
            else:
                x_int8 = _dequant_add_accum_int8(fc2_accum, fc2_accum_scale, prev_x_int8, prev_x_scale, x_scale)
            x = x_int8.astype(np.float32) * _arch_scale(x_scale)

        ln_f_scale = _scale(s, "ln_f")
        ln_f = _qdq(layernorm(x, self.ln_f_w, self.ln_f_b, self.eps), ln_f_scale)
        lm_input = ln_f if return_all_logits else ln_f[-1:]
        if not self._large_vocab_lm_head_reference():
            logits = lm_input @ self.lm_head_w.T
            if self.lm_head_b_fp32 is not None:
                logits = logits + self.lm_head_b_fp32
            logits_i8 = _to_int8_logits(logits, _scale(s, "lm_head"))
            return logits_i8 if return_all_logits else logits_i8[0]
        ln_f_i8 = _fp32_to_int8(lm_input, ln_f_scale)
        logits_accum = ln_f_i8.astype(np.int32) @ self.lm_head_w_q.astype(np.int32).T
        if self.lm_head_b_logit is not None:
            # Inline requant + bias to avoid double-saturation. Computing
            # `_requant_accum_pc_int8` first (which clips to [-128, 127])
            # and THEN adding bias loses precision on saturated logits and
            # accumulates error across positions at long evals.
            requant_pc = self.lm_head_requant_pc.astype(np.float16).astype(np.float32).reshape(1, -1)
            logits_fp32 = logits_accum.astype(np.float32) * requant_pc[:, :logits_accum.shape[-1]]
            logits_fp32 = logits_fp32 + self.lm_head_b_logit
            logits_i8 = np.clip(np.round(logits_fp32), -128, 127).astype(np.int8)
        else:
            logits_i8 = _requant_accum_pc_int8(logits_accum, self.lm_head_requant_pc)
        return logits_i8 if return_all_logits else logits_i8[0]

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

    @staticmethod
    def _mlp_group_active(groups: set[str], name: str, layer_idx: int) -> bool:
        return name in groups or f"{name}_block_{layer_idx}" in groups

    @staticmethod
    def _attn_group_active(groups: set[str], name: str, layer_idx: int, head_idx: int) -> bool:
        return (
            name in groups
            or f"{name}_block_{layer_idx}" in groups
            or f"{name}_head_{layer_idx}_{head_idx}" in groups
        )

    @staticmethod
    def _record_trace(
        trace: Optional[dict[str, dict[str, np.ndarray | float | None]]],
        name: str,
        value: np.ndarray,
        *,
        scale: Optional[float] = None,
        int8: Optional[np.ndarray] = None,
    ) -> None:
        """Append one entry to a `_decode_incremental_step` trace dict (no-op when ``trace is None``)."""
        if trace is None:
            return
        trace[name] = {
            "value": np.asarray(value, dtype=np.float32).copy(),
            "scale": None if scale is None else float(scale),
            "int8": None if int8 is None else np.asarray(int8, dtype=np.int8).copy(),
        }

    def _step_attention_head(
        self,
        L: int,
        H: int,
        ln1: np.ndarray,
        ln1_scale: float,
        ln1_i8: np.ndarray,
        head_weights,
        s: Dict[str, float],
        groups: set[str],
        caches_lh: dict,
        trace,
    ) -> np.ndarray:
        """One attention head: Q/K/V projections, KV-cache append, QKT,
        softmax, attn_v. Returns ``head_out_i8`` (per-head output INT8 row).

        Closure-free version of the original head loop body — every piece of
        state needed (scales dict, fp32_groups, per-head cache, trace sink)
        is passed in. Numerically identical to the inline form.
        """
        q_w, k_w, v_w, q_w_fp32, k_w_fp32, v_w_fp32 = head_weights[:6]
        q_q, k_q, v_q = head_weights[6:9]
        q_scale_w, k_scale_w, v_scale_w = head_weights[9:12]
        q_b_i32, k_b_i32, v_b_i32 = head_weights[15:18]
        q_b_fp32, k_b_fp32, v_b_fp32 = head_weights[18:21]
        q_scale = _scale(s, f"block{L}_head{H}_query")
        k_scale = _scale(s, f"block{L}_head{H}_key")
        v_scale = _scale(s, f"block{L}_head{H}_value")
        attn_v_scale = _scale(s, f"block{L}_head{H}_attn_v")

        value_fp32 = ln1 @ v_w_fp32.T + v_b_fp32
        if "qkv" in groups:
            q = ln1 @ q_w_fp32.T + q_b_fp32
            k = ln1 @ k_w_fp32.T + k_b_fp32
            v = value_fp32
        elif self._integer_linear_reference():
            q_accum = ln1_i8.astype(np.int32) @ q_q.astype(np.int32).T + q_b_i32
            k_accum = ln1_i8.astype(np.int32) @ k_q.astype(np.int32).T + k_b_i32
            v_accum = ln1_i8.astype(np.int32) @ v_q.astype(np.int32).T + v_b_i32
            q_i8 = _requant_accum_int8(q_accum, np.float32(ln1_scale) * np.float32(q_scale_w), q_scale)
            k_i8 = _requant_accum_int8(k_accum, np.float32(ln1_scale) * np.float32(k_scale_w), k_scale)
            v_i8 = _requant_accum_int8(v_accum, np.float32(ln1_scale) * np.float32(v_scale_w), v_scale)
            q = q_i8.astype(np.float32) * _arch_scale(q_scale)
            k = k_i8.astype(np.float32) * _arch_scale(k_scale)
            v = v_i8.astype(np.float32) * _arch_scale(v_scale)
        else:
            q = _qdq(ln1 @ q_w.T, q_scale)
            k = _qdq(ln1 @ k_w.T, k_scale)
            v = _qdq(ln1 @ v_w.T, v_scale)
        q_i8 = _fp32_to_int8(q, q_scale)
        k_i8 = _fp32_to_int8(k, k_scale)
        v_i8 = _fp32_to_int8(v, v_scale)
        self._record_trace(trace, f"block{L}_head{H}_query", q, scale=q_scale, int8=_fp32_to_int8(q, q_scale))
        self._record_trace(trace, f"block{L}_head{H}_key", k, scale=k_scale, int8=k_i8)
        self._record_trace(trace, f"block{L}_head{H}_value", v, scale=v_scale, int8=v_i8)
        caches_lh["k"].append(k_i8[0].copy())
        caches_lh["v"].append(v_i8[0].copy())
        caches_lh.setdefault("v_fp32", []).append(value_fp32[0].copy())

        k_cache_i8 = np.stack(caches_lh["k"], axis=0).astype(np.int8)
        v_cache_i8 = np.stack(caches_lh["v"], axis=0).astype(np.int8)
        v_cache = v_cache_i8.astype(np.float32) * _arch_scale(v_scale)
        v_cache_fp32 = np.stack(caches_lh["v_fp32"], axis=0).astype(np.float32)
        if self._integer_linear_reference():
            qkt_accum = q_i8.astype(np.int32) @ k_cache_i8.astype(np.int32).T
            attn = _dequant_accum_fp32(
                qkt_accum,
                np.float32(q_scale) * np.float32(k_scale) * np.float32(self.attn_scale),
            )
        else:
            k_cache = k_cache_i8.astype(np.float32) * _arch_scale(k_scale)
            attn = (q @ k_cache.T) * self.attn_scale
        row = attn[0].astype(np.float32)
        row_max = float(row.max())
        exp_row = np.exp(row - row_max)
        probs = (exp_row / float(exp_row.sum()))[None, :]
        softmax_scale = _scale(s, f"block{L}_head{H}_softmax", 1.0 / 127.0)
        if not self._attn_group_active(groups, "softmax", L, H):
            probs = _qdq(probs, softmax_scale)
        probs_i8 = _fp32_to_int8(probs, softmax_scale)
        self._record_trace(trace, f"block{L}_head{H}_softmax", probs, scale=softmax_scale, int8=probs_i8)
        attn_v_fp32 = self._attn_group_active(groups, "attn_v", L, H)
        value_cache_fp32 = self._attn_group_active(groups, "attn_v_value_fp32", L, H)
        if not (attn_v_fp32 or value_cache_fp32):
            if self._integer_linear_reference():
                head_accum = probs_i8.astype(np.int32) @ v_cache_i8.astype(np.int32)
                head_out_i8 = _requant_accum_int8(
                    head_accum,
                    np.float32(softmax_scale) * np.float32(v_scale),
                    attn_v_scale,
                )
                head_out = head_out_i8.astype(np.float32) * _arch_scale(attn_v_scale)
            else:
                head_out = _qdq(probs @ v_cache, attn_v_scale)
                head_out_i8 = _fp32_to_int8(head_out, attn_v_scale)
        else:
            head_out = probs @ (v_cache_fp32 if value_cache_fp32 else v_cache)
            head_out_i8 = _fp32_to_int8(head_out, attn_v_scale)
        self._record_trace(trace, f"block{L}_head{H}_attn_v", head_out, scale=attn_v_scale, int8=head_out_i8)
        return head_out_i8

    def _step_attention_block(
        self,
        L: int,
        layer: dict,
        x: np.ndarray,
        x_int8: np.ndarray,
        x_scale_prev: float,
        s: Dict[str, float],
        groups: set[str],
        caches_L: list,
        trace,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """LN1 + multi-head attention + concat + out_proj + residual1.

        ``x``/``x_int8`` come in as the pre-block state (embedding output or
        previous block's residual2). ``x_scale_prev`` is its scale (only
        consulted by the integer ``dequant_add_accum`` residual path).
        Returns the post-residual1 ``(x, x_int8, x_scale)`` triple.
        """
        ln1_scale = _scale(s, f"block{L}_ln1")
        ln1_raw = layernorm(x, layer["ln1_w"], layer["ln1_b"], self.eps)
        ln1 = _qdq(ln1_raw, ln1_scale)
        self._record_trace(trace, f"block{L}_ln1", ln1, scale=ln1_scale, int8=_fp32_to_int8(ln1, ln1_scale))
        ln1_i8 = _fp32_to_int8(ln1, ln1_scale)
        head_outs_int8 = [
            self._step_attention_head(
                L, H, ln1, ln1_scale, ln1_i8, head_weights,
                s, groups, caches_L[H], trace,
            )
            for H, head_weights in enumerate(layer["heads"])
        ]

        concat_int8 = np.concatenate(head_outs_int8, axis=-1)
        concat_scale = _scale(s, f"block{L}_concat")
        concat = concat_int8.astype(np.float32) * _arch_scale(concat_scale)
        self._record_trace(trace, f"block{L}_concat", concat, scale=concat_scale, int8=concat_int8)

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
            if layer["c_proj_requant_pc"] is not None:
                out_proj_i8 = _requant_accum_pc_int8(out_accum, layer["c_proj_requant_pc"])
            else:
                out_proj_i8 = _requant_accum_int8(out_accum, out_accum_scale, out_proj_scale)
            out_proj = out_proj_i8.astype(np.float32) * _arch_scale(out_proj_scale)
        self._record_trace(trace, f"block{L}_out_proj", out_proj, scale=out_proj_scale, int8=out_proj_i8)

        x_scale = _scale(s, f"block{L}_residual1")
        if L in self.raw_residual1_blocks and "residual_vadd" not in groups:
            x_int8 = _int8_saturating_add(x_int8, out_proj_i8)
            x = x_int8.astype(np.float32) * _arch_scale(x_scale)
        elif "residual_vadd" in groups or out_accum is None:
            x = _qdq(x + out_proj, x_scale)
            x_int8 = _fp32_to_int8(x, x_scale)
        else:
            x_int8 = _dequant_add_accum_int8(
                out_accum, out_accum_scale, x_int8, x_scale_prev, x_scale,
            )
            x = x_int8.astype(np.float32) * _arch_scale(x_scale)
        self._record_trace(trace, f"block{L}_residual1", x, scale=x_scale, int8=x_int8)
        return x, x_int8, x_scale

    def _step_mlp_block(
        self,
        L: int,
        layer: dict,
        x: np.ndarray,
        x_int8: np.ndarray,
        x_scale_residual1: float,
        s: Dict[str, float],
        groups: set[str],
        trace,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """LN2 + FC1 + GELU + FC2 + residual2. Returns (x, x_int8, x_scale).

        ``x``/``x_int8`` come in as the post-residual1 state; the method
        produces the post-residual2 state and the f"block{L}_residual2"
        scale. ``mlp_*`` fp32_groups switch individual matmuls back to
        FP32 weights; the rest of the path stays INT8.
        """
        def mlp_group_active(name: str) -> bool:
            return self._mlp_group_active(groups, name, L)

        ln2_scale = _scale(s, f"block{L}_ln2")
        ln2_raw = layernorm(x, layer["ln2_w"], layer["ln2_b"], self.eps)
        ln2 = _qdq(ln2_raw, ln2_scale)
        self._record_trace(trace, f"block{L}_ln2", ln2, scale=ln2_scale, int8=_fp32_to_int8(ln2, ln2_scale))
        fc1_scale = _scale(s, f"block{L}_fc1")
        gelu_scale = _scale(s, f"block{L}_gelu", 1.0 / 127.0)
        fc2_scale = _scale(s, f"block{L}_fc2")
        full_mlp_fp32 = (
            "mlp" in groups
            or "mlp_full" in groups
            or f"mlp_block_{L}" in groups
        )
        fc1_fp32 = (
            full_mlp_fp32
            or mlp_group_active("mlp_fc1")
            or mlp_group_active("mlp_fc1_gelu")
            or mlp_group_active("mlp_fc1_gelu_fc2")
        )
        gelu_fp32 = (
            full_mlp_fp32
            or mlp_group_active("mlp_gelu")
            or mlp_group_active("mlp_fc1_gelu")
            or mlp_group_active("mlp_gelu_fc2")
            or mlp_group_active("mlp_fc1_gelu_fc2")
        )
        fc2_fp32 = (
            full_mlp_fp32
            or mlp_group_active("mlp_fc2")
            or mlp_group_active("mlp_gelu_fc2")
            or mlp_group_active("mlp_fc1_gelu_fc2")
        )

        fc1_accum = None
        if fc1_fp32:
            fc1_raw = ln2 @ layer["fc_w_fp32"].T + layer["fc_b"]
            fc1 = fc1_raw if gelu_fp32 else _qdq(fc1_raw, fc1_scale)
        elif self._use_integer_linear(layer["fc_requant_pc"]):
            ln2_i8 = _fp32_to_int8(ln2, ln2_scale)
            fc1_accum = (
                ln2_i8.astype(np.int32) @ layer["fc_w_q"].astype(np.int32).T
                + layer["fc_b_i32"]
            )
            if L in self.gelu_from_accum_blocks:
                fc1 = _dequant_accum_fp32(
                    fc1_accum,
                    np.float32(ln2_scale) * np.float32(layer["fc_w_scale"]),
                )
            elif layer["fc_requant_pc"] is not None:
                fc1_i8 = _requant_accum_pc_int8(fc1_accum, layer["fc_requant_pc"])
                fc1 = fc1_i8.astype(np.float32) * _arch_scale(fc1_scale)
            else:
                fc1_i8 = _requant_accum_int8(
                    fc1_accum,
                    np.float32(ln2_scale) * np.float32(layer["fc_w_scale"]),
                    fc1_scale,
                )
                fc1 = fc1_i8.astype(np.float32) * _arch_scale(fc1_scale)
        else:
            fc1 = _qdq(ln2 @ layer["fc_w"].T + layer["fc_b"], fc1_scale)

        gelu_raw = self.gelu_fn(fc1)
        gelu = gelu_raw if (gelu_fp32 or fc2_fp32) else _qdq(gelu_raw, gelu_scale)
        fc2_accum = None
        fc2_accum_scale = None
        if fc2_fp32:
            fc2 = gelu @ layer["proj_w_fp32"].T + layer["proj_b"]
            fc2_i8 = _fp32_to_int8(_qdq(fc2, fc2_scale), fc2_scale)
        else:
            gelu = _qdq(gelu, gelu_scale)
            gelu_i8 = _fp32_to_int8(gelu, gelu_scale)
            fc2_accum = (
                gelu_i8.astype(np.int32) @ layer["proj_w_q"].astype(np.int32).T
                + layer["proj_b_i32"]
            )
            fc2_accum_scale = np.float32(gelu_scale) * np.float32(layer["proj_w_scale"])
            if layer["proj_requant_pc"] is not None:
                fc2_i8 = _requant_accum_pc_int8(fc2_accum, layer["proj_requant_pc"])
            else:
                fc2_i8 = _requant_accum_int8(fc2_accum, fc2_accum_scale, fc2_scale)
            fc2 = fc2_i8.astype(np.float32) * _arch_scale(fc2_scale)
        fc1_trace_scale = (
            float(np.float32(ln2_scale) * np.float32(layer["fc_w_scale"]))
            if L in self.gelu_from_accum_blocks and fc1_accum is not None
            else fc1_scale
        )
        self._record_trace(
            trace, f"block{L}_fc1", fc1, scale=fc1_trace_scale,
            int8=None if L in self.gelu_from_accum_blocks and fc1_accum is not None else _fp32_to_int8(fc1, fc1_scale),
        )
        self._record_trace(trace, f"block{L}_gelu", gelu, scale=gelu_scale, int8=_fp32_to_int8(gelu, gelu_scale))
        self._record_trace(trace, f"block{L}_fc2", fc2, scale=fc2_scale, int8=fc2_i8)

        residual1_int8 = x_int8
        residual1_scale = x_scale_residual1
        x_scale = _scale(s, f"block{L}_residual2")
        if L in self.raw_residual2_blocks and "residual_vadd" not in groups:
            x_int8 = _int8_saturating_add(residual1_int8, fc2_i8)
            x = x_int8.astype(np.float32) * _arch_scale(x_scale)
        elif "residual_vadd" in groups or fc2_accum is None:
            x = _qdq(x + fc2, x_scale)
            x_int8 = _fp32_to_int8(x, x_scale)
        else:
            x_int8 = _dequant_add_accum_int8(
                fc2_accum, fc2_accum_scale, residual1_int8, residual1_scale, x_scale,
            )
            x = x_int8.astype(np.float32) * _arch_scale(x_scale)
        self._record_trace(trace, f"block{L}_residual2", x, scale=x_scale, int8=x_int8)
        return x, x_int8, x_scale

    def _step_embedding(
        self,
        token_id: int,
        position_id: int,
        s: Dict[str, float],
        groups: set[str],
        trace,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """tok+pos embedding lookup. Returns (x, x_int8, x_scale)."""
        x_scale = _scale(s, "tok_pos_add")
        if "embeddings" in groups:
            x = self.wte_fp32[[token_id]] + self.wpe_fp32[[position_id]]
            x_int8 = _fp32_to_int8(_qdq(x, x_scale), x_scale)
        else:
            x_int8 = _int8_saturating_add(
                self.wte_int8[[token_id]],
                self.wpe_int8[[position_id]],
            )
            x = x_int8.astype(np.float32) * _arch_scale(x_scale)
        self._record_trace(trace, "tok_pos_add", x, scale=x_scale, int8=x_int8)
        return x, x_int8, x_scale

    def _step_final(
        self,
        x: np.ndarray,
        s: Dict[str, float],
        groups: set[str],
        trace,
    ) -> np.ndarray:
        """ln_f + lm_head. Returns logits in the shape expected by callers
        (FP32 1-D array for fp32-output ``groups``, INT8 1-D row otherwise).

        Six lm_head paths are dispatched by ``fp32_groups``:
          - ``"lm_head"``                — FP32 weights, FP32 output (returned as FP32).
          - ``"lm_head_weight_fp32"``    — FP32 weights → INT8 logits (last row).
          - ``"lm_head_requant_fp32"``   — INT8 accum × FP32 PC scale / lm_head_scale.
          - ``"lm_head_output_fp32"``    — INT8 accum × FP32 PC scale (no clip).
          - Default (small vocab)        — FP32 logits → INT8 (single matmul).
          - Default (large vocab)        — INT8 accum + PC-requant (optionally
                                            with FP32 bias added pre-clip to
                                            avoid double-saturation).
        """
        ln_f_scale = _scale(s, "ln_f")
        ln_f_raw = layernorm(x, self.ln_f_w, self.ln_f_b, self.eps)
        ln_f = ln_f_raw if "ln_f" in groups else _qdq(ln_f_raw, ln_f_scale)
        self._record_trace(trace, "ln_f", ln_f, scale=ln_f_scale, int8=_fp32_to_int8(ln_f, ln_f_scale))
        if "lm_head" in groups:
            logits = ln_f @ self.lm_head_w_fp32.T
            self._record_trace(trace, "lm_head", logits, scale=None, int8=None)
            return logits[0].astype(np.float32)
        if "lm_head_weight_fp32" in groups:
            logits_fp32 = ln_f[-1:] @ self.lm_head_w_fp32.T
            return _to_int8_logits(logits_fp32[0], _scale(s, "lm_head"))
        if "lm_head_requant_fp32" in groups:
            ln_f_i8 = _fp32_to_int8(ln_f, ln_f_scale)
            logits_accum = ln_f_i8.astype(np.int32) @ self.lm_head_w_q.astype(np.int32).T
            requant_f32 = (
                np.float32(ln_f_scale)
                * self.lm_head_w_scales.astype(np.float32)
                / max(np.float32(_scale(s, "lm_head")), np.float32(1e-12))
            )
            return _requant_accum_pc_int8(logits_accum, requant_f32)[0]
        if "lm_head_output_fp32" in groups:
            ln_f_i8 = _fp32_to_int8(ln_f, ln_f_scale)
            logits_accum = ln_f_i8.astype(np.int32) @ self.lm_head_w_q.astype(np.int32).T
            # requant_pc may be padded to 16-wide boundary; slice to actual vocab size.
            # Returns accum * requant_pc = true_logits / lm_head_scale so that the
            # caller's normal dequant (* lm_scale) produces true FP32 logits without INT8 clipping.
            requant_pc = self.lm_head_requant_pc.astype(np.float32)[:logits_accum.shape[-1]]
            return (logits_accum.astype(np.float32) * requant_pc)[0]
        if not self._large_vocab_lm_head_reference():
            logits = ln_f @ self.lm_head_w.T
            if self.lm_head_b_fp32 is not None:
                logits = logits + self.lm_head_b_fp32
            logits_i8 = _to_int8_logits(logits[0], _scale(s, "lm_head"))
            self._record_trace(
                trace, "lm_head",
                logits_i8.astype(np.float32) * _arch_scale(_scale(s, "lm_head")),
                scale=_scale(s, "lm_head"),
                int8=logits_i8,
            )
            return logits_i8
        ln_f_i8 = _fp32_to_int8(ln_f, ln_f_scale)
        logits_accum = ln_f_i8.astype(np.int32) @ self.lm_head_w_q.astype(np.int32).T
        if self.lm_head_b_logit is not None:
            # Inline requant + bias to avoid double-saturation (see forward()
            # for explanation).
            requant_pc = self.lm_head_requant_pc.astype(np.float16).astype(np.float32).reshape(1, -1)
            logits_fp32 = logits_accum.astype(np.float32) * requant_pc[:, :logits_accum.shape[-1]]
            logits_fp32 = logits_fp32 + self.lm_head_b_logit
            logits_i8 = np.clip(np.round(logits_fp32), -128, 127).astype(np.int8)
        else:
            logits_i8 = _requant_accum_pc_int8(logits_accum, self.lm_head_requant_pc)
        logits = logits_i8.astype(np.float32) * _arch_scale(_scale(s, "lm_head"))
        self._record_trace(
            trace, "lm_head", logits,
            scale=_scale(s, "lm_head"),
            int8=logits_i8,
        )
        return logits_i8[0]

    def _decode_incremental_step(
        self,
        token_id: int,
        position_id: int,
        caches,
        trace: Optional[dict[str, dict[str, np.ndarray | float | None]]] = None,
        fp32_groups: Optional[set[str]] = None,
    ) -> np.ndarray:
        """One incremental decode step: tok+pos embedding → 12 blocks of
        attention+MLP → ln_f → lm_head. State (KV cache) flows in/out via
        ``caches``; per-node trace flows via ``trace``.
        """
        s = self.scales
        groups = set(fp32_groups or ())
        x, x_int8, x_scale = self._step_embedding(token_id, position_id, s, groups, trace)
        for L, layer in enumerate(self.layers):
            x, x_int8, x_scale = self._step_attention_block(
                L, layer, x, x_int8, x_scale, s, groups, caches[L], trace,
            )
            x, x_int8, x_scale = self._step_mlp_block(
                L, layer, x, x_int8, x_scale, s, groups, trace,
            )
        return self._step_final(x, s, groups, trace)


# ---------------------------------------------------------------------------
# FP32-only forward for scale calibration (no QDQ)
# ---------------------------------------------------------------------------

def _fp32_forward(
    state_dict: dict,
    model_args: dict,
    token_ids: Sequence[int],
    ln_eps: float | None = None,
) -> Dict[str, np.ndarray]:
    """Run FP32 forward and return per-node outputs keyed by IR node name."""
    n_layer = int(model_args["n_layer"])
    n_head = int(model_args["n_head"])
    d_model = int(model_args["n_embd"])
    d_head = d_model // n_head
    eps = ln_eps if ln_eps is not None else float(model_args.get("layer_norm_epsilon", 1e-5))
    _gelu_fn = _resolve_gelu_fn(model_args)
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
        ln1 = out[f"block{L}_ln1"] = layernorm(x, ln1_w, ln1_b, eps)

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
            probs = np.ones((1, 1), dtype=np.float32) if seq == 1 else softmax_causal(attn)
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
        ln2 = out[f"block{L}_ln2"] = layernorm(x, ln2_w, ln2_b, eps)
        fc_w = _to_f32(sd[f"transformer.h.{L}.mlp.c_fc.weight"])
        fc_b = _to_f32(sd[f"transformer.h.{L}.mlp.c_fc.bias"])
        fc1 = out[f"block{L}_fc1"] = ln2 @ fc_w.T + fc_b
        gelu = out[f"block{L}_gelu"] = _gelu_fn(fc1)
        proj_w = _to_f32(sd[f"transformer.h.{L}.mlp.c_proj.weight"])
        proj_b = _to_f32(sd[f"transformer.h.{L}.mlp.c_proj.bias"])
        fc2 = out[f"block{L}_fc2"] = gelu @ proj_w.T + proj_b
        x = out[f"block{L}_residual2"] = x + fc2

    ln_f = out["ln_f"] = layernorm(x, ln_f_w, ln_f_b, eps)
    # Optional `lm_head.bias` is created by `fold_layernorm_for_quarot`
    # (β-fold of `ln_f` produces this). For models without QuaRot, the key
    # is absent and the bias contribution is zero.
    lm_head_logits = ln_f[-1:] @ lm_head_w.T
    if "lm_head.bias" in sd:
        lm_head_logits = lm_head_logits + _to_f32(sd["lm_head.bias"])
    out["lm_head"] = lm_head_logits
    return out
