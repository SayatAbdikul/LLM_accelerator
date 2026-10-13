"""Independent NumPy reference for the single-stream W8A16 decoder.

The reference models FP16-stored embeddings, normalization parameters, biases
and activations; INT8 weight bins are chosen using FP32 scales before the
scales are stored as FP16. Dynamic matmuls use FP16 inverse/forward activation
scales, INT32 products, and FP32 dequantization followed by an FP16 store.
Streamed FC2 uses a separate scale for each K partition, first-partition bias,
and ordered FP16 partial sums. This differs intentionally from the FP32 weight
QDQ ceiling in WeightOnlyHostRunner.

Attention quantizes with a stored FP16 reciprocal, not division by the original
calibration float. Q/K/V defaults are 6/127; softmax defaults to 1/127. K/V cache
bytes are quantized once using decode-side scale aliases. The first prefill
attention consumes direct projections; later steps consume the cached INT8
bytes. Composite scale products use FP32 arithmetic before FP16 storage.

Pass attention_window equal to the compiled decode key window for exact padded
softmax reductions. If omitted, incremental research callers use the smallest
16-aligned window covering the current prefix. A one-token prefill always has
a 16-column physical window. FP16/reconstructed cache shadows are retained for
TurboQuant diagnostics; that optional perturbation is research-only.

This module does not import or execute compiler emitters or the golden model.
Agreement with the compiled implementation is a separate conformance gate,
not evidence that quantized model quality matches the untouched FP32 model.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from ._ref_ops import cast_fp16, layernorm, softmax_masked
from taccel.isa.opcodes import ABUF_SIZE, ACCUM_SIZE, WBUF_SIZE
from taccel.quantizer.turboquant import TurboQuantKV


def _pad_to_multiple(arr: np.ndarray, axis: int, multiple: int = 16) -> np.ndarray:
    """Right-pad `arr` along `axis` to a multiple of `multiple` with zeros."""
    n = arr.shape[axis]
    pad = (-n) % multiple
    if pad == 0:
        return arr
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (0, pad)
    return np.pad(arr, pad_width, mode="constant", constant_values=0)


def _quant_w_per_channel(w_fp32: np.ndarray, *, bitwidth: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Quantize FP32 weight `[K, N]` to signed integer with per-column FP16 scales.

    Mirrors `_torch_qdq_per_channel` math but returns the raw integer +
    scales (no QDQ back to FP32). The simulator-backed codegen consumes
    these directly via `weight_data`.

    Storage is always ``np.int8`` regardless of ``bitwidth`` — INT4 values
    (``bitwidth=4``) live in the low nibble. Default ``bitwidth=8`` keeps
    every existing callsite byte-identical (W4A16 plan Phase 1).
    """
    if w_fp32.ndim != 2:
        raise ValueError(f"expected 2-D weight, got shape {w_fp32.shape}")
    if bitwidth < 2 or bitwidth > 8:
        raise ValueError(f"bitwidth must be in [2, 8], got {bitwidth}")
    qmax = (1 << (bitwidth - 1)) - 1   # 127 (INT8) or 7 (INT4)
    qmin = -(1 << (bitwidth - 1))      # -128 (INT8) or -8 (INT4)
    max_per_col = np.maximum(np.max(np.abs(w_fp32), axis=0), 1e-8)
    # Quantize using the FP32 scale, then round the separately stored scale.
    # Using the stored FP16 scale to choose integer weights changes RTN bins.
    scales_fp32 = max_per_col / float(qmax)
    w_int8 = np.clip(
        np.round(w_fp32 / scales_fp32.reshape(1, -1)),
        qmin, qmax,
    ).astype(np.int8)
    return w_int8, scales_fp32.astype(np.float16)


def _quant_w_per_channel_int8(w_fp32: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Back-compat alias for ``_quant_w_per_channel(..., bitwidth=8)``.

    Kept so external/test imports stay green; new internal callers should
    use ``_quant_w_per_channel(..., bitwidth=cfg)`` directly.
    """
    return _quant_w_per_channel(w_fp32, bitwidth=8)


def _w8a16_dynamic_matmul(
    x_fp32: np.ndarray,
    w_int8: np.ndarray,
    w_scales_fp16: np.ndarray,
    bias_fp32: Optional[np.ndarray] = None,
    *,
    k_tile_size: Optional[int] = None,
) -> np.ndarray:
    """One INT8-activation × INT8-weight matmul with FP16 dequant.

    Matches the codegen's `emit_matmul_w8a16` (W8A16) semantics exactly:
      - max_abs over the full activation tile (one MAX_ABS_REDUCE) after
        FP16 round-trip on the source (matches simulator's FP16-store path).
      - FP16 inv_scale, FP16 fwd_scale (DEQUANT_ACCUM_FP32_SCALED sreg+1).
      - INT8 round-half-to-even (numpy default → matches QUANT_FP32_INT8).
      - Per-column FP16 weight scales applied after the INT32 accum.
      - Bias read from FP16 storage, added in FP32 BEFORE FP16 cast.
      - Output: FP16 widened to FP32 (matches simulator's FP16-store path).

    With ``k_tile_size``, each K partition has its own dynamic scale and
    FP16 partial output. The bias belongs only to the first partition;
    subsequent partials are added in order with an FP16 store after each
    addition. This is the streamed FC2 path, not one global INT8 matmul.
    """
    if k_tile_size is not None:
        if isinstance(k_tile_size, bool) or int(k_tile_size) != k_tile_size or k_tile_size <= 0:
            raise ValueError("k_tile_size must be a positive integer")
        k_tile_size = int(k_tile_size)
        result = None
        for start in range(0, x_fp32.shape[-1], k_tile_size):
            stop = min(start + k_tile_size, x_fp32.shape[-1])
            partial = _w8a16_dynamic_matmul(
                x_fp32[..., start:stop], w_int8[start:stop], w_scales_fp16,
                bias_fp32 if start == 0 else None,
            )
            result = partial if result is None else cast_fp16(result + partial)
        if result is None:
            raise ValueError("streamed matmul requires a nonempty K dimension")
        return result
    # The simulator reads ABUF FP16 and widens to FP32 internally. Mirror
    # that here so max_abs is computed over the same precision-rounded values.
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
    # Bias added BEFORE FP16 cast (matches the simulator's bias-fold contract).
    if bias_fp32 is not None:
        y_fp32 = y_fp32 + cast_fp16(np.asarray(bias_fp32, dtype=np.float32))
    return y_fp32.astype(np.float16).astype(np.float32)


def _fc2_k_partition_size(d_model: int, mlp_dim: int) -> Optional[int]:
    """Independent single-query-row storage policy for the active FC2 path.

    A tiled FC1 spills its output; GELU preserves that DRAM storage. FC2
    streams it only if FC2 itself requires tiling. The dense decoder uses
    16 physical rows, a maximum 512-column output tile, and half-WBUF
    weight tiles. The full-K admission reserves four bytes per input
    element (the conservative staging policy), even with FP16 storage.
    No compiler/emitter execution is used to construct this reference.
    """
    tile, rows = 16, 16
    width = ((int(d_model) + tile - 1) // tile) * tile
    inner = ((int(mlp_dim) + tile - 1) // tile) * tile

    def needs_tiling(k, n):
        return (rows * n * 2 > ABUF_SIZE or rows * n * 4 > ACCUM_SIZE
                or k * n + 4 * n > WBUF_SIZE)

    if not (needs_tiling(width, inner) and needs_tiling(inner, width)):
        return None
    budget = WBUF_SIZE // 2
    n_cap = min(width, 512, ACCUM_SIZE // (rows * 4), ABUF_SIZE // rows)
    n_cap = max(tile, (n_cap // tile) * tile)
    if rows * inner * 4 <= ABUF_SIZE and budget // inner >= tile:
        return inner
    while budget // n_cap < tile:
        n_cap = max(tile, ((n_cap // 2) // tile) * tile)
    return min(inner, max(tile, ((budget // n_cap) // tile) * tile))


def _gelu_fp16(x: np.ndarray) -> np.ndarray:
    """FP16 storage and explicit FP32 gelu_new arithmetic, including constants."""
    xf = cast_fp16(x)
    inner = np.float32(np.sqrt(2.0 / np.pi)) * (xf + np.float32(0.044715) * xf ** 3)
    return cast_fp16(xf * np.float32(0.5) * (np.float32(1.0) + np.tanh(inner)))


class NanoGPTW8A16SimulatorReference:
    """W8A16 reference matching the single-query-row storage/arithmetic contract.

    See module docstring. Constructs INT8 weights + FP16 scales from
    the FP32 state_dict and runs the forward pass with per-matmul
    dynamic activation quantization for non-attention matmuls and
    static composite scales for QKT / attn_v.
    """

    def __init__(self, payload: dict, *, default_act_scale: float = 6.0 / 127.0,
                 calibration_scales: Optional[Dict[str, float]] = None,
                 kv_quant: Optional[TurboQuantKV] = None,
                 weight_bitwidth: int = 8,
                 lm_head_bitwidth: Optional[int] = None,
                 weight_overrides: Optional[Dict[str, tuple]] = None,
                 attention_window: Optional[int] = None) -> None:
        # The active decoder stores quantized INT8 K/V. Keep FP16/reconstructed
        # shadows for research diagnostics, not as the decode-side operands.
        # When set, the KV cache stores the lossy TurboQuant round-trip.
        # weight_bitwidth=8 (default) preserves the W8A16 contract
        # byte-identical; weight_bitwidth=4 selects W4A16 per-channel.
        #
        # lm_head_bitwidth: optional override for the LM head specifically.
        # If None, defaults to:
        #   - weight_bitwidth itself when weight_bitwidth >= 8 (no change),
        #   - **8** when weight_bitwidth < 8 (standard W4 practice: lm_head
        #     row scale is dominated by one outlier per vocab token, so
        #     per-channel W4 destroys the discriminative info → near-random
        #     logits. Empirically verified on GPT-2 124M 2026-05-23: W4
        #     lm_head gives 40,306 PPL while W8 lm_head + W4 blocks gives
        #     73.59 PPL — a 548× difference traced to lm_head alone).
        # Pass lm_head_bitwidth=weight_bitwidth explicitly to force full-W4
        # (research / ablation only — not production).
        self.kv_quant = kv_quant
        if attention_window is not None:
            if (isinstance(attention_window, bool) or int(attention_window) != attention_window
                    or attention_window < 1):
                raise ValueError("attention_window must be a positive integer")
            attention_window = int(attention_window)
        self.attention_window = attention_window
        self.payload = payload
        self.weight_bitwidth = int(weight_bitwidth)
        if lm_head_bitwidth is None:
            self.lm_head_bitwidth = max(int(weight_bitwidth), 8)
        else:
            self.lm_head_bitwidth = int(lm_head_bitwidth)
        cfg = payload["model_args"]
        sd = payload["state_dict"]
        self.n_layer = int(cfg["n_layer"])
        self.n_head = int(cfg["n_head"])
        self.d_model = int(cfg["n_embd"])
        self.d_head = self.d_model // self.n_head
        self.mlp_dim = int(cfg.get("mlp_dim", 4 * self.d_model))
        self.fc2_k_tile_size = _fc2_k_partition_size(self.d_model, self.mlp_dim)
        self.vocab_size = int(cfg["vocab_size"])
        self.block_size = int(cfg["block_size"])
        self.layer_norm_epsilon = float(cfg.get("layer_norm_epsilon", 1e-5))
        self.inv_sqrt_d_head = self.d_head ** -0.5
        self.default_act_scale = float(default_act_scale)
        self.calibration_scales = dict(calibration_scales or {})
        # Local bound for terser callsite spelling below. Two flavors:
        #   _qpc       — block weights at self.weight_bitwidth
        #   _qpc_head  — lm_head specifically (at self.lm_head_bitwidth)
        # When `weight_overrides` is provided and the sd-key is in it, the
        # override (int8, fp16_scales) is used INSTEAD of internal RTN —
        # this is how GPTQ-quantized weights are injected without losing
        # GPTQ's per-element rounding decisions to the simulator's
        # re-quantization round-trip (W4A16 Tier B).
        self._weight_overrides = dict(weight_overrides or {})

        def _qpc_for(key, w):
            ov = self._weight_overrides.get(key)
            if ov is not None:
                return ov  # (int8_tensor, fp16_scales) already; same shape contract
            return _quant_w_per_channel(w, bitwidth=self.weight_bitwidth)

        def _qpc_head_for(key, w):
            ov = self._weight_overrides.get(key)
            if ov is not None:
                return ov
            return _quant_w_per_channel(w, bitwidth=self.lm_head_bitwidth)

        # Back-compat: keep the bare lambdas for callers that don't pass keys.
        _qpc = lambda w: _quant_w_per_channel(w, bitwidth=self.weight_bitwidth)
        _qpc_head = lambda w: _quant_w_per_channel(w, bitwidth=self.lm_head_bitwidth)
        self._quant_w = _qpc
        self._quant_w_head = _qpc_head
        self._qpc_for = _qpc_for
        self._qpc_head_for = _qpc_head_for

        # Each table is stored separately as FP16 before their FP32 sum.
        self.wte = cast_fp16(np.asarray(sd["transformer.wte.weight"], dtype=np.float32))
        self.wpe = cast_fp16(np.asarray(sd["transformer.wpe.weight"], dtype=np.float32))

        # Final LN + lm_head.
        self.ln_f_w = cast_fp16(np.asarray(sd["transformer.ln_f.weight"], dtype=np.float32))
        self.ln_f_b = cast_fp16(np.asarray(sd["transformer.ln_f.bias"], dtype=np.float32))
        # lm_head: weight is [vocab, d_model] in checkpoint; W8A32 emits
        # `lm_head` as `x @ W.T` where W is [vocab, d_model]. So store
        # the transposed-shape INT8: w_int8 is [d_model, vocab].
        lm_head_fp32 = np.asarray(sd["lm_head.weight"], dtype=np.float32)
        # lm_head uses the head-specific bitwidth (default W8 even under W4).
        self.lm_head_w_int8, self.lm_head_w_scales = self._qpc_head_for(
            "lm_head.weight", lm_head_fp32.T,
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
                "ln1_w": cast_fp16(np.asarray(sd[f"{prefix}.ln_1.weight"], dtype=np.float32)),
                "ln1_b": cast_fp16(np.asarray(sd[f"{prefix}.ln_1.bias"], dtype=np.float32)),
                "ln2_w": cast_fp16(np.asarray(sd[f"{prefix}.ln_2.weight"], dtype=np.float32)),
                "ln2_b": cast_fp16(np.asarray(sd[f"{prefix}.ln_2.bias"], dtype=np.float32)),
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
                    sd.get(f"{head_prefix}.bias_h{head_idx}_query", np.zeros(self.d_head)), dtype=np.float32
                )
                k_b = np.asarray(
                    sd.get(f"{head_prefix}.bias_h{head_idx}_key", np.zeros(self.d_head)), dtype=np.float32
                )
                v_b = np.asarray(
                    sd.get(f"{head_prefix}.bias_h{head_idx}_value", np.zeros(self.d_head)), dtype=np.float32
                )
                # Store INT8 + FP16 scales (transposed to [K, N] = [d_model, d_head]).
                q_int8, q_scales = self._qpc_for(
                    f"{head_prefix}.weight_h{head_idx}_query", q_fp32.T)
                k_int8, k_scales = self._qpc_for(
                    f"{head_prefix}.weight_h{head_idx}_key", k_fp32.T)
                v_int8, v_scales = self._qpc_for(
                    f"{head_prefix}.weight_h{head_idx}_value", v_fp32.T)
                layer["heads"].append({
                    "q_int8": q_int8, "q_scales": q_scales, "q_b": q_b,
                    "k_int8": k_int8, "k_scales": k_scales, "k_b": k_b,
                    "v_int8": v_int8, "v_scales": v_scales, "v_b": v_b,
                })
            c_proj_fp32 = np.asarray(
                sd[f"{prefix}.attn.c_proj.weight"], dtype=np.float32
            )
            layer["c_proj_int8"], layer["c_proj_scales"] = self._qpc_for(
                f"{prefix}.attn.c_proj.weight", c_proj_fp32.T)
            layer["c_proj_b"] = np.asarray(
                sd[f"{prefix}.attn.c_proj.bias"], dtype=np.float32
            )
            fc_fp32 = np.asarray(sd[f"{prefix}.mlp.c_fc.weight"], dtype=np.float32)
            layer["fc1_int8"], layer["fc1_scales"] = self._qpc_for(
                f"{prefix}.mlp.c_fc.weight", fc_fp32.T)
            layer["fc1_b"] = np.asarray(
                sd[f"{prefix}.mlp.c_fc.bias"], dtype=np.float32
            )
            proj_fp32 = np.asarray(sd[f"{prefix}.mlp.c_proj.weight"], dtype=np.float32)
            layer["fc2_int8"], layer["fc2_scales"] = self._qpc_for(
                f"{prefix}.mlp.c_proj.weight", proj_fp32.T)
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
            [{"k": [], "v": [], "k_int8": [], "v_int8": []} for _ in range(self.n_head)]
            for _ in range(self.n_layer)
        ]
        self._next_position = 0

    def _act_scale_for(self, name: str) -> float:
        default = 1.0 / 127.0 if name.endswith("_softmax") else self.default_act_scale
        return float(self.calibration_scales.get(name, default))

    def _qkt_attn_v_quantize(self, x_fp32: np.ndarray, static_scale: float) -> np.ndarray:
        """Static-scale INT8 quantization for QKT / attn_v inputs.

        Matches the codegen path's `q_scale * k_scale * inv_sqrt_d_head`
        composite — Q and K each get quantized with their *own* static
        calibration scale (NOT a dynamic max_abs), then INT8-matmul,
        then dequant with the composite FP16 factor.
        """
        inverse = np.float32(np.float16(1.0 / max(static_scale, 1e-12)))
        return np.clip(
            np.round(cast_fp16(x_fp32) * inverse),
            -128, 127,
        ).astype(np.int8)

    def _attention_head(self, ln1: np.ndarray, layer_idx: int, head_idx: int,
                        position: int, record=None) -> np.ndarray:
        head = self.layers[layer_idx]["heads"][head_idx]
        # Per-head Q/K/V projections (dynamic activation scale).
        q = _w8a16_dynamic_matmul(
            ln1, head["q_int8"], head["q_scales"], head["q_b"],
        )
        k = _w8a16_dynamic_matmul(
            ln1, head["k_int8"], head["k_scales"], head["k_b"],
        )
        v = _w8a16_dynamic_matmul(
            ln1, head["v_int8"], head["v_scales"], head["v_b"],
        )
        if record is not None:
            # Same node names / pre-cache-append order as
            # NanoGPTFP32Reference._decode_incremental_step so the A/B
            # harness can diff sublayer-by-sublayer.
            record(f"block{layer_idx}_head{head_idx}_query", q)
            record(f"block{layer_idx}_head{head_idx}_key", k)
            record(f"block{layer_idx}_head{head_idx}_value", v)
        # Store-time quantization uses the decode-side alias even on prefill.
        # Prefill attention itself still consumes the projected K/V directly.
        # TurboQuant is a separate research perturbation before this boundary;
        # it is not an assertion that a compiled TurboQuant path exists.
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
        prefix = f"block{layer_idx}_head{head_idx}"
        k_decode_scale = self._act_scale_for(f"{prefix}_key_kv_load")
        v_decode_scale = self._act_scale_for(f"{prefix}_value_kv_load")
        self._caches[layer_idx][head_idx]["k_int8"].append(
            self._qkt_attn_v_quantize(k_store, k_decode_scale))
        self._caches[layer_idx][head_idx]["v_int8"].append(
            self._qkt_attn_v_quantize(v_store, v_decode_scale))
        if record is not None and tq is not None:
            # Cache-level reconstruction nodes for the deep verifier
            # (Level-2 diffs these directly, not just downstream attn).
            record(f"block{layer_idx}_head{head_idx}_k_cache_tq", k_store)
            record(f"block{layer_idx}_head{head_idx}_v_cache_tq", v_store)
        k_cache = np.stack(self._caches[layer_idx][head_idx]["k"], axis=0)
        v_cache = np.stack(self._caches[layer_idx][head_idx]["v"], axis=0)

        # QKT: static composite scale.
        q_scale = self._act_scale_for(f"block{layer_idx}_head{head_idx}_query")
        k_scale = (self._act_scale_for(f"{prefix}_key") if position == 0 else k_decode_scale)
        composite_qkt = np.float16(
            np.float32(q_scale) * np.float32(k_scale) * np.float32(self.inv_sqrt_d_head)
        )
        # Q/K source widened from FP16 if applicable (matches QUANT input).
        q_int8 = self._qkt_attn_v_quantize(cast_fp16(q), q_scale)
        k_int8 = (self._qkt_attn_v_quantize(cast_fp16(k_cache), k_scale) if position == 0 else
                  np.stack(self._caches[layer_idx][head_idx]["k_int8"], axis=0))
        scores_int32 = q_int8.astype(np.int32) @ k_int8.astype(np.int32).T
        scores_fp32 = scores_int32.astype(np.float32) * np.float32(composite_qkt)
        # QKT output stored at FP precision.
        scores_fp32 = cast_fp16(scores_fp32)

        # Masked softmax with valid_kv_len = position + 1.
        valid_kv_len = position + 1
        # Softmax's FP32 reduction includes the zero exponentials of masked
        # padded columns. Their presence changes NumPy reduction grouping,
        # so match the compiled physical width, not only the live prefix.
        key_width = 16 if position == 0 else (
            ((self.attention_window or valid_kv_len) + 15) // 16 * 16)
        scores_fp32 = np.pad(scores_fp32, ((0, 0), (0, key_width - valid_kv_len)))
        probs_fp32 = softmax_masked(
            scores_fp32, valid_kv_len, fp16_storage=True,
        )
        if record is not None:
            record(f"block{layer_idx}_head{head_idx}_softmax", probs_fp32[:, :valid_kv_len])

        # attn_v: static composite scale.
        sm_scale = self._act_scale_for(f"block{layer_idx}_head{head_idx}_softmax")
        v_scale = (self._act_scale_for(f"{prefix}_value") if position == 0 else v_decode_scale)
        composite_av = np.float16(np.float32(sm_scale) * np.float32(v_scale))
        sm_int8 = self._qkt_attn_v_quantize(probs_fp32, sm_scale)
        v_cache_int8 = (self._qkt_attn_v_quantize(cast_fp16(v_cache), v_scale) if position == 0 else
                        np.stack(self._caches[layer_idx][head_idx]["v_int8"], axis=0))
        v_cache_int8 = np.pad(v_cache_int8, ((0, key_width - valid_kv_len), (0, 0)))
        head_out_int32 = sm_int8.astype(np.int32) @ v_cache_int8.astype(np.int32)
        head_out_fp32 = head_out_int32.astype(np.float32) * np.float32(composite_av)
        head_out = cast_fp16(head_out_fp32)
        if record is not None:
            record(f"block{layer_idx}_head{head_idx}_attn_v", head_out)
        return head_out

    def run_decode_step(self, token_id: int, position: int,
                        trace: Optional[dict] = None) -> np.ndarray:
        """Run one decode step and return its FP32 logits.

        When `trace` is a dict, records per-sublayer activations under the
        same node names `NanoGPTFP32Reference` uses (see
        `incremental_node_trace`) so the two can be diffed sublayer-by-
        sublayer to localize where W8A16 quantization damage accumulates.
        """
        if position != self._next_position:
            raise ValueError(
                f"NanoGPTW8A16SimulatorReference: position {position} doesn't "
                f"match internal cursor {self._next_position}"
            )
        if self.attention_window is not None and position >= self.attention_window:
            raise ValueError("position exceeds the compiled attention_window")

        def record(name: str, value) -> None:
            if trace is None:
                return
            trace[name] = {"value": np.asarray(value, dtype=np.float32).copy()}

        # Token + position embedding lookups. Storage is FP{32,16} matching
        # how the codegen stages embedding tables.
        x = self.wte[token_id:token_id + 1] + self.wpe[position:position + 1]
        x = cast_fp16(x)
        record("tok_pos_add", x)

        for layer_idx, layer in enumerate(self.layers):
            ln1 = layernorm(
                x, layer["ln1_w"], layer["ln1_b"],
                eps=self.layer_norm_epsilon, fp16_storage=True,
            )
            record(f"block{layer_idx}_ln1", ln1)
            head_outs = []
            for head_idx in range(self.n_head):
                head_outs.append(
                    self._attention_head(
                        ln1, layer_idx, head_idx, position, record=record
                    )
                )
            concat = np.concatenate(head_outs, axis=-1)
            record(f"block{layer_idx}_concat", concat)
            out_proj = _w8a16_dynamic_matmul(
                concat, layer["c_proj_int8"], layer["c_proj_scales"], layer["c_proj_b"],
            )
            record(f"block{layer_idx}_out_proj", out_proj)
            x = cast_fp16(x + out_proj)  # residual1
            record(f"block{layer_idx}_residual1", x)

            ln2 = layernorm(
                x, layer["ln2_w"], layer["ln2_b"],
                eps=self.layer_norm_epsilon, fp16_storage=True,
            )
            record(f"block{layer_idx}_ln2", ln2)
            fc1 = _w8a16_dynamic_matmul(
                ln2, layer["fc1_int8"], layer["fc1_scales"], layer["fc1_b"],
            )
            record(f"block{layer_idx}_fc1", fc1)
            gelu = _gelu_fp16(fc1)
            record(f"block{layer_idx}_gelu", gelu)
            fc2 = _w8a16_dynamic_matmul(
                gelu, layer["fc2_int8"], layer["fc2_scales"], layer["fc2_b"],
                k_tile_size=self.fc2_k_tile_size,
            )
            record(f"block{layer_idx}_fc2", fc2)
            x = cast_fp16(x + fc2)  # residual2
            record(f"block{layer_idx}_residual2", x)

        ln_f = layernorm(
            x, self.ln_f_w, self.ln_f_b,
            eps=self.layer_norm_epsilon, fp16_storage=True,
        )
        record("ln_f", ln_f)
        # lm_head: takes only the last row of ln_f (incremental decode).
        logits = _w8a16_dynamic_matmul(
            ln_f[-1:], self.lm_head_w_int8, self.lm_head_w_scales, self.lm_head_b,
        )
        record("lm_head", logits)
        self._next_position += 1
        return logits[0].astype(np.float32)

    def incremental_node_trace(
        self,
        token_ids: Sequence[int],
        position_ids: Optional[Sequence[int]] = None,
    ) -> List[dict]:
        """Per-node W8A16 trace, node names matching NanoGPTFP32Reference."""
        toks = [int(t) for t in token_ids]
        if not toks:
            return []
        pids = list(position_ids) if position_ids is not None else list(range(len(toks)))
        if len(pids) != len(toks):
            raise ValueError("position_ids length must match token_ids length")
        self._reset_caches()
        traces: List[dict] = []
        for tok, pos in zip(toks, pids):
            step_trace: dict = {}
            self.run_decode_step(tok, pos, trace=step_trace)
            traces.append(step_trace)
        return traces

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
