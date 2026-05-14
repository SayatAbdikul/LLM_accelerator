"""Internal helpers for Stage 3 tiny nanoGPT ProgramBundle tests."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..compiler.decoder_bundle import (
    DecoderBundleBuild,
    build_decoder_program_bundle,
    inject_kv_cache_nodes,
    mark_runtime_embedding_lookups,
)
from ..compiler.frontend.nanogpt_adapter import load_nanogpt
from ..compiler.model_config import ModelConfig
from ..compiler.tiler import pad_dim
from ..quantizer.quantize import quantize_tensor
from ..quantizer.scales import ScalePropagator
from .calibration import build_calibration_scales
from .fake_quant import cosine_similarity
from .fake_quant_reference import NanoGPTFQReference
from .fp32_reference import NanoGPTFP32Reference
from .host_runner import HostRunner
from .stage5_ptq import (
    Stage5PTQPreset,
    apply_stage5_ptq_scale_policy,
    resolve_stage5_ptq_preset,
    stage5_dequant_add_residual1_blocks,
    stage5_dequant_add_residual2_blocks,
    stage5_gelu_from_accum_blocks,
    stage5_raw_residual1_blocks,
    stage5_raw_residual2_blocks,
    stage5_requant_pc_weight_names,
    validate_stage5_ptq_preset_for_model,
)


@dataclass
class TinyFixtureBundle:
    build: DecoderBundleBuild
    config: ModelConfig
    logits_size: int


@dataclass
class TinyDecodeTrace:
    generated: List[int]
    logits: List[np.ndarray]


@dataclass
class TinyE2EResult:
    generated: List[int]
    reference_generated: List[int]
    cosine_per_step: List[float]
    top5_overlap_per_step: List[int]
    logits: List[np.ndarray]
    reference_logits: List[np.ndarray]

    @property
    def min_cosine(self) -> float:
        return min(self.cosine_per_step) if self.cosine_per_step else 1.0


@dataclass
class TinyFP32E2EResult:
    generated: List[int]
    fp32_generated: List[int]
    logits: List[np.ndarray]
    fp32_logits: List[np.ndarray]
    top5_overlap_per_step: List[int]
    fp32_top1_in_golden_top5_per_step: List[bool]
    golden_top1_in_fp32_top5_per_step: List[bool]
    top1_match_per_step: List[bool]
    fp32_cosine_per_step: List[float]

    @property
    def min_top5_overlap(self) -> int:
        return min(self.top5_overlap_per_step) if self.top5_overlap_per_step else 5

    @property
    def fp32_top1_in_golden_top5_all(self) -> bool:
        return all(self.fp32_top1_in_golden_top5_per_step)

    @property
    def golden_top1_in_fp32_top5_all(self) -> bool:
        return all(self.golden_top1_in_fp32_top5_per_step)

    @property
    def top1_match_rate(self) -> float:
        if not self.top1_match_per_step:
            return 1.0
        return float(sum(bool(v) for v in self.top1_match_per_step) / len(self.top1_match_per_step))

    @property
    def min_fp32_cosine(self) -> float:
        return min(self.fp32_cosine_per_step) if self.fp32_cosine_per_step else 1.0


def _mark_gelu_from_accum_inline(graph, blocks: set[int]) -> None:
    """Inline selected nanoGPT FC1->GELU pairs so GELU can consume ACCUM."""
    for block in sorted(int(v) for v in blocks):
        fc1 = graph.get_node(f"block{block}_fc1")
        gelu = graph.get_node(f"block{block}_gelu")
        if fc1 is None or gelu is None:
            continue
        fc1.attrs["inline_gelu"] = gelu.name
        gelu.attrs["inline_with"] = fc1.name


def _to_numpy(tensor) -> np.ndarray:
    if hasattr(tensor, "detach"):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def _quantize_embedding(tensor, scale: Optional[float] = None) -> np.ndarray:
    arr = _to_numpy(tensor).astype(np.float32)
    if scale is None:
        q, _ = quantize_tensor(arr, per_channel=False)
        return q.astype(np.int8)
    if scale <= 0.0:
        raise ValueError("embedding scale must be positive")
    return np.clip(np.round(arr / np.float32(scale)), -128, 127).astype(np.int8)


def _fp32_embedding(tensor) -> np.ndarray:
    """Raw FP32 embedding table for the W8A32 path (M3-prep).

    The codegen DMA-loads `d_model_pad * 4` bytes per row in W8A32 mode
    and the resulting ABUF tile is interpreted as FP32 by the next
    sub-layer op. No quantization, no scale — the embedding output is
    real-units FP32 throughout.
    """
    return _to_numpy(tensor).astype(np.float32)


def _fp16_embedding(tensor) -> np.ndarray:
    """Raw FP16 embedding table for the W8A16 path (M2-W8A16).

    The codegen DMA-loads `d_model_pad * 2` bytes per row in W8A16 mode.
    The resulting ABUF tile is interpreted as FP16 by the next sub-layer
    op (which widens to FP32 internally for LN/GELU/softmax).
    """
    return _to_numpy(tensor).astype(np.float16)


def _quantize_linear_weight(tensor, *, per_channel: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Quantize a PyTorch-layout [out, in] weight into TACCEL [K, N] layout."""
    q, scales = quantize_tensor(_to_numpy(tensor).astype(np.float32), per_channel=per_channel)
    q = np.pad(
        q,
        ((0, (16 - q.shape[0] % 16) % 16), (0, (16 - q.shape[1] % 16) % 16)),
        mode="constant",
    )
    if len(scales) < q.shape[0]:
        scales = np.pad(scales, (0, q.shape[0] - len(scales)), constant_values=scales[-1])
    return np.ascontiguousarray(q.T).astype(np.int8), scales.astype(np.float16)


def _fp16_vector(tensor) -> np.ndarray:
    arr = _to_numpy(tensor).astype(np.float16)
    return np.pad(arr, (0, (16 - len(arr) % 16) % 16), mode="constant")


def _prescale_bias(
    state_dict: Dict[str, object],
    name: str,
    *,
    output_dim: int,
    act_scale: float,
    weight_scales: np.ndarray,
) -> np.ndarray:
    """Return compiler-domain INT32 bias, padded to the matmul output width."""
    output_pad = pad_dim(int(output_dim))
    if name in state_dict:
        arr = _to_numpy(state_dict[name]).astype(np.float32)
    else:
        arr = np.zeros(int(output_dim), dtype=np.float32)
    if arr.size != int(output_dim):
        raise ValueError(f"{name!r} has {arr.size} values, expected {output_dim}")
    scales = np.asarray(weight_scales, dtype=np.float32)
    if scales.size < output_pad:
        scales = np.pad(scales, (0, output_pad - scales.size), constant_values=scales[-1])
    bias_i32 = ScalePropagator().prescale_bias(
        arr,
        np.asarray([float(act_scale)], dtype=np.float32),
        scales[:arr.size],
    )
    return np.pad(bias_i32, (0, output_pad - bias_i32.size), constant_values=0).astype(np.int32)


def _fp32_bias(
    state_dict: Dict[str, object],
    name: str,
    *,
    output_dim: int,
) -> np.ndarray:
    """Return raw FP32 bias for the W8A32 path (M2.5-C), padded to N_pad.

    The W8A32 matmul lowering (`emit_matmul_w8a16`) adds the bias in
    FP32 *after* the DEQUANT epilogue, so the bias is staged unscaled —
    just the original FP32 values from the state dict, padded to the
    matmul output width. Missing biases yield zeros (same fallback as
    `_prescale_bias`).
    """
    output_pad = pad_dim(int(output_dim))
    if name in state_dict:
        arr = _to_numpy(state_dict[name]).astype(np.float32)
    else:
        arr = np.zeros(int(output_dim), dtype=np.float32)
    if arr.size != int(output_dim):
        raise ValueError(f"{name!r} has {arr.size} values, expected {output_dim}")
    return np.pad(arr, (0, output_pad - arr.size), constant_values=0).astype(np.float32)


def model_config_from_fixture_payload(payload: Dict[str, object]) -> ModelConfig:
    cfg = payload["model_args"]
    d_model = int(cfg["n_embd"])
    n_head = int(cfg["n_head"])
    return ModelConfig(
        name="stage3-tiny-fixture",
        model_kind="decoder",
        n_layer=int(cfg["n_layer"]),
        n_head=n_head,
        d_model=d_model,
        d_head=d_model // n_head,
        mlp_dim=4 * d_model,
        vocab_size=int(cfg["vocab_size"]),
        max_seq_len=int(cfg["block_size"]),
        embedding_kind="token_pos",
        norm_epsilon=float(cfg.get("layer_norm_epsilon", 1e-5)),
    )


def quantize_fixture_payload(
    payload: Dict[str, object],
    calibration_scales: Optional[Dict[str, float]] = None,
    per_tensor_fc1_blocks: Optional[set[int]] = None,
    use_fp16_activations: bool = False,
):
    """Return codegen-ready weights, biases, FP32 biases, scales, config, and logits size.

    Returns a 6-tuple:
      (weight_data, prescaled_biases, biases, calibration_scales,
       config, logits_size)

    `biases` holds raw FP32 bias values (padded to N_pad) for
    every matmul bias the W8A32 lowering will need to consume:

      M2.5-C (always staged, since these resolve through
      `emit_matmul_w8a16` once the matmul lowering exists):
        - `transformer.h.{layer}.attn.c_proj.bias` (out_proj)
        - `transformer.h.{layer}.mlp.c_fc.bias`     (fc1)
        - `transformer.h.{layer}.mlp.c_proj.bias`   (fc2)
        - `lm_head.bias` (when present)

      M3-prep (staged for the attention-internal per-head matmuls; sit
      behind the matmul_qkt/matmul_attn_v guardrail until M3 lifts it,
      but the staging is a hidden M3 blocker so we resolve it now):
        - `transformer.h.{layer}.attn.c_attn.bias_h{H}_{proj}`
          for proj ∈ {query, key, value}, h ∈ [0, n_head).

    `use_fp16_activations` (M3-prep, default False): when True, the token and
    position embedding tables are stored as raw FP32 (4 bytes/elem)
    instead of being quantized to INT8. The codegen's
    `_emit_embedding_lookup` reads `d_model_pad * 4` bytes per row in
    that mode so the next op (a sub-layer FP32 op) sees real-units
    FP32 in ABUF. Required for W8A32 graphs with embeddings — without
    it, LN reads INT8 bytes as FP32 = garbage. Full GPT-2 124M still
    streams its embeddings via `runtime_patch` and isn't affected by
    this change (tiny fixture only).

    When *calibration_scales* is None the function runs calibration internally
    so both the compiler and the NanoGPTFQReference share the same per-node scales.
    Pass an explicit dict to avoid re-running calibration when it was already built.
    """
    if calibration_scales is None:
        calibration_scales = build_calibration_scales(payload)
    state_dict = payload["state_dict"]
    config = model_config_from_fixture_payload(payload)
    per_tensor_fc1_blocks = set(int(v) for v in (per_tensor_fc1_blocks or set()))
    # Token and position embeddings: INT8 path shares the tok_pos_add
    # output scale across both tables (otherwise q_token + q_pos isn't
    # a representation of token + position in any single real scale).
    # W8A32 path stores them as raw FP32 — no quantization step, no
    # cross-table scale constraint. The split below picks one or the
    # other based on `use_fp16_activations` (M3-prep).
    if use_fp16_activations:
        # W8A16: embedding output is FP16 (2 bytes/elem). The codegen DMAs
        # `d_model_pad * 2` bytes per row and the resulting ABUF tile is
        # interpreted as FP16 by the first sub-layer op (LN widens FP16
        # to FP32 internally on read).
        weight_data = {
            "transformer.wte.weight": (_fp16_embedding(state_dict["transformer.wte.weight"]), None),
            "transformer.wpe.weight": (_fp16_embedding(state_dict["transformer.wpe.weight"]), None),
            "transformer.ln_f.weight": (_fp16_vector(state_dict["transformer.ln_f.weight"]), None),
            "transformer.ln_f.bias": (_fp16_vector(state_dict["transformer.ln_f.bias"]), None),
        }
    else:
        # INT8 path (W8A8 default): both embedding tables share the
        # output scale used for tok_pos_add; otherwise q_token + q_pos
        # is not a representation of token + position in any single
        # real scale.
        embedding_add_scale = float(calibration_scales.get("tok_pos_add", 6.0 / 127.0))
        weight_data = {
            "transformer.wte.weight": (_quantize_embedding(state_dict["transformer.wte.weight"], embedding_add_scale), None),
            "transformer.wpe.weight": (_quantize_embedding(state_dict["transformer.wpe.weight"], embedding_add_scale), None),
            "transformer.ln_f.weight": (_fp16_vector(state_dict["transformer.ln_f.weight"]), None),
            "transformer.ln_f.bias": (_fp16_vector(state_dict["transformer.ln_f.bias"]), None),
        }
    prescaled_biases: Dict[str, np.ndarray] = {}
    # M2.5-C: raw FP32 biases for the W8A32 simple-matmul lowering.
    # Populated for non-attention matmuls only — see the docstring.
    biases: Dict[str, np.ndarray] = {}

    for layer in range(config.n_layer):
        for ln in ("ln_1", "ln_2"):
            weight_data[f"transformer.h.{layer}.{ln}.weight"] = (
                _fp16_vector(state_dict[f"transformer.h.{layer}.{ln}.weight"]),
                None,
            )
            weight_data[f"transformer.h.{layer}.{ln}.bias"] = (
                _fp16_vector(state_dict[f"transformer.h.{layer}.{ln}.bias"]),
                None,
            )
        for head in range(config.n_head):
            for proj in ("query", "key", "value"):
                name = f"transformer.h.{layer}.attn.c_attn.weight_h{head}_{proj}"
                weight_data[name] = _quantize_linear_weight(state_dict[name])
                bias_name = f"transformer.h.{layer}.attn.c_attn.bias_h{head}_{proj}"
                if bias_name in state_dict:
                    prescaled_biases[bias_name] = _prescale_bias(
                        state_dict,
                        bias_name,
                        output_dim=config.d_head,
                        act_scale=calibration_scales.get(f"block{layer}_ln1", 6.0 / 127.0),
                        weight_scales=weight_data[name][1],
                    )
                    # M3-prep: stage the FP32 sibling for the per-head
                    # Q/K/V matmul that emit_matmul_w8a16 will eventually
                    # consume. Currently behind the matmul_qkt/matmul_attn_v
                    # guardrail (matmul_qkt/attn_v themselves are M3); the
                    # per-head Q/K/V matmuls upstream are `op="matmul"`
                    # nodes and will be lowered by emit_matmul_w8a16 once
                    # M3 lifts the graph-level guardrail.
                    biases[bias_name] = _fp32_bias(
                        state_dict, bias_name, output_dim=config.d_head,
                    )
        weight_data[f"transformer.h.{layer}.attn.c_proj.weight"] = _quantize_linear_weight(
            state_dict[f"transformer.h.{layer}.attn.c_proj.weight"],
        )
        weight_data[f"transformer.h.{layer}.mlp.c_fc.weight"] = _quantize_linear_weight(
            state_dict[f"transformer.h.{layer}.mlp.c_fc.weight"],
            per_channel=layer not in per_tensor_fc1_blocks,
        )
        weight_data[f"transformer.h.{layer}.mlp.c_proj.weight"] = _quantize_linear_weight(
            state_dict[f"transformer.h.{layer}.mlp.c_proj.weight"],
        )
        bias_specs = (
            (
                f"transformer.h.{layer}.attn.c_proj.bias",
                f"transformer.h.{layer}.attn.c_proj.weight",
                config.d_model,
                f"block{layer}_concat",
            ),
            (
                f"transformer.h.{layer}.mlp.c_fc.bias",
                f"transformer.h.{layer}.mlp.c_fc.weight",
                config.mlp_dim,
                f"block{layer}_ln2",
            ),
            (
                f"transformer.h.{layer}.mlp.c_proj.bias",
                f"transformer.h.{layer}.mlp.c_proj.weight",
                config.d_model,
                f"block{layer}_gelu",
            ),
        )
        for bias_name, weight_name, output_dim, act_name in bias_specs:
            prescaled_biases[bias_name] = _prescale_bias(
                state_dict,
                bias_name,
                output_dim=output_dim,
                act_scale=calibration_scales.get(act_name, 6.0 / 127.0),
                weight_scales=weight_data[weight_name][1],
            )
            # M2.5-C: same families, FP32 unscaled — consumed by
            # emit_matmul_w8a16 when use_fp16_activations. The bias names map
            # 1:1 onto the IR node's `attrs["bias"]` values emitted by
            # the nanogpt frontend (out_proj, fc1, fc2 per layer).
            biases[bias_name] = _fp32_bias(
                state_dict, bias_name, output_dim=output_dim,
            )

    weight_data["lm_head.weight"] = _quantize_linear_weight(state_dict["lm_head.weight"])
    # `lm_head.bias` is created by `taccel.quantizer.ln_fold.fold_layernorm_for_quarot`
    # (β-fold of `ln_f`) when `quarot_enabled=True` is set on the preset.
    # Standard GPT-2 has no `lm_head.bias`; we prescale it only when present.
    if "lm_head.bias" in state_dict:
        prescaled_biases["lm_head.bias"] = _prescale_bias(
            state_dict,
            "lm_head.bias",
            output_dim=config.vocab_size,
            act_scale=calibration_scales.get("ln_f", 6.0 / 127.0),
            weight_scales=weight_data["lm_head.weight"][1],
        )
        # M2.5-C: same key, FP32 unscaled.
        biases["lm_head.bias"] = _fp32_bias(
            state_dict, "lm_head.bias", output_dim=config.vocab_size,
        )
    # W8A16: lm_head produces FP16 output (2 bytes/elem). The logits
    # DRAM region grows by 2× vs the INT8 path.
    logits_elem_bytes = 2 if use_fp16_activations else 1
    return (
        weight_data,
        prescaled_biases,
        biases,
        calibration_scales,
        config,
        pad_dim(config.vocab_size) * logits_elem_bytes,
    )


def build_stage3_tiny_decoder_bundle(
    payload: Dict[str, object],
    *,
    smoke_decode_steps: int = 2,
    calibration_scales: Optional[Dict[str, float]] = None,
    ptq_preset: str | Stage5PTQPreset | None = None,
) -> TinyFixtureBundle:
    """Build the full 1-token tiny decoder ProgramBundle used by Stage 3 tests."""
    resolved_preset = resolve_stage5_ptq_preset(ptq_preset)
    activation_percentile_nodes = dict(resolved_preset.activation_percentile_nodes)
    if calibration_scales is None:
        calibration_scales = build_calibration_scales(
            payload,
            activation_percentile_overrides=activation_percentile_nodes or None,
        )
    else:
        calibration_scales = dict(calibration_scales)
    gelu_from_accum_blocks = stage5_gelu_from_accum_blocks(resolved_preset)
    (
        weight_data,
        prescaled_biases,
        biases,
        calibration_scales,
        config,
        logits_size,
    ) = quantize_fixture_payload(
        payload,
        calibration_scales=calibration_scales,
        per_tensor_fc1_blocks=gelu_from_accum_blocks,
        use_fp16_activations=bool(resolved_preset.weight_only_int8),
    )
    validate_stage5_ptq_preset_for_model(config, resolved_preset)
    calibration_scales = apply_stage5_ptq_scale_policy(calibration_scales, config, resolved_preset)
    # Thread state_dict so the frontend can detect optional `lm_head.bias`
    # (created by `taccel.quantizer.ln_fold.fold_layernorm_for_quarot`).
    frontend = load_nanogpt(
        config=payload["model_args"],
        state_dict=payload["state_dict"],
        variant="forward_1token",
    )
    graph = mark_runtime_embedding_lookups(frontend.graph)
    if gelu_from_accum_blocks:
        _mark_gelu_from_accum_inline(graph, gelu_from_accum_blocks)
    decode_key_len = 1 + int(smoke_decode_steps)
    prefill_graph = inject_kv_cache_nodes(graph, config, decode=False, seq_len=1)
    decode_graph = inject_kv_cache_nodes(graph, config, decode=True, seq_len=decode_key_len)
    build = build_decoder_program_bundle(
        prefill_graph=prefill_graph,
        decode_graph=decode_graph,
        weight_data=weight_data,
        calibration_scales=calibration_scales,
        prescaled_biases=prescaled_biases,
        model_config=config,
        max_seq_len=config.max_seq_len,
        logits_size=logits_size,
        requant_pc_weight_names=stage5_requant_pc_weight_names(config, resolved_preset),
        dequant_add_residual1_blocks=stage5_dequant_add_residual1_blocks(config, resolved_preset),
        dequant_add_residual2_blocks=stage5_dequant_add_residual2_blocks(config, resolved_preset),
        gelu_from_accum_blocks=gelu_from_accum_blocks or None,
        use_fp16_activations=bool(resolved_preset.weight_only_int8),
        biases=biases,
    )
    return TinyFixtureBundle(build=build, config=config, logits_size=logits_size)


build_stage3f_tiny_decoder_bundle = build_stage3_tiny_decoder_bundle


def _greedy_token(logits: np.ndarray, vocab_size: int) -> int:
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    active = np.asarray(logits)[:vocab_size]
    if active.size == 0:
        raise ValueError("logits are empty")
    return int(np.argmax(active))


def run_tiny_decode_trace(tiny: TinyFixtureBundle, prompt_ids: Sequence[int], *,
                          max_new_tokens: int,
                          logits_dtype=None) -> TinyDecodeTrace:
    """Run a greedy tiny decode trace and retain prefill + per-step logits.

    `logits_dtype` (M4-F W8A16): infer from `tiny.logits_size` when None —
    pad_dim(vocab) * 1 = INT8, *2 = FP16, *4 = FP32.
    """
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if not prompt_ids:
        raise ValueError("prompt_ids must be non-empty")
    if logits_dtype is None:
        elem_bytes = max(1, int(tiny.logits_size) // pad_dim(int(tiny.config.vocab_size)))
        logits_dtype = {1: np.int8, 2: np.float16, 4: np.float32}.get(elem_bytes, np.int8)
    generated = [int(tok) for tok in prompt_ids]
    runner = HostRunner(tiny.build.bundle, logits_dtype=logits_dtype)

    logits_trace: List[np.ndarray] = []
    logits = runner.run_prefill(generated)
    logits_trace.append(logits)
    next_token = _greedy_token(logits, tiny.config.vocab_size)

    for _ in range(max_new_tokens):
        generated.append(next_token)
        position = len(generated) - 1
        logits = runner.run_decode_step(next_token, position)
        logits_trace.append(logits)
        next_token = _greedy_token(logits, tiny.config.vocab_size)

    return TinyDecodeTrace(generated=generated, logits=logits_trace)


def _topk_overlap(a: np.ndarray, b: np.ndarray, *, vocab_size: int, k: int = 5) -> int:
    lhs = _topk_set(a, vocab_size=vocab_size, k=k)
    rhs = _topk_set(b, vocab_size=vocab_size, k=k)
    return len(lhs.intersection(rhs))


def _topk_set(logits: np.ndarray, *, vocab_size: int, k: int = 5) -> set[int]:
    k = min(int(k), int(vocab_size))
    active = np.asarray(logits)[:vocab_size]
    if k <= 0 or active.size == 0:
        return set()
    threshold = np.sort(active)[-k]
    return set(np.where(active >= threshold)[0].tolist())


def _run_reference_trace(
    ref: NanoGPTFQReference,
    prompt_ids: Sequence[int],
    *,
    max_new_tokens: int,
    vocab_size: int,
) -> TinyDecodeTrace:
    """Greedy decode using the PyTorch fake-quant numpy reference."""
    generated = [int(tok) for tok in prompt_ids]
    logits_trace: List[np.ndarray] = []

    logits = ref.forward_incremental(generated)
    logits_trace.append(logits)
    next_token = _greedy_token(logits, vocab_size)

    for _ in range(max_new_tokens):
        generated.append(next_token)
        logits = ref.forward_incremental(generated)
        logits_trace.append(logits)
        next_token = _greedy_token(logits, vocab_size)

    return TinyDecodeTrace(generated=generated, logits=logits_trace)


def run_stage3_tiny_e2e(payload: Dict[str, object], *,
                        prompt_ids: Sequence[int] = (0,),
                        max_new_tokens: int = 32,
                        ptq_preset: str | Stage5PTQPreset | None = None) -> TinyE2EResult:
    """Run the Stage 3 32-step gate comparing the golden model against a
    PyTorch fake-quant numpy reference that shares calibration scales.

    Both paths use the same per-node INT8 scales derived from calibration on
    the fixture's embedded Shakespeare text, so differences reflect arithmetic
    correctness rather than calibration drift. When ptq_preset is the W8A16
    weight-only variant, the bundle uses FP16 activation storage; otherwise
    INT8 throughout (W8A8 reference).
    """
    # Single calibration pass — shared by compiler and reference
    resolved_preset = resolve_stage5_ptq_preset(ptq_preset)
    calibration_scales = build_calibration_scales(
        payload,
        activation_percentile_overrides=(
            resolved_preset.activation_percentile_nodes or None
        ),
    )
    calibration_scales = apply_stage5_ptq_scale_policy(
        calibration_scales,
        payload["model_args"],
        resolved_preset,
    )

    tiny = build_stage3_tiny_decoder_bundle(
        payload,
        smoke_decode_steps=max_new_tokens,
        calibration_scales=calibration_scales,
        ptq_preset=resolved_preset,
    )
    actual = run_tiny_decode_trace(tiny, prompt_ids, max_new_tokens=max_new_tokens)

    ref = NanoGPTFQReference(
        state_dict=payload["state_dict"],
        model_args=payload["model_args"],
        scales=calibration_scales,
        requant_pc_weight_names=stage5_requant_pc_weight_names(payload["model_args"], resolved_preset),
        raw_residual1_blocks=stage5_raw_residual1_blocks(resolved_preset),
        raw_residual2_blocks=stage5_raw_residual2_blocks(resolved_preset),
        gelu_from_accum_blocks=stage5_gelu_from_accum_blocks(resolved_preset),
    )
    reference = _run_reference_trace(
        ref, prompt_ids, max_new_tokens=max_new_tokens, vocab_size=tiny.config.vocab_size
    )

    vocab = tiny.config.vocab_size
    cosine = [
        cosine_similarity(got[:vocab].astype(np.float32), exp[:vocab].astype(np.float32))
        for got, exp in zip(actual.logits, reference.logits)
    ]
    top5 = [
        _topk_overlap(got, exp, vocab_size=vocab, k=5)
        for got, exp in zip(actual.logits, reference.logits)
    ]
    return TinyE2EResult(
        generated=actual.generated,
        reference_generated=reference.generated,
        cosine_per_step=cosine,
        top5_overlap_per_step=top5,
        logits=actual.logits,
        reference_logits=reference.logits,
    )


run_stage3g_tiny_e2e = run_stage3_tiny_e2e


def run_nanogpt_fp32_e2e(payload: Dict[str, object], *,
                         prompt_ids: Sequence[int] = (0,),
                         max_new_tokens: int = 32) -> TinyFP32E2EResult:
    """Compare golden-model INT8 generation against true PyTorch FP32 nanoGPT.

    The fake-quant gate remains the primary INT8 correctness test.  This helper
    is intentionally rank-based: golden logits are INT8, while the reference is
    true FP32, so top-k agreement is a more stable signal than raw magnitudes.
    """
    calibration_scales = build_calibration_scales(payload)
    tiny = build_stage3_tiny_decoder_bundle(
        payload,
        smoke_decode_steps=max_new_tokens,
        calibration_scales=calibration_scales,
    )
    actual = run_tiny_decode_trace(tiny, prompt_ids, max_new_tokens=max_new_tokens)
    fp32_ref = NanoGPTFP32Reference(payload["state_dict"], payload["model_args"])
    fp32 = fp32_ref.greedy_decode_trace(
        prompt_ids,
        max_new_tokens=max_new_tokens,
    )
    # Rank-quality gates must compare both models on the same prefix.  The
    # independently free-running FP32 text is still returned as a diagnostic,
    # but the logits below are evaluated on the golden-generated prefix so an
    # early greedy-token split does not poison every later rank comparison.
    fp32_logits_on_golden_prefix = fp32_ref.incremental_logits_trace(actual.generated)

    vocab = tiny.config.vocab_size
    lm_head_scale = float(calibration_scales.get("lm_head", 1.0))
    top5 = []
    fp32_in_golden = []
    golden_in_fp32 = []
    top1_match = []
    cosine = []
    for got, ref in zip(actual.logits, fp32_logits_on_golden_prefix):
        got_active = np.asarray(got)[:vocab]
        ref_active = np.asarray(ref)[:vocab]
        got_top5 = _topk_set(got_active, vocab_size=vocab, k=5)
        ref_top5 = _topk_set(ref_active, vocab_size=vocab, k=5)
        got_argmax_set = set(np.where(got_active == np.max(got_active))[0].tolist())
        ref_argmax_set = set(np.where(ref_active == np.max(ref_active))[0].tolist())
        top5.append(len(got_top5.intersection(ref_top5)))
        fp32_in_golden.append(bool(ref_argmax_set.intersection(got_top5)))
        golden_in_fp32.append(bool(got_argmax_set.intersection(ref_top5)))
        top1_match.append(bool(got_argmax_set.intersection(ref_argmax_set)))
        got_dequant = got_active.astype(np.float32) * np.float32(lm_head_scale)
        cosine.append(cosine_similarity(got_dequant, ref_active.astype(np.float32)))

    return TinyFP32E2EResult(
        generated=actual.generated,
        fp32_generated=fp32.generated,
        logits=actual.logits,
        fp32_logits=fp32_logits_on_golden_prefix,
        top5_overlap_per_step=top5,
        fp32_top1_in_golden_top5_per_step=fp32_in_golden,
        golden_top1_in_fp32_top5_per_step=golden_in_fp32,
        top1_match_per_step=top1_match,
        fp32_cosine_per_step=cosine,
    )
