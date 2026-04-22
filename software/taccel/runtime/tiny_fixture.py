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
from .calibration import build_calibration_scales
from .fake_quant import cosine_similarity
from .fake_quant_reference import NanoGPTFQReference
from .fp32_reference import NanoGPTFP32Reference
from .host_runner import HostRunner


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


def _quantize_linear_weight(tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Quantize a PyTorch-layout [out, in] weight into TACCEL [K, N] layout."""
    q, scales = quantize_tensor(_to_numpy(tensor).astype(np.float32), per_channel=True)
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


def _zero_prescaled_bias(state_dict: Dict[str, object], name: str) -> np.ndarray:
    if name not in state_dict:
        return np.zeros(0, dtype=np.int32)
    arr = _to_numpy(state_dict[name]).astype(np.float32)
    if not np.allclose(arr, 0.0):
        raise ValueError(
            f"Stage 3 tiny fixture helper only supports zero matmul biases; "
            f"{name!r} is non-zero"
        )
    return np.zeros(pad_dim(arr.size), dtype=np.int32)


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
):
    """Return codegen-ready weights, zero biases, scales, config, and logits size.

    When *calibration_scales* is None the function runs calibration internally
    so both the compiler and the NanoGPTFQReference share the same per-node scales.
    Pass an explicit dict to avoid re-running calibration when it was already built.
    """
    if calibration_scales is None:
        calibration_scales = build_calibration_scales(payload)
    state_dict = payload["state_dict"]
    config = model_config_from_fixture_payload(payload)
    # Token and position embeddings are added by a raw INT8 VADD in the
    # generated program.  Both tables therefore must share the output scale used
    # for tok_pos_add; otherwise q_token + q_pos is not a representation of
    # token + position in any single real scale.
    embedding_add_scale = float(calibration_scales.get("tok_pos_add", 6.0 / 127.0))
    weight_data = {
        "transformer.wte.weight": (_quantize_embedding(state_dict["transformer.wte.weight"], embedding_add_scale), None),
        "transformer.wpe.weight": (_quantize_embedding(state_dict["transformer.wpe.weight"], embedding_add_scale), None),
        "transformer.ln_f.weight": (_fp16_vector(state_dict["transformer.ln_f.weight"]), None),
        "transformer.ln_f.bias": (_fp16_vector(state_dict["transformer.ln_f.bias"]), None),
    }
    prescaled_biases: Dict[str, np.ndarray] = {}

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
        for name in (
            f"transformer.h.{layer}.attn.c_proj.weight",
            f"transformer.h.{layer}.mlp.c_fc.weight",
            f"transformer.h.{layer}.mlp.c_proj.weight",
        ):
            weight_data[name] = _quantize_linear_weight(state_dict[name])
        for name in (
            f"transformer.h.{layer}.attn.c_proj.bias",
            f"transformer.h.{layer}.mlp.c_fc.bias",
            f"transformer.h.{layer}.mlp.c_proj.bias",
        ):
            prescaled_biases[name] = _zero_prescaled_bias(state_dict, name)

    weight_data["lm_head.weight"] = _quantize_linear_weight(state_dict["lm_head.weight"])
    return weight_data, prescaled_biases, calibration_scales, config, pad_dim(config.vocab_size)


def build_stage3_tiny_decoder_bundle(
    payload: Dict[str, object],
    *,
    smoke_decode_steps: int = 2,
    calibration_scales: Optional[Dict[str, float]] = None,
) -> TinyFixtureBundle:
    """Build the full 1-token tiny decoder ProgramBundle used by Stage 3 tests."""
    weight_data, prescaled_biases, calibration_scales, config, logits_size = quantize_fixture_payload(
        payload, calibration_scales=calibration_scales
    )
    frontend = load_nanogpt(config=payload["model_args"], variant="forward_1token")
    graph = mark_runtime_embedding_lookups(frontend.graph)
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
                          max_new_tokens: int) -> TinyDecodeTrace:
    """Run a greedy tiny decode trace and retain prefill + per-step logits."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if not prompt_ids:
        raise ValueError("prompt_ids must be non-empty")
    generated = [int(tok) for tok in prompt_ids]
    runner = HostRunner(tiny.build.bundle, logits_dtype=np.int8)

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
                        max_new_tokens: int = 32) -> TinyE2EResult:
    """Run the Stage 3 32-step gate comparing the golden model against a
    PyTorch fake-quant numpy reference that shares calibration scales.

    Both paths use the same per-node INT8 scales derived from calibration on
    the fixture's embedded Shakespeare text, so differences reflect arithmetic
    correctness rather than calibration drift.
    """
    # Single calibration pass — shared by compiler and reference
    calibration_scales = build_calibration_scales(payload)

    tiny = build_stage3_tiny_decoder_bundle(
        payload,
        smoke_decode_steps=max_new_tokens,
        calibration_scales=calibration_scales,
    )
    actual = run_tiny_decode_trace(tiny, prompt_ids, max_new_tokens=max_new_tokens)

    ref = NanoGPTFQReference(
        state_dict=payload["state_dict"],
        model_args=payload["model_args"],
        scales=calibration_scales,
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
