"""Independent bounded checks of W8A16 reference arithmetic; no model fixtures."""
import ast
from pathlib import Path

import numpy as np
import pytest

from taccel.runtime import w8a16_simulator_reference as reference


def half(value):
    return np.asarray(value, dtype=np.float16).astype(np.float32)


def bits(value):
    return np.asarray(value, dtype=np.float16).view(np.uint16)


def partial_oracle(x, w, scales, bias=None):
    """Written from the ISA storage contract, not another simulator/emitter."""
    x = half(x)
    maximum = max(float(np.abs(x).max()), 2.0 ** -9)
    inverse = np.float32(np.float16(127.0 / maximum))
    forward = np.float32(np.float16(maximum / 127.0))
    integer = np.clip(np.rint(x * inverse), -128, 127).astype(np.int32)
    accum = integer @ np.asarray(w, dtype=np.int32)
    result = accum.astype(np.float32) * half(scales).reshape(1, -1) * forward
    if bias is not None:
        result += half(bias)
    return half(result)


def payload(*, layers=1, width=16, mlp_width=None, seed=19):
    rng = np.random.default_rng(seed)
    inner = mlp_width or 4 * width
    cfg = {"n_layer": layers, "n_head": 1, "n_embd": width,
           "mlp_dim": inner, "vocab_size": 32, "block_size": 64,
           "layer_norm_epsilon": 1e-5, "split_qkv_bias": True}
    draw = lambda shape: rng.normal(0, 0.2, shape).astype(np.float32)
    sd = {"transformer.wte.weight": draw((32, width)),
          "transformer.wpe.weight": draw((64, width)),
          "transformer.ln_f.weight": 1 + draw(width),
          "transformer.ln_f.bias": draw(width),
          "lm_head.weight": draw((32, width)), "lm_head.bias": draw(32)}
    for layer in range(layers):
        prefix = f"transformer.h.{layer}"
        for ln in ("ln_1", "ln_2"):
            sd[f"{prefix}.{ln}.weight"] = 1 + draw(width)
            sd[f"{prefix}.{ln}.bias"] = draw(width)
        for name, rows, cols in (("attn.c_proj", width, width),
                                 ("mlp.c_fc", inner, width),
                                 ("mlp.c_proj", width, inner)):
            sd[f"{prefix}.{name}.weight"] = draw((rows, cols))
            sd[f"{prefix}.{name}.bias"] = draw(rows)
        for projection in ("query", "key", "value"):
            sd[f"{prefix}.attn.c_attn.weight_h0_{projection}"] = draw((width, width))
            sd[f"{prefix}.attn.c_attn.bias_h0_{projection}"] = draw(width)
    return {"model_args": cfg, "state_dict": sd}


@pytest.mark.parametrize("bitwidth", [4, 8])
def test_weight_integer_bins_use_unrounded_fp32_scale(bitwidth):
    w = np.random.default_rng(7).normal(size=(64, 64)).astype(np.float32)
    limit = (1 << (bitwidth - 1)) - 1
    scale = np.maximum(np.abs(w).max(axis=0), 1e-8) / float(limit)
    expected = np.clip(np.rint(w / scale), -limit - 1, limit).astype(np.int8)
    integer, stored_scale = reference._quant_w_per_channel(w, bitwidth=bitwidth)
    np.testing.assert_array_equal(integer, expected)
    np.testing.assert_array_equal(bits(stored_scale), bits(scale))
    old = np.clip(np.rint(w / half(scale)), -limit - 1, limit).astype(np.int8)
    assert np.any(old != expected), "fixture must detect premature scale rounding"


def test_embedding_operands_round_before_addition_and_ln_parameters_are_stored_half():
    model = payload()
    ref = reference.NanoGPTW8A16SimulatorReference(model)
    trace = ref.incremental_node_trace([3])[0]
    sd = model["state_dict"]
    expected = half(half(sd["transformer.wte.weight"][3]) + half(sd["transformer.wpe.weight"][0]))
    old = half(sd["transformer.wte.weight"][3] + sd["transformer.wpe.weight"][0])
    np.testing.assert_array_equal(bits(trace["tok_pos_add"]["value"][0]), bits(expected))
    assert np.any(bits(expected) != bits(old))
    for attr, name in (("ln_f_w", "weight"), ("ln_f_b", "bias")):
        np.testing.assert_array_equal(getattr(ref, attr), half(sd[f"transformer.ln_f.{name}"]))
    for number in (1, 2):
        for tag, suffix in (("w", "weight"), ("b", "bias")):
            np.testing.assert_array_equal(ref.layers[0][f"ln{number}_{tag}"],
                                          half(sd[f"transformer.h.0.ln_{number}.{suffix}"]))


def test_dynamic_epilogue_reads_bias_from_half_storage():
    rng = np.random.default_rng(13)
    x = rng.normal(size=(1, 16)).astype(np.float32)
    w = rng.integers(-127, 128, size=(16, 64), dtype=np.int8)
    scales = np.full(64, np.float16(0.001))
    bias = rng.normal(size=64).astype(np.float32)
    got = reference._w8a16_dynamic_matmul(x, w, scales, bias)
    expected = partial_oracle(x, w, scales, bias)
    np.testing.assert_array_equal(bits(got), bits(expected))
    maximum = float(np.abs(half(x)).max())
    q = np.clip(np.rint(half(x) * np.float32(np.float16(127 / maximum))), -128, 127).astype(np.int32)
    old = half((q @ w.astype(np.int32)).astype(np.float32) * half(scales)
               * np.float32(np.float16(maximum / 127)) + bias)
    assert np.any(bits(old) != bits(expected)), "fixture must detect FP32-bias mismatch"


def test_static_attention_quantization_uses_stored_reciprocal_not_division():
    ref = reference.NanoGPTW8A16SimulatorReference(payload(layers=0))
    x = half(np.linspace(-6, 6, 10001, dtype=np.float32))
    scale = 6.0 / 127.0
    expected = np.clip(np.rint(x * np.float32(np.float16(1.0 / scale))), -128, 127).astype(np.int8)
    got = ref._qkt_attn_v_quantize(x, scale)
    np.testing.assert_array_equal(got, expected)
    assert np.any(got != np.clip(np.rint(x / scale), -128, 127).astype(np.int8))


def test_missing_softmax_scale_uses_one_over127_but_explicit_override_wins():
    ref = reference.NanoGPTW8A16SimulatorReference(payload(layers=0))
    assert ref._act_scale_for("block0_head0_softmax") == 1.0 / 127.0
    assert ref._act_scale_for("block0_head0_value") == 6.0 / 127.0
    assert ref._qkt_attn_v_quantize(np.array([1.0], dtype=np.float32),
                                  ref._act_scale_for("block0_head0_softmax"))[0] == 127
    ref.calibration_scales["block0_head0_softmax"] = 0.0123
    assert ref._act_scale_for("block0_head0_softmax") == 0.0123


def test_gelu_constants_and_all_intermediates_are_fp32():
    x = half(np.array([-5.41796875, -4.0, -0.25, 0.125, 2.0], dtype=np.float32))
    inner = np.float32(np.sqrt(2.0 / np.pi)) * (x + np.float32(0.044715) * x ** 3)
    expected = half(x * np.float32(0.5) * (np.float32(1) + np.tanh(inner)))
    np.testing.assert_array_equal(bits(reference._gelu_fp16(x)), bits(expected))
    old = half(0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3))))
    assert np.any(bits(old) != bits(expected))


def test_streaming_fc2_3072_by1_has_partition_scales_half_partials_and_ordered_sums():
    rng = np.random.default_rng(27)
    x = rng.normal(size=(1, 3072)).astype(np.float32)
    x[0, :256] *= 20  # A global max must not control all twelve partitions.
    w = rng.integers(-127, 128, size=(3072, 1), dtype=np.int8)
    scales = np.array([0.001], dtype=np.float16)
    bias = np.array([0.312345], dtype=np.float32)
    expected = None
    partials = []
    for start in range(0, 3072, 256):
        part = partial_oracle(x[:, start:start + 256], w[start:start + 256], scales,
                              bias if start == 0 else None)
        partials.append(part)
        expected = part if expected is None else half(expected + part)
    got = reference._w8a16_dynamic_matmul(x, w, scales, bias, k_tile_size=256)
    np.testing.assert_array_equal(bits(got), bits(expected))
    old = reference._w8a16_dynamic_matmul(x, w, scales, bias)
    assert np.any(bits(old) != bits(expected))
    # Bias belongs only to the first partition; each VADD stores to FP16.
    np.testing.assert_array_equal(bits(got), bits(half(expected)))
    unrounded_sum = half(np.sum(np.stack(partials), axis=0, dtype=np.float32))
    assert np.any(bits(unrounded_sum) != bits(expected)), "fixture must expose partial-add rounding"


@pytest.mark.parametrize("width,inner,expected", [(128, 512, None), (384, 1536, 1536),
                                                  (768, 3072, 256), (16, 3072, None)])
def test_fc2_partition_admission_is_shape_specific(width, inner, expected):
    assert reference._fc2_k_partition_size(width, inner) == expected


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_invalid_partition_size_is_explicit(bad):
    with pytest.raises(ValueError, match="positive integer"):
        reference._w8a16_dynamic_matmul(np.ones((1, 16), dtype=np.float32),
                                       np.ones((16, 1), dtype=np.int8),
                                       np.ones(1, dtype=np.float16), k_tile_size=bad)


def test_post_ptq_weight_overrides_preserve_integer_decisions_and_scale_bits():
    model = payload()
    query_name = "transformer.h.0.attn.c_attn.weight_h0_query"
    overrides = {query_name: (np.full((16, 16), -3, dtype=np.int8),
                             np.linspace(0.001, 0.016, 16, dtype=np.float16)),
                 "lm_head.weight": (np.full((16, 32), 5, dtype=np.int8),
                                    np.full(32, 0.0123, dtype=np.float16))}
    # Simulate a PTQ bias mutation before reference construction.
    model["state_dict"]["lm_head.bias"] += np.float32(0.123456)
    ref = reference.NanoGPTW8A16SimulatorReference(model, weight_overrides=overrides)
    head = ref.layers[0]["heads"][0]
    np.testing.assert_array_equal(head["q_int8"], overrides[query_name][0])
    np.testing.assert_array_equal(bits(head["q_scales"]), bits(overrides[query_name][1]))
    np.testing.assert_array_equal(ref.lm_head_w_int8, overrides["lm_head.weight"][0])
    np.testing.assert_array_equal(bits(ref.lm_head_w_scales), bits(overrides["lm_head.weight"][1]))
    traces = ref.incremental_node_trace([2, 7])
    for trace in traces:
        expected = partial_oracle(trace["ln_f"]["value"], overrides["lm_head.weight"][0],
                                  overrides["lm_head.weight"][1], model["state_dict"]["lm_head.bias"])
        np.testing.assert_array_equal(bits(trace["lm_head"]["value"]), bits(expected))


def test_prefill_and_decode_use_their_actual_scale_keys_and_stored_int8_cache(monkeypatch):
    scales = {"block0_head0_key": 0.003, "block0_head0_key_kv_load": 0.031,
              "block0_head0_value": 0.017, "block0_head0_value_kv_load": 0.071,
              "block0_head0_softmax": 1.0 / 127.0}
    ref = reference.NanoGPTW8A16SimulatorReference(
        payload(), calibration_scales=scales, attention_window=17)
    widths = []
    original = reference.softmax_masked

    def observe(scores, valid_kv_len, **kwargs):
        widths.append(scores.shape[-1])
        return original(scores, valid_kv_len, **kwargs)

    monkeypatch.setattr(reference, "softmax_masked", observe)
    traces = ref.incremental_node_trace([3, 7])
    assert widths == [16, 32], "prefill is one key; decode uses the complete compiled window"
    cache = ref._caches[0][0]
    for kind, projection in (("k", "key"), ("v", "value")):
        scale = scales[f"block0_head0_{projection}_kv_load"]
        inverse = np.float32(np.float16(1.0 / scale))
        for index, trace in enumerate(traces):
            expected = np.clip(np.rint(half(trace[f"block0_head0_{projection}"]["value"])[0]
                                       * inverse), -128, 127).astype(np.int8)
            assert cache[f"{kind}_int8"][index].dtype == np.int8
            np.testing.assert_array_equal(cache[f"{kind}_int8"][index], expected)
    for index, trace in enumerate(traces):
        probabilities = trace["block0_head0_softmax"]["value"]
        p_int = np.clip(np.rint(probabilities * np.float32(127)), -128, 127).astype(np.int32)
        if index == 0:
            value_scale = scales["block0_head0_value"]
            values = trace["block0_head0_value"]["value"]
            v_int = np.clip(np.rint(values * np.float32(np.float16(1 / value_scale))),
                             -128, 127).astype(np.int32)
        else:
            value_scale = scales["block0_head0_value_kv_load"]
            v_int = np.stack(cache["v_int8"]).astype(np.int32)
        composite = np.float32(np.float16(np.float32(1 / 127) * np.float32(value_scale)))
        expected = half((p_int @ v_int).astype(np.float32) * composite)
        np.testing.assert_array_equal(bits(trace["block0_head0_attn_v"]["value"]), bits(expected))


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_attention_window_must_be_positive_integer(bad):
    with pytest.raises(ValueError, match="positive integer"):
        reference.NanoGPTW8A16SimulatorReference(payload(layers=0), attention_window=bad)


def test_attention_window_bounds_are_explicit():
    ref = reference.NanoGPTW8A16SimulatorReference(payload(layers=0), attention_window=1)
    ref.run_prefill([2])
    with pytest.raises(ValueError, match="exceeds"):
        ref.run_decode_step(3, 1)


@pytest.mark.parametrize("scales", [{}, {
    "block0_head0_query": 0.009,
    "block0_head0_key": 0.003, "block0_head0_key_kv_load": 0.003,
    "block0_head0_value": 0.017, "block0_head0_value_kv_load": 0.017,
    "block0_head0_softmax": 0.011,
}])
def test_two_token_compiled_synthetic_logits_and_complete_int8_kv_match_independent_reference(scales):
    # The compiled simulator is the implementation under test, never a source
    # of expected operands, quantization decisions, or arithmetic results.
    from taccel.runtime.host_runner import HostRunner
    from taccel.runtime.tiny_fixture import build_stage3_tiny_decoder_bundle

    model = payload()
    ref = reference.NanoGPTW8A16SimulatorReference(
        model, calibration_scales=scales, attention_window=17)
    expected = ref.run_teacher_forced([3, 7])
    tiny = build_stage3_tiny_decoder_bundle(
        model, calibration_scales=scales, ptq_preset="weight_only_int8", smoke_decode_steps=16)
    runner = HostRunner(tiny.build.bundle, logits_dtype=np.float16)
    actual = [runner.run_prefill([3]), runner.run_decode_step(7, 1)]
    for got, want in zip(actual, expected):
        assert np.isfinite(got).all() and np.isfinite(want).all()
        np.testing.assert_array_equal(bits(got), bits(want))
    for key, kind in (("k_int8", "key"), ("v_int8", "value")):
        entry = tiny.build.kv_layout.entry(0, kind, 0)
        start = tiny.build.bundle.kv_cache_base + entry.byte_offset
        got = np.frombuffer(runner.simulator.state.dram[start:start + entry.span_bytes], dtype=np.int8)
        want = np.zeros((64, 16), dtype=np.int8)
        want[:2] = np.stack(ref._caches[0][0][key])
        np.testing.assert_array_equal(got, want.ravel())


def test_reference_has_no_compiler_or_golden_implementation_dependency():
    tree = ast.parse(Path(reference.__file__).read_text())
    imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imports += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    assert not any("compiler" in name or "golden_model" in name for name in imports)
