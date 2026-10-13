"""Quality criteria reject invalid data and bad metrics independently."""
import numpy as np
import pytest

from taccel.runtime.numerical_quality import teacher_forced_quality_report
from taccel.runtime import gpt2_perplexity as ppl


def _report(**changes):
    rows = [np.arange(8, dtype=np.float32)] * 2
    values = dict(token_ids=[0, 1, 2], vocab_size=8, compiled_logits=rows,
                  reference_logits=rows, fp32_logits=rows,
                  max_fp32_ppl_increase=0.10)
    values.update(changes)
    return teacher_forced_quality_report(**values)


def test_identical_three_rails_pass_without_dropping_steps():
    result = _report()
    assert result["passed"] and result["target_count"] == 2
    assert result["reference_ppl_relative_delta"] == 0
    assert result["fp32_ppl_increase"] == 0
    assert len(result["per_step"]) == 2


@pytest.mark.parametrize("rail", ["compiled_logits", "reference_logits", "fp32_logits"])
@pytest.mark.parametrize("bad", [[], [np.zeros(8)], [np.zeros((1, 8))] * 2,
                                [np.zeros(7)] * 2, [np.full(8, np.nan)] * 2,
                                [np.full(8, np.inf)] * 2])
def test_invalid_or_truncated_rails_fail_closed(rail, bad):
    with pytest.raises(ValueError):
        _report(**{rail: bad})


@pytest.mark.parametrize("rail", ["compiled_logits", "reference_logits", "fp32_logits"])
@pytest.mark.parametrize("row", [
    np.arange(8, dtype=np.complex64) + 1j,
    np.arange(8).astype(object),
    np.arange(8).astype("U"),
    np.arange(8).astype("S"),
    np.arange(8).astype(bool),
], ids=["complex", "object", "string", "bytes", "boolean"])
def test_logits_cannot_be_coerced_into_valid_evidence(rail, row):
    with pytest.raises(ValueError, match="logits dtype"):
        _report(**{rail: [row, row]})


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.float16, np.float32, np.float64])
def test_real_numeric_row_types_remain_supported(dtype):
    rows = [np.arange(8, dtype=dtype)] * 2
    assert _report(compiled_logits=rows, reference_logits=rows, fp32_logits=rows)["passed"]


@pytest.mark.parametrize("budget", [np.nan, np.inf, -0.01, False, True,
                                     np.bool_(False), np.bool_(True)])
def test_invalid_quality_budget_rejected(budget):
    with pytest.raises(ValueError, match="budget"):
        _report(max_fp32_ppl_increase=budget)


def test_reference_gate_does_not_accept_any_finite_gap():
    result = _report(reference_logits=[np.arange(8, dtype=np.float32) * 2] * 2)
    assert not result["passed"]
    assert "reference_perplexity" in result["violations"]


def test_cosine_gate_rejects_even_when_perplexity_and_ranking_match():
    reference = np.arange(8, dtype=np.float32) - 3.5
    # A constant shift leaves softmax probabilities and token ranking unchanged,
    # but not cosine similarity. Only the cosine criterion should reject it.
    actual = reference + 100.0
    result = _report(compiled_logits=[actual] * 2,
                     reference_logits=[reference] * 2, fp32_logits=[reference] * 2)
    assert not result["passed"]
    assert result["violations"] == ["reference_cosine"]
    assert result["reference_ppl_relative_delta"] == pytest.approx(0, abs=1e-12)
    assert result["fp32_ppl_increase"] == pytest.approx(0, abs=1e-12)
    assert result["min_reference_top5_overlap"] == 5
    assert result["min_reference_cosine"] < 0.995


def test_top5_gate_rejects_even_when_perplexity_and_cosine_pass():
    reference = np.arange(10, dtype=np.float32) + 1000.0
    actual = reference.copy()
    # Preserve the target (token 0) and the multiset of logits, hence its
    # probability, while replacing two top-five tokens. The shared offset
    # keeps cosine above its threshold without hiding the ranking mismatch.
    actual[[1, 9]] = actual[[9, 1]]
    actual[[2, 8]] = actual[[8, 2]]
    result = _report(token_ids=[0, 0, 0], vocab_size=10,
                     compiled_logits=[actual] * 2,
                     reference_logits=[reference] * 2, fp32_logits=[reference] * 2)
    assert not result["passed"]
    assert result["violations"] == ["reference_top5_overlap"]
    # FP32 exp-sum reduction may round differently after permutation; require
    # the unchanged declared PPL policies to pass, not bit-identical metrics.
    assert result["reference_ppl_relative_delta"] <= result["thresholds"]["max_reference_ppl_relative_delta"]
    assert result["fp32_ppl_increase"] <= result["thresholds"]["max_fp32_ppl_increase"]
    assert result["min_reference_cosine"] >= 0.995
    assert result["min_reference_top5_overlap"] == 3


def test_fp32_loss_fails_even_when_compiled_and_reference_are_identical():
    fp32 = [np.eye(8, dtype=np.float32)[target] * 10 for target in [1, 2]]
    result = _report(fp32_logits=fp32)
    assert result["reference_ppl_relative_delta"] == 0
    assert "fp32_perplexity" in result["violations"]
    assert not result["passed"]


def test_unused_padding_is_excluded_but_live_nan_is_not():
    rows = [np.r_[np.arange(8, dtype=np.float32), np.nan]] * 2
    assert _report(compiled_logits=rows)["passed"]


def test_simulator_comparison_resolves_implicit_calibration_once(monkeypatch):
    calls, captured = [], []
    scales = {"block0_head0_softmax": 1 / 127, "block0_head0_key": 0.02}
    def calibrate(payload):
        calls.append(payload)
        return scales
    def logits(payload, tokens, *, calibration_scales, **kwargs):
        captured.append(calibration_scales)
        return [np.arange(4, dtype=np.float32)]
    monkeypatch.setattr(ppl, "build_calibration_scales", calibrate)
    monkeypatch.setattr(ppl, "run_weight_only_int8_simulator_teacher_forced_logits", logits)
    monkeypatch.setattr(ppl, "run_weight_only_int8_simulator_reference_teacher_forced_logits", logits)
    payload = {"model_args": {"vocab_size": 4}, "state_dict": {}}
    result = ppl.evaluate_gpt2_perplexity(
        payload, calibration_token_ids=[0, 1], eval_token_ids=[0, 1],
        tokenizer_dir=".", ptq_preset="weight_only_int8", simulator_backed=True,
        compute_fp32_ceiling=False)
    assert calls == [payload] and len(captured) == 2
    assert captured[0] is scales and captured[1] is scales
    assert result.relative_delta == 0


@pytest.mark.parametrize("sm_scale", [None, 0.02])
def test_attention_default_composite_matches_probability_quantization(sm_scale):
    from taccel.compiler.codegen import CodeGenerator
    from taccel.compiler.ir import IRGraph, IRNode
    from taccel.compiler.model_config import ModelConfig
    from taccel.golden_model import memory
    from taccel.golden_model.simulator import Simulator
    from taccel.golden_model.state import MachineState
    from taccel.isa.opcodes import BUF_ABUF

    config = ModelConfig(model_kind="decoder", n_layer=1, n_head=1,
                         d_model=16, d_head=16, mlp_dim=16, vocab_size=16,
                         max_seq_len=16, embedding_kind="token_pos")
    scales = {} if sm_scale is None else {"sm": sm_scale}
    cg = CodeGenerator({}, scales, {}, model_config=config)
    graph = IRGraph()
    graph.add_node(IRNode("matmul_attn_v", "av", ["sm", "v"], (16, 16),
                          {"query_len": 16, "key_len": 16, "head_idx": 0}))
    instructions, blob = cg.generate(graph)
    sm = np.eye(16, dtype=np.float32)
    v = np.random.default_rng(3).normal(size=(16, 16)).astype(np.float16).astype(np.float32)
    state = MachineState()
    state.dram[:len(blob)] = blob
    memory.write_fp16_tile(state, BUF_ABUF, cg.mem.abuf.get("sm").offset_units, sm)
    memory.write_fp16_tile(state, BUF_ABUF, cg.mem.abuf.get("v").offset_units, v)
    sim = Simulator(state)
    for insn in instructions:
        sim._execute(insn)
    actual = memory.read_fp16_tile(state, BUF_ABUF, cg.mem.abuf.get("av").offset_units, 16, 16)
    probability_scale = 1 / 127 if sm_scale is None else sm_scale
    value_scale = 6 / 127
    q_sm = np.clip(np.rint(sm * np.float32(np.float16(1 / probability_scale))), -128, 127).astype(np.int32)
    q_v = np.clip(np.rint(v * np.float32(np.float16(1 / value_scale))), -128, 127).astype(np.int32)
    composite = np.float32(np.float16(np.float32(probability_scale) * np.float32(value_scale)))
    expected = ((q_sm @ q_v).astype(np.float32) * composite).astype(np.float16).astype(np.float32)
    np.testing.assert_array_equal(actual, expected)
