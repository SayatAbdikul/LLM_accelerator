"""Short-context packed attention must bound delayed-gather AV residency."""
from dataclasses import asdict

import pytest

from taccel.compiler.codegen import CodeGenerator
from taccel.compiler.ir import IRGraph, IRNode
from taccel.compiler.model_config import deit_tiny_config
from taccel.compiler.options import CompilerOptions
from taccel.compiler.w8a16_emit import attention
from taccel.golden_model.simulator import Simulator
from taccel.runtime.tiny_fixture import build_stage3_tiny_decoder_bundle
from tools.optimization_matrix import build_program, compare_outputs, inspect_output, workload_steps
from tools.train_tiny_fixture import _make_checkpoint_payload


@pytest.fixture(scope="module")
def payloads():
    return {key: _make_checkpoint_payload(n_layer=1, n_head=heads, d_model=width,
                                         block_size=128, weight_scale=.02)
            for key, width, heads in ((64, 64, 1), (768, 768, 12), ("tiny4", 64, 4))}


def _build(payload, batch, key_len, optimized=False, scales=None):
    options = CompilerOptions(**{name: optimized for name in asdict(CompilerOptions())})
    return build_stage3_tiny_decoder_bundle(payload, batch=batch, smoke_decode_steps=key_len - 1,
                                           calibration_scales=scales or {}, compiler_options=options)


def _old_policy(cg, node, m_pad, n_pad, key_pad):
    return cg.current_node_idx >= 0 and key_pad >= 64


@pytest.mark.parametrize("batch", [16, 32])
@pytest.mark.parametrize("key_len", [1, 3, 16, 17, 32, 33, 48, 49])
@pytest.mark.parametrize("optimized", [False, True])
def test_short_window_full_width_cohort_fits_by_spilling(payloads, batch, key_len, optimized):
    tiny = _build(payloads[768], batch, key_len, optimized)
    cg = tiny.build.decode_codegen
    assert cg.mem.abuf.high_water_units * 16 <= cg.mem.abuf.capacity
    events = cg.compilation_metrics()["optimization_events"]
    decisions = [event for event in events if event["kind"] == "batched_av_residency"]
    if key_len <= 48:
        assert len(decisions) == 1 and decisions[0]["spill"]
        assert decisions[0]["cohort_bytes"] == batch * 12 * 16 * 64 * 2
    else:
        assert not decisions  # unchanged established >=64-key spill path
    assert not any("fallback" in event["kind"] for event in events)
    if optimized:
        assert any(event["kind"] == "fc2_quantization_hit" for event in events)


def test_old_short_window_policy_reproduces_next_packed_q_failure(payloads, monkeypatch):
    monkeypatch.setattr(attention, "_spill_batched_av_cohort", _old_policy)
    with pytest.raises(MemoryError, match="block0_s4_g0_pqkt__qpack"):
        _build(payloads[768], 16, 3)


@pytest.mark.parametrize("batch", [16, 32])
@pytest.mark.parametrize("width,key_len", [(64, 3), (768, 64), (768, 128)])
def test_previously_fitting_or_large_window_schedules_are_byte_unchanged(
        payloads, monkeypatch, batch, width, key_len):
    candidate = _build(payloads[width], batch, key_len)
    monkeypatch.setattr(attention, "_spill_batched_av_cohort", _old_policy)
    baseline = _build(payloads[width], batch, key_len)
    assert candidate.build.bundle.prefill_instrs == baseline.build.bundle.prefill_instrs
    assert candidate.build.bundle.decode_instrs == baseline.build.bundle.decode_instrs
    assert candidate.build.bundle.shared_data == baseline.build.bundle.shared_data
    assert candidate.build.bundle.materialize() == baseline.build.bundle.materialize()


@pytest.mark.parametrize("batch", [16, 32])
def test_spill_reload_preserves_logical_logits_and_contiguous_kv_with_poisoned_abuf(
        payloads, monkeypatch, batch):
    from taccel.runtime.calibration import build_calibration_scales
    scales = build_calibration_scales(payloads["tiny4"])
    resident = _build(payloads["tiny4"], batch, 3, scales=scales)
    monkeypatch.setattr(attention, "_spill_batched_av_cohort",
                        lambda cg, node, *dims: cg.current_node_idx >= 0)
    spilled = _build(payloads["tiny4"], batch, 3, scales=scales)
    outputs = []
    for tiny in (resident, spilled):
        previous_kv, sequence = None, []
        for step in workload_steps(batch, 1, [], 2, 1, 128):
            program, layout = build_program(tiny, step, previous_kv)
            sim = Simulator()
            sim.load_program(program)
            sim.state.abuf[:] = b"\x35" * len(sim.state.abuf)
            sim.run()
            assert sim.state.halted
            raw = bytes(sim.state.dram[layout["data_base"]:layout["data_base"] + layout["data_bytes"]])
            metrics, previous_kv = inspect_output(raw, layout, tiny.build.bundle.shared_data)
            assert any(previous_kv)
            sequence.append(metrics)
        outputs.append(sequence)
    assert all(compare_outputs(a, b)["passed"] for a, b in zip(*outputs))


def test_residency_decision_resets_on_regeneration_and_whole_graph_retry(monkeypatch):
    cg = CodeGenerator({}, {}, {}, model_config=deit_tiny_config(),
                       options=CompilerOptions(reuse_qkv_quantization=True))
    graph = IRGraph()
    graph.add_node(IRNode("gelu", "output", ["external"], (1, 16)))
    calls = []
    monkeypatch.setattr(CodeGenerator, "_layout_weights", lambda self, graph: None)

    def emit(self, node):
        assert not self.batched_av_residency
        self.batched_av_residency[0] = True
        calls.append(self)
        if len(calls) == 1:
            raise MemoryError("test whole-graph retry")

    monkeypatch.setattr(CodeGenerator, "_emit_node", emit)
    cg.batched_av_residency[0] = False
    cg.generate(graph)
    assert len(calls) == 2 and calls[0] is not calls[1]
    cg.generate(graph)
    assert len(calls) == 3 and calls[-1] is cg


def test_cohort_admission_uses_contiguous_space_and_fails_closed():
    cg = CodeGenerator({}, {}, {}, model_config=deit_tiny_config())
    cg.current_node_idx = 0
    node = IRNode("matmul_attn_v", "av", [], (1, 64),
                  {"block_idx": 0, "av_cohort_tiles": 16, "av_cohort_streams": 16,
                   "av_cohort_pack_bytes": 1024})
    for i in range(4):
        cg.mem.abuf.alloc(f"hole{i}", 24 * 1024)
        cg.mem.abuf.alloc(f"live{i}", 8 * 1024)
    for i in range(4):
        cg.mem.abuf.free(f"hole{i}")
    assert cg.mem.abuf.free_bytes == 96 * 1024
    assert attention._spill_batched_av_cohort(cg, node, 16, 64, 16)
    assert cg.optimization_events[-1]["largest_free_bytes"] == 24 * 1024
    cg.batched_av_residency.clear()
    node.attrs.pop("av_cohort_streams")
    assert attention._spill_batched_av_cohort(cg, node, 16, 64, 16)


def test_half_capacity_boundary_and_independent_blocks():
    cg = CodeGenerator({}, {}, {}, model_config=deit_tiny_config())
    cg.current_node_idx = 0
    node = IRNode("matmul_attn_v", "av", [], (1, 64),
                  {"block_idx": 0, "av_cohort_tiles": 32, "av_cohort_streams": 32,
                   "av_cohort_pack_bytes": 1024})
    assert not attention._spill_batched_av_cohort(cg, node, 16, 64, 16)
    assert cg.optimization_events[-1]["cohort_bytes"] == cg.mem.abuf.capacity // 2
    node.attrs.update(block_idx=1, av_cohort_tiles=33)
    assert attention._spill_batched_av_cohort(cg, node, 16, 64, 16)
    assert cg.batched_av_residency == {0: False, 1: True}
    # The established padded-key threshold takes precedence over the cache.
    node.attrs["block_idx"] = 0
    assert attention._spill_batched_av_cohort(cg, node, 16, 64, 64)
