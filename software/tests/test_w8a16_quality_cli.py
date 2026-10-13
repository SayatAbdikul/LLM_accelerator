"""Bounded mocked orchestration tests; no model weights or native execution."""
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest

from tools import w8a16_quality_gate as gate


def _argv(tmp_path):
    return ["--checkpoint", str(tmp_path / "model.pt"),
            "--tokenizer-dir", str(tmp_path / "tokenizer"),
            "--calibration-text", str(tmp_path / "calib.txt"),
            "--eval-text", str(tmp_path / "eval.txt"),
            "--ptq-preset", "weight_only_int8",
            "--max-fp32-ppl-increase-percent", "5", "--max-eval-tokens", "3",
            "--out", str(tmp_path / "evidence")]


def _fake_rails(args, state, rails):
    state["input_ids"] = {"evaluation": [0, 1, 2], "inputs": [0, 1], "targets": [1, 2]}
    state["model_args"] = {"vocab_size": 8}
    state["model_args_sha256"] = gate.canonical_hash(state["model_args"])
    for name in rails:
        rails[name].extend([np.arange(8, dtype=np.float32)] * 2)


def test_budget_is_required_and_active_presets_are_closed(tmp_path):
    args = _argv(tmp_path)
    at = args.index("--max-fp32-ppl-increase-percent")
    with pytest.raises(SystemExit):
        gate.parser().parse_args(args[:at] + args[at + 2:])
    args[args.index("--ptq-preset") + 1] = "control"
    with pytest.raises(SystemExit):
        gate.parser().parse_args(args)


def test_canonical_window_default_and_debug_window_are_explicit(tmp_path):
    args = _argv(tmp_path)
    at = args.index("--max-eval-tokens")
    assert gate.parser().parse_args(args[:at] + args[at + 2:]).max_eval_tokens == 257
    assert gate.parser().parse_args(args).max_eval_tokens == 3
    assert gate.parser().parse_args(args).rtl_export_steps == 0


def test_success_retains_three_rails_ids_and_unchanged_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "run_three_rails", _fake_rails)
    assert gate.main(_argv(tmp_path)) == 0
    out = tmp_path / "evidence"
    report = json.loads((out / "report.json").read_text())
    assert report["passed"] and report["status"] == "passed"
    assert report["scope"]["debug_noncanonical_window"]
    assert not report["scope"]["official_corpus_wide_quality"]
    assert report["quality"]["thresholds"] == {
        "max_reference_ppl_relative_delta": 0.02, "min_reference_cosine": 0.995,
        "min_reference_top5_overlap": 4, "max_fp32_ppl_increase": 0.05}
    assert report["logits_artifact"]["row_counts"] == {"compiled": 2, "reference": 2, "fp32": 2}
    with np.load(out / "logits.npz", allow_pickle=False) as saved:
        assert saved["token_ids"].tolist() == [0, 1, 2]
        assert len(saved.files) == 7
        np.testing.assert_array_equal(saved["compiled_0001"], np.arange(8))
    for name, expected in report["artifact_sha256"].items():
        assert gate.sha256(out / name) == expected


@pytest.mark.parametrize("failure", ["reference_gap", "fp32_gap", "nonfinite", "missing_row", "wrong_shape"])
def test_failing_metrics_or_rows_keep_raw_evidence(tmp_path, monkeypatch, failure):
    def run(args, state, rails):
        _fake_rails(args, state, rails)
        if failure == "reference_gap":
            rails["reference"] = [np.arange(8, dtype=np.float32) * 3] * 2
        elif failure == "fp32_gap":
            rails["fp32"] = [np.eye(8, dtype=np.float32)[target] * 10 for target in (1, 2)]
        elif failure == "nonfinite":
            rails["compiled"][0] = np.full(8, np.inf)
        elif failure == "missing_row":
            rails["compiled"].pop()
        else:
            rails["compiled"][0] = np.zeros((1, 8))
    monkeypatch.setattr(gate, "run_three_rails", run)
    assert gate.main(_argv(tmp_path)) != 0
    report = json.loads((tmp_path / "evidence/report.json").read_text())
    assert not report["passed"] and report["status"] in {"failed", "error"}
    assert report["model_args_sha256"] == gate.canonical_hash({"vocab_size": 8})
    assert report["logits_artifact"]["row_counts"]["compiled"] == (1 if failure == "missing_row" else 2)
    assert (tmp_path / "evidence/input_ids.json").is_file()


def test_missing_prerequisite_retains_failure_report(tmp_path, monkeypatch):
    def missing(args, state, rails):
        state["sources"] = {"sha256": {"source": "fixed"}}
        raise FileNotFoundError("mandatory local fixture")
    monkeypatch.setattr(gate, "run_three_rails", missing)
    monkeypatch.setattr(gate, "_source_identity", lambda: {"sha256": {"source": "fixed"}})
    assert gate.main(_argv(tmp_path)) == 2
    report = json.loads((tmp_path / "evidence/report.json").read_text())
    assert report["error"]["type"] == "FileNotFoundError"
    assert report["sources"]["sha256"] == {"source": "fixed"}
    assert report["logits_artifact"]["row_counts"] == {"compiled": 0, "reference": 0, "fp32": 0}


def test_source_drift_overrides_metric_pass(tmp_path, monkeypatch):
    def run(args, state, rails):
        _fake_rails(args, state, rails)
        state["sources"] = {"sha256": {"source": "old"}}
    monkeypatch.setattr(gate, "run_three_rails", run)
    monkeypatch.setattr(gate, "_source_identity", lambda: {"sha256": {"source": "new"}})
    assert gate.main(_argv(tmp_path)) == 2
    report = json.loads((tmp_path / "evidence/report.json").read_text())
    assert report["quality"]["passed"] and not report["passed"]
    assert report["identities_unchanged"] is False


def test_existing_output_is_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "run_three_rails", _fake_rails)
    assert gate.main(_argv(tmp_path)) == 0
    path = tmp_path / "evidence/report.json"
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        gate.main(_argv(tmp_path))
    assert path.read_bytes() == before


@pytest.mark.parametrize("budget", ["nan", "inf", "-1"])
def test_invalid_budget_fails_before_model_work(tmp_path, monkeypatch, budget):
    monkeypatch.setattr(gate, "run_three_rails", lambda *args: pytest.fail("model work must not start"))
    args = _argv(tmp_path)
    args[args.index("--max-fp32-ppl-increase-percent") + 1] = budget
    assert gate.main(args) == 2
    report = json.loads((tmp_path / "evidence/report.json").read_text())
    assert report["passed"] is False
    assert report["error"]["type"] == "ValueError"
    if budget in {"nan", "inf"}:
        assert report["arguments"]["max_fp32_ppl_increase_percent"] == budget


def test_invalid_json_does_not_create_partial_artifact(tmp_path):
    path = tmp_path / "invalid.json"
    with pytest.raises(ValueError):
        gate._write_json(path, {"invalid": float("nan")})
    assert not path.exists()


def test_exact_window_overlap_is_rejected_with_explicit_scope():
    match = gate.calibration_overlap_report([7, 1, 2, 3, 6], [1, 2, 3], [[7, 1]])
    assert not match["passed"] and match["matches"][0]["kind"] == "evaluation_window_in_calibration_text"
    match = gate.calibration_overlap_report([6, 7], [0, 1, 2, 3], [[1, 2]])
    assert not match["passed"]
    clean = gate.calibration_overlap_report([6, 7], [0, 1, 2, 3], [[6, 7]])
    assert clean["passed"] and "partial shorter overlaps" in clean["scope"]


def test_shared_scales_require_softmax_and_matching_kv_aliases():
    scales = {"block0_head0_" + suffix: 0.02 for suffix in
              ("query", "key", "value", "softmax", "key_kv_load", "value_kv_load")}
    model = {"n_layer": 1, "n_head": 1}
    gate._validate_scales(scales, model)
    del scales["block0_head0_softmax"]
    with pytest.raises(ValueError, match="softmax"):
        gate._validate_scales(scales, model)
    scales["block0_head0_softmax"] = 0.02
    scales["block0_head0_key_kv_load"] = 0.03
    with pytest.raises(ValueError, match="prefill/decode"):
        gate._validate_scales(scales, model)


@pytest.mark.parametrize("preset", gate.PRESETS)
@pytest.mark.parametrize("unhalted_position", [None, 0, 1])
def test_orchestration_fp32_before_rotation_one_shared_map_and_continuous_kv(
        tmp_path, monkeypatch, preset, unhalted_position):
    from dataclasses import asdict
    from taccel.compiler.options import CompilerOptions
    from taccel.runtime import calibration, fp32_reference, host_runner, tiny_fixture, w8a16_simulator_reference
    events, map_ids, captures = [], [], []
    payload = {"model_args": {"vocab_size": 8, "n_layer": 1, "n_head": 1, "block_size": 8},
               "state_dict": {"rotated": False}}
    scales = {"block0_head0_" + suffix: 0.02 for suffix in
              ("query", "key", "value", "softmax", "key_kv_load", "value_kv_load")}
    def prepare(args, state):
        state["model_args"] = payload["model_args"]
        state["input_ids"] = {"evaluation": [1, 2, 3]}
        return payload, [7, 6, 5, 4], [1, 2, 3]
    class FP32:
        def __init__(self, state_dict, model_args):
            assert state_dict["rotated"] is False
        def incremental_logits_trace(self, inputs):
            assert inputs == [1, 2]
            events.append("fp32")
            return [np.arange(8, dtype=np.float32)] * 2
    def rotate(model, tokens, **kwargs):
        assert events == ["fp32"]
        events.append("rotate")
        model["state_dict"]["rotated"] = True
        return ["mock_weight"], {"seed": kwargs["seed"]}
    def calibrate(model, *args, **kwargs):
        events.append("calibrate")
        assert kwargs["seq_len"] == (16 if preset == "weight_only_int8" else 32)
        return scales
    class Reference:
        def __init__(self, model, *, calibration_scales, attention_window):
            assert model["state_dict"]["rotated"] == (preset == "weight_only_int8_quarot")
            assert attention_window == 2
            map_ids.append(id(calibration_scales))
            self.positions = []
        def run_decode_step(self, token, position):
            assert position == len(self.positions) and token == [1, 2][position]
            self.positions.append(position)
            return np.arange(8, dtype=np.float32)
    bundle = SimpleNamespace(prefill_instrs=b"p", decode_instrs=b"d", shared_data=b"data")
    def build(model, *, calibration_scales, smoke_decode_steps, batch, prefill_tokens, target, **kwargs):
        assert smoke_decode_steps == 1 and batch == prefill_tokens == 1 and target == "rtl-gen2"
        assert isinstance(kwargs["compiler_options"], CompilerOptions)
        assert asdict(kwargs["compiler_options"]) == asdict(CompilerOptions())
        assert not any(asdict(kwargs["compiler_options"]).values())
        map_ids.append(id(calibration_scales))
        return SimpleNamespace(build=SimpleNamespace(bundle=bundle))
    class Runner:
        def __init__(self, value, *, logits_dtype, vocab_size):
            assert value is bundle and logits_dtype == np.float16 and vocab_size == 8
            self.next_position = 0
            self.simulator = SimpleNamespace(state=SimpleNamespace(halted=False))
        def run_prefill(self, tokens):
            assert tokens == [1] and self.next_position == 0
            self.next_position = 1
            self.simulator.state.halted = unhalted_position != 0
            return np.arange(8, dtype=np.float16)
        def run_decode_step(self, token, position):
            assert token == 2 and position == self.next_position == 1
            self.next_position += 1
            self.simulator.state.halted = unhalted_position != position
            return np.arange(8, dtype=np.float16)
    monkeypatch.setattr(gate, "prepare_inputs", prepare)
    monkeypatch.setattr(fp32_reference, "NanoGPTFP32Reference", FP32)
    monkeypatch.setattr(calibration, "apply_quarot_rotation_from_token_ids", rotate)
    monkeypatch.setattr(calibration, "build_calibration_scales", calibrate)
    monkeypatch.setattr(calibration, "build_calibration_scales_from_token_ids", calibrate)
    monkeypatch.setattr(calibration, "build_calibration_seqs", lambda *a, **kw: [[7, 6]])
    monkeypatch.setattr(calibration, "build_calibration_seqs_from_token_ids", lambda *a, **kw: [[7, 6]])
    monkeypatch.setattr(w8a16_simulator_reference, "NanoGPTW8A16SimulatorReference", Reference)
    monkeypatch.setattr(tiny_fixture, "build_stage3_tiny_decoder_bundle", build)
    monkeypatch.setattr(host_runner, "HostRunner", Runner)
    monkeypatch.setattr(gate, "_capture_rtl_export_golden", lambda args, export, runner, position, row: captures.append(position))
    args = _argv(tmp_path)
    args[args.index("--ptq-preset") + 1] = preset
    assert gate.main(args) == (0 if unhalted_position is None else 2)
    assert events.count("calibrate") == 1 and len(map_ids) == 2 and map_ids[0] == map_ids[1]
    report = json.loads((tmp_path / "evidence/report.json").read_text())
    assert report["transforms"]["fp32_captured_before_ptq"]
    assert report["transforms"]["quarot"]["applied"] == (preset == "weight_only_int8_quarot")
    assert report["original_fp32_trace"]["row_count"] == 2
    assert report["reference"]["attention_window"] == report["compiled"]["attention_window"] == 2
    assert report["compiled"]["compiler_options"] == asdict(CompilerOptions())
    accepted_count = 2 if unhalted_position is None else unhalted_position
    assert report["logits_artifact"]["row_counts"]["compiled"] == accepted_count
    assert captures == list(range(accepted_count))
    if unhalted_position is None:
        assert all(step["halted"] for step in report["compiled_step_execution"])
    else:
        assert not report["passed"] and report["status"] == "error"
        assert "quality" not in report
        assert report["error"]["type"] == "RuntimeError"
        assert f"position {unhalted_position} did not HALT" in report["error"]["message"]
        assert report["compiled_step_execution"][-1] == {
            "position": unhalted_position, "token_id": [1, 2][unhalted_position],
            "stream": "prefill" if unhalted_position == 0 else "decode", "halted": False}
        with np.load(tmp_path / "evidence/logits.npz", allow_pickle=False) as saved:
            assert f"compiled_{unhalted_position:04d}" not in saved.files


def _export_fixture(tmp_path):
    from taccel.assembler.assembler import ProgramBundle, RuntimeConfigAttnSite, RuntimePatchSite
    from taccel.isa.encoding import encode
    from taccel.isa.instructions import ConfigAttnInsn, HaltInsn, SetAddrHiInsn, SetAddrLoInsn
    instructions = []
    for register in range(3):
        instructions.extend([SetAddrLoInsn(addr_reg=register, imm28=0),
                             SetAddrHiInsn(addr_reg=register, imm28=0)])
    instructions.extend([ConfigAttnInsn(query_row_base=0, valid_kv_len=1, mode=3), HaltInsn()])
    raw = b"".join(encode(insn) for insn in instructions)
    bundle = ProgramBundle(
        prefill_instrs=raw, decode_instrs=raw, shared_data=bytes(range(256)) * 64,
        logits_size=32, kv_cache_size=256 * 32, embedding_row_bytes=32, kv_step_bytes=32,
        symbol_offsets={"tokens": 0, "positions": 512, "kv": 0}, symbol_regions={"kv": "kv_cache"},
        runtime_patch_sites=[
            RuntimePatchSite(stream, kind, 2 * register, 2 * register + 1, 0, 0, register, symbol)
            for stream in ("prefill", "decode")
            for register, (kind, symbol) in enumerate((("token_embed", "tokens"),
                                                     ("pos_embed", "positions"), ("kv_base", "kv")))],
        runtime_config_attn_sites=[RuntimeConfigAttnSite(stream, 6, 0, 3)
                                  for stream in ("prefill", "decode")])
    args = gate.parser().parse_args(_argv(tmp_path))
    args.out.mkdir()
    args.rtl_export_steps = 3
    state = {"model_args": {"vocab_size": 16}, "transforms": {"quarot": {"applied": True}},
             "scales_sha256": "fixed-shared-map", "compiled": {"attention_window": 256}}
    return args, state, bundle, [1, 2, 3] + [0] * 253


def test_rtl_export_retains_data_addresses_full_window_and_zero_initial_kv(tmp_path):
    from taccel.assembler.assembler import ProgramBinary, read_set_addr_pair
    from taccel.isa.encoding import decode
    args, state, bundle, inputs = _export_fixture(tmp_path)
    before = bundle.materialize()
    manifest = gate._begin_rtl_export(args, state, bundle, inputs)
    directory = args.out / "rtl_export"
    data = (directory / "static_data.bin").read_bytes()
    assert manifest["attention_window"] == 256 and manifest["token_ids"] == [1, 2, 3]
    assert manifest["transforms"]["quarot"]["applied"]
    assert manifest["scales_sha256"] == "fixed-shared-map"
    assert data == before[bundle.data_base:]
    kv_offset = bundle.kv_cache_base - bundle.data_base
    assert not any(data[kv_offset:])
    assert bundle.materialize(reset_runtime=False) == before
    assert len(list(directory.glob("static_data*"))) == 1
    for position, step in enumerate(manifest["steps"]):
        path = directory / step["program"]["file"]
        program = ProgramBinary.from_bytes(path.read_bytes())
        assert program.data == b"" and program.entry_point == 0
        assert program.data_base == bundle.data_base and program.insn_count == 8
        assert read_set_addr_pair(program.instructions, 0, 1, 0) == bundle.data_base + inputs[position] * 32
        assert read_set_addr_pair(program.instructions, 2, 3, 1) == bundle.data_base + 512 + position * 32
        assert read_set_addr_pair(program.instructions, 4, 5, 2) == bundle.kv_cache_base + position * 32
        config = decode(program.instructions[48:56])
        assert (config.query_row_base, config.valid_kv_len) == (position, position + 1)
        assert gate.sha256(path) == step["program"]["sha256"]
        program.data = data
        assert program.to_dram_image()[bundle.data_base:] == data
    assert "preceding RTL" in manifest["replay_contract"]["continuation"]
    assert "never use golden KV" in manifest["replay_contract"]["golden_outputs"]
    assert not (directory / "manifest.json").exists()


def test_rtl_export_golden_capture_is_contiguous_raw_and_comparison_only(tmp_path):
    from taccel.runtime.host_runner import BundlePatcher
    args, state, bundle, inputs = _export_fixture(tmp_path)
    manifest = gate._begin_rtl_export(args, state, bundle, inputs)
    dram = bytearray(bundle.materialize())
    runner = SimpleNamespace(bundle=bundle, simulator=SimpleNamespace(state=SimpleNamespace(dram=dram)))
    patcher = BundlePatcher(bundle)
    golden_rows = []
    for position in range(3):
        stream = "prefill" if position == 0 else "decode"
        patcher._patch_embeddings(stream, [inputs[position]], [position])
        patcher._patch_kv_bases(position, stream=stream)
        patcher._patch_attention_context(stream, position, position + 1)
        row = np.arange(16, dtype=np.float16) + position
        golden_rows.append(row.tobytes())
        dram[bundle.logits_base:bundle.logits_base + 32] = row.tobytes()
        start = bundle.kv_cache_base + position * 32
        dram[start:start + 32] = row.tobytes()
        gate._capture_rtl_export_golden(args, manifest, runner, position, row)
    directory = args.out / "rtl_export"
    saved = json.loads((directory / "manifest.json").read_text())
    assert saved["status"] == "golden_complete_rtl_not_run"
    assert len(saved["golden_steps"]) == 3
    for position, record in enumerate(saved["golden_steps"]):
        assert (directory / record["logits"]["file"]).read_bytes() == golden_rows[position]
        kv = (directory / record["kv"]["file"]).read_bytes()
        prefix = b"".join(golden_rows[:position + 1])
        assert kv[:len(prefix)] == prefix and not any(kv[len(prefix):])
        assert len(kv) == bundle.kv_cache_size_bytes
    static_data = (directory / "static_data.bin").read_bytes()
    assert not any(static_data[bundle.kv_cache_base - bundle.data_base:])
    with pytest.raises(ValueError, match="contiguous"):
        gate._capture_rtl_export_golden(args, manifest, runner, 1, row)


@pytest.mark.parametrize("failure", ["wrong_dtype", "nonfinite", "instruction_drift", "dram_mismatch", "short_kv"])
def test_rtl_export_capture_rejects_inconsistent_evidence(tmp_path, failure):
    from taccel.runtime.host_runner import BundlePatcher
    args, state, bundle, inputs = _export_fixture(tmp_path)
    manifest = gate._begin_rtl_export(args, state, bundle, inputs)
    dram = bytearray(bundle.materialize())
    row = np.arange(16, dtype=np.float16)
    dram[bundle.logits_base:bundle.logits_base + 32] = row.tobytes()
    patcher = BundlePatcher(bundle)
    patcher._patch_embeddings("prefill", [1], [0])
    patcher._patch_kv_bases(0, stream="prefill")
    patcher._patch_attention_context("prefill", 0, 1)
    if failure == "wrong_dtype":
        row = row.astype(np.float32)
    elif failure == "nonfinite":
        row[0] = np.nan
    elif failure == "instruction_drift":
        patcher._patch_embeddings("prefill", [2], [0])
    elif failure == "dram_mismatch":
        dram[bundle.logits_base] ^= 1
    else:
        del dram[-1]
    runner = SimpleNamespace(bundle=bundle, simulator=SimpleNamespace(state=SimpleNamespace(dram=dram)))
    with pytest.raises(ValueError):
        gate._capture_rtl_export_golden(args, manifest, runner, 0, row)
    assert not (args.out / "rtl_export/manifest.json").exists()


def test_export_count_cannot_exceed_available_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "run_three_rails", lambda *args: pytest.fail("model work must not start"))
    assert gate.main(_argv(tmp_path) + ["--rtl-export-steps", "3"]) == 2
    report = json.loads((tmp_path / "evidence/report.json").read_text())
    assert not report["passed"] and "input count" in report["error"]["message"]
