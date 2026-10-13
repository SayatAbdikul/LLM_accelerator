"""Fail-closed offline W8A16 quality gate with retained three-rail evidence.

Example (the FP32-loss budget is an explicit acceptance decision)::

  PYTHONPATH=software .venv/bin/python software/tools/w8a16_quality_gate.py \
    --checkpoint PATH.pt --tokenizer-dir LOCAL_TOKENIZER \
    --calibration-text CALIB.txt --eval-text EVAL.txt \
    --ptq-preset weight_only_int8_calibrated \
    --max-fp32-ppl-increase-percent 5 --out NEW_DIRECTORY

Runs the compiled golden bundle, independent W8A16 reference, and original
FP32 model on identical teacher-forced prefixes. This is not RTL execution or
official corpus-wide evaluation. The caller supervises time/RSS/disk limits.
Nothing is downloaded and an existing output directory is never overwritten.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "software"))
PRESETS = ("weight_only_int8", "weight_only_int8_calibrated", "weight_only_int8_quarot")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                   "added_tokens.json", "vocab.json", "merges.txt", "config.json",
                   "config.provenance.json")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _write_json(path, value):
    # Serialize before creating the exclusive artifact so an invalid value never
    # leaves a truncated JSON document that looks like retained evidence.
    serialized = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8") as output:
        output.write(serialized)
        output.flush()
        os.fsync(output.fileno())


def _phase_snapshot(args, state, name):
    _write_json(args.out / f"phase_{name}.json", state)


def _check_rows(rows, expected, vocab_size, label):
    import numpy as np
    if len(rows) != expected:
        raise ValueError(f"{label}: expected exactly {expected} rows, got {len(rows)}")
    for position, row in enumerate(rows):
        values = np.asarray(row)
        if values.ndim != 1 or values.size < vocab_size or values.dtype.kind not in "fiu":
            raise ValueError(f"{label} step {position}: invalid logits row")
        if not np.isfinite(values[:vocab_size]).all():
            raise ValueError(f"{label} step {position}: nonfinite live logits")


def _identity(path):
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"required regular file missing: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def _source_identity():
    paths = sorted((REPO / "software/taccel").rglob("*.py"))
    paths += [Path(__file__).resolve(), REPO / "software/tools/check_reference_manifest.py",
              REPO / "software/tests/fixtures/gen2_reference_manifest.json"]
    result = {"sha256": {p.relative_to(REPO).as_posix(): sha256(p) for p in paths}}
    try:
        result["commit"] = subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        result["commit"] = None
    result["scope"] = "all Python package sources, this CLI, and frozen-reference checker/manifest"
    return result


def _runtime_identity():
    packages = {}
    for name in ("numpy", "torch", "transformers", "safetensors"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "packages": packages,
            "dependencies": "existing installed environment; no installation or network"}


def _tokens_valid(tokens, vocab_size, label):
    if not tokens:
        raise ValueError(f"{label}: no tokens")
    if any(type(t) is not int or not 0 <= t < vocab_size for t in tokens):
        raise ValueError(f"{label}: invalid token outside live vocabulary")


def _find_window(haystack, needle):
    """Return the first exact contiguous occurrence, without hashing collisions."""
    if not needle or len(needle) > len(haystack):
        return None
    for start in range(len(haystack) - len(needle) + 1):
        if haystack[start] == needle[0] and haystack[start:start + len(needle)] == needle:
            return start
    return None


def calibration_overlap_report(calibration_tokens, evaluation_tokens, windows):
    """Check the retained eval window and every actually selected calibration window.

    A token-ID corpus has no reliable source offsets unless its provenance is
    known. These exact-window checks do not assert disjoint Wikipedia articles
    or exclude shorter shared phrases; the limitation is explicit in evidence.
    """
    if not calibration_tokens or not evaluation_tokens or not windows:
        raise ValueError("nonoverlap check requires nonempty corpora and calibration windows")
    overlap = []
    full = _find_window(calibration_tokens, evaluation_tokens)
    if full is not None:
        overlap.append({"kind": "evaluation_window_in_calibration_text", "calibration_offset": full})
    for index, window in enumerate(windows):
        if not window:
            raise ValueError("empty calibration window")
        if len(window) <= len(evaluation_tokens):
            at = _find_window(evaluation_tokens, window)
            if at is not None:
                overlap.append({"kind": "selected_calibration_window_in_evaluation", "window": index,
                                "evaluation_offset": at, "tokens": len(window)})
        else:
            at = _find_window(window, evaluation_tokens)
            if at is not None:
                overlap.append({"kind": "evaluation_window_in_selected_calibration", "window": index,
                                "calibration_offset": at, "tokens": len(evaluation_tokens)})
    return {"passed": not overlap, "matches": overlap,
            "calibration_window_count": len(windows),
            "calibration_window_lengths": [len(w) for w in windows],
            "scope": "exact full evaluated/selected calibration token windows only; "
                     "partial shorter overlaps and dataset split provenance are not established"}


def _corpus_identity(path):
    result = _identity(path)
    result["origin"] = "unverified local text; filename is not dataset/split provenance"
    sidecar = Path(str(path) + ".provenance.json")
    if sidecar.is_file():
        result["supplied_provenance"] = {**_identity(sidecar), "contents": json.loads(sidecar.read_text())}
        result["origin"] = "supplied provenance retained, not independently verified"
    return result


def prepare_inputs(args, state):
    """Load only local inputs and pin them before any model computation."""
    state["sources"] = _source_identity()
    state["runtime"] = _runtime_identity()
    import torch
    from taccel.runtime.gpt2_perplexity import load_gpt2_tokenizer
    from tools.check_reference_manifest import verify_reference_manifest

    frozen = verify_reference_manifest()
    state["frozen_reference"] = {"files": len(frozen["sha256"]),
                                  "manifest_sha256": sha256(REPO / "software/tests/fixtures/gen2_reference_manifest.json")}
    checkpoint = _identity(args.checkpoint)
    state["checkpoint"] = checkpoint
    metadata_path = Path(str(args.checkpoint) + ".json")
    metadata_identity = _identity(metadata_path)
    state["checkpoint_metadata"] = metadata_identity
    metadata = json.loads(metadata_path.read_text())
    state["checkpoint_metadata"]["contents"] = metadata
    if metadata.get("checkpoint_sha256") != checkpoint["sha256"]:
        raise ValueError("checkpoint does not match mandatory paired metadata SHA-256")
    tokenizer_dir = args.tokenizer_dir.resolve(strict=True)
    if not tokenizer_dir.is_dir():
        raise ValueError("tokenizer directory missing")
    tokenizer_files = {name: _identity(tokenizer_dir / name) for name in TOKENIZER_FILES
                       if (tokenizer_dir / name).is_file()}
    if "tokenizer.json" not in tokenizer_files and not {"vocab.json", "merges.txt"} <= tokenizer_files.keys():
        raise ValueError("offline GPT-2 tokenizer requires tokenizer.json or vocab.json plus merges.txt")
    state["tokenizer"] = {"path": str(tokenizer_dir), "files": tokenizer_files,
                          "local_files_only": True}
    state["corpora"] = {"calibration": _corpus_identity(args.calibration_text),
                         "evaluation": _corpus_identity(args.eval_text)}
    if state["corpora"]["calibration"]["sha256"] == state["corpora"]["evaluation"]["sha256"]:
        raise ValueError("calibration and evaluation text must not be identical")
    tokenizer = load_gpt2_tokenizer(tokenizer_dir)
    calibration_ids = [int(t) for t in tokenizer.encode(args.calibration_text.read_text(encoding="utf-8"))]
    eval_all = tokenizer.encode(args.eval_text.read_text(encoding="utf-8"))
    eval_ids = [int(t) for t in eval_all[:args.max_eval_tokens]]
    state["input_ids"] = {"evaluation": eval_ids, "inputs": eval_ids[:-1], "targets": eval_ids[1:],
                          "calibration_token_count": len(calibration_ids),
                          "calibration_token_ids_sha256": canonical_hash(calibration_ids),
                          "evaluation_token_ids_sha256": canonical_hash(eval_ids)}
    if len(eval_ids) != args.max_eval_tokens:
        raise ValueError(f"expected exactly {args.max_eval_tokens} evaluation tokens, got {len(eval_ids)}")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_args = dict(payload["model_args"])
    state["model_args"] = model_args
    state["model_args_sha256"] = canonical_hash(model_args)
    if "model_args" in metadata and metadata["model_args"] != model_args:
        raise ValueError("paired metadata model_args do not equal checkpoint model_args")
    vocab_size = int(model_args["vocab_size"])
    _tokens_valid(eval_ids, vocab_size, "evaluation")
    _tokens_valid(calibration_ids, vocab_size, "calibration")
    if len(eval_ids) - 1 > int(model_args["block_size"]):
        raise ValueError("evaluation inputs exceed checkpoint positional context")
    state["attention_window"] = len(eval_ids) - 1
    _phase_snapshot(args, state, "inputs")
    return payload, calibration_ids, eval_ids


def _validate_scales(scales, model_args):
    """Require explicit Q/K/V/probability scales and decode aliases on both rails."""
    for name, value in scales.items():
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"invalid calibration scale: {name}")
    for layer in range(int(model_args["n_layer"])):
        for head in range(int(model_args["n_head"])):
            prefix = f"block{layer}_head{head}_"
            for suffix in ("query", "key", "value", "softmax", "key_kv_load", "value_kv_load"):
                if prefix + suffix not in scales:
                    raise ValueError(f"explicit shared scale missing: {prefix + suffix}")
            for name in ("key", "value"):
                if scales[prefix + name] != scales[prefix + name + "_kv_load"]:
                    raise ValueError(f"prefill/decode shared scale mismatch: {prefix + name}")


def _write_binary(path, data):
    with Path(path).open("xb") as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())
    return {"file": Path(path).name, "bytes": len(data), "sha256": sha256(path)}


def _begin_rtl_export(args, state, bundle, inputs):
    """Export static data once and standard instruction-only ProgramBinaries.

    The envelopes are NOT standalone runnable inputs: a replay must attach the
    static data at the retained absolute data_base. For later steps it must
    replace only the whole KV region with the preceding RTL output, never the
    separately captured golden KV. No new binary header format is introduced.
    """
    from taccel.assembler.assembler import ProgramBinary
    from taccel.runtime.host_runner import BundlePatcher

    count = args.rtl_export_steps
    if not count:
        return None
    if not 1 <= count <= min(3, len(inputs)):
        raise ValueError("RTL export requires 1..3 available teacher-forced inputs")
    if bundle.logits_rows != 1 or bundle.prefill_store_rows != 1:
        raise ValueError("RTL quality export supports B1, one-row prefill only")
    data_base, kv_base = int(bundle.data_base), int(bundle.kv_cache_base)
    kv_size, dram_size = int(bundle.kv_cache_size_bytes), int(bundle.required_dram_bytes)
    if data_base % 16 or not data_base <= kv_base < kv_base + kv_size == dram_size:
        raise ValueError("invalid RTL export data/KV layout")
    expected_bytes = (dram_size - data_base + count * (kv_size + int(bundle.logits_row_bytes)) +
                      len(bundle.prefill_instrs) + max(0, count - 1) * len(bundle.decode_instrs) + 4096)
    if expected_bytes > 1200 * 1024 * 1024:
        raise ValueError("RTL export exceeds the 1200 MiB artifact bound")
    directory = args.out / "rtl_export"
    directory.mkdir(exist_ok=False)
    manifest = {
        "schema_version": 1, "status": "programs_exported_golden_pending", "steps_requested": count,
        "scope": "first teacher-forced inputs of this exact compiled quality run; not RTL execution",
        "attention_window": len(inputs), "token_ids": list(inputs[:count]),
        "sources": state.get("sources", {}), "checkpoint": state.get("checkpoint", {}),
        "checkpoint_metadata": state.get("checkpoint_metadata", {}),
        "model_args": state["model_args"], "transforms": state["transforms"],
        "scales_sha256": state["scales_sha256"], "compiled": state["compiled"],
        "parent_report": "../report.json", "parent_report_must_have_unchanged_identities": True,
        "data_base": data_base, "required_dram_bytes": dram_size,
        "shared_data_bytes": len(bundle.shared_data),
        "shared_data_sha256": hashlib.sha256(bundle.shared_data).hexdigest(),
        "kv_cache_base": kv_base, "kv_cache_size": kv_size,
        "kv_offset_in_static_data": kv_base - data_base,
        "logits_dtype": "float16", "logits_row_bytes": int(bundle.logits_row_bytes),
        "vocab_size": int(state["model_args"]["vocab_size"]),
        "runtime_patch_sites": [asdict(site) for site in bundle.runtime_patch_sites],
        "runtime_config_attn_sites": [asdict(site) for site in bundle.runtime_config_attn_sites],
        "replay_contract": {
            "program_envelopes": "ProgramBinary with empty data; attach static_data before execution",
            "entry_point": 0, "instruction_pc": "stream-local; absolute DMA data addresses preserved",
            "initial_kv": "all zero in static_data",
            "continuation": "replace only KV with the preceding RTL whole-KV dump, at unchanged absolute addresses",
            "golden_outputs": "comparison only; never use golden KV as RTL history",
            "golden_logits": "raw full FP16 row including padding; quality comparisons use live vocabulary",
            "golden_kv": "whole KV region after each step, including unused rows and banks"},
        "estimated_binary_artifact_bytes": expected_bytes, "steps": [], "golden_steps": [],
    }
    state["rtl_export"] = manifest
    patcher = BundlePatcher(bundle)
    try:
        image = bundle.materialize(reset_runtime=True)
        if len(image) != dram_size or any(memoryview(image)[kv_base:kv_base + kv_size]):
            raise ValueError("RTL initial data image must have exactly zero whole KV")
        manifest["static_data"] = _write_binary(directory / "static_data.bin", memoryview(image)[data_base:])
        del image
        for position, token in enumerate(inputs[:count]):
            stream = "prefill" if position == 0 else "decode"
            patcher._patch_embeddings(stream, [int(token)], [position])
            patcher._patch_kv_bases(position, stream=stream)
            patcher._patch_attention_context(stream, position, position + 1)
            instructions = bundle.stream_bytes(stream)
            program = ProgramBinary(instructions=instructions, data=b"", entry_point=0,
                                    insn_count=len(instructions) // 8, data_base=data_base)
            artifact = _write_binary(directory / f"step_{position:03d}.program.bin", program.to_bytes())
            manifest["steps"].append({
                "position": position, "token_id": int(token), "stream": stream,
                "query_row_base": position, "valid_kv_len": position + 1,
                "instructions_sha256": hashlib.sha256(instructions).hexdigest(),
                "instructions_bytes": len(instructions), "insn_count": program.insn_count,
                "logits_offset": int(bundle.prefill_logits_offset if position == 0 else bundle.decode_logits_offset),
                "program": artifact})
        _write_json(directory / "manifest_initial.json", manifest)
    finally:
        # Exporting must not change how the subsequent HostRunner initializes.
        bundle.reset_runtime_images()
    return manifest


def _capture_rtl_export_golden(args, export, runner, position, row):
    import numpy as np
    if export is None or position >= export["steps_requested"]:
        return
    if position != len(export["golden_steps"]):
        raise ValueError("RTL export golden captures must be contiguous from position zero")
    values = np.asarray(row)
    if values.dtype != np.dtype(np.float16) or values.ndim != 1 or values.nbytes != export["logits_row_bytes"]:
        raise ValueError("RTL export requires an exact raw FP16 logits row")
    _check_rows([values], 1, export["vocab_size"], "RTL export golden")
    step = export["steps"][position]
    if hashlib.sha256(runner.bundle.stream_bytes(step["stream"])).hexdigest() != step["instructions_sha256"]:
        raise ValueError("golden runtime instructions differ from the RTL export")
    dram = runner.simulator.state.dram
    offset = step["logits_offset"]
    if bytes(dram[offset:offset + values.nbytes]) != values.tobytes():
        raise ValueError("golden returned logits differ from the actual DRAM row")
    base, size = export["kv_cache_base"], export["kv_cache_size"]
    kv = bytes(dram[base:base + size])
    if len(kv) != size:
        raise ValueError("RTL export golden whole-KV capture is truncated")
    directory = args.out / "rtl_export"
    record = {"position": position, "token_id": export["token_ids"][position],
              "logits": _write_binary(directory / f"golden_{position:03d}.logits.fp16.bin", values.tobytes()),
              "kv": _write_binary(directory / f"golden_{position:03d}.kv.bin", kv)}
    export["golden_steps"].append(record)
    _write_json(directory / f"golden_{position:03d}.json", record)
    if len(export["golden_steps"]) == export["steps_requested"]:
        export["status"] = "golden_complete_rtl_not_run"
        _write_json(directory / "manifest.json", export)


def run_three_rails(args, state, rails):
    import numpy as np
    from taccel.compiler.options import CompilerOptions
    from taccel.runtime.calibration import (
        apply_quarot_rotation_from_token_ids, build_calibration_scales,
        build_calibration_scales_from_token_ids, build_calibration_seqs,
        build_calibration_seqs_from_token_ids,
    )
    from taccel.runtime.fp32_reference import NanoGPTFP32Reference
    from taccel.runtime.host_runner import HostRunner
    from taccel.runtime.stage5_ptq import apply_stage5_ptq_scale_policy, resolve_stage5_ptq_preset
    from taccel.runtime.tiny_fixture import build_stage3_tiny_decoder_bundle
    from taccel.runtime.w8a16_simulator_reference import NanoGPTW8A16SimulatorReference

    payload, calibration_ids, eval_ids = prepare_inputs(args, state)
    inputs, vocab = eval_ids[:-1], int(payload["model_args"]["vocab_size"])
    preset = resolve_stage5_ptq_preset(args.ptq_preset)
    state["preset"] = asdict(preset)
    state["transforms"] = {"quarot": {"requested": preset.quarot_enabled, "applied": False},
                           "input_checkpoint_transform_history": "unverified", "fp32_captured_before_ptq": False}
    print("quality gate: original FP32 teacher-forced reference", flush=True)
    original = NanoGPTFP32Reference(payload["state_dict"], payload["model_args"])
    rails["fp32"].extend(original.incremental_logits_trace(inputs))
    del original
    _check_rows(rails["fp32"], len(inputs), vocab, "original FP32")
    state["transforms"]["fp32_captured_before_ptq"] = True
    state["original_fp32_trace"] = {"row_count": len(rails["fp32"]),
                                    "input_token_ids_sha256": canonical_hash(inputs),
                                    "positions": list(range(len(inputs)))}
    _phase_snapshot(args, state, "fp32_before_ptq")
    if preset.quarot_enabled:
        keys, diagnostics = apply_quarot_rotation_from_token_ids(
            payload, calibration_ids, seed=preset.quarot_seed, kind=preset.quarot_kind)
        state["transforms"]["quarot"].update(applied=True, modified_keys=keys, diagnostics=diagnostics)
    if preset.weight_only_int8_calibrate:
        windows = build_calibration_seqs_from_token_ids(
            calibration_ids, n_seqs=args.calibration_n_seqs, seq_len=args.calibration_seq_len)
        calibration_scope = {"source": "explicit calibration text token windows",
                             "n_seqs": args.calibration_n_seqs, "seq_len": args.calibration_seq_len}
    else:
        windows = build_calibration_seqs(
            payload, n_seqs=args.implicit_calibration_n_seqs, seq_len=args.implicit_calibration_seq_len)
        calibration_scope = {"source": "existing bundle implicit payload calibration policy",
                             "n_seqs": args.implicit_calibration_n_seqs,
                             "seq_len": args.implicit_calibration_seq_len,
                             "zero_token_fallback": not bool(payload.get("text") and payload.get("stoi")),
                             "external_calibration_text_used_for_scales": False}
    for window in windows:
        _tokens_valid(window, vocab, "selected calibration window")
    overlap = calibration_overlap_report(calibration_ids, eval_ids, windows)
    state["input_ids"]["calibration_windows"] = windows
    state["input_ids"]["calibration_windows_sha256"] = canonical_hash(windows)
    state["calibration"] = {**calibration_scope, "percentile": args.calibration_percentile,
                            "window_overlap_check": overlap}
    if not overlap["passed"]:
        raise ValueError("calibration/evaluation exact token windows overlap")
    print("quality gate: resolve one explicit shared calibration map", flush=True)
    if preset.weight_only_int8_calibrate:
        scales = build_calibration_scales_from_token_ids(
            payload, calibration_ids, n_seqs=args.calibration_n_seqs,
            seq_len=args.calibration_seq_len, percentile=args.calibration_percentile,
            activation_percentile_overrides=preset.activation_percentile_nodes or None)
    else:
        scales = build_calibration_scales(
            payload, n_seqs=args.implicit_calibration_n_seqs, seq_len=args.implicit_calibration_seq_len,
            percentile=args.calibration_percentile,
            activation_percentile_overrides=preset.activation_percentile_nodes or None)
    scales = apply_stage5_ptq_scale_policy(scales, payload["model_args"], preset)
    scales = {str(key): float(value) for key, value in scales.items()}
    _validate_scales(scales, payload["model_args"])
    state["scales"] = scales
    state["scales_sha256"] = canonical_hash(scales)
    _phase_snapshot(args, state, "shared_scales")
    state["reference"] = {"implementation": "NanoGPTW8A16SimulatorReference",
                           "independent_weight_quantization_from_post_ptq_payload": True,
                           "attention_window": len(inputs), "shared_scales_sha256": canonical_hash(scales)}
    print("quality gate: independent W8A16 teacher-forced reference", flush=True)
    reference = NanoGPTW8A16SimulatorReference(payload, calibration_scales=scales, attention_window=len(inputs))
    for position, token in enumerate(inputs):
        rails["reference"].append(np.asarray(reference.run_decode_step(token, position)).copy())
        _check_rows(rails["reference"][-1:], 1, vocab, f"W8A16 reference position {position}")
    del reference
    _check_rows(rails["reference"], len(inputs), vocab, "W8A16 reference")
    _phase_snapshot(args, state, "reference_complete")
    print("quality gate: compile baseline W8A16 bundle", flush=True)
    compiler_options = CompilerOptions()
    tiny = build_stage3_tiny_decoder_bundle(
        payload, smoke_decode_steps=len(inputs) - 1, calibration_scales=scales,
        ptq_preset=preset, batch=1, prefill_tokens=1, target="rtl-gen2",
        compiler_options=compiler_options)
    bundle = tiny.build.bundle
    state["compiled"] = {"target": "rtl-gen2", "compiler_options": asdict(compiler_options),
                          "batch": 1, "prefill_tokens": 1, "decode_steps": len(inputs) - 1,
                          "attention_window": len(inputs), "logits_dtype": "float16",
                          "shared_scales_sha256": canonical_hash(scales),
                          "program_sha256": {name: hashlib.sha256(getattr(bundle, name)).hexdigest()
                                             for name in ("prefill_instrs", "decode_instrs", "shared_data")}}
    _phase_snapshot(args, state, "compiled")
    rtl_export = _begin_rtl_export(args, state, bundle, inputs)
    runner = HostRunner(bundle, logits_dtype=np.float16, vocab_size=vocab)
    # Keep execution observations outside the static export manifest. That
    # manifest closes after its first few steps, while this list covers the
    # complete evaluation and must not mutate an already pinned export.
    state["compiled_step_execution"] = []
    for position, token in enumerate(inputs):
        row = runner.run_prefill([token]) if position == 0 else runner.run_decode_step(token, position)
        halted = runner.simulator.state.halted is True
        state["compiled_step_execution"].append({"position": position, "token_id": int(token),
                                                  "stream": "prefill" if position == 0 else "decode",
                                                  "halted": halted})
        if not halted:
            raise RuntimeError(f"compiled position {position} did not HALT; partial logits are not accepted")
        rails["compiled"].append(np.asarray(row).copy())
        _check_rows(rails["compiled"][-1:], 1, vocab, f"compiled position {position}")
        _capture_rtl_export_golden(args, rtl_export, runner, position, rails["compiled"][-1])
        if position == 0 or (position + 1) % 16 == 0 or position + 1 == len(inputs):
            print(f"quality gate: compiled position {position + 1}/{len(inputs)}", flush=True)
    if canonical_hash(scales) != state["scales_sha256"]:
        raise ValueError("shared scale map changed during execution")


def _verify_inputs_unchanged(state):
    if "sources" in state:
        current = _source_identity()["sha256"]
        if current != state["sources"]["sha256"]:
            raise ValueError("source identities changed during quality evaluation")
    records = [state[k] for k in ("checkpoint", "checkpoint_metadata") if k in state]
    records += list(state.get("tokenizer", {}).get("files", {}).values())
    for corpus in state.get("corpora", {}).values():
        records.append(corpus)
        if "supplied_provenance" in corpus:
            records.append(corpus["supplied_provenance"])
    for record in records:
        if sha256(record["path"]) != record["sha256"]:
            raise ValueError(f"input identity changed during quality evaluation: {record['path']}")


def _retain_logits(out, rails, token_ids):
    import numpy as np
    arrays = {"token_ids": np.asarray(token_ids, dtype=np.int64)}
    shapes = {}
    for name, rows in rails.items():
        shapes[name] = []
        for i, row in enumerate(rows):
            values = np.asarray(row)
            if values.dtype.kind not in "biufc":
                raise ValueError(f"cannot safely retain nonnumeric {name} logits")
            arrays[f"{name}_{i:04d}"] = values
            shapes[name].append({"shape": list(values.shape), "dtype": str(values.dtype)})
    with (out / "logits.npz").open("xb") as output:
        np.savez(output, **arrays)
        output.flush()
        os.fsync(output.fileno())
    return {"file": "logits.npz", "sha256": sha256(out / "logits.npz"),
            "row_counts": {name: len(rows) for name, rows in rails.items()}, "rows": shapes,
            "scope": "unmodified captured rows, including padding/nonfinite values and partial failed runs"}


def execute(args):
    """Return nonzero for metric failure, exceptions, missing inputs or incomplete rails."""
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    rails = {name: [] for name in ("compiled", "reference", "fp32")}
    state = {"schema_version": 1, "status": "error", "passed": False,
             # Invalid numeric CLI inputs must still be representable in the
             # failure report; validation below uses the original values.
             "arguments": {k: str(v) if isinstance(v, Path) or
                           (isinstance(v, float) and not math.isfinite(v)) else v
                           for k, v in vars(args).items()},
             "scope": {"execution": "compiled golden simulator, not RTL",
                       "canonical_257_token_window": args.max_eval_tokens == 257,
                       "debug_noncanonical_window": args.max_eval_tokens != 257,
                       "official_corpus_wide_quality": False,
                       "legacy_tests_superseded": False, "default_preset_promotion": False},
             "resource_guard": "external supervisor required; CLI does not enforce RSS/time/disk bounds"}
    exit_code = 2
    try:
        if not math.isfinite(args.max_fp32_ppl_increase_percent) or args.max_fp32_ppl_increase_percent < 0:
            raise ValueError("FP32 PPL increase percent must be finite and nonnegative")
        if args.max_eval_tokens < 2:
            raise ValueError("max-eval-tokens must be at least two")
        if not 0 <= args.rtl_export_steps <= min(3, args.max_eval_tokens - 1):
            raise ValueError("rtl-export-steps must be 0..3 and fit the teacher-forced input count")
        if min(args.calibration_n_seqs, args.calibration_seq_len,
               args.implicit_calibration_n_seqs, args.implicit_calibration_seq_len) < 1:
            raise ValueError("calibration counts and lengths must be positive")
        if not math.isfinite(args.calibration_percentile) or not 0 < args.calibration_percentile <= 100:
            raise ValueError("calibration percentile must be in (0,100]")
        run_three_rails(args, state, rails)
        from taccel.runtime.numerical_quality import teacher_forced_quality_report
        state["quality"] = teacher_forced_quality_report(
            token_ids=state["input_ids"]["evaluation"], vocab_size=int(state["model_args"]["vocab_size"]),
            compiled_logits=rails["compiled"], reference_logits=rails["reference"], fp32_logits=rails["fp32"],
            max_fp32_ppl_increase=args.max_fp32_ppl_increase_percent / 100.0)
        state["passed"] = state["quality"]["passed"]
        state["status"] = "passed" if state["passed"] else "failed"
        exit_code = 0 if state["passed"] else 1
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            exit_code = 130
        state["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        try:
            _verify_inputs_unchanged(state)
            state["identities_unchanged"] = True
        except Exception as exc:
            state.update(passed=False, status="error", identities_unchanged=False)
            state["identity_error"] = {"type": type(exc).__name__, "message": str(exc)}
            exit_code = 2
        try:
            state["logits_artifact"] = _retain_logits(out, rails, state.get("input_ids", {}).get("evaluation", []))
            _write_json(out / "input_ids.json", state.get("input_ids", {}))
            _write_json(out / "scales.json", state.get("scales", {}))
            state["artifact_sha256"] = {name: sha256(out / name) for name in ("input_ids.json", "scales.json", "logits.npz")}
        except Exception as exc:
            state.update(passed=False, status="error")
            state["retention_error"] = {"type": type(exc).__name__, "message": str(exc)}
            exit_code = 2
        state["elapsed_s"] = time.monotonic() - started
        state["exit_code"] = exit_code
        _write_json(out / "report.json", state)
    print(json.dumps({"status": state["status"], "passed": state["passed"],
                      "report": str(out / "report.json"), "exit_code": exit_code}, sort_keys=True), flush=True)
    return exit_code


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--tokenizer-dir", type=Path, required=True)
    result.add_argument("--calibration-text", type=Path, required=True)
    result.add_argument("--eval-text", type=Path, required=True)
    result.add_argument("--ptq-preset", choices=PRESETS, required=True)
    result.add_argument("--max-fp32-ppl-increase-percent", type=float, required=True,
                        help="explicit nonnegative allowed PPL increase over original FP32; no default")
    result.add_argument("--max-eval-tokens", type=int, default=257,
                        help="257 is canonical; any explicit different length is a debug/noncanonical run")
    result.add_argument("--rtl-export-steps", type=int, default=0, choices=(0, 1, 2, 3),
                        help="export up to three initial inputs with this run's full attention window; no RTL execution")
    result.add_argument("--calibration-n-seqs", type=int, default=8)
    result.add_argument("--calibration-seq-len", type=int, default=32)
    result.add_argument("--calibration-percentile", type=float, default=99.9)
    result.add_argument("--implicit-calibration-n-seqs", type=int, default=8,
                        help="plain weight_only_int8 preserves the bundle payload-calibration default")
    result.add_argument("--implicit-calibration-seq-len", type=int, default=16)
    result.add_argument("--out", type=Path, required=True, help="new exclusive evidence directory")
    return result


def main(argv=None):
    return execute(parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
