"""Shared provenance and numerical checks for accelerator cycle experiments."""
from __future__ import annotations

import hashlib
from dataclasses import asdict
import importlib.metadata
import json
from numbers import Integral
import platform
import subprocess
from pathlib import Path

import numpy as np


def compiler_options_from_names(names: str):
    from taccel.compiler.options import CompilerOptions
    allowed = set(asdict(CompilerOptions()))
    requested = {name.strip() for name in names.split(",") if name.strip()}
    unknown = requested - allowed
    if unknown:
        raise ValueError(f"unknown compiler optimizations: {sorted(unknown)}; choose from {sorted(allowed)}")
    return CompilerOptions(**{name: True for name in requested})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def runtime_toolchain_identity() -> dict:
    """Installed versions at capture time, not an attestation of runner builds."""
    versions = {}
    for package in ("numpy", "torch", "transformers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    command = ["verilator", "--version"]
    verilator = {"command": command, "available": False, "version": None,
                 "scope": "installed_tool_at_manifest_time_not_runner_build_attestation"}
    try:
        version = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=10).strip()
        if version:
            verilator.update(available=True, version=version)
        else:
            verilator["unavailable_reason"] = "empty version output"
    except (OSError, subprocess.SubprocessError) as exc:
        verilator["unavailable_reason"] = f"{type(exc).__name__}: {exc}"
    return {"python": platform.python_version(), "packages": versions,
            "package_version_source": "installed importlib.metadata; null means not installed",
            "toolchain": {"verilator": verilator}}


def bundle_ptq_transform_manifest(preset_name: str | None) -> dict:
    """Describe direct bundle construction, not the full PTQ evaluation flow.

    Bundle builders consume the supplied weights and calibration. They do not
    invoke the high-level QuaRot preprocessing step, even when the preset's
    name/flags request it. A caller-supplied checkpoint may already be rotated;
    absent separate provenance, its prior transformation history is unknown.
    """
    from taccel.runtime.stage5_ptq import resolve_stage5_ptq_preset
    requested = (bool(resolve_stage5_ptq_preset(preset_name).quarot_enabled)
                 if preset_name is not None else None)
    return {"requested_preset": preset_name,
            "quarot": {"requested_by_preset": requested,
                       "applied_in_bundle_build": False,
                       "input_checkpoint_transform_history": "unverified"},
            "scope": "direct_bundle_build_not_full_ptq_preprocessing"}


def source_identity(repo: Path) -> dict:
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], timeout=30)

    try:
        patch = git("diff", "HEAD", "--binary")
        # Include untracked source files as well as tracked edits in the identity.
        untracked = git("ls-files", "--others", "--exclude-standard", "-z").split(b"\0")
        hashes = {name.decode(): sha256_file(repo / name.decode())
                  for name in untracked if name and (repo / name.decode()).is_file()}
        return {"commit": git("rev-parse", "HEAD").decode().strip(),
                "patch_sha256": hashlib.sha256(patch).hexdigest(),
                "untracked_sha256": hashes,
                "dirty": bool(patch or hashes)}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None, "unavailable": "not a Git checkout"}


def benchmark_manifest(repo: Path, fixture: Path, rtl: Path, *, batch: int,
                       positions, beat_interval: int, fmax_mhz, compiler_options,
                       ptq_preset: str | None = None) -> dict:
    fixture_info = {"path": str(fixture), "sha256": sha256_file(fixture)}
    metadata_path = fixture.with_suffix(fixture.suffix + ".json")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("checkpoint_sha256") != fixture_info["sha256"]:
            raise ValueError("benchmark checkpoint does not match its paired metadata SHA-256")
        fixture_info["metadata_sha256"] = sha256_file(metadata_path)
        fixture_info["provenance"] = metadata.get("provenance")
    else:
        fixture_info["metadata_sha256"] = None
    return {"schema_version": 2, "source": source_identity(repo),
            **runtime_toolchain_identity(),
            "fixture": fixture_info,
            "runner": {"path": str(rtl.resolve()), "sha256": sha256_file(rtl)},
            "batch": batch, "positions": list(positions),
            "memory": {"beat_bytes": 16, "core_cycles_per_read_beat": beat_interval,
                       "write_pacing": "unthrottled_by_beat_interval",
                       "model_kind": "independent_axi_channels_not_shared_total_bandwidth"},
            "clock_mhz": fmax_mhz, "clock_kind": "user_assumed" if fmax_mhz else "unmeasured",
            "compiler_options": compiler_options,
            "ptq_transforms": bundle_ptq_transform_manifest(ptq_preset),
            "tokenizer": {"used": False, "reason": "fixed token-id shape workload"},
            "workload_kind": "shape_only_zero_kv", "accuracy_validated": False}


def logical_decoder_work(*, batch: int, layers: int, d_model: int, mlp_dim: int,
                         vocab_size: int, position: int, summary: dict | None = None) -> dict:
    """Estimate canonical dense-decoder MACs, not executed ISA work.

    One MAC means one multiplication plus accumulation. The estimate counts
    Q/K/V/output projections, both MLP projections, QK/AV over position+1
    keys, and the vocabulary projection. It excludes PTQ transformations,
    SFU/helper operations and padding. It is therefore not measured array
    utilization, a numerical-quality claim, or a frequency/energy estimate.
    """
    geometry = dict(batch=batch, layers=layers, d_model=d_model, mlp_dim=mlp_dim,
                    vocab_size=vocab_size, position=position)
    for name, value in geometry.items():
        minimum = 0 if name == "position" else 1
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            qualifier = "non-negative" if name == "position" else "positive"
            raise ValueError(f"{name} must be a {qualifier} integer")
        geometry[name] = int(value)
    batch, layers, d_model, mlp_dim, vocab_size, position = geometry.values()
    components = {
        "qkv_and_output_projections": batch * layers * 4 * d_model ** 2,
        "mlp_projections": batch * layers * 2 * d_model * mlp_dim,
        "attention_qk_and_av": batch * layers * 2 * d_model * (position + 1),
        "vocabulary_projection": batch * d_model * vocab_size,
    }
    useful_macs = sum(components.values())
    result = {
        "kind": "canonical_decoder_algorithm_estimate",
        "formula": "B * (layers * (4*D^2 + 2*D*MLP + 2*D*(position+1)) + D*vocab)",
        "unit": "MAC (one multiply-accumulate)",
        "measured_isa_op_count": False,
        "excludes": ["PTQ_transforms", "SFU_operations", "helper_operations", "padding"],
        "geometry": geometry,
        "estimated_useful_macs": useful_macs,
        "component_macs": components,
        "capacity_ratios": {},
        "ratio_interpretation": "algorithmic useful MACs / 16x16 capacity; not measured array utilization",
    }
    # Old runner summaries may not contain compute-step instrumentation. Never
    # substitute systolic busy cycles, nor invent a denominator for old data.
    if summary is not None:
        counters = {
            "enabled_compute_steps": (summary.get("systolic_phases") or {}).get("compute_steps"),
            "total_cycles": summary.get("cycles"),
        }
        for name, value in counters.items():
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer counter")
            # A zero counter is present but cannot define a capacity ratio.
            if value:
                result["capacity_ratios"][name] = {
                    "counter": int(value), "array_macs_per_step": 16 ** 2,
                    "value": useful_macs / (16 ** 2 * int(value)),
                }
    return result


def validate_runner_summary(summary: dict, *, require_hardware: bool = True) -> None:
    if summary.get("status") != "halted" or summary.get("fault"):
        raise ValueError("runner did not halt successfully")
    if summary.get("timeout") or summary.get("parse_error") or summary.get("violations"):
        raise ValueError(f"runner reported invalid execution or counter violations: {summary.get('violations', [])}")
    if summary.get("forbidden_overlap_violation") or summary.get("port_a", {}).get("sys_lost", 0):
        raise ValueError("runner reported forbidden overlap or lost SRAM accesses")
    if int(summary.get("cycles", 0)) <= 0:
        raise ValueError("runner returned no executed cycles")
    phases = summary.get("systolic_phases")
    if phases is not None:
        names = ("load_cycles", "data_cycles", "k_boundary_cycles", "flush_cycles",
                 "drain_cycles", "other_cycles", "compute_steps")
        if any(not isinstance(phases.get(k), int) or phases[k] < 0 for k in names):
            raise ValueError("incomplete or negative systolic phase counters")
        if sum(phases[k] for k in names[:-1]) != summary.get("busy_cycles", {}).get("systolic"):
            raise ValueError("systolic phase counters do not sum to busy cycles")
        if phases["compute_steps"] != phases["data_cycles"] + phases["flush_cycles"]:
            raise ValueError("systolic compute-step counter mismatch")
    dma = summary.get("dma", {})
    if "read_beat_count" in dma:
        for suffix in ("beat_count", "burst_count"):
            values = [dma.get(prefix + suffix) for prefix in ("", "read_", "write_")]
            if any(not isinstance(value, int) or value < 0 for value in values) or values[0] != sum(values[1:]):
                raise ValueError(f"DMA direction counter mismatch: {suffix}")
        for count, byte_key in (("beat_count", "accepted_bytes"),
                                ("read_beat_count", "read_bytes"),
                                ("write_beat_count", "write_bytes")):
            if dma.get(byte_key) != dma[count] * 16:
                raise ValueError(f"DMA byte counter mismatch: {byte_key}")
    config = summary.get("configuration", {})
    if require_hardware and (config.get("sfu_synth_mode") != 1 or
                             config.get("helper_synth_mode") != 1):
        raise ValueError("hardware measurements require reported SFU_SYNTH_MODE=1 and HELPER_SYNTH_MODE=1")


def decoder_logits_metrics(raw: bytes, *, rows: int, vocab_size: int,
                           row_bytes: int) -> dict:
    """Read actual DRAM FP16 logits, excluding padded vocabulary columns."""
    if len(raw) != rows * row_bytes or row_bytes % 2 or vocab_size * 2 > row_bytes:
        raise ValueError("decoder logits dump does not match its declared layout")
    logits = np.frombuffer(raw, dtype="<f2").reshape(rows, row_bytes // 2)[:, :vocab_size]
    finite = np.isfinite(logits)
    return {"dtype": "float16", "rows": rows, "vocab_size": vocab_size,
            "sha256": hashlib.sha256(logits.tobytes()).hexdigest(),
            "dump_sha256": hashlib.sha256(raw).hexdigest(),
            "finite_count": int(finite.sum()), "element_count": int(logits.size),
            "all_finite": bool(finite.all()),
            "argmax": [int(np.argmax(row)) if np.isfinite(row).all() else None for row in logits]}


def check_runner_freshness(repo: Path, rtl: Path) -> None:
    """Reject stale binaries, including custom variants and changed filelists."""
    paths = [repo / "rtl/verilator/Makefile", repo / "rtl/common/filelists/core.f"]
    paths.extend((repo / "rtl/common/src").rglob("*.sv"))
    paths.extend((repo / "rtl/common/src").rglob("*.svh"))
    # The program runner does not compile focused test_*.cpp translation units.
    # Match its Make prerequisites so unrelated bench edits cannot make a
    # binary appear stale when `make run_program_synth` would not rebuild it.
    paths.append(repo / "rtl/verilator/run_program.cpp")
    paths.extend((repo / "rtl/verilator/include").glob("*.h"))
    newer = [p for p in paths if p.exists() and p.stat().st_mtime > rtl.stat().st_mtime]
    if newer:
        raise ValueError(f"stale runner {rtl}: rebuild after {newer[0].relative_to(repo)}")
