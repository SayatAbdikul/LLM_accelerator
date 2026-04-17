#!/usr/bin/env python3
"""Run N images through the golden model and RTL, compare logits.

Compiles the model once from weights, then iterates over N images:
  patch embed → golden inference → RTL simulation → compare logits.

Prints per-image pass/fail and an aggregate summary table.

Usage:
    python software/tools/batch_compare_rtl_golden.py \\
        --weights pytorch_model.bin \\
        --image-dir software/images/frozen_benchmark/ \\
        --summary-out /tmp/batch_results.json

    python software/tools/batch_compare_rtl_golden.py \\
        --weights pytorch_model.bin \\
        --image path/to/img1.jpg --image path/to/img2.jpg \\
        --max-images 5
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# ─── path setup ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parents[1]
SOFTWARE_DIR = REPO_ROOT / "software"
sys.path.insert(0, str(SOFTWARE_DIR))

DEFAULT_RUNNER = (
    REPO_ROOT / "rtl" / "verilator" / "build" / "run_program" / "Vtaccel_top"
)
DEFAULT_WEIGHTS = SOFTWARE_DIR / "pytorch_model.bin"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ─── helpers ─────────────────────────────────────────────────────────────────

def _scenario_overrides(name: str) -> dict[str, Any]:
    scenarios: dict[str, dict[str, Any]] = {
        "baseline_default": {},
        "experimental_requant_pc": {
            "requant_pc_out_proj": True,
            "requant_pc_out_proj_blocks": {11},
        },
        "experimental_dequant_add": {
            "dequant_add_residual1_blocks": {11},
        },
        "experimental_softmax_attnv": {
            "fused_softmax_attnv_blocks": {11},
        },
        "experimental_fused_out_proj": {
            "fused_softmax_attnv_blocks": {11},
            "fused_softmax_attnv_accum_out_proj": True,
            "requant_pc_out_proj": True,
            "requant_pc_out_proj_blocks": {11},
        },
    }
    if name not in scenarios:
        raise ValueError(f"Unknown compile scenario: {name!r}. "
                         f"Choices: {sorted(scenarios)}")
    return dict(scenarios[name])


def _load_processor(model_name: str):
    from transformers import AutoImageProcessor
    try:
        return AutoImageProcessor.from_pretrained(model_name, local_files_only=True)
    except OSError as exc:
        print(f"[batch_compare] Falling back to local DeiT processor: {exc}",
              file=sys.stderr)
        # Import local fallback from compare_rtl_golden.py
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "compare_rtl_golden",
            str(SCRIPT_DIR / "compare_rtl_golden.py"),
        )
        crg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(crg)  # type: ignore[union-attr]
        return crg._LocalDeiTProcessor()


def _cycle_budget(min_budget: int, golden_cycles: int) -> int:
    return max(int(min_budget), int(golden_cycles) * 6 + 1_000_000)


def _collect_image_paths(args: argparse.Namespace) -> list[Path]:
    """Return an ordered list of image paths from --image-dir and/or --image."""
    paths: list[Path] = []
    if args.image_dir:
        root = Path(args.image_dir)
        for p in sorted(root.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(p)
    for explicit in args.image or []:
        p = Path(explicit)
        if not p.is_file():
            print(f"[batch_compare] WARNING: image not found, skipping: {p}",
                  file=sys.stderr)
            continue
        paths.append(p)
    if not paths:
        raise SystemExit(
            "No images found. Provide --image-dir and/or --image <path>."
        )
    if args.max_images:
        paths = paths[: args.max_images]
    return paths


def _rtl_execution_ok(runner_rc: int, rtl: dict[str, Any]) -> bool:
    violations = rtl.get("violations") or []
    return (
        runner_rc == 0
        and rtl.get("status") == "halted"
        and not bool(rtl.get("fault", False))
        and not bool(rtl.get("timeout", False))
        and not violations
    )


def _compare_logits_with_execution(
    golden_logits: list[int],
    rtl_logits: list[int],
    runner_rc: int,
    rtl: dict[str, Any],
) -> dict[str, Any]:
    golden_top1 = int(np.argmax(golden_logits)) if golden_logits else -1
    rtl_top1 = int(np.argmax(rtl_logits)) if rtl_logits else -1
    raw_logits_exact = golden_logits == rtl_logits
    raw_top1_match = golden_top1 == rtl_top1
    execution_ok = _rtl_execution_ok(runner_rc, rtl)

    return {
        "golden_top1": golden_top1,
        "rtl_top1": rtl_top1,
        "raw_logits_exact_match": raw_logits_exact,
        "raw_top1_match": raw_top1_match,
        "execution_ok": execution_ok,
        "logits_exact_match": execution_ok and raw_logits_exact,
        "top1_match": execution_ok and raw_top1_match,
        "rtl_violations": list(rtl.get("violations") or []),
    }


# ─── per-image runner ─────────────────────────────────────────────────────────

def run_single_image(
    idx: int,
    img_path: Path,
    model,
    processor,
    program,
    embed_scale: float,
    program_path: Path,
    runner_path: Path,
    work_dir: Path,
    num_classes: int,
    min_cycles: int,
    fold_cls_pos_embed: bool,
    keep_work: bool,
) -> dict[str, Any]:
    """
    Patch-embed one image, run golden inference, run RTL, compare.
    Returns a result dict.
    """
    import compare_golden as cg

    img_work = work_dir / f"img_{idx:04d}"
    img_work.mkdir(parents=True, exist_ok=True)

    # ── 1. Patch embed ────────────────────────────────────────────────────────
    try:
        with Image.open(img_path) as img:
            patches_int8, cls_int8, _ = cg.patch_embed_int8(
                model, processor, img.convert("RGB"),
                act_scale=embed_scale,
                fold_cls_pos_embed=fold_cls_pos_embed,
            )
    except Exception as exc:
        return _error_result(img_path, f"patch_embed failed: {exc}")

    # ── 2. Golden inference ───────────────────────────────────────────────────
    try:
        golden_logits, _, golden_cycles, _ = cg.golden_inference(
            program, patches_int8, cls_int8=cls_int8,
            num_classes=num_classes,
        )
        golden_logits_list = np.asarray(golden_logits, dtype=np.int32).tolist()
    except Exception as exc:
        return _error_result(img_path, f"golden_inference failed: {exc}")

    # ── 3. Write patches to disk for runner ──────────────────────────────────
    patches_raw = img_work / "patches.raw"
    np.asarray(patches_int8, dtype=np.int8).reshape(-1).tofile(patches_raw)
    cls_raw: Path | None = None
    if cls_int8 is not None:
        cls_raw = img_work / "cls.raw"
        np.asarray(cls_int8, dtype=np.int8).reshape(-1).tofile(cls_raw)

    # ── 4. RTL simulation ─────────────────────────────────────────────────────
    rtl_summary_path = img_work / "rtl_summary.json"
    budget = _cycle_budget(min_cycles, int(golden_cycles))

    cmd = [
        str(runner_path),
        "--program",     str(program_path),
        "--json-out",    str(rtl_summary_path),
        "--num-classes", str(num_classes),
        "--max-cycles",  str(budget),
        "--patches-raw", str(patches_raw),
        "--patch-rows",  str(patches_int8.shape[0]),
        "--patch-cols",  str(patches_int8.shape[1]),
        "--folded-pos-embed",   # patch_embed_int8 always folds B3
    ]
    if cls_raw is not None:
        cmd.extend(["--cls-raw", str(cls_raw)])

    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=600)
        runner_rc = proc.returncode
        rtl = json.loads(rtl_summary_path.read_text())
    except subprocess.TimeoutExpired:
        return _error_result(img_path, "RTL runner timed out (>600s)")
    except Exception as exc:
        return _error_result(img_path, f"RTL runner error: {exc}")

    # ── 5. Compare ────────────────────────────────────────────────────────────
    rtl_logits = (rtl.get("logits") or [])[:num_classes]
    compare_fields = _compare_logits_with_execution(
        golden_logits=golden_logits_list,
        rtl_logits=rtl_logits,
        runner_rc=runner_rc,
        rtl=rtl,
    )

    result = {
        "image":              str(img_path),
        **compare_fields,
        "rtl_status":         rtl.get("status", "unknown"),
        "rtl_fault":          bool(rtl.get("fault", False)),
        "rtl_timeout":        bool(rtl.get("timeout", False)),
        "rtl_cycles":         int(rtl.get("cycles", 0)),
        "golden_cycles":      int(golden_cycles),
        "runner_exit_code":   runner_rc,
        "error":              None,
    }

    if not keep_work:
        import shutil
        shutil.rmtree(img_work, ignore_errors=True)

    return result


def _error_result(img_path: Path, msg: str) -> dict[str, Any]:
    print(f"  ERROR: {msg}", file=sys.stderr)
    return {
        "image":              str(img_path),
        "golden_top1":        -1,
        "rtl_top1":           -1,
        "raw_top1_match":     False,
        "raw_logits_exact_match": False,
        "execution_ok":       False,
        "top1_match":         False,
        "logits_exact_match": False,
        "rtl_status":         "error",
        "rtl_fault":          False,
        "rtl_timeout":        False,
        "rtl_violations":     [],
        "rtl_cycles":         0,
        "golden_cycles":      0,
        "runner_exit_code":   -1,
        "error":              msg,
    }


# ─── output helpers ───────────────────────────────────────────────────────────

def _print_progress(idx: int, total: int, result: dict[str, Any]) -> None:
    img_name = Path(result["image"]).name
    status   = "PASS" if result["logits_exact_match"] else (
               "TOP1" if result["top1_match"]         else "FAIL")
    rtl_m    = result["rtl_cycles"] / 1e6
    note = ""
    if result["error"]:
        note = f"  [{result['error']}]"
    elif result["rtl_fault"]:
        note = "  [RTL FAULT]"
    elif result["rtl_timeout"]:
        note = "  [RTL TIMEOUT]"
    elif result["runner_exit_code"] != 0:
        note = f"  [RTL EXIT {result['runner_exit_code']}]"
    elif result["rtl_status"] != "halted":
        note = f"  [RTL STATUS {result['rtl_status']}]"
    elif result.get("rtl_violations"):
        note = f"  [RTL VIOLATIONS: {','.join(result['rtl_violations'])}]"
    elif not result["logits_exact_match"] and not result["top1_match"]:
        note = f"  [golden={result['golden_top1']} rtl={result['rtl_top1']}]"
    elif not result["logits_exact_match"]:
        note = f"  [top1 OK but logits differ]"

    print(f"[{idx+1:3d}/{total}] {status:<4}  {img_name:<45}  "
          f"cycles={rtl_m:.1f}M{note}")


def _print_summary(results: list[dict[str, Any]]) -> None:
    n = len(results)
    exact    = sum(1 for r in results if r["logits_exact_match"])
    top1     = sum(1 for r in results if r["top1_match"])
    clean    = sum(1 for r in results if r.get("execution_ok"))
    faults   = sum(1 for r in results if r["rtl_fault"])
    timeouts = sum(1 for r in results if r["rtl_timeout"])
    errors   = sum(1 for r in results if r["error"])

    print()
    print("=" * 45)
    print("          Batch Summary")
    print("=" * 45)
    print(f"  Images tested:        {n}")
    print(f"  Clean RTL executions: {clean}/{n}  ({100*clean/n:.1f}%)")
    print(f"  Logits exact match:   {exact}/{n}  ({100*exact/n:.1f}%)")
    print(f"  Top-1 agreement:      {top1}/{n}  ({100*top1/n:.1f}%)")
    if faults:
        print(f"  RTL faults:           {faults}/{n}")
    if timeouts:
        print(f"  RTL timeouts:         {timeouts}/{n}")
    if errors:
        print(f"  Script errors:        {errors}/{n}")
    print("=" * 45)


# ─── main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch RTL vs golden model comparison over N images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--weights",        required=True,
                   help="Path to pytorch_model.bin")
    p.add_argument("--runner",         default=str(DEFAULT_RUNNER),
                   help="Path to RTL runner binary (Vtaccel_top)")
    p.add_argument("--image-dir",      metavar="DIR",
                   help="Directory of images to test (sorted order)")
    p.add_argument("--image",          action="append", metavar="PATH",
                   help="Explicit image path (repeatable)")
    p.add_argument("--max-images",     type=int, default=0,
                   help="Maximum number of images to process (0 = all)")
    p.add_argument("--num-classes",    type=int, default=1000,
                   help="Number of output logit classes")
    p.add_argument("--min-cycles",     type=int, default=500_000,
                   help="Minimum RTL cycle budget (auto-scales with golden count)")
    p.add_argument("--scenario",       default="baseline_default",
                   choices=["baseline_default", "experimental_requant_pc",
                            "experimental_dequant_add", "experimental_softmax_attnv",
                            "experimental_fused_out_proj"],
                   help="Compilation scenario")
    p.add_argument("--fold-cls-pos-embed", action="store_true",
                   help="Fold CLS + position embeddings on host before quantisation")
    p.add_argument("--calibration-image", action="append", metavar="PATH",
                   help="Extra calibration image(s); defaults to first runtime image")
    p.add_argument("--summary-out",    metavar="PATH",
                   help="Write JSON summary to this file")
    p.add_argument("--work-dir",       metavar="DIR",
                   help="Root working directory (default: auto tempdir)")
    p.add_argument("--keep-work",      action="store_true",
                   help="Keep per-image working directories after run")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import compare_golden as cg

    runner_path = Path(args.runner).resolve()
    if not runner_path.exists():
        raise SystemExit(
            f"RTL runner not found: {runner_path}\n"
            "Build it with:  make -C rtl/verilator run_program"
        )

    # ── Discover images ───────────────────────────────────────────────────────
    images = _collect_image_paths(args)
    print(f"[batch_compare] Testing {len(images)} image(s).")

    # ── Working directory ─────────────────────────────────────────────────────
    work_dir = Path(args.work_dir) if args.work_dir else \
        Path(tempfile.mkdtemp(prefix="batch_compare_"))
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"[batch_compare] Work dir: {work_dir}")

    # ── Load model ────────────────────────────────────────────────────────────
    print("[batch_compare] Loading model weights…")
    cg.WEIGHTS_PATH = args.weights
    model, state_dict = cg.load_model()
    processor = _load_processor(cg.MODEL_NAME)

    # ── Choose calibration images ─────────────────────────────────────────────
    cal_paths = args.calibration_image or [str(images[0])]
    sample_images: list = []
    for cp in cal_paths:
        with Image.open(cp) as img:
            sample_images.append(img.convert("RGB"))

    # ── Compile once ──────────────────────────────────────────────────────────
    print("[batch_compare] Compiling model (once)…")
    compile_kwargs = _scenario_overrides(args.scenario)
    program, cal_scales = cg.compile_model(
        model, state_dict, sample_images, processor, **compile_kwargs
    )
    embed_scale: float = cal_scales.get("pos_embed_add", 14.0 / 127.0)

    program_path = work_dir / "program.bin"
    program_path.write_bytes(program.to_bytes())
    print(f"[batch_compare] Program binary: {program_path} "
          f"({program_path.stat().st_size // 1024} KB)")

    # ── Per-image loop ────────────────────────────────────────────────────────
    print()
    results: list[dict[str, Any]] = []
    for idx, img_path in enumerate(images):
        result = run_single_image(
            idx=idx,
            img_path=img_path,
            model=model,
            processor=processor,
            program=program,
            embed_scale=embed_scale,
            program_path=program_path,
            runner_path=runner_path,
            work_dir=work_dir,
            num_classes=args.num_classes,
            min_cycles=args.min_cycles,
            fold_cls_pos_embed=args.fold_cls_pos_embed,
            keep_work=args.keep_work,
        )
        results.append(result)
        _print_progress(idx, len(images), result)

    # ── Summary ───────────────────────────────────────────────────────────────
    _print_summary(results)

    # ── JSON output ───────────────────────────────────────────────────────────
    if args.summary_out:
        n = len(results)
        summary = {
            "total_images":       n,
            "execution_ok":       sum(1 for r in results if r.get("execution_ok")),
            "logits_exact_match": sum(1 for r in results if r["logits_exact_match"]),
            "top1_agreement":     sum(1 for r in results if r["top1_match"]),
            "rtl_faults":         sum(1 for r in results if r["rtl_fault"]),
            "rtl_timeouts":       sum(1 for r in results if r["rtl_timeout"]),
            "script_errors":      sum(1 for r in results if r["error"]),
            "scenario":           args.scenario,
            "results":            results,
        }
        Path(args.summary_out).write_text(json.dumps(summary, indent=2))
        print(f"\n[batch_compare] Summary written to: {args.summary_out}")


if __name__ == "__main__":
    main()
