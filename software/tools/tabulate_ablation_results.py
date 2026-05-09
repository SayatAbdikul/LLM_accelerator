#!/usr/bin/env python3
"""Tabulate Phase 0C / Phase 0D ablation results from the logs/ directory.

Reads JSON files produced by `evaluate_gpt2_perplexity.py --json`, extracts the
key fields, and prints a markdown table sorted by fake_quant_perplexity.

Usage:
    python3 software/tools/tabulate_ablation_results.py software/logs/phase0c/
    python3 software/tools/tabulate_ablation_results.py software/logs/phase0c/ software/logs/phase0d/
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def safe_get(data: dict[str, Any], key: str, default: Any = None) -> Any:
    val = data.get(key, default)
    if isinstance(val, float) and math.isnan(val):
        return default
    return val


def fmt(val: Any, *, fmt_spec: str = ".2f") -> str:
    if val is None:
        return "—"
    if isinstance(val, float):
        if math.isnan(val):
            return "NaN"
        return format(val, fmt_spec)
    return str(val)


def load_results(directory: Path) -> list[dict[str, Any]]:
    results = []
    for json_path in sorted(directory.glob("*.json")):
        try:
            text = json_path.read_text()
            # Skip leading transformers warnings if tee'd in
            brace_idx = text.find("{")
            if brace_idx == -1:
                continue
            data = json.loads(text[brace_idx:])
            data["_log_file"] = json_path.name
            results.append(data)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"WARN: skipping {json_path.name}: {exc}")
    return results


def tabulate(results: list[dict[str, Any]]) -> str:
    if not results:
        return "(no results)"

    # Sort: ones with valid fake_quant_perplexity first, ascending
    def sort_key(r: dict) -> tuple:
        fq = safe_get(r, "fake_quant_perplexity")
        return (fq is None, fq if fq is not None else float("inf"))

    rows = sorted(results, key=sort_key)

    headers = [
        "log",
        "preset",
        "tokens",
        "fp32_ppl",
        "fake_quant_ppl",
        "golden_ppl",
        "rel_delta",
        "ratio_fq/fp32",
    ]

    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        fp32 = safe_get(r, "fp32_perplexity")
        fq = safe_get(r, "fake_quant_perplexity")
        ratio = (fq / fp32) if (fp32 and fq) else None
        row = [
            r.get("_log_file", ""),
            r.get("ptq_preset", ""),
            fmt(r.get("token_count"), fmt_spec="d"),
            fmt(fp32, fmt_spec=".2f"),
            fmt(fq, fmt_spec=".1f"),
            fmt(safe_get(r, "golden_perplexity"), fmt_spec=".1f"),
            fmt(safe_get(r, "relative_delta"), fmt_spec=".6f"),
            fmt(ratio, fmt_spec=".1f") + "×" if ratio else "—",
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dirs", nargs="+", type=Path, help="Directory/directories of JSON logs")
    args = parser.parse_args(argv)

    all_results: list[dict[str, Any]] = []
    for d in args.dirs:
        if not d.is_dir():
            print(f"ERROR: {d} is not a directory")
            return 1
        results = load_results(d)
        print(f"\n## {d}\n")
        print(tabulate(results))
        all_results.extend(results)

    if len(args.dirs) > 1:
        print("\n## All combined\n")
        print(tabulate(all_results))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
