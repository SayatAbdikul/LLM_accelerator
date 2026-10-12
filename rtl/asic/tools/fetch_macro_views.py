#!/usr/bin/env python3
"""Fetch/verify only the explicitly pinned experimental SRAM collateral."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ASIC = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    manifest = json.loads((ASIC / "macro_views.json").read_text())
    cache = ASIC / "cache/sky130_sram"
    cache.mkdir(parents=True, exist_ok=True)
    base = ("https://raw.githubusercontent.com/VLSIDA/sky130_sram_macros/"
            f"{manifest['revision']}/{manifest['macro']}/")
    for name, expected in manifest["views"].items():
        path = cache / name
        if not path.exists() and not args.verify_only:
            # Validate in memory before creating the cache entry. Never overwrite
            # existing mismatched files or silently switch upstream revisions.
            with urllib.request.urlopen(base + name, timeout=120) as response:
                raw = response.read()
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError(f"upstream view hash mismatch: {name}")
            path.write_bytes(raw)
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"missing or mismatched pinned SRAM view: {path}")
        print(f"verified {name}")


if __name__ == "__main__":
    main()
