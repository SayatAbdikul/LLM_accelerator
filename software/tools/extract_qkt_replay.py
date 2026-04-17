#!/usr/bin/env python3
"""Extract exact-state QK^T replay payloads from a first-divergence bundle."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.compare_rtl_golden import extract_qkt_replay_payloads


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--first-divergence",
        required=True,
        type=Path,
        help="Path to first_divergence.json from a compare_rtl_golden mismatch run",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Directory where replay payload binaries and metadata should be written",
    )
    parser.add_argument(
        "--node-prefix",
        default=None,
        help="Optional QK^T node prefix, e.g. block0_head0_qkt (defaults from first divergence)",
    )
    parser.add_argument(
        "--strip-row-start",
        type=int,
        default=0,
        help="Row start for the strip to extract (default: 0)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = extract_qkt_replay_payloads(
        first_divergence_path=args.first_divergence.resolve(),
        out_dir=args.out_dir.resolve(),
        node_prefix=args.node_prefix,
        strip_row_start=args.strip_row_start,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
