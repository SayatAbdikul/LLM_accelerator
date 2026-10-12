"""Compile-only indexed-liveness A/B microbenchmark; no RTL execution.

Uses a long dependency chain of small matmuls, with fixed weights and no
arithmetic rewrites. Alternates candidate execution order and checks exact
instruction/data hashes in every sample. Results describe this synthetic
compiler workload, not inference speed or complete-model compile speed.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "software"))

import numpy as np

from taccel.compiler.codegen import CodeGenerator
from taccel.compiler.ir import IRGraph, IRNode
from taccel.compiler.model_config import deit_tiny_config
from taccel.compiler.options import CompilerOptions
from taccel.isa.encoding import encode
from tools.benchmark_support import source_identity
from tools.optimization_matrix import host_peak_rss, parse_ints


def graph_chain(nodes):
    if nodes < 1:
        raise ValueError("nodes must be positive")
    graph = IRGraph()
    previous = "external_input"
    for index in range(nodes):
        name = f"projection_{index}"
        graph.add_node(IRNode("matmul", name, [previous, "w"], (16, 16)))
        previous = name
    weights = {"w": (np.eye(16, dtype=np.int8), np.ones(16, dtype=np.float16))}
    return graph, weights


def compile_sample(graph, weights, indexed):
    gc.collect()
    started = time.perf_counter()
    cg = CodeGenerator(weights, {}, {}, model_config=deit_tiny_config(), target="rtl-gen2",
                       options=CompilerOptions(indexed_liveness=indexed),
                       external_inputs={"external_input"})
    instructions, data = cg.generate(graph)
    elapsed = time.perf_counter() - started
    # Exclude encoding/hashing from the measured generate+allocation interval.
    encoded = b"".join(encode(insn) for insn in instructions)
    return {"elapsed_s": elapsed, "instructions": len(instructions), "data_bytes": len(data),
            "instruction_sha256": hashlib.sha256(encoded).hexdigest(),
            "data_sha256": hashlib.sha256(data).hexdigest()}


def benchmark(nodes, repeats):
    if repeats < 1:
        raise ValueError("repeats must be positive")
    graph, weights = graph_chain(nodes)
    compile_sample(graph, weights, False)  # untimed import/allocator warmup
    samples = {"baseline": [], "indexed": []}
    for repeat in range(repeats):
        for indexed in ((False, True) if repeat % 2 == 0 else (True, False)):
            samples["indexed" if indexed else "baseline"].append(compile_sample(graph, weights, indexed))
    expected = samples["baseline"][0]
    identity_fields = ("instructions", "data_bytes", "instruction_sha256", "data_sha256")
    identical = all(all(sample[key] == expected[key] for key in identity_fields)
                    for group in samples.values() for sample in group)
    if not identical:
        raise AssertionError("indexed liveness changed compiled program/data bytes")
    medians = {name: statistics.median(sample["elapsed_s"] for sample in group)
               for name, group in samples.items()}
    return {"nodes": nodes, "repeats": repeats, "byte_identical": identical,
            "identity": {key: expected[key] for key in identity_fields},
            "median_seconds": medians, "median_speedup": medians["baseline"] / medians["indexed"],
            "samples_seconds": {name: [sample["elapsed_s"] for sample in group]
                                for name, group in samples.items()}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", default="500,1000,2000")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("output already exists; choose a new evidence path")
    report = {"schema_version": 1, "source": source_identity(REPO),
              "python": sys.version, "numpy": np.__version__,
              "workload": "synthetic 16x16 matmul chain; compile only; alternating A/B order",
              "timed_interval": "CodeGenerator construction + generate; excludes GC, encoding and hashing",
              "runs": [benchmark(nodes, args.repeats) for nodes in parse_ints(args.nodes)],
              "host_peak_rss": host_peak_rss()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for run in report["runs"]:
        print(f"{run['nodes']} nodes: {run['median_speedup']:.3f}x median speedup; exact program/data bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
