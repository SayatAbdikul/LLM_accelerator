"""Read-only diagnostic helper for the Stage 3 tiny nanoGPT e2e gate."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import List, Sequence

import numpy as np

from taccel.runtime.calibration import build_calibration_scales
from taccel.runtime.fake_quant import cosine_similarity
from taccel.runtime.fake_quant_reference import NanoGPTFQReference
from taccel.runtime.host_runner import HostRunner
from taccel.runtime.tiny_fixture import build_stage3_tiny_decoder_bundle, run_tiny_decode_trace


TOOL_PATH = Path(__file__).resolve().parent / "train_tiny_fixture.py"


def _load_fixture_tool():
    spec = importlib.util.spec_from_file_location("train_tiny_fixture", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _greedy_token(logits: np.ndarray, vocab_size: int) -> int:
    return int(np.argmax(np.asarray(logits)[:vocab_size]))


def _topk_overlap(a: np.ndarray, b: np.ndarray, *, vocab_size: int, k: int = 5) -> int:
    lhs = set(np.argsort(np.asarray(a)[:vocab_size])[-k:].tolist())
    rhs = set(np.argsort(np.asarray(b)[:vocab_size])[-k:].tolist())
    return len(lhs.intersection(rhs))


def _reference_trace(ref: NanoGPTFQReference, prompt_ids: Sequence[int], *,
                     max_new_tokens: int, vocab_size: int, mode: str) -> tuple[List[int], List[np.ndarray]]:
    generated = [int(tok) for tok in prompt_ids]
    logits_trace: List[np.ndarray] = []

    def forward(tokens: Sequence[int]) -> np.ndarray:
        if mode == "full":
            return ref.forward(tokens)
        if mode == "incremental":
            return ref.forward_incremental(tokens)
        raise ValueError(f"unknown reference mode: {mode}")

    logits = forward(generated)
    logits_trace.append(logits)
    next_token = _greedy_token(logits, vocab_size)
    for _ in range(max_new_tokens):
        generated.append(next_token)
        logits = forward(generated)
        logits_trace.append(logits)
        next_token = _greedy_token(logits, vocab_size)
    return generated, logits_trace


def _stats(lhs: Sequence[np.ndarray], rhs: Sequence[np.ndarray], *, vocab_size: int) -> dict:
    cosines = [
        float(cosine_similarity(a[:vocab_size].astype(np.float32), b[:vocab_size].astype(np.float32)))
        for a, b in zip(lhs, rhs)
    ]
    top5 = [
        int(_topk_overlap(a, b, vocab_size=vocab_size, k=5))
        for a, b in zip(lhs, rhs)
    ]
    return {
        "steps": len(cosines),
        "min_cosine": min(cosines) if cosines else 1.0,
        "argmin_step": int(np.argmin(cosines)) if cosines else 0,
        "mean_cosine": float(np.mean(cosines)) if cosines else 1.0,
        "max_cosine": max(cosines) if cosines else 1.0,
        "min_top5_overlap": min(top5) if top5 else 5,
        "first10_cosine": cosines[:10],
        "first10_top5": top5[:10],
    }


def _trace_node(payload: dict, prompt_ids: Sequence[int], *, max_new_tokens: int,
                node_name: str, step: int) -> dict:
    scales = build_calibration_scales(payload)
    tiny = build_stage3_tiny_decoder_bundle(
        payload,
        smoke_decode_steps=max_new_tokens,
        calibration_scales=scales,
    )
    bundle = tiny.build.bundle
    runner = HostRunner(bundle, logits_dtype=np.int8)
    generated = [int(tok) for tok in prompt_ids]

    if step == 0:
        runner.simulator.trace_manifest = {
            bundle.prefill_pc + pc: events
            for pc, events in tiny.build.prefill_codegen.trace_manifest.items()
        }
        runner.simulator.enable_trace([node_name])
        runner.run_prefill(generated)
    else:
        logits = runner.run_prefill(generated)
        next_token = _greedy_token(logits, tiny.config.vocab_size)
        for decode_step in range(1, step + 1):
            generated.append(next_token)
            runner.simulator.trace_manifest = {
                bundle.decode_pc + pc: events
                for pc, events in tiny.build.decode_codegen.trace_manifest.items()
            }
            if decode_step == step:
                runner.simulator.enable_trace([node_name])
            logits = runner.run_decode_step(next_token, len(generated) - 1)
            next_token = _greedy_token(logits, tiny.config.vocab_size)

    trace = runner.simulator.get_trace_payload()
    raw = trace["raw_tensors"].get(node_name)
    tensor = trace["tensors"].get(node_name)
    if raw is None and tensor is None:
        return {"node": node_name, "captured": False}
    out = {"node": node_name, "captured": True}
    if raw is not None:
        out["raw_shape"] = list(raw.shape)
        out["raw_dtype"] = str(raw.dtype)
        out["raw_first16"] = raw.ravel()[:16].astype(int).tolist()
    if tensor is not None:
        out["tensor_shape"] = list(tensor.shape)
        out["tensor_first16"] = tensor.ravel()[:16].astype(float).tolist()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--prompt-id", type=int, action="append", default=None)
    parser.add_argument("--node", default=None, help="Optional trace node to dump")
    parser.add_argument("--step", type=int, default=1, help="Trace step: 0=prefill, >0=decode step")
    args = parser.parse_args()

    tool = _load_fixture_tool()
    fixture = args.fixture or tool.DEFAULT_FIXTURE
    if not fixture.exists():
        raise SystemExit(
            "tiny fixture is missing; run "
            "PYTHONPATH=software python software/tools/train_tiny_fixture.py"
        )

    import torch

    payload = torch.load(fixture, map_location="cpu")
    prompt_ids = args.prompt_id if args.prompt_id is not None else [0]
    scales = build_calibration_scales(payload)
    tiny = build_stage3_tiny_decoder_bundle(
        payload,
        smoke_decode_steps=args.max_new_tokens,
        calibration_scales=scales,
    )
    actual = run_tiny_decode_trace(tiny, prompt_ids, max_new_tokens=args.max_new_tokens)
    ref = NanoGPTFQReference(payload["state_dict"], payload["model_args"], scales)
    full_generated, full_logits = _reference_trace(
        ref,
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        vocab_size=tiny.config.vocab_size,
        mode="full",
    )
    inc_generated, inc_logits = _reference_trace(
        ref,
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        vocab_size=tiny.config.vocab_size,
        mode="incremental",
    )

    summary = {
        "fixture": str(fixture),
        "prompt_ids": prompt_ids,
        "max_new_tokens": args.max_new_tokens,
        "actual_generated": actual.generated,
        "full_reference_generated": full_generated,
        "incremental_reference_generated": inc_generated,
        "actual_vs_incremental": _stats(actual.logits, inc_logits, vocab_size=tiny.config.vocab_size),
        "full_vs_incremental": _stats(full_logits, inc_logits, vocab_size=tiny.config.vocab_size),
    }
    if args.node:
        summary["trace"] = _trace_node(
            payload,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            node_name=args.node,
            step=args.step,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
