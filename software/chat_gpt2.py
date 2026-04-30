#!/usr/bin/env python3
"""Interactive command-line chat for local GPT-2 FP32 and golden-model paths."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch

from taccel.runtime.calibration import (
    apply_fc2_aware_gelu_scale_search_from_token_ids,
    apply_output_aware_gelu_scale_search_from_token_ids,
    apply_output_aware_mlp_scale_search_from_token_ids,
    build_calibration_scales_from_token_ids,
)
from taccel.runtime.fp32_reference import NanoGPTFP32Reference
from taccel.runtime.gpt2_perplexity import (
    CALIBRATION_N_SEQS_LARGE,
    CALIBRATION_PERCENTILE_DEFAULT,
    CALIBRATION_SEQ_LEN_LARGE,
    GPT2_DEFAULT_PTQ_PRESET,
    load_gpt2_tokenizer,
    tokenize_text_file,
)
from taccel.runtime.host_runner import HostRunner
from taccel.runtime.stage5_ptq import (
    apply_stage5_ptq_scale_policy,
    resolve_stage5_ptq_preset,
    stage5_gelu_from_accum_blocks,
)
from taccel.runtime.tiny_fixture import build_stage3_tiny_decoder_bundle


DEFAULT_CHECKPOINT = Path("software/tests/fixtures/generated/gpt2_converted_nanogpt.pt")
DEFAULT_TOKENIZER_DIR = Path("software/tests/fixtures/generated/hf_gpt2")
DEFAULT_CALIBRATION_TEXT = Path("software/tests/fixtures/generated/wikitext2_stage5_calibration.txt")
CHAT_CALIBRATION_N_SEQS = 2
CHAT_CALIBRATION_SEQ_LEN = 16
CHAT_CALIBRATION_MAX_TOKENS = 512
CHAT_DEFAULT_PTQ_PRESET = "fc2_8_to_11_raw_vadd"


def _active_logits(logits: np.ndarray, vocab_size: int) -> np.ndarray:
    return np.asarray(logits, dtype=np.float32)[: int(vocab_size)]


def _greedy_token(logits: np.ndarray, vocab_size: int) -> int:
    active = _active_logits(logits, vocab_size)
    if active.size == 0:
        raise ValueError("empty logits")
    return int(np.argmax(active))


def _truncate_context(token_ids: Sequence[int], max_context_tokens: int) -> List[int]:
    tokens = [int(tok) for tok in token_ids]
    if max_context_tokens <= 0:
        return tokens
    return tokens[-int(max_context_tokens):]


def _trim_at_stop(text: str, stop_strings: Sequence[str]) -> str:
    limit = len(text)
    for stop in stop_strings:
        if not stop:
            continue
        idx = text.find(stop)
        if idx >= 0:
            limit = min(limit, idx)
    return text[:limit]


def _format_turn(history: str, user_text: str) -> str:
    return f"{history}User: {user_text.strip()}\nGPT-2:"


def _append_turn(history: str, user_text: str, response: str) -> str:
    return f"{history}User: {user_text.strip()}\nGPT-2:{response}\n"


def _build_stage5_scales(
    payload: Dict[str, object],
    calibration_ids: Sequence[int],
    *,
    ptq_preset,
    calibration_n_seqs: int,
    calibration_seq_len: int,
    calibration_percentile: float,
) -> Dict[str, float]:
    scales = build_calibration_scales_from_token_ids(
        payload,
        calibration_ids,
        n_seqs=calibration_n_seqs,
        seq_len=calibration_seq_len,
        percentile=calibration_percentile,
        activation_percentile_overrides=ptq_preset.activation_percentile_nodes or None,
        hessian_gelu_blocks=ptq_preset.hessian_gelu_blocks,
    )
    scales = apply_stage5_ptq_scale_policy(scales, payload["model_args"], ptq_preset)
    if ptq_preset.fc2_aware_gelu_blocks:
        scales, _ = apply_fc2_aware_gelu_scale_search_from_token_ids(
            payload,
            calibration_ids,
            scales,
            blocks=ptq_preset.fc2_aware_gelu_blocks,
            n_seqs=calibration_n_seqs,
            seq_len=calibration_seq_len,
        )
    if ptq_preset.output_aware_gelu_blocks:
        scales, _ = apply_output_aware_gelu_scale_search_from_token_ids(
            payload,
            calibration_ids,
            scales,
            blocks=ptq_preset.output_aware_gelu_blocks,
            ptq_preset=ptq_preset,
            n_seqs=calibration_n_seqs,
            seq_len=calibration_seq_len,
        )
    if ptq_preset.output_aware_mlp_blocks:
        scales, _ = apply_output_aware_mlp_scale_search_from_token_ids(
            payload,
            calibration_ids,
            scales,
            blocks=ptq_preset.output_aware_mlp_blocks,
            ptq_preset=ptq_preset,
            n_seqs=calibration_n_seqs,
            seq_len=calibration_seq_len,
        )
    return scales


def _fp32_generate(
    ref: NanoGPTFP32Reference,
    prompt_ids: Sequence[int],
    *,
    max_new_tokens: int,
    vocab_size: int,
) -> List[int]:
    trace = ref.greedy_decode_trace(prompt_ids, max_new_tokens=max_new_tokens)
    return [int(tok) for tok in trace.generated]


def _golden_generate(
    payload: Dict[str, object],
    prompt_ids: Sequence[int],
    *,
    max_new_tokens: int,
    ptq_preset,
    calibration_scales: Dict[str, float],
    vocab_size: int,
) -> List[int]:
    prompt = [int(tok) for tok in prompt_ids]
    if not prompt:
        raise ValueError("prompt_ids must be non-empty")
    max_decode_position = max(0, len(prompt) + int(max_new_tokens) - 1)
    tiny = build_stage3_tiny_decoder_bundle(
        payload,
        smoke_decode_steps=max_decode_position,
        calibration_scales=calibration_scales,
        ptq_preset=ptq_preset,
    )
    runner = HostRunner(tiny.build.bundle, logits_dtype=np.int8)

    generated = list(prompt)
    logits = runner.run_prefill([prompt[0]])
    for position, token in enumerate(prompt[1:], start=1):
        logits = runner.run_decode_step(int(token), position)

    next_token = _greedy_token(logits, vocab_size)
    for _ in range(max_new_tokens):
        generated.append(next_token)
        position = len(generated) - 1
        logits = runner.run_decode_step(next_token, position)
        next_token = _greedy_token(logits, vocab_size)
    return generated


def _response_text(tokenizer, generated_ids: Sequence[int], prompt_len: int, stop_strings: Sequence[str]) -> str:
    new_ids = [int(tok) for tok in generated_ids[int(prompt_len):]]
    text = tokenizer.decode(new_ids)
    return _trim_at_stop(text, stop_strings)


def _parse_stop_strings(raw: Iterable[str]) -> List[str]:
    return [item.encode("utf-8").decode("unicode_escape") for item in raw]


def _tokenize_calibration_for_chat(
    tokenizer,
    tokenizer_dir: Path,
    text_path: Path,
    *,
    max_tokens: int | None,
) -> List[int]:
    if max_tokens is None:
        return tokenize_text_file(tokenizer_dir, text_path)
    # Interactive chat only needs a stable calibration sample, not the full
    # multi-megabyte corpus.  Read a generous character prefix to avoid the slow
    # "encode all then slice" path used by verification tools.
    char_budget = max(1024, int(max_tokens) * 4)
    text = Path(text_path).read_text(encoding="utf-8")[:char_budget]
    token_ids = [int(tok) for tok in tokenizer.encode(text)]
    return token_ids[: int(max_tokens)]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, nargs="?", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--calibration-text", type=Path, default=DEFAULT_CALIBRATION_TEXT)
    parser.add_argument("--mode", choices=("fp32", "golden", "both"), default="both")
    parser.add_argument(
        "--prompt-style",
        choices=("completion", "chat"),
        default="completion",
        help="completion sends your text directly to base GPT-2; chat uses a User/GPT-2 transcript wrapper",
    )
    parser.add_argument(
        "--ptq-preset",
        default=None,
        help=f"golden PTQ preset; chat default {CHAT_DEFAULT_PTQ_PRESET}",
    )
    parser.add_argument(
        "--quality-preset",
        action="store_true",
        help=f"use the current Stage 5 promoted preset ({GPT2_DEFAULT_PTQ_PRESET}); slower startup",
    )
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--context-tokens", type=int, default=96)
    parser.add_argument("--prompt", default=None, help="Run one turn non-interactively")
    parser.add_argument("--stop", action="append", default=["\\nUser:"], help="Stop string; may be passed more than once")
    parser.add_argument(
        "--quality-calibration",
        action="store_true",
        help=f"use the full Stage 5 gate calibration budget ({CALIBRATION_N_SEQS_LARGE}x{CALIBRATION_SEQ_LEN_LARGE}); slower startup",
    )
    parser.add_argument(
        "--calibration-seq-len",
        type=int,
        default=None,
        help=f"golden calibration sequence length; chat default {CHAT_CALIBRATION_SEQ_LEN}",
    )
    parser.add_argument(
        "--calibration-n-seqs",
        type=int,
        default=None,
        help=f"golden calibration sequence count; chat default {CHAT_CALIBRATION_N_SEQS}",
    )
    parser.add_argument("--calibration-percentile", type=float, default=CALIBRATION_PERCENTILE_DEFAULT)
    args = parser.parse_args(argv)

    if args.max_new_tokens < 0:
        raise ValueError("--max-new-tokens must be non-negative")
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    if not args.tokenizer_dir.exists():
        raise FileNotFoundError(args.tokenizer_dir)

    calibration_n_seqs = (
        CALIBRATION_N_SEQS_LARGE
        if args.quality_calibration and args.calibration_n_seqs is None
        else CHAT_CALIBRATION_N_SEQS
        if args.calibration_n_seqs is None
        else int(args.calibration_n_seqs)
    )
    calibration_seq_len = (
        CALIBRATION_SEQ_LEN_LARGE
        if args.quality_calibration and args.calibration_seq_len is None
        else CHAT_CALIBRATION_SEQ_LEN
        if args.calibration_seq_len is None
        else int(args.calibration_seq_len)
    )

    payload = torch.load(args.checkpoint, map_location="cpu")
    tokenizer = load_gpt2_tokenizer(args.tokenizer_dir)
    vocab_size = int(payload["model_args"]["vocab_size"])
    block_size = int(payload["model_args"].get("block_size", 1024))
    context_limit = min(int(args.context_tokens), block_size)
    stop_strings = _parse_stop_strings(args.stop)

    fp32_ref = None
    if args.mode in {"fp32", "both"}:
        print("Loading FP32 reference...")
        fp32_ref = NanoGPTFP32Reference(payload["state_dict"], payload["model_args"])

    golden_scales = None
    preset_name = (
        GPT2_DEFAULT_PTQ_PRESET
        if args.quality_preset and args.ptq_preset is None
        else CHAT_DEFAULT_PTQ_PRESET
        if args.ptq_preset is None
        else args.ptq_preset
    )
    ptq_preset = resolve_stage5_ptq_preset(preset_name)
    if args.mode in {"golden", "both"}:
        if not args.calibration_text.exists():
            raise FileNotFoundError(args.calibration_text)
        print(
            "Preparing golden model calibration "
            f"({ptq_preset.name}, {calibration_n_seqs}x{calibration_seq_len})..."
        )
        if not args.quality_calibration and args.calibration_n_seqs is None and args.calibration_seq_len is None:
            print("Using fast chat calibration. Pass --quality-calibration for the slower Stage 5 gate settings.")
        if not args.quality_preset and args.ptq_preset is None:
            print("Using fast chat PTQ preset. Pass --quality-preset for the slower promoted Stage 5 preset.")
        calibration_max_tokens = None if args.quality_calibration else CHAT_CALIBRATION_MAX_TOKENS
        calibration_ids = _tokenize_calibration_for_chat(
            tokenizer,
            args.tokenizer_dir,
            args.calibration_text,
            max_tokens=calibration_max_tokens,
        )
        golden_scales = _build_stage5_scales(
            payload,
            calibration_ids,
            ptq_preset=ptq_preset,
            calibration_n_seqs=calibration_n_seqs,
            calibration_seq_len=calibration_seq_len,
            calibration_percentile=args.calibration_percentile,
        )

    print("Ready. Type /quit to exit.")
    if args.prompt_style == "completion":
        print("Tip: completion mode sends your prompt directly to base GPT-2.")
    else:
        print("Tip: chat mode is only a transcript-style prompt; this is not an instruction-tuned model.")

    fp32_history = ""
    golden_history = ""
    prompts = [args.prompt] if args.prompt is not None else None
    while True:
        if prompts is None:
            try:
                user_text = input("\nyou> ")
            except EOFError:
                print()
                break
        else:
            if not prompts:
                break
            user_text = prompts.pop(0)
            print(f"\nyou> {user_text}")

        if user_text is None:
            break
        if user_text.strip() in {"/quit", "/exit"}:
            break
        if not user_text.strip():
            continue

        if args.mode in {"fp32", "both"}:
            assert fp32_ref is not None
            prompt_text = (
                _format_turn(fp32_history, user_text)
                if args.prompt_style == "chat"
                else user_text
            )
            prompt_ids = _truncate_context(tokenizer.encode(prompt_text), context_limit)
            generated = _fp32_generate(
                fp32_ref,
                prompt_ids,
                max_new_tokens=args.max_new_tokens,
                vocab_size=vocab_size,
            )
            response = _response_text(tokenizer, generated, len(prompt_ids), stop_strings)
            if args.prompt_style == "chat":
                fp32_history = _append_turn(fp32_history, user_text, response)
            print(f"fp32> {response.strip()}")

        if args.mode in {"golden", "both"}:
            assert golden_scales is not None
            prompt_text = (
                _format_turn(golden_history, user_text)
                if args.prompt_style == "chat"
                else user_text
            )
            prompt_ids = _truncate_context(tokenizer.encode(prompt_text), context_limit)
            generated = _golden_generate(
                payload,
                prompt_ids,
                max_new_tokens=args.max_new_tokens,
                ptq_preset=ptq_preset,
                calibration_scales=golden_scales,
                vocab_size=vocab_size,
            )
            response = _response_text(tokenizer, generated, len(prompt_ids), stop_strings)
            if args.prompt_style == "chat":
                golden_history = _append_turn(golden_history, user_text, response)
            print(f"golden> {response.strip()}")

        if prompts is not None:
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
