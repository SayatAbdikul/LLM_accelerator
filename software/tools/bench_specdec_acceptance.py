#!/usr/bin/env python3
"""Measure `t` — the tokens a spec-dec verify pass actually confirms — on real text.

`bench_specdec_cycles.py` measures `r` (the pass/step cost ratio, a hardware
property: 1.4531 on 124M at P=16). This measures the OTHER term:

    speedup = t / r          break-even at t = r

`t` is a property of the DRAFT and the TEXT, not of the chip, and it is where a
spec-dec claim usually turns into fiction. So it gets measured, on real text,
with the real model's numerics -- not assumed.

Why this needs no RTL
---------------------
Acceptance depends only on the model's GREEDY continuation. A candidate is
accepted exactly where the draft's guess equals the model's own greedy next
token, and the correction token IS the model's greedy next token -- so every
token spec-dec emits is a token plain greedy decoding would have emitted. Given
the greedy sequence, the driver is a deterministic walk over it, reproducing
`speculative_generate` exactly. One greedy decode in torch therefore yields the
exact `t` the chip would see, in seconds instead of hours of RTL simulation.

Which model
-----------
`t` is only meaningful on the model the chip actually runs. The default is
`--model fake-quant`: `NanoGPTFQReference` under the frozen `weight_only_int8_quarot`
preset with calibration scales -- i.e. **W8A16**: INT8 weights, FP16 activations,
static scales, QuaRot. That is the accelerator's numerics.

`--model w8a32` (INT8 weights but FP32 activations, no QuaRot, no scales) and
`--model fp32` are cross-checks, NOT the chip. Quoting `t` from either as "the
chip's numerics" would be wrong: a different model produces a different greedy
sequence, hence a different acceptance rate.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "software"))

from taccel.runtime.speculative import PromptLookupDraft  # noqa: E402

FIXTURES = REPO_ROOT / "software" / "tests" / "fixtures" / "generated"
DEFAULT_CKPT = FIXTURES / "gpt2_converted_nanogpt.pt"
DEFAULT_TOKENIZER = FIXTURES / "hf_gpt2"
DEFAULT_TEXT = FIXTURES / "wikitext2_stage5_eval.txt"

# Measured on GPT-2 124M, P=16, base_pos=496 (bench_specdec_cycles.py).
DEFAULT_R = 1.4531
DEFAULT_P = 16


FREEZE_PTQ_PRESET = "weight_only_int8_quarot"


def _build_reference(payload, model: str):
    """The model whose GREEDY sequence defines acceptance.

    `fake-quant` is the accelerator: W8A16 (INT8 weights, FP16 activations, static
    calibration scales, QuaRot) under the frozen preset. The others are cross-checks.
    """
    from taccel.runtime.fp32_reference import (
        NanoGPTFP32Reference,
        build_weight_only_int8_reference,
    )

    if model == "fp32":
        return NanoGPTFP32Reference(payload["state_dict"], payload["model_args"])
    if model == "w8a32":
        return build_weight_only_int8_reference(payload, weight_mode="per_channel")

    from taccel.runtime.calibration import build_calibration_scales
    from taccel.runtime.fake_quant_reference import NanoGPTFQReference
    from taccel.runtime.stage5_ptq import (
        resolve_stage5_ptq_preset,
        stage5_gelu_from_accum_blocks,
        stage5_raw_residual1_blocks,
        stage5_raw_residual2_blocks,
        stage5_requant_pc_weight_names,
    )

    preset = resolve_stage5_ptq_preset(FREEZE_PTQ_PRESET)
    args = payload["model_args"]
    return NanoGPTFQReference(
        payload["state_dict"], args, build_calibration_scales(payload),
        requant_pc_weight_names=stage5_requant_pc_weight_names(args, preset),
        raw_residual1_blocks=stage5_raw_residual1_blocks(preset),
        raw_residual2_blocks=stage5_raw_residual2_blocks(preset),
        gelu_from_accum_blocks=stage5_gelu_from_accum_blocks(preset),
    )


def greedy_continuation(payload, prompt, n_new, *, model="fake-quant"):
    """Greedy-decode `n_new` tokens. Returns prompt + generated."""
    ref = _build_reference(payload, model)
    ctx = list(prompt)
    for _ in range(n_new):
        logits = ref.incremental_logits_trace(ctx)[-1]
        ctx.append(int(np.argmax(np.asarray(logits))))
    return ctx


def simulate(prompt, greedy, draft, p_rows, r):
    """Walk the greedy sequence exactly as `speculative_generate` would.

    Returns (passes, fallbacks, tokens, accepted, proposed, per_pass).
    """
    n_prompt = len(prompt)
    seq = list(greedy)
    n_total = len(seq) - n_prompt

    i = 0                    # tokens emitted so far
    passes = fallbacks = accepted = proposed = 0
    per_pass = []
    while i < n_total:
        # Mirror `speculative_generate` EXACTLY. `cur` -- the token the driver holds
        # but has not emitted yet -- is seq[n_prompt + i], and the driver drafts on
        # a context that INCLUDES it (`tokens + generated + [cur]`). Dropping it
        # shifts both the draft context and the truth alignment by one; the tiny-model
        # test `test_acceptance_bench_simulator_matches_the_shipped_driver` pins this.
        context = seq[:n_prompt + i + 1]          # ... + [cur]
        budget = n_total - i
        k = min(p_rows - 1, budget - 1)
        guesses = draft.propose(context, k) if k > 0 else []

        if not guesses:
            # adaptive fallback: a plain 1-token step, cost 1.0
            fallbacks += 1
            i += 1
            continue

        # chain = [cur] + guesses, so guess j is the model's token at
        # seq[n_prompt + i + 1 + j].
        base = n_prompt + i + 1
        truth = seq[base: base + len(guesses)]
        n_acc = 0
        for g, t in zip(guesses, truth):
            if g == t:
                n_acc += 1
            else:
                break
        proposed += len(guesses)
        accepted += n_acc
        emitted = min(n_acc + 1, budget)   # cur + the accepted guesses
        passes += 1
        per_pass.append(emitted)
        i += emitted
    return passes, fallbacks, i, accepted, proposed, per_pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER)
    ap.add_argument("--text", type=Path, default=DEFAULT_TEXT)
    ap.add_argument("--prompt-tokens", type=int, default=128)
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--samples", type=int, default=4,
                    help="independent prompts drawn from the text")
    ap.add_argument("--prefill-tokens", type=int, default=DEFAULT_P)
    ap.add_argument("--sweep-p", default=None,
                    help="comma-separated P:r pairs, e.g. '4:1.10,8:1.22,16:1.4531'. "
                         "The greedy decode is the slow part and is INDEPENDENT of "
                         "P, so one decode prices every P.")
    ap.add_argument("--max-ngram", type=int, default=3)
    ap.add_argument("--r", type=float, default=DEFAULT_R,
                    help="measured pass/step cost ratio (bench_specdec_cycles.py)")
    ap.add_argument("--fmax-mhz", type=float, default=34.41)
    ap.add_argument("--step-cycles", type=int, default=19_794_743)
    ap.add_argument("--model", choices=["fake-quant", "w8a32", "fp32"],
                    default="fake-quant",
                    help="fake-quant = THE CHIP (W8A16: int8 weights, fp16 acts, "
                         "static scales, QuaRot). w8a32/fp32 are cross-checks, "
                         "NOT the chip.")
    args = ap.parse_args()

    import torch
    from taccel.runtime.gpt2_perplexity import tokenize_text_file

    label = {"fake-quant": "W8A16 fake-quant == THE CHIP",
             "w8a32": "W8A32 cross-check (fp32 activations -- NOT the chip)",
             "fp32": "FP32 cross-check -- NOT the chip"}[args.model]
    print(f"loading {args.checkpoint.name}  [{label}] ...", flush=True)
    payload = torch.load(args.checkpoint, map_location="cpu")

    need = args.samples * (args.prompt_tokens + 8) + args.prompt_tokens
    tokens = tokenize_text_file(args.tokenizer_dir, args.text, max_tokens=need + 64)
    if len(tokens) < args.prompt_tokens + 8:
        print("not enough tokens in the eval text", file=sys.stderr)
        return 1

    draft = PromptLookupDraft(max_ngram=args.max_ngram)

    # The greedy decode is the slow part and does NOT depend on P (acceptance is
    # a property of the model's own continuation), so decode once and price every
    # P against it.
    decoded = []
    stride = max(1, (len(tokens) - args.prompt_tokens) // max(1, args.samples))
    for s in range(args.samples):
        off = s * stride
        prompt = tokens[off: off + args.prompt_tokens]
        if len(prompt) < args.prompt_tokens:
            break
        print(f"  sample {s + 1}/{args.samples}: greedy-decoding "
              f"{args.new_tokens} tokens ...", flush=True)
        decoded.append((prompt,
                        greedy_continuation(payload, prompt, args.new_tokens,
                                            model=args.model)))

    base_tok_s0 = args.fmax_mhz * 1e6 / args.step_cycles
    if args.sweep_p:
        print()
        print("=== P sweep (r measured per P by bench_specdec_cycles.py) ===")
        print(f"  {'P':>4} {'r':>8} {'break-even':>11} {'tok/pass':>9} "
              f"{'passes':>7} {'fb':>5} {'speedup':>8} {'tok/s':>8}")
        for pair in args.sweep_p.split(","):
            ps, rs = pair.split(":")
            pp, rr = int(ps), float(rs)
            a = b = c = d = e = 0
            pps = []
            for prompt, greedy in decoded:
                x = simulate(prompt, greedy, draft, pp, rr)
                a += x[0]; b += x[1]; c += x[2]; d += x[3]; e += x[4]
                pps += x[5]
            cost = a * rr + b * 1.0
            sp = c / cost if cost else 0.0
            print(f"  {pp:>4} {rr:>8.4f} {rr:>11.2f} "
                  f"{np.mean(pps) if pps else 0:>9.2f} {a:>7} {b:>5} {sp:>7.2f}x "
                  f"{base_tok_s0 * sp:>8.3f}")
        print()
        print("  NOTE: r is NOT linear past P=16. The systolic verifies P tokens for")
        print("  the price of ONE only up to the MESH HEIGHT (SYSTOLIC_DIM=16). At")
        print("  P=32, M_pad=32 walks TWO m-tiles at full price and sys DOUBLES")
        print("  (measured r(32)=2.6331, sys 2.00x) -- the same 16-row wall that")
        print("  made batching (lever H, B=32) a dud. Always MEASURE r at a new P.")
        print()
        print("  Emitted tokens are IDENTICAL to greedy at every P (accept rule).")
        return 0

    p = args.prefill_tokens
    r = float(args.r)
    tot_passes = tot_fb = tot_tokens = tot_acc = tot_prop = 0
    all_per_pass = []
    for prompt, greedy in decoded:
        passes, fb, emitted, acc, prop, per_pass = simulate(prompt, greedy, draft, p, r)
        tot_passes += passes
        tot_fb += fb
        tot_tokens += emitted
        tot_acc += acc
        tot_prop += prop
        all_per_pass += per_pass

    # Cost model: a verify pass costs r steps, a fallback costs 1 step.
    cost_steps = tot_passes * r + tot_fb * 1.0
    speedup = tot_tokens / cost_steps if cost_steps else 0.0
    base_tok_s = args.fmax_mhz * 1e6 / args.step_cycles

    print()
    print("=== draft (prompt-lookup n-gram) on real text ===")
    print(f"  tokens emitted        {tot_tokens:>10,}")
    print(f"  verify passes         {tot_passes:>10,}")
    print(f"  fallback steps        {tot_fb:>10,}   (draft had no candidate)")
    print(f"  draft acceptance      {tot_acc / tot_prop if tot_prop else 0:>10.3f}"
          f"   ({tot_acc:,}/{tot_prop:,} guessed tokens confirmed)")
    if all_per_pass:
        hist = {k: all_per_pass.count(k) for k in sorted(set(all_per_pass))}
        print(f"  tokens/verify pass    {np.mean(all_per_pass):>10.3f}   hist {hist}")
    print()
    print("=== throughput ===")
    print(f"  measured r (pass/step)          {r:>8.4f}   break-even {r:.2f} tok/pass")
    print(f"  effective cost (steps/token)    {cost_steps / tot_tokens:>8.4f}")
    print(f"  SPEEDUP over sequential greedy  {speedup:>8.3f}x")
    print(f"  b1 tok/s: {base_tok_s:.3f} -> {base_tok_s * speedup:.3f} "
          f"@ {args.fmax_mhz} MHz")
    print()
    if speedup < 1.0:
        print("  NOTE: below 1.0x -- the draft is not clearing break-even on this")
        print("  text. The adaptive fallback bounds the loss, but on this workload")
        print("  spec-dec is not paying for itself.")
    print("  Emitted tokens are IDENTICAL to greedy by construction (accept rule),")
    print("  so this is a pure throughput number -- no quality trade.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
