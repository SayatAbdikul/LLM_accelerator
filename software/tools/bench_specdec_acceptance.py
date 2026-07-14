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

The model is the INT8-weight reference (`build_weight_only_int8_reference`) --
the same weight numerics the chip runs -- so `t` is not measured on a model the
accelerator does not implement. `--fp32` cross-checks that the quantization is
not what is driving acceptance.
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


def greedy_continuation(payload, prompt, n_new, *, fp32=False):
    """Greedy-decode `n_new` tokens. Returns the token list."""
    from taccel.runtime.fp32_reference import (
        NanoGPTFP32Reference,
        build_weight_only_int8_reference,
    )

    if fp32:
        ref = NanoGPTFP32Reference(payload["state_dict"], payload["model_args"])
    else:
        ref = build_weight_only_int8_reference(payload, weight_mode="per_channel")

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
        cur = seq[n_prompt + i - 1] if i > 0 else seq[n_prompt - 1]
        # context as the driver sees it: everything decided so far, plus `cur`
        context = seq[:n_prompt + i]
        budget = n_total - i
        k = min(p_rows - 1, budget - 1)
        guesses = draft.propose(context, k) if k > 0 else []

        if not guesses:
            # adaptive fallback: a plain 1-token step, cost 1.0
            fallbacks += 1
            i += 1
            continue

        truth = seq[n_prompt + i: n_prompt + i + len(guesses)]
        n_acc = 0
        for g, t in zip(guesses, truth):
            if g == t:
                n_acc += 1
            else:
                break
        proposed += len(guesses)
        accepted += n_acc
        emitted = min(n_acc + 1, budget)   # accepted + the correction token
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
    ap.add_argument("--fp32", action="store_true",
                    help="use the FP32 model instead of the INT8-weight one")
    args = ap.parse_args()

    import torch
    from taccel.runtime.gpt2_perplexity import tokenize_text_file

    print(f"loading {args.checkpoint.name} "
          f"({'fp32' if args.fp32 else 'int8-weight (chip numerics)'}) ...", flush=True)
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
                                            fp32=args.fp32)))

    base_tok_s0 = args.fmax_mhz * 1e6 / args.step_cycles
    if args.sweep_p:
        print()
        print("=== P sweep (r measured per P by bench_specdec_cycles.py) ===")
        print(f"  {'P':>4} {'r':>8} {'break-even':>11} {'tok/pass':>9} "
              f"{'speedup':>8} {'tok/s':>8}")
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
                  f"{np.mean(pps) if pps else 0:>9.2f} {sp:>7.2f}x "
                  f"{base_tok_s0 * sp:>8.3f}")
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
