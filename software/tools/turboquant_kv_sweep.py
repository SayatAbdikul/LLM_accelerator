"""TurboQuant KV-cache Pareto sweep (Tier-1 verification driver).

Grid: variant {mse,ip} × target {k,v,kv} × bits {2,2.5,3,3.5,4}
      × apply_rotation {True,False} × base {weight_only_int8,
      weight_only_int8_quarot}.

`prepare()` runs the expensive rotate+calibrate ONCE per base; every config
is then a cheap reference forward. Output: per-config PPL Δ vs that base's
kv_quant=None anchor, the Pareto frontier (best Δ per effective-bits), and
the two ablations the plan calls for:
  * apply_rotation False vs True  — does TurboQuant's Π earn its keep?
  * base quarot vs plain          — does QuaRot's residual-stream rotation
                                     already Gaussianize the KV (making Π
                                     redundant)?
Writes a JSON + a markdown table to software/logs/turboquant_kv/.
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

from taccel.quantizer.turboquant import TurboQuantKV
from taccel.runtime._turboquant_eval import prepare, ppl_for

FIXTURE = Path("software/tests/fixtures/generated/gpt2_converted_nanogpt.pt")
TOKENIZER_DIR = Path("software/tests/fixtures/generated/hf_gpt2")
CALIB_TEXT = Path("software/tests/fixtures/generated/wikitext2_stage5_calibration.txt")
EVAL_TEXT = Path("software/tests/fixtures/generated/wikitext2_stage5_eval.txt")
OUT = Path("software/logs/turboquant_kv")
SEED = 20260515
D_HEAD = 64
MAX_TOKENS = 33  # fast grid; rerun the winner at 257 via the L3 slow gate

BASES = ["weight_only_int8", "weight_only_int8_quarot"]
VARIANTS = ["mse", "ip"]
TARGETS = ["k", "v", "kv"]
BITS = [2.0, 2.5, 3.0, 3.5, 4.0]
ROT = [True, False]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for base in BASES:
        t0 = time.time()
        print(f"\n=== base={base} : prepare (rotate+calibrate once) ===", flush=True)
        prep = prepare(
            FIXTURE, TOKENIZER_DIR, EVAL_TEXT, max_tokens=MAX_TOKENS,
            ptq_preset=base, calibration_text=CALIB_TEXT,
        )
        anchor = ppl_for(prep, kv_quant=None).perplexity
        print(f"  anchor(kv=None)={anchor:.4f}  prep {time.time()-t0:.0f}s", flush=True)
        for variant, target, bits, rot in itertools.product(
            VARIANTS, TARGETS, BITS, ROT
        ):
            tq = TurboQuantKV(d=D_HEAD, bits=bits, variant=variant,
                              target=target, apply_rotation=rot, seed=SEED)
            try:
                ppl = ppl_for(prep, kv_quant=tq).perplexity
                rel = 100.0 * (ppl - anchor) / anchor
                ok = True
            except Exception as e:  # one bad config must not kill the sweep
                ppl, rel, ok = float("nan"), float("nan"), False
                print(f"  FAIL {variant}/{target}/{bits}/rot={rot}: {e}", flush=True)
            row = dict(base=base, anchor=anchor, variant=variant, target=target,
                       bits=bits, apply_rotation=rot, ppl=ppl, rel_pct=rel, ok=ok)
            rows.append(row)
            if ok:
                print(f"  {variant:3} {target:2} b={bits:<3} rot={int(rot)} "
                      f"ppl={ppl:8.3f}  Δ={rel:+6.1f}%", flush=True)
        (OUT / "sweep.json").write_text(json.dumps(rows, indent=2))  # incremental

    # ---- Pareto frontier (min |Δ| per (base,bits), quality-neutral first) ----
    md = ["# TurboQuant KV sweep (33-tok, Δ vs per-base kv=None anchor)\n"]
    for base in BASES:
        br = [r for r in rows if r["base"] == base and r["ok"]]
        if not br:
            continue
        anchor = br[0]["anchor"]
        md.append(f"\n## base={base}  anchor={anchor:.3f}\n")
        md.append("| eff.bits | best cfg | ppl | Δ% |")
        md.append("|---|---|---|---|")
        for b in BITS:
            cand = [r for r in br if r["bits"] == b]
            best = min(cand, key=lambda r: abs(r["rel_pct"]))
            md.append(f"| {b} | {best['variant']}/{best['target']}/"
                      f"rot={int(best['apply_rotation'])} | {best['ppl']:.3f} "
                      f"| {best['rel_pct']:+.1f} |")
        # ablation: Π on vs off, matched otherwise (kv, mse)
        for b in (3.0, 4.0):
            on = next((r for r in br if r["variant"] == "mse" and r["target"] == "kv"
                       and r["bits"] == b and r["apply_rotation"]), None)
            off = next((r for r in br if r["variant"] == "mse" and r["target"] == "kv"
                        and r["bits"] == b and not r["apply_rotation"]), None)
            if on and off:
                md.append(f"\nΠ ablation @ mse/kv/{b}b: rot=1 Δ={on['rel_pct']:+.1f}% "
                          f"vs rot=0 Δ={off['rel_pct']:+.1f}%  "
                          f"(Π {'earns its keep' if abs(on['rel_pct'])+0.5 < abs(off['rel_pct']) else 'looks redundant'})")
    (OUT / "pareto.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nwrote {OUT/'sweep.json'} and {OUT/'pareto.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
