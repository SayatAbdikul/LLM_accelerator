# Performance roadmap — the architecture campaign, honest re-base (2026-07-16)

> **Historical campaign record, reconciled 2026-09-03.** Later pipeline work
> changed the latest cycle counts to **18,318,261 cycles** for b1 at position
> 511 and **49,998,042 cycles per 16-token step** for b16 at position 510
> (3,124,878 cycles/token). The logits SHA-1 prefixes are `eeab004014642d14`
> and `205682b6515f7e85`, respectively. The 34.41 MHz peg and every derived
> token rate below came from earlier partial-design timing; later integrated
> exp/GELU/full-lane paths have no successful full-chip PNR sign-off on the
> available 15 GB host. Use [current project status](project_status.md) for the
> live state and [the documentation index](README.md) for authority.

Successor to `perf_roadmap_2026-07-10.md`. Two things happened since that doc:
**(1) the Port-A corruption discovery re-based every number** (the old machine
silently dropped systolic drain writes during DMA‖systolic overlap — all
pre-`daef072` tok/s carried a ~7.3% subsidy and 99.78%-wrong logits at b16);
**(2) the effort re-organized from single levers into a tiered campaign** —
occupancy (T1/T2) → clock (T3) → width (T4) — after measurement showed the
machine is idle, slow-clocked, and narrow, in that order.

**Scope pin (the standing goal): make the CHIP faster — RTL architecture +
compiler only.** Model changes (better spec-dec drafts, sampling, prompts) and
host software are out of scope; weights stay INT8. Numbers are GPT-2 124M
decode at the ctx-512 budget, honest-BW `--fast-beats`, 34.41 MHz unless
noted. The DRAM-BW model is PINNED: bandwidth scales with core clock.

## 0. Measured state at HEAD `9a82e34`

| shape | cyc/step | cyc/tok | tok/s | partition |
|---|---:|---:|---:|---|
| decode b1, pos 511 | 18,300,503 | 18,300,503 | **1.880** | sys 11.11M (61%) · DMA 10.53M beats (co-busy 5.64M) · ctl ~2.0M · SFU 0.63M |
| decode b16, pos 510 | 49,803,314 | 3,112,707 | **11.055** (aggregate) | sys 22.32M (45%) · SFU 10.07M (20%) · DMA 19.66M beats (co-busy ~8.6M) |

Adjacent capabilities (both byte-inert by default): **chunked prefill** — TTFT
12.38× on a 128-token prompt (69.9 s → 5.6 s, lever I-b; ratio measured
pre-re-base, both sides on the same machine, so it carries); **speculative
decode** — an opt-in HOST track (`prefill_tokens>1`), b1 1.746 → 2.906 tok/s
at P=4 exact-greedy on the chip's own W8A16 numerics (`54c0314` fence,
`docs/lever_b3_specdec.md`).

⚠️ **The 34.41 MHz peg is CONTINGENT** (T0.3, `docs/t0_sfu_fmax_audit.md`):
it is a div/sqrt-primitive STA number that excludes the un-pipelined
`fp32_exp` clouds (~412–490 ns single-cycle). Cycle counts and byte-exactness
are unaffected; the ns→tok/s conversion becomes honest when T3 step 0's
integration lands. All cycle numbers above are exact, from direct
`--fast-beats` runs.

## 1. The honest re-base (what invalidated the old waterfall)

- **Phase 0** (`0c31852`, `a2201e3`, `docs/phase0_measurement.md`): with
  DMA‖systolic overlap enabled, the systolic's ACCUM drain writes lost the
  shared Port-A bus to the DMA with no backpressure — **255,818 silently
  dropped drain writes per b16 step**, 99.78% of logits bytes wrong. The
  "byte-exact" tiny gate had ZERO co-busy cycles and was structurally blind.
- **Lever 1a** (`4a58734`): deleted the dead ACCUM preclear — byte-exact by
  proof, −2,318,647 cyc.
- **The fix** (`daef072`, `docs/porta_bus_split.md`): Port A fanned out per
  buffer + the systolic got its own Port-S channel; same-buffer collision now
  FAULTS. Logits byte-identical to a correct machine, lost writes 257,406 → 0,
  co-busy **unchanged** (the concurrency was kept, not re-serialized).
  Honest b16: **10.145 tok/s**.
- **The doctrine** (standing gate for every overlap change): total cycles ↓
  (primary, un-fakeable) + co-busy quoted and moving the right way + Port-A
  audit `lost=0` + RTL-vs-RTL logits sha1 unchanged. Never "byte-exact" alone.

## 2. Landed since the 07-10 roadmap (ledger)

| commit(s) | what | result |
|---|---|---|
| `9f1bd2c` | lever E fmax cluster (div_p6 + sqrt restructure) | SFU primitive floor 32.10 → 29.64 ns (+8.3% fmax), cycle-neutral; PNR stamp deferred (≥24 GB) |
| `5618577` | I-a logits×N | batched decode emits all 16 per-stream logits rows, +0.51% cyc |
| `2554d63`, `4cc7942` | H: B=32 | **+3.45% only ⇒ batching MINED OUT** (sys scales exactly 2×; ACCUM at 100%); also fixed a latent batched-prefill KV-store weight-corruption bug |
| `b566733`, `67c7f46`, `b187549` | I-b chunked multi-token prefill | **TTFT 12.38×** on 124M, byte-exact at every prompt length; P=16 chunk = the decode graph with 16 query rows |
| `0c31852`, `a2201e3`, `4a58734`, `daef072` | phase 0 + 1a + Port-A split | the honest re-base above |
| `7f22243`…`54c0314` | B3 speculative decode | b1 2.906 @P=4 exact-greedy (W8A16); r(P)=1+0.0302(P−1) up to the 16-row mesh; fenced opt-in, byte-inert by default |
| `23c8eb0` | T0 instrumentation | fetch-stall + per-op DMA-exposure counters, `--beat-interval` BW knob; b1 fetch-stall 1.35M (6.9%) |
| `77e5803`, `c5f4d40` | T0.3 STA audit | **the real SFU binder is exp/EXPSUM (~412/~490 ns), not div/sqrt**; fp32_add 28.49 ns / mul 27.34 ns; "helper 109 MHz" is a non-fp path |
| `316621e`, `ba5f129` | T1 measured redirect | re-sized the T1 items from measurement (two conclusions later overturned — see banner in that doc) |
| `e6f7006` | **T3 step 0 primitive**: `fp32_exp_p18` | 18-stage pipelined exp, bit-IDENTICAL (pure retiming), zero-diff gate; **not yet instantiated in the SFU** |
| `dde6f19` | **T1 item 1**: b16 KV V-prefetch | b16 10.145 → 10.733 (+5.8%), byte-exact, co-busy +2.96M ≈ Δcyc |
| `9a82e34` | **T1 item 2**: FC2 weight prefetch | b1 1.738 → **1.880 (+8.17%)**, b16 → **11.055 (+3.0%)**; identical −1,494,240 cyc at both shapes (batch-invariant weight stream) |

Details: `docs/t1_overlap_items.md`, `docs/lever_h_b32.md`,
`docs/lever_i_serving.md`, `docs/lever_b3_specdec.md`.

## 3. The campaign: T0 → T4 (status and remaining work)

**T0 — instrumentation + gates: ✅ DONE** (`23c8eb0` + the T0.3 audit). The
DRAM-BW model is pinned and checkable (`--beat-interval`); exposed-DMA,
fetch-stall, co-busy, and Port-A-audit counters exist and are byte-exact.

**T1 — compiler overlap (zero RTL): items 1+2 LANDED, vein still open.**
- ✅ item 1 (b16 KV V-prefetch, +5.8%) — capacity wall caps the whole KV vein
  at +8–11% total (KV 12.3 MB/layer ≫ 384 KB SRAM ⇒ KV overlaps only
  attention's own systolic, ~5.5M).
- ✅ item 2 (FC2 weight prefetch, b1 +8.2% / b16 +3.0%) — refuted "b1 is
  walled": measured slack 6.96M > exposed 5.89M.
- Next, in order: **item 2b** FC2 input hoist (byte-exact, REDUCES DMA
  ~0.28M at b1, small+strict); **increment 2** next-group K^T prefetch into
  the freed `__ktall` WBUF region (b16, ~+3–5% est.); then the ~2M
  prologue/tail exposed DMA on the pipelined full-K paths (harder).
- Dead/deprioritized: item 3 per-head DMA-transpose is NET-NEGATIVE at b1
  (helper transposes are already hidden — 986K busy / 10K exposed); item 4
  SET_ADDR dedup ceiling is small (the 32.4 cyc/op is the following LOAD's
  startup latency, not a bubble).
- b1 scheduling ceiling ≈ ~2.5 tok/s; b16 T1 remainder ≈ +4–8%.

**T2 — small RTL (byte-exact): not started.** (1) instruction prefetch buffer
in `fetch_unit.sv` (each 16-byte beat holds TWO instructions; today the
sibling is discarded and re-fetched — T0 measured 1.35M/6.9% fetch-stall at
b1); (2) A-load reuse across n-tiles (`systolic_controller.sv`, ~0.25M);
(3) attention-shape drain/flush overlap (fixed strip overhead is 45–64% of
decode QKT/AV strips; ~1–2M); (4) optional SFU‖DMA legalization as
schedule-slack insurance (Port-S playbook: own channel, collision FAULT,
audit counters — the silent-drop class must not return).

**T3 — the clock (single-domain fmax, the multiplier): step 0 half-landed.**
Re-ordered by T0.3 — the binder is exp, then div/sqrt:
0. **Pipeline exp**: primitive ✅ (`fp32_exp_p18`, bit-identical, 18-stage);
   **remaining = instantiate it at the three single-cycle sites** (EXPSUM
   `sfu_synth_datapath.svh:400`, attention-V weight `:568`, GELU
   `fp32_gelu_new.sv`) with the feed-1/collect-18 software pipeline the
   div_p6 softmax-OUT drain already uses (order preserved ⇒ byte-exact).
   This makes the current 34.41 MHz honest BEFORE any speedup.
1. Deepen `fp32_div_p6` → ~12–15 stages and `fp32_sqrt_p6` similarly
   (uniform iterations; latency plumbing at ~10 known sites).
2. `fp32_add` (28.49 ns) + `fp32_mul` (27.34 ns) 2–3-stage splits — required
   only for the 70–90 MHz stretch, not at 34.41.
3. Audit the rest (GELU beyond exp, scalar div sites, fp16↔fp32 converts);
   helper/DMA never bind ≤90 MHz.
4. CDC slow-SFU island: **REJECTED** (supersedes the 07-10 roadmap's lever F)
   — at a 70–90 MHz single-domain target the island would cap exactly the
   block being sped up.
- Verification per step: bit-exact 10M-vector primitive diff, cycle-neutral
  same-position A/B, standalone synth+STA; **full-chip PNR stamp deferred to
  a ≥24 GB box** (the full-SFU flatten OOMs at 15 GB — do not retry here).
- Expected (pinned BW model): ×2.0–2.6 on both shapes once 70–90 MHz closes.

**T4 — memory width 32 B/beat: conditional, last.** Only if post-T2/T3
measurement shows sys < DMA-cycles. ISA/compiler are width-invisible (all
size fields count 16-byte SRAM rows); the lever-D transpose byte-routing and
~30 literal sites need rework; re-STA the DMA at 32 B.

**Post-T3, b16-specific: 2-lane SFU.** At b16 the SFU is ~10.1M/step (20%)
and no tier above touches it; elementwise SFU work is order-insensitive ⇒
2-laning is byte-exact (only EXPSUM's ordered accumulate is locked to 1
elem/cyc). Scope after T3.

**Endgame (recorded, out of scope):** persistent on-chip KV (one layer at
ctx-512 ≈ 786 KB > the chip's 448 KB total SRAM — floorplan change), >16-row
mesh, spatial-reduction attention engine. Each KV byte is read exactly once
per step (verified) — no cache lever exists at this SRAM size.

## 4. Floors and the mined-out ledger

Hard floors at 34.41 MHz:
- **b1 systolic floor 11.11M cyc ⇒ ~3.1 tok/s ceiling** — M=1 leaves the
  16-row mesh 15/16 empty; only batching/spec-dec/clock touch it. The DMA
  wall (~3.27, the whole 124 MB model per token) sits right behind it.
- **b16 KV capacity wall**: KV can never ride under the MLP matmuls
  (12.3 MB/layer vs 384 KB SRAM), so the KV-overlap vein tops out at +8–11%.

Mined out (do not revisit without new structure): batching beyond B=16
(lever H: sys scales exactly 2.000×); QK^T packing (~8.4 tok/s ceiling,
group=7); elementwise SFU load-fusion (QUANT/DEQUANT/VADD all load-bound);
the serialized helper K^T pass (lever D deleted it); the ACCUM preclear (1a);
SFU row-walk padding (m_exact).

## 5. Verification (standing gates, every item)

1. Byte-exact = tiny RTL==golden + 124M A/B; at 124M use RTL-vs-RTL logits
   sha1 (RTL-vs-golden byte-match is ill-posed past the first fp16 overflow —
   golden saturates; rtl_cosim #109).
2. Overlap changes additionally gate on: direct `--fast-beats` cycles ↓,
   co-busy quoted + moving, Port-A audit `lost=0`
   (`software/tools/fast_gate_b16.py`, ~15–20 min/run).
3. Inertness pin: `test_specdec_is_inert_at_the_default` — default bundles
   byte-pinned (current goldens: b1 `172b4aa61a3de54e`, b16
   `fa75a8991ee385b3`); re-pins are user-owned sign-offs.
4. fmax: standalone sky130 synth+STA per changed path; bit-exactness by
   vector diff (the div_p6/exp_p18 pattern); full PNR deferred to ≥24 GB.
5. Known-failure baseline: the pre-existing suite failures (fixture SHA drift,
   `n_embd` KeyError, W4 tile_config tuple, fp16-embedding-era synthetic)
   must not grow — verify against a clean-HEAD stash baseline, not memory.
6. Ops box notes: never two yosys jobs (or synth+PNR, or two 124M sims)
   concurrently on a 15 GB box; rebuild `run_program_synth` with
   `-GDRAM_SIZE=1073741824` for 124M.

## 6. Waterfall

```
b16 aggregate (as-measured then, ctx-512):
  2.79 → A 3.80 → C 7.19 → B 8.38 → D 9.78 → E ~10.6   [pre-re-base machine]
honest chain (post-daef072, 34.41 MHz):
  10.145 → item-1 10.733 → item-2 11.055
  → T1 remainder ~11.3-11.9 → T2 ~12-13 → T3 ×2.0-2.6 ⇒ ~24-31 → 2-lane SFU
b1:
  1.016 → A 1.126 → C 1.633 → (re-base era) 1.738-1.746 → item-2 1.880
  → T1 remainder ~2.0 → scheduling ceiling ~2.5 → sys floor 3.1 → T3 ⇒ ~3.8-4.9
  (spec-dec, opt-in host track: 2.906 today at P=4)
```

The T3 range here is REVISED DOWN from the planning estimate (which assumed
the pre-capacity-wall T1 endpoint). Fixed-GB/s sensitivity (if the BW pin ever
changes): T3's gain compresses toward ~×1.4 at b1 and T4 becomes core.
