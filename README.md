# LLM Accelerator (TACCEL)

An LLM inference accelerator, measured end-to-end on GPT-2 124M: SystemVerilog
RTL (16×16 INT8 systolic array + FP32 special-function unit + DMA, byte-exact
against a pinned golden model) plus a Python toolchain (ISA, compiler,
quantizer, golden simulator, host runtime). The primary target is ASIC
(sky130: yosys synth + STA, per-block OpenROAD PNR); an FPGA wrapper exists as
a smoke-tested skeleton.

**Measured decode throughput (GPT-2 124M, W8A16, ctx-512 budget, post-PNR
34.41 MHz, honest-bandwidth DRAM model):**

| shape | tok/s | note |
|---|---:|---|
| batch-16 decode (16 streams) | **11.055** aggregate | 49,803,314 cyc/step, 3.11M cyc/token |
| batch-1 decode | **1.880** | 18,300,503 cyc/token; systolic floor ~3.1 |
| prefill (chunked, 128-tok prompt) | TTFT **5.6 s** | 12.4× faster than sequential prefill |
| batch-1 + speculative decode (opt-in host track) | 2.906 | P=4, exact-greedy, byte-inert unless enabled |

The project started as an INT8 ViT accelerator (DeiT-tiny) and grew a decoder
ISA + GPT-2 frontend on top; both lineages remain in the tree.

---

## Status snapshot

| Area | State |
|---|---|
| **ISA** | gen-2 frozen 2026-05-19 (19 emitted opcodes, FP32 sub-layer `0x17–0x1F`), 8-byte fixed encoding, 32-slot space. Post-freeze revisions via the freeze §6 mechanism: `m_exact` (CONFIG_TILE[27:16], exact SFU row count) and the DMA transpose-LOAD (reserved M-type bits). Contract: `software/docs/isa_generation_freeze.md`. |
| **RTL** | All engines synthesizable (yosys `synth-check` GREEN, no `real`/DPI in the netlist path). 16×16 INT8 systolic (chained default), gen-2 FP32 SFU with pipelined div/sqrt (`fp32_div_p6`/`fp32_sqrt_p6`), DMA with transpose-load, per-buffer Port-A fan-out + dedicated systolic drain channel (Port S) with collision-FAULT. ~14.5 k LOC SV. |
| **Correctness** | Tiny fixture: RTL == golden **byte-identical** (freeze cosim). 124M: RTL-vs-RTL logits SHA gates + argmax/PPL conformance (RTL-vs-golden byte-match is ill-posed past the first fp16 overflow — the golden saturates). The 2026-07 Port-A audit found and fixed a months-old silent-corruption bug in DMA‖systolic overlap; the audit counters (`lost=0`, co-busy) are now part of every overlap gate. |
| **Performance** | Table above; measured by direct RTL cycle counts (`run_program_synth --fast-beats`) × post-PNR fmax. History and next levers: `docs/perf_roadmap_2026-07-16.md`. |
| **fmax** | 34.41 MHz post-PNR (sky130, per-block flow; full-chip PNR needs a ≥24 GB box). **Contingent**: the peg excludes the un-pipelined `fp32_exp` clouds; the pipelined replacement (`fp32_exp_p18`, bit-identical) is landed but not yet integrated — `docs/t0_sfu_fmax_audit.md`. Standalone div/sqrt floor after lever E: 29.64 ns. |
| **Quantization** | W8A16 + QuaRot production preset: **56.23 PPL** (FP32 ceiling 53.42, GPT-2 124M 257-tok). W4A16 + AWQ + GPTQ: 63.04 (tier-1 refinements: ~55.04). KV cache stored INT8 (store-time quant, bit-exact — static scales). |
| **Serving** | Batched decode B=16 (per-stream KV, all 16 logits rows emitted); chunked multi-token prefill (any prompt length, byte-exact); speculative decoding fenced as an opt-in host track. B=32 measured +3.45% ⇒ batching is mined out at B=16. |
| **Roadmap** | `docs/perf_roadmap_2026-07-16.md` — occupancy (T1/T2) → clock (T3) → width (T4). Done: T0 instrumentation, T1 items 1+2. Next: T3 step 0 integration (exp pipelining into the SFU), T1 remainder. |

---

## Repository layout

```
LLM_accelerator/
├── software/                       # Python toolchain + tests + tools
│   ├── taccel/
│   │   ├── isa/                    # Opcodes, instruction dataclasses, 8-byte encoding
│   │   ├── assembler/              # Two-pass assembler + disassembler, ProgramBinary/Bundle
│   │   ├── quantizer/              # W8/W4, AWQ, GPTQ, QuaRot, SmoothQuant, TurboQuant, …
│   │   ├── compiler/               # Graph frontend (nanogpt/GPT-2/DeiT), tiling, memory
│   │   │                           #   alloc, codegen (emit/ + w8a16_emit/), decoder bundle,
│   │   │                           #   KV-cache layout, packed attention, prefetch scheduling
│   │   ├── golden_model/           # Bit-accurate Python simulator (content-pinned)
│   │   └── runtime/                # HostRunner (decode loop, chunked prefill, spec-dec),
│   │                               #   PPL eval, calibration, W4 path, disk caches
│   ├── tests/                      # 58 pytest files: freeze cosim, batched decode,
│   │                               #   spec-dec inertness pin, ISA/codegen byte-identity, PPL
│   ├── tools/                      # CLIs: bench_decode_cycles, fast_gate_b16,
│   │                               #   profile_decode_step, rtl_cosim, evaluate_gpt2_perplexity,
│   │                               #   asm/disasm, gen_gen2_fixtures, …
│   └── docs/                       # ISA freeze contract + design notes
│
├── rtl/
│   ├── common/
│   │   ├── src/                    # The core: taccel_top, fetch/decode/control, dma_engine,
│   │   │   │                       #   blocking_helper_engine, sfu_engine (+3 .svh partitions)
│   │   │   ├── include/            # taccel_pkg.sv (params, opcodes, faults, SYNC bits)
│   │   │   ├── systolic/           # 16×16 PE array + controller
│   │   │   ├── memory/             # sram_subsystem (Port A/S/B/W), sram_dp, register_file
│   │   │   ├── fp32/               # 20 synthesizable IEEE-754 primitives (add, mul,
│   │   │   │                       #   div_p2..p6, sqrt_p2..p6, exp, exp_p18, gelu, cvt, …)
│   │   │   └── tb/                 # SV bench models (AXI slave, decode tb)
│   │   └── filelists/              # core.f — the single source-of-truth filelist
│   ├── verilator/                  # Native C++ benches + run_program harness (primary sim);
│   │                               #   run_program_synth = the perf-measurement build
│   ├── cocotb/                     # Python-driven ISA-visible benches
│   ├── asic/                       # ASIC wrappers, sky130 SRAM macro stub, synth + STA +
│   │                               #   OpenROAD per-block PNR flows (build/synth_blocks,
│   │                               #   build/openroad), OpenLane config
│   ├── fpga/                       # FPGA wrapper + vendor stubs; yosys-fpga smoke gate
│   └── synth/                      # Generic yosys synth-check gate + Phase-2/3 records
│
└── docs/                           # Roadmaps, landed-lever reports, audits (see Doc index)
```

---

## Quickstart

### Environment

```sh
python -m venv .venv && source .venv/bin/activate
pip install -r software/requirements.txt
```

You'll also need Verilator (4.2+) for the RTL benches, and yosys + sv2v for
the synth gates. All commands run from the repo root; `pytest` picks up
`software/tests/conftest.py` (adds `software/` to `sys.path`), and direct
script invocations use `PYTHONPATH=software`.

### Run the correctness gates (golden pins + RTL == golden byte-match)

```sh
make -C rtl/verilator run_program run_program_synth   # build the RTL runners once
pytest software/tests/test_compare_rtl_golden.py software/tests/test_batched_decode.py -v
```

`test_compare_rtl_golden.py` pins the golden simulator's content SHA (freeze
§6) and asserts the frozen `weight_only_int8_quarot` bundle builds
deterministically with gen-2 opcodes (its single-stream RTL leg is bridged
out — task #105; the opt-in 124M logits-metric leg is gated by
`PYTEST_124M=1`). `test_batched_decode.py` is the live RTL==golden
byte-match: tiny decode bundles — including the packed batch-16 attention
path — run on the Verilator RTL and must produce logits byte-identical to
the golden simulator.

### Measure decode performance (the numbers in this README)

```sh
make -C rtl/verilator run_program_synth        # mode-1 (synth SFU), 1 GiB DRAM
PYTHONPATH=software python3 software/tools/bench_decode_cycles.py ...   # decode @ ctx-512
PYTHONPATH=software python3 software/tools/fast_gate_b16.py            # b16 gate: cycles,
                                               # tok/s, co-busy, Port-A audit, logits sha1
```

`run_program_synth` is the runner every perf number is measured on:
`SFU_SYNTH_MODE=1` (the real datapath, not the DPI reference) and always
`--fast-beats` (the pinned honest-bandwidth DRAM model — 1 beat/cycle,
bandwidth scales with core clock; `--beat-interval N` simulates fixed-GB/s).
`profile_decode_step.py` adds per-opcode retire-gap attribution + the
exposed-DMA census (slower). Don't run two 124M sims (or a sim + a yosys
job) concurrently on a 15 GB box.

### Evaluate GPT-2 124M perplexity

```sh
PYTHONPATH=software python3 software/tools/evaluate_gpt2_perplexity.py \
    software/tests/fixtures/generated/gpt2_converted_nanogpt.pt \
    --tokenizer-dir software/tests/fixtures/generated/hf_gpt2 \
    --calibration-text <calib.txt> --eval-text <eval.txt> \
    --max-eval-tokens 257 --ptq-preset weight_only_int8_quarot
```

Presets: `weight_only_int8`, `weight_only_int8_quarot` (production),
`weight_only_int4_awq_gptq`, and activation-aware variants — see
`software/taccel/runtime/stage5_ptq.py`.

### RTL benches

```sh
make -C rtl/verilator all             # decode/control/dma/helpers/sfu/systolic C++ benches
make -C rtl/verilator test_sfu_synth  # mode-1 SFU datapath gate (11 cases)
make -C rtl/verilator test_fp32_div_p6 test_fp32_sqrt_p6 test_fp32_exp_p18
                                      # pipelined-primitive bit-exactness gates
make -C rtl/cocotb test_all SIM=verilator      # Python-driven ISA benches
```

Bench guide: [`rtl/TESTBENCHES.md`](rtl/TESTBENCHES.md).

### Synthesis / timing / PNR

```sh
make -C rtl/verilator synth-check              # whole-design yosys elaboration gate
make -C rtl/asic yosys-asic                    # ASIC wrapper smoke gate
rtl/asic/build/synth_blocks/synth_full.sh <top>    # sky130 synth (abc -D 5000) + .log
rtl/asic/build/synth_blocks/run_block_sta.sh ...   # OpenSTA per block
openroad ... rtl/asic/build/openroad/block_pnr.tcl # per-block PNR + STA (sky130A PDK)
```

Memory caveats: yosys's SHARE pass hangs on the SFU (use the `-noshare`
flows); the full-SFU flatten + PNR OOMs below ~24 GB RAM — per-block and
per-primitive flows are the supported path on small boxes. Never run two
yosys jobs concurrently on a 15 GB machine.

### Full software test suite

```sh
pytest software/tests/ -n auto
```

`PYTEST_SLOW=1` enables long PPL gates; `PYTEST_124M=1` enables the 124M
cosim leg. A small pinned set of failures is pre-existing (fixture SHA drift,
`n_embd` KeyError cases, W4 `tile_config` tuple, fp16-embedding-era synthetic
tests) — gate new work against a clean-HEAD baseline, not against zero
failures.

---

## Architecture

### Hardware (rtl/common/src)

One in-order control plane driving four engines over three on-chip SRAMs:

- **`taccel_top.sv`** — fetch/decode/issue glue, AXI-read arbitration
  (fetch vs DMA), SRAM port arbitration, fault collapse, and the
  observability counters (`obs_*`: DMA/systolic co-busy, Port-A lost-write
  audit, fetch-stall, per-engine busy) that the perf tooling samples.
- **`fetch_unit.sv`** — single-beat AXI reads; 8-byte instructions, two per
  16-byte beat (the sibling is currently re-fetched — a known T2 lever).
- **`control_unit.sv`** — in-order issue. Concurrency contract: **DMA ‖
  systolic only** (`SYS_DMA_OVERLAP=1`); SFU and helper are serialized.
  `SYNC` waits on a resource mask (bit0=DMA, bit1=systolic, bit2=SFU).
- **`dma_engine.sv`** — AXI4 LOAD/STORE bursts, whole-transfer OOB
  prevalidation, and the lever-D **transpose-LOAD** (16-row-stripe byte
  transpose on the fly — this deleted the serialized K^T helper pass).
- **`systolic/`** — 16×16 INT8 PEs (chained skew default), tile-walk
  controller with double-buffered A-tiles, K-split accumulate (RMW drain),
  and dead-preclear elimination.
- **`sfu_engine.sv`** (+ `sfu_synth_datapath.svh`, `sfu_g2_compute.svh`,
  `sfu_dpi_helpers.svh`) — the gen-2 FP32 ops (dequant/quant, LN, GELU,
  masked softmax, max-abs). Mode 0 = DPI reference; mode 1 = the
  synthesizable fp32-primitive datapath (the chip). `m_exact` bounds row
  walks to real rows (the lever that removed the 16-row padding tax).
  Divider/sqrt are 6-stage pipelined; **exp is still combinational in the
  datapath** — its pipelined replacement `fp32_exp_p18` is verified and
  awaiting integration (T3 step 0).
- **`blocking_helper_engine.sv`** — gen-1 helper ops (BUF_COPY, VADD,
  REQUANT[_PC], SCALE_MUL, DEQUANT_ADD); fully blocking.
- **`memory/sram_subsystem.sv`** — ABUF **128 KB** / WBUF **256 KB** / ACCUM
  **64 KB** (448 KB total, 16-byte rows) as three dual-port macros. Channels:
  shared Port A (DMA/helper/SFU, fanned out per buffer), **Port S**
  (systolic-dedicated writes — the 2026-07 bus split that fixed the silent
  drain-drop corruption; same-buffer A/S collision now FAULTs), Port B
  (systolic src1 reads), **Port W** (systolic src2 reads from WBUF — what
  makes DMA-prefetch-under-MATMUL port-safe).
- **`fp32/`** — 20 synthesizable IEEE-754 primitives, each bit-exact to its
  DPI golden or its combinational parent (`fp32_add/mul/div/sqrt/exp/
  gelu_new/quantize_i8/…`, pipelined `div_p2..p6`, `sqrt_p2..p6`,
  `exp_p18`).

Key parameters (`include/taccel_pkg.sv`): `SYS_DIM=16`, `AXI_DATA_W=128`
(16 B/beat), 56-bit DRAM addressing, 16 FP16 scale regs, 4 addr regs.
Peak: 256 MACs × 2 ops × 34.41 MHz ≈ **0.018 int8 TOPS** (useful utilization
~17% at b16, ~3% at b1 — M=1 fills 1 of 16 mesh rows, which is why batching
and the clock are the levers, not more MACs).

### Software (software/taccel)

1. **ISA** (`isa/`) — 32-slot opcode space, 8-byte big-endian encoding.
   Frozen gen-2 contract + dated §6 revisions (`m_exact`, transpose-LOAD).
2. **Assembler** (`assembler/`) — two-pass; `ProgramBinary` (cosim) and
   two-stream `ProgramBundle` (host decode path with runtime patch sites).
3. **Quantizer** (`quantizer/`) — per-channel W8/W4, AWQ, GPTQ, QuaRot,
   SmoothQuant, AdaRound, TurboQuant-KV, LN-fold, bias correction.
4. **Compiler** (`compiler/`) — graph frontend (`frontend/nanogpt_adapter.py`
   builds the decode/prefill graphs: KV-cache injection at batch=1, packed
   multi-head attention groups at batch≥2, chunked-prefill graphs, prefetch
   attributes); tile planning + memory allocation with eviction; codegen
   split into `emit/` (generic) and `w8a16_emit/` (the W8A16 production
   path: attention, matmuls with weight-prefetch double-buffering, shared
   ops); KV layout; decoder bundle with runtime patch sites (KV bases,
   position, valid_kv_len).
5. **Golden model** (`golden_model/`) — bit-accurate Python simulator,
   content-pinned (SHA gate) as the conformance arbiter.
6. **Runtime** (`runtime/`) — `HostRunner` (decode loop, batched decode,
   `run_prefill_chunk`, opt-in speculative decoding), PPL evaluator,
   calibration, fixtures (`tiny_fixture.py` builds tiny + 124M bundles).
7. **Tools** (`tools/`) — see Quickstart; plus `rtl_cosim.py`,
   `audit_porta_argmax.py`, spec-dec benches, fixture generators.
8. **Tests** (`tests/`) — 58 files: encoding round-trips, quantizer parity,
   codegen byte-identity, freeze cosim, batched-decode RTL==golden,
   spec-dec inertness byte-pin, PPL gates.

Numerics end to end: weights INT8 (per-channel, QuaRot); activations FP16;
KV cache INT8 (store-time quant, static scales ⇒ bit-exact); ACCUM INT32;
SFU FP32 with characterized ULP bands (freeze §7).

---

## Performance measurement doctrine

Hard-won rules, enforced by the tooling (the full story:
`docs/phase0_measurement.md` → `docs/porta_bus_split.md`):

1. **Cycles come from direct runs** (`run_program_synth --fast-beats`,
   `--json-out`), never from cached profiles. tok/s = fmax / cyc-per-token.
2. **"Byte-exact" alone is insufficient for concurrency changes.** The tiny
   cosim gate has zero DMA‖systolic co-busy cycles and is structurally blind
   — exactly how a Port-A bus bug silently corrupted overlapped matmuls for
   months while every gate stayed green. Overlap changes must additionally
   show: total cycles ↓, co-busy quoted and moving the right way, Port-A
   audit `lost=0`, and an unchanged RTL-vs-RTL logits sha1.
3. **At 124M, compare RTL to RTL.** RTL-vs-golden byte-match is ill-posed
   past the first fp16 overflow (the golden saturates); use logits SHA
   against a baseline RTL run, plus argmax/PPL conformance.
4. **Default bundles are byte-pinned** (`test_specdec_is_inert_at_the_default`)
   so opt-in features (spec-dec) provably don't perturb the default path;
   re-pins are deliberate, reviewed events.
5. The DRAM model is pinned: bandwidth scales with core clock
   (`--fast-beats`). Fixed-GB/s sensitivity via `--beat-interval`.

---

## Quality results — GPT-2 124M, 257-token PPL

| Preset | PPL | Notes |
|---|---|---|
| FP32 reference | **53.42** | Ceiling. NumPy reference, no quantization. |
| `weight_only_int8` | ~175 | Zero-calibration W8A16 baseline. |
| `weight_only_int8_quarot` | **56.23** | + data-free residual-stream rotation; the production preset (all perf numbers above run this). |
| `weight_only_int4_awq_gptq` | **63.04** | W4 blocks + W8 lm_head; AWQ α=0.40 + GPTQ. ~50% DRAM weight savings. |
| W4 + Tier-1 refinements | **~55.04** | + act-order GPTQ, 16 K calibration, AdaRound, bias correction (non-default kwargs, `w4_quant.py`). |

The RTL currently executes W8 bundles; W4 exists in golden/ISA
(`CONFIG_TILE` bit 28) but the RTL doesn't decode it (roadmap item G).

---

## Roadmap (condensed — full detail in `docs/perf_roadmap_2026-07-16.md`)

The machine is idle, slow-clocked, and narrow — in that order:

- **T1 — compiler overlap (occupancy), in progress.** Items 1+2 landed
  (+8.96% b16, +8.17% b1, byte-exact, zero RTL): KV V-prefetch inside packed
  attention groups and FC2 weight-prefetch double-buffering. Remaining: FC2
  input hoist, next-group K^T prefetch.
- **T2 — small RTL.** Instruction prefetch buffer (6.9% of b1 is
  fetch-stall), A-load reuse, drain/flush overlap.
- **T3 — the clock (the multiplier).** Single-domain 70–90 MHz: pipeline exp
  into the SFU (step 0 — primitive done, integration pending; also makes the
  current 34.41 MHz honest), deepen div/sqrt, split add/mul for the stretch.
  ×2.0–2.6 on everything under the pinned BW model. Full-chip PNR stamp
  needs a ≥24 GB box.
- **T4 — 32-byte AXI (conditional).** Only if occupancy+clock make DMA the
  binding floor. ISA/compiler are width-invisible.
- **Mined out** (don't revisit): batching beyond B=16, QK^T packing,
  elementwise SFU fusion, the helper K^T pass, SFU row-padding, preclears.
- **Hard floors:** b1 systolic 11.11M cyc (M=1: 15/16 of the mesh idle ⇒
  ~3.1 tok/s at this clock); KV capacity wall (12.3 MB/layer vs 384 KB
  SRAM) caps KV overlap at attention scope.

---

## Doc index

### Current state-of-truth

- [`docs/perf_roadmap_2026-07-16.md`](docs/perf_roadmap_2026-07-16.md) —
  **the current roadmap**: measured state, the honest re-base, T0–T4.
- [`software/docs/isa_generation_freeze.md`](software/docs/isa_generation_freeze.md)
  — normative ISA contract (+ dated §6 revisions).
- [`docs/porta_bus_split.md`](docs/porta_bus_split.md) — the Port-A/Port-S
  memory architecture and why it exists (the corruption fix).
- [`docs/t0_sfu_fmax_audit.md`](docs/t0_sfu_fmax_audit.md) — what actually
  binds fmax (exp/EXPSUM), the 34.41 MHz contingency, T3 step-0 status.
- [`rtl/TESTBENCHES.md`](rtl/TESTBENCHES.md) — RTL bench ownership.

### Landed-lever reports (dated records, each with measured gates)

- [`docs/t1_overlap_items.md`](docs/t1_overlap_items.md) — T1 items 1+2
  (KV V-prefetch, FC2 weight prefetch).
- [`docs/lever_d_dma_transpose.md`](docs/lever_d_dma_transpose.md),
  [`docs/lever_e_fmax_cluster.md`](docs/lever_e_fmax_cluster.md),
  [`docs/lever_h_b32.md`](docs/lever_h_b32.md),
  [`docs/lever_i_serving.md`](docs/lever_i_serving.md),
  [`docs/lever_b3_specdec.md`](docs/lever_b3_specdec.md) — DMA
  transpose-load, the fmax cluster, B=32 (negative result), serving
  (logits×N + chunked prefill), speculative decoding.
- [`docs/phase0_measurement.md`](docs/phase0_measurement.md) — the overlap
  corruption discovery and its bounds.
- [`docs/t1_measured_redirect.md`](docs/t1_measured_redirect.md) — the T1
  re-scope (read its outcome banner: two conclusions were later overturned).

### Historical (preserved as record, superseded for planning)

- [`docs/perf_roadmap_2026-07-10.md`](docs/perf_roadmap_2026-07-10.md)
  (levers A/C/B/D/E era), [`docs/perf_roadmap_2026-07-08.md`](docs/perf_roadmap_2026-07-08.md)
  (single-stream era) — pre-re-base numbers; see their banners.
- [`docs/accelerator_completion_review.md`](docs/accelerator_completion_review.md)
  — the 2026-05-19 gap analysis (its gaps have since closed; see banner).
- [`docs/llm_isa_plan.md`](docs/llm_isa_plan.md), `docs/rtl_plan.md`,
  `docs/rtl_debug_plan.md`, `docs/rtl_debugging_plan.md`,
  `docs/stage5_readiness_2026-04-22.md` — ISA/RTL bring-up planning.
- `rtl/synth/BASELINE.md`, `PHASE2_INTEGRATION.md`, `PHASE3_CLOSEOUT.md` —
  the synthesizability campaign (2026-05); cell counts predate the July RTL.
- [`software/CODEBASE.md`](software/CODEBASE.md) — ViT-era architecture
  overview; ISA mechanics and assembler/compiler internals remain accurate.

---

## Known open items

- **T3 step 0 integration**: `fp32_exp_p18` is verified but the SFU still
  elaborates combinational `fp32_exp` at three sites; until integrated, the
  34.41 MHz peg is contingent (cycle counts unaffected).
- **Full-chip/full-SFU PNR** is blocked on this class of machine (OOMs at
  15 GB; needs ≥24 GB). fmax evidence is per-block PNR + standalone
  primitive STA, calibrated (see `docs/t0_sfu_fmax_audit.md`).
- **Pre-existing test failures** (pinned set — fixture/environment debt, not
  product regressions): fixture SHA drift, `test_stage5_ptq_presets`
  `n_embd` KeyErrors, W4 `tile_config` tuple cases, fp16-embedding-era
  synthetic tests. Gate new work against a clean-HEAD baseline, not zero.
  (The golden-simulator SHA pin itself is green at the §6 m_exact revision.)
- **W4 on RTL** not implemented (golden/ISA only).
- **FPGA path** is a smoke-tested skeleton; no part chosen. The measured
  track is ASIC/sky130.

---

## Layout invariants worth knowing

- **The golden model is the conformance arbiter** on the tiny fixture; don't
  edit `taccel/golden_model/simulator.py` casually (SHA-pinned). At 124M the
  arbiter is RTL-vs-RTL SHA + argmax/PPL (golden saturates fp16).
- **The freeze cosim gate is the bright line**: commits touching
  `rtl/common/src/`, `taccel/isa/`, `taccel/golden_model/`, or the codegen
  path must keep it byte-identical.
- **Overlap/concurrency changes** carry the extra gate: co-busy + Port-A
  audit + RTL-vs-RTL SHA (see the doctrine section — this is not optional;
  the repo has the scar).
- **Byte-pinned defaults**: productized PPL presets and the spec-dec
  inertness pin are byte-identity gates; drift of 1e-4 PPL or one schedule
  byte is a real signal, and re-pins are deliberate events.
- **Box limits**: never two yosys jobs, or a yosys job + a 124M sim, or two
  124M sims, concurrently on a 15 GB machine.

---

## License

No license file is present in the tree; the code is © the repository owner.
Contact before depending on or redistributing.
