# LLM Accelerator

An LLM inference accelerator project: Python toolchain (ISA, compiler, quantizer,
golden model) plus SystemVerilog RTL, targeting GPT-2 124M end-to-end. The
target end-state is a working FPGA demo (tokens/sec on board); the present
state is a byte-exact behavioral RTL implementation with a synthesizable
datapath rewrite in progress.

The project started as an INT8 ViT accelerator (DeiT-tiny) and grew a decoder
ISA + GPT-2 frontend on top. Both lineages remain in the tree.

---

## Status snapshot

| Area | State |
|---|---|
| **ISA generation** | gen-2 frozen 2026-05-19 — 19 emitted opcodes, FP32 sub-layer ops `0x17–0x1F` (`0x1C` reserved). Contract: `software/docs/isa_generation_freeze.md`. |
| **Golden model** | Content-pinned (`taccel/golden_model/simulator.py`); SHA enforced by `test_frozen_golden_sha_pin`. |
| **RTL behavioral** | All 19 emitted opcodes implemented across fetch / decode / control / DMA / 16×16 systolic / SFU / helper. ~9 k LOC SV. |
| **Freeze cosim** | `test_compare_rtl_golden.py` — **6+1 byte-identical** vs golden on the tiny fixture (P6b + P6m). Closes freeze §5 definition-of-done. |
| **RTL synthesizability** | Whole-design `make synth-check` GREEN (yosys + sv2v, `-DSFU_SYNTH_NO_DPI`). Phase-3 close-out 2026-05-21: `real`/DPI removed from the SFU/helper datapaths; first synth fp32 primitive (`rtl/src/fp32/fp32_add.sv`) bit-exact vs DPI golden. **No FPGA part chosen yet — no fmax / area / tokens-per-sec number exists.** |
| **W8A16+QuaRot** | GPT-2 124M 257-tok = **56.23 PPL** (FP32 ceiling 53.42). Productized preset `weight_only_int8_quarot`. |
| **W4A16+AWQ+GPTQ** | GPT-2 124M 257-tok = **63.04 PPL**. Productized preset `weight_only_int4_awq_gptq` (W4 blocks + W8 lm_head; AWQ α=0.40 + GPTQ act-order). |
| **TurboQuant KV** | Reference-only Tier-1 KV quantization characterized at 257-tok. Quality-neutral at ~4 bits on the QuaRot base (~3.76× KV compression). |
| **Roadmap** | `docs/accelerator_completion_review.md` (2026-05-19). Phase-0 correctness DONE; Phase-2 synthesizable SFU STARTED (first brick landed). FPGA part / platform integration not yet started. |

---

## Repository layout

```
LLM_accelerator/
├── software/                       # Python toolchain + tests + tools
│   ├── taccel/
│   │   ├── isa/                    # Opcodes, instruction dataclasses, encoding
│   │   ├── assembler/              # Two-pass assembler + disassembler
│   │   ├── quantizer/              # W8 / W4, AWQ, GPTQ, QuaRot, SmoothQuant, TurboQuant, …
│   │   ├── compiler/               # IR, tiling, memory alloc, codegen, decoder bundle
│   │   ├── golden_model/           # Bit-accurate cycle-faithful Python simulator
│   │   └── runtime/                # PPL eval, calibration, fake-quant ref, host runner,
│   │                               # W4 productized path, prep/Hessian disk caches
│   ├── tests/                      # pytest gates (freeze cosim, byte-identity, E2E, …)
│   ├── tools/                      # CLIs (asm, disasm, run_golden, evaluate_gpt2_perplexity,
│   │                               # rtl_cosim, compare_rtl_golden, w8a8_ppl_gate, …)
│   ├── docs/                       # ISA freeze + supporting design notes (state-of-truth)
│   ├── run_gpt2.py / run_nanogpt.py / chat_gpt2.py    # Top-level scripts
│   ├── CODEBASE.md                 # Historical architecture overview (ViT-era; still
│   │                               # accurate for ISA mechanics + assembler/compiler)
│   └── requirements.txt
│
├── rtl/                            # SystemVerilog RTL + testbenches
│   ├── src/                        # taccel_top, fetch/decode/control/DMA, systolic,
│   │                               # sfu_engine, blocking_helper_engine, fp32/, memory/
│   ├── verilator/                  # Native C++ benches + run_program harness
│   ├── cocotb/                     # Python-driven ISA-visible benches
│   ├── synth/                      # yosys synth-check + Phase-3 baseline notes
│   └── TESTBENCHES.md
│
└── docs/                           # Long-form planning + review docs
    ├── accelerator_completion_review.md     # Current roadmap (2026-05-19)
    ├── llm_isa_plan.md                      # Historical ISA v1.1 plan
    ├── rtl_plan.md, rtl_debug_plan.md, rtl_debugging_plan.md   # Historical RTL planning
    └── stage5_readiness_2026-04-22.md       # Stage-5 baseline note
```

---

## Quickstart

### Environment

```sh
python -m venv .venv && source .venv/bin/activate
pip install -r software/requirements.txt
```

You'll also need a Verilator install (4.2+) for the RTL benches and yosys + sv2v
(via Homebrew on macOS) for the synth-check gate.

All invocations below assume you're running from the repo root. `pytest`
auto-discovers `software/tests/conftest.py` and adds `software/` to `sys.path`,
so no `PYTHONPATH` is needed for the test commands. Direct script invocations
that import `taccel.*` set `PYTHONPATH=software` explicitly.

### Run the gen-2 freeze cosim gate (6+1 byte-identical)

```sh
pytest software/tests/test_compare_rtl_golden.py -v
```

This builds the frozen `weight_only_int8_quarot` bundle, runs it end-to-end on
both the pinned golden and the RTL via Verilator, and asserts byte-identity on
prefill + decode logits. The opt-in 124M leg is gated by `PYTEST_124M=1`.

### Evaluate GPT-2 124M perplexity

```sh
PYTHONPATH=software python3 software/tools/evaluate_gpt2_perplexity.py \
    software/tests/fixtures/generated/gpt2_converted_nanogpt.pt \
    --tokenizer-dir software/tests/fixtures/generated/hf_gpt2 \
    --calibration-text <path-to-calib.txt> \
    --eval-text       <path-to-eval.txt> \
    --max-eval-tokens 257 \
    --ptq-preset      weight_only_int8_quarot
```

Available presets include `weight_only_int8`, `weight_only_int8_quarot`,
`weight_only_int4_awq_gptq`, and a stack of activation-aware variants — see
`software/taccel/runtime/stage5_ptq.py`.

### Run an RTL bench

```sh
make -C rtl/verilator test_sfu        # Native C++ SFU bench
make -C rtl/verilator run_program     # Native program runner (used by cosim)
make -C rtl/cocotb test_all SIM=verilator      # Python-driven ISA benches
```

Bench-by-bench guide: [`rtl/TESTBENCHES.md`](rtl/TESTBENCHES.md).

### Synthesizability gate

```sh
make -C rtl/verilator synth-check
```

Runs `sv2v -DSFU_SYNTH_NO_DPI` over the whole design, then `yosys -p
"hierarchy -check -top taccel_top; proc; opt -fast; check -assert; stat"`. The
gate is GREEN on `main` (see `rtl/synth/BASELINE.md`,
`rtl/synth/PHASE3_CLOSEOUT.md`); it returns non-zero on any `real`, DPI, system
task, or unbounded-loop drift.

### Full software test suite

```sh
pytest software/tests/ -n auto
```

Slow / underpowered tests are gated behind `PYTEST_SLOW=1` (notably the 257-tok
and 33-tok TurboQuant-KV perplexity checks). The big 124M leg of cosim is gated
behind `PYTEST_124M=1`.

---

## Quality results — GPT-2 124M, 257-token PPL

| Preset | PPL | Notes |
|---|---|---|
| FP32 reference | **53.42** | Ceiling. NumPy reference, no quantization. |
| `weight_only_int8` | ~175 | Zero-calibration W8A16 baseline. |
| `weight_only_int8_quarot` | **56.23** | + data-free residual-stream rotation; current W8 production preset. |
| `weight_only_int4_awq_gptq` | **63.04** | W4 blocks + W8 lm_head; AWQ α=0.40 (c_attn + c_fc + lm_head) + GPTQ. ~50 % DRAM weight savings vs W8. |
| W4 + Tier-1 refinements | **~55.04** | Same productized stack + GPTQ `act_order=True` + 16 K calibration + AdaRound + per-channel bias correction. Beats W8 baseline; non-default kwargs (see `software/taccel/runtime/w4_quant.py`). |

(Other preset combinations — output-aware scale searches, FC2 ladders, raw VADD
modes — are in `stage5_ptq.py`; the table above lists what's referenced by the
production gates.)

The W8A16+QuaRot productized number is the post-`eps`-bug-fix baseline; earlier
reports of 55.76 are pre-fix.

---

## Architecture brief

Eight discrete software layers, each with its own pytest gate:

1. **ISA** (`taccel/isa/`) — 32-slot opcode space, fixed 6-byte encoding.
   Frozen contract: `software/docs/isa_generation_freeze.md`.
2. **Assembler / disassembler** (`taccel/assembler/`) — two-pass, plus a
   `ProgramBinary` container; the cosim runner consumes a single-stream
   ProgramBinary, the GPT-2 host path uses a two-stream `ProgramBundle`.
3. **Quantizer** (`taccel/quantizer/`) — per-channel symmetric W8 / W4, AWQ
   activation-aware scaling, GPTQ Hessian descent (with optional act-order +
   precomputed Hessian/gram inputs), QuaRot data-free rotation, SmoothQuant,
   AdaRound, TurboQuant KV, LN-fold, bias correction.
4. **Compiler** (`taccel/compiler/`) — IR + frontend registry (`ModelConfig`
   for nanoGPT / GPT-2 / DeiT), tile schedules, memory allocator with eviction,
   codegen (now split into `emit/` and `w8a16_emit/` subpackages), decoder
   bundle with runtime patch sites.
5. **Golden model** (`taccel/golden_model/`) — cycle-faithful Python simulator.
   Content-pinned (`simulator.py` blob `131d3ef1…`); the SHA pin is the
   freeze § 6 mechanism for catching golden drift.
6. **Runtime / PPL** (`taccel/runtime/`) — PPL evaluator, calibration adapters,
   W4 productized path (`w4_quant.py`), host runner, fake-quant reference,
   disk caches (`_prep_cache.py` for the full `Prepared`,
   `_hg_cache.py` for the per-source Hessians inside the W4 stack).
7. **Tools** (`software/tools/`) — CLIs for assemble/disassemble, evaluate
   perplexity, drive RTL cosim, generate gen-2 fixtures, characterize W8A8
   gates, and so on.
8. **Tests** (`software/tests/`) — ~125+ pytest cases covering encoding
   round-trips, quantizer parity, codegen byte-identity, E2E PPL gates,
   freeze cosim, ISA encoding, byte-identity invariants for productized presets.

RTL stack (`rtl/src/`):

- **`taccel_top.sv`** — top-level glue (start/done/fault + AXI4 ideal slave).
- **`fetch_unit.sv`, `decode_unit.sv`, `control_unit.sv`** — front-end and
  control plane.
- **`dma_engine.sv`** — DMA with the BUG1 (`dma_busy` gating) fix.
- **`systolic/`** — 16×16 INT8 array + controller (per-tile `clear_acc` +
  DRAIN RMW fix for the latent `flags=1`-multitile bug).
- **`sfu_engine.sv`** + sibling `.svh` partitions
  (`sfu_dpi_helpers / sfu_synth_datapath / sfu_g2_compute`) — gen-2 SFU.
- **`blocking_helper_engine.sv`** — gen-1 helper ops (still RTL-legal per
  freeze §3).
- **`fp32/`** — synthesizable IEEE-754 primitives (`fp32_add.sv` first brick;
  the rest of the swap from DPI-C to synth RTL is the Phase-2 long pole).

End-to-end data flow: fixture → quantizer rewrite + calibration → compiler
emit → ProgramBundle → either the golden model (`HostRunner`) or the RTL
(`compare_rtl_golden.py` → `run_program.cpp` → Verilator). The freeze cosim
gate runs both legs side-by-side and asserts byte-identity (with characterized
per-op ULP bands per freeze §7).

For deeper architectural detail, see [`software/CODEBASE.md`](software/CODEBASE.md)
(ViT-era reference; ISA mechanics + hardware model + assembler/compiler
internals are still accurate) and
[`docs/accelerator_completion_review.md`](docs/accelerator_completion_review.md)
(current roadmap with the Phase 0–4 plan).

---

## Roadmap

**Phase 0 — Correctness foundation** *(done)*. gen-2 ISA freeze locked,
golden SHA-pinned, RTL byte-exact on tiny fixture, BUG1 (`#108`) and BUG2
(`#115`) fixed, flags=1 latent bug (`#116`) fixed, ACCUM-snapshot capture
(`#114`) fixed, logits-metric leg of `#109` landed.

**Phase 1 — RTL correctness generality** *(done)*. Folded into Phase 0 once
`#116` and `#114` landed.

**Phase 2 — Synthesizable datapath** *(in progress)*. First synth fp32
primitive (`fp32_add.sv`) bit-exact vs DPI golden; whole-design synth-check
GREEN; all transcendentals (`exp`, `gelu_new`, `softmax`, `layernorm`) routed
through synthesizable sub-FSMs with characterized ULP bands per freeze §7.
Remaining: synth fp32_mul / fp16↔fp32 cvt, then swap the non-transcendental
gen-2 ops to 0-ULP synth paths.

**Phase 3 — FPGA platform integration** *(not started)*. Memory controller
(MIG / EMIF), ABUF/WBUF/ACCUM → BRAM/URAM mapping, host link (PCIe / XDMA or
UART / JTAG), synthesizable top wrapper. **Gated on FPGA part / board choice**,
which is not yet made.

**Phase 4 — On-hardware validation + performance** *(not started)*. No fmax,
tokens/sec, or area number exists yet. Establishing those is part of Phase 3.

---

## Recurring development gates

| Gate | Command | Purpose |
|---|---|---|
| Freeze cosim | `pytest software/tests/test_compare_rtl_golden.py` | gen-2 freeze §5 definition-of-done. Must be 6+1 byte-identical. |
| Synth-check | `make -C rtl/verilator synth-check` | RTL elaborates with zero `real` / DPI / system-tasks / unbounded-loops. |
| Productized PPL | `pytest software/tests/test_stage5_ptq_presets.py` | W8A16+QuaRot 56.2256 + W4A16+AWQ+GPTQ 63.0353 byte-identical to recorded baseline. |
| Encoding round-trip | `pytest software/tests/test_isa_encoding.py` | All instruction formats encode/decode losslessly. |
| RTL unit benches | `make -C rtl/verilator test_sfu test_helpers test_systolic ...` | Per-engine native C++ benches. |
| cocotb ISA benches | `make -C rtl/cocotb test_all SIM=verilator` | Python-driven ISA-visible flow tests. |

`PYTEST_SLOW=1` enables long-running PPL gates; `PYTEST_124M=1` enables the
opt-in 124M leg of cosim.

---

## Doc index

### Current state-of-truth

- [`software/docs/isa_generation_freeze.md`](software/docs/isa_generation_freeze.md)
  — normative ISA contract. The RTL implements **exactly** the opcode set
  listed here.
- [`docs/accelerator_completion_review.md`](docs/accelerator_completion_review.md)
  — FPGA-demo roadmap (Phase 0 → 4) with the gap analysis and risk register.
- [`rtl/synth/BASELINE.md`](rtl/synth/BASELINE.md) — synth-check gate
  definition + how it went RED → GREEN.
- [`rtl/synth/PHASE3_CLOSEOUT.md`](rtl/synth/PHASE3_CLOSEOUT.md) — Phase-3
  closeout ledger: `real`/DPI removal, fp32_exp / gelu_new tightening,
  synth sub-FSM cascade.
- [`rtl/TESTBENCHES.md`](rtl/TESTBENCHES.md) — RTL bench ownership and
  required shape for new benches.

### Architecture reference

- [`software/CODEBASE.md`](software/CODEBASE.md) — historical architecture
  overview. ISA mechanics, hardware model, assembler/compiler/quantizer/
  golden-model internals are still accurate; the GPT-2-era runtime / W4 /
  freeze content is **not** covered here (lives in this README + the
  freeze doc).

### Historical planning (preserved as record, not state-of-truth)

- [`docs/llm_isa_plan.md`](docs/llm_isa_plan.md) — ISA v1.1 plan that grew the
  decoder ops on top of the ViT base. Superseded for the normative spec by
  the freeze doc.
- [`docs/rtl_plan.md`](docs/rtl_plan.md),
  [`docs/rtl_debug_plan.md`](docs/rtl_debug_plan.md),
  [`docs/rtl_debugging_plan.md`](docs/rtl_debugging_plan.md) — RTL bring-up
  and debugging plans from the gen-1 era.
- [`docs/stage5_readiness_2026-04-22.md`](docs/stage5_readiness_2026-04-22.md)
  — pre-freeze Stage-5 readiness baseline.

---

## Known open items

- **`#109` — 257-tok 124M logits conformance.** Logits-metric scaffold landed;
  the full 257-tok 124M leg of cosim is gated behind `PYTEST_124M=1` and runs
  under the characterized real-data `layernorm_fp32` ≤1-ULP band (same
  discipline as `gelu_new` ≤3-ULP, freeze §7).
- **`test_frozen_golden_sha_pin` debt.** The simulator `tile_config` was
  extended from a 3-tuple to a 4-tuple (W4 weight-dtype dispatch) in the W4A16
  Phase-2 codegen+golden landing; the freeze §6 SHA pin owns the rebake and is
  currently expected-fail until rebaked.
- **`test_stage5_ptq_presets.py` — 4 `KeyError: 'n_embd'` cases.** Pre-existing,
  independent of the W4 / iteration-speed work; tracked.
- **No FPGA part chosen.** Gates Phase 3 (synthesis toolchain, memory
  controller, SRAM mapping) and Phase 4 (fmax, tokens/sec).

---

## Layout invariants worth knowing

- The golden model is **the conformance arbiter**, not the RTL — the RTL is
  measured against the golden via the freeze cosim gate. Don't edit
  `taccel/golden_model/simulator.py` casually; the SHA pin (`test_frozen_golden_sha_pin`)
  will fire on any drift.
- The W8A16+QuaRot and W4A16+AWQ+GPTQ productized presets are pinned by
  byte-identity gates against recorded baselines. PPL drift of even 1e-4 is
  a real signal — usually a numerical reduction-order change.
- The cosim test `test_compare_rtl_golden.py` is the bright line for the
  gen-2 freeze. Any commit touching `rtl/src/`, `taccel/isa/`,
  `taccel/golden_model/`, or `taccel/compiler/{codegen,decoder_bundle}.py`
  must keep this gate 6+1 byte-identical.
- All commands run from the repo root. `pytest` picks up
  `software/tests/conftest.py` and adds `software/` to `sys.path` on its own,
  so no `PYTHONPATH` is needed for the test suite. Direct script invocations
  that need to import `taccel.*` use `PYTHONPATH=software` explicitly.

---

## License

No license file is present in the tree; the code is © the repository owner.
Contact before depending on or redistributing.
