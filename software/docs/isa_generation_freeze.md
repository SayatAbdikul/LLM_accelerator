# ISA Generation Freeze — gen-2 (W8A32 Phase 3 (c.1))

**Status: FREEZE DECISION LOCKED (2026-05-15). Becomes the effective repo
baseline on commit of the §5 dependencies** — the generation decision is
fixed and final; the repo is not yet the frozen baseline only because the
spec files in §5 are still uncommitted.

Frozen generation: **gen-2** — the FP32 sub-layer / dynamic-per-matmul-scale
generation introduced by W8A32 Phase 3 **(c.1)**, **including the in-flight
M2.5-A additions** (0x1E DEQUANT_ACCUM_FP32_SCALED, 0x1F MAX_ABS_REDUCE_FP32)
currently uncommitted in the §5 files.

This document is the normative ISA contract the RTL must implement. Until a
superseding freeze revision is published, the opcode set below is fixed; any
ISA change requires a new dated revision of this file.

## 1. Decision record

- **What:** lock the ISA to gen-2 (FP32 sub-layer ops 0x17–0x1F + the
  still-emitted gen-1 infrastructure ops). Do **not** revert to gen-1.
- **Basis:** `w8a32_deployment_scope.md` recommended option **(c.1)**;
  commits `47141fb` (Phase 3 (c.1) M1: ISA extension + simulator dispatch)
  and `5babbaa` (M2: codegen lowering for sub-layer ops) implement it; the
  toolchain and golden model already emit/dispatch gen-2. User confirmed
  "Lock gen-2 (c.1)" on 2026-05-15.
- **Why not gen-1:** reverting would undo committed Phase 3 (c.1) work and
  risk the activation-range accuracy the FP32 sub-layer path was added for
  (dynamic per-matmul scaling, `MAX_ABS_REDUCE_FP32`).

## 2. Normative opcode set (RTL MUST implement exactly these)

Ground truth = empirical histogram of a compiled GPT-2 124M bundle
(prefill + decode), **both** `weight_only_int8` and
`weight_only_int8_quarot` presets — **byte-identical** (QuaRot is data-free
weight prep, zero ISA surface). 19 opcodes are emitted:

| Opcode | Name | Gen | RTL today |
|---|---|---|---|
| 0x01 | HALT | infra | implemented |
| 0x02 | SYNC | infra | implemented |
| 0x03 | CONFIG_TILE | infra | implemented |
| 0x04 | SET_SCALE | infra | implemented |
| 0x05 | SET_ADDR_LO | infra | implemented |
| 0x06 | SET_ADDR_HI | infra | implemented |
| 0x07 | LOAD | infra | implemented |
| 0x08 | STORE | infra | implemented |
| 0x09 | BUF_COPY | infra | implemented |
| 0x0A | MATMUL | infra | implemented |
| 0x14 | CONFIG_ATTN | infra | implemented |
| 0x17 | DEQUANT_ACCUM_FP32 | **gen-2** | **MISSING — reserved/illegal** |
| 0x18 | QUANT_FP32_INT8 | **gen-2** | **MISSING — reserved/illegal** |
| 0x19 | VADD_FP32 | **gen-2** | **MISSING — reserved/illegal** |
| 0x1A | LAYERNORM_FP32 | **gen-2** | **MISSING — reserved/illegal** |
| 0x1B | GELU_FP32 | **gen-2** | **MISSING — reserved/illegal** |
| 0x1D | MASKED_SOFTMAX_FP32 | **gen-2** | **MISSING — reserved/illegal** |
| 0x1E | DEQUANT_ACCUM_FP32_SCALED | **gen-2** | **MISSING — reserved/illegal** |
| 0x1F | MAX_ABS_REDUCE_FP32 | **gen-2** | **MISSING — reserved/illegal** |

Per-bundle emission counts (prefill, GPT-2 124M, indicative of hotness):
SYNC 11765, SET_ADDR_LO/HI 11061, BUF_COPY 8977, LOAD 6656, STORE 4405,
CONFIG_TILE 3000, MATMUL 1254, QUANT_FP32_INT8 1177, DEQUANT_ACCUM_FP32_SCALED
771, MAX_ABS_REDUCE_FP32 601, SET_SCALE 576, DEQUANT_ACCUM_FP32 288,
CONFIG_ATTN 144, MASKED_SOFTMAX_FP32 144, VADD_FP32 145, GELU_FP32 36,
LAYERNORM_FP32 25, HALT 1.

## 3. Non-normative for the gen-2 RTL target

These 13 opcodes are **not emitted** by the gen-2 toolchain. The opcode enum,
assemblers (`isa/instructions.py`), and golden model keep them for
back-compat / non-causal models, but **the gen-2 RTL is NOT required to
implement them** and MAY treat them as illegal:

- **Superseded gen-1 sub-layer ops** (replaced by their FP32 analogue):
  0x0B REQUANT, 0x0C SCALE_MUL, 0x0D VADD (→0x19), 0x0F LAYERNORM (→0x1A),
  0x10 GELU (→0x1B), 0x11 REQUANT_PC, 0x13 DEQUANT_ADD,
  0x15 MASKED_SOFTMAX (→0x1D).
- **Non-causal only** (no current frontend; GPT-2 is causal):
  0x0E SOFTMAX, 0x12 SOFTMAX_ATTNV, 0x1C SOFTMAX_FP32.
- **Unused fused-causal variant** (toolchain chose unfused 0x1D + separate
  attn·V matmul; no FP32 fused analogue defined): 0x16 MASKED_SOFTMAX_ATTNV.
- **Retained no-op:** 0x00 NOP — keep as a defined decode-to-no-op
  (conventional, zero RTL cost); not emitted but not deprecated.

Note 0x1C SOFTMAX_FP32 is gen-2-numbered but **non-normative** (no causal-LM
consumer) — RTL need not implement it; the normative gen-2 additions are
0x17–0x1B and 0x1D–0x1F (**8 opcodes**, not 9).

## 4. RTL reconciliation requirement (the blocking work item)

The RTL implements gen-1 (0x00–0x16); `taccel_pkg.sv:45` and
`decode_unit.sv:32` declare **0x17–0x1F = "reserved — illegal instruction
fault"**. A current bundle therefore illegal-faults on the first gen-2 op.
To make golden-vs-RTL cosim possible on the production path:

1. **Implement the 8 normative gen-2 opcodes in RTL:** 0x17, 0x18, 0x19,
   0x1A, 0x1B, 0x1D, 0x1E, 0x1F. Remove them from the reserved/illegal
   range in `taccel_pkg.sv` / `decode_unit.sv`. (0x1C stays reserved.)
2. **Validate the 11 already-implemented normative ops** (§2 "implemented")
   in cosim — no new RTL, but they must pass against the gen-2 golden model.
3. The SFU "all internal ops FP32" contract (`sfu_engine.sv:18-19`) is
   consistent with gen-2; the gen-2 ops are FP32-I/O, so the change is the
   instruction-level decode/datapath, not the internal precision model.
4. Area note (unmeasured): the gen-2 FP32 sub-layer datapath is expected to
   cost more SFU area than the gen-1 ops it supersedes. No synthesis number
   exists. This is acknowledged, not blocking the freeze, but should be
   measured before RTL sign-off.
5. **Conformance / definition of done.** Gen-2 RTL is conformant when
   `software/tests/test_compare_rtl_golden.py` passes an **end-to-end**
   (not block-level) byte-match within FP16 ULP on the GPT-2 W8A16
   teacher-forced reference bundle (`weight_only_int8_quarot`, 257-tok)
   against the pinned golden model below. Block-level cosim greens are
   necessary but **not** sufficient — the freeze is not satisfied until the
   end-to-end bundle byte-matches.
   **Status (2026-05-16, P6b) — substantive property GREEN; literal §5 bar
   NOT yet met (two explicit gaps, named below).** The end-to-end
   RTL-vs-golden gate is built and green:
   `test_compare_rtl_golden.py::test_rtl_cosim_gen2_byte_match[0,5]` runs
   the *real* compiled bundle on Verilator `run_program` and byte-matches
   the pinned golden across all 71 captured gen-1+gen-2 nodes (LayerNorm/
   GELU/residual/dequant/quant/masked-softmax/max-abs-reduce/attn) at
   **0 fp16 ULP on every op-class incl. `gelu_new`** — strictly stronger
   than the §7 ≤3 gelu band (e2e fp16 rounding collapses the sub-ULP tanh
   delta, same mechanism §7 noted for masked-softmax); RTL run clean
   (`status=halted`, `fault=False`, `forbidden_overlap=False`).
   **Model & scope actually exercised (do not over-read):** the *tiny
   2-layer nanoGPT shakespeare-char fixture* (`tools/train_tiny_fixture.py`
   `DEFAULT_FIXTURE`, d128/l2; ~3114-insn prefill, ~5 s/run) compiled with
   the `weight_only_int8_quarot` **preset**, **single-token PREFILL only**.
   This is meaningful evidence — it is the *same gen-2 ISA* the GPT-2 W8A16
   bundle emits, exercised through the real compiler/codegen/SYNC path — so
   the gen-2 **datapath** is conformant. But the **literal §5 bar is not
   met on two axes**: (1) **model size** — tiny d128/l2 fixture, *not*
   GPT-2 124M (12-layer); (2) **sequence** — single-token prefill, *not*
   257-tok prefill+decode (the decode stream needs PC-rebased trace
   manifests + per-step kv/attn runtime patching; prefill_pc=0 made prefill
   rebase-free). Closing both is tracked as **P6c / task #106**; the freeze
   §5 definition-of-done remains formally open until the GPT-2 124M 257-tok
   bundle byte-matches.
   **Status (2026-05-17, P6c — GPT-2 124M single-tok prefill:
   byte-EXACT to golden up to the first fp16 overflow; one isolated bug;
   §5 still open).** Headline: on the real **GPT-2 124M**
   `weight_only_int8_quarot` bundle (single-token prefill), **RTL is
   byte-EXACT (0 fp16 ULP) to the pinned golden across the entire prefill
   for every captured node before golden's first fp16 overflow** — the
   only exception being **one localized bug (`block0_head0_query`,
   propagating to its direct child `block0_head0_qkt`)**. RTL runs clean
   (`status=halted`, `fault=False`, `forbidden_overlap=False`;
   `run_program` rebuilt with `DRAM_SIZE≥1<<30` — the 16 MB default
   `FAULT_DRAM_OOB`s on the 392 MB image). Findings (advisor-reviewed,
   primary-source): **(a) well-posed boundary** — golden first overflows
   fp16 at `block0_out_proj` (pc 2298, 48/768 → ±65504/NaN; the W8A16
   storage format genuinely saturates at 124M MLP dynamic range, model
   still 55.76 PPL). The `block0_out_proj` *trace snapshot* matches
   byte-identically at its capture point, but that snapshot is a
   non-functional tile: it is requantized → stored → reloaded (int8)
   before the residual path consumes it, and the **functional** out_proj
   is that reloaded int8 representation — which is exactly where BUG2
   enters (P6d ground truth, #107). The full pre-boundary prefill is
   0 ULP modulo the one bug above. **(b) BUG1** —
   `block0_head0_query` (first Q-projection, head-0 only; heads 1–11 all
   0 ULP) miscomputes; architecturally inert at seq=1 (`softmax(1,1)`).
   **P6e (2026-05-17) CONFIRMED the root cause** via a chain of 8
   fidelity-gated / control-validated refutations (input/path,
   int8/scale_regs[0], bias/0x1E-src2, scalar scale_regs[1],
   wrong-weight-swap) culminating in Step3c: a non-perturbing
   `cg._record_trace_event` probe of the *loaded* weight in WBUF under
   the canonical cosim — ALL controls 0-ULP including `head0_key__wprobe`
   byte-exact (validates the WBUF-snapshot instrument), while
   **`head0_query__wprobe` is 48569/49152 bytes WRONG golden-vs-RTL**.
   ⇒ **BUG1 = the FIRST matmul-weight DMA-load into WBUF is wrong in
   RTL** (golden loads the same staged Wq correctly — same instruction
   stream; golden reproduces head0_query at fid 1.9e-4 with the correct
   Wq). The systolic array then computes correct_int8 @ WRONG_Wq →
   the observed head0_query divergence. Scope: specifically the first
   matmul-weight WBUF load — NOT generic first-WBUF (ln1 γ/β WBUF loads
   ran earlier, 0-ULP), NOT large-load (head0_key Wk: same 49152 B /
   same SET_ADDR+LOAD+SYNC1 structure, byte-EXACT). **Root mechanism
   CONFIRMED BIT-EXACT (Step4/5, direct observation):** RTL's wrong
   head0_query Wq == golden DRAM[`0x14c12ca0`] at 93.76% vs 1.19% at the
   intended `0x4c12ca0`; addr-reg trace shows golden's `pc=51 LOAD` reads
   `0x4c12ca0` (post-`SET_ADDR_HI`=0) while RTL reads `0x14c12ca0`
   (pre-HI, stale HI=1). ⇒ **BUG1 = a `SET_ADDR_HI`(pc=50)→`LOAD`(pc=51)
   address-register read-after-write hazard** — the consuming LOAD
   samples the addr reg before the immediately-preceding `SET_ADDR_HI`
   write is visible → stale HI=1 → wrong DRAM source → wrong Wq → wrong
   matmul. (The earlier "port-A mux steal" fix-class was **refuted**: a
   DMA back-pressure fix was a provable no-op — head0_query unchanged,
   controls 0-ULP, no port-A contention; reverted, run_program rebuilt
   clean. Lesson: never patch on inference — confirm the mechanism
   directly.) seq=1-inert ⇒ decode/#109 item, NOT a §5 prefill-byte-match
   blocker (freeze gate stays 4/4). FIX (scoped, HIGH-RISK shared
   control-unit/addr-regfile): the LOAD/STORE consuming an addr reg must
   see the preceding `SET_ADDR` write — control-unit dispatch stall until
   commit / addr-regfile bypass / registered-after-commit base_addr.
   MUST first build a bit-exact predicted-buggy unit repro
   (`SET_ADDR_LO;SET_ADDR_HI;LOAD` ⇒ stale-HI; 2nd LOAD = known-good
   control) to pin the exact race + confirm BEFORE patching, then
   canonical 124M cosim + freeze gate 4/4 + suites. Task **#108 (P6e)**.
   **P6l (2026-05-18, #108 DONE). BUG1 FIXED — corrected mechanism.**
   The freeze doc's earlier "register-file RAW hazard" framing (stale HI=1
   from `SET_ADDR_HI`→`LOAD`) was an incorrect mechanism attribution:
   `SET_ADDR_HI`'s NBA register write commits at the posedge that ends its
   S_ISSUE cycle; the S_FETCH stage plus AXI fetch latency before the next
   `LOAD` reaches S_ISSUE spans ≥2 posedges — the write is fully visible. The "RTL reads 0x14c12ca0 (stale HI=1)" was a correct WBUF
   content observation (the stale data *in WBUF* matched that address) but
   a wrong mechanism attribution (no AXI read was issued to 0x14c12ca0
   from pc=51; the DMA read was silently dropped, explained below). The
   simple `SET_ADDR_HI`→`LOAD` unit reproducers never triggered the bug
   because they had no concurrent DMA. **Confirmed root cause:**
   `dma_dispatch = !sfu_busy` in `control_unit.sv` lacked a `!dma_busy`
   guard. When a STORE DMA is active, the AXI write channel is independent
   of `rd_inflight_q`, so instruction fetches proceed freely. A subsequent
   LOAD reaches S_ISSUE with `dma_busy=1`, fires `dma_dispatch=1`, but the
   DMA engine (in D_STORE_W, not D_IDLE) silently ignores the pulse. The
   control unit advances; the Wq WBUF load never executes; WBUF retains the
   stale Wk from the prior LOAD (which had used addr=0x14c12ca0, HI=1).
   **Fix (`control_unit.sv`):** (i) combinational: `OP_LOAD, OP_STORE:
   dma_dispatch = !sfu_busy && !dma_busy`; (ii) sequential S_ISSUE stall:
   `if (sfu_busy || dma_busy) state <= S_ISSUE`. **ISA-contract change:**
   LOAD/STORE now serialize through DMA idle. Previously they were
   fire-and-forget (SYNC was the sole ordering point); a program issuing
   LOAD while a prior STORE-DMA was still in-flight silently dropped the
   LOAD. Programs that correctly bracket every LOAD/STORE with SYNC are
   unaffected; this fix makes the ISA contract explicit and enforced in HW.
   **Reproducer (bit-exact):** `rtl/verilator/test_addr_raw_hazard.cpp`
   `test_dispatch_drop_via_store` — pre-fix: ABUF=0xBB (stale, second LOAD
   dropped), FAIL; post-fix: ABUF=0xAA (correct), PASS. Full suite 6/6.
   **Verified:** 124M cosim post-fix → `block0_head0_query` 0 fp16 ULP
   (was 490 ULP); all controls 0-ULP; freeze gate 4/4
   (`test_compare_rtl_golden.py`). #108 DONE. The decode path + logits
   metric are tracked as **#109 (P6f)**.
   **(c) BUG2** — past the overflow boundary,
   non-finite-operand handling in the requant path diverges (golden
   `np.clip`/cast-NaN vs RTL); a localized op-semantics edge for
   non-finite inputs, *not* a finite-path datapath error. Task **#107
   (P6d)**. **No freeze edit is shipped:** per §7 discipline, a
   logits-level / characterized 124M conformance metric (per-tensor
   byte-match is well-posed only up to the overflow boundary) must be
   *proposed with its supporting measurement* before §5/§7 are amended —
   tracked as **#109 (P6f)**, which also carries the 257-tok decode
   integration. §5 definition-of-done remains formally open.
6. **Pinned reference golden.** RTL conformance is measured against
   `software/taccel/golden_model/simulator.py` **at the commit created by
   the §5 commit**: `frozen_golden_sha = aa9a9c0fa389d77598acfe68f4ac1347bd9fc9ef`
   (recorded 2026-05-16; §5 spec files committed/clean). Golden fixtures
   under `rtl/verilator/fixtures/gen2/` are pinned to this SHA via
   `software/tools/gen_gen2_fixtures.py`. Any later `simulator.py` change
   requires a new freeze revision + fixture regen — otherwise "cosim vs the
   gen-2 golden" is a moving target, the exact failure this freeze closes.
   **Content-level pin (enforced, not just documented).** The commit SHA
   above does not detect a `simulator.py` edit made *after* `aa9a9c0`
   within the same or a later commit, so the pin is enforced at the
   content level: the SHA-1 git-blob hash of
   `software/taccel/golden_model/simulator.py` is
   `131d3ef1a6009519976cf99baf9157a434e67f6f` (**freeze §6 REVISION
   2026-05-17, P6g / Option B, task #110**: the 0x18 `QUANT_FP32_INT8`
   non-finite requant contract was made explicit per §7 item 8 Option B —
   NaN→0, ±inf→±127/−128, finite-overflow→saturate. Numerically identical
   to the prior `np.clip(...).astype(np.int8)` on the pinned numpy: all
   gen-2 `.raw` fixtures regenerated **byte-identical**, only the
   `meta.json` sha-stamp changed. Prior pin
   `7746e65598961ac8430f8eeece45d7ec976584cd` (pre-P6g, blob at `aa9a9c0`).
   New commit SHA is set by the user on external commit; this blob hash is
   the authoritative enforced pin and is what the test recomputes).
   `software/tests/test_compare_rtl_golden.py::test_frozen_golden_sha_pin`
   recomputes this blob hash and **fails loud** on any drift — the gen-2
   conformance gate refuses to run a comparison against an unpinned
   golden, closing the moving-target hole even inside an unchanged commit.
7. **Conformance tolerance — per op-class (REVISION 2026-05-16).** The
   original "byte-match within FP16 ULP" (item 5) left the band a single
   guessed number (≤1). Measurement on the P2 ops replaces the guess:
   - **Non-transcendental ops → exactly 0 fp16 ULP (bit-exact).** Verified:
     `VADD_FP32` (0x19) and `LAYERNORM_FP32` (0x1A, eps=1e-5, γβ, mean/var)
     are byte-identical to the golden fixtures. This is mandatory; any
     drift is a real bug. Applies to 0x17/0x18/0x19/0x1A/0x1E.
   - **Transcendental-heavy ops → small empirically-characterized band.**
     `GELU_FP32` (0x1B `gelu_new`) is byte-identical to a libm-`tanhf`
     implementation of the exact golden formula; it differs from the
     golden's `np.tanh` (numpy vectorized float32 tanh) by **≤3 fp16 ULP
     on 53/1024 elements** (measured; `/tmp/verify_gelu.py` reproduces it:
     RTL ≡ libm-gelu exactly, libm-vs-golden ≡ RTL-vs-golden = (53, 3)).
     The datapath is correct — the residual is purely a float32-tanh
     library difference. Conformance band for `gelu_new`: **|ulp| ≤ 3**.
   - `MASKED_SOFTMAX_FP32` (0x1D, `exp`) — **measured P4: BIT-EXACT
     (0 ULP)** on the characterized fixture (uniform(-8,8), qrb=0,
     valid_kv_len=64). Unlike `gelu_new`, the `row_max` subtraction +
     softmax normalization + final FP16 round collapse the numpy-vs-libm
     `expf` differences. Conformance band for `masked_softmax_fp32`:
     **0 ULP** on the fixture. Caveat: this is one input distribution;
     the end-to-end bundle (P6) over real activations is the final
     arbiter — if real-data `exp` drift appears there it gets its own
     characterized band, same discipline. Band is fixture-asserted, not
     a universal claim.
   - Bit-replicating numpy's tanh is rejected: numpy-version-fragile and
     the freeze already pins a golden SHA. Characterizing the band per
     op-class is the disciplined resolution the §6 revision mechanism is
     for. `expect_fp16_ulp` enforces these bands per op in `test_sfu.cpp`.
8. **Non-finite requant contract — Option B CHOSEN & IMPLEMENTED
   (2026-05-17, user pick; P6g / #110).** At
   GPT-2 124M the W8A16 fp16 storage genuinely overflows in the MLP/attn
   path (golden too). The requant op (`QUANT_FP32_INT8` 0x18 / the
   `DEQUANT_ACCUM_*` int8-clamp paths — same idiom
   `np.clip(np.round(x·s),-128,127).astype(np.int8)`) then has to map a
   **non-finite** fp32 to int8. **Golden, measured (numpy 1.26.4):** the
   `np.clip(…,-128,127)` saturates finite-overflow and ±inf → ±127/−128;
   **NaN passes the clip and `NaN.astype(int8)` → 0**. RTL's 0x18
   datapath does *not* reproduce this (hardware clip comparisons are
   all-false for NaN → garbage int8, not 0). This is BUG2. Pick one
   (each is mechanical; A and B converge on identical code/RTL, differ
   only in how §7 reads; C changes the rule instead of the RTL):
   - **Option A — pin the (current) golden semantics.** Make golden
     explicit: `np.where(np.isfinite(x), np.clip(np.round(x·s),-128,127),
     0).astype(np.int8)` (NaN→0, ±inf/overflow→saturate), numpy-version
     independent; specify it in this §; RTL implements NaN→0 / saturate.
   - **Option B — choose a hw-sane contract from scratch.** §7 specifies
     NaN→0, ±inf→±127, overflow→saturate; golden enforced explicitly
     (same patch as A); RTL implements it. Identical end state to A; only
     this paragraph's framing differs (B = "designed", A = "pins numpy").
   - **Option C — don't fix in RTL; characterize at logits.** Declare the
     §5 well-posed region = "up to the first golden non-finite tensor";
     past it, conformance is the logits-level metric (#109 / P6f). RTL's
     non-finite handling is allowed to differ. The gelu-band move applied
     to overflow; no RTL/golden code change, the metric moves instead.
   Same §6 revision discipline as the §7 bands: this is a *characterized
   contract decision*, not a silent relax.
   **Decision & implementation (P6g / #110).** User chose **Option B**
   (FPGA-deployment rationale: deterministic, explicitly-specified,
   version-independent — A's "pin numpy `NaN.astype`" is the same
   fragility the freeze rejected for gelu; C alone leaves silicon
   behavior unspecified in a reachable regime). Implemented & staged:
   golden `_exec_quant_fp32_int8` (:856) made explicit
   `np.where(np.isnan(scaled), 0, np.clip(scaled,-128,127)).astype(int8)`
   (NaN→0, ±inf/overflow→saturate); the synthesizable SV
   `quantize_to_i8` (`sfu_engine.sv`) gets NaN→0 / ±inf→±127 guards;
   the DPI `sfu_fp32_quantize_i8` kept consistent. **Behavior-preserving
   on the pinned numpy** — all gen-2 `.raw` fixtures regenerated
   byte-identical (only the `meta.json` sha-stamp changed); content pin
   re-pinned (§6 revision, blob `131d3ef1…`). The contract is correct
   and FPGA-deployable on its own merits.
   **Empirical caveat (do not over-read):** implementing B did **not**
   move the GPT-2 124M cosim boundary (`block0_residual1` 18188 ULP,
   gNaN=rNaN=0 finite-vs-finite). **B is not BUG2's cause** (P6g:
   provably 0 non-finite over 1177 functional 0x18 calls). **BUG2's
   root is still OPEN — two roots refuted, each caught pre-fix by
   instrument discipline:** (i) P6d "0x18 non-finite" → refuted by P6g
   (zero cosim change); (ii) P6h "systolic MATMUL deep-K" → refuted by
   P6i: the bisect relied on an **ACCUM-via-snapshot read that is
   unreliable in RTL** (RTL ACCUM snapshots all-zeros / garbage at every
   phase & pc, while the *same* matmul's downstream fp16 is P6c-proven
   0-ULP exact ⇒ the real-datapath ACCUM is correct; only the snapshot
   read is wrong; golden snapshot is self-consistent). No unit test
   validates ACCUM *readback* (P5 gen-2 ACCUM tests preload ACCUM + read
   ABUF; only `test_systolic_chained` validates post-MATMUL ACCUM). So
   the MATMUL is **not** proven buggy. **(iii) "deep-K" also refuted; BUG2
   LOCALIZED (P6j/#113)** via a clean control-validated `run_program
   --dram-dump-*` probe to the **functional `out_proj`** chain
   (`concat[768]@Wo[768×768]` → MATMUL@2131 → 0x1E@2141 → DMA-store):
   diverges; downstream innocent. **(iv) BUG2 ROOT CAUSE LOCKED &amp; FIXED
   — P6k / #115 (2026-05-17).** The "wide-N" framing (iii) was a proxy:
   wide N merely makes the weight exceed WBUF, routing out_proj/fc1/
   lm_head to the **tiled lowering** `emit_matmul_w8a16_large_weight_tiled`
   (a *different* path from the byte-exact-proven simple `emit_matmul_w8a16`
   that head Q/K/V use). `_large_weight_tile_plan(768,768)` = 2 N-tiles ×
   2 K-tiles ⇒ the 2nd MatmulInsn is **flags=1** (K-split accumulate).
   Real root: `systolic_controller.sv:344`
   `clear_acc=(ST_INIT_TILE)&&!flags_accumulate_q` suppresses the
   per-output-tile PE-accumulator clear for the **entire** multi-(m,n)-tile
   walk when flags=1; `systolic_array` `pe_acc` has no per-tile preload and
   DRAIN_WR overwrites (no RMW) ⇒ flags=1 is correct **only for a single
   16×16 output tile**. **Proven** by
   `rtl/verilator/test_systolic_chained.cpp::test_ksplit_accumulate_diagnostic`
   (chained = 124M cosim mode): K-split flags0/flags1 on the same data as
   the passing single-shot `test_matmul_multitile_2x2x2` (V1 known-exact
   contrast) ⇒ **1014/1024 ≠ correct, 0/1024 == the predicted cross-tile-
   leak model** (byte-exact the broken mechanism, all cells — not merely
   "diverges"). Pre-existing `test_matmul_accumulate_flag` only covered the
   single-tile degenerate-correct case (false confidence). **Fix S
   shipped (#115):** `_large_weight_tile_plan` now prefers a full-K weight
   tile (single MatmulInsn flags=0 per N-tile) whenever a full-K
   activation strip fits ABUF — the frozen GPT-2 124M bundle now emits
   **2340 MATMULs, 0 with flags=1** (out_proj/fc1/lm_head → full-K
   flags=0; fc2 K=3072 → large-input *streaming* path, flags=0-only by
   construction). K-tiling is integer-exact tiling-invariant ⇒ the golden
   is **numerically identical** (frozen `.raw` fixtures byte-unchanged,
   §6 simulator.py blob untouched). **VERIFIED — canonical 124M cosim
   after S (controls `head1_query`/`concat` 0-ULP — instrument valid):
   the canonical `block0_out_proj` *manifest node* is now 0-ULP
   byte-exact to golden across all 768 cols / all 3 S N-tiles, including
   the 48 NaN/overflow cols matched golden↔RTL.** Fix S fully resolved
   the out_proj tiled-path miscompute (BUG2). The remaining
   `block0_residual1` ~18139 ULP (finite, 0 NaN) is the FIRST node *past*
   out_proj — which is precisely the W8A16 fp16 overflow boundary (§4.x /
   P6c, pre-existing: golden itself saturates ±65504/NaN at out_proj
   @124M). Per-tensor byte-match past the first golden overflow is
   already documented ill-posed → **logits-level metric, task #109 /
   P6f**; it is **not** BUG2 and not a regression. This run also proves
   the entire pre-overflow datapath *through out_proj* is now byte-exact
   (a major de-risking for #109). **BUG2 root-caused, fixed (S),
   verified — #115 DONE.** The latent RTL flags=1-multi-tile HW
   bug (FPGA generality) is tracked as **R-min / task #116** (controller
   clear_acc-per-tile + DRAIN read-modify-write); the DIAGNOSTIC test is
   its permanent regression evidence (flips to a hard assertion when #116
   lands). Four bespoke-probe instrument failures across this saga, all
   caught by the `block0_head1_query`/`multitile_2x2x2`=exact control;
   only canonical `rc.*` + existing manifest + `--dram-dump-*` + the
   unit-level `test_systolic_chained` ACCUM readback are trustworthy
   (ACCUM-snapshot debt **#114**). The Option B non-finite contract stands
   independently as the correct, FPGA-deployable behavior.

## 5. Actions required to *complete* the freeze (owner: user — I do not commit)

The freeze is not real until the in-flight spec is committed. As of the
session-start snapshot these are uncommitted-modified and MUST be committed
as the frozen baseline:

- `software/taccel/isa/opcodes.py`
- `software/taccel/isa/instructions.py`
- `software/taccel/isa/encoding.py`
- `software/taccel/golden_model/simulator.py`
- `software/tests/test_compare_rtl_golden.py`

After commit, any opcode change requires a new dated revision of this file
and re-opens the gen-2 RTL contract.

## 6. Cross-references

- Decision basis: `software/docs/w8a32_deployment_scope.md` (§"Decision
  required", recommends (c.1)).
- ISA opcode definitions: `software/taccel/isa/opcodes.py:86` (`Opcode`).
- RTL opcode table: `rtl/src/include/taccel_pkg.sv:22-45`.
- Cosim harness / debug state: `docs/rtl_debug_plan.md` (current first
  divergence ~`pos_embed_add`, gen-1 — predates this freeze).
- Empirical probe: `software/tools/isa_coverage_probe.py` (re-runnable;
  compiles GPT-2 via `build_stage3_tiny_decoder_bundle`, histograms
  `codegen.instructions`; basis for §2).
