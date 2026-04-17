# TACCEL RTL Debugging Plan

## Context

The TACCEL RTL implements a 20-instruction ISA for an INT8 transformer accelerator targeting DeiT-tiny. The RTL has grown through Phases A-E (fetch/decode/control, DMA, helpers, SFU, experimental ops) and now needs systematic debugging to achieve functional parity with the Python golden model. The git status shows uncommitted changes across the RTL core (`control_unit.sv`, `taccel_pkg.sv`, `taccel_top.sv`), test infrastructure, and golden model — indicating active development that needs verification.

The goal is a structured, bottom-up debugging workflow that establishes correctness at each layer before moving up, ultimately reaching program-level RTL-vs-golden sign-off.

---

## Phase 1: Establish Baseline — Run All Existing Tests

Before debugging anything, capture the current pass/fail state of every test.

### 1a. Verilator C++ tests (unit/subsystem level)
```
make -C rtl/verilator test_decode
make -C rtl/verilator test_control
make -C rtl/verilator test_dma
make -C rtl/verilator test_helpers
make -C rtl/verilator test_sfu
make -C rtl/verilator test_systolic
make -C rtl/verilator test_systolic_array_chained
make -C rtl/verilator test_systolic_chained
```

### 1b. cocotb integration tests (top-level ISA-visible behavior)
```
make -C rtl/cocotb test_all SIM=verilator
make -C rtl/cocotb test_systolic_chained SIM=verilator
```

### 1c. Software golden model tests
```
cd software && python -m pytest tests/test_golden_model.py -v
cd software && python -m pytest tests/test_compare_rtl_golden.py -v
```

### 1d. Record results
Create a checklist of PASS/FAIL per test target. This is the debugging scoreboard — every fix should flip at least one FAIL to PASS.

---

## Phase 2: Bottom-Up Module Verification

Debug in strict dependency order. Do NOT move to the next layer until the current layer is green.

### Layer 0: Leaf modules (no RTL dependencies)

**`rtl/src/memory/sram_dp.sv`** — Memory primitive everything depends on.
- Verify write-first semantics: port A write + read same address → new data appears next cycle
- Verify port B read-only: data returns one cycle after `b_en`
- Verify simultaneous port A write + port B read to same address → port B sees OLD data
- Test boundary: row 0, row DEPTH-1

**`rtl/src/systolic/systolic_pe.sv`** — Single INT8 MAC cell.
- Verify `acc_clear` zeroes accumulator AND forwarding outputs (critical for chained mode)
- Verify signed 8-bit × signed 8-bit → sign-extended 32-bit accumulate
- Verify no saturation (wraps, per ISA contract)

**`rtl/src/decode_unit.sv`** — Combinational decoder.
- Verify all 20 opcodes decode correctly vs `insn_builder.py` / `testbench.h` encoding
- Verify `illegal` raised for opcodes 0x14–0x1F
- Verify `illegal` raised for buffer ID `2'b11` in R-type, M-type, B-type only (not A/C/S-type)

### Layer 1: Memory subsystem

**`rtl/src/memory/sram_subsystem.sv`**
- Verify OOB check boundaries: row 8191 valid for ABUF, row 8192 faults
- Verify `a_fault` gates SRAM write enable — no write occurs on OOB
- **Critical**: read-data mux uses registered `a_buf_q`/`b_buf_q`. If requestor changes `a_buf` between request and read-back cycles, wrong SRAM is selected. Verify buffer ID stability.

**`rtl/src/memory/register_file.sv`**
- Verify `tile_valid` = 0 after reset, = 1 after first `tile_we`, never returns to 0
- Verify `addr_lo_we` writes only bits [27:0], preserves [55:28]; same for `addr_hi_we`
- Verify scale register read port wrapping (sreg=15 → reads S15, then S0 for dual-read)

### Layer 2: Individual execution units

**`rtl/src/fetch_unit.sv`**
- Verify `bswap64` byte-swap converts AXI LE to BE instruction words
- Verify PC lane select: even PC → `rdata[63:0]`, odd PC → `rdata[127:64]`
- Verify fault is sticky (stays in S_FAULT until reset)
- Verify `r_last` must be asserted on single beat (`ar_len=0`)

**`rtl/src/dma_engine.sv`** — HIGH PRIORITY
- Verify multi-burst: `xfer_len=513` requires 3 bursts (256+256+1), all 513 rows in SRAM
- Verify `curr_dram_addr_q` increments by `burst_beats × 16` after each burst
- Verify `curr_sram_row_q` increments by `burst_beats` after each burst
- Verify `ARLEN = burst_beats - 1` truncation is safe when `burst_beats=256` (yields 0xFF)
- Verify STORE read-ahead timing: `D_STORE_SRAM_PRE` issues SRAM read, data valid one cycle later in `D_STORE_W`
- Verify DRAM OOB prevalidation: check `>` vs `>=` for off-by-one
- Verify `xfer_len=0` is no-op

**`rtl/src/systolic/systolic_controller.sv` + `systolic_array.sv`** — HIGH PRIORITY
- **Data layout mismatch (known issue)**: Controller reads src1 from port B (ABUF), src2 from port A (WBUF). Verify convention matches software's A[M,K] × B[K,N] without requiring caller to transpose
- Verify chained mode flush: `CHAIN_TOTAL_STEPS = 46`, steps 16-45 inject zeros, skew pipeline produces correct final values
- Verify tile base addresses: `src1 << 4` (16 rows/INT8 tile), `dst << 6` (64 rows/INT32 tile)
- Verify drain writeback: 4 SRAM rows per PE row (4 groups of 4 INT32), 64 rows per 16×16 tile
- Verify accumulate flag (`flags[0]=1`) suppresses `clear_acc` across K tiles

**`rtl/src/blocking_helper_engine.sv`** — HIGH PRIORITY
- **BUF_COPY flat**: Verify forward/backward overlap handling (`flat_backward_q` flag)
- **BUF_COPY transpose**: Verify byte-lane insertion matches golden model's `np.transpose`
- **REQUANT rounding**: Verify `fp16_mul_round_even` matches golden model for edge cases (exact 0.5, negatives, zero scale)
- **VADD mode selection**: INT8 saturating (both buffers INT8) vs INT32 broadcast bias-add (src=ACCUM)
- **DEQUANT_ADD**: Verify scale application order matches golden model (dual-scale from S[sreg], S[sreg+1])
- **SRAM read timing**: Every FSM state that reads SRAM must wait one cycle (REQ→LATCH pattern) before using data

**`rtl/src/sfu_engine.sv`**
- Uses `real`-valued arithmetic — simulation-only. Primary concern is output INT8 rounding.
- **SOFTMAX**: Verify row-max subtraction, exp, normalization, round-half-to-even to INT8
- **LAYERNORM**: Verify gamma/beta loading from WBUF as FP16 pairs, mean/variance normalization
- **GELU**: Verify erf polynomial coefficients match `ERF_A1`–`ERF_A5`, `ERF_P` from `taccel_pkg.sv`
- **SOFTMAX_ATTNV**: Verify SRAM access pattern (QK^T from src1, V from src2) and output addressing
- Verify SFU reads through port B only, writes through port A only

### Layer 3: Control and arbitration

**`rtl/src/control_unit.sv`**
- Map each opcode's dispatch readiness conditions:
  - LOAD/STORE: blocked only by `sfu_busy`
  - MATMUL: blocked by `sfu_busy`, requires `tile_valid`
  - Helpers (BUF_COPY/REQUANT/VADD/etc.): blocked by `dma_busy || sys_busy || helper_busy || sfu_busy`
  - SFU ops: blocked by all four units, requires `tile_valid`
- Verify LOAD can overlap with MATMUL (both async, only serialized by SYNC)
- **SYNC_WAIT deadlock risk**: If mask selects unit that was never dispatched, SYNC should retire immediately (all idle → clear)
- **DISP_WAIT retirement**: Verify fires on exact cycle `helper_busy` drops, not one cycle late
- Verify `ext_fault` breaks out of SYNC_WAIT (stuck unit scenario)

**`rtl/src/taccel_top.sv` — AXI arbitration** (lines 118–174)
- Verify `prefer_fetch_after_dma_q` allows one fetch between DMA bursts
- Verify `dma_r_owner_q` correctly gates `dma_r_valid` vs `fetch_r_valid`
- Verify no combinational loop through AR ready signals
- Verify `rd_inflight_q` toggles correctly with each AR-R pair

**`rtl/src/taccel_top.sv` — SRAM arbitration** (lines 334–375)
- Priority: Helper > SFU > DMA > Systolic (port A); Helper > SFU > Systolic (port B)
- `sram_a_rdata` is broadcast to all consumers (lines 367–369) — each consumer must only use data when it was the active requester
- Check `obs_forbidden_overlap_violation_q` — must ALWAYS be 0

**`rtl/src/taccel_top.sv` — Fault collapse** (lines 386–403)
- Priority: fetch > DMA > helper > SFU > systolic SRAM
- Verify simultaneous faults report the higher-priority one
- Verify `sys_sram_fault_latched` is sticky and reaches S_FAULT

---

## Phase 3: Critical Bug Patterns to Hunt

### 3a. SRAM arbitration races
**Where**: `taccel_top.sv:336-375`
**Check**: `obs_forbidden_overlap_violation_q` after every test run. If ever nonzero → control-level serialization bug.

### 3b. AXI protocol violations
**Where**: `fetch_unit.sv`, `dma_engine.sv`, `taccel_top.sv` AR mux
**Check**: ARVALID stable with ARADDR/ARLEN; RREADY not deasserted while RVALID high; WLAST on final write beat; AWLEN matches actual W beats.

### 3c. Off-by-one in tile addressing
**Where**: `systolic_controller.sv`, `blocking_helper_engine.sv`, `dma_engine.sv`
**Check**: INT8 tiles use `<< 4` (16 rows), INT32 tiles use `<< 6` (64 rows). Mixing these corrupts data silently. Verify ACCUM drain addressing is consistent with REQUANT reader.

### 3d. Rounding mismatches (most likely source of golden-model divergence)
**Where**: `blocking_helper_engine.sv` `fp16_mul_round_even`, `sfu_engine.sv` `round_half_even`
**Check**: Negative values — helper uses abs/round/restore-sign, SFU operates on real directly. Must agree for all inputs. Test `value=2.5→2`, `value=3.5→4`, both positive and negative.

### 3e. FSM deadlocks
**Where**: `control_unit.sv`, `blocking_helper_engine.sv`, `sfu_engine.sv`
**Check**: SYNC_WAIT with mask for never-dispatched unit should clear immediately. Helper stuck in non-terminal state. Use `obs_sync_wait_*_cycles_q` — unbounded growth indicates stuck unit.

### 3f. Instruction register stability
**Where**: `taccel_top.sv:585-590`
**Check**: `insn_data_q` changes only when `insn_valid_w` fires; stable throughout S_ISSUE and S_DISP_WAIT. Spurious `insn_valid_w` corrupts executing instruction's operands.

---

## Phase 4: RTL-vs-Golden Comparison Strategy

### 4a. Instruction-level trace alignment
Use `run_program.cpp` to produce per-instruction trace. Compare:
1. Instruction count: `obs_retired_insn_count_q` == golden model's executed count
2. PC sequence: `obs_retire_pc` at each step
3. Find first divergence point

### 4b. SRAM snapshot comparison
Use `run_program.cpp --snapshot-request-path` to capture SRAM at specific PCs:
- After SYNC(010): compare ACCUM (MATMUL output)
- After blocking helper: compare target buffer
- After SYNC(100): compare ABUF (SFU output)

### 4c. Narrowing numerical mismatches
When byte-level mismatch found:
1. Identify producing instruction from trace
2. Extract inputs from pre-execution SRAM snapshot
3. Run same inputs through golden model's function
4. In RTL, use Verilator hierarchical signal access to read internal registers

### 4d. Binary bisection for large programs
For DeiT-tiny programs (500+ instructions):
1. Insert artificial HALTs at midpoints
2. Run both RTL and golden model to HALT
3. Compare SRAM dumps
4. Binary-search for first diverging instruction

---

## Phase 5: Observability Hooks Reference

Key `verilator public_flat_rd` signals in `taccel_top.sv` for debugging:

| Signal | Line | Use |
|--------|------|-----|
| `obs_retire_pulse_w` / `obs_retire_pc_w` / `obs_retire_opcode_w` | 242–244 | Instruction trace, compare vs golden |
| `obs_ctrl_fault_pulse_w` / `obs_ctrl_fault_code_w` / `obs_ctrl_fault_pc_w` | 245–247 | First fault localization |
| `obs_fault_source_q` | 268 | Which unit faulted (1=FETCH, 2=DMA, 3=HELPER, 4=SFU, 5=SRAM, 6=CTRL) |
| `obs_forbidden_overlap_violation_q` | 270 | SRAM serialization violation — MUST always be 0 |
| `obs_sync_wait_*_cycles_q` | 256–258 | Deadlock detection — unbounded growth = stuck unit |
| `obs_cycle_count_q` / `obs_retired_insn_count_q` | 254–255 | Sanity check, cycle/instruction ratio |
| `obs_*_issue_pc_q` / `obs_*_issue_opcode_q` | 271–278 | Trace back faulting unit to dispatching instruction |

**Debugging workflow**:
1. After any run: check `obs_forbidden_overlap_violation_q` — if nonzero, stop everything, fix serialization bug
2. If fault: read `obs_fault_source_q` + `obs_fault_code_q` + `obs_fault_pc_q`
3. Compare `obs_retired_insn_count_q` vs golden model count for divergence point
4. Use `obs_sync_wait_*_cycles_q` to detect deadlock-prone SYNCs

---

## Phase 6: Inter-Module Integration Hotspots

### Hotspot 1: insn_data_q registration timing
`insn_data_q` captured on `insn_valid_w` (`taccel_top.sv` ~line 588). Decode is combinational on `insn_data_q`. Control FSM acts on decoded struct in S_ISSUE. If `insn_valid_w` fires while FSM is not in S_FETCH, instruction register is overwritten during execution.

### Hotspot 2: Helper-to-SRAM-to-Systolic data path
When helper is active, it owns both SRAM ports. But `sram_a_rdata` is broadcast (lines 367–369). If systolic issues a read on the same cycle (prevented by serialization), it sees helper's data. Verify serialization invariant holds.

### Hotspot 3: DMA read channel ownership
`dma_r_owner_q` determines R-channel routing. If AXI slave has unexpected latency or `r_last` is incorrect, `rd_inflight_q` won't clear → fetch permanently blocked.

### Hotspot 4: Scale register read timing during dispatch
Scale registers read combinationally. Helper/SFU latches scale on dispatch cycle. If `insn_data_q` changes between dispatch and latch (spurious `insn_valid_w`), wrong scale is captured.

---

## Verification Checklist

1. All 8 Verilator test targets pass:
   ```
   make -C rtl/verilator test_decode test_control test_dma test_helpers test_sfu test_systolic test_systolic_array_chained test_systolic_chained
   ```
2. All cocotb tests pass:
   ```
   make -C rtl/cocotb test_all SIM=verilator
   make -C rtl/cocotb test_systolic_chained SIM=verilator
   ```
3. `obs_forbidden_overlap_violation_q == 0` in every test run
4. Golden model comparison passes:
   ```
   python -m pytest software/tests/test_compare_rtl_golden.py -v
   ```
5. Program-level sign-off: `run_program` on a compiler-generated binary matches golden model at all SRAM snapshot points

### Key files to modify during debugging

| File | Area |
|------|------|
| `rtl/src/taccel_top.sv` | SRAM arbitration, fault plumbing, AXI mux |
| `rtl/src/control_unit.sv` | Dispatch conditions, SYNC logic, FSM transitions |
| `rtl/src/systolic/systolic_controller.sv` | Tile addressing, data layout |
| `rtl/src/blocking_helper_engine.sv` | Rounding, SRAM timing, operation semantics |
| `rtl/src/dma_engine.sv` | Multi-burst, addressing, prevalidation |
| `rtl/src/sfu_engine.sv` | Rounding, SRAM access patterns |
