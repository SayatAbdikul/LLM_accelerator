// BUG1 reproducer: SET_ADDR_HI → LOAD read-after-write hazard.
//
// Two DRAM regions differ only in the upper-28 address bits (SET_ADDR_HI):
//   REGION_A: {HI=0, LO=LO_VAL} → filled 0xAA
//   REGION_B: {HI=1, LO=LO_VAL} → filled 0xBB
//
// Tests:
//   stable_lo_only           — baseline: HI=0, LOAD→region A. Always passes.
//   raw_hazard_separated     — SET_ADDR_HI(1), NOP, LOAD. Always passes.
//   raw_hazard_consecutive   — SET_ADDR_HI(1) immediately before LOAD (0→1).
//   raw_hazard_hi_to_zero    — pre-set HI=1, NOP, SET_ADDR_HI(0), LOAD (1→0).
//                              Matches freeze-doc bug: RTL reads stale HI=1
//                              (0x14c12ca0) when golden expects HI=0 (0x4c12ca0).
//   raw_hazard_hi_to_zero_tight — HI=1, HI=0 back-to-back, LOAD (tightest).
//
// Mechanism pin: s.dram.read_addr_log() reveals the exact AXI address the DMA
// issued. Stale HI=1 → 0x10001000; correct HI=0 → 0x001000.

#include "Vtaccel_top.h"
#include "Vtaccel_top___024root.h"
#include "verilated.h"
#include "include/testbench.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int tests_run  = 0;
static int tests_pass = 0;

#define TEST_PASS(name) do { \
    printf("PASS: %s\n", (name)); tests_pass++; tests_run++; } while (0)
#define TEST_FAIL(name, msg) do { \
    fprintf(stderr, "FAIL: %s — %s\n", (name), (msg)); std::exit(1); } while (0)

using tbutil::SimHarness;
using tbutil::sram_read_row;
constexpr int BUF_ABUF_ID = tbutil::BUF_ABUF_ID;

// HI=0 → region A at LO_VAL; HI=1 → region B at (1<<28)|LO_VAL.
// LO_VAL must be > max instruction byte offset (8 bytes × ~16 insns = 128 B).
constexpr uint32_t LO_VAL        = 0x001000;
constexpr uint64_t REGION_A_ADDR = uint64_t(LO_VAL);
constexpr uint64_t REGION_B_ADDR = (uint64_t(1) << 28) | LO_VAL;
// DRAM_SIZE must match -GDRAM_SIZE in the Makefile target.
// 512 MB covers region B at 256 MB + 4 KB.
constexpr size_t   DRAM_BYTES    = 512ULL * 1024 * 1024;
constexpr uint8_t  FILL_A        = 0xAA;
constexpr uint8_t  FILL_B        = 0xBB;

// Print the first AXI read address that looks like a DMA data access
// (address >= REGION_A_ADDR, so well past the instruction region at 0x0).
static void print_dma_addr(const SimHarness& s) {
    for (uint64_t a : s.dram.read_addr_log()) {
        if (a >= REGION_A_ADDR) {
            printf("  [mechanism] DMA addr=0x%016llx  (region_a=0x%016llx  region_b=0x%016llx)\n",
                   (unsigned long long)a,
                   (unsigned long long)REGION_A_ADDR,
                   (unsigned long long)REGION_B_ADDR);
            if (a == REGION_A_ADDR)
                printf("  [mechanism] BUG CONFIRMED: DMA sampled stale HI=0\n");
            else if (a == REGION_B_ADDR)
                printf("  [mechanism] FIX VERIFIED: DMA sampled correct HI=1\n");
            return;
        }
    }
    printf("  [mechanism] no DMA data read logged\n");
}

// Write both fill regions and return a freshly constructed harness.
static SimHarness make_harness() {
    SimHarness s(DRAM_BYTES);
    uint8_t fa[16], fb[16];
    std::memset(fa, FILL_A, sizeof(fa));
    std::memset(fb, FILL_B, sizeof(fb));
    s.dram.write_bytes(REGION_A_ADDR, fa, sizeof(fa));
    s.dram.write_bytes(REGION_B_ADDR, fb, sizeof(fb));
    return s;
}

// Baseline: no SET_ADDR_HI. After reset HI=0, so LOAD reads region A.
static void test_stable_lo_only() {
    const char* name = "stable_lo_only";
    auto s = make_harness();
    s.load({
        insn::SET_ADDR_LO(0, LO_VAL),
        insn::LOAD(BUF_ABUF_ID, 0, 1, 0, 0),
        insn::SYNC(0b001),
        insn::HALT(),
    });
    s.run(2000000);
    if (!s.dut->done || s.dut->fault)
        TEST_FAIL(name, "did not halt cleanly");
    uint8_t got[16];
    sram_read_row(s.dut.get(), BUF_ABUF_ID, 0, got);
    if (got[0] != FILL_A)
        TEST_FAIL(name, "ABUF has unexpected fill (expected 0xAA for region A)");
    TEST_PASS(name);
}

// Control: NOP separator between SET_ADDR_HI and LOAD. Must always pass.
static void test_raw_hazard_separated() {
    const char* name = "raw_hazard_separated";
    auto s = make_harness();
    s.load({
        insn::SET_ADDR_LO(0, LO_VAL),
        insn::SET_ADDR_HI(0, 1),
        insn::NOP(),                        // separator — one instruction gap
        insn::LOAD(BUF_ABUF_ID, 0, 1, 0, 0),
        insn::SYNC(0b001),
        insn::HALT(),
    });
    s.run(2000000);
    if (!s.dut->done || s.dut->fault)
        TEST_FAIL(name, "did not halt cleanly");
    print_dma_addr(s);
    uint8_t got[16];
    sram_read_row(s.dut.get(), BUF_ABUF_ID, 0, got);
    if (got[0] != FILL_B)
        TEST_FAIL(name, got[0] == FILL_A
                        ? "ABUF=0xAA: SET_ADDR_HI write not visible even with NOP separator"
                        : "ABUF has unexpected fill byte");
    TEST_PASS(name);
}

// Bug case: SET_ADDR_HI immediately before LOAD (no gap), 0→1 direction.
// Pre-fix : ABUF=0xAA → stale HI=0 → reads region A → FAIL (BUG1 confirmed).
// Post-fix: ABUF=0xBB → correct HI=1 → reads region B → PASS.
static void test_raw_hazard_consecutive() {
    const char* name = "raw_hazard_consecutive";
    auto s = make_harness();
    s.load({
        insn::SET_ADDR_LO(0, LO_VAL),
        insn::SET_ADDR_HI(0, 1),
        insn::LOAD(BUF_ABUF_ID, 0, 1, 0, 0),
        insn::SYNC(0b001),
        insn::HALT(),
    });
    s.run(2000000);
    if (!s.dut->done || s.dut->fault)
        TEST_FAIL(name, "did not halt cleanly");
    print_dma_addr(s);
    uint8_t got[16];
    sram_read_row(s.dut.get(), BUF_ABUF_ID, 0, got);
    if (got[0] != FILL_B)
        TEST_FAIL(name, got[0] == FILL_A
                        ? "ABUF=0xAA: LOAD read stale HI=0 (BUG1 confirmed — apply the stall fix)"
                        : "ABUF has unexpected fill byte");
    TEST_PASS(name);
}

// Bug candidate: pre-set HI=1 (with NOP separator so it commits), then
// SET_ADDR_HI(R0,0) immediately before LOAD — the 1→0 direction.
//
// This matches the freeze-doc observation: RTL samples stale HI=1 (0x14c12ca0)
// when the golden expects HI=0 (0x4c12ca0).  If RTL latches the wrong address
// ABUF will contain FILL_B (0xBB) instead of FILL_A (0xAA).
static void test_raw_hazard_hi_to_zero() {
    const char* name = "raw_hazard_hi_to_zero";
    auto s = make_harness();
    s.load({
        insn::SET_ADDR_LO(0, LO_VAL),
        insn::SET_ADDR_HI(0, 1),            // commit HI=1 first …
        insn::NOP(),                         // … separator so the write settles
        insn::SET_ADDR_HI(0, 0),            // then clear HI → should point to region A
        insn::LOAD(BUF_ABUF_ID, 0, 1, 0, 0),
        insn::SYNC(0b001),
        insn::HALT(),
    });
    s.run(2000000);
    if (!s.dut->done || s.dut->fault)
        TEST_FAIL(name, "did not halt cleanly");
    print_dma_addr(s);
    uint8_t got[16];
    sram_read_row(s.dut.get(), BUF_ABUF_ID, 0, got);
    if (got[0] != FILL_A)
        TEST_FAIL(name, got[0] == FILL_B
                        ? "ABUF=0xBB: LOAD read stale HI=1 — BUG1 (1→0 direction) confirmed"
                        : "ABUF has unexpected fill byte");
    TEST_PASS(name);
}

// Root-cause reproducer: LOAD dispatch silently dropped when DMA is busy.
//
// The control unit stalls on sfu_busy but NOT dma_busy for LOAD/STORE:
//   OP_LOAD, OP_STORE: dma_dispatch = !sfu_busy;   // no !dma_busy guard
// So a LOAD dispatched while a STORE's DMA is in progress fires dma_dispatch=1,
// but the DMA engine (in D_STORE_W, not D_IDLE) ignores the pulse.  The
// control unit still advances, SYNC(001) drains the STORE, and the LOAD's
// destination buffer is never written.
//
// Mechanism:
//   STORE-DMA does not block instruction fetches (write channel ≠ read channel,
//   no rd_inflight_q held).  So 3 fetch+issue cycles complete while 32 AXI write
//   beats are still in flight, bringing the second LOAD to S_ISSUE while
//   dma_busy=1.  dma_dispatch=1 fires but is ignored; ABUF retains the stale
//   FILL_B from the earlier load.
//
// Pre-fix: ABUF[0] == FILL_B (0xBB) — second LOAD dropped → BUG confirmed.
// Post-fix: ABUF[0] == FILL_A (0xAA) — second LOAD ran correctly.
static void test_dispatch_drop_via_store() {
    const char* name = "dispatch_drop_via_store";

    // REGION_STORE is within the 512 MB window, distinct from A and B.
    constexpr uint32_t LO_STORE_VAL   = 0x002000;
    constexpr uint64_t REGION_STORE_ADDR = (uint64_t(1) << 28) | LO_STORE_VAL;

    SimHarness s(DRAM_BYTES);
    // make_harness() pattern: fill A and B; B is 1 row only so extend to 1 row
    uint8_t fa[16], fb[16];
    std::memset(fa, FILL_A, sizeof(fa));
    std::memset(fb, FILL_B, sizeof(fb));
    s.dram.write_bytes(REGION_A_ADDR, fa, sizeof(fa));
    s.dram.write_bytes(REGION_B_ADDR, fb, sizeof(fb));

    s.load({
        // 1. Load ABUF row 0 from REGION_B (HI=1) → ABUF[0] = FILL_B.
        insn::SET_ADDR_LO(0, LO_VAL),
        insn::SET_ADDR_HI(0, 1),
        insn::LOAD(BUF_ABUF_ID, 0, 1, 0, 0),
        insn::SYNC(0b001),

        // 2. STORE 32 rows from ABUF to REGION_STORE — long-running DMA write.
        //    32 AXI beats keep dma_busy=1 for ~35+ cycles.
        insn::SET_ADDR_LO(1, LO_STORE_VAL),
        insn::SET_ADDR_HI(1, 1),
        insn::STORE(BUF_ABUF_ID, 0, 32, 1, 0),

        // 3. No SYNC — DMA write still in progress.
        //    3 fetch+issue cycles complete in ~12 cycles, well before STORE finishes.
        insn::SET_ADDR_LO(0, LO_VAL),
        insn::SET_ADDR_HI(0, 0),              // R0 = REGION_A (HI=0)
        // dma_dispatch fires here but DMA is in D_STORE_W → dispatch DROPPED.
        insn::LOAD(BUF_ABUF_ID, 0, 1, 0, 0),

        // 4. SYNC drains the STORE DMA; if LOAD was dropped, ABUF[0] stays FILL_B.
        insn::SYNC(0b001),
        insn::HALT(),
    });

    s.run(2000000);
    if (!s.dut->done || s.dut->fault)
        TEST_FAIL(name, "did not halt cleanly");

    uint8_t got[16];
    sram_read_row(s.dut.get(), BUF_ABUF_ID, 0, got);
    printf("  [mechanism] ABUF[0]=0x%02x  (FILL_A=0x%02x expect; FILL_B=0x%02x=stale)\n",
           got[0], FILL_A, FILL_B);
    if (got[0] == FILL_B)
        TEST_FAIL(name, "ABUF=0xBB: LOAD dispatch silently dropped while STORE DMA busy (BUG1 root cause confirmed)");
    if (got[0] != FILL_A)
        TEST_FAIL(name, "ABUF has unexpected fill byte");
    TEST_PASS(name);
}

// Same as test_raw_hazard_hi_to_zero but without the NOP separator between
// the two SET_ADDR_HI writes (most aggressive back-to-back case).
static void test_raw_hazard_hi_to_zero_tight() {
    const char* name = "raw_hazard_hi_to_zero_tight";
    auto s = make_harness();
    s.load({
        insn::SET_ADDR_LO(0, LO_VAL),
        insn::SET_ADDR_HI(0, 1),            // HI=1
        insn::SET_ADDR_HI(0, 0),            // immediately overwrite with HI=0
        insn::LOAD(BUF_ABUF_ID, 0, 1, 0, 0),
        insn::SYNC(0b001),
        insn::HALT(),
    });
    s.run(2000000);
    if (!s.dut->done || s.dut->fault)
        TEST_FAIL(name, "did not halt cleanly");
    print_dma_addr(s);
    uint8_t got[16];
    sram_read_row(s.dut.get(), BUF_ABUF_ID, 0, got);
    if (got[0] != FILL_A)
        TEST_FAIL(name, got[0] == FILL_B
                        ? "ABUF=0xBB: LOAD read stale HI=1 — BUG1 (tight 1→0) confirmed"
                        : "ABUF has unexpected fill byte");
    TEST_PASS(name);
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);

    // Baseline and control first — always pass, validate harness soundness.
    test_stable_lo_only();
    test_raw_hazard_separated();
    // 0→1 direction (original hypothesis).
    test_raw_hazard_consecutive();
    // 1→0 direction — matches freeze-doc: RTL stale HI=1, golden expects HI=0.
    test_raw_hazard_hi_to_zero();
    test_raw_hazard_hi_to_zero_tight();
    // Root-cause: LOAD dispatch dropped when STORE DMA is running.
    test_dispatch_drop_via_store();

    printf("\n%d/%d tests passed\n", tests_pass, tests_run);
    return (tests_pass == tests_run) ? 0 : 1;
}
