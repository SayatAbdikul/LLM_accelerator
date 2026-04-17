// Verilator integration tests for the Phase A control/top-level contract.
//
// Tests the fetch → decode → issue pipeline, including:
//   - NOP then HALT: done asserted, no fault
//   - CONFIG_TILE: tile registers updated correctly
//   - SET_SCALE (immediate): scale register updated
//   - SET_ADDR_LO + SET_ADDR_HI: address register composed correctly
//   - SYNC 0b000 (NOP-like): passes without stall
//   - SYNC with all-idle units: completes immediately
//   - Illegal opcode: fault asserted, correct fault code
//   - Remaining legal but unsupported ops: FAULT_UNSUPPORTED_OP
//   - Stage D SFU dispatch / no-config behavior
//   - MATMUL without CONFIG_TILE: FAULT_NO_CONFIG
//   - Fetch / DMA / systolic SRAM fault plumbing
//   - Multi-instruction sequence

#include "Vtaccel_top.h"
#include "verilated.h"
#include "include/testbench.h"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <memory>

static int tests_run  = 0;
static int tests_pass = 0;

#define TEST_PASS(name) do { \
    printf("PASS: %s\n", name); tests_pass++; tests_run++; } while(0)
#define TEST_FAIL(name, msg) do { \
    fprintf(stderr, "FAIL: %s — %s\n", name, msg); std::exit(1); } while(0)

// ============================================================================
// Helper: create fresh DUT + DRAM, reset, load program, run
// ============================================================================
using tbutil::SimHarness;

static void expect_fault_program(const char* name,
                                 const std::vector<uint64_t>& prog,
                                 uint32_t expected_fault_code,
                                 int timeout = 5000) {
    SimHarness s;
    s.load(prog);
    s.run(timeout);
    EXPECT(s.dut->fault == 1, "fault should assert");
    EXPECT(s.dut->done == 0, "done should remain 0 on fault");
    EXPECT(s.dut->fault_code == expected_fault_code, "unexpected fault code");
    TEST_PASS(name);
}

// ============================================================================
// Test: NOP → HALT
// ============================================================================
static void test_nop_halt() {
    const char* name = "nop_then_halt";
    SimHarness s;
    s.load({ insn::NOP(), insn::HALT() });
    s.run();
    EXPECT(s.dut->done  == 1, "done should be 1 after HALT");
    EXPECT(s.dut->fault == 0, "no fault expected");
    TEST_PASS(name);
}

// ============================================================================
// Test: Three NOPs then HALT
// ============================================================================
static void test_multi_nop_halt() {
    const char* name = "multi_nop_halt";
    SimHarness s;
    s.load({ insn::NOP(), insn::NOP(), insn::NOP(), insn::HALT() });
    s.run();
    EXPECT(s.dut->done  == 1, "done after 3 NOPs + HALT");
    EXPECT(s.dut->fault == 0, "no fault");
    TEST_PASS(name);
}

// ============================================================================
// Test: HALT immediately (no NOP)
// ============================================================================
static void test_immediate_halt() {
    const char* name = "immediate_halt";
    SimHarness s;
    s.load({ insn::HALT() });
    s.run();
    EXPECT(s.dut->done  == 1, "done on first HALT");
    EXPECT(s.dut->fault == 0, "no fault");
    TEST_PASS(name);
}

// ============================================================================
// Test: CONFIG_TILE registers visible in tile_m / tile_n / tile_k
// ============================================================================
static void test_config_tile() {
    const char* name = "config_tile_registers";
    SimHarness s;
    // CONFIG_TILE M=3, N=7, K=5; then HALT
    s.load({ insn::CONFIG_TILE(3, 7, 5), insn::HALT() });
    s.run();
    EXPECT(s.dut->done == 1, "done after CONFIG_TILE + HALT");
    EXPECT(s.dut->fault == 0, "no fault");
    // Tile registers are exposed via taccel_top → register_file
    // We verify indirectly: MATMUL after CONFIG_TILE should NOT fault
    TEST_PASS(name);
}

// ============================================================================
// Test: CONFIG_TILE → MATMUL → HALT  (MATMUL dispatched, completes immediately
//       with SYNC waiting for completion)
// ============================================================================
static void test_config_tile_then_matmul() {
    const char* name = "config_tile_then_matmul_no_fault";
    SimHarness s;
    s.load({
        insn::CONFIG_TILE(1, 1, 1),
        insn::MATMUL(/*src1*/0, 0, /*src2*/1, 0, /*dst*/2, 0, /*sreg*/0),
        insn::SYNC(0b010),   // wait for systolic (immediately clear since sys_busy=0)
        insn::HALT()
    });
    s.run();
    EXPECT(s.dut->done  == 1, "done");
    EXPECT(s.dut->fault == 0, "MATMUL with valid tile should not fault");
    TEST_PASS(name);
}

// ============================================================================
// Test: SET_SCALE immediate — opcode executes without fault
// ============================================================================
static void test_set_scale() {
    const char* name = "set_scale_immediate";
    SimHarness s;
    // SET_SCALE S0 = 0x3C00 (1.0 FP16), then HALT
    s.load({ insn::SET_SCALE(0, 0x3C00), insn::HALT() });
    s.run();
    EXPECT(s.dut->done  == 1, "done after SET_SCALE + HALT");
    EXPECT(s.dut->fault == 0, "no fault");
    TEST_PASS(name);
}

// ============================================================================
// Test: SET_ADDR_LO + SET_ADDR_HI + HALT
// ============================================================================
static void test_set_addr() {
    const char* name = "set_addr_lo_hi";
    SimHarness s;
    constexpr uint32_t LO = 0x0100000;
    constexpr uint32_t HI = 0x0000123;
    s.load({
        insn::SET_ADDR_LO(0, LO),  // R0[27:0]
        insn::SET_ADDR_HI(0, HI),  // R0[55:28]
        insn::HALT()
    });
    s.run();
    EXPECT(s.dut->done  == 1, "done");
    EXPECT(s.dut->fault == 0, "no fault on SET_ADDR");

    const uint64_t expected = ((uint64_t(HI & 0x0FFFFFFF) << 28) | uint64_t(LO & 0x0FFFFFFF));
    const uint64_t got = uint64_t(s.dut->rootp->taccel_top__DOT__u_regfile__DOT__addr_regs[0])
                         & 0x00FFFFFFFFFFFFFFULL;
    EXPECT(got == expected, "R0 56-bit composition mismatch");
    TEST_PASS(name);
}

// ============================================================================
// Test: SYNC 0b000 (no-op SYNC)
// ============================================================================
static void test_sync_nop() {
    const char* name = "sync_mask_zero";
    SimHarness s;
    s.load({ insn::SYNC(0b000), insn::HALT() });
    s.run();
    EXPECT(s.dut->done  == 1, "done after SYNC(0) + HALT");
    EXPECT(s.dut->fault == 0, "no fault");
    TEST_PASS(name);
}

// ============================================================================
// Test: SYNC 0b111 (all units) — clears immediately since all busy=0
// ============================================================================
static void test_sync_all_idle() {
    const char* name = "sync_all_units_idle";
    SimHarness s;
    s.load({ insn::SYNC(0b111), insn::HALT() });
    s.run(500);
    EXPECT(s.dut->done  == 1, "done after SYNC(7) when all units idle");
    EXPECT(s.dut->fault == 0, "no fault");
    TEST_PASS(name);
}

// ============================================================================
// Test: Illegal opcode 0x14 → fault
// ============================================================================
static void test_illegal_opcode() {
    const char* name = "illegal_opcode_fault";
    SimHarness s;
    s.load({ insn::ILLEGAL_OP() });
    s.run(500);
    EXPECT(s.dut->fault == 1, "fault asserted for illegal opcode");
    EXPECT(s.dut->done  == 0, "done should be 0 on fault");
    EXPECT(s.dut->fault_code == 1, "fault_code = FAULT_ILLEGAL_OP (1)");
    TEST_PASS(name);
}

// ============================================================================
// Test: Stage E helper/SFU paths are now legal and complete cleanly
// ============================================================================
static void test_stage_e_paths() {
    struct StageECase {
        const char* name;
        std::vector<uint64_t> prog;
    };
    const StageECase cases[] = {
        { "requant_pc_dispatch", {
            insn::CONFIG_TILE(1, 1, 1),
            insn::REQUANT_PC(2, 0, 1, 0, 0, 0, 0),
            insn::HALT()
        }},
        { "scale_mul_dispatch", {
            insn::CONFIG_TILE(1, 1, 1),
            insn::SET_SCALE(2, 0x3800),
            insn::SCALE_MUL(0, 0, 1, 0, 2),
            insn::HALT()
        }},
        { "dequant_add_dispatch", {
            insn::CONFIG_TILE(1, 1, 1),
            insn::SET_SCALE(4, 0x2C00),
            insn::SET_SCALE(5, 0x3400),
            insn::DEQUANT_ADD(2, 0, 0, 0, 1, 0, 4),
            insn::HALT()
        }},
        { "softmax_attnv_dispatch_sync", {
            insn::CONFIG_TILE(1, 1, 1),
            insn::SET_SCALE(8, 0x3400),
            insn::SET_SCALE(9, 0x3400),
            insn::SET_SCALE(10, 0x3400),
            insn::SET_SCALE(11, 0x3000),
            insn::SOFTMAX_ATTNV(2, 0, 0, 0, 1, 0, 8),
            insn::SYNC(0b100),
            insn::HALT()
        }},
    };

    for (const auto& tc : cases) {
        SimHarness s;
        s.load(tc.prog);
        s.run(100000);
        EXPECT(s.dut->done == 1, "stage E path should halt cleanly");
        EXPECT(s.dut->fault == 0, "stage E path should not fault");
        TEST_PASS(tc.name);
    }
}

// ============================================================================
// Test: Stage D SFU ops require CONFIG_TILE
// ============================================================================
static void test_sfu_no_config_faults() {
    expect_fault_program("softmax_without_config_tile",
                         { insn::SOFTMAX(2, 0, 0, 0, 0) }, 4, 1000);
    expect_fault_program("layernorm_without_config_tile",
                         { insn::LAYERNORM(0, 0, 1, 0, 0, 0, 0) }, 4, 1000);
    expect_fault_program("gelu_without_config_tile",
                         { insn::GELU(0, 0, 0, 0, 0) }, 4, 1000);
}

// ============================================================================
// Test: Stage D SFU ops dispatch and complete through SYNC(100)
// ============================================================================
static void test_sfu_dispatch_paths() {
    {
        SimHarness s;
        s.load({
            insn::CONFIG_TILE(1, 1, 1),
            insn::SET_SCALE(0, 0x3800),
            insn::SET_SCALE(1, 0x3400),
            insn::SOFTMAX(0, 0, 1, 0, 0),
            insn::SYNC(0b100),
            insn::HALT()
        });
        s.run(50000);
        EXPECT(s.dut->done == 1, "softmax dispatch path should halt cleanly");
        EXPECT(s.dut->fault == 0, "softmax dispatch path should not fault");
        TEST_PASS("softmax_dispatch_sync");
    }

    {
        SimHarness s;
        s.load({
            insn::CONFIG_TILE(1, 1, 1),
            insn::SET_SCALE(0, 0x3800),
            insn::SET_SCALE(1, 0x3400),
            insn::LAYERNORM(0, 0, 1, 0, 0, 0, 0),
            insn::SYNC(0b100),
            insn::HALT()
        });
        s.run(50000);
        EXPECT(s.dut->done == 1, "layernorm dispatch path should halt cleanly");
        EXPECT(s.dut->fault == 0, "layernorm dispatch path should not fault");
        TEST_PASS("layernorm_dispatch_sync");
    }

    {
        SimHarness s;
        s.load({
            insn::CONFIG_TILE(1, 1, 1),
            insn::SET_SCALE(0, 0x3800),
            insn::SET_SCALE(1, 0x3400),
            insn::GELU(0, 0, 0, 16, 0),
            insn::SYNC(0b100),
            insn::HALT()
        });
        s.run(50000);
        EXPECT(s.dut->done == 1, "gelu dispatch path should halt cleanly");
        EXPECT(s.dut->fault == 0, "gelu dispatch path should not fault");
        TEST_PASS("gelu_dispatch_sync");
    }
}

// ============================================================================
// Test: SET_SCALE from SRAM is rejected in Phase A
// ============================================================================
static void test_set_scale_from_buffer_unsupported() {
    expect_fault_program("unsupported_set_scale_abuf",
                         { insn::SET_SCALE(0, 7, 1) }, 6, 500);
    expect_fault_program("unsupported_set_scale_wbuf",
                         { insn::SET_SCALE(0, 7, 2) }, 6, 500);
    expect_fault_program("unsupported_set_scale_accum",
                         { insn::SET_SCALE(0, 7, 3) }, 6, 500);
}

// ============================================================================
// Test: multi-burst DMA lengths are supported in Phase B
// ============================================================================
static void test_multiburst_dma_supported() {
    const char* name = "multiburst_dma_supported";
    SimHarness s;
    s.load({
        insn::SET_ADDR_LO(0, 0),
        insn::SET_ADDR_HI(0, 0),
        insn::LOAD(0, 0, 257, 0, 0),
        insn::SYNC(0b001),
        insn::STORE(0, 0, 257, 0, 0),
        insn::SYNC(0b001),
        insn::HALT()
    });
    s.run(20000);
    EXPECT(s.dut->done == 1, "done after multi-burst LOAD/STORE");
    EXPECT(s.dut->fault == 0, "no fault for multi-burst DMA");
    TEST_PASS(name);
}

// ============================================================================
// Test: MATMUL without prior CONFIG_TILE → FAULT_NO_CONFIG
// ============================================================================
static void test_matmul_no_config() {
    const char* name = "matmul_without_config_tile";
    SimHarness s;
    s.load({ insn::MATMUL(0, 0, 1, 0, 2, 0, 0) });
    s.run(500);
    EXPECT(s.dut->fault == 1, "fault for missing CONFIG_TILE");
    EXPECT(s.dut->fault_code == 4, "fault_code = FAULT_NO_CONFIG (4)");
    TEST_PASS(name);
}

// ============================================================================
// Test: fetch RRESP fault is surfaced architecturally
// ============================================================================
static void test_fetch_rresp_fault() {
    const char* name = "fetch_rresp_fault";
    SimHarness s;
    s.load({ insn::NOP(), insn::HALT() });
    s.dram.inject_next_read(/*resp=*/2);
    s.run(500);
    EXPECT(s.dut->fault == 1, "fault asserted for fetch RRESP error");
    EXPECT(s.dut->fault_code == 2, "fault_code = FAULT_DRAM_OOB (2)");
    TEST_PASS(name);
}

// ============================================================================
// Test: single-beat fetch without RLAST faults
// ============================================================================
static void test_fetch_missing_rlast_fault() {
    const char* name = "fetch_missing_rlast_fault";
    SimHarness s;
    s.load({ insn::NOP(), insn::HALT() });
    s.dram.inject_next_read(/*resp=*/0, /*force_last=*/0);
    s.run(500);
    EXPECT(s.dut->fault == 1, "fault asserted for missing fetch RLAST");
    EXPECT(s.dut->fault_code == 2, "fault_code = FAULT_DRAM_OOB (2)");
    TEST_PASS(name);
}

// ============================================================================
// Test: DMA Port A SRAM OOB faults as FAULT_SRAM_OOB
// ============================================================================
static void test_dma_sram_oob_fault() {
    const char* name = "dma_sram_oob_fault";
    SimHarness s;
    s.load({
        insn::SET_ADDR_LO(0, 0),
        insn::SET_ADDR_HI(0, 0),
        insn::LOAD(0/*ABUF*/, 8192, 1, 0, 0),
        insn::SYNC(0b001),
        insn::HALT(),
    });
    s.run(2000);
    EXPECT(s.dut->fault == 1, "fault asserted for DMA SRAM OOB");
    EXPECT(s.dut->fault_code == 3, "fault_code = FAULT_SRAM_OOB (3)");
    TEST_PASS(name);
}

// ============================================================================
// Test: systolic SRAM OOB faults as FAULT_SRAM_OOB
// ============================================================================
static void test_systolic_sram_oob_fault() {
    const char* name = "systolic_sram_oob_fault";
    SimHarness s;
    s.load({
        insn::CONFIG_TILE(1, 1, 1),
        insn::MATMUL(0/*ABUF*/, 8192, 1/*WBUF*/, 0, 2/*ACCUM*/, 0, 0),
        insn::SYNC(0b010),
        insn::HALT(),
    });
    s.run(5000);
    EXPECT(s.dut->fault == 1, "fault asserted for systolic SRAM OOB");
    EXPECT(s.dut->fault_code == 3, "fault_code = FAULT_SRAM_OOB (3)");
    TEST_PASS(name);
}

// ============================================================================
// Test: Long sequence — 10 NOPs + HALT
// ============================================================================
static void test_long_nop_sequence() {
    const char* name = "ten_nops_then_halt";
    SimHarness s;
    std::vector<uint64_t> prog(10, insn::NOP());
    prog.push_back(insn::HALT());
    s.load(prog);
    s.run(10000);
    EXPECT(s.dut->done  == 1, "done after 10 NOPs + HALT");
    EXPECT(s.dut->fault == 0, "no fault");
    TEST_PASS(name);
}

// ============================================================================
// Test: LOAD instruction dispatched (DMA stub → no stall)
// ============================================================================
static void test_load_dispatch() {
    const char* name = "load_dispatch_no_stall";
    SimHarness s;
    s.load({
        insn::SET_ADDR_LO(0, 0),
        insn::SET_ADDR_HI(0, 0),
        insn::LOAD(/*ABUF*/0, /*sram_off*/0, /*xfer_len*/16, /*addr_reg*/0, /*dram_off*/0),
        insn::SYNC(0b001),  // wait for DMA (immediately done, busy=0)
        insn::HALT()
    });
    s.run();
    EXPECT(s.dut->done  == 1, "done after LOAD + SYNC + HALT");
    EXPECT(s.dut->fault == 0, "no fault");
    TEST_PASS(name);
}

// ============================================================================
// main
// ============================================================================
int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);

    test_nop_halt();
    test_multi_nop_halt();
    test_immediate_halt();
    test_config_tile();
    test_config_tile_then_matmul();
    test_set_scale();
    test_set_addr();
    test_sync_nop();
    test_sync_all_idle();
    test_illegal_opcode();
    test_stage_e_paths();
    test_sfu_no_config_faults();
    test_sfu_dispatch_paths();
    test_set_scale_from_buffer_unsupported();
    test_multiburst_dma_supported();
    test_matmul_no_config();
    test_fetch_rresp_fault();
    test_fetch_missing_rlast_fault();
    test_dma_sram_oob_fault();
    test_systolic_sram_oob_fault();
    test_long_nop_sequence();
    test_load_dispatch();

    printf("\n%d / %d tests passed\n", tests_pass, tests_run);
    if (tests_pass != tests_run) std::exit(1);
    return 0;
}
