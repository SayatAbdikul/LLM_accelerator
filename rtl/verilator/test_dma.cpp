// Verilator DMA tests for TACCEL Phase 2.
//
// Tests:
//   load_single_beat   — LOAD 1×16 bytes; verify via STORE roundtrip
//   load_multi_beat    — LOAD 16×16 bytes (256 B); verify via STORE roundtrip
//   store_roundtrip    — LOAD src→SRAM, STORE SRAM→dst; src==dst bytes
//   load_to_wbuf       — LOAD to WBUF buffer
//   load_to_accum      — LOAD to ACCUM buffer
//   addr_reg_r1        — R1 used for LOAD, R0 untouched
//   dram_oob_fault     — STORE with end_addr > DRAM_SIZE → fault_code=2
//   load_dispatch_async — LOAD dispatched (non-blocking) → SYNC(001) waits → HALT

#include "Vtaccel_top.h"
#include "Vtaccel_top___024root.h"
#include "verilated.h"
#include "include/testbench.h"

#include <cassert>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

static int tests_run  = 0;
static int tests_pass = 0;

#define TEST_PASS(name) do { \
    printf("PASS: %s\n", name); tests_pass++; tests_run++; } while(0)
#define TEST_FAIL(name, msg) do { \
    fprintf(stderr, "FAIL: %s — %s\n", name, msg); std::exit(1); } while(0)

using tbutil::SimHarness;
using tbutil::sram_write_row;
using tbutil::sram_read_row;
using tbutil::make_pattern;
constexpr int BUF_ABUF_ID  = tbutil::BUF_ABUF_ID;
constexpr int BUF_WBUF_ID  = tbutil::BUF_WBUF_ID;
constexpr int BUF_ACCUM_ID = tbutil::BUF_ACCUM_ID;

static void expect_dram_bytes(AXI4SlaveModel& dram, uint64_t addr,
                              const std::vector<uint8_t>& expected,
                              const char* name) {
    for (size_t i = 0; i < expected.size(); ++i) {
        if (dram.read_byte(addr + i) != expected[i])
            TEST_FAIL(name, "DRAM data mismatch");
    }
}

static void expect_dram_bytes_16(AXI4SlaveModel& dram, uint64_t addr,
                                 const uint8_t expected[16], const char* name) {
    for (int i = 0; i < 16; ++i) {
        if (dram.read_byte(addr + i) != expected[i])
            TEST_FAIL(name, "DRAM data mismatch");
    }
}

static void run_roundtrip_case(const char* name, int buf_id, int beats,
                               uint64_t src_addr, uint64_t dst_addr,
                               uint8_t seed) {
    SimHarness s;
    const size_t nbytes = size_t(beats) * 16;
    std::vector<uint8_t> src = make_pattern(nbytes, seed);
    s.dram.write_bytes(src_addr, src.data(), src.size());

    s.load({
        insn::SET_ADDR_LO(0, src_addr), insn::SET_ADDR_HI(0, 0),
        insn::LOAD(buf_id, 0, beats, 0, 0),
        insn::SYNC(0b001),
        insn::SET_ADDR_LO(1, dst_addr), insn::SET_ADDR_HI(1, 0),
        insn::STORE(buf_id, 0, beats, 1, 0),
        insn::SYNC(0b001),
        insn::HALT(),
    });

    s.run(2000000);
    if (!s.dut->done || s.dut->fault) TEST_FAIL(name, "did not halt cleanly");
    expect_dram_bytes(s.dram, dst_addr, src, name);
    TEST_PASS(name);
}

static void expect_dma_fault_after_injection(const char* name, int xfer_len,
                                             int beat_index, int resp,
                                             int force_last) {
    SimHarness s;
    constexpr uint64_t SRC = 0x1D0000;
    std::vector<uint8_t> src = make_pattern(size_t(xfer_len) * 16, 0x44);
    s.dram.write_bytes(SRC, src.data(), src.size());

    s.load({
        insn::SET_ADDR_LO(0, SRC), insn::SET_ADDR_HI(0, 0),
        insn::LOAD(BUF_ABUF_ID, 0, xfer_len, 0, 0),
        insn::SYNC(0b001),
        insn::HALT(),
    });

    s.start();

    bool dma_ar_seen = false;
    bool injected = false;
    uint64_t dma_read_base = 0;

    for (int cycle = 0; cycle < 20000; ++cycle) {
      s.step();

      const auto& read_log = s.dram.read_addr_log();
      if (!dma_ar_seen && !read_log.empty() && read_log.back() == SRC) {
          dma_ar_seen = true;
          dma_read_base = s.dram.read_beat_count();
      }

      if (dma_ar_seen && !injected &&
          s.dram.read_beat_count() == dma_read_base + uint64_t(beat_index)) {
          s.dram.inject_next_read(resp, force_last);
          injected = true;
      }

      if (s.dut->done || s.dut->fault)
          break;
    }

    if (!injected) TEST_FAIL(name, "did not inject DMA read fault");
    if (!s.dut->fault) TEST_FAIL(name, "expected fault=1");
    if (s.dut->fault_code != 2) TEST_FAIL(name, "expected fault_code=2 (DRAM_OOB)");
    TEST_PASS(name);
}

// ============================================================================
// test: LOAD 1 beat writes expected bytes directly into ABUF SRAM row 0
// ============================================================================
static void test_load_16bytes_to_sram_direct() {
    const char* name = "load_16bytes_to_sram_direct";
    SimHarness s;

    constexpr uint64_t SRC = 0x110000;
    uint8_t src[16];
    for (int i = 0; i < 16; i++) src[i] = (uint8_t)(0x55 + i);
    s.dram.write_bytes(SRC, src, 16);

    s.load({
        insn::SET_ADDR_LO(0, SRC), insn::SET_ADDR_HI(0, 0),
        insn::LOAD(0/*ABUF*/, 0, 1, 0, 0),
        insn::SYNC(0b001),
        insn::HALT(),
    });

    s.run();
    if (!s.dut->done || s.dut->fault) TEST_FAIL(name, "did not halt cleanly");

    uint8_t got[16] = {};
    sram_read_row(s.dut.get(), BUF_ABUF_ID, 0, got);
    if (std::memcmp(got, src, 16) != 0) TEST_FAIL(name, "ABUF row mismatch after LOAD");
    TEST_PASS(name);
}

// ============================================================================
// test: STORE 1 beat reads bytes directly from ABUF SRAM row 0 into DRAM
// ============================================================================
static void test_store_16bytes_from_sram_direct() {
    const char* name = "store_16bytes_from_sram_direct";
    SimHarness s;

    constexpr uint64_t DST = 0x210000;
    uint8_t src[16];
    for (int i = 0; i < 16; i++) src[i] = (uint8_t)(0xA5 ^ i);

    sram_write_row(s.dut.get(), BUF_ABUF_ID, 0, src);

    s.load({
        insn::SET_ADDR_LO(1, DST), insn::SET_ADDR_HI(1, 0),
        insn::STORE(0/*ABUF*/, 0, 1, 1, 0),
        insn::SYNC(0b001),
        insn::HALT(),
    });

    s.run();
    if (!s.dut->done || s.dut->fault) TEST_FAIL(name, "did not halt cleanly");
    for (int i = 0; i < 16; i++) {
        if (s.dram.read_byte(DST + i) != src[i]) TEST_FAIL(name, "DRAM mismatch after STORE");
    }
    TEST_PASS(name);
}

// ============================================================================
// test: LOAD 1 beat → STORE roundtrip (ABUF)
// ============================================================================
static void test_load_single_beat() {
    const char* name = "load_single_beat";
    SimHarness s;

    // Write 16 known bytes at DRAM address 0x10000 (src)
    constexpr uint64_t SRC_ADDR  = 0x10000;
    constexpr uint64_t DST_ADDR  = 0x20000;
    uint8_t src_data[16];
    for (int i = 0; i < 16; i++) src_data[i] = (uint8_t)(0xA0 + i);
    s.dram.write_bytes(SRC_ADDR, src_data, 16);

    // Program:
    //   SET_ADDR_LO R0, SRC_ADDR   ; R0 = SRC
    //   LOAD ABUF[0], xfer=1, R0, dram_off=0
    //   SYNC 001
    //   SET_ADDR_LO R1, DST_ADDR   ; R1 = DST
    //   STORE ABUF[0], xfer=1, R1, dram_off=0
    //   SYNC 001
    //   HALT
    s.load({
        insn::SET_ADDR_LO(0, SRC_ADDR),
        insn::SET_ADDR_HI(0, 0),
        insn::LOAD (0/*ABUF*/, 0, 1, 0, 0),
        insn::SYNC (0b001),
        insn::SET_ADDR_LO(1, DST_ADDR),
        insn::SET_ADDR_HI(1, 0),
        insn::STORE(0/*ABUF*/, 0, 1, 1, 0),
        insn::SYNC (0b001),
        insn::HALT(),
    });

    s.run();
    if (!s.dut->done) TEST_FAIL(name, "DUT did not halt");
    if (s.dut->fault) TEST_FAIL(name, "unexpected fault");

    // Verify DRAM[DST] == src_data
    for (int i = 0; i < 16; i++) {
        uint8_t got = s.dram.read_byte(DST_ADDR + i);
        if (got != src_data[i]) TEST_FAIL(name, "data mismatch");
    }
    TEST_PASS(name);
}

// ============================================================================
// test: LOAD 16 beats (256 bytes) → STORE roundtrip
// ============================================================================
static void test_load_multi_beat() {
    const char* name = "load_multi_beat";
    SimHarness s;

    constexpr uint64_t SRC = 0x30000;
    constexpr uint64_t DST = 0x40000;
    constexpr int NBYTES = 256;   // 16 rows × 16 bytes

    uint8_t src[NBYTES];
    for (int i = 0; i < NBYTES; i++) src[i] = (uint8_t)i;
    s.dram.write_bytes(SRC, src, NBYTES);

    s.load({
        insn::SET_ADDR_LO(0, SRC),
        insn::SET_ADDR_HI(0, 0),
        insn::LOAD (0, 0, 16, 0, 0),   // xfer_len=16 → 256 bytes
        insn::SYNC (0b001),
        insn::SET_ADDR_LO(1, DST),
        insn::SET_ADDR_HI(1, 0),
        insn::STORE(0, 0, 16, 1, 0),
        insn::SYNC (0b001),
        insn::HALT(),
    });

    s.run();
    if (!s.dut->done) TEST_FAIL(name, "DUT did not halt");
    if (s.dut->fault) TEST_FAIL(name, "unexpected fault");

    for (int i = 0; i < NBYTES; i++) {
        if (s.dram.read_byte(DST + i) != src[i]) TEST_FAIL(name, "data mismatch");
    }
    TEST_PASS(name);
}

// ============================================================================
// test: LOAD to WBUF
// ============================================================================
static void test_load_to_wbuf() {
    const char* name = "load_to_wbuf";
    SimHarness s;

    constexpr uint64_t SRC = 0x50000;
    constexpr uint64_t DST = 0x60000;
    uint8_t src[16];
    for (int i = 0; i < 16; i++) src[i] = (uint8_t)(0xB0 + i);
    s.dram.write_bytes(SRC, src, 16);

    s.load({
        insn::SET_ADDR_LO(0, SRC), insn::SET_ADDR_HI(0, 0),
        insn::LOAD (1/*WBUF*/, 0, 1, 0, 0),
        insn::SYNC (0b001),
        insn::SET_ADDR_LO(1, DST), insn::SET_ADDR_HI(1, 0),
        insn::STORE(1/*WBUF*/, 0, 1, 1, 0),
        insn::SYNC (0b001),
        insn::HALT(),
    });

    s.run();
    if (!s.dut->done || s.dut->fault) TEST_FAIL(name, "did not halt cleanly");
    for (int i = 0; i < 16; i++) {
        if (s.dram.read_byte(DST + i) != src[i]) TEST_FAIL(name, "data mismatch");
    }
    TEST_PASS(name);
}

// ============================================================================
// test: LOAD to ACCUM
// ============================================================================
static void test_load_to_accum() {
    const char* name = "load_to_accum";
    SimHarness s;

    constexpr uint64_t SRC = 0x70000;
    constexpr uint64_t DST = 0x80000;
    uint8_t src[16];
    for (int i = 0; i < 16; i++) src[i] = (uint8_t)(0xC0 + i);
    s.dram.write_bytes(SRC, src, 16);

    s.load({
        insn::SET_ADDR_LO(0, SRC), insn::SET_ADDR_HI(0, 0),
        insn::LOAD (2/*ACCUM*/, 0, 1, 0, 0),
        insn::SYNC (0b001),
        insn::SET_ADDR_LO(1, DST), insn::SET_ADDR_HI(1, 0),
        insn::STORE(2/*ACCUM*/, 0, 1, 1, 0),
        insn::SYNC (0b001),
        insn::HALT(),
    });

    s.run();
    if (!s.dut->done || s.dut->fault) TEST_FAIL(name, "did not halt cleanly");
    for (int i = 0; i < 16; i++) {
        if (s.dram.read_byte(DST + i) != src[i]) TEST_FAIL(name, "data mismatch");
    }
    TEST_PASS(name);
}

// ============================================================================
// test: R1 used independently from R0
// ============================================================================
static void test_addr_reg_independence() {
    const char* name = "addr_reg_independence";
    SimHarness s;

    // Data at two separate DRAM regions
    constexpr uint64_t SRC_A = 0x90000;
    constexpr uint64_t DST_A = 0xA0000;
    constexpr uint64_t SRC_B = 0xB0000;
    constexpr uint64_t DST_B = 0xC0000;

    uint8_t pa[16], pb[16];
    for (int i = 0; i < 16; i++) { pa[i] = (uint8_t)(0x11 + i); pb[i] = (uint8_t)(0x88 + i); }
    s.dram.write_bytes(SRC_A, pa, 16);
    s.dram.write_bytes(SRC_B, pb, 16);

    // Use R2 for A (src), R3 for B (src), R0 for A (dst), R1 for B (dst)
    s.load({
        insn::SET_ADDR_LO(2, SRC_A), insn::SET_ADDR_HI(2, 0),
        insn::SET_ADDR_LO(3, SRC_B), insn::SET_ADDR_HI(3, 0),
        insn::LOAD(0/*ABUF*/, 0, 1, 2, 0),   // load A to ABUF[0]
        insn::SYNC(0b001),
        insn::LOAD(1/*WBUF*/, 0, 1, 3, 0),   // load B to WBUF[0]
        insn::SYNC(0b001),
        insn::SET_ADDR_LO(0, DST_A), insn::SET_ADDR_HI(0, 0),
        insn::SET_ADDR_LO(1, DST_B), insn::SET_ADDR_HI(1, 0),
        insn::STORE(0/*ABUF*/, 0, 1, 0, 0),
        insn::SYNC(0b001),
        insn::STORE(1/*WBUF*/, 0, 1, 1, 0),
        insn::SYNC(0b001),
        insn::HALT(),
    });

    s.run();
    if (!s.dut->done || s.dut->fault) TEST_FAIL(name, "did not halt cleanly");
    for (int i = 0; i < 16; i++) {
        if (s.dram.read_byte(DST_A + i) != pa[i]) TEST_FAIL(name, "A data mismatch");
        if (s.dram.read_byte(DST_B + i) != pb[i]) TEST_FAIL(name, "B data mismatch");
    }
    TEST_PASS(name);
}

// ============================================================================
// test: DRAM OOB → FAULT_DRAM_OOB (code 2)
// ============================================================================
static void test_dram_oob_fault() {
    const char* name = "dram_oob_fault";
    SimHarness s;

    // Set R0 to near end of 16 MB DRAM; xfer_len=2 → 32 bytes past end
    // DRAM_SIZE = 16 MB = 0x1000000
    // addr = 0xFFFFF0 = 16777200; end = 16777200 + 32 = 0x1000010 > 0x1000000
    constexpr uint64_t NEAR_END = 0xFFFFF0;

    s.load({
        insn::SET_ADDR_LO(0, NEAR_END),
        insn::SET_ADDR_HI(0, 0),
        insn::LOAD(0, 0, 2, 0, 0),   // OOB: 2 × 16 bytes past NEAR_END
        insn::SYNC(0b001),            // will see dma_fault in SYNC_WAIT
        insn::HALT(),                 // unreachable
    });

    s.run(200000);
    if (!s.dut->fault)      TEST_FAIL(name, "expected fault=1");
    if (s.dut->fault_code != 2) TEST_FAIL(name, "expected fault_code=2 (DRAM_OOB)");
    TEST_PASS(name);
}

// ============================================================================
// test: STORE OOB -> FAULT_DRAM_OOB (code 2)
// ============================================================================
static void test_store_oob_fault() {
    const char* name = "store_oob_fault";
    SimHarness s;

    constexpr uint64_t NEAR_END = 0xFFFFF0;

    s.load({
        insn::SET_ADDR_LO(0, NEAR_END),
        insn::SET_ADDR_HI(0, 0),
        insn::STORE(0, 0, 2, 0, 0),  // OOB: 2 x 16B from 0xFFFFF0 crosses 16MB
        insn::SYNC(0b001),
        insn::HALT(),
    });

    s.run(200000);
    if (!s.dut->fault) TEST_FAIL(name, "expected fault=1");
    if (s.dut->fault_code != 2) TEST_FAIL(name, "expected fault_code=2 (DRAM_OOB)");
    TEST_PASS(name);
}

// ============================================================================
// test: LOAD dispatch is non-blocking; SYNC(001) waits correctly
// ============================================================================
static void test_load_dispatch_async() {
    const char* name = "load_dispatch_async";
    SimHarness s;

    constexpr uint64_t SRC = 0xD0000;
    uint8_t src[16] = {};
    s.dram.write_bytes(SRC, src, 16);

    // LOAD dispatched → NOP → SYNC(001) → HALT
    // SYNC should stall until DMA completes, then HALT.
    s.load({
        insn::SET_ADDR_LO(0, SRC), insn::SET_ADDR_HI(0, 0),
        insn::LOAD(0, 0, 1, 0, 0),
        insn::NOP(),                  // prove pipeline continues after dispatch
        insn::SYNC(0b001),
        insn::HALT(),
    });

    s.run();
    if (!s.dut->done) TEST_FAIL(name, "DUT did not halt");
    if (s.dut->fault) TEST_FAIL(name, "unexpected fault");
    TEST_PASS(name);
}

// ============================================================================
// test: dram_off field shifts address correctly
// ============================================================================
static void test_dram_offset() {
    const char* name = "dram_offset";
    SimHarness s;

    // R0 = 0x50000 (base); dram_off=4 → effective = 0x50000 + 4*16 = 0x50040
    constexpr uint64_t BASE = 0x50040;   // effective address
    constexpr uint64_t DST  = 0xE0000;
    uint8_t src[16];
    for (int i = 0; i < 16; i++) src[i] = (uint8_t)(0xD0 + i);
    s.dram.write_bytes(BASE, src, 16);   // write at effective address

    s.load({
        // base address is 0x50000; dram_off=4 → 0x50000 + 64 = 0x50040
        insn::SET_ADDR_LO(0, 0x50000), insn::SET_ADDR_HI(0, 0),
        insn::LOAD(0, 0, 1, 0, 4),   // dram_off=4 → byte +64
        insn::SYNC(0b001),
        insn::SET_ADDR_LO(1, DST), insn::SET_ADDR_HI(1, 0),
        insn::STORE(0, 0, 1, 1, 0),
        insn::SYNC(0b001),
        insn::HALT(),
    });

    s.run();
    if (!s.dut->done || s.dut->fault) TEST_FAIL(name, "did not halt cleanly");
    for (int i = 0; i < 16; i++) {
        if (s.dram.read_byte(DST + i) != src[i]) TEST_FAIL(name, "data mismatch");
    }
    TEST_PASS(name);
}

// ============================================================================
// test: xfer_len=0 LOAD leaves SRAM untouched and completes without DMA busy
// ============================================================================
static void test_zero_length_load_noop() {
    const char* name = "zero_length_load_noop";
    SimHarness s;

    constexpr uint64_t SRC = 0x102000;
    constexpr uint64_t DST = 0x104000;
    uint8_t sentinel[16];
    for (int i = 0; i < 16; ++i) sentinel[i] = uint8_t(0xE0 + i);
    sram_write_row(s.dut.get(), BUF_ABUF_ID, 0, sentinel);

    s.load({
        insn::SET_ADDR_LO(0, SRC), insn::SET_ADDR_HI(0, 0),
        insn::LOAD(BUF_ABUF_ID, 0, 0, 0, 0),
        insn::SET_ADDR_LO(1, DST), insn::SET_ADDR_HI(1, 0),
        insn::STORE(BUF_ABUF_ID, 0, 1, 1, 0),
        insn::SYNC(0b001),
        insn::HALT(),
    });

    s.run(20000);
    if (!s.dut->done || s.dut->fault) TEST_FAIL(name, "did not halt cleanly");
    expect_dram_bytes_16(s.dram, DST, sentinel, name);
    TEST_PASS(name);
}

// ============================================================================
// test: xfer_len=0 STORE leaves DRAM untouched and completes without fault
// ============================================================================
static void test_zero_length_store_noop() {
    const char* name = "zero_length_store_noop";
    SimHarness s;

    constexpr uint64_t DST = 0x106000;
    std::vector<uint8_t> sentinel = make_pattern(16, 0xA3);
    s.dram.write_bytes(DST, sentinel.data(), sentinel.size());

    s.load({
        insn::SET_ADDR_LO(0, DST), insn::SET_ADDR_HI(0, 0),
        insn::STORE(BUF_ABUF_ID, 0, 0, 0, 0),
        insn::HALT(),
    });

    s.run(5000);
    if (!s.dut->done || s.dut->fault) TEST_FAIL(name, "did not halt cleanly");
    expect_dram_bytes(s.dram, DST, sentinel, name);
    TEST_PASS(name);
}

// ============================================================================
// test: boundary and compiler-scale DMA transfers roundtrip correctly
// ============================================================================
static void test_large_roundtrip_lengths() {
    run_roundtrip_case("roundtrip_257_beats",  BUF_ABUF_ID, 257,  0x200000, 0x220000, 0x11);
    run_roundtrip_case("roundtrip_511_beats",  BUF_ABUF_ID, 511,  0x240000, 0x260000, 0x21);
    run_roundtrip_case("roundtrip_512_beats",  BUF_ABUF_ID, 512,  0x280000, 0x2A0000, 0x31);
    run_roundtrip_case("roundtrip_2304_beats", BUF_WBUF_ID, 2304, 0x2C0000, 0x300000, 0x41);
    run_roundtrip_case("roundtrip_2352_beats", BUF_WBUF_ID, 2352, 0x340000, 0x380000, 0x51);
    run_roundtrip_case("roundtrip_9216_beats", BUF_WBUF_ID, 9216, 0x3C0000, 0x500000, 0x61);
}

// ============================================================================
// test: fetch can interleave between DMA read bursts
// ============================================================================
static void test_fetch_interleave_between_dma_bursts() {
    const char* name = "fetch_interleave_between_dma_bursts";
    SimHarness s;

    constexpr uint64_t SRC = 0x540000;
    std::vector<uint8_t> src = make_pattern(size_t(257) * 16, 0x77);
    s.dram.write_bytes(SRC, src.data(), src.size());

    s.load({
        insn::SET_ADDR_LO(0, SRC), insn::SET_ADDR_HI(0, 0),
        insn::LOAD(BUF_WBUF_ID, 0, 257, 0, 0),
        insn::NOP(),
        insn::SYNC(0b001),
        insn::HALT(),
    });

    s.run(200000);
    if (!s.dut->done || s.dut->fault) TEST_FAIL(name, "did not halt cleanly");

    const auto& read_log = s.dram.read_addr_log();
    bool interleaved = false;
    for (size_t i = 0; i + 2 < read_log.size(); ++i) {
        if (read_log[i] == SRC &&
            read_log[i + 1] < 0x1000 &&
            read_log[i + 2] == (SRC + 256 * 16)) {
            interleaved = true;
            break;
        }
    }

    if (!interleaved) TEST_FAIL(name, "expected fetch AR between DMA bursts");
    TEST_PASS(name);
}

// ============================================================================
// test: DMA read faults are surfaced for first/middle/final beats and bad RLAST
// ============================================================================
static void test_dma_read_faults() {
    expect_dma_fault_after_injection("load_rresp_fault_first_beat", 4, 0, 2, -1);
    expect_dma_fault_after_injection("load_rresp_fault_middle_beat", 4, 1, 2, -1);
    expect_dma_fault_after_injection("load_rresp_fault_final_beat", 4, 3, 2, -1);
    expect_dma_fault_after_injection("load_early_rlast_fault", 4, 1, 0, 1);
    expect_dma_fault_after_injection("load_missing_final_rlast_fault", 1, 0, 0, 0);
}

// ============================================================================
// test: non-OKAY BRESP faults STORE
// ============================================================================
static void test_store_bresp_fault() {
    const char* name = "store_bresp_fault";
    SimHarness s;

    constexpr uint64_t SRC = 0x580000;
    constexpr uint64_t DST = 0x5A0000;
    std::vector<uint8_t> src = make_pattern(16, 0x99);
    s.dram.write_bytes(SRC, src.data(), src.size());
    s.dram.inject_next_bresp(2);

    s.load({
        insn::SET_ADDR_LO(0, SRC), insn::SET_ADDR_HI(0, 0),
        insn::LOAD(BUF_ABUF_ID, 0, 1, 0, 0),
        insn::SYNC(0b001),
        insn::SET_ADDR_LO(1, DST), insn::SET_ADDR_HI(1, 0),
        insn::STORE(BUF_ABUF_ID, 0, 1, 1, 0),
        insn::SYNC(0b001),
        insn::HALT(),
    });

    s.run(20000);
    if (!s.dut->fault) TEST_FAIL(name, "expected fault=1");
    if (s.dut->fault_code != 2) TEST_FAIL(name, "expected fault_code=2 (DRAM_OOB)");
    TEST_PASS(name);
}

// ============================================================================
// main
// ============================================================================
int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);

    test_load_single_beat();
    test_load_16bytes_to_sram_direct();
    test_store_16bytes_from_sram_direct();
    test_load_multi_beat();
    test_load_to_wbuf();
    test_load_to_accum();
    test_addr_reg_independence();
    test_dram_oob_fault();
    test_store_oob_fault();
    test_load_dispatch_async();
    test_dram_offset();
    test_zero_length_load_noop();
    test_zero_length_store_noop();
    test_large_roundtrip_lengths();
    test_fetch_interleave_between_dma_bursts();
    test_dma_read_faults();
    test_store_bresp_fault();

    printf("\n%d / %d tests passed\n", tests_pass, tests_run);
    return (tests_pass == tests_run) ? 0 : 1;
}
