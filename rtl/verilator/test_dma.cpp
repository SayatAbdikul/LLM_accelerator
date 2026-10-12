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

#include "test_runner.h"

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
// test: Lever D transposed LOAD — LOAD_T (R,C)→(C,R) into WBUF, STORE back,
// compare against the CPU transpose. Covers cols_log2=0 (C=16) and =2 (C=64).
// ============================================================================
static void run_transpose_case(const char* name, int rows, int cols_log2) {
    SimHarness s;
    const int C = 16 << cols_log2;
    const int R = rows;
    const int nbytes = R * C;
    const int beats = nbytes / 16;
    constexpr uint64_t SRC = 0x40000;
    constexpr uint64_t DST = 0x90000;

    std::vector<uint8_t> src(nbytes);
    for (int i = 0; i < nbytes; i++) src[i] = (uint8_t)((i * 7 + 3) & 0xFF);
    s.dram.write_bytes(SRC, src.data(), src.size());

    // reference: transposed[c*R + r] = src[r*C + c]
    std::vector<uint8_t> ref(nbytes);
    for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++)
            ref[c * R + r] = src[r * C + c];

    s.load({
        insn::SET_ADDR_LO(0, SRC), insn::SET_ADDR_HI(0, 0),
        insn::LOAD_T(1/*WBUF*/, 0, beats, 0, 0, cols_log2),
        insn::SYNC(0b001),
        insn::SET_ADDR_LO(1, DST), insn::SET_ADDR_HI(1, 0),
        insn::STORE(1/*WBUF*/, 0, beats, 1, 0),
        insn::SYNC(0b001),
        insn::HALT(),
    });

    s.run(2000000);
    if (!s.dut->done || s.dut->fault) TEST_FAIL(name, "did not halt cleanly");
    for (int i = 0; i < nbytes; i++) {
        if (s.dram.read_byte(DST + i) != ref[i]) TEST_FAIL(name, "transpose mismatch");
    }
    TEST_PASS(name);
}

static void test_transpose_load_c16() { run_transpose_case("transpose_load_c16_r32", 32, 0); }
static void test_transpose_load_c64() { run_transpose_case("transpose_load_c64_r32", 32, 2); }
static void test_transpose_load_c32() { run_transpose_case("transpose_load_c32_r16", 16, 1); }

// A strict, edge-sampled AXI slave for protocol regression. Unlike the legacy
// convenience model, this rejects illegal page-crossing bursts, varies all
// five channel delays, and checks that stalled master payloads remain stable.
// It is deliberately local: changing the timing of every historical fixture
// would obscure whether the DMA fix or the test harness changed a result.
class ProtocolMemory {
public:
    Vtaccel_top dut;
    std::vector<uint8_t> bytes;
    std::vector<std::pair<uint64_t, unsigned>> reads, writes;
    bool hold_instruction_reads_during_store = false;
    uint64_t instruction_read_wait_for_wvalid_from = ~uint64_t(0);
    unsigned minimum_w_stalls = 1;
    bool randomize_w_ready = true;
    unsigned stalled_w_same_buffer_s_reads = 0;
    uint64_t cycles = 0;
    std::vector<uint64_t> aw_cycles, w_cycles;
    uint64_t error_read_addr = ~uint64_t(0);
    int error_read_resp = 2;
    int error_read_last = -1;
    int error_bresp = 0;
    unsigned ar_stalls = 0, aw_stalls = 0, w_stalls = 0, r_gaps = 0, b_gaps = 0;

    explicit ProtocolMemory(uint32_t seed = 1) : bytes(16 * 1024 * 1024), rng(seed) {
        do_reset(&dut);
    }

    void program(const std::vector<uint64_t>& words) {
        for (size_t i = 0; i < words.size(); ++i)
            for (int j = 0; j < 8; ++j)
                bytes[i * 8 + j] = uint8_t(words[i] >> (56 - j * 8));
    }

    void tick() {
        dut.clk = 0;
        dut.eval();
        if (dut.m_axi_w_valid) write_valid_seen = true;
        if (read_active && !r_hold) {
            if (r_gaps != 0 && (random() & 3) != 0) {
                r_hold = true;
                const uint64_t addr = read_addr + read_index * 16;
                r_resp = addr == error_read_addr ? error_read_resp : 0;
                r_last = read_index + 1 == read_beats;
                if (addr == error_read_addr && error_read_last >= 0)
                    r_last = error_read_last;
                for (unsigned i = 0; i < 4; ++i) {
                    r_data[i] = 0;
                    for (unsigned j = 0; j < 4; ++j)
                        r_data[i] |= uint32_t(bytes.at(addr + i * 4 + j)) << (8 * j);
                }
            } else ++r_gaps;
        }
        if (b_pending && !b_hold) {
            if (b_gaps != 0 && (random() & 3) != 0) b_hold = true;
            else ++b_gaps;
        }
        // Directed first stalls make coverage independent of instruction
        // cache timing; random delays then explore additional alignments.
        dut.m_axi_ar_ready = ar_stalls != 0 && !read_active && !r_hold &&
                             ((random() & 3) != 0);
        if (hold_instruction_reads_during_store && dut.m_axi_ar_addr < 0x1000 &&
            (write_active || b_pending || b_hold || dut.m_axi_aw_valid))
            dut.m_axi_ar_ready = 0;
        if (!write_valid_seen && dut.m_axi_ar_addr < 0x1000 &&
            dut.m_axi_ar_addr >= instruction_read_wait_for_wvalid_from)
            dut.m_axi_ar_ready = 0;
        // Force at least one AW stall even for the one-burst shapes, then
        // randomize its remaining delay.
        dut.m_axi_aw_ready = aw_wait && !write_active && !b_pending && ((random() & 3) != 0);
        dut.m_axi_w_ready = w_stalls >= minimum_w_stalls && write_active &&
                            (!randomize_w_ready || (random() & 3) != 0);
        dut.m_axi_r_valid = r_hold;
        dut.m_axi_r_resp = r_resp;
        dut.m_axi_r_last = r_last;
        for (unsigned i = 0; i < 4; ++i) dut.m_axi_r_data[i] = r_data[i];
        dut.m_axi_b_valid = b_hold;
        dut.m_axi_b_resp = error_bresp;
        dut.eval();

        const auto* root = dut.rootp;
        if (dut.m_axi_w_valid && !dut.m_axi_w_ready &&
            root->taccel_top__DOT__sys_sram_a_en && !root->taccel_top__DOT__sys_sram_a_we &&
            !root->taccel_top__DOT__dma_sram_en &&
            root->taccel_top__DOT__sys_sram_a_buf == root->taccel_top__DOT__dma_sram_buf &&
            root->taccel_top__DOT__sys_sram_a_row != root->taccel_top__DOT__dma_sram_row)
            ++stalled_w_same_buffer_s_reads;

        if (ar_wait && (!dut.m_axi_ar_valid || dut.m_axi_ar_addr != stalled_ar ||
                        dut.m_axi_ar_len != stalled_ar_len))
            TEST_FAIL("axi_protocol", "AR changed while backpressured");
        if (aw_wait && (!dut.m_axi_aw_valid || dut.m_axi_aw_addr != stalled_aw ||
                        dut.m_axi_aw_len != stalled_aw_len))
            TEST_FAIL("axi_protocol", "AW changed while backpressured");
        if (w_wait) {
            if (!dut.m_axi_w_valid || dut.m_axi_w_last != stalled_wlast ||
                dut.m_axi_w_strb != stalled_strb)
                TEST_FAIL("axi_protocol", "W control changed while backpressured");
            for (int i = 0; i < 4; ++i)
                if (dut.m_axi_w_data[i] != stalled_wdata[i]) {
                    std::fprintf(stderr, "Concurrent same-buffer different-row Port-S reads: %u\n",
                                 stalled_w_same_buffer_s_reads);
                    TEST_FAIL("axi_protocol", "W data changed while backpressured");
                }
        }
        ar_wait = dut.m_axi_ar_valid && !dut.m_axi_ar_ready;
        aw_wait = dut.m_axi_aw_valid && !dut.m_axi_aw_ready;
        w_wait = dut.m_axi_w_valid && !dut.m_axi_w_ready;
        if (ar_wait) {
            ++ar_stalls; stalled_ar = dut.m_axi_ar_addr; stalled_ar_len = dut.m_axi_ar_len;
        }
        if (aw_wait) {
            ++aw_stalls; stalled_aw = dut.m_axi_aw_addr; stalled_aw_len = dut.m_axi_aw_len;
        }
        if (w_wait) {
            ++w_stalls; stalled_wlast = dut.m_axi_w_last; stalled_strb = dut.m_axi_w_strb;
            for (int i = 0; i < 4; ++i) stalled_wdata[i] = dut.m_axi_w_data[i];
        }
        if (dut.m_axi_r_valid && dut.m_axi_r_ready) {
            r_hold = false;
            if (++read_index == read_beats) read_active = false;
        }
        if (dut.m_axi_b_valid && dut.m_axi_b_ready) b_pending = b_hold = false;
        if (dut.m_axi_ar_valid && dut.m_axi_ar_ready) {
            check_burst(dut.m_axi_ar_addr, dut.m_axi_ar_len,
                        dut.m_axi_ar_size, dut.m_axi_ar_burst);
            read_addr = dut.m_axi_ar_addr;
            read_beats = unsigned(dut.m_axi_ar_len) + 1;
            read_index = 0;
            read_active = true;
            reads.emplace_back(read_addr, read_beats);
        }
        if (dut.m_axi_aw_valid && dut.m_axi_aw_ready) {
            check_burst(dut.m_axi_aw_addr, dut.m_axi_aw_len,
                        dut.m_axi_aw_size, dut.m_axi_aw_burst);
            write_addr = dut.m_axi_aw_addr;
            write_beats = unsigned(dut.m_axi_aw_len) + 1;
            write_index = 0;
            write_active = true;
            writes.emplace_back(write_addr, write_beats);
            aw_cycles.push_back(cycles);
        }
        if (dut.m_axi_w_valid && dut.m_axi_w_ready) {
            w_cycles.push_back(cycles);
            if (!write_active || bool(dut.m_axi_w_last) != (write_index + 1 == write_beats))
                TEST_FAIL("axi_protocol", "unexpected W beat or WLAST");
            const uint64_t addr = write_addr + write_index * 16;
            for (unsigned i = 0; i < 16; ++i)
                if ((dut.m_axi_w_strb >> i) & 1)
                    bytes.at(addr + i) = uint8_t(dut.m_axi_w_data[i / 4] >> (8 * (i % 4)));
            if (++write_index == write_beats) {
                write_active = false;
                b_pending = true;
            }
        }
        dut.clk = 1;
        dut.eval();
        ++cycles;
    }

    void run() {
        dut.start = 1;
        tick();
        dut.start = 0;
        for (int i = 0; i < 300000; ++i) {
            tick();
            if (dut.done || dut.fault) return;
        }
        TEST_FAIL("axi_protocol", "simulation timeout");
    }

private:
    uint32_t rng;
    bool write_valid_seen = false;
    bool read_active = false, write_active = false, r_hold = false;
    bool b_pending = false, b_hold = false, r_last = false;
    unsigned read_beats = 0, read_index = 0, write_beats = 0, write_index = 0;
    uint64_t read_addr = 0, write_addr = 0;
    int r_resp = 0;
    uint32_t r_data[4] = {};
    bool ar_wait = false, aw_wait = false, w_wait = false, stalled_wlast = false;
    uint64_t stalled_ar = 0, stalled_aw = 0;
    unsigned stalled_ar_len = 0, stalled_aw_len = 0, stalled_strb = 0;
    uint32_t stalled_wdata[4] = {};

    uint32_t random() {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        return rng;
    }
    void check_burst(uint64_t addr, unsigned len, unsigned size, unsigned burst) {
        if ((addr & 15) || size != 4 || burst != 1 ||
            (addr & 4095) + (len + 1) * 16 > 4096 || addr + (len + 1) * 16 > bytes.size())
            TEST_FAIL("axi_protocol", "unaligned, out-of-range or 4-KiB-crossing AXI burst");
    }
};

using BurstLog = std::vector<std::pair<uint64_t, unsigned>>;

static uint8_t strict_source_byte(uint64_t address) {
    // Mix the full address so adjacent transpose stripes and 4-KiB pages
    // cannot accidentally share the old fixture's 256-byte-period pattern.
    uint64_t mixed = address + 0x9e3779b97f4a7c15ULL;
    mixed = (mixed ^ (mixed >> 30)) * 0xbf58476d1ce4e5b9ULL;
    mixed = (mixed ^ (mixed >> 27)) * 0x94d049bb133111ebULL;
    return uint8_t(mixed ^ (mixed >> 31));
}

static BurstLog expected_burst_log(uint64_t address, unsigned nbytes,
                                  unsigned stripe_bytes) {
    BurstLog result;
    // A normal transfer has one logical stripe. Transpose loads fetch each
    // 16-row stripe separately, splitting only where an AXI page ends.
    for (unsigned stripe = 0; stripe < nbytes; stripe += stripe_bytes) {
        unsigned left = stripe_bytes;
        while (left != 0) {
            const unsigned page_left = 4096 - unsigned(address & 4095);
            const unsigned bytes = left < page_left ? left : page_left;
            result.emplace_back(address, bytes / 16);
            address += bytes;
            left -= bytes;
        }
    }
    return result;
}

static void test_page_boundaries_with_backpressure() {
    // Exercise short first bursts, more than one page, one-beat burst tails,
    // and each supported transpose width. Addresses deliberately differ
    // between LOAD and STORE so their burst decompositions differ too.
    const unsigned offsets[] = {0xff0, 0xfe0, 0x130, 0};
    for (int transpose = -1; transpose < 3; ++transpose) {
        for (unsigned off : offsets) {
            ProtocolMemory s(1 + off + 17 * (transpose + 1));
            const uint64_t src = 0x100000 + off;
            const uint64_t dst = 0x200000 + ((off + 0x10) & 0xfff);
            const unsigned columns = transpose < 0 ? 16 : (16u << transpose);
            const unsigned rows = transpose < 0 ? 513 : 80;
            const unsigned nbytes = transpose < 0 ? rows * 16 : rows * columns;
            std::vector<uint8_t> expected(nbytes);
            for (unsigned i = 0; i < nbytes; ++i)
                s.bytes[src + i] = strict_source_byte(src + i);
            for (unsigned i = 0; i < nbytes; ++i) {
                const unsigned dest = transpose < 0 ? i : (i % columns) * rows + i / columns;
                expected[dest] = s.bytes[src + i];
            }
            s.program({
                insn::SET_ADDR_LO(0, src), insn::SET_ADDR_HI(0, 0),
                transpose < 0 ? insn::LOAD(BUF_WBUF_ID, 0, nbytes / 16, 0, 0)
                              : insn::LOAD_T(BUF_WBUF_ID, 0, nbytes / 16, 0, 0, transpose),
                insn::NOP(), insn::NOP(), insn::SYNC(1),
                insn::SET_ADDR_LO(1, dst), insn::SET_ADDR_HI(1, 0),
                insn::STORE(BUF_WBUF_ID, 0, nbytes / 16, 1, 0), insn::SYNC(1), insn::HALT(),
            });
            s.run();
            if (!s.dut.done || s.dut.fault)
                TEST_FAIL("page_boundary_backpressure", "unexpected fault");
            BurstLog dma_reads;
            for (const auto& burst : s.reads)
                if (burst.first >= 0x1000) dma_reads.push_back(burst);
            if (dma_reads != expected_burst_log(src, nbytes,
                    transpose < 0 ? nbytes : 16 * columns))
                TEST_FAIL("page_boundary_backpressure", "DMA AR addresses/lengths differ from expected page/stripe sequence");
            if (s.writes != expected_burst_log(dst, nbytes, nbytes))
                TEST_FAIL("page_boundary_backpressure", "DMA AW addresses/lengths differ from expected page sequence");
            for (unsigned i = 0; i < nbytes; ++i)
                if (s.bytes[dst + i] != expected[i])
                    TEST_FAIL("page_boundary_backpressure", "roundtrip/transpose mismatch");
            if (s.ar_stalls == 0 || s.aw_stalls == 0 || s.w_stalls == 0 ||
                s.r_gaps == 0 || s.b_gaps == 0)
                TEST_FAIL("page_boundary_backpressure", "backpressure scenario not exercised");
        }
    }
    TEST_PASS("page_boundary_backpressure_16_shapes");
}

static void test_effective_address_overflow() {
    for (bool store : {false, true}) {
        ProtocolMemory s;
        s.program({
            insn::SET_ADDR_LO(0, 0x0ffffff0), insn::SET_ADDR_HI(0, 0x0fffffff),
            store ? insn::STORE(BUF_ABUF_ID, 0, 1, 0, 1)
                  : insn::LOAD(BUF_ABUF_ID, 0, 1, 0, 1),
            insn::SYNC(1), insn::HALT(),
        });
        const uint8_t sentinel[16] = {0x5a};
        sram_write_row(&s.dut, BUF_ABUF_ID, 0, sentinel);
        s.run();
        if (!s.dut.fault || s.dut.fault_code != 2 || !s.writes.empty())
            TEST_FAIL("effective_address_overflow", "missing pre-side-effect DRAM fault");
        uint8_t got[16];
        sram_read_row(&s.dut, BUF_ABUF_ID, 0, got);
        if (std::memcmp(got, sentinel, 16))
            TEST_FAIL("effective_address_overflow", "overflow LOAD modified SRAM");
    }
    TEST_PASS("effective_address_overflow_load_store");
}

static void test_split_transpose_response_errors() {
    for (int kind = 0; kind < 3; ++kind) {
        ProtocolMemory s(10 + kind);
        constexpr uint64_t SRC = 0x100ff0;
        s.program({insn::SET_ADDR_LO(0, SRC), insn::SET_ADDR_HI(0, 0),
                   insn::LOAD_T(BUF_WBUF_ID, 0, 64, 0, 0, 2),
                   insn::SYNC(1), insn::HALT()});
        s.error_read_addr = kind == 0 ? SRC + 16 : SRC;
        s.error_read_resp = kind == 0 ? 2 : 0;
        s.error_read_last = kind == 0 ? -1 : (kind == 1 ? 0 : 1);
        // kind 2 injects early RLAST into the second, longer subburst.
        if (kind == 2) s.error_read_addr = SRC + 16;
        const uint8_t sentinel[16] = {0xa5};
        sram_write_row(&s.dut, BUF_WBUF_ID, 0, sentinel);
        s.run();
        if (!s.dut.fault || s.dut.fault_code != 2)
            TEST_FAIL("split_transpose_response_error", "missing DRAM fault");
        uint8_t got[16];
        sram_read_row(&s.dut, BUF_WBUF_ID, 0, got);
        if (std::memcmp(got, sentinel, 16))
            TEST_FAIL("split_transpose_response_error", "incomplete stripe was written to SRAM");
    }
    TEST_PASS("split_transpose_response_errors");
}

static void test_store_instruction_coherence() {
    ProtocolMemory s;
    s.hold_instruction_reads_during_store = true;
    constexpr uint64_t SRC = 0x100000;
    const uint64_t store = insn::STORE(BUF_ABUF_ID, 0, 1, 1, 0);
    const uint64_t patch[] = {store, insn::HALT()};
    for (unsigned i = 0; i < 16; ++i)
        s.bytes[SRC + i] = uint8_t(patch[i / 8] >> (56 - 8 * (i % 8)));
    // Fetching STORE at even PC 6 caches the old SYNC at odd PC 7. The
    // store replaces that same beat. AXI deliberately delays the next fetch
    // until the write completes; the enabled cache must also observe HALT.
    s.program({insn::SET_ADDR_LO(0, SRC), insn::SET_ADDR_HI(0, 0),
               insn::LOAD(BUF_ABUF_ID, 0, 1, 0, 0), insn::SYNC(1),
               insn::SET_ADDR_LO(1, 48), insn::SET_ADDR_HI(1, 0),
               store, insn::SYNC(1), uint64_t(0x0e) << 59});
    s.run();
    if (!s.dut.done || s.dut.fault)
        TEST_FAIL("store_instruction_coherence", "cached pre-STORE instruction was executed");
    TEST_PASS("store_instruction_coherence");
}

static void check_unstalled_store_cadence() {
    const char* name = "unstalled_store_cadence";
    ProtocolMemory s;
    constexpr uint64_t DST = 0x200ff0;
    constexpr unsigned BEATS = 7;
    s.minimum_w_stalls = 0;
    s.randomize_w_ready = false;
    for (unsigned row = 0; row < BEATS; ++row) {
        uint8_t data[16];
        for (unsigned i = 0; i < 16; ++i) data[i] = strict_source_byte(DST + row * 16 + i);
        sram_write_row(&s.dut, BUF_WBUF_ID, row, data);
    }
    s.program({insn::SET_ADDR_LO(0, DST), insn::SET_ADDR_HI(0, 0),
               insn::STORE(BUF_WBUF_ID, 0, BEATS, 0, 0), insn::SYNC(1), insn::HALT()});
    s.run();
    if (!s.dut.done || s.dut.fault || s.w_stalls || s.w_cycles.size() != BEATS ||
        s.writes != expected_burst_log(DST, BEATS * 16, BEATS * 16))
        TEST_FAIL(name, "unexpected transfer, stall or fault");
    unsigned beat = 0;
    for (unsigned burst = 0; burst < s.writes.size(); ++burst)
        for (unsigned local = 0; local < s.writes[burst].second; ++local, ++beat)
            if (s.w_cycles[beat] != s.aw_cycles[burst] + 2 + 2 * local)
                TEST_FAIL(name, "unstalled AW/PRE/W cadence changed");
    for (unsigned i = 0; i < BEATS * 16; ++i)
        if (s.bytes[DST + i] != strict_source_byte(DST + i))
            TEST_FAIL(name, "STORE payload mismatch");
    std::printf("INFO: unstalled STORE total cycles=%llu fetch_buffer=%u\n",
                static_cast<unsigned long long>(s.cycles),
                unsigned(s.dut.rootp->taccel_top__DOT__obs_fetch_buffer_enable_w));
    std::fflush(stdout);
}

static void test_stalled_store_with_systolic_same_buffer_read() {
    const char* name = "stalled_store_with_systolic_same_buffer_read";
    check_unstalled_store_cadence();
    ProtocolMemory s;
    constexpr uint64_t DST = 0x200000;
    uint8_t stored[16], ones[16], twos[16];
    for (unsigned i = 0; i < 16; ++i) {
        stored[i] = strict_source_byte(DST + i);
        ones[i] = 1;
        twos[i] = 2;
    }
    sram_write_row(&s.dut, BUF_ABUF_ID, 64, stored);
    for (unsigned row = 0; row < 16; ++row) {
        sram_write_row(&s.dut, BUF_ABUF_ID, row, ones);
        sram_write_row(&s.dut, BUF_ABUF_ID, 16 + row, twos);
    }
    // STORE is the odd instruction in its beat, so the following MATMUL
    // requires a new AR with either fetch mode. Delay that AR until the DMA
    // has completed its SRAM pre-read, then allow legal Port-S reads of
    // different rows in the same physical ABUF while WVALID is backpressured.
    s.instruction_read_wait_for_wvalid_from = 32;
    s.minimum_w_stalls = 1024;
    s.program({insn::SET_ADDR_LO(0, DST), insn::SET_ADDR_HI(0, 0),
               insn::CONFIG_TILE(1, 1, 1), insn::STORE(BUF_ABUF_ID, 64, 1, 0, 0),
               insn::MATMUL(BUF_ABUF_ID, 0, BUF_ABUF_ID, 16, BUF_ACCUM_ID, 0, 0),
               insn::SYNC(3), insn::HALT()});
    s.run();
    if (!s.dut.done || s.dut.fault) TEST_FAIL(name, "did not halt cleanly");
    if (s.stalled_w_same_buffer_s_reads == 0)
        TEST_FAIL(name, "concurrent same-buffer different-row Port-S reads were not exercised");
    for (unsigned i = 0; i < 16; ++i)
        if (s.bytes[DST + i] != stored[i]) TEST_FAIL(name, "STORE payload was corrupted");
    for (unsigned row = 0; row < 64; ++row) {
        uint8_t got[16];
        sram_read_row(&s.dut, BUF_ACCUM_ID, row, got);
        for (unsigned lane = 0; lane < 4; ++lane)
            if (got[lane * 4] != 32 || got[lane * 4 + 1] ||
                got[lane * 4 + 2] || got[lane * 4 + 3])
                TEST_FAIL(name, "concurrent MATMUL output mismatch");
    }
    TEST_PASS(name);
}

static void test_fetch_counter_ignores_busy_start() {
    const char* name = "fetch_counter_ignores_busy_start";
    ProtocolMemory s;
    std::vector<uint64_t> program(64, insn::NOP());
    program.push_back(insn::HALT());
    s.program(program);
    auto* root = s.dut.rootp;
    if (root->taccel_top__DOT__obs_fetch_read_count_q != 0)
        TEST_FAIL(name, "reset did not clear the fetch-read counter");
    s.dut.start = 1;
    s.tick();
    if (root->taccel_top__DOT__obs_fetch_read_count_q != 0)
        TEST_FAIL(name, "accepted start did not begin with an empty counter");

    unsigned reads_with_start_high = 0;
    bool repeated_start_with_prior_reads = false;
    for (unsigned cycle = 0; cycle < 2000 && !s.dut.done && !s.dut.fault; ++cycle) {
        // Keep start high after the initial acceptance, then reassert it once
        // as a held level and once as a pulse while the same program is busy.
        s.dut.start = cycle < 20 || (cycle >= 40 && cycle < 50) || cycle == 70;
        const size_t reads_before = s.reads.size();
        if (cycle == 40 && reads_before != 0)
            repeated_start_with_prior_reads = true;
        s.tick();
        if (s.dut.start && s.reads.size() != reads_before) ++reads_with_start_high;
        // This program has no DMA reads: every accepted external AR is fetch.
        if (root->taccel_top__DOT__obs_fetch_read_count_q != s.reads.size())
            TEST_FAIL(name, "held/repeated busy start discarded accepted fetch reads");
    }
    if (!s.dut.done || s.dut.fault || !repeated_start_with_prior_reads ||
        reads_with_start_high == 0)
        TEST_FAIL(name, "program or held/repeated-start coverage was incomplete");

    const uint64_t completed_reads = root->taccel_top__DOT__obs_fetch_read_count_q;
    s.dut.start = 1;
    for (unsigned cycle = 0; cycle < 3; ++cycle) s.tick();
    if (!s.dut.done || root->taccel_top__DOT__obs_fetch_read_count_q != completed_reads)
        TEST_FAIL(name, "ignored terminal start changed the completed-run count");

    // Restart is reset-qualified: neither the controller nor the other run
    // counters accept a new run directly from HALT. Keep that behavior intact.
    s.dut.start = 0;
    do_reset(&s.dut);
    if (root->taccel_top__DOT__obs_fetch_read_count_q != 0)
        TEST_FAIL(name, "second reset did not clear the counter");
    const size_t read_base = s.reads.size();
    s.program({insn::NOP(), insn::HALT()});
    s.run();
    if (!s.dut.done || s.dut.fault || s.reads.size() == read_base ||
        root->taccel_top__DOT__obs_fetch_read_count_q != s.reads.size() - read_base)
        TEST_FAIL(name, "reset-qualified new run counted prior-run reads");
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
    test_transpose_load_c16();
    test_transpose_load_c32();
    test_transpose_load_c64();
    test_page_boundaries_with_backpressure();
    test_effective_address_overflow();
    test_split_transpose_response_errors();
    test_store_instruction_coherence();
    test_stalled_store_with_systolic_same_buffer_read();
    test_fetch_counter_ignores_busy_start();

    printf("\n%d / %d tests passed\n", tests_pass, tests_run);
    return (tests_pass == tests_run) ? 0 : 1;
}
