// Verilator tests for the Stage D SFU engine.

#include "Vtaccel_top.h"
#include "Vtaccel_top___024root.h"
#include "verilated.h"
#include "include/testbench.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "test_runner.h"

using tbutil::SimHarness;
using tbutil::sram_write_row;
using tbutil::sram_read_row;
using tbutil::sram_write_bytes;
using tbutil::sram_read_bytes;
using tbutil::pack_i32_le;
using tbutil::pack_u16_le;
constexpr int BUF_ABUF_ID  = tbutil::BUF_ABUF_ID;
constexpr int BUF_WBUF_ID  = tbutil::BUF_WBUF_ID;
constexpr int BUF_ACCUM_ID = tbutil::BUF_ACCUM_ID;

static std::vector<uint8_t> read_binary_file(const char* name, const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        std::string msg = "could not open " + path;
        TEST_FAIL(name, msg.c_str());
    }
    return std::vector<uint8_t>(
        std::istreambuf_iterator<char>(stream),
        std::istreambuf_iterator<char>());
}

static void expect_equal_bytes(const char* name,
                               const std::vector<uint8_t>& got,
                               const std::vector<uint8_t>& exp) {
    if (got.size() != exp.size())
        TEST_FAIL(name, "length mismatch");
    for (size_t i = 0; i < got.size(); ++i) {
        if (got[i] != exp[i]) {
            std::fprintf(stderr, "%s mismatch at byte %zu: got=%d exp=%d\n",
                         name, i, int(got[i]), int(exp[i]));
            TEST_FAIL(name, "byte mismatch");
        }
    }
}


// Geometry constants shared by the gen-2 fixture tests below.
static constexpr int G2_M = 16;
static constexpr int G2_N = 64;
static constexpr int G2_S1 = 0;
static constexpr int G2_S2 = 512;
static constexpr int G2_DST = 1024;
static constexpr int G2_OUT_BYTES = G2_M * G2_N * 2;   // FP16 dst tile

// FP16-ULP conformance check (isa_generation_freeze.md §7): exact byte-match
// is the target; the accepted band is |ulp| <= max_ulp for finite same-sign
// values; NaN/Inf must match exactly. Prints the observed max ULP + count.
// Used by every test_g2_* test below.
static void expect_fp16_ulp(const char* name,
                            const std::vector<uint8_t>& got,
                            const std::vector<uint8_t>& exp,
                            int max_ulp) {
    if (got.size() != exp.size()) TEST_FAIL(name, "length mismatch");
    if (std::getenv("RTL_G2_DUMP")) {
        std::string p = std::string("/tmp/") + name + "_got.raw";
        FILE* f = std::fopen(p.c_str(), "wb");
        if (f) { std::fwrite(got.data(), 1, got.size(), f); std::fclose(f); }
    }
    auto ord = [](uint16_t h) -> int {
        int mag = int(h & 0x7FFF);
        return (h & 0x8000) ? -mag : mag;
    };
    int gmax = 0, nbad = 0; size_t gmax_i = 0;
    for (size_t i = 0; i + 1 < got.size(); i += 2) {
        uint16_t a = uint16_t(got[i]) | (uint16_t(got[i + 1]) << 8);
        uint16_t b = uint16_t(exp[i]) | (uint16_t(exp[i + 1]) << 8);
        if (a == b) continue;
        bool a_special = ((a & 0x7C00) == 0x7C00);
        bool b_special = ((b & 0x7C00) == 0x7C00);
        if (a_special || b_special) {           // NaN/Inf: exact only
            nbad++; gmax = 0x7fff; gmax_i = i;
            continue;
        }
        int u = ord(a) - ord(b); if (u < 0) u = -u;
        if (u > gmax) { gmax = u; gmax_i = i; }   // true max over ALL mismatches
        if (u > max_ulp) nbad++;
    }
    std::fprintf(stderr,
        "[%s] fp16-ulp: %d elem(s) over band (max_ulp=%d); TRUE max=%d ULP "
        "@byte %zu\n", name, nbad, max_ulp, gmax, gmax_i);
    if (nbad != 0) TEST_FAIL(name, "fp16-ulp band exceeded");
}

static void test_g2_vadd_fp32() {
    const char* name = "g2_vadd_fp32";
    const std::string d = "fixtures/gen2/vadd_fp32/std";
    const int band = 0;   // non-transcendental: bit-exact (freeze §7)
    auto s1 = read_binary_file(name, d + "/input_src1.raw");
    auto s2 = read_binary_file(name, d + "/input_src2.raw");
    auto expected = read_binary_file(name, d + "/expected_out.raw");
    SimHarness s;
    sram_write_bytes(s.dut.get(), BUF_ABUF_ID, size_t(G2_S1) * 16, s1);
    sram_write_bytes(s.dut.get(), BUF_ABUF_ID, size_t(G2_S2) * 16, s2);
    s.load({
        insn::CONFIG_TILE(1, 4, 1),
        insn::R_TYPE(0x19, BUF_ABUF_ID, G2_S1, BUF_ABUF_ID, G2_S2,
                     BUF_ABUF_ID, G2_DST, 0, 1),
        insn::SYNC(0b100),
        insn::HALT(),
    });
    s.run(250000);
    if (s.dut->fault) TEST_FAIL(name, "unexpected fault");
    auto got = sram_read_bytes(s.dut.get(), BUF_ABUF_ID,
                               size_t(G2_DST) * 16, G2_OUT_BYTES);
    expect_fp16_ulp(name, got, expected, band);   // freeze §7 per-op band
    TEST_PASS(name);
}

static void test_g2_layernorm_fp32() {
    const char* name = "g2_layernorm_fp32";
    const std::string d = "fixtures/gen2/layernorm_fp32/std";
    const int band = 0;   // non-transcendental: bit-exact (freeze §7)
    auto s1 = read_binary_file(name, d + "/input_src1.raw");
    auto gb = read_binary_file(name, d + "/input_src2.raw");  // 2N fp16
    auto expected = read_binary_file(name, d + "/expected_out.raw");
    SimHarness s;
    sram_write_bytes(s.dut.get(), BUF_ABUF_ID, size_t(G2_S1) * 16, s1);
    sram_write_bytes(s.dut.get(), BUF_WBUF_ID, size_t(G2_S2) * 16, gb);
    s.load({
        insn::CONFIG_TILE(1, 4, 1),
        insn::R_TYPE(0x1A, BUF_ABUF_ID, G2_S1, BUF_WBUF_ID, G2_S2,
                     BUF_ABUF_ID, G2_DST, 0, 1),
        insn::SYNC(0b100),
        insn::HALT(),
    });
    s.run(250000);
    if (s.dut->fault) TEST_FAIL(name, "unexpected fault");
    auto got = sram_read_bytes(s.dut.get(), BUF_ABUF_ID,
                               size_t(G2_DST) * 16, G2_OUT_BYTES);
    expect_fp16_ulp(name, got, expected, band);   // freeze §7 per-op band
    TEST_PASS(name);
}

static void test_g2_gelu_fp32() {
    const char* name = "g2_gelu_fp32";
    const std::string d = "fixtures/gen2/gelu_fp32/std";
    const int band = 3;   // gelu_new: numpy-tanh vs libm-tanh (freeze §7)
    auto s1 = read_binary_file(name, d + "/input_src1.raw");
    auto expected = read_binary_file(name, d + "/expected_out.raw");
    SimHarness s;
    sram_write_bytes(s.dut.get(), BUF_ABUF_ID, size_t(G2_S1) * 16, s1);
    s.load({
        insn::CONFIG_TILE(1, 4, 1),
        insn::R_TYPE(0x1B, BUF_ABUF_ID, G2_S1, BUF_ABUF_ID, 0,
                     BUF_ABUF_ID, G2_DST, 0, 1),
        insn::SYNC(0b100),
        insn::HALT(),
    });
    s.run(250000);
    if (s.dut->fault) TEST_FAIL(name, "unexpected fault");
    auto got = sram_read_bytes(s.dut.get(), BUF_ABUF_ID,
                               size_t(G2_DST) * 16, G2_OUT_BYTES);
    expect_fp16_ulp(name, got, expected, band);   // freeze §7 per-op band
    TEST_PASS(name);
}

static void test_g2_dequant_accum_fp32() {
    const char* name = "g2_dequant_accum_fp32";
    const int band = 0;   // non-transcendental: bit-exact (freeze §7)
    const std::string d = "fixtures/gen2/dequant_accum_fp32/std";
    auto s1 = read_binary_file(name, d + "/input_src1.raw");  // M*N int32
    auto sc = read_binary_file(name, d + "/input_src2.raw");  // N fp16
    auto expected = read_binary_file(name, d + "/expected_out.raw");
    SimHarness s;
    sram_write_bytes(s.dut.get(), BUF_ACCUM_ID, 0, s1);
    sram_write_bytes(s.dut.get(), BUF_WBUF_ID, size_t(G2_S2) * 16, sc);
    s.load({
        insn::CONFIG_TILE(1, 4, 1),
        insn::R_TYPE(0x17, BUF_ACCUM_ID, 0, BUF_WBUF_ID, G2_S2,
                     BUF_ABUF_ID, G2_DST, 0, 1),
        insn::SYNC(0b100),
        insn::HALT(),
    });
    s.run(250000);
    if (s.dut->fault) TEST_FAIL(name, "unexpected fault");
    auto got = sram_read_bytes(s.dut.get(), BUF_ABUF_ID,
                               size_t(G2_DST) * 16, G2_OUT_BYTES);
    expect_fp16_ulp(name, got, expected, band);   // freeze §7 per-op band
    TEST_PASS(name);
}

static void run_g2_quant(const char* name, const std::string& d) {
    auto s1 = read_binary_file(name, d + "/input_src1.raw");      // M*N fp16
    auto pre = read_binary_file(name, d + "/scale_regs_pre.raw"); // 16 fp16
    auto expected = read_binary_file(name, d + "/expected_out.raw"); // M*N i8
    uint16_t s3 = uint16_t(pre[6]) | (uint16_t(pre[7]) << 8);     // reg 3
    SimHarness s;
    sram_write_bytes(s.dut.get(), BUF_ABUF_ID, size_t(G2_S1) * 16, s1);
    s.load({
        insn::SET_SCALE(3, s3),
        insn::CONFIG_TILE(1, 4, 1),
        insn::R_TYPE(0x18, BUF_ABUF_ID, G2_S1, BUF_ABUF_ID, 0,
                     BUF_ABUF_ID, G2_DST, 3, 1),
        insn::SYNC(0b100),
        insn::HALT(),
    });
    s.run(250000);
    if (s.dut->fault) TEST_FAIL(name, "unexpected fault");
    // INT8 output: exact byte-match (one byte/elem, M*N bytes).
    auto got = sram_read_bytes(s.dut.get(), BUF_ABUF_ID,
                               size_t(G2_DST) * 16, size_t(G2_M) * G2_N);
    expect_equal_bytes(name, got, expected);
    TEST_PASS(name);
}

static void test_g2_quant_fp32_int8() {
    run_g2_quant("g2_quant_fp32_int8_std",
                 "fixtures/gen2/quant_fp32_int8/std");
    run_g2_quant("g2_quant_fp32_int8_rhe",
                 "fixtures/gen2/quant_fp32_int8/round_half_even");
}

static void test_g2_masked_softmax_fp32() {
    const char* name = "g2_masked_softmax_fp32";
    const int band = 0;   // exp: measured BIT-EXACT on this fixture (§7)
    const std::string d = "fixtures/gen2/masked_softmax_fp32/std";
    auto s1 = read_binary_file(name, d + "/input_src1.raw");
    auto expected = read_binary_file(name, d + "/expected_out.raw");
    SimHarness s;
    sram_write_bytes(s.dut.get(), BUF_ABUF_ID, size_t(G2_S1) * 16, s1);
    s.load({
        insn::CONFIG_TILE(1, 4, 1),
        insn::CONFIG_ATTN(0, 64, 1),   // qrb=0, valid_kv_len=64, mode!=0
        insn::R_TYPE(0x1D, BUF_ABUF_ID, G2_S1, BUF_ABUF_ID, 0,
                     BUF_ABUF_ID, G2_DST, 0, 1),
        insn::SYNC(0b100),
        insn::HALT(),
    });
    s.run(250000);
    if (s.dut->fault) TEST_FAIL(name, "unexpected fault");
    auto got = sram_read_bytes(s.dut.get(), BUF_ABUF_ID,
                               size_t(G2_DST) * 16, G2_OUT_BYTES);
    expect_fp16_ulp(name, got, expected, band);   // freeze §7 per-op band
    TEST_PASS(name);
}

// Peek a scale register (register_file.scale_regs is verilator public).
static uint16_t scale_reg(SimHarness& s, int idx) {
    return s.dut->rootp->taccel_top__DOT__u_regfile__DOT__scale_regs[idx];
}

static void run_g2_max_abs_reduce(const char* name, const std::string& d) {
    auto s1 = read_binary_file(name, d + "/input_src1.raw");      // M*N fp16
    auto post = read_binary_file(name, d + "/scale_regs_post.raw"); // 16 fp16
    uint16_t exp5 = uint16_t(post[10]) | (uint16_t(post[11]) << 8); // reg 5
    uint16_t exp6 = uint16_t(post[12]) | (uint16_t(post[13]) << 8); // reg 6
    SimHarness s;
    sram_write_bytes(s.dut.get(), BUF_ABUF_ID, size_t(G2_S1) * 16, s1);
    s.load({
        insn::CONFIG_TILE(1, 4, 1),
        insn::R_TYPE(0x1F, BUF_ABUF_ID, G2_S1, BUF_ABUF_ID, 0,
                     BUF_ABUF_ID, 0, 5, 1),   // sreg=5 -> regs 5,6
        insn::SYNC(0b100),
        insn::HALT(),
    });
    s.run(250000);
    if (s.dut->fault) TEST_FAIL(name, "unexpected fault");
    uint16_t g5 = scale_reg(s, 5), g6 = scale_reg(s, 6);
    if (g5 != exp5 || g6 != exp6) {
        std::fprintf(stderr, "%s: reg5 got=%04x exp=%04x  reg6 got=%04x "
                     "exp=%04x\n", name, g5, exp5, g6, exp6);
        TEST_FAIL(name, "scale-reg writeback mismatch");
    }
    TEST_PASS(name);
}

static void test_g2_max_abs_reduce_fp32() {
    run_g2_max_abs_reduce("g2_max_abs_reduce_std",
                          "fixtures/gen2/max_abs_reduce_fp32/std");
    run_g2_max_abs_reduce("g2_max_abs_reduce_zero",
                          "fixtures/gen2/max_abs_reduce_fp32/all_zero");
}

static void test_g2_dequant_accum_fp32_scaled() {
    const char* name = "g2_dequant_accum_fp32_scaled";
    const int band = 0;   // non-transcendental: bit-exact (freeze §7)
    const std::string d = "fixtures/gen2/dequant_accum_fp32_scaled/std";
    auto s1 = read_binary_file(name, d + "/input_src1.raw");      // M*N i32
    auto s2 = read_binary_file(name, d + "/input_src2.raw");      // 2N fp16
    auto pre = read_binary_file(name, d + "/scale_regs_pre.raw"); // 16 fp16
    auto expected = read_binary_file(name, d + "/expected_out.raw");
    uint16_t a7 = uint16_t(pre[14]) | (uint16_t(pre[15]) << 8);   // reg 7 act
    SimHarness s;
    sram_write_bytes(s.dut.get(), BUF_ACCUM_ID, 0, s1);
    sram_write_bytes(s.dut.get(), BUF_WBUF_ID, size_t(G2_S2) * 16, s2);
    s.load({
        insn::SET_SCALE(7, a7),
        insn::CONFIG_TILE(1, 4, 1),
        insn::R_TYPE(0x1E, BUF_ACCUM_ID, 0, BUF_WBUF_ID, G2_S2,
                     BUF_ABUF_ID, G2_DST, 7, 1),
        insn::SYNC(0b100),
        insn::HALT(),
    });
    s.run(250000);
    if (s.dut->fault) TEST_FAIL(name, "unexpected fault");
    auto got = sram_read_bytes(s.dut.get(), BUF_ABUF_ID,
                               size_t(G2_DST) * 16, G2_OUT_BYTES);
    expect_fp16_ulp(name, got, expected, band);   // freeze §7 per-op band
    TEST_PASS(name);
}

// 0x1F write-back determinism + survival across a following op/SYNC:
// both 0x1F calls (same input, different sreg) write the SAME value, and
// the first result survives the second op. NOT a consumer-visibility test
// (no 0x18/0x1E reads the written scale here) — the real H2 chain
// (0x1F -> SYNC -> consumer reading that scale) is exercised by the
// end-to-end gate (P6) via the compiled bundle.
static void test_g2_scale_chain() {
    const char* name = "g2_scale_chain_1f_sync";
    const std::string d = "fixtures/gen2/max_abs_reduce_fp32/std";
    auto s1 = read_binary_file(name, d + "/input_src1.raw");
    auto post = read_binary_file(name, d + "/scale_regs_post.raw");
    uint16_t exp5 = uint16_t(post[10]) | (uint16_t(post[11]) << 8);
    SimHarness s;
    sram_write_bytes(s.dut.get(), BUF_ABUF_ID, size_t(G2_S1) * 16, s1);
    s.load({
        insn::CONFIG_TILE(1, 4, 1),
        insn::R_TYPE(0x1F, BUF_ABUF_ID, G2_S1, BUF_ABUF_ID, 0,
                     BUF_ABUF_ID, 0, 5, 1),
        insn::SYNC(0b100),               // wait for 0x1F (SFU) to retire
        insn::R_TYPE(0x1F, BUF_ABUF_ID, G2_S1, BUF_ABUF_ID, 0,
                     BUF_ABUF_ID, 0, 8, 1),   // 2nd 0x1F, different sreg
        insn::SYNC(0b100),
        insn::HALT(),
    });
    s.run(250000);
    if (s.dut->fault) TEST_FAIL(name, "unexpected fault");
    // First 0x1F's result must survive the second op; both write same val.
    if (scale_reg(s, 5) != exp5 || scale_reg(s, 8) != exp5)
        TEST_FAIL(name, "scale not visible across SYNC");
    // H1: the SFU-vs-SET_SCALE write-port overlap invariant must hold
    // (raised obs bit => toolchain serialization broken). Fail loudly.
    if (s.dut->rootp->taccel_top__DOT__obs_forbidden_overlap_violation_q)
        TEST_FAIL(name, "obs_forbidden_overlap_violation_q set");
    TEST_PASS(name);
}

// ===================================================================
// P6e / BUG1 reproducer. Mirrors head0_query's real flow: the FIRST
// gen-2 scale-writing chain 0x1F(MAX_ABS_REDUCE -> scale_regs[sreg]) ->
// SYNC(0b100) -> 0x18(QUANT, reads scale_regs[sreg]). The existing
// test_g2_scale_chain has NO consumer reading the scale between the
// SYNC and end-of-run, so it misses the first-instance scale-write ->
// consumer VISIBILITY race. Here chain1 is that first chain; chain2 is
// an IDENTICAL warm chain (different sreg) = in-binary known-exact
// contrast. Predicted-buggy: QUANT reads scale_regs reset 0.0 ->
// quantize_to_i8(out_scale==0.0)=0x00 -> int8 ALL ZERO. DIAGNOSTIC:
// never aborts the suite; prints a verdict.
static void bug1_build_X(std::vector<uint8_t>& x) {
    // 16x64 fp16, contiguous rows*cols*2, LE. Varied sign/magnitude with
    // one clear global max so scale=127/maxabs is well-defined and the
    // quantization is non-degenerate (full INT8 range used).
    x.assign(size_t(G2_M) * G2_N * 2, 0);
    for (int i = 0; i < G2_M; ++i)
        for (int j = 0; j < G2_N; ++j) {
            double v = (double(((i * 7 + j * 3) % 17) - 8)) * 0.05;
            if (i == 3 && j == 29) v = 1.75;          // unique max|.|
            int bits = sfu_fp32_to_fp16_bits(v);
            size_t o = (size_t(i) * G2_N + j) * 2;
            x[o] = uint8_t(bits & 0xFF);
            x[o + 1] = uint8_t((bits >> 8) & 0xFF);
        }
}

static void test_bug1_scale_visibility_diagnostic() {
    const char* name = "bug1_first_scale_write_visibility_DIAG";
    const int GD = 160, D1 = 400, D2 = 520;   // ABUF unit offsets
    const size_t I8 = size_t(G2_M) * G2_N;    // int8 bytes per chain
    std::vector<uint8_t> X;
    bug1_build_X(X);

    auto run_variant = [&](bool with_prefix, const char* tag) {
        SimHarness s;
        sram_write_bytes(s.dut.get(), BUF_ABUF_ID, size_t(G2_S1) * 16, X);
        std::vector<uint64_t> prog;
        if (with_prefix) {
            // A prior SFU op completing via SYNC(0b100) before the first
            // scale-write 0x1F (mirrors ln1 LAYERNORM before head0_query's
            // 0x1F). GELU (0x1B) has no src2; sreg 8 = unused.
            prog.push_back(insn::CONFIG_TILE(1, 4, 1));
            prog.push_back(insn::R_TYPE(0x1B, BUF_ABUF_ID, G2_S1, BUF_ABUF_ID,
                                        0, BUF_ABUF_ID, GD, 8, 1));
            prog.push_back(insn::SYNC(0b100));
        }
        // chain1: FIRST scale-writing chain (sreg 0 -> regs 0,1).
        prog.push_back(insn::CONFIG_TILE(1, 4, 1));
        prog.push_back(insn::R_TYPE(0x1F, BUF_ABUF_ID, G2_S1, BUF_ABUF_ID, 0,
                                    BUF_ABUF_ID, 0, 0, 1));
        prog.push_back(insn::SYNC(0b100));
        prog.push_back(insn::R_TYPE(0x18, BUF_ABUF_ID, G2_S1, BUF_ABUF_ID, 0,
                                    BUF_ABUF_ID, D1, 0, 1));
        prog.push_back(insn::SYNC(0b100));
        // chain2: IDENTICAL, warm (sreg 4 -> regs 4,5) = known-exact ref.
        prog.push_back(insn::CONFIG_TILE(1, 4, 1));
        prog.push_back(insn::R_TYPE(0x1F, BUF_ABUF_ID, G2_S1, BUF_ABUF_ID, 0,
                                    BUF_ABUF_ID, 0, 4, 1));
        prog.push_back(insn::SYNC(0b100));
        prog.push_back(insn::R_TYPE(0x18, BUF_ABUF_ID, G2_S1, BUF_ABUF_ID, 0,
                                    BUF_ABUF_ID, D2, 4, 1));
        prog.push_back(insn::SYNC(0b100));
        prog.push_back(insn::HALT());
        s.load(prog);
        s.run(300000);
        if (s.dut->fault) { std::printf("DIAG %s: FAULT\n", tag); return; }

        auto c1 = sram_read_bytes(s.dut.get(), BUF_ABUF_ID, size_t(D1) * 16, I8);
        auto c2 = sram_read_bytes(s.dut.get(), BUF_ABUF_ID, size_t(D2) * 16, I8);
        bool c1_zero = std::all_of(c1.begin(), c1.end(),
                                   [](uint8_t b) { return b == 0; });
        bool c2_zero = std::all_of(c2.begin(), c2.end(),
                                   [](uint8_t b) { return b == 0; });
        bool c1_eq_c2 = (c1 == c2);
        uint16_t r0 = scale_reg(s, 0), r1 = scale_reg(s, 1);
        uint16_t r4 = scale_reg(s, 4), r5 = scale_reg(s, 5);
        bool regs_landed = (r0 == r4) && (r1 == r5) && (r0 != 0);
        int nz1 = 0;
        for (auto b : c1) if (b) ++nz1;
        std::printf("DIAG %s: chain1_zero=%d chain2_zero=%d chain1==chain2=%d "
                    "nz1=%d/%zu  sreg[0,1]=%04x,%04x [4,5]=%04x,%04x "
                    "regs_landed=%d\n", tag, c1_zero, c2_zero, c1_eq_c2,
                    nz1, I8, r0, r1, r4, r5, regs_landed);
        if (!regs_landed || c2_zero)
            std::printf("DIAG %s VERDICT: INSTRUMENT INVALID (warm chain/"
                        "regs not sane) — inconclusive\n", tag);
        else if (c1_zero && !c1_eq_c2)
            std::printf("DIAG %s VERDICT: BUG1 REPRODUCED — first scale-write "
                        "NOT visible to first consumer (chain1 int8 all-zero "
                        "== predicted scale=0.0; chain2 warm correct)\n", tag);
        else if (c1_eq_c2)
            std::printf("DIAG %s VERDICT: not reproduced with this prefix "
                        "(chain1 == warm chain2)\n", tag);
        else
            std::printf("DIAG %s VERDICT: chain1 wrong DIFFERENTLY (not the "
                        "predicted all-zero) — investigate, do NOT patch\n",
                        tag);
    };

    run_variant(false, "A_pure_cold");
    run_variant(true, "B_prefix(GELU+SYNC then first 0x1F)");
    (void)name;
}

// P6e/BUG1 FULL-CHAIN reproducer: head0_query's real op sequence
// 0x1F(s)->SYNC->0x18(s)->SYNC->MATMUL(flags=0)->SYNC->0x1E(s+1) with
// PRELOADED buffers (no DMA). chain1 = FIRST (cold); chain2 = identical
// warm (s=4/5, separate ACCUM region) = in-binary known-exact contrast.
// Captures int8 / ACCUM / 0x1E-out for both; first differing stage
// localizes the first-instance bug. 0x1E reads scale_regs[s+1] = the
// 0x1F PHASE-1 write (committed 1 cyc after phase-0, untested by the
// 0x18-only repro). Predicted-buggy if phase-1 not visible: 0x1E =
// pc*0.0*ACCUM + bias = the preloaded bias vector, bit-exact. DIAGNOSTIC.
static void test_bug1_full_chain_diagnostic() {
    const char* name = "bug1_full_chain_DIAG";
    const int X1 = 0;                 // X fp16 16x64  (units; 128u)
    const int I1 = 160, I2 = 240;     // int8 16x64    (64u each)
    const int Q1 = 340, Q2 = 480;     // 0x1E out fp16 (128u each)
    const int WB = 0;                 // WBUF weight 64x64 i8 (256u)
    const int PCB = 300;              // WBUF 2N=128 fp16 (pc||bias) (16u)
    const int AC1 = 0, AC2 = 256;     // ACCUM dst units (4096B apart)
    const size_t I8 = size_t(G2_M) * G2_N;        // 1024
    const size_t OB = size_t(G2_M) * G2_N * 2;    // 0x1E out bytes 2048
    const size_t ACB = size_t(G2_M) * G2_N * 4;   // ACCUM bytes 4096

    std::vector<uint8_t> X;
    bug1_build_X(X);                              // 16x64 fp16

    // Weight 64x64 int8 (K=64,N=64), deterministic.
    std::vector<uint8_t> W(64 * 64);
    for (int k = 0; k < 64; ++k)
        for (int j = 0; j < 64; ++j)
            W[k * 64 + j] = uint8_t(int8_t(((k * 5 + j * 3) % 13) - 6));
    // src2 for 0x1E: 2N FP16 = [N pc-scale | N bias]. pc=1/128 (clean
    // pow2); bias = distinctive ramp so predicted-buggy (==bias) is
    // unambiguous and won't coincide with the real dequant.
    std::vector<uint8_t> PCBv(2 * 64 * 2);
    for (int j = 0; j < 64; ++j) {
        int pc = sfu_fp32_to_fp16_bits(1.0 / 128.0);
        int bs = sfu_fp32_to_fp16_bits(-2.0 + 0.05 * j);
        PCBv[j * 2] = pc & 0xFF;            PCBv[j * 2 + 1] = (pc >> 8) & 0xFF;
        PCBv[(64 + j) * 2] = bs & 0xFF;     PCBv[(64 + j) * 2 + 1] = (bs >> 8) & 0xFF;
    }

    SimHarness s;
    sram_write_bytes(s.dut.get(), BUF_ABUF_ID, size_t(X1) * 16, X);
    sram_write_bytes(s.dut.get(), BUF_WBUF_ID, size_t(WB) * 16, W);
    sram_write_bytes(s.dut.get(), BUF_WBUF_ID, size_t(PCB) * 16, PCBv);

    auto chain = [&](int si, int i8dst, int acdst, int qdst) {
        return std::vector<uint64_t>{
            insn::CONFIG_TILE(1, 4, 1),
            insn::R_TYPE(0x1F, BUF_ABUF_ID, X1, BUF_ABUF_ID, 0,
                         BUF_ABUF_ID, 0, si, 1),
            insn::SYNC(0b100),
            insn::R_TYPE(0x18, BUF_ABUF_ID, X1, BUF_ABUF_ID, 0,
                         BUF_ABUF_ID, i8dst, si, 1),
            insn::SYNC(0b100),
            insn::CONFIG_TILE(1, 4, 4),
            insn::MATMUL(BUF_ABUF_ID, i8dst, BUF_WBUF_ID, WB,
                         BUF_ACCUM_ID, acdst, 0, 0),
            insn::SYNC(0b010),
            insn::CONFIG_TILE(1, 4, 1),
            insn::R_TYPE(0x1E, BUF_ACCUM_ID, acdst, BUF_WBUF_ID, PCB,
                         BUF_ABUF_ID, qdst, si + 1, 1),
            insn::SYNC(0b100),
        };
    };
    std::vector<uint64_t> prog;
    for (auto v : chain(0, I1, AC1, Q1)) prog.push_back(v);   // chain1 cold
    for (auto v : chain(4, I2, AC2, Q2)) prog.push_back(v);   // chain2 warm
    prog.push_back(insn::HALT());
    s.load(prog);
    s.run(400000);
    if (s.dut->fault) { std::printf("DIAG %s: FAULT\n", name); return; }

    auto i1 = sram_read_bytes(s.dut.get(), BUF_ABUF_ID, size_t(I1) * 16, I8);
    auto i2 = sram_read_bytes(s.dut.get(), BUF_ABUF_ID, size_t(I2) * 16, I8);
    auto a1 = sram_read_bytes(s.dut.get(), BUF_ACCUM_ID, size_t(AC1) * 16, ACB);
    auto a2 = sram_read_bytes(s.dut.get(), BUF_ACCUM_ID, size_t(AC2) * 16, ACB);
    auto q1 = sram_read_bytes(s.dut.get(), BUF_ABUF_ID, size_t(Q1) * 16, OB);
    auto q2 = sram_read_bytes(s.dut.get(), BUF_ABUF_ID, size_t(Q2) * 16, OB);
    bool i_eq = (i1 == i2), a_eq = (a1 == a2), q_eq = (q1 == q2);
    // predicted-buggy: chain1 0x1E == bias broadcast (every 64-col row ==
    // PCBv bias half). Compare row0 of q1 to the bias fp16 vector.
    std::vector<uint8_t> bias_row(PCBv.begin() + 64 * 2, PCBv.end());
    std::vector<uint8_t> q1_row0(q1.begin(), q1.begin() + 128);
    bool q1_is_bias = (q1_row0 == bias_row);
    uint16_t r0 = scale_reg(s, 0), r1 = scale_reg(s, 1);
    uint16_t r4 = scale_reg(s, 4), r5 = scale_reg(s, 5);
    bool regs_ok = (r0 == r4) && (r1 == r5) && (r0 != 0) && (r1 != 0);

    std::printf("DIAG %s: int8_eq=%d ACCUM_eq=%d dequant_eq=%d  "
                "chain1_0x1E==bias=%d  sreg[0,1]=%04x,%04x [4,5]=%04x,%04x "
                "regs_ok=%d\n", name, i_eq, a_eq, q_eq, q1_is_bias,
                r0, r1, r4, r5, regs_ok);
    if (!regs_ok)
        std::printf("DIAG %s VERDICT: INSTRUMENT INVALID (scale regs not "
                    "sane/equal) — inconclusive\n", name);
    else if (i_eq && a_eq && q_eq)
        std::printf("DIAG %s VERDICT: NOT reproduced in fresh-sim full chain "
                    "(all 3 stages chain1==chain2) — BUG1 needs full bundle "
                    "context; DEFER (do NOT patch)\n", name);
    else if (i_eq && a_eq && !q_eq)
        std::printf("DIAG %s VERDICT: localized to 0x1E / sreg+1 phase-1 "
                    "visibility (int8+ACCUM match, dequant differs)%s\n",
                    name, q1_is_bias
                    ? " — chain1==bias, predicted-buggy shape EXACT: ROOT LOCKED"
                    : " — but chain1 != bias: wrong DIFFERENTLY, investigate");
    else if (i_eq && !a_eq)
        std::printf("DIAG %s VERDICT: localized to first-MATMUL cold-start "
                    "(int8 match, ACCUM differs) — investigate that path\n",
                    name);
    else
        std::printf("DIAG %s VERDICT: int8 already differs (0x18/phase-0) — "
                    "contradicts prior repro; investigate, do NOT patch\n",
                    name);
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);

    test_bug1_scale_visibility_diagnostic();
    test_bug1_full_chain_diagnostic();
    // 2026-05-23 Phase B/C: gen-1 SFU test functions stripped (the 9 tests
    // for OP_SOFTMAX, OP_LAYERNORM, OP_GELU, OP_SOFTMAX_ATTNV,
    // OP_MASKED_SOFTMAX, OP_MASKED_SOFTMAX_ATTNV — now illegal at decode).
    test_g2_vadd_fp32();
    test_g2_layernorm_fp32();
    test_g2_gelu_fp32();
    test_g2_dequant_accum_fp32();
    test_g2_quant_fp32_int8();
    test_g2_masked_softmax_fp32();
    test_g2_max_abs_reduce_fp32();
    test_g2_dequant_accum_fp32_scaled();
    test_g2_scale_chain();

    std::printf("\n%d / %d tests passed\n", tests_pass, tests_run);
    if (tests_pass != tests_run) std::exit(1);
    return 0;
}
