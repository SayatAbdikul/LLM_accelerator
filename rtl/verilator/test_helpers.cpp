// Verilator helper-engine tests for Phase C.

#include "Vtaccel_top.h"
#include "Vtaccel_top___024root.h"
#include "verilated.h"
#include "include/systolic_test_utils.h"
#include "include/testbench.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
using tbutil::sram_write_bytes;
using tbutil::sram_read_bytes;
using tbutil::pack_i32_le;
using tbutil::pack_u16_le;
using tbutil::unpack_i32_le;
constexpr int BUF_ABUF_ID  = tbutil::BUF_ABUF_ID;
constexpr int BUF_WBUF_ID  = tbutil::BUF_WBUF_ID;
constexpr int BUF_ACCUM_ID = tbutil::BUF_ACCUM_ID;

static int8_t sat_add_ref(int8_t a, int8_t b) {
    int sum = int(a) + int(b);
    if (sum > 127) return int8_t(127);
    if (sum < -128) return int8_t(-128);
    return int8_t(sum);
}

static int64_t fp16_mul_round_even_ref(int32_t src, uint16_t fp16_val) {
    bool sign = (fp16_val >> 15) & 1;
    uint32_t exp = (fp16_val >> 10) & 0x1F;
    uint32_t frac = fp16_val & 0x3FF;
    if (exp == 0 && frac == 0)
        return 0;

    int32_t shift = 0;
    int64_t mant = 0;
    if (exp == 0) {
        mant = int64_t(frac);
        shift = -24;
    } else {
        mant = int64_t(1024 + frac);
        shift = int32_t(exp) - 25;
    }

    int64_t prod = int64_t(src) * mant;
    if (sign)
        prod = -prod;

    if (shift >= 0)
        return prod << shift;

    int rshift = -shift;
    int64_t abs_prod = prod < 0 ? -prod : prod;
    int64_t q = abs_prod >> rshift;
    int64_t rem = abs_prod & ((int64_t(1) << rshift) - 1);
    int64_t half = int64_t(1) << (rshift - 1);
    if (rem > half || (rem == half && (q & 1)))
        q += 1;
    return prod < 0 ? -q : q;
}

static double fp16_to_double_ref(uint16_t bits) {
    bool sign = (bits >> 15) & 1;
    int exp = (bits >> 10) & 0x1F;
    int frac = bits & 0x3FF;
    double sign_v = sign ? -1.0 : 1.0;
    if (exp == 0 && frac == 0)
        return 0.0;
    if (exp == 0)
        return sign_v * (double(frac) / 1024.0) * std::ldexp(1.0, -14);
    if (exp == 31)
        return sign_v * 65504.0;
    return sign_v * (1.0 + double(frac) / 1024.0) * std::ldexp(1.0, exp - 15);
}

static int round_half_even_ref(double x) {
    long long floor_i = static_cast<long long>(std::floor(x));
    double frac = x - static_cast<double>(floor_i);
    if (frac > 0.5)
        return int(floor_i + 1);
    if (frac < 0.5)
        return int(floor_i);
    return (floor_i & 1LL) ? int(floor_i + 1) : int(floor_i);
}

static int8_t requant_ref(int32_t src, uint16_t scale) {
    int64_t scaled = fp16_mul_round_even_ref(src, scale);
    if (scaled > 127) return int8_t(127);
    if (scaled < -128) return int8_t(-128);
    return int8_t(scaled);
}

static int8_t scale_mul_i8_ref(int8_t src, uint16_t scale) {
    return requant_ref(int32_t(src), scale);
}

static int32_t scale_mul_i32_ref(int32_t src, uint16_t scale) {
    int64_t scaled = fp16_mul_round_even_ref(src, scale);
    if (scaled > int64_t(INT32_MAX)) return INT32_MAX;
    if (scaled < int64_t(INT32_MIN)) return INT32_MIN;
    return int32_t(scaled);
}

static int8_t dequant_add_ref(int32_t accum, int8_t skip,
                              uint16_t accum_scale, uint16_t skip_scale) {
    double x = double(accum) * fp16_to_double_ref(accum_scale) +
               double(skip) * fp16_to_double_ref(skip_scale);
    int q = round_half_even_ref(x);
    if (q > 127) return int8_t(127);
    if (q < -128) return int8_t(-128);
    return int8_t(q);
}

static void expect_fault_program(const char* name,
                                 const std::vector<uint64_t>& prog,
                                 uint32_t expected_fault_code,
                                 int timeout = 5000) {
    SimHarness s;
    s.load(prog);
    s.run(timeout);
    EXPECT(s.dut->fault == 1, "fault should assert");
    EXPECT(s.dut->done == 0, "done should remain low");
    EXPECT(s.dut->fault_code == expected_fault_code, "unexpected fault code");
    TEST_PASS(name);
}

static int32_t read_accum_ij(Vtaccel_top* dut, int dst_off, int i, int j) {
    auto* r = dut->rootp;
    int grp = j / 4;
    int lane = j % 4;
    int row = dst_off + i * 4 + grp;
    uint32_t word = r->taccel_top__DOT__u_sram__DOT__u_accum__DOT__mem[row][lane];
    return int32_t(word);
}

static void matmul_ref(const int8_t a[16][16], const int8_t b[16][16], int32_t c[16][16]) {
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            int32_t acc = 0;
            for (int k = 0; k < 16; ++k)
                acc += int32_t(a[i][k]) * int32_t(b[k][j]);
            c[i][j] = acc;
        }
    }
}

static void test_buf_copy_flat_interbuffer() {
    const char* name = "buf_copy_flat_interbuffer";
    SimHarness s;
    std::vector<uint8_t> src(48);
    for (size_t i = 0; i < src.size(); ++i)
        src[i] = uint8_t((0x20 + 7 * i) & 0xFF);
    sram_write_bytes(s.dut.get(), BUF_ABUF_ID, 0, src);

    s.load({
        insn::BUF_COPY(BUF_ABUF_ID, 0, BUF_WBUF_ID, 10, 3, 0, 0),
        insn::HALT(),
    });
    s.run();

    EXPECT(s.dut->done == 1 && s.dut->fault == 0, "copy should halt cleanly");
    auto got = sram_read_bytes(s.dut.get(), BUF_WBUF_ID, 10 * 16, src.size());
    if (got != src) TEST_FAIL(name, "flat copy mismatch");
    TEST_PASS(name);
}

static void test_buf_copy_overlap_compaction() {
    const char* name = "buf_copy_overlap_compaction";
    SimHarness s;
    std::vector<uint8_t> bytes(6 * 16);
    for (size_t i = 0; i < bytes.size(); ++i)
        bytes[i] = uint8_t((0x51 + 11 * i) & 0xFF);
    auto expected = bytes;
    std::memmove(expected.data() + 16, expected.data() + 32, 48);
    sram_write_bytes(s.dut.get(), BUF_ABUF_ID, 0, bytes);

    s.load({
        insn::BUF_COPY(BUF_ABUF_ID, 2, BUF_ABUF_ID, 1, 3, 0, 0),
        insn::HALT(),
    });
    s.run();

    EXPECT(s.dut->done == 1 && s.dut->fault == 0, "overlap copy should halt cleanly");
    auto got = sram_read_bytes(s.dut.get(), BUF_ABUF_ID, 0, bytes.size());
    if (got != expected) TEST_FAIL(name, "overlap compaction mismatch");
    TEST_PASS(name);
}

static void test_buf_copy_zero_length() {
    const char* name = "buf_copy_zero_length";
    SimHarness s;
    std::vector<uint8_t> before(64);
    for (size_t i = 0; i < before.size(); ++i)
        before[i] = uint8_t((0x91 + 13 * i) & 0xFF);
    sram_write_bytes(s.dut.get(), BUF_WBUF_ID, 8 * 16, before);

    s.load({
        insn::BUF_COPY(BUF_ABUF_ID, 0, BUF_WBUF_ID, 8, 0, 0, 0),
        insn::HALT(),
    });
    s.run();

    auto got = sram_read_bytes(s.dut.get(), BUF_WBUF_ID, 8 * 16, before.size());
    if (got != before) TEST_FAIL(name, "zero-length copy modified memory");
    TEST_PASS(name);
}

static void test_buf_copy_transpose_unaligned_source() {
    const char* name = "buf_copy_transpose_unaligned_source";
    SimHarness s;
    constexpr int rows = 16;
    constexpr int cols = 18;
    std::vector<uint8_t> src(rows * cols);
    std::vector<uint8_t> expected(cols * rows);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            src[r * cols + c] = uint8_t((r * 19 + c * 7 + 3) & 0xFF);
            expected[c * rows + r] = src[r * cols + c];
        }

    sram_write_bytes(s.dut.get(), BUF_ABUF_ID, 0, src);
    s.load({
        insn::BUF_COPY(BUF_ABUF_ID, 0, BUF_WBUF_ID, 0, 18, 1, 1),
        insn::HALT(),
    });
    s.run(100000);

    auto got = sram_read_bytes(s.dut.get(), BUF_WBUF_ID, 0, expected.size());
    if (got != expected) TEST_FAIL(name, "transpose mismatch");
    TEST_PASS(name);
}

static void test_buf_copy_transpose_same_buffer_fault() {
    expect_fault_program("buf_copy_transpose_same_buffer_fault",
                         { insn::BUF_COPY(BUF_ABUF_ID, 0, BUF_ABUF_ID, 32, 16, 1, 1) }, 6, 5000);
}

static void test_vadd_int8_saturating() {
    const char* name = "vadd_int8_saturating";
    SimHarness s;
    std::vector<uint8_t> src_a(256), src_b(256), expected(256);
    for (int i = 0; i < 256; ++i) {
        int8_t a = (i % 5 == 0) ? int8_t(120) :
                   (i % 7 == 0) ? int8_t(-120) : int8_t((i % 17) - 8);
        int8_t b = (i % 5 == 0) ? int8_t(30) :
                   (i % 7 == 0) ? int8_t(-30) : int8_t((i % 11) - 5);
        src_a[i] = uint8_t(a);
        src_b[i] = uint8_t(b);
        expected[i] = uint8_t(sat_add_ref(a, b));
    }

    sram_write_bytes(s.dut.get(), BUF_ABUF_ID, 0, src_a);
    sram_write_bytes(s.dut.get(), BUF_WBUF_ID, 0, src_b);
    s.load({
        insn::CONFIG_TILE(1, 1, 1),
        insn::VADD(BUF_ABUF_ID, 0, BUF_WBUF_ID, 0, BUF_ABUF_ID, 32, 0),
        insn::HALT(),
    });
    s.run(100000);

    auto got = sram_read_bytes(s.dut.get(), BUF_ABUF_ID, 32 * 16, expected.size());
    if (got != expected) TEST_FAIL(name, "INT8 VADD mismatch");
    TEST_PASS(name);
}

static void test_vadd_bias_int32_wrap() {
    const char* name = "vadd_bias_int32_wrap";
    SimHarness s;
    std::vector<int32_t> accum(16 * 16), bias(16), expected(16 * 16);
    for (int i = 0; i < 16 * 16; ++i)
        accum[i] = (i % 9 == 0) ? int32_t(0x7FFFFFF0u) :
                   (i % 13 == 0) ? int32_t(0x80000010u) :
                   int32_t((i * 97) - 3000);
    for (int j = 0; j < 16; ++j)
        bias[j] = (j % 4 == 0) ? int32_t(0x30u) :
                  (j % 5 == 0) ? int32_t(0xFFFFFFE0u) :
                  int32_t(j * 11 - 40);

    for (int r = 0; r < 16; ++r)
        for (int c = 0; c < 16; ++c)
            expected[r * 16 + c] = int32_t(uint32_t(accum[r * 16 + c]) + uint32_t(bias[c]));

    sram_write_bytes(s.dut.get(), BUF_ACCUM_ID, 0, pack_i32_le(accum));
    sram_write_bytes(s.dut.get(), BUF_WBUF_ID, 200 * 16, pack_i32_le(bias));
    s.load({
        insn::CONFIG_TILE(1, 1, 1),
        insn::VADD(BUF_ACCUM_ID, 0, BUF_WBUF_ID, 200, BUF_ACCUM_ID, 128, 0),
        insn::HALT(),
    });
    s.run(120000);

    auto got = unpack_i32_le(sram_read_bytes(s.dut.get(), BUF_ACCUM_ID, 128 * 16, expected.size() * 4ULL));
    if (got != expected) TEST_FAIL(name, "bias VADD mismatch");
    TEST_PASS(name);
}

static void test_requant_rounding_and_clipping() {
    const char* name = "requant_rounding_and_clipping";
    SimHarness s;
    constexpr uint16_t SCALE_HALF = 0x3800; // 0.5
    const int32_t pattern[16] = {
        1, 3, 5, -1, -3, -5, 255, 257,
        -255, -257, 300, -300, 0, 2, -2, 7
    };
    std::vector<int32_t> src(16 * 16);
    std::vector<uint8_t> expected(16 * 16);
    for (int r = 0; r < 16; ++r) {
        for (int c = 0; c < 16; ++c) {
            int32_t v = pattern[c] + r;
            src[r * 16 + c] = v;
            expected[r * 16 + c] = uint8_t(requant_ref(v, SCALE_HALF));
        }
    }

    sram_write_bytes(s.dut.get(), BUF_ACCUM_ID, 0, pack_i32_le(src));
    s.load({
        insn::CONFIG_TILE(1, 1, 1),
        insn::SET_SCALE(0, SCALE_HALF),
        insn::REQUANT(BUF_ACCUM_ID, 0, BUF_ABUF_ID, 64, 0),
        insn::HALT(),
    });
    s.run(120000);

    auto got = sram_read_bytes(s.dut.get(), BUF_ABUF_ID, 64 * 16, expected.size());
    if (got != expected) TEST_FAIL(name, "requant rounding mismatch");
    TEST_PASS(name);
}

static void test_requant_subnormal_negative_zero() {
    const char* name = "requant_subnormal_negative_zero";
    SimHarness s;
    constexpr uint16_t SCALE_SUBN = 0x0001;
    constexpr uint16_t SCALE_NEG1 = 0xBC00;
    std::vector<int32_t> src(16 * 16, 0);
    std::vector<uint8_t> expected_subn(16 * 16, 0);
    std::vector<uint8_t> expected_neg(16, 0);

    const int32_t row0[16] = {
        1 << 23, 3 << 23, -(1 << 23), -(3 << 23),
        1, -1, 0, 4 << 23,
        -(4 << 23), 5 << 23, -(5 << 23), 127,
        -127, 128, -128, 1000
    };
    for (int c = 0; c < 16; ++c) {
        src[c] = row0[c];
        expected_subn[c] = uint8_t(requant_ref(row0[c], SCALE_SUBN));
    }
    for (int c = 0; c < 16; ++c) {
        src[16 + c] = c - 8;
        expected_subn[16 + c] = uint8_t(requant_ref(src[16 + c], SCALE_SUBN));
        expected_neg[c] = uint8_t(requant_ref(src[16 + c], SCALE_NEG1));
    }

    sram_write_bytes(s.dut.get(), BUF_ACCUM_ID, 0, pack_i32_le(src));
    s.load({
        insn::CONFIG_TILE(1, 1, 1),
        insn::SET_SCALE(0, SCALE_SUBN),
        insn::REQUANT(BUF_ACCUM_ID, 0, BUF_ABUF_ID, 128, 0),
        insn::SET_SCALE(0, SCALE_NEG1),
        insn::REQUANT(BUF_ACCUM_ID, 4, BUF_WBUF_ID, 256, 0),
        insn::SET_SCALE(0, 0x0000),
        insn::REQUANT(BUF_ACCUM_ID, 0, BUF_ABUF_ID, 160, 0),
        insn::HALT(),
    });
    s.run(200000);

    auto got_subn = sram_read_bytes(s.dut.get(), BUF_ABUF_ID, 128 * 16, 256);
    if (!std::equal(expected_subn.begin(), expected_subn.end(), got_subn.begin())) {
        for (size_t i = 0; i < expected_subn.size(); ++i) {
            if (expected_subn[i] != got_subn[i]) {
                fprintf(stderr, "subnormal mismatch idx=%zu got=%d exp=%d\n",
                        i, int(got_subn[i]), int(expected_subn[i]));
                break;
            }
        }
        TEST_FAIL(name, "subnormal requant mismatch");
    }
    auto got_neg = sram_read_bytes(s.dut.get(), BUF_WBUF_ID, 256 * 16, 256);
    if (!std::equal(expected_neg.begin(), expected_neg.end(), got_neg.begin())) {
        for (size_t i = 0; i < 16; ++i) {
            if (expected_neg[i] != got_neg[i]) {
                fprintf(stderr, "negative mismatch idx=%zu got=%d exp=%d\n",
                        i, int(got_neg[i]), int(expected_neg[i]));
                break;
            }
        }
        TEST_FAIL(name, "negative requant mismatch");
    }
    auto got_zero = sram_read_bytes(s.dut.get(), BUF_ABUF_ID, 160 * 16, 256);
    if (!std::all_of(got_zero.begin(), got_zero.end(), [](uint8_t v) { return v == 0; }))
        TEST_FAIL(name, "zero-scale requant mismatch");
    TEST_PASS(name);
}

static void test_requant_pc_per_column() {
    const char* name = "requant_pc_per_column";
    SimHarness s;
    constexpr int SCALE_OFF = 320;
    constexpr int DST_OFF = 512;
    std::vector<int32_t> src(16 * 16);
    std::vector<uint16_t> scales(16);
    std::vector<uint8_t> expected(16 * 16);

    for (int c = 0; c < 16; ++c)
        scales[c] = (c % 4 == 0) ? 0x3C00 :
                    (c % 4 == 1) ? 0x3800 :
                    (c % 4 == 2) ? 0xBC00 : 0x0000;
    for (int r = 0; r < 16; ++r) {
        for (int c = 0; c < 16; ++c) {
            int32_t v = (c - 6) * 37 + r * 5;
            src[r * 16 + c] = v;
            expected[r * 16 + c] = uint8_t(requant_ref(v, scales[c]));
        }
    }

    sram_write_bytes(s.dut.get(), BUF_ACCUM_ID, 0, pack_i32_le(src));
    sram_write_bytes(s.dut.get(), BUF_WBUF_ID, size_t(SCALE_OFF) * 16, pack_u16_le(scales));
    s.load({
        insn::CONFIG_TILE(1, 1, 1),
        insn::REQUANT_PC(BUF_ACCUM_ID, 0, BUF_WBUF_ID, SCALE_OFF, BUF_ABUF_ID, DST_OFF, 0),
        insn::HALT(),
    });
    s.run(100000);

    auto got = sram_read_bytes(s.dut.get(), BUF_ABUF_ID, size_t(DST_OFF) * 16, expected.size());
    if (got != expected) TEST_FAIL(name, "REQUANT_PC mismatch");
    TEST_PASS(name);
}

static void test_scale_mul_int8_roundtrip() {
    const char* name = "scale_mul_int8_roundtrip";
    SimHarness s;
    constexpr int DST_OFF = 640;
    constexpr uint16_t SCALE = 0xB800;  // -0.5
    std::vector<uint8_t> src(16 * 16);
    std::vector<uint8_t> expected(16 * 16);
    for (int i = 0; i < 256; ++i) {
        int8_t v = int8_t(((i * 9) % 61) - 30);
        src[i] = uint8_t(v);
        expected[i] = uint8_t(scale_mul_i8_ref(v, SCALE));
    }

    sram_write_bytes(s.dut.get(), BUF_ABUF_ID, 0, src);
    s.load({
        insn::CONFIG_TILE(1, 1, 1),
        insn::SET_SCALE(2, SCALE),
        insn::SCALE_MUL(BUF_ABUF_ID, 0, BUF_WBUF_ID, DST_OFF, 2),
        insn::HALT(),
    });
    s.run(100000);

    auto got = sram_read_bytes(s.dut.get(), BUF_WBUF_ID, size_t(DST_OFF) * 16, expected.size());
    if (got != expected) TEST_FAIL(name, "SCALE_MUL INT8 mismatch");
    TEST_PASS(name);
}

static void test_scale_mul_accum_roundtrip() {
    const char* name = "scale_mul_accum_roundtrip";
    SimHarness s;
    constexpr int DST_OFF = 256;
    constexpr uint16_t SCALE = 0x4200;  // 3.0
    std::vector<int32_t> src(16 * 16);
    std::vector<int32_t> expected(16 * 16);
    for (int i = 0; i < 256; ++i) {
        src[i] = (i % 13 == 0) ? (INT32_MAX / 2) :
                 (i % 17 == 0) ? (INT32_MIN / 2) :
                 int32_t(i * 1234 - 150000);
        expected[i] = scale_mul_i32_ref(src[i], SCALE);
    }

    sram_write_bytes(s.dut.get(), BUF_ACCUM_ID, 0, pack_i32_le(src));
    s.load({
        insn::CONFIG_TILE(1, 1, 1),
        insn::SET_SCALE(3, SCALE),
        insn::SCALE_MUL(BUF_ACCUM_ID, 0, BUF_ACCUM_ID, DST_OFF, 3),
        insn::HALT(),
    });
    s.run(100000);

    auto got = unpack_i32_le(sram_read_bytes(s.dut.get(), BUF_ACCUM_ID, size_t(DST_OFF) * 16, 16 * 16 * 4ULL));
    if (got != expected) TEST_FAIL(name, "SCALE_MUL ACCUM mismatch");
    TEST_PASS(name);
}

static void test_dequant_add_roundtrip() {
    const char* name = "dequant_add_roundtrip";
    SimHarness s;
    constexpr int DST_OFF = 768;
    constexpr uint16_t ACC_SCALE = 0x2C00;   // 0.0625
    constexpr uint16_t SKIP_SCALE = 0x3400;  // 0.25
    std::vector<int32_t> accum(16 * 16);
    std::vector<uint8_t> skip(16 * 16);
    std::vector<uint8_t> expected(16 * 16);

    for (int i = 0; i < 256; ++i) {
        accum[i] = (i - 120) * 11;
        int8_t skip_i8 = int8_t(((i * 5) % 29) - 14);
        skip[i] = uint8_t(skip_i8);
        expected[i] = uint8_t(dequant_add_ref(accum[i], skip_i8, ACC_SCALE, SKIP_SCALE));
    }

    sram_write_bytes(s.dut.get(), BUF_ACCUM_ID, 0, pack_i32_le(accum));
    sram_write_bytes(s.dut.get(), BUF_ABUF_ID, 0, skip);
    s.load({
        insn::CONFIG_TILE(1, 1, 1),
        insn::SET_SCALE(4, ACC_SCALE),
        insn::SET_SCALE(5, SKIP_SCALE),
        insn::DEQUANT_ADD(BUF_ACCUM_ID, 0, BUF_ABUF_ID, 0, BUF_WBUF_ID, DST_OFF, 4),
        insn::HALT(),
    });
    s.run(100000);

    auto got = sram_read_bytes(s.dut.get(), BUF_WBUF_ID, size_t(DST_OFF) * 16, expected.size());
    if (got != expected) TEST_FAIL(name, "DEQUANT_ADD mismatch");
    TEST_PASS(name);
}

static void test_helper_oob_faults() {
    expect_fault_program("helper_buf_copy_oob_fault",
                         { insn::BUF_COPY(BUF_ABUF_ID, 8191, BUF_WBUF_ID, 0, 2, 0, 0) }, 3, 5000);
    expect_fault_program("helper_vadd_bad_mode_fault",
                         { insn::CONFIG_TILE(1, 1, 1),
                           insn::VADD(BUF_WBUF_ID, 0, BUF_WBUF_ID, 0, BUF_ABUF_ID, 0, 0) }, 6, 5000);
    expect_fault_program("helper_requant_bad_mode_fault",
                         { insn::CONFIG_TILE(1, 1, 1),
                           insn::REQUANT(BUF_ABUF_ID, 0, BUF_WBUF_ID, 0, 0) }, 6, 5000);
    expect_fault_program("helper_dequant_add_bad_sreg_fault",
                         { insn::CONFIG_TILE(1, 1, 1),
                           insn::DEQUANT_ADD(BUF_ACCUM_ID, 0, BUF_ABUF_ID, 0, BUF_WBUF_ID, 0, 15) }, 6, 5000);
}

static void test_matmul_then_requant() {
    const char* name = "matmul_then_requant";
    SimHarness s;
    int8_t a[16][16] = {};
    int8_t eye[16][16] = {};
    int32_t exp_acc[16][16] = {};
    std::vector<uint8_t> expected(256);
    std::vector<uint64_t> prog;
    constexpr uint64_t src_a_addr = 0x280000ULL;
    constexpr uint64_t src_b_addr = 0x281000ULL;

    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            a[i][j] = int8_t((i * 5 + j) - 20);
            eye[i][j] = (i == j) ? 1 : 0;
        }
    }
    matmul_ref(a, eye, exp_acc);
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 16; ++j)
            expected[i * 16 + j] = uint8_t(requant_ref(exp_acc[i][j], 0x3C00));

    systolic_test::prepare_logical_16x16(s.dram, prog, a, eye, src_a_addr, src_b_addr, 128, 0);
    prog.insert(prog.end(), {
        insn::CONFIG_TILE(1, 1, 1),
        insn::MATMUL(BUF_ABUF_ID, 128, BUF_WBUF_ID, 0, BUF_ACCUM_ID, 0, 0),
        insn::SYNC(0b010),
        insn::SET_SCALE(0, 0x3C00),
        insn::REQUANT(BUF_ACCUM_ID, 0, BUF_ABUF_ID, 256, 0),
        insn::HALT(),
    });
    s.load(prog);
    s.run(200000);

    auto got = sram_read_bytes(s.dut.get(), BUF_ABUF_ID, 256 * 16, expected.size());
    if (got != expected) TEST_FAIL(name, "matmul->requant mismatch");
    TEST_PASS(name);
}

static void test_transpose_then_matmul() {
    const char* name = "transpose_then_matmul";
    SimHarness s;
    int8_t a[16][16] = {};
    int8_t k[16][16] = {};
    int8_t kt[16][16] = {};
    int32_t exp[16][16] = {};
    std::vector<uint64_t> prog;
    constexpr uint64_t src_a_addr = 0x282000ULL;
    constexpr uint64_t src_k_addr = 0x283000ULL;

    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            a[i][j] = int8_t(((i * 3 + j * 5) % 11) - 5);
            k[i][j] = int8_t(((i * 7 + j * 2 + 1) % 13) - 6);
            kt[j][i] = k[i][j];
        }
    }

    s.dram.write_bytes(src_k_addr, systolic_test::flatten_16x16(k).data(), 256);
    systolic_test::append_load_sync(prog, 0, src_k_addr, BUF_ABUF_ID, 0, 16);
    systolic_test::append_prepare_a_tile(prog, 1, src_a_addr, 128);
    matmul_ref(a, kt, exp);

    s.dram.write_bytes(src_a_addr, systolic_test::flatten_16x16(a).data(), 256);
    prog.insert(prog.end(), {
        insn::BUF_COPY(BUF_ABUF_ID, 0, BUF_WBUF_ID, 0, 16, 1, 1),
        insn::CONFIG_TILE(1, 1, 1),
        insn::MATMUL(BUF_ABUF_ID, 128, BUF_WBUF_ID, 0, BUF_ACCUM_ID, 0, 0),
        insn::SYNC(0b010),
        insn::HALT(),
    });
    s.load(prog);
    s.run(200000);

    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            int32_t got = read_accum_ij(s.dut.get(), 0, i, j);
            if (got != exp[i][j]) {
                fprintf(stderr, "transpose->matmul mismatch i=%d j=%d got=%d exp=%d\n",
                        i, j, got, exp[i][j]);
                TEST_FAIL(name, "transpose-fed MATMUL mismatch");
            }
        }
    }
    TEST_PASS(name);
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);

    test_buf_copy_flat_interbuffer();
    test_buf_copy_overlap_compaction();
    test_buf_copy_zero_length();
    test_buf_copy_transpose_unaligned_source();
    test_buf_copy_transpose_same_buffer_fault();
    test_vadd_int8_saturating();
    test_vadd_bias_int32_wrap();
    test_requant_rounding_and_clipping();
    test_requant_subnormal_negative_zero();
    test_requant_pc_per_column();
    test_scale_mul_int8_roundtrip();
    test_scale_mul_accum_roundtrip();
    test_dequant_add_roundtrip();
    test_helper_oob_faults();
    test_matmul_then_requant();
    test_transpose_then_matmul();

    printf("\n%d / %d tests passed\n", tests_pass, tests_run);
    if (tests_pass != tests_run) std::exit(1);
    return 0;
}
