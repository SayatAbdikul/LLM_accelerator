// Verilator tests for the Stage D SFU engine.

#include "Vtaccel_top.h"
#include "Vtaccel_top___024root.h"
#include "verilated.h"
#include "include/layernorm_replay_utils.h"
#include "include/testbench.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <array>
#include <memory>
#include <string>
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
constexpr int BUF_ABUF_ID  = tbutil::BUF_ABUF_ID;
constexpr int BUF_WBUF_ID  = tbutil::BUF_WBUF_ID;
constexpr int BUF_ACCUM_ID = tbutil::BUF_ACCUM_ID;

struct Ln1ReplayFixture {
    std::string base;
    std::string metadata_text;
    std::vector<uint8_t> input_bytes;
    std::vector<uint8_t> output_bytes;
    std::vector<uint8_t> gamma_bytes;
    std::vector<uint8_t> beta_bytes;
    std::vector<uint8_t> gamma_beta_bytes;
    int input_off_units = 0;
    int output_off_units = 0;
    int gamma_beta_off_units = 0;
    int sreg_base = 0;
    int in_scale_fp16 = 0;
    int out_scale_fp16 = 0;
    int rows = 0;
    int cols = 0;
    int m_tiles = 0;
    int n_tiles = 0;
};

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

static std::string read_text_file(const char* name, const std::string& path) {
    std::ifstream stream(path);
    if (!stream) {
        std::string msg = "could not open " + path;
        TEST_FAIL(name, msg.c_str());
    }
    return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

static int extract_json_int(const char* name, const std::string& text, const std::string& key) {
    const std::string marker = "\"" + key + "\"";
    const std::size_t key_pos = text.find(marker);
    if (key_pos == std::string::npos) {
        std::string msg = "missing metadata key " + key;
        TEST_FAIL(name, msg.c_str());
    }
    const std::size_t colon = text.find(':', key_pos + marker.size());
    if (colon == std::string::npos) {
        std::string msg = "malformed metadata for key " + key;
        TEST_FAIL(name, msg.c_str());
    }
    std::size_t value_pos = colon + 1u;
    while (value_pos < text.size() &&
           (text[value_pos] == ' ' || text[value_pos] == '\n' ||
            text[value_pos] == '\r' || text[value_pos] == '\t')) {
        ++value_pos;
    }
    std::size_t value_end = value_pos;
    if (value_end < text.size() && text[value_end] == '-')
        ++value_end;
    while (value_end < text.size() && text[value_end] >= '0' && text[value_end] <= '9')
        ++value_end;
    if (value_end == value_pos) {
        std::string msg = "metadata value is not an integer for key " + key;
        TEST_FAIL(name, msg.c_str());
    }
    return std::stoi(text.substr(value_pos, value_end - value_pos));
}

static Ln1ReplayFixture load_ln1_replay_fixture(const char* name) {
    const char* replay_dir = std::getenv("RTL_QKT_REPLAY_DIR");
    if (replay_dir == nullptr || replay_dir[0] == '\0') {
        std::printf("SKIP: %s (set RTL_QKT_REPLAY_DIR to enable)\n", name);
        return {};
    }

    Ln1ReplayFixture fixture;
    fixture.base = std::string(replay_dir);
    fixture.metadata_text = read_text_file(name, fixture.base + "/replay_metadata.json");
    fixture.input_bytes = read_binary_file(name, fixture.base + "/ln1_input_padded.raw");
    fixture.output_bytes = read_binary_file(name, fixture.base + "/ln1_output_padded.raw");
    fixture.gamma_bytes = read_binary_file(name, fixture.base + "/ln1_gamma.raw");
    fixture.beta_bytes = read_binary_file(name, fixture.base + "/ln1_beta.raw");
    fixture.gamma_beta_bytes = fixture.gamma_bytes;
    fixture.gamma_beta_bytes.insert(
        fixture.gamma_beta_bytes.end(),
        fixture.beta_bytes.begin(),
        fixture.beta_bytes.end());

    fixture.input_off_units = extract_json_int(name, fixture.metadata_text, "ln1_input_padded_offset_units");
    fixture.output_off_units = extract_json_int(name, fixture.metadata_text, "ln1_output_padded_offset_units");
    fixture.gamma_beta_off_units = extract_json_int(name, fixture.metadata_text, "ln1_gamma_beta_wbuf_offset_units");
    fixture.sreg_base = extract_json_int(name, fixture.metadata_text, "ln1_sreg_base");
    fixture.in_scale_fp16 = extract_json_int(name, fixture.metadata_text, "ln1_in_scale_fp16");
    fixture.out_scale_fp16 = extract_json_int(name, fixture.metadata_text, "ln1_out_scale_fp16");
    fixture.rows = extract_json_int(name, fixture.metadata_text, "ln1_input_padded_rows");
    fixture.cols = extract_json_int(name, fixture.metadata_text, "ln1_input_padded_cols");
    fixture.m_tiles = fixture.rows / 16;
    fixture.n_tiles = fixture.cols / 16;

    if (fixture.rows != 208 || fixture.cols != 192)
        TEST_FAIL(name, "unexpected LayerNorm replay shape");
    if (fixture.input_bytes.size() != std::size_t(fixture.rows * fixture.cols))
        TEST_FAIL(name, "unexpected ln1_input_padded.raw size");
    if (fixture.output_bytes.size() != std::size_t(fixture.rows * fixture.cols))
        TEST_FAIL(name, "unexpected ln1_output_padded.raw size");
    if (fixture.gamma_bytes.size() != std::size_t(fixture.cols * 2) ||
        fixture.beta_bytes.size() != std::size_t(fixture.cols * 2))
        TEST_FAIL(name, "unexpected LayerNorm gamma/beta payload size");

    return fixture;
}

static bool close_enough(float got, float exp, float tol) {
    return std::fabs(got - exp) <= tol;
}

static std::array<uint8_t, 16> expected_ln_row_chunk(
    const std::vector<uint8_t>& row_major_bytes,
    int cols,
    int row_idx,
    int chunk_idx
) {
    std::array<uint8_t, 16> out{};
    const std::size_t base = std::size_t(row_idx) * std::size_t(cols) + std::size_t(chunk_idx) * 16u;
    for (std::size_t i = 0; i < out.size(); ++i)
        out[i] = row_major_bytes[base + i];
    return out;
}

static std::array<uint8_t, 16> wide_to_bytes(const VlWide<4>& data) {
    std::array<uint8_t, 16> out{};
    for (int w = 0; w < 4; ++w) {
        const uint32_t word = data[w];
        out[std::size_t(w) * 4u + 0u] = uint8_t(word & 0xFFu);
        out[std::size_t(w) * 4u + 1u] = uint8_t((word >> 8) & 0xFFu);
        out[std::size_t(w) * 4u + 2u] = uint8_t((word >> 16) & 0xFFu);
        out[std::size_t(w) * 4u + 3u] = uint8_t((word >> 24) & 0xFFu);
    }
    return out;
}

static float fp16_to_float(uint16_t bits) {
    bool sign = (bits >> 15) & 1;
    int exp = (bits >> 10) & 0x1F;
    int frac = bits & 0x3FF;
    float sign_v = sign ? -1.0f : 1.0f;
    if (exp == 0 && frac == 0)
        return 0.0f;
    if (exp == 0)
        return sign_v * (static_cast<float>(frac) / 1024.0f) * std::ldexp(1.0f, -14);
    if (exp == 31)
        return sign_v * 65504.0f;
    return sign_v * (1.0f + static_cast<float>(frac) / 1024.0f) * std::ldexp(1.0f, exp - 15);
}

static int round_half_even(float x) {
    long long floor_i = static_cast<long long>(std::floor(x));
    float frac = x - static_cast<float>(floor_i);
    if (frac > 0.5f)
        return int(floor_i + 1);
    if (frac < 0.5f)
        return int(floor_i);
    if (floor_i & 1LL)
        return int(floor_i + 1);
    return int(floor_i);
}

static int8_t quantize_ref(float value, float out_scale) {
    if (out_scale == 0.0f)
        return 0;
    int q = round_half_even(value / out_scale);
    if (q > 127) return int8_t(127);
    if (q < -128) return int8_t(-128);
    return int8_t(q);
}

static std::vector<int8_t> softmax_ref(const std::vector<int32_t>& src,
                                       int M, int N, float in_scale, float out_scale) {
    std::vector<int8_t> out(size_t(M) * size_t(N));
    for (int r = 0; r < M; ++r) {
        float row_max = static_cast<float>(src[size_t(r) * size_t(N)]) * in_scale;
        for (int c = 1; c < N; ++c) {
            float x = static_cast<float>(src[size_t(r) * size_t(N) + size_t(c)]) * in_scale;
            row_max = std::max(row_max, x);
        }

        float exp_sum = 0.0f;
        for (int c = 0; c < N; ++c) {
            float x = static_cast<float>(src[size_t(r) * size_t(N) + size_t(c)]) * in_scale;
            exp_sum += std::exp(x - row_max);
        }

        for (int c = 0; c < N; ++c) {
            float x = static_cast<float>(src[size_t(r) * size_t(N) + size_t(c)]) * in_scale;
            out[size_t(r) * size_t(N) + size_t(c)] =
                quantize_ref(std::exp(x - row_max) / exp_sum, out_scale);
        }
    }
    return out;
}

static std::vector<int8_t> softmax_attnv_ref(const std::vector<int32_t>& qkt_i32,
                                             const std::vector<int8_t>& v_i8,
                                             int M, int K, int N,
                                             float qkt_scale, float v_scale, float out_scale) {
    std::vector<int8_t> out(size_t(M) * size_t(N), 0);
    for (int r = 0; r < M; ++r) {
        float row_max = static_cast<float>(qkt_i32[size_t(r) * size_t(K)]) * qkt_scale;
        for (int k = 1; k < K; ++k) {
            float q = static_cast<float>(qkt_i32[size_t(r) * size_t(K) + size_t(k)]) * qkt_scale;
            row_max = std::max(row_max, q);
        }

        std::vector<float> weights(K, 0.0f);
        float exp_sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            float q = static_cast<float>(qkt_i32[size_t(r) * size_t(K) + size_t(k)]) * qkt_scale;
            weights[size_t(k)] = std::exp(q - row_max);
            exp_sum += weights[size_t(k)];
        }

        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                float prob = weights[size_t(k)] / exp_sum;
                float v = static_cast<float>(v_i8[size_t(k) * size_t(N) + size_t(n)]) * v_scale;
                acc += prob * v;
            }
            out[size_t(r) * size_t(N) + size_t(n)] = quantize_ref(acc, out_scale);
        }
    }
    return out;
}

static std::vector<int8_t> layernorm_ref(const std::vector<int8_t>& src,
                                         const std::vector<uint16_t>& gamma_bits,
                                         const std::vector<uint16_t>& beta_bits,
                                         int M, int N, float in_scale, float out_scale) {
    std::vector<float> gamma(N), beta(N);
    for (int i = 0; i < N; ++i) {
        gamma[i] = fp16_to_float(gamma_bits[size_t(i)]);
        beta[i] = fp16_to_float(beta_bits[size_t(i)]);
    }

    std::vector<int8_t> out(size_t(M) * size_t(N));
    for (int r = 0; r < M; ++r) {
        float sum = 0.0f;
        for (int c = 0; c < N; ++c)
            sum += static_cast<float>(src[size_t(r) * size_t(N) + size_t(c)]) * in_scale;
        float mean = sum / static_cast<float>(N);

        float var = 0.0f;
        for (int c = 0; c < N; ++c) {
            float x = static_cast<float>(src[size_t(r) * size_t(N) + size_t(c)]) * in_scale;
            float d = x - mean;
            var += d * d;
        }
        var /= static_cast<float>(N);
        float denom = std::sqrt(var + 1.0e-6f);

        for (int c = 0; c < N; ++c) {
            float x = static_cast<float>(src[size_t(r) * size_t(N) + size_t(c)]) * in_scale;
            float y = ((x - mean) / denom) * gamma[c] + beta[c];
            out[size_t(r) * size_t(N) + size_t(c)] = quantize_ref(y, out_scale);
        }
    }
    return out;
}

static std::vector<int8_t> gelu_ref_i8(const std::vector<int8_t>& src,
                                       float in_scale, float out_scale) {
    std::vector<int8_t> out(src.size());
    const float inv_sqrt2 = 1.0f / std::sqrt(2.0f);
    for (size_t i = 0; i < src.size(); ++i) {
        float x = static_cast<float>(src[i]) * in_scale;
        float y = x * 0.5f * (1.0f + std::erf(x * inv_sqrt2));
        out[i] = quantize_ref(y, out_scale);
    }
    return out;
}

static std::vector<int8_t> gelu_ref_i32(const std::vector<int32_t>& src,
                                        float in_scale, float out_scale) {
    std::vector<int8_t> out(src.size());
    const float inv_sqrt2 = 1.0f / std::sqrt(2.0f);
    for (size_t i = 0; i < src.size(); ++i) {
        float x = static_cast<float>(src[i]) * in_scale;
        float y = x * 0.5f * (1.0f + std::erf(x * inv_sqrt2));
        out[i] = quantize_ref(y, out_scale);
    }
    return out;
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

static void test_softmax_accum_large_row() {
    const char* name = "softmax_accum_large_row";
    constexpr int M = 16;
    constexpr int N = 208;
    constexpr int DST_OFF = 256;
    const float in_scale = fp16_to_float(0x3400);
    const float out_scale = fp16_to_float(0x3400);

    std::vector<int32_t> src(size_t(M) * size_t(N), 0);
    for (int r = 0; r < M; ++r)
        src[size_t(r) * size_t(N) + size_t(N - 1)] = 32;

    auto expected_i8 = softmax_ref(src, M, N, in_scale, out_scale);
    std::vector<uint8_t> expected(expected_i8.size());
    for (size_t i = 0; i < expected.size(); ++i)
        expected[i] = uint8_t(expected_i8[i]);

    SimHarness s;
    sram_write_bytes(s.dut.get(), BUF_ACCUM_ID, 0, pack_i32_le(src));
    s.load({
        insn::CONFIG_TILE(1, 13, 1),
        insn::SET_SCALE(2, 0x3400),
        insn::SET_SCALE(3, 0x3400),
        insn::SOFTMAX(BUF_ACCUM_ID, 0, BUF_ABUF_ID, DST_OFF, 2),
        insn::SYNC(0b100),
        insn::HALT(),
    });
    s.run(200000);
    if (s.dut->fault) TEST_FAIL(name, "unexpected fault");

    auto got = sram_read_bytes(s.dut.get(), BUF_ABUF_ID, size_t(DST_OFF) * 16, expected.size());
    expect_equal_bytes(name, got, expected);
    TEST_PASS(name);
}

static void test_layernorm_identity() {
    const char* name = "layernorm_identity";
    constexpr int M = 16;
    constexpr int N = 192;
    constexpr int DST_OFF = 512;
    const float in_scale = fp16_to_float(0x3400);
    const float out_scale = fp16_to_float(0x3400);

    std::vector<int8_t> src(size_t(M) * size_t(N));
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < N; ++c)
            src[size_t(r) * size_t(N) + size_t(c)] = int8_t(((r * 17 + c * 5) % 41) - 20);

    std::vector<uint16_t> gamma(N, 0x3C00);
    std::vector<uint16_t> beta(N, 0x0000);
    auto expected_i8 = layernorm_ref(src, gamma, beta, M, N, in_scale, out_scale);

    std::vector<uint8_t> src_bytes(src.size());
    std::vector<uint8_t> expected(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        src_bytes[i] = uint8_t(src[i]);
        expected[i] = uint8_t(expected_i8[i]);
    }

    std::vector<uint8_t> gb_bytes = pack_u16_le(gamma);
    auto beta_bytes = pack_u16_le(beta);
    gb_bytes.insert(gb_bytes.end(), beta_bytes.begin(), beta_bytes.end());

    SimHarness s;
    sram_write_bytes(s.dut.get(), BUF_ABUF_ID, 0, src_bytes);
    sram_write_bytes(s.dut.get(), BUF_WBUF_ID, 0, gb_bytes);
    s.load({
        insn::CONFIG_TILE(1, 12, 1),
        insn::SET_SCALE(3, 0x3400),
        insn::SET_SCALE(4, 0x3400),
        insn::LAYERNORM(BUF_ABUF_ID, 0, BUF_WBUF_ID, 0, BUF_ABUF_ID, DST_OFF, 3),
        insn::SYNC(0b100),
        insn::HALT(),
    });
    s.run(250000);
    if (s.dut->fault) TEST_FAIL(name, "unexpected fault");

    auto got = sram_read_bytes(s.dut.get(), BUF_ABUF_ID, size_t(DST_OFF) * 16, expected.size());
    expect_equal_bytes(name, got, expected);
    TEST_PASS(name);
}

static void test_layernorm_replay_probe() {
    const char* name = "layernorm_replay_probe";
    const Ln1ReplayFixture fixture = load_ln1_replay_fixture(name);
    if (fixture.base.empty())
        return;

    constexpr int F_LN_PARAM_REQ = 1;
    constexpr int F_ROW_I8_REQ = 3;
    constexpr int F_ROW_COMPUTE = 7;
    constexpr int F_ROW_PACK = 8;
    constexpr int F_ROW_WRITE = 9;

    const auto row0_ref = lnreplay::compute_ln_row_reference(
        fixture.input_bytes,
        fixture.gamma_bytes,
        fixture.beta_bytes,
        fixture.rows,
        fixture.cols,
        0,
        fixture.in_scale_fp16,
        fixture.out_scale_fp16);

    SimHarness s;
    sram_write_bytes(
        s.dut.get(),
        BUF_ABUF_ID,
        size_t(fixture.input_off_units) * 16u,
        fixture.input_bytes);
    sram_write_bytes(
        s.dut.get(),
        BUF_WBUF_ID,
        size_t(fixture.gamma_beta_off_units) * 16u,
        fixture.gamma_beta_bytes);

    s.load({
        insn::CONFIG_TILE(fixture.m_tiles, fixture.n_tiles, 1),
        insn::SET_SCALE(fixture.sreg_base, uint16_t(fixture.in_scale_fp16), 0),
        insn::SET_SCALE(fixture.sreg_base + 1, uint16_t(fixture.out_scale_fp16), 0),
        insn::LAYERNORM(
            BUF_ABUF_ID,
            fixture.input_off_units,
            BUF_WBUF_ID,
            fixture.gamma_beta_off_units,
            BUF_ABUF_ID,
            fixture.output_off_units,
            fixture.sreg_base,
            0),
        insn::SYNC(0b100),
        insn::HALT(),
    });

    auto fail_stage = [&](const char* stage, const char* detail) {
        std::fprintf(stderr, "%s stage failure: %s\n", stage, detail);
        TEST_FAIL(name, "replay probe mismatch");
    };

    auto compare_real = [&](const char* stage, const char* field, float got, float exp, float tol) {
        if (!close_enough(got, exp, tol)) {
            std::fprintf(stderr, "%s mismatch in %s: got=%0.12f exp=%0.12f tol=%g\n",
                         stage, field, got, exp, tol);
            TEST_FAIL(name, "replay probe mismatch");
        }
    };

    auto* r = s.dut->rootp;
    bool checked_scales = false;
    bool checked_params = false;
    bool checked_row_data = false;
    bool checked_compute = false;
    bool checked_writeback = false;
    bool row0_write_window_complete = false;
    int observed_row0_writes = 0;
    int sim_cycle = 0;
    struct PendingReadback {
        bool valid = false;
        int abs_row = -1;
        int row_idx = -1;
        int chunk_idx = -1;
        std::array<uint8_t, 16> expected{};
    } pending_readback;

    auto probe = [&]() {
        const int state = int(r->taccel_top__DOT__u_sfu__DOT__state);
        const int row_idx = int(r->taccel_top__DOT__u_sfu__DOT__row_idx_q);
        const int read_idx = int(r->taccel_top__DOT__u_sfu__DOT__read_idx_q);
        const int write_chunk = int(r->taccel_top__DOT__u_sfu__DOT__write_chunk_q);

        if (pending_readback.valid) {
            auto got = sram_read_bytes(
                s.dut.get(),
                BUF_ABUF_ID,
                std::size_t(pending_readback.abs_row) * 16u,
                pending_readback.expected.size());
            for (std::size_t i = 0; i < pending_readback.expected.size(); ++i) {
                if (got[i] != pending_readback.expected[i]) {
                    std::fprintf(stderr,
                                 "row_write_readback mismatch row=%d chunk=%d byte=%zu got=%d exp=%d\n",
                                 pending_readback.row_idx,
                                 pending_readback.chunk_idx,
                                 i,
                                 int(int8_t(got[i])),
                                 int(int8_t(pending_readback.expected[i])));
                    TEST_FAIL(name, "replay probe mismatch");
                }
            }
            if (pending_readback.row_idx == 0 && pending_readback.chunk_idx == fixture.n_tiles - 1)
                row0_write_window_complete = true;
            pending_readback.valid = false;
        }

        if (!checked_scales && state == F_LN_PARAM_REQ) {
            compare_real("scale_setup", "scale0_q", r->taccel_top__DOT__u_sfu__DOT__scale0_q, row0_ref.scale0, 1.0e-6f);
            compare_real("scale_setup", "scale1_q", r->taccel_top__DOT__u_sfu__DOT__scale1_q, row0_ref.scale1, 1.0e-6f);
            checked_scales = true;
        }

        if (!checked_params && state == F_ROW_I8_REQ && row_idx == 0 && read_idx == 0) {
            for (int i = 0; i < fixture.cols; ++i) {
                const float got_gamma = r->taccel_top__DOT__u_sfu__DOT__gamma_q[i];
                const float got_beta = r->taccel_top__DOT__u_sfu__DOT__beta_q[i];
                const float exp_gamma = row0_ref.gamma[size_t(i)];
                const float exp_beta = row0_ref.beta[size_t(i)];
                if (!close_enough(got_gamma, exp_gamma, 1.0e-6f)) {
                    std::fprintf(stderr, "param_latch gamma mismatch idx=%d got=%0.12f exp=%0.12f\n",
                                 i, got_gamma, exp_gamma);
                    TEST_FAIL(name, "replay probe mismatch");
                }
                if (!close_enough(got_beta, exp_beta, 1.0e-6f)) {
                    std::fprintf(stderr, "param_latch beta mismatch idx=%d got=%0.12f exp=%0.12f\n",
                                 i, got_beta, exp_beta);
                    TEST_FAIL(name, "replay probe mismatch");
                }
            }
            checked_params = true;
        }

        if (!checked_row_data && state == F_ROW_COMPUTE && row_idx == 0) {
            for (int i = 0; i < fixture.cols; ++i) {
                const float got = r->taccel_top__DOT__u_sfu__DOT__row_data_q[i];
                const float exp = row0_ref.row_data[size_t(i)];
                if (!close_enough(got, exp, 1.0e-5f)) {
                    std::fprintf(stderr, "row_ingest mismatch idx=%d got=%0.12f exp=%0.12f\n",
                                 i, got, exp);
                    TEST_FAIL(name, "replay probe mismatch");
                }
            }
            checked_row_data = true;
        }

        if (!checked_compute && state == F_ROW_PACK && row_idx == 0 && write_chunk == 0) {
            compare_real("row_compute", "mean", r->taccel_top__DOT__u_sfu__DOT__ln_debug_mean_q, row0_ref.mean, 1.0e-5f);
            compare_real("row_compute", "var", r->taccel_top__DOT__u_sfu__DOT__ln_debug_var_q, row0_ref.var, 1.0e-5f);
            compare_real("row_compute", "denom", r->taccel_top__DOT__u_sfu__DOT__ln_debug_denom_q, row0_ref.denom, 1.0e-5f);
            for (int i = 0; i < 16; ++i) {
                const float got_y = r->taccel_top__DOT__u_sfu__DOT__ln_debug_y_q[i];
                const float exp_y = row0_ref.y_prefix[size_t(i)];
                if (!close_enough(got_y, exp_y, 1.0e-5f)) {
                    std::fprintf(stderr, "row_compute y mismatch idx=%d got=%0.12f exp=%0.12f\n",
                                 i, got_y, exp_y);
                    TEST_FAIL(name, "replay probe mismatch");
                }
                const uint8_t got_out = r->taccel_top__DOT__u_sfu__DOT__out_bytes_q[i];
                const uint8_t exp_out = row0_ref.out_row_bytes[size_t(i)];
                if (got_out != exp_out) {
                    std::fprintf(stderr, "row_compute out_bytes mismatch idx=%d got=%d exp=%d\n",
                                 i, int(int8_t(got_out)), int(int8_t(exp_out)));
                    TEST_FAIL(name, "replay probe mismatch");
                }
            }
            checked_compute = true;
        }

        if (state == F_ROW_WRITE) {
            const int sram_row = int(r->taccel_top__DOT__sfu_sram_a_row);
            const auto payload = wide_to_bytes(r->taccel_top__DOT__sfu_sram_a_wdata);

            if (row_idx == 0) {
                const int exp_row = fixture.output_off_units + write_chunk;
                const auto exp_chunk = expected_ln_row_chunk(fixture.output_bytes, fixture.cols, 0, write_chunk);
                if (sram_row != exp_row) {
                    std::fprintf(stderr,
                                 "row_write_addr mismatch chunk=%d got_row=%d exp_row=%d\n",
                                 write_chunk, sram_row, exp_row);
                    TEST_FAIL(name, "replay probe mismatch");
                }
                for (std::size_t i = 0; i < exp_chunk.size(); ++i) {
                    if (payload[i] != exp_chunk[i]) {
                        std::fprintf(stderr,
                                     "row_write_payload mismatch chunk=%d byte=%zu got=%d exp=%d\n",
                                     write_chunk,
                                     i,
                                     int(int8_t(payload[i])),
                                     int(int8_t(exp_chunk[i])));
                        TEST_FAIL(name, "replay probe mismatch");
                    }
                }
                pending_readback.valid = true;
                pending_readback.abs_row = sram_row;
                pending_readback.row_idx = row_idx;
                pending_readback.chunk_idx = write_chunk;
                pending_readback.expected = exp_chunk;
                observed_row0_writes++;
                checked_writeback = true;
            } else if (row0_write_window_complete &&
                       sram_row >= fixture.output_off_units &&
                       sram_row < fixture.output_off_units + fixture.n_tiles) {
                std::fprintf(stderr,
                             "post_write_overwrite detected cycle=%d row_idx=%d chunk=%d sram_row=%d\n",
                             sim_cycle, row_idx, write_chunk, sram_row);
                TEST_FAIL(name, "replay probe mismatch");
            }
        }
    };

    s.start_once();
    r = s.dut->rootp;
    probe();

    for (int cycle = 0; cycle < 1000000; ++cycle) {
        if (s.dut->done || s.dut->fault)
            break;
        s.step();
        r = s.dut->rootp;
        sim_cycle = cycle + 1;
        probe();
    }

    if (!checked_scales)
        fail_stage("scale_setup", "did not observe F_LN_PARAM_REQ");
    if (!checked_params)
        fail_stage("param_latch", "did not observe completed gamma/beta latch");
    if (!checked_row_data)
        fail_stage("row_ingest", "did not observe row 0 compute entry");
    if (!checked_compute)
        fail_stage("row_compute", "did not observe row 0 compute completion");
    if (!checked_writeback)
        fail_stage("row_write", "did not observe row 0 writeback");
    if (observed_row0_writes != fixture.n_tiles) {
        std::fprintf(stderr, "row_write count mismatch got=%d exp=%d\n",
                     observed_row0_writes, fixture.n_tiles);
        TEST_FAIL(name, "replay probe mismatch");
    }
    if (s.dut->fault)
        TEST_FAIL(name, "unexpected fault");
    if (!s.dut->done)
        TEST_FAIL(name, "did not halt");

    auto got = sram_read_bytes(
        s.dut.get(),
        BUF_ABUF_ID,
        size_t(fixture.output_off_units) * 16u,
        fixture.output_bytes.size());
    if (got.size() != fixture.output_bytes.size())
        TEST_FAIL(name, "length mismatch");
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (got[i] != fixture.output_bytes[i]) {
            const int row = int(i / std::size_t(fixture.cols));
            const int col = int(i % std::size_t(fixture.cols));
            std::fprintf(stderr,
                         "final_output mismatch row=%d col=%d got=%d exp=%d\n",
                         row, col, int(int8_t(got[i])), int(int8_t(fixture.output_bytes[i])));
            TEST_FAIL(name, "replay probe mismatch");
        }
    }

    TEST_PASS(name);
}

static void test_gelu_abuf_roundtrip() {
    const char* name = "gelu_abuf_roundtrip";
    constexpr int M = 16;
    constexpr int N = 16;
    constexpr int DST_OFF = 1024;
    const float in_scale = fp16_to_float(0x3400);
    const float out_scale = fp16_to_float(0x3400);

    std::vector<int8_t> src(size_t(M) * size_t(N));
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < N; ++c)
            src[size_t(r) * size_t(N) + size_t(c)] = int8_t((c - 8) + r);

    auto expected_i8 = gelu_ref_i8(src, in_scale, out_scale);
    std::vector<uint8_t> src_bytes(src.size()), expected(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        src_bytes[i] = uint8_t(src[i]);
        expected[i] = uint8_t(expected_i8[i]);
    }

    SimHarness s;
    sram_write_bytes(s.dut.get(), BUF_ABUF_ID, 0, src_bytes);
    s.load({
        insn::CONFIG_TILE(1, 1, 1),
        insn::SET_SCALE(5, 0x3400),
        insn::SET_SCALE(6, 0x3400),
        insn::GELU(BUF_ABUF_ID, 0, BUF_WBUF_ID, DST_OFF, 5),
        insn::SYNC(0b100),
        insn::HALT(),
    });
    s.run(100000);
    if (s.dut->fault) TEST_FAIL(name, "unexpected fault");

    auto got = sram_read_bytes(s.dut.get(), BUF_WBUF_ID, size_t(DST_OFF) * 16, expected.size());
    expect_equal_bytes(name, got, expected);
    TEST_PASS(name);
}

static void test_gelu_accum_roundtrip() {
    const char* name = "gelu_accum_roundtrip";
    constexpr int M = 16;
    constexpr int N = 16;
    constexpr int DST_OFF = 1280;
    const float in_scale = fp16_to_float(0x3400);
    const float out_scale = fp16_to_float(0x3400);

    std::vector<int32_t> src(size_t(M) * size_t(N));
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < N; ++c)
            src[size_t(r) * size_t(N) + size_t(c)] = (c - 8) * 2 + r;

    auto expected_i8 = gelu_ref_i32(src, in_scale, out_scale);
    std::vector<uint8_t> expected(expected_i8.size());
    for (size_t i = 0; i < expected.size(); ++i)
        expected[i] = uint8_t(expected_i8[i]);

    SimHarness s;
    sram_write_bytes(s.dut.get(), BUF_ACCUM_ID, 0, pack_i32_le(src));
    s.load({
        insn::CONFIG_TILE(1, 1, 1),
        insn::SET_SCALE(8, 0x3400),
        insn::SET_SCALE(9, 0x3400),
        insn::GELU(BUF_ACCUM_ID, 0, BUF_ABUF_ID, DST_OFF, 8),
        insn::SYNC(0b100),
        insn::HALT(),
    });
    s.run(100000);
    if (s.dut->fault) TEST_FAIL(name, "unexpected fault");

    auto got = sram_read_bytes(s.dut.get(), BUF_ABUF_ID, size_t(DST_OFF) * 16, expected.size());
    expect_equal_bytes(name, got, expected);
    TEST_PASS(name);
}

static void test_softmax_attnv_roundtrip() {
    const char* name = "softmax_attnv_roundtrip";
    constexpr int M = 16;
    constexpr int K = 16;
    constexpr int N = 16;
    constexpr int DST_OFF = 1536;
    constexpr uint16_t QKT_SCALE_BITS = 0x2C00;   // 0.0625
    constexpr uint16_t V_SCALE_BITS   = 0x3400;   // 0.25
    constexpr uint16_t OUT_SCALE_BITS = 0x3400;   // 0.25
    constexpr uint16_t TRACE_BITS     = 0x3000;   // trace-only
    const float qkt_scale = fp16_to_float(QKT_SCALE_BITS);
    const float v_scale = fp16_to_float(V_SCALE_BITS);
    const float out_scale = fp16_to_float(OUT_SCALE_BITS);

    std::vector<int32_t> qkt(size_t(M) * size_t(K));
    std::vector<int8_t> v(size_t(K) * size_t(N));
    for (int r = 0; r < M; ++r)
        for (int k = 0; k < K; ++k)
            qkt[size_t(r) * size_t(K) + size_t(k)] = ((r * 3 + k * 5) % 19) - 9;
    for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n)
            v[size_t(k) * size_t(N) + size_t(n)] = int8_t(((k * 7 + n * 11) % 23) - 11);

    auto expected_i8 = softmax_attnv_ref(qkt, v, M, K, N, qkt_scale, v_scale, out_scale);
    std::vector<uint8_t> v_bytes(v.size()), expected(expected_i8.size());
    for (size_t i = 0; i < v.size(); ++i)
        v_bytes[i] = uint8_t(v[i]);
    for (size_t i = 0; i < expected.size(); ++i)
        expected[i] = uint8_t(expected_i8[i]);

    SimHarness s;
    sram_write_bytes(s.dut.get(), BUF_ACCUM_ID, 0, pack_i32_le(qkt));
    sram_write_bytes(s.dut.get(), BUF_ABUF_ID, 0, v_bytes);
    s.load({
        insn::CONFIG_TILE(1, 1, 1),
        insn::SET_SCALE(4, QKT_SCALE_BITS),
        insn::SET_SCALE(5, V_SCALE_BITS),
        insn::SET_SCALE(6, OUT_SCALE_BITS),
        insn::SET_SCALE(7, TRACE_BITS),
        insn::SOFTMAX_ATTNV(BUF_ACCUM_ID, 0, BUF_ABUF_ID, 0, BUF_WBUF_ID, DST_OFF, 4),
        insn::SYNC(0b100),
        insn::HALT(),
    });
    s.run(250000);
    if (s.dut->fault) TEST_FAIL(name, "unexpected fault");

    auto got = sram_read_bytes(s.dut.get(), BUF_WBUF_ID, size_t(DST_OFF) * 16, expected.size());
    expect_equal_bytes(name, got, expected);
    TEST_PASS(name);
}

static void test_softmax_attnv_zero_out_scale() {
    const char* name = "softmax_attnv_zero_out_scale";
    constexpr int DST_OFF = 1664;
    SimHarness s;
    std::vector<int32_t> qkt(16 * 16, 0);
    std::vector<uint8_t> v(16 * 16, 0x7F);
    sram_write_bytes(s.dut.get(), BUF_ACCUM_ID, 0, pack_i32_le(qkt));
    sram_write_bytes(s.dut.get(), BUF_ABUF_ID, 0, v);
    s.load({
        insn::CONFIG_TILE(1, 1, 1),
        insn::SET_SCALE(8, 0x3400),
        insn::SET_SCALE(9, 0x3400),
        insn::SET_SCALE(10, 0x0000),
        insn::SET_SCALE(11, 0x3000),
        insn::SOFTMAX_ATTNV(BUF_ACCUM_ID, 0, BUF_ABUF_ID, 0, BUF_WBUF_ID, DST_OFF, 8),
        insn::SYNC(0b100),
        insn::HALT(),
    });
    s.run(250000);
    if (s.dut->fault) TEST_FAIL(name, "unexpected fault");

    auto got = sram_read_bytes(s.dut.get(), BUF_WBUF_ID, size_t(DST_OFF) * 16, 256);
    if (!std::all_of(got.begin(), got.end(), [](uint8_t vbyte) { return vbyte == 0; }))
        TEST_FAIL(name, "zero out-scale should zero destination");
    TEST_PASS(name);
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);

    test_softmax_accum_large_row();
    test_layernorm_identity();
    test_layernorm_replay_probe();
    test_gelu_abuf_roundtrip();
    test_gelu_accum_roundtrip();
    test_softmax_attnv_roundtrip();
    test_softmax_attnv_zero_out_scale();

    std::printf("\n%d / %d tests passed\n", tests_pass, tests_run);
    if (tests_pass != tests_run) std::exit(1);
    return 0;
}
