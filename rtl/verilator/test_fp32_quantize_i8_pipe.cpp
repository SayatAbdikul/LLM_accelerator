// Bit-exact gate for fp32_quantize_i8_pipe (single-input variant: assumes
// FSM has done value/out_scale first; out_scale==0 short-circuit handled
// at the SFU level). Golden mirrors sfu_fp32_quantize_i8 with out_scale=1.0:
//   NaN -> 0, +inf -> 127, -inf -> -128, finite -> clamp(RNE(v), -128, 127).

#include "Vfp32_quantize_i8_pipe.h"
#include "verilated.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <queue>
#include <random>
#include <vector>

#ifndef PIPE_LATENCY
#define PIPE_LATENCY 1
#endif

static int g_fail = 0;
static long g_checked = 0;

static inline float bits2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

// Mirror sfu_round_half_even_fp32_scalar from testbench.h
static int rne_int(float x) {
  long long floor_i = (long long)std::floor(x);
  float frac = x - (float)floor_i;
  if (frac > 0.5f) return (int)(floor_i + 1);
  if (frac < 0.5f) return (int)floor_i;
  // tie: pick even
  if (floor_i & 1LL) return (int)(floor_i + 1);
  return (int)floor_i;
}

static int8_t golden(uint32_t bits) {
  float v = bits2f(bits);
  if (std::isnan(v)) return 0;
  if (std::isinf(v)) return v > 0.0f ? 127 : -128;
  // Pre-saturate before UB territory (testbench.h's clamp catches the int-range
  // overflow but only if (long long)floor(v) doesn't itself UB; on huge |v|
  // the cast is UB, so we apply the SAME semantic clamp at the float level
  // here. 127.5 RNE-ties to 128, clamps to 127; -128.5 RNE-ties to -128.):
  if (v >=  127.5f) return  127;
  if (v <= -128.5f) return -128;
  int q = rne_int(v);
  if (q > 127) q = 127;
  if (q < -128) q = -128;
  return (int8_t)q;
}

static void tick(Vfp32_quantize_i8_pipe* dut) {
  dut->clk = 0; dut->eval();
  dut->clk = 1; dut->eval();
}

static void reset(Vfp32_quantize_i8_pipe* dut) {
  dut->rst_n = 0; dut->valid_in = 0; dut->a = 0;
  for (int i = 0; i < 4; ++i) tick(dut);
  dut->rst_n = 1;
}

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  Vfp32_quantize_i8_pipe* dut = new Vfp32_quantize_i8_pipe;
  reset(dut);

  std::vector<uint32_t> ops;
  // Edges:
  const uint32_t E[] = {
    0x00000000u, 0x80000000u,                          // ±0
    0x3F000000u, 0xBF000000u,                          // ±0.5 (tie to even 0)
    0x3F800000u, 0xBF800000u,                          // ±1.0 (tie to even 1?)
    0x3FC00000u, 0xBFC00000u,                          // ±1.5 (tie to 2/-2)
    0x40000000u, 0xC0000000u,                          // ±2.0
    0x40200000u,                                       // 2.5 (tie to 2)
    0x40600000u,                                       // 3.5 (tie to 4)
    0x42FE0000u, 0x43000000u,                          // 127.0, 128.0 (saturate)
    0xC3000000u, 0xC3000001u,                          // -128.0, just below
    0x47F00000u, 0xC7F00000u,                          // large finite (>>127)
    0x7F800000u, 0xFF800000u,                          // ±inf
    0x7FC00000u, 0xFFC00000u,                          // NaN
    0x33800000u,                                       // 2^-24 (tiny -> 0)
    0x00000001u, 0x80000001u, 0x007FFFFFu,             // fp32 subnormals -> 0
    0x42FFFFFFu,                                       // 127.99... (RNE up to 128 -> 127 sat)
  };
  for (auto v : E) ops.push_back(v);

  std::mt19937 rng(0xA8B132u);
  std::uniform_int_distribution<uint32_t> d32(0, 0xFFFFFFFFu);
  for (long n = 0; n < 3000000; ++n) ops.push_back(d32(rng));
  // Many in the "interesting" range [0.25, 256] to stress saturation + ties.
  for (long n = 0; n < 1000000; ++n) {
    uint32_t s = d32(rng) & 1u;
    uint32_t e = 125u + (d32(rng) % 15u);    // E in [-2, 12]
    uint32_t m = d32(rng) & 0x7FFFFFu;
    ops.push_back((s << 31) | (e << 23) | m);
  }
  // Tie territory: exact halves (E in [-1, 6], m = 0)
  for (int E_off = -1; E_off <= 6; ++E_off) {
    for (uint32_t s = 0; s < 2; ++s) {
      uint32_t e = (uint32_t)(127 + E_off);
      ops.push_back((s << 31) | (e << 23));    // exact tie
    }
  }

  std::queue<int8_t> in_flight;
  for (size_t k = 0; k < ops.size(); ++k) {
    dut->valid_in = 1;
    dut->a = ops[k];
    int8_t exp = golden(ops[k]);
    in_flight.push(exp);
    tick(dut);
    if (dut->valid_out) {
      int8_t got = (int8_t)dut->y;
      int8_t e   = in_flight.front(); in_flight.pop();
      ++g_checked;
      if (got != e) {
        if (g_fail < 20)
          std::fprintf(stderr,
                       "MISMATCH op=%zu a=%08x got=%d exp=%d  (%g)\n",
                       k, ops[k], (int)got, (int)e, (double)bits2f(ops[k]));
        ++g_fail;
      }
    }
  }
  dut->valid_in = 0;
  while (!in_flight.empty()) {
    tick(dut);
    if (dut->valid_out) {
      int8_t got = (int8_t)dut->y;
      int8_t e   = in_flight.front(); in_flight.pop();
      ++g_checked;
      if (got != e) ++g_fail;
    }
  }

  delete dut;
  std::printf("fp32_quantize_i8_pipe (LATENCY=%d): checked=%ld  mismatches=%d\n",
              PIPE_LATENCY, g_checked, g_fail);
  if (g_fail == 0 && g_checked == (long)ops.size()) {
    std::printf("PASS: fp32_quantize_i8_pipe bit-exact vs Option-B golden\n");
    return 0;
  }
  std::fprintf(stderr, "FAIL: fp32_quantize_i8_pipe %d mismatches (checked=%ld)\n",
               g_fail, g_checked);
  return 1;
}
