// Bit-exact gate for fp32_to_fp16_pipe (RNE fp32->fp16). Golden = the DPI
// `sfu_fp32_to_fp16_bits` semantics: (_Float16)(float) cast, NaN canonicalized
// to fp16 qNaN 0x7E00. Uses Apple clang's __fp16 IEEE half type which casts
// from float with IEEE RNE matching `_Float16`.
//
// Test vector pattern: directed edges (zero/inf/NaN, ±FLT_MAX, overflow
// boundaries, RNE ties at fp16 ULPs, fp16 subnormal range boundaries
// E in [-25,-15], smallest fp16 subnormal/normal boundary) + millions
// random full + millions random normals.

#include "Vfp32_to_fp16_pipe.h"
#include "verilated.h"

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

static uint16_t golden_fp32_to_fp16(uint32_t bits) {
  float f = bits2f(bits);
  if (std::isnan(f)) return 0x7E00u;             // canonical fp16 qNaN
  __fp16 h = (__fp16)f;
  uint16_t hb;
  std::memcpy(&hb, &h, 2);
  return hb;
}

static void tick(Vfp32_to_fp16_pipe* dut) {
  dut->clk = 0; dut->eval();
  dut->clk = 1; dut->eval();
}

static void reset(Vfp32_to_fp16_pipe* dut) {
  dut->rst_n = 0; dut->valid_in = 0; dut->a = 0;
  for (int i = 0; i < 4; ++i) tick(dut);
  dut->rst_n = 1;
}

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  Vfp32_to_fp16_pipe* dut = new Vfp32_to_fp16_pipe;
  reset(dut);

  std::vector<uint32_t> ops;

  // Directed edges:
  const uint32_t E[] = {
    0x00000000u, 0x80000000u,                   // ±0
    0x00000001u, 0x80000001u,                   // smallest fp32 subnormals
    0x007FFFFFu, 0x00800000u, 0x00800001u,      // sub/normal boundary
    0x3F800000u, 0xBF800000u, 0x40000000u,      // ±1.0, +2.0
    0x3F7FFFFFu, 0x3F800001u,                   // 1-eps, 1+eps
    0x7F7FFFFFu, 0xFF7FFFFFu,                   // ±FLT_MAX (overflow)
    0x7F800000u, 0xFF800000u,                   // ±inf
    0x7FC00000u, 0x7F800001u, 0xFFC00000u,      // NaN forms
    // fp16 overflow boundary: fp16 max ≈ 65504 = 0x477FE000 (fp32); overflow above.
    0x477FE000u, 0x477FF000u, 0x477FF800u,      // around fp16 max, RNE-overflow ties
    0x477FFFFFu, 0x477FE001u,                   // just below/at fp16 max
    // fp16 normal/subnormal boundary: 2^-14 = 0x38800000 (fp32); below -> sub.
    0x38800000u, 0x38000000u, 0x38400000u,
    // fp16 smallest subnormal = 2^-24 = 0x33800000; below -> 0 (with RNE tie at 2^-25=0x33000000).
    0x33800000u, 0x33000000u, 0x32FFFFFFu, 0x33800001u,
    // Subnormal fp32 (should -> fp16 ±0)
    0x00400000u, 0x80400000u,
  };
  for (auto v : E) ops.push_back(v);

  std::mt19937 rng(0x132F1632u);
  std::uniform_int_distribution<uint32_t> d32(0, 0xFFFFFFFFu);
  for (long n = 0; n < 3000000; ++n) ops.push_back(d32(rng));
  for (long n = 0; n < 3000000; ++n) {
    uint32_t s = d32(rng) & 1u;
    uint32_t e = 1u + (d32(rng) % 254u);
    uint32_t m = d32(rng) & 0x7FFFFFu;
    ops.push_back((s << 31) | (e << 23) | m);
  }
  // Bias a million toward the subnormal/edge range (E in [-30, 16])
  // to stress the subnormal-output branch.
  for (long n = 0; n < 1000000; ++n) {
    uint32_t s = d32(rng) & 1u;
    uint32_t e = 97u + (d32(rng) % 60u);         // E in [-30, 30]
    uint32_t m = d32(rng) & 0x7FFFFFu;
    ops.push_back((s << 31) | (e << 23) | m);
  }

  std::queue<uint16_t> in_flight;
  for (size_t k = 0; k < ops.size(); ++k) {
    dut->valid_in = 1;
    dut->a = ops[k];
    uint16_t exp = golden_fp32_to_fp16(ops[k]);
    in_flight.push(exp);
    tick(dut);
    if (dut->valid_out) {
      uint16_t got = dut->y;
      uint16_t e   = in_flight.front(); in_flight.pop();
      ++g_checked;
      if (got != e) {
        if (g_fail < 20)
          std::fprintf(stderr,
                       "MISMATCH op=%zu a=%08x got=%04x exp=%04x  (%g)\n",
                       k, ops[k], got, e, (double)bits2f(ops[k]));
        ++g_fail;
      }
    }
  }
  dut->valid_in = 0;
  while (!in_flight.empty()) {
    tick(dut);
    if (dut->valid_out) {
      uint16_t got = dut->y;
      uint16_t e   = in_flight.front(); in_flight.pop();
      ++g_checked;
      if (got != e) ++g_fail;
    }
  }

  delete dut;
  std::printf("fp32_to_fp16_pipe (LATENCY=%d): checked=%ld  mismatches=%d\n",
              PIPE_LATENCY, g_checked, g_fail);
  if (g_fail == 0 && g_checked == (long)ops.size()) {
    std::printf("PASS: fp32_to_fp16_pipe bit-exact vs host __fp16 cast\n");
    return 0;
  }
  std::fprintf(stderr, "FAIL: fp32_to_fp16_pipe %d mismatches (checked=%ld)\n",
               g_fail, g_checked);
  return 1;
}
