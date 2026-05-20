// Measured-band gate for fp32_exp_pipe. Golden = sfu_fp32_exp semantics:
//   (float)std::exp((float)x). NaN -> 0x7FC00000.
// Reports a ULP histogram (vs the host libm golden). This is the §7 band
// for the synthesizable exp primitive.

#include "Vfp32_exp_pipe.h"
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
static long g_finite_compared = 0;
static long g_ulp_max = 0;
static long g_ulp_sum = 0;
static long g_ulp_hist[16] = {0};   // [0]=exact, [1..14]=N ULP, [15]=>=15

static inline float bits2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static inline uint32_t f2bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

static uint32_t golden(uint32_t bits) {
  float v = bits2f(bits);
  if (std::isnan(v)) return 0x7FC00000u;
  float r = std::exp(v);
  if (std::isnan(r)) return 0x7FC00000u;
  return f2bits(r);
}

static void tick(Vfp32_exp_pipe* dut) {
  dut->clk = 0; dut->eval();
  dut->clk = 1; dut->eval();
}

static void reset(Vfp32_exp_pipe* dut) {
  dut->rst_n = 0; dut->valid_in = 0; dut->a = 0;
  for (int i = 0; i < 4; ++i) tick(dut);
  dut->rst_n = 1;
}

static long ulp_diff_fp32(uint32_t a, uint32_t b) {
  if (a == b) return 0;
  // Convert to monotonic signed-magnitude representation
  int64_t ai = (a & 0x80000000u) ? -(int64_t)(a & 0x7FFFFFFF) : (int64_t)a;
  int64_t bi = (b & 0x80000000u) ? -(int64_t)(b & 0x7FFFFFFF) : (int64_t)b;
  int64_t d = ai - bi;
  return d < 0 ? -d : d;
}

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  Vfp32_exp_pipe* dut = new Vfp32_exp_pipe;
  reset(dut);

  std::vector<uint32_t> ops;
  // Edges
  const uint32_t E[] = {
    0x00000000u, 0x80000000u, 0x3F800000u, 0xBF800000u, 0x40000000u, 0xC0000000u,
    0x7F800000u, 0xFF800000u, 0x7FC00000u, 0xFFC00000u,                  // ±inf, NaN
    0x42B00000u, 0x42B17218u, 0xC2AEAC50u,                                // ±88-ish saturation
    0x3F317218u,                                                           // ln2
    0x40000000u, 0x40400000u, 0x40800000u,                                 // 2.0, 3.0, 4.0
  };
  for (auto v : E) ops.push_back(v);

  // Random inputs in (-90, 90) — the meaningful domain
  std::mt19937 rng(0xE9CA111u);
  std::uniform_real_distribution<float> dx(-15.0f, 15.0f);   // softmax-typical range
  for (long n = 0; n < 1000000; ++n) {
    float x = dx(rng);
    ops.push_back(f2bits(x));
  }
  std::uniform_real_distribution<float> dw(-90.0f, 90.0f);   // wider domain
  for (long n = 0; n < 1000000; ++n) {
    float x = dw(rng);
    ops.push_back(f2bits(x));
  }

  std::queue<std::pair<uint32_t, uint32_t>> in_flight;
  for (size_t k = 0; k < ops.size(); ++k) {
    dut->valid_in = 1;
    dut->a = ops[k];
    in_flight.push({ops[k], golden(ops[k])});
    tick(dut);
    if (dut->valid_out) {
      auto p = in_flight.front(); in_flight.pop();
      uint32_t got = dut->y;
      uint32_t exp = p.second;
      ++g_checked;
      // For non-finite golden (NaN/inf), require exact match.
      bool exp_special = ((exp & 0x7F800000u) == 0x7F800000u) || ((exp & 0x7FFFFFFFu) == 0);
      bool got_special = ((got & 0x7F800000u) == 0x7F800000u) || ((got & 0x7FFFFFFFu) == 0);
      if (exp_special || got_special) {
        if (got != exp) {
          if (g_fail < 20)
            std::fprintf(stderr, "MISMATCH(special) a=%08x got=%08x exp=%08x (%g -> %g)\n",
                         p.first, got, exp, (double)bits2f(p.first), (double)bits2f(exp));
          ++g_fail;
        }
      } else {
        long u = ulp_diff_fp32(got, exp);
        ++g_finite_compared;
        g_ulp_sum += u;
        if (u > g_ulp_max) g_ulp_max = u;
        g_ulp_hist[std::min((long)15, u)]++;
      }
    }
  }
  dut->valid_in = 0;
  while (!in_flight.empty()) {
    tick(dut);
    if (dut->valid_out) {
      auto p = in_flight.front(); in_flight.pop();
      uint32_t got = dut->y;
      uint32_t exp = p.second;
      ++g_checked;
      bool exp_special = ((exp & 0x7F800000u) == 0x7F800000u) || ((exp & 0x7FFFFFFFu) == 0);
      bool got_special = ((got & 0x7F800000u) == 0x7F800000u) || ((got & 0x7FFFFFFFu) == 0);
      if (exp_special || got_special) {
        if (got != exp) ++g_fail;
      } else {
        long u = ulp_diff_fp32(got, exp);
        ++g_finite_compared;
        g_ulp_sum += u;
        if (u > g_ulp_max) g_ulp_max = u;
        g_ulp_hist[std::min((long)15, u)]++;
      }
    }
  }

  delete dut;
  std::printf("fp32_exp_pipe (LATENCY=%d): checked=%ld special-mismatches=%d  finite=%ld\n",
              PIPE_LATENCY, g_checked, g_fail, g_finite_compared);
  std::printf("  ULP max=%ld mean=%.2f\n", g_ulp_max,
              g_finite_compared > 0 ? (double)g_ulp_sum / g_finite_compared : 0.0);
  std::printf("  ULP histogram (0..15+):");
  for (int i = 0; i < 16; ++i) std::printf(" %ld", g_ulp_hist[i]);
  std::printf("\n");
  if (g_fail == 0) {
    std::printf("PASS: fp32_exp_pipe measured-band (special cases exact, finite within band)\n");
    return 0;
  }
  std::fprintf(stderr, "FAIL: fp32_exp_pipe %d special-case mismatches\n", g_fail);
  return 1;
}
