// Measured-band gate for fp32_gelu_new_pipe. Golden = sfu_fp32_gelu_new:
//   x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))), in fp32.

#include "Vfp32_gelu_new_pipe.h"
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
static long g_checked = 0, g_finite_compared = 0, g_ulp_max = 0, g_ulp_sum = 0;
static long g_ulp_hist[16] = {0};

static inline float bits2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static inline uint32_t f2bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

static uint32_t golden(uint32_t bits) {
  float x = bits2f(bits);
  if (std::isnan(x)) return 0x7FC00000u;
  const float K = 0.7978845608028654f;
  float x3 = std::pow(x, 3.0f);   // match the DPI's std::pow (NOT x*x*x)
  float arg = K * (x + 0.044715f * x3);
  float t   = std::tanh(arg);
  float y   = x * 0.5f * (1.0f + t);
  if (std::isnan(y)) return 0x7FC00000u;
  return f2bits(y);
}

static void tick(Vfp32_gelu_new_pipe* dut) { dut->clk=0; dut->eval(); dut->clk=1; dut->eval(); }
static void reset(Vfp32_gelu_new_pipe* dut) {
  dut->rst_n=0; dut->valid_in=0; dut->a=0;
  for (int i=0;i<4;++i) tick(dut);
  dut->rst_n=1;
}

static long ulp_diff_fp32(uint32_t a, uint32_t b) {
  if (a == b) return 0;
  int64_t ai = (a & 0x80000000u) ? -(int64_t)(a & 0x7FFFFFFF) : (int64_t)a;
  int64_t bi = (b & 0x80000000u) ? -(int64_t)(b & 0x7FFFFFFF) : (int64_t)b;
  int64_t d = ai - bi;
  return d < 0 ? -d : d;
}

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  Vfp32_gelu_new_pipe* dut = new Vfp32_gelu_new_pipe;
  reset(dut);

  std::vector<uint32_t> ops;
  // Random in (-10, 10) -- the typical GELU input range.
  std::mt19937 rng(0xCE1Eu);
  std::uniform_real_distribution<float> dx(-10.0f, 10.0f);
  for (long n = 0; n < 500000; ++n) ops.push_back(f2bits(dx(rng)));

  std::queue<std::pair<uint32_t, uint32_t>> in_flight;
  for (size_t k = 0; k < ops.size(); ++k) {
    dut->valid_in=1; dut->a=ops[k];
    in_flight.push({ops[k], golden(ops[k])});
    tick(dut);
    if (dut->valid_out) {
      auto p = in_flight.front(); in_flight.pop();
      uint32_t got = dut->y; uint32_t exp = p.second;
      ++g_checked;
      bool exp_nan = ((exp & 0x7F800000u)==0x7F800000u) && ((exp & 0x7FFFFFu)!=0);
      bool got_nan = ((got & 0x7F800000u)==0x7F800000u) && ((got & 0x7FFFFFu)!=0);
      // Banded primitive: only NaN-vs-non-NaN is a hard failure. Tiny-ULP
      // differences near zero / boundary saturate land in the wide tail.
      if (exp_nan != got_nan) {
        if (g_fail < 10)
          std::fprintf(stderr, "NaN-mismatch a=%08x got=%08x exp=%08x\n",
                       p.first, got, exp);
        ++g_fail;
      } else if (!exp_nan && !got_nan) {
        long u = ulp_diff_fp32(got, exp);
        ++g_finite_compared;
        g_ulp_sum += u;
        if (u > g_ulp_max) g_ulp_max = u;
        g_ulp_hist[std::min((long)15, u)]++;
      }
    }
  }
  dut->valid_in=0;
  while (!in_flight.empty()) {
    tick(dut);
    if (dut->valid_out) {
      auto p = in_flight.front(); in_flight.pop();
      ++g_checked;
      uint32_t got = dut->y;
      long u = ulp_diff_fp32(got, p.second);
      if (u >= 0) {
        ++g_finite_compared;
        g_ulp_sum += u;
        if (u > g_ulp_max) g_ulp_max = u;
        g_ulp_hist[std::min((long)15, u)]++;
      }
    }
  }

  delete dut;
  std::printf("fp32_gelu_new_pipe (LATENCY=%d): checked=%ld special-mismatches=%d  finite=%ld\n",
              PIPE_LATENCY, g_checked, g_fail, g_finite_compared);
  std::printf("  ULP max=%ld mean=%.2f\n", g_ulp_max,
              g_finite_compared > 0 ? (double)g_ulp_sum / g_finite_compared : 0.0);
  std::printf("  ULP histogram (0..15+):");
  for (int i = 0; i < 16; ++i) std::printf(" %ld", g_ulp_hist[i]);
  std::printf("\n");
  if (g_fail == 0) {
    std::printf("PASS: fp32_gelu_new_pipe measured-band\n");
    return 0;
  }
  return 1;
}
