// Bit-exact gate for fp32_sqrt_pipe.

#include "Vfp32_sqrt_pipe.h"
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
static inline uint32_t f2bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

static uint32_t golden(uint32_t bits) {
  float v = bits2f(bits);
  if (std::isnan(v)) return 0x7FC00000u;
  float r = std::sqrt(v);
  if (std::isnan(r)) return 0x7FC00000u;
  return f2bits(r);
}

static void tick(Vfp32_sqrt_pipe* dut) {
  dut->clk = 0; dut->eval();
  dut->clk = 1; dut->eval();
}

static void reset(Vfp32_sqrt_pipe* dut) {
  dut->rst_n = 0; dut->valid_in = 0; dut->a = 0;
  for (int i = 0; i < 4; ++i) tick(dut);
  dut->rst_n = 1;
}

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  Vfp32_sqrt_pipe* dut = new Vfp32_sqrt_pipe;
  reset(dut);

  std::vector<uint32_t> ops;
  const uint32_t E[] = {
    0x00000000u, 0x80000000u, 0x3F800000u, 0xBF800000u, 0x40000000u,
    0x40800000u, 0x41200000u,                              // 4.0, 10.0
    0x00000001u, 0x80000001u, 0x007FFFFFu, 0x00800000u,
    0x3F7FFFFFu, 0x3F800001u, 0x7F7FFFFFu, 0xFF7FFFFFu,
    0x7F800000u, 0xFF800000u, 0x7FC00000u, 0xFFC00000u,
    0x4B000000u, 0x4B000001u, 0x34000000u, 0x749DC5AEu,
    0x0DA24260u, 0x33800000u, 0x33000000u,
    0x40A00000u, 0x40C00000u,                              // 5.0, 6.0 (non-perfect)
    0x42C80000u,                                            // 100.0 -> sqrt = 10.0 exact
    0x49742400u,                                            // 1000000.0
  };
  for (auto v : E) ops.push_back(v);

  std::mt19937 rng(0xD15A4u);
  std::uniform_int_distribution<uint32_t> d32(0, 0xFFFFFFFFu);
  for (long n = 0; n < 3000000; ++n) ops.push_back(d32(rng));
  for (long n = 0; n < 3000000; ++n) {
    uint32_t s = 0;  // positive only (negative -> NaN)
    uint32_t e = 1u + (d32(rng) % 254u);
    uint32_t m = d32(rng) & 0x7FFFFFu;
    ops.push_back((s << 31) | (e << 23) | m);
  }

  std::queue<uint32_t> in_flight;
  for (size_t k = 0; k < ops.size(); ++k) {
    dut->valid_in = 1;
    dut->a = ops[k];
    in_flight.push(golden(ops[k]));
    tick(dut);
    if (dut->valid_out) {
      uint32_t got = dut->y;
      uint32_t e   = in_flight.front(); in_flight.pop();
      ++g_checked;
      if (got != e) {
        if (g_fail < 20)
          std::fprintf(stderr, "MISMATCH op=%zu a=%08x got=%08x exp=%08x  (sqrt %g)\n",
                       k, ops[k], got, e, (double)bits2f(ops[k]));
        ++g_fail;
      }
    }
  }
  dut->valid_in = 0;
  while (!in_flight.empty()) {
    tick(dut);
    if (dut->valid_out) {
      uint32_t got = dut->y;
      uint32_t e   = in_flight.front(); in_flight.pop();
      ++g_checked;
      if (got != e) ++g_fail;
    }
  }

  delete dut;
  std::printf("fp32_sqrt_pipe (LATENCY=%d): checked=%ld  mismatches=%d\n",
              PIPE_LATENCY, g_checked, g_fail);
  if (g_fail == 0 && g_checked == (long)ops.size()) {
    std::printf("PASS: fp32_sqrt_pipe bit-exact vs host float\n");
    return 0;
  }
  std::fprintf(stderr, "FAIL: fp32_sqrt_pipe %d mismatches (checked=%ld)\n",
               g_fail, g_checked);
  return 1;
}
