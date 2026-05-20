// Bit-exact gate for fp32_div_pipe — golden = sfu_fp32_div semantics
// (float)((float)a / (float)b) with NaN canonicalized to 0x7FC00000.

#include "Vfp32_div_pipe.h"
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

static uint32_t golden(uint32_t au, uint32_t bu) {
  float r = bits2f(au) / bits2f(bu);
  if (std::isnan(r)) return 0x7FC00000u;
  return f2bits(r);
}

static void tick(Vfp32_div_pipe* dut) {
  dut->clk = 0; dut->eval();
  dut->clk = 1; dut->eval();
}

static void reset(Vfp32_div_pipe* dut) {
  dut->rst_n = 0; dut->valid_in = 0; dut->a = 0; dut->b = 0;
  for (int i = 0; i < 4; ++i) tick(dut);
  dut->rst_n = 1;
}

struct Op { uint32_t a, b, exp; };

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  Vfp32_div_pipe* dut = new Vfp32_div_pipe;
  reset(dut);

  const uint32_t E[] = {
    0x00000000u, 0x80000000u,
    0x3F800000u, 0xBF800000u, 0x40000000u, 0xC0000000u,
    0x00000001u, 0x80000001u, 0x007FFFFFu, 0x00800000u, 0x3F7FFFFFu,
    0x3F800001u, 0x7F7FFFFFu, 0xFF7FFFFFu, 0x7F800000u, 0xFF800000u,
    0x7FC00000u, 0x7F800001u, 0xFFC00000u, 0x4B000000u, 0x4B000001u,
    0x34000000u, 0x749DC5AEu, 0x0DA24260u, 0x33800000u, 0x33000000u,
  };
  const int NE = sizeof(E) / sizeof(E[0]);

  std::vector<Op> ops;
  ops.reserve(NE * NE + 6000000);

  for (int i = 0; i < NE; ++i)
    for (int j = 0; j < NE; ++j)
      ops.push_back({E[i], E[j], golden(E[i], E[j])});

  std::mt19937 rng(0xD1F32u);
  std::uniform_int_distribution<uint32_t> d32(0, 0xFFFFFFFFu);
  for (long n = 0; n < 3000000; ++n) {
    uint32_t a = d32(rng), b = d32(rng);
    ops.push_back({a, b, golden(a, b)});
  }
  for (long n = 0; n < 3000000; ++n) {
    auto mk = [&]() -> uint32_t {
      uint32_t s = d32(rng) & 1u;
      uint32_t e = 1u + (d32(rng) % 254u);
      uint32_t m = d32(rng) & 0x7FFFFFu;
      return (s << 31) | (e << 23) | m;
    };
    uint32_t a = mk(), b = mk();
    ops.push_back({a, b, golden(a, b)});
  }

  std::queue<uint32_t> in_flight;
  for (size_t k = 0; k < ops.size(); ++k) {
    dut->valid_in = 1;
    dut->a = ops[k].a;
    dut->b = ops[k].b;
    in_flight.push(ops[k].exp);
    tick(dut);
    if (dut->valid_out) {
      uint32_t got = dut->y;
      uint32_t e   = in_flight.front(); in_flight.pop();
      ++g_checked;
      if (got != e) {
        if (g_fail < 20)
          std::fprintf(stderr,
                       "MISMATCH op=%zu a=%08x b=%08x got=%08x exp=%08x  (%g / %g)\n",
                       k, ops[k].a, ops[k].b, got, e,
                       (double)bits2f(ops[k].a), (double)bits2f(ops[k].b));
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
  std::printf("fp32_div_pipe (LATENCY=%d): checked=%ld  mismatches=%d\n",
              PIPE_LATENCY, g_checked, g_fail);
  if (g_fail == 0 && g_checked == (long)ops.size()) {
    std::printf("PASS: fp32_div_pipe bit-exact vs host float\n");
    return 0;
  }
  std::fprintf(stderr, "FAIL: fp32_div_pipe %d mismatches (checked=%ld)\n",
               g_fail, g_checked);
  return 1;
}
