// Bit-exact gate for i32_to_fp32_pipe — golden = (float)int32 (host RNE cast).
// Edge cases (INT32_MIN/MAX, ±powers of 2, ±2^23, ±(2^23+1) round-half-even
// boundaries) + millions random + signed-distribution randoms.

#include "Vi32_to_fp32_pipe.h"
#include "verilated.h"

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

static inline uint32_t f2bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

static uint32_t golden(int32_t a) { return f2bits((float)a); }

static void tick(Vi32_to_fp32_pipe* dut) {
  dut->clk = 0; dut->eval();
  dut->clk = 1; dut->eval();
}

static void reset(Vi32_to_fp32_pipe* dut) {
  dut->rst_n = 0; dut->valid_in = 0; dut->a = 0;
  for (int i = 0; i < 4; ++i) tick(dut);
  dut->rst_n = 1;
}

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  Vi32_to_fp32_pipe* dut = new Vi32_to_fp32_pipe;
  reset(dut);

  std::vector<int32_t> ops;

  // Directed edges
  ops.push_back(0);
  ops.push_back(1);
  ops.push_back(-1);
  ops.push_back(INT32_MAX);
  ops.push_back(INT32_MIN);
  ops.push_back(INT32_MIN + 1);
  // Powers of 2 around the 24-bit mantissa boundary (RNE tie territory).
  for (int k = 0; k < 32; ++k) {
    ops.push_back((int32_t)((uint32_t)1 << k));
    if (k < 31) {
      ops.push_back((int32_t)(((uint32_t)1 << k) | 1u));        // 2^k + 1
      ops.push_back(-(int32_t)((uint32_t)1 << k));
      ops.push_back(-(int32_t)(((uint32_t)1 << k) | 1u));
    }
  }
  // RNE half-to-even ties at 2^24 boundary
  ops.push_back(0x01000000);  // 2^24 (exact)
  ops.push_back(0x01000001);  // 2^24+1 (round-half-even to 2^24)
  ops.push_back(0x01000002);  // 2^24+2 (exact)
  ops.push_back(0x01000003);  // 2^24+3 (round to 2^24+4)
  ops.push_back(0x02000001);  // 2^25+1 -> tie territory
  ops.push_back(0x02000003);
  ops.push_back(0x02000005);

  std::mt19937 rng(0x132F32u);
  std::uniform_int_distribution<uint32_t> d32(0, 0xFFFFFFFFu);
  for (long n = 0; n < 4000000; ++n) {
    ops.push_back((int32_t)d32(rng));
  }

  std::queue<uint32_t> in_flight;
  for (size_t k = 0; k < ops.size(); ++k) {
    dut->valid_in = 1;
    dut->a = (uint32_t)ops[k];
    uint32_t exp = golden(ops[k]);
    in_flight.push(exp);
    tick(dut);
    if (dut->valid_out) {
      uint32_t got = dut->y;
      uint32_t e   = in_flight.front(); in_flight.pop();
      ++g_checked;
      if (got != e) {
        if (g_fail < 20)
          std::fprintf(stderr,
                       "MISMATCH op=%zu a=%d (0x%08x) got=%08x exp=%08x\n",
                       k, ops[k], (uint32_t)ops[k], got, e);
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
  std::printf("i32_to_fp32_pipe (LATENCY=%d): checked=%ld  mismatches=%d\n",
              PIPE_LATENCY, g_checked, g_fail);
  if (g_fail == 0 && g_checked == (long)ops.size()) {
    std::printf("PASS: i32_to_fp32_pipe bit-exact vs host (float)int32\n");
    return 0;
  }
  std::fprintf(stderr, "FAIL: i32_to_fp32_pipe %d mismatches (checked=%ld)\n",
               g_fail, g_checked);
  return 1;
}
