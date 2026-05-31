// Standalone bit-exact gate for the 4-stage pipelined fp32_sqrt_p4 module.
//
// Golden = the DPI semantics (testbench.h sfu_fp32_sqrt), identical to the
// combinational fp32_sqrt contract:
//   (float)std::sqrt((float)x),  with NaN canonicalized to 0x7FC00000
//   (covers sqrt(neg)->qNaN and -0->-0, +inf->+inf via the host).
// The pipelined sqrt must be BYTE-EXACT to this over directed edge cases +
// millions of randomized inputs. Fully streamed: a new operand is driven every
// cycle and outputs are checked against a golden queue (LATENCY=4, but the
// in-flight deque auto-handles any latency).
// Links ONLY fp32_sqrt_p4 (no taccel_top) — zero risk to the proven cosim.

#include "Vfp32_sqrt_p4.h"
#include "verilated.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <random>
#include <vector>

static int g_fail = 0;
static long g_checked = 0;

static inline float bits2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static inline uint32_t f2bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

static uint32_t golden_sqrt(uint32_t au) {
  float r = std::sqrt(bits2f(au));
  if (std::isnan(r)) return 0x7FC00000u;
  return f2bits(r);
}

static Vfp32_sqrt_p4* g_dut = nullptr;

// in-flight expectations: {expected_y, a} presented LATENCY cycles ago.
struct Exp { uint32_t y, a; const char* tag; };
static std::deque<Exp> g_inflight;

static void tick(uint32_t au, bool drive, const char* tag) {
  g_dut->valid_in = drive ? 1 : 0;
  g_dut->a = au;
  g_dut->clk = 0; g_dut->eval();
  g_dut->clk = 1; g_dut->eval();

  if (drive) g_inflight.push_back({golden_sqrt(au), au, tag});

  if (g_dut->valid_out) {
    if (g_inflight.empty()) {
      if (g_fail < 30) std::fprintf(stderr, "valid_out with empty queue\n");
      ++g_fail;
      return;
    }
    Exp e = g_inflight.front(); g_inflight.pop_front();
    ++g_checked;
    const uint32_t got = g_dut->y;
    if (got != e.y) {
      if (g_fail < 30)
        std::fprintf(stderr,
                     "MISMATCH [%s] a=%08x  got=%08x exp=%08x  (sqrt(%g))\n",
                     e.tag, e.a, got, e.y, (double)bits2f(e.a));
      ++g_fail;
    }
  }
}

static void drain() {
  for (int i = 0; i < 8 && !g_inflight.empty(); ++i)
    tick(0x3f800000u, false, "drain");
}

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  g_dut = new Vfp32_sqrt_p4;

  // reset
  g_dut->rst_n = 0; g_dut->valid_in = 0; g_dut->a = 0;
  for (int i = 0; i < 4; ++i) { g_dut->clk = 0; g_dut->eval(); g_dut->clk = 1; g_dut->eval(); }
  g_dut->rst_n = 1;

  const uint32_t E[] = {
    0x00000000u, 0x80000000u, 0x3F800000u, 0xBF800000u, 0x40000000u,
    0x00000001u, 0x80000001u, 0x007FFFFFu, 0x00800000u, 0x00800001u,
    0x3F7FFFFFu, 0x3F800001u, 0x7F7FFFFFu, 0xFF7FFFFFu, 0x7F800000u,
    0xFF800000u, 0x7FC00000u, 0x7F800001u, 0xFFC00000u, 0x4B000000u,
    0x4B000001u, 0x34000000u, 0x749DC5AEu, 0x0DA24260u, 0x33800000u,
    0x33000000u, 0x00000002u, 0x007FFFFEu, 0x3EAAAAABu, 0x42F60000u,
    0x40800000u, 0x41200000u, 0x447A0000u, 0x3DCCCCCDu, 0x4B7FFFFFu,
  };
  const int NE = sizeof(E) / sizeof(E[0]);
  for (int i = 0; i < NE; ++i)
    tick(E[i], true, "edge");

  // perfect squares: sqrt should be exact, exercises sticky==0 path
  for (uint32_t k = 1; k < 4096; ++k) {
    float v = (float)(k * k);
    tick(f2bits(v), true, "perfect_sq");
  }

  std::mt19937 rng(0x5417C0DEu);
  std::uniform_int_distribution<uint32_t> d32(0, 0xFFFFFFFFu);
  for (long n = 0; n < 4000000; ++n)
    tick(d32(rng), true, "rand_full");

  for (long n = 0; n < 4000000; ++n) {
    auto mk = [&]() -> uint32_t {
      uint32_t s = d32(rng) & 1u;
      uint32_t e = 1u + (d32(rng) % 254u);
      uint32_t m = d32(rng) & 0x7FFFFFu;
      return (s << 31) | (e << 23) | m;
    };
    tick(mk(), true, "rand_normal");
  }

  // subnormal-heavy: tiny inputs exercise the leading-1 normalize path
  for (long n = 0; n < 2000000; ++n) {
    auto mk = [&]() -> uint32_t {
      uint32_t s = d32(rng) & 1u;
      uint32_t e = d32(rng) % 8u;                 // 0..7 -> subnormal / tiny
      uint32_t m = d32(rng) & 0x7FFFFFu;
      return (s << 31) | (e << 23) | m;
    };
    tick(mk(), true, "rand_sub");
  }

  drain();

  delete g_dut;
  std::printf("fp32_sqrt_p4: checked=%ld  mismatches=%d\n", g_checked, g_fail);
  if (g_fail == 0) { std::printf("PASS: fp32_sqrt_p4 bit-exact vs host float\n"); return 0; }
  std::fprintf(stderr, "FAIL: fp32_sqrt_p4 %d mismatches\n", g_fail);
  return 1;
}
