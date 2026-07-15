// Standalone bit-exact gate for the 18-stage pipelined fp32_exp_p18.
//
// Golden = the COMBINATIONAL fp32_exp RTL (not std::exp): fp32_exp is a banded
// approximation, and this module is a PURE RETIMING of it, so the pipeline must
// be BYTE-IDENTICAL to the combinational core over directed edge cases + millions
// of randomized inputs. The harness (fp32_exp_p18_tb.sv) drives one `a` per cycle
// into both; y_pipe (valid LATENCY=18 later) is compared to the y_comb that was
// captured when that `a` was presented (an in-flight deque handles the latency).
//
// Links ONLY the fp32 primitives (fp32_add/mul/exp/exp_p18) — no taccel_top.

#include "Vfp32_exp_p18_tb.h"
#include "verilated.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <random>
#include <vector>

static int  g_fail = 0;
static long g_checked = 0;

static inline float    bits2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static inline uint32_t f2bits(float f)     { uint32_t u; std::memcpy(&u, &f, 4); return u; }

static Vfp32_exp_p18_tb* g_dut = nullptr;

struct Exp { uint32_t a; uint32_t y_comb; const char* tag; };
static std::deque<Exp> g_inflight;

static void tick() {
  g_dut->clk = 0; g_dut->eval();
  g_dut->clk = 1; g_dut->eval();
}

// Drive one input, then advance one cycle; check any output that becomes valid.
static void step(uint32_t a, const char* tag) {
  g_dut->valid_in = 1;
  g_dut->a = a;
  g_dut->eval();  // settle combinational y_comb for THIS a
  uint32_t y_comb_now = g_dut->y_comb;
  g_inflight.push_back({a, y_comb_now, tag});

  tick();

  if (g_dut->valid_out) {
    Exp e = g_inflight.front();
    g_inflight.pop_front();
    uint32_t got = g_dut->y_pipe;
    g_checked++;
    if (got != e.y_comb) {
      if (g_fail < 40) {
        std::printf("MISMATCH [%s] a=%08x  pipe=%08x (%.9g)  comb=%08x (%.9g)\n",
                    e.tag, e.a, got, bits2f(got), e.y_comb, bits2f(e.y_comb));
      }
      g_fail++;
    }
  }
}

// Drain the remaining in-flight results after the input stream ends.
static void drain() {
  while (!g_inflight.empty()) {
    g_dut->valid_in = 0;
    g_dut->eval();
    tick();
    if (g_dut->valid_out) {
      Exp e = g_inflight.front();
      g_inflight.pop_front();
      uint32_t got = g_dut->y_pipe;
      g_checked++;
      if (got != e.y_comb) {
        if (g_fail < 40)
          std::printf("MISMATCH(drain) [%s] a=%08x  pipe=%08x  comb=%08x\n",
                      e.tag, e.a, got, e.y_comb);
        g_fail++;
      }
    }
  }
}

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  g_dut = new Vfp32_exp_p18_tb;

  // Reset.
  g_dut->rst_n = 0; g_dut->valid_in = 0; g_dut->a = 0; g_dut->clk = 0;
  g_dut->eval();
  for (int i = 0; i < 4; ++i) tick();
  g_dut->rst_n = 1; g_dut->eval();

  // --- Directed edge cases (exp saturations, specials, boundaries) ---
  std::vector<uint32_t> directed = {
    0x00000000u,                 // +0 -> 1
    0x80000000u,                 // -0 -> 1
    0x3F800000u,                 // 1.0
    0xBF800000u,                 // -1.0
    0x40000000u,                 // 2.0
    0xC0000000u,                 // -2.0
    0x7F800000u,                 // +inf -> +inf
    0xFF800000u,                 // -inf -> +0
    0x7FC00000u,                 // qNaN -> qNaN
    0x7FA00000u,                 // sNaN -> qNaN
    0x42B17218u,                 // ~88.72 overflow boundary
    0x42B17219u,                 // just past overflow -> +inf
    0x42CFF1B5u,                 // underflow-to-subnormal boundary
    0xC2B17218u,                 // -88.72
    0xC2CFF1B5u,                 // -103.97 underflow
    0xC3000000u,                 // -128
    0x43000000u,                 //  128 -> +inf
    0x3DCCCCCDu,                 // 0.1
    0xBDCCCCCDu,                 // -0.1
    0x40490FDBu,                 // pi
    0xC0490FDBu,                 // -pi
    0x00000001u,                 // smallest subnormal
    0x007FFFFFu,                 // largest subnormal
    0x33800000u,                 // ~5.96e-8 (tiny positive)
  };
  for (uint32_t a : directed) step(a, "directed");

  // --- Randomized: uniform over the interesting exp input range [-104, 104] ---
  std::mt19937 rng(0xC0FFEEu);
  std::uniform_real_distribution<float> ur(-104.0f, 104.0f);
  for (long i = 0; i < 4000000; ++i) {
    float x = ur(rng);
    step(f2bits(x), "rand-range");
  }
  // --- Randomized: full 32-bit bit patterns (specials, huge, denormal) ---
  std::uniform_int_distribution<uint32_t> ubits(0, 0xFFFFFFFFu);
  for (long i = 0; i < 4000000; ++i) {
    step(ubits(rng), "rand-bits");
  }
  // --- Randomized: near the saturation boundaries (dense) ---
  std::uniform_real_distribution<float> ubound(87.0f, 90.0f);
  for (long i = 0; i < 500000; ++i) {
    float x = ubound(rng);
    step(f2bits(x),  "bound+");
    step(f2bits(-x), "bound-");
  }

  drain();

  std::printf("fp32_exp_p18 bit-exact gate: checked=%ld  fails=%d\n", g_checked, g_fail);
  delete g_dut;
  if (g_fail) { std::printf("FAIL\n"); return 1; }
  std::printf("PASS (byte-identical to combinational fp32_exp)\n");
  return 0;
}
