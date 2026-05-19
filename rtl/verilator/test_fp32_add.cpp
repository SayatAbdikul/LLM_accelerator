// Standalone bit-exact gate for the synthesizable fp32_add module.
//
// Golden = the DPI semantics (testbench.h sfu_fp32_add):
//   (float)( (float)a + (float)b ),  with NaN canonicalized to 0x7FC00000.
// The module must be BYTE-EXACT to this over directed edge cases + millions
// of randomized pairs. This is the reproducer/gate-first discipline applied
// to the synthesizable SFU primitive library: the primitive is proven in
// isolation before any sfu_engine.sv integration (zero risk to the proven
// byte-exact cosim — this links only fp32_add, no taccel_top).

#include "Vfp32_add.h"
#include "verilated.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int g_fail = 0;
static int g_checked = 0;

static inline float bits2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static inline uint32_t f2bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

// DPI sfu_fp32_add semantics + NaN canonicalization (module contract).
static uint32_t golden_add(uint32_t au, uint32_t bu) {
  float r = bits2f(au) + bits2f(bu);
  if (std::isnan(r)) return 0x7FC00000u;
  return f2bits(r);
}

static void check(Vfp32_add* dut, uint32_t au, uint32_t bu, const char* tag) {
  dut->a = au;
  dut->b = bu;
  dut->eval();
  const uint32_t got = dut->y;
  const uint32_t exp = golden_add(au, bu);
  ++g_checked;
  if (got != exp) {
    if (g_fail < 30)
      std::fprintf(stderr,
                   "MISMATCH [%s] a=%08x b=%08x  got=%08x exp=%08x  "
                   "(%g + %g)\n",
                   tag, au, bu, got, exp,
                   (double)bits2f(au), (double)bits2f(bu));
    ++g_fail;
  }
}

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  Vfp32_add* dut = new Vfp32_add;

  // --- directed edge cases ---
  const uint32_t E[] = {
    0x00000000u,  // +0
    0x80000000u,  // -0
    0x3F800000u,  // +1.0
    0xBF800000u,  // -1.0
    0x40000000u,  // +2.0
    0x00000001u,  // smallest +subnormal
    0x80000001u,  // smallest -subnormal
    0x007FFFFFu,  // largest +subnormal
    0x00800000u,  // smallest +normal
    0x00800001u,
    0x3F7FFFFFu,  // 1 - eps
    0x3F800001u,  // 1 + eps
    0x7F7FFFFFu,  // FLT_MAX
    0xFF7FFFFFu,  // -FLT_MAX
    0x7F800000u,  // +inf
    0xFF800000u,  // -inf
    0x7FC00000u,  // qNaN
    0x7F800001u,  // sNaN
    0xFFC00000u,  // -qNaN
    0x4B000000u,  // 2^23 (RNE tie territory)
    0x4B000001u,
    0x34000000u,  // ~1.19e-7 (small vs 1.0 -> guard/round/sticky)
    0x749DC5AEu,  // ~1e32
    0x0DA24260u,  // ~1e-30
    0x33800000u,  // 2^-24 (exact tie when added to 1.0)
    0x33000000u,  // 2^-25
  };
  const int NE = sizeof(E) / sizeof(E[0]);
  for (int i = 0; i < NE; ++i)
    for (int j = 0; j < NE; ++j)
      check(dut, E[i], E[j], "edge");

  // x + (-x) exact cancellation -> +0 (RNE), all magnitudes.
  for (int i = 0; i < NE; ++i)
    check(dut, E[i], E[i] ^ 0x80000000u, "cancel");

  // --- randomized: full 32-bit space (incl. NaN/inf/subnormal) ---
  std::mt19937 rng(0xF32ADDu);
  std::uniform_int_distribution<uint32_t> d32(0, 0xFFFFFFFFu);
  for (long n = 0; n < 3000000; ++n)
    check(dut, d32(rng), d32(rng), "rand_full");

  // --- randomized: biased to finite normals (the SFU common path) ---
  for (long n = 0; n < 3000000; ++n) {
    // exponents in [1,254], random sign+mantissa -> always finite normal.
    auto mk = [&]() -> uint32_t {
      uint32_t s = d32(rng) & 1u;
      uint32_t e = 1u + (d32(rng) % 254u);
      uint32_t m = d32(rng) & 0x7FFFFFu;
      return (s << 31) | (e << 23) | m;
    };
    check(dut, mk(), mk(), "rand_normal");
  }

  delete dut;
  std::printf("fp32_add: checked=%d  mismatches=%d\n", g_checked, g_fail);
  if (g_fail == 0) { std::printf("PASS: fp32_add bit-exact vs host float\n"); return 0; }
  std::fprintf(stderr, "FAIL: fp32_add %d mismatches\n", g_fail);
  return 1;
}
