// Standalone bit-exact gate for the 33-stage pipelined fp32_gelu_p33.
//
// Golden = the COMBINATIONAL fp32_gelu_new RTL (not a host-float GELU):
// fp32_gelu_new is a measured-band approximation (freeze §7), and this module
// is a PURE RETIMING of it, so the pipeline must be BYTE-IDENTICAL to the
// combinational core over directed edge cases + millions of randomized inputs.
// The harness (fp32_gelu_p33_tb.sv) drives one `a` per cycle into both; y_pipe
// (valid LATENCY=33 later) is compared to the y_comb that was captured when
// that `a` was presented. Timestamped expectations check valid_out every cycle,
// including bubbles and a bounded empty-pipeline drain, not merely FIFO order.
//
// BUBBLE COVERAGE is not optional here. fp32_gelu_p33 carries a 32-deep `x`
// delay line whose taps (1, 3, 32) must stay aligned with the arithmetic under
// arbitrary stall patterns — that alignment is the module's core invariant, and
// a wrong tap or a stray clock-enable shows up ONLY when valid_in has gaps in
// it. The bubble phase drives fresh random garbage on `a` while valid_in is
// low, so any leakage of an unfed operand into a live result is caught.
//
// Links ONLY the fp32 primitives (add/mul/exp/exp_p18/div/div_p6/gelu_new/
// gelu_p33) — no taccel_top.
//
// Usage: ./Vfp32_gelu_p33_tb [--quick]   (--quick shrinks the random phases
// ~20x for a fast smoke; the committed gate runs the full stream.)

#include "Vfp32_gelu_p33_tb.h"
#include "verilated.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <random>
#include <vector>

static int  g_fail = 0;
static long g_checked = 0;
static long g_tail_checked = 0;
static long g_finite_checked = 0;
static long g_nan_checked = 0;
static long g_inputs = 0;
static long g_valid_checked = 0;
static constexpr uint64_t LATENCY = 33;
static uint64_t g_cycle = 0;  // Number of completed rising edges, including reset.

static inline float    bits2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static inline uint32_t f2bits(float f)    { uint32_t u; std::memcpy(&u, &f, 4); return u; }

static Vfp32_gelu_p33_tb* g_dut = nullptr;

struct Exp { uint32_t a; uint32_t y_comb; const char* tag; uint64_t due_cycle; };
static std::deque<Exp> g_inflight;

static void tick() {
  g_dut->clk = 0; g_dut->eval();
  g_dut->clk = 1; g_dut->eval();
  ++g_cycle;
}

static void collect(const char* where) {
  const bool expected_valid = !g_inflight.empty() &&
                              g_inflight.front().due_cycle == g_cycle;
  const bool actual_valid = g_dut->valid_out != 0;
  ++g_valid_checked;
  if (actual_valid != expected_valid) {
    if (g_fail < 40)
      std::printf("VALID MISMATCH%s cycle=%llu expected=%d actual=%d\n",
                  where, static_cast<unsigned long long>(g_cycle),
                  int(expected_valid), int(actual_valid));
    ++g_fail;
  }
  // Retire by the specified cycle even when the DUT misses its valid pulse.
  // An extra pulse must not consume a future expectation or read an empty FIFO.
  if (!expected_valid) return;
  Exp e = g_inflight.front();
  g_inflight.pop_front();
  if (!actual_valid) return;
  uint32_t got = g_dut->y_pipe;
  g_checked++;
  if (got != e.y_comb) {
    if (g_fail < 40) {
      std::printf("MISMATCH%s [%s] a=%08x (%.9g)  pipe=%08x (%.9g)  comb=%08x (%.9g)\n",
                  where, e.tag, e.a, bits2f(e.a), got, bits2f(got),
                  e.y_comb, bits2f(e.y_comb));
    }
    g_fail++;
  }
  // Independent mathematical oracle, not RTL-vs-RTL agreement: for x>=10,
  // tanh's positive tail rounds to1 in FP32, so GELU(x) is exactly x. This
  // catches the prior exp-overflow Inf/Inf bug even if BOTH RTL variants agree.
  if (e.a >= 0x41200000u && e.a <= 0x7f800000u) {
    ++g_tail_checked;
    if (got != e.a || e.y_comb != e.a) {
      if (g_fail < 40)
        std::printf("TAIL MISMATCH [%s] a=%08x pipe=%08x comb=%08x expected=x\n",
                    e.tag, e.a, got, e.y_comb);
      ++g_fail;
    }
  }
  if ((e.a & 0x7f800000u) != 0x7f800000u) {
    ++g_finite_checked;
    if ((got & 0x7f800000u) == 0x7f800000u ||
        (e.y_comb & 0x7f800000u) == 0x7f800000u) {
      if (g_fail < 40)
        std::printf("FINITE MISMATCH [%s] a=%08x pipe=%08x comb=%08x\n",
                    e.tag, e.a, got, e.y_comb);
      ++g_fail;
    }
  }
  // Independent special-value oracle: neither implementation may silently
  // turn a NaN input into a finite value or infinity. Bit parity above still
  // checks their existing payload/sign behavior without inventing a new policy.
  const auto is_nan = [](uint32_t u) {
    return (u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu) != 0;
  };
  if (is_nan(e.a)) {
    ++g_nan_checked;
    if (!is_nan(got) || !is_nan(e.y_comb)) {
      if (g_fail < 40)
        std::printf("NAN MISMATCH [%s] a=%08x pipe=%08x comb=%08x\n",
                    e.tag, e.a, got, e.y_comb);
      ++g_fail;
    }
  }
}

// Drive one input, then advance one cycle; check any output that becomes valid.
static void step(uint32_t a, const char* tag) {
  g_dut->valid_in = 1;
  g_dut->a = a;
  g_dut->eval();  // settle combinational y_comb for THIS a
  // With C edges already completed, this input enters stage0 on edge C+1
  // and exits stage32 on edge C+33: 33 captures, not 33 *additional* edges.
  g_inflight.push_back({a, g_dut->y_comb, tag, g_cycle + LATENCY});
  ++g_inputs;
  tick();
  collect("");
}

// Drive invalid input with unrelated garbage. Older valid inputs can still
// produce outputs now; this bubble must appear at the output 33 captures later.
static void bubble(uint32_t garbage) {
  g_dut->valid_in = 0;
  g_dut->a = garbage;
  g_dut->eval();
  tick();
  collect("(bubble)");
}

// A bounded drain catches missing outputs without hanging and checks an extra
// full latency of invalid output after all pending results must have emerged.
static void drain() {
  for (uint64_t i = 0; i < 2 * LATENCY; ++i) bubble(0x7fc00000u);
  if (!g_inflight.empty()) {
    std::printf("DRAIN MISMATCH: %zu expected results remain\n", g_inflight.size());
    ++g_fail;
  }
  if (g_checked != g_inputs) {
    std::printf("COUNT MISMATCH: presented=%ld checked=%ld\n", g_inputs, g_checked);
    ++g_fail;
  }
}

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  bool quick = false;
  for (int i = 1; i < argc; ++i)
    if (std::strcmp(argv[i], "--quick") == 0) quick = true;
  const long DIV = quick ? 20 : 1;

  g_dut = new Vfp32_gelu_p33_tb;

  // Reset.
  g_dut->rst_n = 0; g_dut->valid_in = 0; g_dut->a = 0; g_dut->clk = 0;
  g_dut->eval();
  for (int i = 0; i < 4; ++i) tick();
  g_dut->rst_n = 1; g_dut->eval();

  // --- Directed edge cases -------------------------------------------------
  // Include the retained GPT-2 failure and neighbors. Large positive finite
  // activations are legal and occur in the trained model; they must saturate
  // the ratio to1, never generate NaN from an overflowing exponential.
  std::vector<uint32_t> directed = {
    0x00000000u,                 // +0
    0x80000000u,                 // -0
    0x3F800000u,                 // 1.0
    0xBF800000u,                 // -1.0
    0x40000000u,                 // 2.0
    0xC0000000u,                 // -2.0
    0x40A00000u,                 // 5.0
    0xC0A00000u,                 // -5.0
    0x41200000u,                 // 10.0  (just under the exp-overflow knee)
    0xC1200000u,                 // -10.0
    0x412AAAABu,                 // ~10.67 (past the exp-overflow knee)
    0x412DE000u,                 // 10.8671875, captured PC3994/column588
    0x41400000u,                 // 12.0  (past it: exp->inf)
    0xC1400000u,                 // -12.0
    0x42000000u,                 // 32.0
    0xC2000000u,                 // -32.0
    0x477FE000u,                 // 65504 = fp16 max
    0xC77FE000u,                 // -65504
    0x7F800000u,                 // +inf
    0xFF800000u,                 // -inf
    0x7FC00000u,                 // qNaN
    0x7FA00000u,                 // sNaN
    0x3DCCCCCDu,                 // 0.1
    0xBDCCCCCDu,                 // -0.1
    0x38D1B717u,                 // 1e-4
    0xB8D1B717u,                 // -1e-4
    0x00000001u,                 // smallest subnormal
    0x80000001u,                 // -smallest subnormal
    0x007FFFFFu,                 // largest subnormal
    0x33800000u,                 // ~5.96e-8
    0x40490FDBu,                 // pi
    0xC0490FDBu,                 // -pi
  };
  for (uint32_t a : directed) step(a, "directed");

  // Exhaust every FP16 value in the positive saturation tail (and its negative
  // partner), including bubble transitions across the former overflow knee.
  for (uint32_t h = 0x4900u; h <= 0x7bffu; ++h) {
    const uint32_t a = (((h >> 10) + 112u) << 23) | ((h & 0x3ffu) << 13);
    step(a, "fp16-tail+");
    if ((h & 31u) == 0) bubble(0x7fc00000u);
    step(a | 0x80000000u, "fp16-tail-");
  }

  std::mt19937 rng(0x6E1Bu);

  // --- Randomized: the live activation band (fp16 GELU inputs) -------------
  {
    std::uniform_real_distribution<float> ur(-30.0f, 30.0f);
    for (long i = 0; i < 3000000 / DIV; ++i) step(f2bits(ur(rng)), "rand-band");
  }
  // --- Randomized: dense around the exp-overflow knee and around 0 ---------
  {
    std::uniform_real_distribution<float> uk(9.5f, 11.5f);
    for (long i = 0; i < 500000 / DIV; ++i) {
      float x = uk(rng);
      step(f2bits(x),  "knee+");
      step(f2bits(-x), "knee-");
    }
    std::uniform_real_distribution<float> uz(-1.0f, 1.0f);
    for (long i = 0; i < 1000000 / DIV; ++i) step(f2bits(uz(rng)), "rand-near0");
  }
  // --- Randomized: full 32-bit bit patterns (specials, huge, subnormal) ----
  {
    std::uniform_int_distribution<uint32_t> ubits(0, 0xFFFFFFFFu);
    for (long i = 0; i < 3000000 / DIV; ++i) step(ubits(rng), "rand-bits");
  }
  // --- Randomized WITH BUBBLES: the delay-line alignment gate --------------
  // Mixed valid/invalid with garbage on `a` during the gaps, including runs of
  // gaps longer than LATENCY so the pipe fully empties and refills.
  {
    std::uniform_real_distribution<float> ur(-30.0f, 30.0f);
    std::uniform_int_distribution<uint32_t> ubits(0, 0xFFFFFFFFu);
    std::uniform_int_distribution<int> ugap(0, 40);
    for (long i = 0; i < 500000 / DIV; ++i) {
      step(f2bits(ur(rng)), "bubbled");
      int gap = ugap(rng);
      for (int g = 0; g < gap; ++g) bubble(ubits(rng));
    }
  }

  drain();

  std::printf("fp32_gelu_p33 bit-exact gate: checked=%ld  fails=%d\n",
              g_checked, g_fail);
  std::printf("independent positive-tail checks=%ld; finite-input checks=%ld\n",
              g_tail_checked, g_finite_checked);
  std::printf("exact 33-stage valid checks=%ld; independent NaN-input checks=%ld\n",
              g_valid_checked, g_nan_checked);
  delete g_dut;
  if (g_fail) { std::printf("FAIL\n"); return 1; }
  std::printf("PASS (byte-identical to combinational fp32_gelu_new)\n");
  return 0;
}
