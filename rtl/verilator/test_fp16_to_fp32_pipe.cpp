// Exhaustive bit-exact gate for fp16_to_fp32_pipe. fp16 has only 65536 bit
// patterns — every one is verified vs the canonical conversion semantics
// matching the DPI golden `sfu_fp16_bits_to_fp32` (testbench.h).
//
// fp16 -> fp32 is EXACT (every fp16 value has an exact fp32 representation),
// so the contract is 0-ULP byte-match on all 65536 inputs; NaN canonicalized
// to qNaN 0x7FC00000 (matches the SFU Option-B non-finite philosophy).

#include "Vfp16_to_fp32_pipe.h"
#include "verilated.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <queue>

#ifndef PIPE_LATENCY
#define PIPE_LATENCY 1
#endif

static int g_fail = 0;
static long g_checked = 0;

static uint32_t fp16_bits_to_fp32_ref(uint16_t h) {
  uint32_t s = (h >> 15) & 1u;
  uint32_t e = (h >> 10) & 0x1Fu;
  uint32_t m = (uint32_t)h & 0x3FFu;

  if (e == 0u) {
    if (m == 0u) return (s << 31);  // ±0
    // Subnormal: locate MSB of m (msb in [0,9]), then exp32 = msb + 103,
    // mant23 = m bits below msb, shifted to top of 23-bit field.
    int msb = 9;
    while (!(m & (1u << msb))) --msb;
    uint32_t exp32  = (uint32_t)msb + 103u;
    uint32_t mant23 = (m << (23 - msb)) & 0x7FFFFFu;  // [22:0] drops the leading 1 at bit 23
    return (s << 31) | (exp32 << 23) | mant23;
  }
  if (e == 31u) {
    if (m == 0u) return (s << 31) | (0xFFu << 23);  // ±inf
    return 0x7FC00000u;                              // qNaN canonical
  }
  // Normal
  uint32_t exp32  = e + 112u;
  uint32_t mant23 = m << 13;
  return (s << 31) | (exp32 << 23) | mant23;
}

static void tick(Vfp16_to_fp32_pipe* dut) {
  dut->clk = 0; dut->eval();
  dut->clk = 1; dut->eval();
}

static void reset(Vfp16_to_fp32_pipe* dut) {
  dut->rst_n = 0; dut->valid_in = 0; dut->a = 0;
  for (int i = 0; i < 4; ++i) tick(dut);
  dut->rst_n = 1;
}

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  Vfp16_to_fp32_pipe* dut = new Vfp16_to_fp32_pipe;
  reset(dut);

  std::queue<std::pair<uint16_t, uint32_t>> in_flight;

  for (uint32_t bits = 0; bits < 0x10000u; ++bits) {
    uint16_t h = (uint16_t)bits;
    uint32_t exp = fp16_bits_to_fp32_ref(h);
    dut->valid_in = 1;
    dut->a = h;
    in_flight.push({h, exp});
    tick(dut);
    if (dut->valid_out) {
      auto p = in_flight.front(); in_flight.pop();
      uint32_t got = dut->y;
      ++g_checked;
      if (got != p.second) {
        if (g_fail < 30)
          std::fprintf(stderr,
                       "MISMATCH h=%04x got=%08x exp=%08x\n",
                       p.first, got, p.second);
        ++g_fail;
      }
    }
  }
  dut->valid_in = 0;
  while (!in_flight.empty()) {
    tick(dut);
    if (dut->valid_out) {
      auto p = in_flight.front(); in_flight.pop();
      uint32_t got = dut->y;
      ++g_checked;
      if (got != p.second) ++g_fail;
    }
  }

  delete dut;
  std::printf("fp16_to_fp32_pipe (LATENCY=%d): checked=%ld  mismatches=%d\n",
              PIPE_LATENCY, g_checked, g_fail);
  if (g_fail == 0 && g_checked == 0x10000) {
    std::printf("PASS: fp16_to_fp32_pipe EXHAUSTIVE (65536/65536) bit-exact\n");
    return 0;
  }
  std::fprintf(stderr,
               "FAIL: fp16_to_fp32_pipe %d mismatches (checked=%ld)\n",
               g_fail, g_checked);
  return 1;
}
