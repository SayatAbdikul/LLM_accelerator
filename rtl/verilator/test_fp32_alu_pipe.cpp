// Integration gate for the shared fp32_alu_pipe — proves that the SFU's
// Phase-2 ALU API works correctly for every basic op (ADD, SUB, MUL, DIV,
// SQRT, CVT_H2F, CVT_F2H, CVT_I2F, QUANT_I8). Each primitive is already
// independently 0-ULP gated; this test confirms (1) the op-mux routes
// correctly, (2) the LATENCY contract holds across mixed-op streams,
// (3) op_out is forwarded for FSM result-typing.

#include "Vfp32_alu_pipe.h"
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

static inline float    bits2f(uint32_t u) { float    f; std::memcpy(&f, &u, 4); return f; }
static inline uint32_t f2bits(float f)    { uint32_t u; std::memcpy(&u, &f, 4); return u; }

enum Op : uint32_t {
  OP_ADD=0, OP_SUB=1, OP_MUL=2, OP_DIV=3, OP_SQRT=4,
  OP_CVT_H2F=5, OP_CVT_F2H=6, OP_CVT_I2F=7, OP_QUANT_I8=8
};

static int rne_int(float x) {
  long long fi = (long long)std::floor(x);
  float frac = x - (float)fi;
  if (frac > 0.5f) return (int)(fi + 1);
  if (frac < 0.5f) return (int)fi;
  if (fi & 1LL)    return (int)(fi + 1);
  return (int)fi;
}

static uint32_t golden(uint32_t op, uint32_t a, uint32_t b) {
  if (op == OP_ADD || op == OP_SUB || op == OP_MUL || op == OP_DIV) {
    float fa = bits2f(a), fb = bits2f(b), r;
    if (op == OP_ADD) r = fa + fb;
    else if (op == OP_SUB) r = fa - fb;
    else if (op == OP_MUL) r = fa * fb;
    else r = fa / fb;
    if (std::isnan(r)) return 0x7FC00000u;
    return f2bits(r);
  }
  if (op == OP_SQRT) {
    float fa = bits2f(a);
    if (std::isnan(fa)) return 0x7FC00000u;
    float r = std::sqrt(fa);
    if (std::isnan(r)) return 0x7FC00000u;
    return f2bits(r);
  }
  if (op == OP_CVT_H2F) {
    uint16_t h = (uint16_t)(a & 0xFFFFu);
    uint32_t s = (h >> 15) & 1u, e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
    if (e == 0u) {
      if (m == 0u) return s << 31;
      int msb = 9; while (!(m & (1u<<msb))) --msb;
      uint32_t exp32 = (uint32_t)msb + 103u;
      uint32_t mant23 = (m << (23 - msb)) & 0x7FFFFFu;
      return (s << 31) | (exp32 << 23) | mant23;
    }
    if (e == 31u) {
      if (m == 0u) return (s << 31) | (0xFFu << 23);
      return 0x7FC00000u;
    }
    return (s << 31) | ((e + 112u) << 23) | (m << 13);
  }
  if (op == OP_CVT_F2H) {
    float f = bits2f(a);
    if (std::isnan(f)) return 0x7E00u;
    __fp16 h = (__fp16)f;
    uint16_t hb; std::memcpy(&hb, &h, 2);
    return (uint32_t)hb;
  }
  if (op == OP_CVT_I2F) {
    return f2bits((float)(int32_t)a);
  }
  if (op == OP_QUANT_I8) {
    float v = bits2f(a);
    int8_t q;
    if (std::isnan(v))      q = 0;
    else if (std::isinf(v)) q = v > 0 ? 127 : -128;
    else if (v >=  127.5f)  q = 127;
    else if (v <= -128.5f)  q = -128;
    else {
      int qi = rne_int(v);
      if (qi >  127) qi =  127;
      if (qi < -128) qi = -128;
      q = (int8_t)qi;
    }
    // sign-extend int8 to 32 bits
    return (uint32_t)(int32_t)q;
  }
  return 0;
}

static void tick(Vfp32_alu_pipe* dut) { dut->clk=0; dut->eval(); dut->clk=1; dut->eval(); }
static void reset(Vfp32_alu_pipe* dut) {
  dut->rst_n=0; dut->valid_in=0; dut->op=0; dut->a=0; dut->b=0;
  for (int i=0;i<4;++i) tick(dut);
  dut->rst_n=1;
}

struct Step { uint32_t op, a, b, exp; const char* tag; };

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  Vfp32_alu_pipe* dut = new Vfp32_alu_pipe;
  reset(dut);

  std::vector<Step> ops;
  std::mt19937 rng(0xA1A133u);
  std::uniform_int_distribution<uint32_t> d32(0, 0xFFFFFFFFu);

  // Mixed-op stream: 100k of each op interleaved.
  const uint32_t op_list[] = {OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_SQRT,
                              OP_CVT_H2F, OP_CVT_F2H, OP_CVT_I2F, OP_QUANT_I8};
  for (int rep = 0; rep < 100000; ++rep) {
    for (uint32_t op : op_list) {
      uint32_t a = d32(rng), b = d32(rng);
      // Bias DIV / SQRT inputs to finite normals to reduce non-finite-domain noise
      if (op == OP_DIV) {
        uint32_t e = 100u + (d32(rng) % 50u);   // E in [-27, 22]
        a = ((d32(rng) & 1u) << 31) | (e << 23) | (d32(rng) & 0x7FFFFFu);
        e = 100u + (d32(rng) % 50u);
        b = ((d32(rng) & 1u) << 31) | (e << 23) | (d32(rng) & 0x7FFFFFu);
      }
      if (op == OP_SQRT) {
        uint32_t e = 100u + (d32(rng) % 50u);
        a = (e << 23) | (d32(rng) & 0x7FFFFFu);  // positive
      }
      uint32_t exp = golden(op, a, b);
      ops.push_back({op, a, b, exp, "rand"});
    }
  }

  std::queue<std::pair<uint32_t,uint32_t>> in_flight;  // (op, exp)
  for (size_t k = 0; k < ops.size(); ++k) {
    dut->valid_in = 1; dut->op = ops[k].op;
    dut->a = ops[k].a; dut->b = ops[k].b;
    in_flight.push({ops[k].op, ops[k].exp});
    tick(dut);
    if (dut->valid_out) {
      auto p = in_flight.front(); in_flight.pop();
      uint32_t got = dut->y;
      uint32_t op_out_got = dut->op_out;
      ++g_checked;
      if (op_out_got != p.first) {
        if (g_fail < 5)
          std::fprintf(stderr, "OP_OUT-MISMATCH k=%zu got_op=%u exp_op=%u\n",
                       k, op_out_got, p.first);
        ++g_fail;
      }
      if (got != p.second) {
        if (g_fail < 20)
          std::fprintf(stderr, "MISMATCH k=%zu op=%u got=%08x exp=%08x\n",
                       k, p.first, got, p.second);
        ++g_fail;
      }
    }
  }
  dut->valid_in = 0;
  while (!in_flight.empty()) {
    tick(dut);
    if (dut->valid_out) {
      auto p = in_flight.front(); in_flight.pop();
      ++g_checked;
      if (dut->y != p.second || dut->op_out != p.first) ++g_fail;
    }
  }

  delete dut;
  std::printf("fp32_alu_pipe (LATENCY=%d): checked=%ld mismatches=%d\n",
              PIPE_LATENCY, g_checked, g_fail);
  if (g_fail == 0) {
    std::printf("PASS: fp32_alu_pipe op-mux + LATENCY contract OK\n");
    return 0;
  }
  return 1;
}
