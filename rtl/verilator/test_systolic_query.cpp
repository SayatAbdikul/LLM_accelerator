// Focused native regression for the first query-projection ACCUM path.

#include "Vtaccel_top.h"
#include "verilated.h"
#include "include/systolic_test_utils.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace systolic_test;

static int tests_run = 0;
static int tests_pass = 0;

#define TEST_PASS(name) do { \
  std::printf("PASS: %s\n", name); tests_pass++; tests_run++; \
} while (0)

#define TEST_FAIL(name, msg) do { \
  std::fprintf(stderr, "FAIL: %s - %s\n", name, msg); std::exit(1); \
} while (0)

namespace {

using Act16x192 = std::array<std::array<int8_t, 192>, 16>;
using Wgt192x16 = std::array<std::array<int8_t, 16>, 192>;

std::vector<uint8_t> flatten_row_major_a(const Act16x192& a) {
  std::vector<uint8_t> out(16 * 192);
  for (int r = 0; r < 16; ++r) {
    for (int c = 0; c < 192; ++c)
      out[r * 192 + c] = static_cast<uint8_t>(a[r][c]);
  }
  return out;
}

std::vector<uint8_t> flatten_row_major_b(const Wgt192x16& b) {
  std::vector<uint8_t> out(192 * 16);
  for (int r = 0; r < 192; ++r) {
    for (int c = 0; c < 16; ++c)
      out[r * 16 + c] = static_cast<uint8_t>(b[r][c]);
  }
  return out;
}

void prepare_compiler_visible_16x192x16(AXI4SlaveModel& dram, std::vector<uint64_t>& prog,
                                        const Act16x192& a, const Wgt192x16& b,
                                        uint64_t a_addr, uint64_t b_addr,
                                        int abuf_off = 0, int wbuf_off = 0) {
  write_dram_bytes(dram, a_addr, flatten_row_major_a(a));
  write_dram_bytes(dram, b_addr, flatten_row_major_b(b));
  append_load_sync(prog, 0, a_addr, BUF_ABUF_ID, abuf_off, (16 * 192) / 16);
  append_load_sync(prog, 1, b_addr, BUF_WBUF_ID, wbuf_off, (192 * 16) / 16);
}

void matmul_ref_16x192x16(const Act16x192& a, const Wgt192x16& b, int32_t (&c)[16][16]) {
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 16; ++j) {
      int32_t acc = 0;
      for (int k = 0; k < 192; ++k)
        acc += int32_t(a[i][k]) * int32_t(b[k][j]);
      c[i][j] = acc;
    }
  }
}

void expect_clean_halt(const char* name, Vtaccel_top* dut) {
  if (!dut->done || dut->fault)
    TEST_FAIL(name, "did not halt cleanly");
}

void test_query_projection_k12() {
  const char* name = "matmul_query_projection_contract_k12";
  Sim s;

  Act16x192 a{};
  Wgt192x16 b{};
  int32_t exp[16][16] = {};
  std::vector<uint64_t> prog;

  for (int i = 0; i < 16; ++i) {
    for (int k = 0; k < 192; ++k)
      a[i][k] = static_cast<int8_t>(((i * 17 + k * 5 + 3) % 15) - 7);
  }
  for (int k = 0; k < 192; ++k) {
    for (int j = 0; j < 16; ++j)
      b[k][j] = static_cast<int8_t>(((k * 11 + j * 7 + 1) % 13) - 6);
  }

  prepare_compiler_visible_16x192x16(s.dram, prog, a, b, 0x220000, 0x240000);
  matmul_ref_16x192x16(a, b, exp);

  prog.push_back(insn::CONFIG_TILE(1, 1, 12));
  prog.push_back(insn::MATMUL(BUF_ABUF_ID, 0, BUF_WBUF_ID, 0, BUF_ACCUM_ID, 0, 0, 0));
  prog.push_back(insn::SYNC(0b010));
  prog.push_back(insn::HALT());

  s.load_program(prog);
  s.run(1500000);
  expect_clean_halt(name, s.dut.get());
  if (!check_accum_16x16(s.dut.get(), 0, exp, "query-k12"))
    TEST_FAIL(name, "ACCUM mismatch");
  TEST_PASS(name);
}

}  // namespace

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);

  test_query_projection_k12();

  std::printf("\n%d / %d tests passed\n", tests_pass, tests_run);
  return (tests_pass == tests_run) ? 0 : 1;
}
