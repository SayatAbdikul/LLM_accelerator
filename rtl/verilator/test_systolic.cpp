// Verilator systolic tests for the default top-level MATMUL contract.

#include "Vtaccel_top.h"
#include "verilated.h"
#include "include/systolic_test_utils.h"

#include <cstdio>
#include <cstdlib>
#include <random>
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

static void expect_clean_halt(const char* name, Vtaccel_top* dut) {
  if (!dut->done || dut->fault)
    TEST_FAIL(name, "did not halt cleanly");
}

static void test_matmul_identity() {
  const char* name = "matmul_identity_16x16";
  Sim s;

  int8_t a[16][16] = {};
  int8_t eye[16][16] = {};
  int32_t exp[16][16] = {};
  std::vector<uint64_t> prog;

  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 16; ++j) {
      a[i][j] = static_cast<int8_t>((i * 3 + j) & 0x7F);
      eye[i][j] = (i == j) ? 1 : 0;
    }
  }

  prepare_logical_16x16(s.dram, prog, a, eye, 0x100000, 0x110000);
  matmul_ref(a, eye, exp);

  prog.push_back(insn::CONFIG_TILE(1, 1, 1));
  prog.push_back(insn::MATMUL(BUF_ABUF_ID, 0, BUF_WBUF_ID, 0, BUF_ACCUM_ID, 0, 0, 0));
  prog.push_back(insn::SYNC(0b010));
  prog.push_back(insn::HALT());

  s.load_program(prog);
  s.run(500000);
  expect_clean_halt(name, s.dut.get());
  if (!check_accum_16x16(s.dut.get(), 0, exp, "identity"))
    TEST_FAIL(name, "ACCUM mismatch");
  TEST_PASS(name);
}

static void test_matmul_ones() {
  const char* name = "matmul_ones_16x16";
  Sim s;

  int8_t a[16][16] = {};
  int8_t b[16][16] = {};
  int32_t exp[16][16] = {};
  std::vector<uint64_t> prog;

  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 16; ++j) {
      a[i][j] = 1;
      b[i][j] = 1;
    }
  }

  prepare_logical_16x16(s.dram, prog, a, b, 0x120000, 0x130000);
  matmul_ref(a, b, exp);

  prog.push_back(insn::CONFIG_TILE(1, 1, 1));
  prog.push_back(insn::MATMUL(BUF_ABUF_ID, 0, BUF_WBUF_ID, 0, BUF_ACCUM_ID, 0, 0, 0));
  prog.push_back(insn::SYNC(0b010));
  prog.push_back(insn::HALT());

  s.load_program(prog);
  s.run(500000);
  expect_clean_halt(name, s.dut.get());
  if (!check_accum_16x16(s.dut.get(), 0, exp, "ones"))
    TEST_FAIL(name, "ACCUM mismatch");
  TEST_PASS(name);
}

static void test_matmul_accumulate_flag() {
  const char* name = "matmul_accumulate_flag";
  Sim s;

  int8_t a[16][16] = {};
  int8_t b[16][16] = {};
  int32_t exp[16][16] = {};
  std::vector<uint64_t> prog;

  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 16; ++j) {
      a[i][j] = 1;
      b[i][j] = (i == j) ? 2 : 0;
    }
  }

  prepare_logical_16x16(s.dram, prog, a, b, 0x140000, 0x150000);
  matmul_ref(a, b, exp);

  prog.push_back(insn::CONFIG_TILE(1, 1, 1));
  prog.push_back(insn::MATMUL(BUF_ABUF_ID, 0, BUF_WBUF_ID, 0, BUF_ACCUM_ID, 0, 0, 0));
  prog.push_back(insn::SYNC(0b010));
  prog.push_back(insn::MATMUL(BUF_ABUF_ID, 0, BUF_WBUF_ID, 0, BUF_ACCUM_ID, 0, 0, 1));
  prog.push_back(insn::SYNC(0b010));
  prog.push_back(insn::HALT());

  s.load_program(prog);
  s.run(600000);
  expect_clean_halt(name, s.dut.get());

  for (int i = 0; i < 16; ++i)
    for (int j = 0; j < 16; ++j)
      exp[i][j] *= 2;

  if (!check_accum_16x16(s.dut.get(), 0, exp, "accumulate"))
    TEST_FAIL(name, "accumulate mismatch");
  TEST_PASS(name);
}

static void test_matmul_multitile_2x2x2() {
  const char* name = "matmul_multitile_2x2x2";
  Sim s;

  int8_t a[32][32] = {};
  int8_t b[32][32] = {};
  int32_t exp[32][32] = {};
  std::vector<uint64_t> prog;

  for (int i = 0; i < 32; ++i) {
    for (int j = 0; j < 32; ++j) {
      a[i][j] = static_cast<int8_t>(((i * 7 + j * 5 + 3) % 11) - 5);
      b[i][j] = static_cast<int8_t>(((i * 3 + j * 9 + 1) % 13) - 6);
    }
  }

  prepare_logical_32x32(s.dram, prog, a, b, 0x160000, 0x180000);
  matmul_ref_32(a, b, exp);

  prog.push_back(insn::CONFIG_TILE(2, 2, 2));
  prog.push_back(insn::MATMUL(BUF_ABUF_ID, 0, BUF_WBUF_ID, 0, BUF_ACCUM_ID, 0, 0, 0));
  prog.push_back(insn::SYNC(0b010));
  prog.push_back(insn::HALT());

  s.load_program(prog);
  s.run(900000);
  expect_clean_halt(name, s.dut.get());
  if (!check_accum_32x32(s.dut.get(), 0, exp, "multitile"))
    TEST_FAIL(name, "ACCUM mismatch");
  TEST_PASS(name);
}

static void test_matmul_signed_extremes_16x16() {
  const char* name = "matmul_signed_extremes_16x16";
  Sim s;

  int8_t a[16][16] = {};
  int8_t b[16][16] = {};
  int32_t exp[16][16] = {};
  std::vector<uint64_t> prog;

  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 16; ++j) {
      a[i][j] = ((i + j) & 1) ? int8_t(-128) : int8_t(127);
      b[i][j] = ((i * 3 + j * 5) & 1) ? int8_t(127) : int8_t(-128);
    }
  }

  prepare_logical_16x16(s.dram, prog, a, b, 0x1A0000, 0x1B0000);
  matmul_ref(a, b, exp);

  prog.push_back(insn::CONFIG_TILE(1, 1, 1));
  prog.push_back(insn::MATMUL(BUF_ABUF_ID, 0, BUF_WBUF_ID, 0, BUF_ACCUM_ID, 0, 0, 0));
  prog.push_back(insn::SYNC(0b010));
  prog.push_back(insn::HALT());

  s.load_program(prog);
  s.run(500000);
  expect_clean_halt(name, s.dut.get());
  if (!check_accum_16x16(s.dut.get(), 0, exp, "signed-ext"))
    TEST_FAIL(name, "ACCUM mismatch");
  TEST_PASS(name);
}

static void test_matmul_random_regression_16x16() {
  const char* name = "matmul_random_regression_16x16";
  std::mt19937 rng(12345);
  std::uniform_int_distribution<int> dist(-128, 127);

  for (int tc = 0; tc < 8; ++tc) {
    Sim s;
    int8_t a[16][16] = {};
    int8_t b[16][16] = {};
    int32_t exp[16][16] = {};
    std::vector<uint64_t> prog;

    for (int i = 0; i < 16; ++i) {
      for (int j = 0; j < 16; ++j) {
        a[i][j] = static_cast<int8_t>(dist(rng));
        b[i][j] = static_cast<int8_t>(dist(rng));
      }
    }

    prepare_logical_16x16(s.dram, prog, a, b, 0x1C0000 + tc * 0x4000, 0x1C2000 + tc * 0x4000);
    matmul_ref(a, b, exp);

    prog.push_back(insn::CONFIG_TILE(1, 1, 1));
    prog.push_back(insn::MATMUL(BUF_ABUF_ID, 0, BUF_WBUF_ID, 0, BUF_ACCUM_ID, 0, 0, 0));
    prog.push_back(insn::SYNC(0b010));
    prog.push_back(insn::HALT());

    s.load_program(prog);
    s.run(500000);
    expect_clean_halt(name, s.dut.get());
    if (!check_accum_16x16(s.dut.get(), 0, exp, "random"))
      TEST_FAIL(name, "ACCUM mismatch");
  }

  TEST_PASS(name);
}

static void test_matmul_k4_boundary_stress() {
  const char* name = "matmul_k4_boundary_stress";
  Sim s;

  int8_t a[16][64] = {};
  int8_t b[64][16] = {};
  int32_t exp[16][16] = {};
  std::vector<uint64_t> prog;

  for (int i = 0; i < 16; ++i) {
    for (int k = 0; k < 64; ++k)
      a[i][k] = ((i + k) & 1) ? int8_t(127) : int8_t(-128);
  }
  for (int k = 0; k < 64; ++k) {
    for (int j = 0; j < 16; ++j)
      b[k][j] = ((k * 7 + j) & 1) ? int8_t(-128) : int8_t(127);
  }

  prepare_logical_16x64x16(s.dram, prog, a, b, 0x200000, 0x210000);
  matmul_ref_16x64x16(a, b, exp);

  prog.push_back(insn::CONFIG_TILE(1, 1, 4));
  prog.push_back(insn::MATMUL(BUF_ABUF_ID, 0, BUF_WBUF_ID, 0, BUF_ACCUM_ID, 0, 0, 0));
  prog.push_back(insn::SYNC(0b010));
  prog.push_back(insn::HALT());

  s.load_program(prog);
  s.run(900000);
  expect_clean_halt(name, s.dut.get());
  if (!check_accum_16x16(s.dut.get(), 0, exp, "k4"))
    TEST_FAIL(name, "ACCUM mismatch");
  TEST_PASS(name);
}

static void test_matmul_multitile_random_regression_2x2x2() {
  const char* name = "matmul_multitile_random_regression_2x2x2";
  std::mt19937 rng(67890);
  std::uniform_int_distribution<int> dist(-128, 127);

  for (int tc = 0; tc < 4; ++tc) {
    Sim s;
    int8_t a[32][32] = {};
    int8_t b[32][32] = {};
    int32_t exp[32][32] = {};
    std::vector<uint64_t> prog;

    for (int i = 0; i < 32; ++i) {
      for (int j = 0; j < 32; ++j) {
        a[i][j] = static_cast<int8_t>(dist(rng));
        b[i][j] = static_cast<int8_t>(dist(rng));
      }
    }

    prepare_logical_32x32(s.dram, prog, a, b, 0x220000 + tc * 0x8000, 0x224000 + tc * 0x8000);
    matmul_ref_32(a, b, exp);

    prog.push_back(insn::CONFIG_TILE(2, 2, 2));
    prog.push_back(insn::MATMUL(BUF_ABUF_ID, 0, BUF_WBUF_ID, 0, BUF_ACCUM_ID, 0, 0, 0));
    prog.push_back(insn::SYNC(0b010));
    prog.push_back(insn::HALT());

    s.load_program(prog);
    s.run(1000000);
    expect_clean_halt(name, s.dut.get());
    if (!check_accum_32x32(s.dut.get(), 0, exp, "multitile-random"))
      TEST_FAIL(name, "ACCUM mismatch");
  }

  TEST_PASS(name);
}

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);

  test_matmul_identity();
  test_matmul_ones();
  test_matmul_accumulate_flag();
  test_matmul_multitile_2x2x2();
  test_matmul_signed_extremes_16x16();
  test_matmul_random_regression_16x16();
  test_matmul_k4_boundary_stress();
  test_matmul_multitile_random_regression_2x2x2();

  std::printf("\n%d / %d tests passed\n", tests_pass, tests_run);
  return (tests_pass == tests_run) ? 0 : 1;
}
