// R4-split (2026-05-22) test_systolic_qkt_basic.cpp —
// transpose + matmul prep regressions (3 tests).
//
// Each test body is verbatim from the pre-split test_systolic_qkt.cpp.
// Shared helpers, the `using tbutil::*` decls, and the BUF_*_ID constants
// live in include/systolic_qkt_utils.h (which also pulls in test_runner.h
// for TEST_PASS / TEST_FAIL and the inline tests_pass / tests_run
// counters).

#include "include/systolic_qkt_utils.h"

namespace {

void test_qkt_key_transpose_208x64() {
  const char* name = "qkt_key_transpose_208x64";
  SimHarness s;
  Key208x64 key{};
  KeyT64x208 expected{};

  for (int r = 0; r < 208; ++r) {
    for (int c = 0; c < 64; ++c) {
      int8_t v = (r >= 197) ? int8_t(0) : int8_t(((r * 9 + c * 5 + 3) % 31) - 15);
      key[r][c] = v;
      expected[c][r] = v;
    }
  }

  sram_write_bytes(s.dut.get(), BUF_ABUF_ID, 0, flatten_key_row_major(key));
  s.load({
      insn::BUF_COPY(BUF_ABUF_ID, 0, BUF_WBUF_ID, 0, (208 * 64) / 16, 13, 1),
      insn::HALT(),
  });
  s.run(200000);
  expect_clean_halt(name, s.dut.get());

  auto got = sram_read_bytes(s.dut.get(), BUF_WBUF_ID, 0, 64 * 208);
  auto exp = flatten_keyt_row_major(expected);
  if (got != exp)
    TEST_FAIL(name, "transposed K mismatch");
  TEST_PASS(name);
}

void test_qkt_matmul_pretransposed_16x64x208() {
  const char* name = "qkt_matmul_pretransposed_16x64x208";
  SimHarness s;
  QStrip16x64 query{};
  KeyT64x208 key_t{};
  int32_t expected[16][208] = {};

  for (int r = 0; r < 16; ++r) {
    for (int c = 0; c < 64; ++c)
      query[r][c] = int8_t(((r * 11 + c * 7 + 1) % 27) - 13);
  }
  for (int r = 0; r < 64; ++r) {
    for (int c = 0; c < 208; ++c)
      key_t[r][c] = int8_t(((r * 5 + c * 3 + 2) % 29) - 14);
  }

  matmul_ref_16x64x208(query, key_t, expected);
  sram_write_bytes(s.dut.get(), BUF_ABUF_ID, 0, flatten_qstrip_row_major(query));
  sram_write_bytes(s.dut.get(), BUF_WBUF_ID, 0, flatten_keyt_row_major(key_t));
  s.load({
      insn::CONFIG_TILE(1, 13, 4),
      insn::MATMUL(BUF_ABUF_ID, 0, BUF_WBUF_ID, 0, BUF_ACCUM_ID, 0, 0, 0),
      insn::SYNC(0b010),
      insn::HALT(),
  });
  s.run(1500000);
  expect_clean_halt(name, s.dut.get());

  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 208; ++j) {
      int32_t got = read_accum_wide(s.dut.get(), 0, i, j, 208);
      if (got != expected[i][j]) {
        std::fprintf(stderr,
                     "wide QK^T mismatch row=%d col=%d got=%d exp=%d\n",
                     i, j, got, expected[i][j]);
        TEST_FAIL(name, "pre-transposed QK^T MATMUL mismatch");
      }
    }
  }
  TEST_PASS(name);
}

void test_qkt_matmul_pretransposed_nonzero_qoff_16x64x208() {
  const char* name = "qkt_matmul_pretransposed_nonzero_qoff_16x64x208";
  SimHarness s;
  constexpr int Q_OFF_UNITS = 4992;
  QStrip16x64 query{};
  KeyT64x208 key_t{};
  int32_t expected[16][208] = {};

  for (int r = 0; r < 16; ++r) {
    for (int c = 0; c < 64; ++c)
      query[r][c] = int8_t(((r * 11 + c * 7 + 1) % 27) - 13);
  }
  for (int r = 0; r < 64; ++r) {
    for (int c = 0; c < 208; ++c)
      key_t[r][c] = int8_t(((r * 5 + c * 3 + 2) % 29) - 14);
  }

  matmul_ref_16x64x208(query, key_t, expected);
  sram_write_bytes(s.dut.get(), BUF_ABUF_ID, size_t(Q_OFF_UNITS) * 16, flatten_qstrip_row_major(query));
  sram_write_bytes(s.dut.get(), BUF_WBUF_ID, 0, flatten_keyt_row_major(key_t));
  s.load({
      insn::CONFIG_TILE(1, 13, 4),
      insn::MATMUL(BUF_ABUF_ID, Q_OFF_UNITS, BUF_WBUF_ID, 0, BUF_ACCUM_ID, 0, 0, 0),
      insn::SYNC(0b010),
      insn::HALT(),
  });
  s.run(1500000);
  expect_clean_halt(name, s.dut.get());

  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 208; ++j) {
      int32_t got = read_accum_wide(s.dut.get(), 0, i, j, 208);
      if (got != expected[i][j]) {
        std::fprintf(stderr,
                     "wide nonzero-qoff QK^T mismatch row=%d col=%d got=%d exp=%d\n",
                     i, j, got, expected[i][j]);
        TEST_FAIL(name, "pre-transposed nonzero-qoff QK^T MATMUL mismatch");
      }
    }
  }
  TEST_PASS(name);
}

}  // namespace

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);

  test_qkt_key_transpose_208x64();
  test_qkt_matmul_pretransposed_16x64x208();
  test_qkt_matmul_pretransposed_nonzero_qoff_16x64x208();

  std::printf("\n%d / %d tests passed\n", tests_pass, tests_run);
  return (tests_pass == tests_run) ? 0 : 1;
}
