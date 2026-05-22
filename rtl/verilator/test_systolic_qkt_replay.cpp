// R4-split (2026-05-22) test_systolic_qkt_replay.cpp —
// single-op bias/requant/exact_state replays (5 tests).
//
// Each test body is verbatim from the pre-split test_systolic_qkt.cpp.
// Shared helpers, the `using tbutil::*` decls, and the BUF_*_ID constants
// live in include/systolic_qkt_utils.h (which also pulls in test_runner.h
// for TEST_PASS / TEST_FAIL and the inline tests_pass / tests_run
// counters).

#include "include/systolic_qkt_utils.h"

namespace {

void test_qkt_exact_state_replay() {
  const char* name = "qkt_exact_state_replay";
  const char* replay_dir = std::getenv("RTL_QKT_REPLAY_DIR");
  if (replay_dir == nullptr || replay_dir[0] == '\0') {
    std::printf("SKIP: %s (set RTL_QKT_REPLAY_DIR to enable)\n", name);
    return;
  }

  const std::string base(replay_dir);
  auto query_bytes = read_binary_file(base + "/query_input.raw");
  auto key_t_bytes = read_binary_file(base + "/key_transposed.raw");
  auto accum_pre_bytes = read_binary_file(base + "/accum_pre_matmul.raw");
  auto golden_qkt_bytes = read_binary_file(base + "/golden_qkt.raw");

  if (query_bytes.size() != 16u * 64u)
    TEST_FAIL(name, "unexpected query_input.raw size");
  if (key_t_bytes.size() != 64u * 208u)
    TEST_FAIL(name, "unexpected key_transposed.raw size");
  if (accum_pre_bytes.size() != 16u * 197u * sizeof(int32_t))
    TEST_FAIL(name, "unexpected accum_pre_matmul.raw size");
  if (golden_qkt_bytes.size() != 16u * 197u * sizeof(int32_t))
    TEST_FAIL(name, "unexpected golden_qkt.raw size");

  SimHarness s;
  constexpr int Q_OFF_UNITS = 4992;
  sram_write_bytes(s.dut.get(), BUF_ABUF_ID, size_t(Q_OFF_UNITS) * 16, query_bytes);
  sram_write_bytes(s.dut.get(), BUF_WBUF_ID, 0, key_t_bytes);
  sram_write_bytes(s.dut.get(), BUF_ACCUM_ID, 0, accum_pre_bytes);

  s.load({
      insn::CONFIG_TILE(1, 13, 4),
      insn::MATMUL(BUF_ABUF_ID, Q_OFF_UNITS, BUF_WBUF_ID, 0, BUF_ACCUM_ID, 0, 0, 0),
      insn::SYNC(0b010),
      insn::HALT(),
  });
  s.run(1500000);
  expect_clean_halt(name, s.dut.get());

  const auto* golden = reinterpret_cast<const int32_t*>(golden_qkt_bytes.data());
  bool accum_pre_nonzero = false;
  const auto* accum_pre = reinterpret_cast<const int32_t*>(accum_pre_bytes.data());
  for (int i = 0; i < 16 * 197; ++i) {
    if (accum_pre[i] != 0) {
      accum_pre_nonzero = true;
      break;
    }
  }
  if (accum_pre_nonzero)
    std::printf("INFO: %s replay starts from nonzero ACCUM pre-state\n", name);

  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 197; ++j) {
      int32_t got = read_accum_wide(s.dut.get(), 0, i, j, 208);
      int32_t exp = golden[size_t(i) * 197 + size_t(j)];
      if (got != exp) {
        std::fprintf(stderr,
                     "exact-state replay mismatch row=%d col=%d got=%d exp=%d",
                     i, j, got, exp);
        if (i == 1 && j == 0)
          std::fprintf(stderr, " [known baseline coordinate]");
        std::fprintf(stderr, "\n");
        TEST_FAIL(name, "exact-state QK^T replay mismatch");
      }
    }
  }
  TEST_PASS(name);
}

void test_qkt_query_bias_replay() {
  const char* name = "qkt_query_bias_replay";
  const char* replay_dir = std::getenv("RTL_QKT_REPLAY_DIR");
  if (replay_dir == nullptr || replay_dir[0] == '\0') {
    std::printf("SKIP: %s (set RTL_QKT_REPLAY_DIR to enable)\n", name);
    return;
  }
  run_projection_bias_replay(name, replay_dir, "query");
}

void test_qkt_query_requant_replay() {
  const char* name = "qkt_query_requant_replay";
  const char* replay_dir = std::getenv("RTL_QKT_REPLAY_DIR");
  if (replay_dir == nullptr || replay_dir[0] == '\0') {
    std::printf("SKIP: %s (set RTL_QKT_REPLAY_DIR to enable)\n", name);
    return;
  }
  run_projection_requant_replay(name, replay_dir, "query");
}

void test_qkt_key_bias_replay() {
  const char* name = "qkt_key_bias_replay";
  const char* replay_dir = std::getenv("RTL_QKT_REPLAY_DIR");
  if (replay_dir == nullptr || replay_dir[0] == '\0') {
    std::printf("SKIP: %s (set RTL_QKT_REPLAY_DIR to enable)\n", name);
    return;
  }
  run_projection_bias_replay(name, replay_dir, "key");
}

void test_qkt_key_requant_replay() {
  const char* name = "qkt_key_requant_replay";
  const char* replay_dir = std::getenv("RTL_QKT_REPLAY_DIR");
  if (replay_dir == nullptr || replay_dir[0] == '\0') {
    std::printf("SKIP: %s (set RTL_QKT_REPLAY_DIR to enable)\n", name);
    return;
  }
  run_projection_requant_replay(name, replay_dir, "key");
}

}  // namespace

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);

  test_qkt_exact_state_replay();
  test_qkt_query_bias_replay();
  test_qkt_query_requant_replay();
  test_qkt_key_bias_replay();
  test_qkt_key_requant_replay();

  std::printf("\n%d / %d tests passed\n", tests_pass, tests_run);
  return (tests_pass == tests_run) ? 0 : 1;
}
