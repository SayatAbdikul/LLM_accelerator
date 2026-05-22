// R4-split (2026-05-22) test_systolic_qkt_padded.cpp —
// padded query/key/value bias/matmul/requant replays (12 tests).
//
// Each test body is verbatim from the pre-split test_systolic_qkt.cpp.
// Shared helpers, the `using tbutil::*` decls, and the BUF_*_ID constants
// live in include/systolic_qkt_utils.h (which also pulls in test_runner.h
// for TEST_PASS / TEST_FAIL and the inline tests_pass / tests_run
// counters).

#include "include/systolic_qkt_utils.h"

namespace {

void test_qkt_query_padded_bias_replay() {
  const char* name = "qkt_query_padded_bias_replay";
  const char* replay_dir = std::getenv("RTL_QKT_REPLAY_DIR");
  if (replay_dir == nullptr || replay_dir[0] == '\0') {
    std::printf("SKIP: %s (set RTL_QKT_REPLAY_DIR to enable)\n", name);
    return;
  }
  run_projection_padded_bias_replay(name, replay_dir, "query");
}

void test_qkt_query_padded_matmul_exact_replay() {
  const char* name = "qkt_query_padded_matmul_exact_replay";
  const char* replay_dir = std::getenv("RTL_QKT_REPLAY_DIR");
  if (replay_dir == nullptr || replay_dir[0] == '\0') {
    std::printf("SKIP: %s (set RTL_QKT_REPLAY_DIR to enable)\n", name);
    return;
  }
  run_projection_padded_matmul_replay(name, replay_dir, "query", false);
}

void test_qkt_query_padded_matmul_clean_replay() {
  const char* name = "qkt_query_padded_matmul_clean_replay";
  const char* replay_dir = std::getenv("RTL_QKT_REPLAY_DIR");
  if (replay_dir == nullptr || replay_dir[0] == '\0') {
    std::printf("SKIP: %s (set RTL_QKT_REPLAY_DIR to enable)\n", name);
    return;
  }
  run_projection_padded_matmul_replay(name, replay_dir, "query", true);
}

void test_qkt_query_padded_requant_replay() {
  const char* name = "qkt_query_padded_requant_replay";
  const char* replay_dir = std::getenv("RTL_QKT_REPLAY_DIR");
  if (replay_dir == nullptr || replay_dir[0] == '\0') {
    std::printf("SKIP: %s (set RTL_QKT_REPLAY_DIR to enable)\n", name);
    return;
  }
  run_projection_padded_requant_replay(name, replay_dir, "query");
}

void test_qkt_key_padded_bias_replay() {
  const char* name = "qkt_key_padded_bias_replay";
  const char* replay_dir = std::getenv("RTL_QKT_REPLAY_DIR");
  if (replay_dir == nullptr || replay_dir[0] == '\0') {
    std::printf("SKIP: %s (set RTL_QKT_REPLAY_DIR to enable)\n", name);
    return;
  }
  run_projection_padded_bias_replay(name, replay_dir, "key");
}

void test_qkt_key_padded_matmul_exact_replay() {
  const char* name = "qkt_key_padded_matmul_exact_replay";
  const char* replay_dir = std::getenv("RTL_QKT_REPLAY_DIR");
  if (replay_dir == nullptr || replay_dir[0] == '\0') {
    std::printf("SKIP: %s (set RTL_QKT_REPLAY_DIR to enable)\n", name);
    return;
  }
  run_projection_padded_matmul_replay(name, replay_dir, "key", false);
}

void test_qkt_key_padded_matmul_clean_replay() {
  const char* name = "qkt_key_padded_matmul_clean_replay";
  const char* replay_dir = std::getenv("RTL_QKT_REPLAY_DIR");
  if (replay_dir == nullptr || replay_dir[0] == '\0') {
    std::printf("SKIP: %s (set RTL_QKT_REPLAY_DIR to enable)\n", name);
    return;
  }
  run_projection_padded_matmul_replay(name, replay_dir, "key", true);
}

void test_qkt_key_padded_requant_replay() {
  const char* name = "qkt_key_padded_requant_replay";
  const char* replay_dir = std::getenv("RTL_QKT_REPLAY_DIR");
  if (replay_dir == nullptr || replay_dir[0] == '\0') {
    std::printf("SKIP: %s (set RTL_QKT_REPLAY_DIR to enable)\n", name);
    return;
  }
  run_projection_padded_requant_replay(name, replay_dir, "key");
}

void test_qkt_value_padded_bias_replay() {
  const char* name = "qkt_value_padded_bias_replay";
  const char* replay_dir = std::getenv("RTL_QKT_REPLAY_DIR");
  if (replay_dir == nullptr || replay_dir[0] == '\0') {
    std::printf("SKIP: %s (set RTL_QKT_REPLAY_DIR to enable)\n", name);
    return;
  }
  run_projection_padded_bias_replay(name, replay_dir, "value");
}

void test_qkt_value_padded_matmul_exact_replay() {
  const char* name = "qkt_value_padded_matmul_exact_replay";
  const char* replay_dir = std::getenv("RTL_QKT_REPLAY_DIR");
  if (replay_dir == nullptr || replay_dir[0] == '\0') {
    std::printf("SKIP: %s (set RTL_QKT_REPLAY_DIR to enable)\n", name);
    return;
  }
  run_projection_padded_matmul_replay(name, replay_dir, "value", false);
}

void test_qkt_value_padded_matmul_clean_replay() {
  const char* name = "qkt_value_padded_matmul_clean_replay";
  const char* replay_dir = std::getenv("RTL_QKT_REPLAY_DIR");
  if (replay_dir == nullptr || replay_dir[0] == '\0') {
    std::printf("SKIP: %s (set RTL_QKT_REPLAY_DIR to enable)\n", name);
    return;
  }
  run_projection_padded_matmul_replay(name, replay_dir, "value", true);
}

void test_qkt_value_padded_requant_replay() {
  const char* name = "qkt_value_padded_requant_replay";
  const char* replay_dir = std::getenv("RTL_QKT_REPLAY_DIR");
  if (replay_dir == nullptr || replay_dir[0] == '\0') {
    std::printf("SKIP: %s (set RTL_QKT_REPLAY_DIR to enable)\n", name);
    return;
  }
  run_projection_padded_requant_replay(name, replay_dir, "value");
}

}  // namespace

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);

  test_qkt_query_padded_bias_replay();
  test_qkt_query_padded_matmul_exact_replay();
  test_qkt_query_padded_matmul_clean_replay();
  test_qkt_query_padded_requant_replay();
  test_qkt_key_padded_bias_replay();
  test_qkt_key_padded_matmul_exact_replay();
  test_qkt_key_padded_matmul_clean_replay();
  test_qkt_key_padded_requant_replay();
  test_qkt_value_padded_bias_replay();
  test_qkt_value_padded_matmul_exact_replay();
  test_qkt_value_padded_matmul_clean_replay();
  test_qkt_value_padded_requant_replay();

  std::printf("\n%d / %d tests passed\n", tests_pass, tests_run);
  return (tests_pass == tests_run) ? 0 : 1;
}
