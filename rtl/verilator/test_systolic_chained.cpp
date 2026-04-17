// Verilator top-level tests for chained systolic mode.

#include "Vtaccel_top.h"
#include "Vtaccel_top___024root.h"
#include "verilated.h"
#include "include/systolic_test_utils.h"

#include <array>
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

namespace {

constexpr int ST_READ_REQ = 2;
constexpr int ST_READ_USE = 3;
constexpr int ST_DRAIN_WR = 5;
constexpr int ST_A_LOAD_REQ = 6;
constexpr int ST_INIT_TILE = 1;
constexpr int CHAIN_TOTAL_STEPS = 16 + (2 * (16 - 1));

using Row = std::array<int8_t, 16>;

struct ChainedArrayRef {
  std::array<std::array<int8_t, 15>, 16> a_skew{};
  std::array<std::array<int8_t, 15>, 16> b_skew{};
  std::array<std::array<int8_t, 16>, 16> a_out{};
  std::array<std::array<int8_t, 16>, 16> b_out{};
  std::array<std::array<int32_t, 16>, 16> acc{};

  void reset() {
    for (int i = 0; i < 16; ++i) {
      for (int j = 0; j < 16; ++j) {
        a_out[i][j] = 0;
        b_out[i][j] = 0;
        acc[i][j] = 0;
      }
      for (int s = 0; s < 15; ++s) {
        a_skew[i][s] = 0;
        b_skew[i][s] = 0;
      }
    }
  }

  void step(const Row& a_row, const Row& b_row, bool step_en, bool clear_acc) {
    Row a_edge{};
    Row b_edge{};
    auto next_a_out = a_out;
    auto next_b_out = b_out;
    auto next_acc = acc;
    auto next_a_skew = a_skew;
    auto next_b_skew = b_skew;

    for (int i = 0; i < 16; ++i) {
      a_edge[i] = (i == 0) ? a_row[i] : a_skew[i][i - 1];
      b_edge[i] = (i == 0) ? b_row[i] : b_skew[i][i - 1];
    }

    for (int i = 0; i < 16; ++i) {
      for (int j = 0; j < 16; ++j) {
        int8_t pe_a = (j == 0) ? a_edge[i] : a_out[i][j - 1];
        int8_t pe_b = (i == 0) ? b_edge[j] : b_out[i - 1][j];
        next_a_out[i][j] = clear_acc ? 0 : pe_a;
        next_b_out[i][j] = clear_acc ? 0 : pe_b;
        if (clear_acc)
          next_acc[i][j] = 0;
        else if (step_en)
          next_acc[i][j] = acc[i][j] + int32_t(pe_a) * int32_t(pe_b);
      }
    }

    if (clear_acc) {
      for (int i = 0; i < 16; ++i) {
        for (int s = 0; s < 15; ++s) {
          next_a_skew[i][s] = 0;
          next_b_skew[i][s] = 0;
        }
      }
    } else if (step_en) {
      for (int i = 0; i < 16; ++i) {
        next_a_skew[i][0] = a_row[i];
        next_b_skew[i][0] = b_row[i];
        for (int s = 1; s < 15; ++s) {
          next_a_skew[i][s] = a_skew[i][s - 1];
          next_b_skew[i][s] = b_skew[i][s - 1];
        }
      }
    }

    a_out = next_a_out;
    b_out = next_b_out;
    acc = next_acc;
    a_skew = next_a_skew;
    b_skew = next_b_skew;
  }
};

struct ScheduleTrace {
  std::vector<int> a_load_rows;
  std::vector<int> read_use_lanes;
  std::vector<int> b_stream_rows;
  std::vector<int> zero_read_lanes;
  std::vector<int> drain_rows;
  int read_use_count_before_drain = -1;
};

Row sample_vec16(Vtaccel_top* dut, bool sample_a) {
  auto* r = dut->rootp;
  Row out{};
  for (int idx = 0; idx < 16; ++idx)
    out[idx] = static_cast<int8_t>(sample_a
             ? r->taccel_top__DOT__u_systolic__DOT__u_array__DOT__a_vec[idx]
             : r->taccel_top__DOT__u_systolic__DOT__u_array__DOT__b_vec[idx]);
  return out;
}

void expect_clean_halt(const char* name, Vtaccel_top* dut) {
  if (!dut->done || dut->fault)
    TEST_FAIL(name, "did not halt cleanly");
}

bool check_identity_read_use_inputs(Vtaccel_top* dut,
                                    const int8_t (&a)[16][16],
                                    const int8_t (&eye)[16][16],
                                    int lane) {
  auto* r = dut->rootp;
  for (int idx = 0; idx < 16; ++idx) {
    int8_t exp_a = 0;
    int8_t exp_b = 0;
    if ((lane >= 0) && (lane < 16)) {
      exp_a = a[idx][lane];
      exp_b = eye[lane][idx];
    }

    int8_t got_a = static_cast<int8_t>(r->taccel_top__DOT__u_systolic__DOT__u_array__DOT__a_vec[idx]);
    int8_t got_b = static_cast<int8_t>(r->taccel_top__DOT__u_systolic__DOT__u_array__DOT__b_vec[idx]);
    if ((got_a != exp_a) || (got_b != exp_b)) {
      std::fprintf(stderr,
                   "lane=%d vec[%d] got_a=%d exp_a=%d got_b=%d exp_b=%d\n",
                   lane, idx, int(got_a), int(exp_a), int(got_b), int(exp_b));
      return false;
    }
  }
  return true;
}

void assert_identity_schedule(const char* name, const ScheduleTrace& trace) {
  if (trace.read_use_lanes.size() != CHAIN_TOTAL_STEPS)
    TEST_FAIL(name, "unexpected chained READ_USE length");

  for (int lane = 0; lane < CHAIN_TOTAL_STEPS; ++lane) {
    if (trace.read_use_lanes[lane] != lane) {
      std::fprintf(stderr, "lane progression mismatch idx=%d got=%d exp=%d\n",
                   lane, trace.read_use_lanes[lane], lane);
      TEST_FAIL(name, "lane progression mismatch");
    }
  }

  if (trace.a_load_rows.size() != 16 || trace.b_stream_rows.size() != 16)
    TEST_FAIL(name, "unexpected chained source row count");

  for (int row = 0; row < 16; ++row) {
    if (trace.a_load_rows[row] != row || trace.b_stream_rows[row] != row) {
      std::fprintf(stderr, "read row mismatch idx=%d a_load=%d b_stream=%d exp=%d\n",
                   row, trace.a_load_rows[row], trace.b_stream_rows[row], row);
      TEST_FAIL(name, "source row progression mismatch");
    }
  }

  if (trace.zero_read_lanes.size() != (CHAIN_TOTAL_STEPS - 16))
    TEST_FAIL(name, "unexpected zero-read window length");

  for (size_t i = 0; i < trace.zero_read_lanes.size(); ++i) {
    int exp_lane = 16 + static_cast<int>(i);
    if (trace.zero_read_lanes[i] != exp_lane) {
      std::fprintf(stderr, "zero-read lane mismatch idx=%zu got=%d exp=%d\n",
                   i, trace.zero_read_lanes[i], exp_lane);
      TEST_FAIL(name, "flush zero window mismatch");
    }
  }

  if (trace.read_use_count_before_drain != CHAIN_TOTAL_STEPS)
    TEST_FAIL(name, "drain started before chained flush completed");

  if (trace.drain_rows.size() != 64)
    TEST_FAIL(name, "unexpected drain row count");

  for (int row = 0; row < 64; ++row) {
    if (trace.drain_rows[row] != row) {
      std::fprintf(stderr, "drain row mismatch idx=%d got=%d exp=%d\n",
                   row, trace.drain_rows[row], row);
      TEST_FAIL(name, "writeback row progression mismatch");
    }
  }
}

bool check_internal_acc_flat(Vtaccel_top* dut, const int32_t (&exp)[16][16], const char* tag) {
  auto* r = dut->rootp;
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 16; ++j) {
      int idx = i * 16 + j;
      int32_t got = static_cast<int32_t>(r->taccel_top__DOT__u_systolic__DOT__acc_flat[idx]);
      if (got != exp[i][j]) {
        std::fprintf(stderr, "%s acc_flat mismatch i=%d j=%d got=%d exp=%d\n",
                     tag, i, j, got, exp[i][j]);
        return false;
      }
    }
  }
  return true;
}

bool compare_ref_acc(Vtaccel_top* dut, const ChainedArrayRef& ref, const char* tag,
                     int cycle, int state, int lane) {
  auto* r = dut->rootp;
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 16; ++j) {
      int idx = i * 16 + j;
      int32_t got = static_cast<int32_t>(r->taccel_top__DOT__u_systolic__DOT__acc_flat[idx]);
      if (got != ref.acc[i][j]) {
        std::fprintf(stderr,
                     "%s ref mismatch cycle=%d state=%d lane=%d i=%d j=%d got=%d exp=%d\n",
                     tag, cycle, state, lane, i, j, got, ref.acc[i][j]);
        return false;
      }
    }
  }
  return true;
}

void test_matmul_identity_schedule() {
  const char* name = "matmul_chained_identity_schedule";
  Sim s;
  ChainedArrayRef ref;
  ScheduleTrace trace;

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
  ref.reset();
  s.dut->start = 1;
  tick(s.dut.get(), s.dram);
  s.dut->start = 0;

  for (int cycle = 0; cycle < 700000; ++cycle) {
    auto* r = s.dut->rootp;
    int state = r->taccel_top__DOT__u_systolic__DOT__state;
    int lane = r->taccel_top__DOT__u_systolic__DOT__lane_q;

    if (state == ST_INIT_TILE) {
      Row zero{};
      ref.step(zero, zero, false, true);
    } else if (state == ST_READ_USE) {
      trace.read_use_lanes.push_back(lane);
      if (!check_identity_read_use_inputs(s.dut.get(), a, eye, lane))
        TEST_FAIL(name, "unexpected READ_USE inputs");
      ref.step(sample_vec16(s.dut.get(), true), sample_vec16(s.dut.get(), false), true, false);
    } else if (state == ST_A_LOAD_REQ) {
      if (r->taccel_top__DOT__sys_sram_b_en)
        trace.a_load_rows.push_back(r->taccel_top__DOT__sys_sram_b_row);
    } else if (state == ST_READ_REQ) {
      if (r->taccel_top__DOT__sys_sram_a_en) {
        trace.b_stream_rows.push_back(r->taccel_top__DOT__sys_sram_a_row);
      } else {
        trace.zero_read_lanes.push_back(lane);
      }
    } else if ((state == ST_DRAIN_WR) && r->taccel_top__DOT__sys_sram_a_we) {
      if (trace.read_use_count_before_drain < 0)
        trace.read_use_count_before_drain = static_cast<int>(trace.read_use_lanes.size());
      trace.drain_rows.push_back(r->taccel_top__DOT__sys_sram_a_row);
    }

    tick(s.dut.get(), s.dram);
    if ((state == ST_INIT_TILE) || (state == ST_READ_USE)) {
      if (!compare_ref_acc(s.dut.get(), ref, "chained-identity", cycle, state, lane))
        TEST_FAIL(name, "reference acc mismatch");
    }

    if (s.dut->done || s.dut->fault)
      break;
  }
  expect_clean_halt(name, s.dut.get());
  assert_identity_schedule(name, trace);
  if (!check_internal_acc_flat(s.dut.get(), exp, "chained-identity"))
    TEST_FAIL(name, "internal acc_flat mismatch");
  if (!check_accum_16x16(s.dut.get(), 0, exp, "chained-identity"))
    TEST_FAIL(name, "ACCUM mismatch");
  TEST_PASS(name);
}

void test_matmul_accumulate_flag() {
  const char* name = "matmul_chained_accumulate_flag";
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

  prepare_logical_16x16(s.dram, prog, a, b, 0x120000, 0x130000);
  matmul_ref(a, b, exp);

  prog.push_back(insn::CONFIG_TILE(1, 1, 1));
  prog.push_back(insn::MATMUL(BUF_ABUF_ID, 0, BUF_WBUF_ID, 0, BUF_ACCUM_ID, 0, 0, 0));
  prog.push_back(insn::SYNC(0b010));
  prog.push_back(insn::MATMUL(BUF_ABUF_ID, 0, BUF_WBUF_ID, 0, BUF_ACCUM_ID, 0, 0, 1));
  prog.push_back(insn::SYNC(0b010));
  prog.push_back(insn::HALT());

  s.load_program(prog);
  s.run(800000);
  expect_clean_halt(name, s.dut.get());

  for (int i = 0; i < 16; ++i)
    for (int j = 0; j < 16; ++j)
      exp[i][j] *= 2;

  if (!check_accum_16x16(s.dut.get(), 0, exp, "chained-accumulate"))
    TEST_FAIL(name, "accumulate mismatch");
  TEST_PASS(name);
}

void test_matmul_signed_extremes() {
  const char* name = "matmul_chained_signed_extremes";
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

  prepare_logical_16x16(s.dram, prog, a, b, 0x140000, 0x150000);
  matmul_ref(a, b, exp);

  prog.push_back(insn::CONFIG_TILE(1, 1, 1));
  prog.push_back(insn::MATMUL(BUF_ABUF_ID, 0, BUF_WBUF_ID, 0, BUF_ACCUM_ID, 0, 0, 0));
  prog.push_back(insn::SYNC(0b010));
  prog.push_back(insn::HALT());

  s.load_program(prog);
  s.run(700000);
  expect_clean_halt(name, s.dut.get());
  if (!check_accum_16x16(s.dut.get(), 0, exp, "chained-signed"))
    TEST_FAIL(name, "ACCUM mismatch");
  TEST_PASS(name);
}

void test_matmul_random_regression() {
  const char* name = "matmul_chained_random_regression";
  std::mt19937 rng(24680);
  std::uniform_int_distribution<int> dist(-128, 127);

  for (int tc = 0; tc < 4; ++tc) {
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

    prepare_logical_16x16(s.dram, prog, a, b, 0x160000 + tc * 0x4000, 0x162000 + tc * 0x4000);
    matmul_ref(a, b, exp);

    prog.push_back(insn::CONFIG_TILE(1, 1, 1));
    prog.push_back(insn::MATMUL(BUF_ABUF_ID, 0, BUF_WBUF_ID, 0, BUF_ACCUM_ID, 0, 0, 0));
    prog.push_back(insn::SYNC(0b010));
    prog.push_back(insn::HALT());

    s.load_program(prog);
    s.run(700000);
    expect_clean_halt(name, s.dut.get());
    if (!check_accum_16x16(s.dut.get(), 0, exp, "chained-random"))
      TEST_FAIL(name, "ACCUM mismatch");
  }

  TEST_PASS(name);
}

void test_matmul_k4_boundary_stress() {
  const char* name = "matmul_chained_k4_boundary_stress";
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

  prepare_logical_16x64x16(s.dram, prog, a, b, 0x180000, 0x190000);
  matmul_ref_16x64x16(a, b, exp);

  prog.push_back(insn::CONFIG_TILE(1, 1, 4));
  prog.push_back(insn::MATMUL(BUF_ABUF_ID, 0, BUF_WBUF_ID, 0, BUF_ACCUM_ID, 0, 0, 0));
  prog.push_back(insn::SYNC(0b010));
  prog.push_back(insn::HALT());

  s.load_program(prog);
  s.run(1100000);
  expect_clean_halt(name, s.dut.get());
  if (!check_accum_16x16(s.dut.get(), 0, exp, "chained-k4"))
    TEST_FAIL(name, "ACCUM mismatch");
  TEST_PASS(name);
}

void test_matmul_multitile_2x2x2() {
  const char* name = "matmul_chained_multitile_2x2x2";
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

  prepare_logical_32x32(s.dram, prog, a, b, 0x1A0000, 0x1C0000);
  matmul_ref_32(a, b, exp);

  prog.push_back(insn::CONFIG_TILE(2, 2, 2));
  prog.push_back(insn::MATMUL(BUF_ABUF_ID, 0, BUF_WBUF_ID, 0, BUF_ACCUM_ID, 0, 0, 0));
  prog.push_back(insn::SYNC(0b010));
  prog.push_back(insn::HALT());

  s.load_program(prog);
  s.run(1200000);
  expect_clean_halt(name, s.dut.get());
  if (!check_accum_32x32(s.dut.get(), 0, exp, "chained-multitile"))
    TEST_FAIL(name, "ACCUM mismatch");
  TEST_PASS(name);
}

}  // namespace

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);

  test_matmul_identity_schedule();
  test_matmul_accumulate_flag();
  test_matmul_signed_extremes();
  test_matmul_random_regression();
  test_matmul_k4_boundary_stress();
  test_matmul_multitile_2x2x2();

  std::printf("\n%d / %d tests passed\n", tests_pass, tests_run);
  return (tests_pass == tests_run) ? 0 : 1;
}
