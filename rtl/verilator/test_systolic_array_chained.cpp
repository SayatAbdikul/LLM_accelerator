// Unit-level chained systolic-array tests.

#include "Vsystolic_array.h"
#include "verilated.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>

static int tests_run = 0;
static int tests_pass = 0;

#define TEST_PASS(name) do { \
  std::printf("PASS: %s\n", name); tests_pass++; tests_run++; \
} while (0)

#define TEST_FAIL(name, msg) do { \
  std::fprintf(stderr, "FAIL: %s - %s\n", name, msg); std::exit(1); \
} while (0)

namespace {

constexpr int SYS_DIM = 16;
constexpr int CHAIN_FLUSH_CYCLES = 2 * (SYS_DIM - 1);
constexpr int CHAIN_TOTAL_STEPS = SYS_DIM + CHAIN_FLUSH_CYCLES;

using Row = std::array<int8_t, SYS_DIM>;
using AccMatrix = std::array<std::array<int32_t, SYS_DIM>, SYS_DIM>;

struct ChainedArrayRef {
  std::array<std::array<int8_t, SYS_DIM - 1>, SYS_DIM> a_skew{};
  std::array<std::array<int8_t, SYS_DIM - 1>, SYS_DIM> b_skew{};
  std::array<std::array<int8_t, SYS_DIM>, SYS_DIM> a_out{};
  std::array<std::array<int8_t, SYS_DIM>, SYS_DIM> b_out{};
  AccMatrix acc{};

  void reset() {
    for (int i = 0; i < SYS_DIM; ++i) {
      for (int j = 0; j < SYS_DIM; ++j) {
        a_out[i][j] = 0;
        b_out[i][j] = 0;
        acc[i][j] = 0;
      }
      for (int s = 0; s < SYS_DIM - 1; ++s) {
        a_skew[i][s] = 0;
        b_skew[i][s] = 0;
      }
    }
  }

  void step(const Row& a_row, const Row& b_row, bool step_en, bool clear_acc) {
    Row a_edge{};
    Row b_edge{};
    std::array<std::array<int8_t, SYS_DIM>, SYS_DIM> pe_a_in{};
    std::array<std::array<int8_t, SYS_DIM>, SYS_DIM> pe_b_in{};
    auto next_a_out = a_out;
    auto next_b_out = b_out;
    auto next_acc = acc;
    auto next_a_skew = a_skew;
    auto next_b_skew = b_skew;

    for (int i = 0; i < SYS_DIM; ++i) {
      a_edge[i] = (i == 0) ? a_row[i] : a_skew[i][i - 1];
      b_edge[i] = (i == 0) ? b_row[i] : b_skew[i][i - 1];
    }

    for (int i = 0; i < SYS_DIM; ++i) {
      for (int j = 0; j < SYS_DIM; ++j) {
        pe_a_in[i][j] = (j == 0) ? a_edge[i] : a_out[i][j - 1];
        pe_b_in[i][j] = (i == 0) ? b_edge[j] : b_out[i - 1][j];
        next_a_out[i][j] = pe_a_in[i][j];
        next_b_out[i][j] = pe_b_in[i][j];

        if (clear_acc) {
          next_acc[i][j] = 0;
        } else if (step_en) {
          next_acc[i][j] = acc[i][j] + int32_t(pe_a_in[i][j]) * int32_t(pe_b_in[i][j]);
        }
      }
    }

    if (clear_acc) {
      for (int i = 0; i < SYS_DIM; ++i) {
        for (int s = 0; s < SYS_DIM - 1; ++s) {
          next_a_skew[i][s] = 0;
          next_b_skew[i][s] = 0;
        }
      }
    } else if (step_en) {
      for (int i = 0; i < SYS_DIM; ++i) {
        next_a_skew[i][0] = a_row[i];
        next_b_skew[i][0] = b_row[i];
        for (int s = 1; s < SYS_DIM - 1; ++s) {
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

void set_row_data(Vsystolic_array* dut, const Row& row, VlWide<4>& port) {
  for (int word = 0; word < 4; ++word) {
    uint32_t packed = 0;
    for (int byte = 0; byte < 4; ++byte) {
      int idx = word * 4 + byte;
      packed |= uint32_t(uint8_t(row[idx])) << (8 * byte);
    }
    port[word] = packed;
  }
}

void drive_rows(Vsystolic_array* dut, const Row& a_row, const Row& b_row) {
  set_row_data(dut, a_row, dut->a_row_data);
  set_row_data(dut, b_row, dut->b_row_data);
}

void tick(Vsystolic_array* dut) {
  dut->clk = 0;
  dut->eval();
  dut->clk = 1;
  dut->eval();
}

void reset(Vsystolic_array* dut, ChainedArrayRef& ref) {
  Row zero{};
  dut->rst_n = 0;
  dut->step_en = 0;
  dut->clear_acc = 0;
  drive_rows(dut, zero, zero);
  for (int i = 0; i < 4; ++i)
    tick(dut);
  dut->rst_n = 1;
  dut->step_en = 0;
  dut->clear_acc = 1;
  tick(dut);
  ref.reset();
  dut->clear_acc = 0;
  dut->step_en = 0;
  tick(dut);
}

void compare_acc(const char* name, Vsystolic_array* dut, const ChainedArrayRef& ref, int cycle) {
  for (int i = 0; i < SYS_DIM; ++i) {
    for (int j = 0; j < SYS_DIM; ++j) {
      int idx = i * SYS_DIM + j;
      int32_t got = static_cast<int32_t>(dut->acc_flat[idx]);
      if (got != ref.acc[i][j]) {
        std::fprintf(stderr,
                     "%s cycle=%d pe=(%d,%d) got=%d exp=%d\n",
                     name, cycle, i, j, got, ref.acc[i][j]);
        TEST_FAIL(name, "chained array mismatch");
      }
    }
  }
}

void step_and_check(const char* name, Vsystolic_array* dut, ChainedArrayRef& ref,
                    const Row& a_row, const Row& b_row, int cycle,
                    bool step_en = true, bool clear_acc = false) {
  dut->rst_n = 1;
  dut->step_en = step_en ? 1 : 0;
  dut->clear_acc = clear_acc ? 1 : 0;
  drive_rows(dut, a_row, b_row);
  tick(dut);
  ref.step(a_row, b_row, step_en, clear_acc);
  compare_acc(name, dut, ref, cycle);
}

void test_single_impulse_timing() {
  const char* name = "systolic_array_chained_single_impulse";
  Vsystolic_array dut;
  ChainedArrayRef ref;
  reset(&dut, ref);

  Row a_row{};
  Row b_row{};
  constexpr int target_row = 7;
  constexpr int target_col = 11;
  constexpr int8_t a_val = 3;
  constexpr int8_t b_val = 5;
  int first_nonzero_cycle = -1;

  a_row[target_row] = a_val;
  b_row[target_col] = b_val;
  step_and_check(name, &dut, ref, a_row, b_row, 0);

  for (int cycle = 1; cycle < CHAIN_TOTAL_STEPS; ++cycle) {
    Row zero{};
    step_and_check(name, &dut, ref, zero, zero, cycle);
    int32_t got = static_cast<int32_t>(dut.acc_flat[target_row * SYS_DIM + target_col]);
    if ((first_nonzero_cycle < 0) && (got != 0))
      first_nonzero_cycle = cycle;
  }

  if (first_nonzero_cycle != (target_row + target_col)) {
    std::fprintf(stderr, "first nonzero cycle got=%d exp=%d\n",
                 first_nonzero_cycle, target_row + target_col);
    TEST_FAIL(name, "unexpected chained arrival timing");
  }
  if (static_cast<int32_t>(dut.acc_flat[target_row * SYS_DIM + target_col]) != int32_t(a_val) * int32_t(b_val))
    TEST_FAIL(name, "final impulse value mismatch");

  Row zero{};
  step_and_check(name, &dut, ref, zero, zero, CHAIN_TOTAL_STEPS);
  TEST_PASS(name);
}

void test_identity_stream() {
  const char* name = "systolic_array_chained_identity_stream";
  Vsystolic_array dut;
  ChainedArrayRef ref;
  reset(&dut, ref);

  int8_t a[16][16] = {};
  int8_t eye[16][16] = {};

  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 16; ++j) {
      a[i][j] = static_cast<int8_t>((i * 3 + j) & 0x7F);
      eye[i][j] = (i == j) ? 1 : 0;
    }
  }

  for (int cycle = 0; cycle < SYS_DIM; ++cycle) {
    Row a_row{};
    Row b_row{};
    for (int lane = 0; lane < SYS_DIM; ++lane) {
      a_row[lane] = a[lane][cycle];
      b_row[lane] = eye[cycle][lane];
    }
    step_and_check(name, &dut, ref, a_row, b_row, cycle);
  }

  for (int cycle = SYS_DIM; cycle < CHAIN_TOTAL_STEPS; ++cycle) {
    Row zero{};
    step_and_check(name, &dut, ref, zero, zero, cycle);
  }

  for (int i = 0; i < SYS_DIM; ++i) {
    for (int j = 0; j < SYS_DIM; ++j) {
      int32_t got = static_cast<int32_t>(dut.acc_flat[i * SYS_DIM + j]);
      int32_t exp = (i == j) ? a[i][j] : a[i][j];
      if (got != exp) {
        std::fprintf(stderr, "identity result mismatch i=%d j=%d got=%d exp=%d\n",
                     i, j, got, exp);
        TEST_FAIL(name, "identity stream mismatch");
      }
    }
  }

  Row zero{};
  step_and_check(name, &dut, ref, zero, zero, CHAIN_TOTAL_STEPS);
  TEST_PASS(name);
}

void test_signed_extremes() {
  const char* name = "systolic_array_chained_signed_extremes";
  Vsystolic_array dut;
  ChainedArrayRef ref;
  reset(&dut, ref);

  int8_t a[16][16] = {};
  int8_t b[16][16] = {};

  for (int i = 0; i < 16; ++i) {
    for (int k = 0; k < 16; ++k)
      a[i][k] = ((i + k) & 1) ? int8_t(-128) : int8_t(127);
  }
  for (int k = 0; k < 16; ++k) {
    for (int j = 0; j < 16; ++j)
      b[k][j] = ((k * 5 + j) & 1) ? int8_t(127) : int8_t(-128);
  }

  for (int cycle = 0; cycle < SYS_DIM; ++cycle) {
    Row a_row{};
    Row b_row{};
    for (int lane = 0; lane < SYS_DIM; ++lane) {
      a_row[lane] = a[lane][cycle];
      b_row[lane] = b[cycle][lane];
    }
    step_and_check(name, &dut, ref, a_row, b_row, cycle);
  }

  for (int cycle = SYS_DIM; cycle < CHAIN_TOTAL_STEPS; ++cycle) {
    Row zero{};
    step_and_check(name, &dut, ref, zero, zero, cycle);
  }

  TEST_PASS(name);
}

void test_random_seeded() {
  const char* name = "systolic_array_chained_random_seeded";
  Vsystolic_array dut;
  ChainedArrayRef ref;
  reset(&dut, ref);

  std::mt19937 rng(123456);
  std::uniform_int_distribution<int> dist(-128, 127);
  int8_t a[16][16] = {};
  int8_t b[16][16] = {};

  for (int i = 0; i < 16; ++i) {
    for (int k = 0; k < 16; ++k)
      a[i][k] = static_cast<int8_t>(dist(rng));
  }
  for (int k = 0; k < 16; ++k) {
    for (int j = 0; j < 16; ++j)
      b[k][j] = static_cast<int8_t>(dist(rng));
  }

  for (int cycle = 0; cycle < SYS_DIM; ++cycle) {
    Row a_row{};
    Row b_row{};
    for (int lane = 0; lane < SYS_DIM; ++lane) {
      a_row[lane] = a[lane][cycle];
      b_row[lane] = b[cycle][lane];
    }
    step_and_check(name, &dut, ref, a_row, b_row, cycle);
  }

  for (int cycle = SYS_DIM; cycle < CHAIN_TOTAL_STEPS; ++cycle) {
    Row zero{};
    step_and_check(name, &dut, ref, zero, zero, cycle);
  }

  Row zero{};
  step_and_check(name, &dut, ref, zero, zero, CHAIN_TOTAL_STEPS);
  TEST_PASS(name);
}

}  // namespace

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);

  test_single_impulse_timing();
  test_identity_stream();
  test_signed_extremes();
  test_random_seeded();

  std::printf("\n%d / %d tests passed\n", tests_pass, tests_run);
  return (tests_pass == tests_run) ? 0 : 1;
}
