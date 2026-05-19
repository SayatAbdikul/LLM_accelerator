// #114 reproducer: RTL ACCUM-snapshot readback fidelity.
//
// The gen-2 cosim --snapshot-request path (run_program.cpp) returns garbage
// for ACCUM captures (buf_id==BUF_ACCUM, dtype "int32") even though the
// datapath is proven 0-ULP exact and the direct-memory read_accum_32x32 used
// by test_systolic_chained reads ACCUM correctly. Two unconfirmed theories:
//   (a) TIMING  — run_program captures at a SYNC/instruction retire that
//       happens before the asynchronous systolic ST_DRAIN_WR has settled
//       ACCUM, so tbutil::accum_read_logical_i32 reads stale/zero state;
//   (b) LAYOUT  — accum_read_logical_i32's m_stride/n_tile/offset_units
//       rearrange disagrees with the ST_DRAIN_WR physical addressing.
//
// This reproducer pins the mechanism with a KNOWN-EXACT contrast (the trusted
// read_accum_32x32 == integer-matmul golden), replicating run_program's exact
// retire-observe loop (tick_with_negedge_observer; obs_retire_pulse_w/pc_w
// read post-posedge) and capturing ACCUM via the SAME tbutil::
// accum_read_logical_i32 run_program calls, at several anchor points:
//   snap@matmul   : at the MATMUL retire (ACCUM NOT yet drained)
//   snap@sync     : at the systolic-draining SYNC(0b010) retire (drained)
//   snap@sync+1   : one cycle later
//   post-halt-log : accum_read_logical_i32 after HALT (definitively settled)
//   GT            : read_accum_32x32 after HALT (trusted; == golden)
//
// Discriminator: GT==golden validates the instrument. If post-halt-log==golden
// then the layout formula is correct => the bug is TIMING/anchor (snap@matmul
// wrong, snap@sync correct). If post-halt-log!=golden it is a LAYOUT bug.
//
// Part 1 (this commit) is a DIAGNOSTIC (prints the verdict, only the
// instrument-validity GT==golden is a hard assert) — same reproducer-first
// discipline as #116's B1. Part 3 flips snap-path-vs-GT to a hard assertion.

#include "Vtaccel_top.h"
#include "Vtaccel_top___024root.h"
#include "verilated.h"
#include "include/systolic_test_utils.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
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

constexpr int OP_SYNC   = 0x02;
constexpr int OP_MATMUL = 0x0A;

// One captured ACCUM grid (logical row-major int32), variable size.
struct Grid {
  int rows = 0, cols = 0;
  std::vector<int32_t> v;
  int32_t at(int i, int j) const { return v[size_t(i) * cols + j]; }
  void resize(int r, int c) { rows = r; cols = c; v.assign(size_t(r) * c, 0); }
};

// Convert the logical row-major LE-int32 byte buffer that
// tbutil::accum_read_logical_i32 returns into a Grid.
Grid grid_from_logical_bytes(const std::vector<uint8_t>& b, int rows, int cols) {
  Grid g;
  g.resize(rows, cols);
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j) {
      const size_t o = (size_t(i) * cols + j) * 4u;
      int32_t w;
      std::memcpy(&w, &b[o], 4);
      g.v[size_t(i) * cols + j] = w;
    }
  return g;
}

int mismatch_count(const Grid& a, const Grid& b) {
  int n = 0;
  for (int i = 0; i < a.rows; ++i)
    for (int j = 0; j < a.cols; ++j)
      if (a.at(i, j) != b.at(i, j)) ++n;
  return n;
}

// Run `prog` with run_program.cpp's EXACT loop, capturing ACCUM via
// tbutil::accum_read_logical_i32(dut, off=0, mem_cols=N, N, N) at the named
// anchors. Faithful timing: obs_retire_* read AFTER tick_with_negedge_observer
// (post-posedge), identical to run_program's process_current_cycle.
struct CaptureSet {
  Grid snap_at_matmul;       // raw capture at MATMUL retire (pre-fix garbage)
  Grid snap_at_sync;         // at the systolic-draining SYNC retire
  Grid snap_at_sync_plus1;
  Grid snap_drain_gated;     // #114 FIX MODEL: armed at MATMUL retire, captured
                             // at the first sys_busy==0 (mirrors run_program's
                             // wait_sys_idle ACCUM deferral exactly)
  bool got_matmul = false, got_sync = false, got_sync1 = false;
  bool got_drain_gated = false;
};

CaptureSet run_and_capture(Sim& s, const std::vector<uint64_t>& prog,
                           uint64_t matmul_pc, uint64_t sync_pc, int N) {
  CaptureSet cs;
  s.load_program(prog);
  s.dut->start = 1;
  tick(s.dut.get(), s.dram);
  s.dut->start = 0;

  auto* root = s.dut->rootp;
  bool arm_sync_plus1 = false;
  bool drain_gate_armed = false;  // #114 fix model: armed at MATMUL retire
  const int timeout = 1200000;
  for (int i = 0; i < timeout; ++i) {
    tick(s.dut.get(), s.dram);
    // run_program reads these post-posedge (process_current_cycle).
    const bool retire_valid = root->taccel_top__DOT__obs_retire_pulse_w;
    const uint64_t retire_pc = root->taccel_top__DOT__obs_retire_pc_w;
    const bool sys_busy = root->taccel_top__DOT__sys_busy;

    if (arm_sync_plus1) {  // exactly one cycle after the SYNC retire
      cs.snap_at_sync_plus1 = grid_from_logical_bytes(
          tbutil::accum_read_logical_i32(s.dut.get(), 0, N, N, N), N, N);
      cs.got_sync1 = true;
      arm_sync_plus1 = false;
    }
    if (retire_valid && retire_pc == matmul_pc && !cs.got_matmul) {
      cs.snap_at_matmul = grid_from_logical_bytes(
          tbutil::accum_read_logical_i32(s.dut.get(), 0, N, N, N), N, N);
      cs.got_matmul = true;
      // #114 FIX MODEL: run_program defers an ACCUM capture armed at an anchor
      // retire until sys_busy==0. Mirror that exactly: if busy now, arm and
      // capture at the first idle cycle; if already idle, capture immediately.
      if (sys_busy) {
        drain_gate_armed = true;
      } else {
        cs.snap_drain_gated = cs.snap_at_matmul;
        cs.got_drain_gated = true;
      }
    }
    if (drain_gate_armed && !sys_busy && !cs.got_drain_gated) {
      cs.snap_drain_gated = grid_from_logical_bytes(
          tbutil::accum_read_logical_i32(s.dut.get(), 0, N, N, N), N, N);
      cs.got_drain_gated = true;
      drain_gate_armed = false;
    }
    if (retire_valid && retire_pc == sync_pc && !cs.got_sync) {
      cs.snap_at_sync = grid_from_logical_bytes(
          tbutil::accum_read_logical_i32(s.dut.get(), 0, N, N, N), N, N);
      cs.got_sync = true;
      arm_sync_plus1 = true;
    }
    if (s.dut->done || s.dut->fault) break;
  }
  return cs;
}

void diag_one(const char* tag, const Grid& golden, const Grid& got) {
  const int n = mismatch_count(golden, got);
  std::printf("  %-16s mismatch vs golden = %d / %d   sample[0][0]=%d "
              "[0][%d]=%d [%d][%d]=%d\n",
              tag, n, golden.rows * golden.cols, got.at(0, 0),
              golden.cols - 1, got.at(0, golden.cols - 1),
              golden.rows - 1, golden.cols - 1,
              got.at(golden.rows - 1, golden.cols - 1));
}

// 32x32 multitile (V1 known-exact, same data as test_matmul_multitile_2x2x2).
void diag_accum_snapshot_32x32() {
  const char* name = "accum_snapshot_readback_32x32_multitile";
  Sim s;
  int8_t a[32][32] = {};
  int8_t b[32][32] = {};
  int32_t exp[32][32] = {};
  std::vector<uint64_t> prog;
  for (int i = 0; i < 32; ++i)
    for (int j = 0; j < 32; ++j) {
      a[i][j] = static_cast<int8_t>(((i * 7 + j * 5 + 3) % 11) - 5);
      b[i][j] = static_cast<int8_t>(((i * 3 + j * 9 + 1) % 13) - 6);
    }
  prepare_logical_32x32(s.dram, prog, a, b, 0x1A0000, 0x1C0000);
  matmul_ref_32(a, b, exp);

  prog.push_back(insn::CONFIG_TILE(2, 2, 2));
  const uint64_t matmul_pc = prog.size();
  prog.push_back(insn::MATMUL(BUF_ABUF_ID, 0, BUF_WBUF_ID, 0, BUF_ACCUM_ID, 0, 0, 0));
  const uint64_t sync_pc = prog.size();
  prog.push_back(insn::SYNC(0b010));
  prog.push_back(insn::HALT());

  CaptureSet cs = run_and_capture(s, prog, matmul_pc, sync_pc, 32);
  if (!s.dut->done || s.dut->fault)
    TEST_FAIL(name, "RTL did not cleanly halt");

  Grid golden; golden.resize(32, 32);
  for (int i = 0; i < 32; ++i)
    for (int j = 0; j < 32; ++j) golden.v[i * 32 + j] = exp[i][j];

  // GT: trusted direct read post-halt — the known-exact contrast.
  Grid gt; gt.resize(32, 32);
  for (int i = 0; i < 32; ++i)
    for (int j = 0; j < 32; ++j)
      gt.v[i * 32 + j] = read_accum_32x32(s.dut.get(), 0, i, j);

  // post-halt logical: the snapshot read path, but on settled ACCUM.
  Grid post = grid_from_logical_bytes(
      tbutil::accum_read_logical_i32(s.dut.get(), 0, 32, 32, 32), 32, 32);

  std::printf("[%s]\n", name);
  if (mismatch_count(golden, gt) != 0)
    TEST_FAIL(name, "INSTRUMENT INVALID: trusted read_accum_32x32 != "
                    "integer-matmul golden (cannot trust this reproducer)");
  std::printf("  GT (read_accum_32x32, post-halt)  == golden  [instrument OK]\n");
  diag_one("post-halt-logical", golden, post);
  if (cs.got_matmul) diag_one("snap@matmul", golden, cs.snap_at_matmul);
  if (cs.got_sync)   diag_one("snap@sync",   golden, cs.snap_at_sync);
  if (cs.got_sync1)  diag_one("snap@sync+1", golden, cs.snap_at_sync_plus1);

  const bool layout_ok = mismatch_count(golden, post) == 0;
  const bool sync_ok   = cs.got_sync && mismatch_count(golden, cs.snap_at_sync) == 0;
  const bool matmul_bad = cs.got_matmul && mismatch_count(golden, cs.snap_at_matmul) != 0;
  if (layout_ok && sync_ok && matmul_bad)
    std::printf("  VERDICT: TIMING — accum_read_logical_i32 is correct on a "
                "settled ACCUM (post-halt & at the draining SYNC), garbage at "
                "the MATMUL retire. #114 = capture before systolic drain "
                "completes (real cosim anchors ACCUM nodes at PCs that retire "
                "pre-drain). Fix = ACCUM-capture drain-complete gate.\n");
  else if (!layout_ok)
    std::printf("  VERDICT: LAYOUT — accum_read_logical_i32 disagrees with the "
                "trusted read even on a fully-settled (post-halt) ACCUM. "
                "#114 = a layout/offset formula bug in accum_read_logical_i32. "
                "Fix = correct the m_stride/n_tile/offset mapping.\n");
  else
    std::printf("  VERDICT: INCONCLUSIVE — layout_ok=%d sync_ok=%d "
                "matmul_bad=%d; investigate before any fix.\n",
                layout_ok, sync_ok, matmul_bad);

  // #114 HARD REGRESSION: the run_program fix model — an ACCUM capture armed
  // at the anchor retire, deferred to the first sys_busy==0 — must be
  // byte-exact to golden. Pre-fix the raw snap@matmul is all-zeros (printed
  // above as failure context); the drain-gated capture is the fix invariant.
  if (!cs.got_drain_gated)
    TEST_FAIL(name, "drain-gated ACCUM capture never fired");
  if (mismatch_count(golden, cs.snap_drain_gated) != 0) {
    std::fprintf(stderr, "drain-gated snapshot != golden (%d/%d); raw "
                 "snap@matmul mismatch=%d — #114 fix invariant broken\n",
                 mismatch_count(golden, cs.snap_drain_gated),
                 golden.rows * golden.cols,
                 cs.got_matmul ? mismatch_count(golden, cs.snap_at_matmul) : -1);
    TEST_FAIL(name, "drain-gated ACCUM snapshot not byte-exact to golden");
  }
  std::printf("  drain-gated      mismatch vs golden = 0 / %d   [#114 FIX OK]\n",
              golden.rows * golden.cols);
  TEST_PASS(name);
}

// 16x16 single-tile (mem_cols=16; accum_read_logical_i32 vs read_accum_ij).
void diag_accum_snapshot_16x16() {
  const char* name = "accum_snapshot_readback_16x16_single";
  Sim s;
  int8_t a[16][16] = {};
  int8_t b[16][16] = {};
  int32_t exp[16][16] = {};
  std::vector<uint64_t> prog;
  for (int i = 0; i < 16; ++i)
    for (int j = 0; j < 16; ++j) {
      a[i][j] = static_cast<int8_t>(((i * 5 + j * 3 + 2) % 9) - 4);
      b[i][j] = static_cast<int8_t>(((i * 2 + j * 7 + 1) % 11) - 5);
    }
  prepare_logical_16x16(s.dram, prog, a, b, 0x1A0000, 0x1C0000);
  matmul_ref(a, b, exp);

  prog.push_back(insn::CONFIG_TILE(1, 1, 1));
  const uint64_t matmul_pc = prog.size();
  prog.push_back(insn::MATMUL(BUF_ABUF_ID, 0, BUF_WBUF_ID, 0, BUF_ACCUM_ID, 0, 0, 0));
  const uint64_t sync_pc = prog.size();
  prog.push_back(insn::SYNC(0b010));
  prog.push_back(insn::HALT());

  CaptureSet cs = run_and_capture(s, prog, matmul_pc, sync_pc, 16);
  if (!s.dut->done || s.dut->fault)
    TEST_FAIL(name, "RTL did not cleanly halt");

  Grid golden; golden.resize(16, 16);
  for (int i = 0; i < 16; ++i)
    for (int j = 0; j < 16; ++j) golden.v[i * 16 + j] = exp[i][j];
  Grid gt; gt.resize(16, 16);
  for (int i = 0; i < 16; ++i)
    for (int j = 0; j < 16; ++j)
      gt.v[i * 16 + j] = read_accum_ij(s.dut.get(), 0, i, j);
  Grid post = grid_from_logical_bytes(
      tbutil::accum_read_logical_i32(s.dut.get(), 0, 16, 16, 16), 16, 16);

  std::printf("[%s]\n", name);
  if (mismatch_count(golden, gt) != 0)
    TEST_FAIL(name, "INSTRUMENT INVALID: trusted read_accum_ij != golden");
  std::printf("  GT (read_accum_ij, post-halt)     == golden  [instrument OK]\n");
  diag_one("post-halt-logical", golden, post);
  if (cs.got_matmul) diag_one("snap@matmul", golden, cs.snap_at_matmul);
  if (cs.got_sync)   diag_one("snap@sync",   golden, cs.snap_at_sync);
  if (cs.got_sync1)  diag_one("snap@sync+1", golden, cs.snap_at_sync_plus1);
  if (!cs.got_drain_gated)
    TEST_FAIL(name, "drain-gated ACCUM capture never fired");
  if (mismatch_count(golden, cs.snap_drain_gated) != 0) {
    std::fprintf(stderr, "drain-gated snapshot != golden (%d/%d) — #114 fix "
                 "invariant broken\n",
                 mismatch_count(golden, cs.snap_drain_gated),
                 golden.rows * golden.cols);
    TEST_FAIL(name, "drain-gated ACCUM snapshot not byte-exact to golden");
  }
  std::printf("  drain-gated      mismatch vs golden = 0 / %d   [#114 FIX OK]\n",
              golden.rows * golden.cols);
  TEST_PASS(name);
}

}  // namespace

int main() {
  Verilated::traceEverOn(false);
  std::printf("--- #114 ACCUM-snapshot readback regression "
              "(reproducer + hard fix-invariant gate) ---\n");
  diag_accum_snapshot_32x32();
  diag_accum_snapshot_16x16();
  std::printf("\n%d / %d tests passed\n", tests_pass, tests_run);
  return (tests_pass == tests_run) ? 0 : 1;
}
