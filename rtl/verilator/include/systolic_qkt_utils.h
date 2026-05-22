// Shared helpers for R4-split test_systolic_qkt_{basic,replay,padded,history}.cpp.
//
// R4 (2026-05-22): the 3,672-LOC test_systolic_qkt.cpp was split into 4 topic-
// based translation units. Helpers that the original file shared across its
// 31 tests live here. The header opens an anonymous namespace at file scope —
// each TU including it gets an internal-linkage copy (one per binary, since
// the four split files build into four separate Verilator-wrapped binaries).
//
// Layout below matches the original file's L1-L817 verbatim (prelude +
// helpers), with the trailing `}  // namespace` from L3633 inserted here so
// the anonymous namespace is closed within the header.

#ifndef RTL_VERILATOR_SYSTOLIC_QKT_UTILS_H
#define RTL_VERILATOR_SYSTOLIC_QKT_UTILS_H

// Focused native regressions for the QK^T debug split.

#include "Vtaccel_top.h"
#include "Vtaccel_top___024root.h"
#include "verilated.h"
#include "systolic_debug_artifacts.h"
#include "systolic_window_trace.h"
#include "testbench.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using tbutil::SimHarness;
using tbutil::sram_read_bytes;
using tbutil::sram_write_bytes;
constexpr int BUF_ABUF_ID = tbutil::BUF_ABUF_ID;
constexpr int BUF_WBUF_ID = tbutil::BUF_WBUF_ID;
constexpr int BUF_ACCUM_ID = tbutil::BUF_ACCUM_ID;

#include "test_runner.h"  // sibling under include/ (R3); supplies inline tests_run/tests_pass + TEST_PASS/TEST_FAIL.

namespace {

using QStrip16x64 = std::array<std::array<int8_t, 64>, 16>;
using Key208x64 = std::array<std::array<int8_t, 64>, 208>;
using KeyT64x208 = std::array<std::array<int8_t, 208>, 64>;
constexpr int ST_IDLE = 0;

int32_t read_accum_wide(Vtaccel_top* dut, int dst_off, int row_idx, int col_idx, int cols);
void expect_clean_halt(const char* name, Vtaccel_top* dut);

std::vector<uint8_t> flatten_qstrip_row_major(const QStrip16x64& q) {
  std::vector<uint8_t> out(16 * 64);
  for (int r = 0; r < 16; ++r) {
    for (int c = 0; c < 64; ++c)
      out[size_t(r) * 64 + size_t(c)] = static_cast<uint8_t>(q[r][c]);
  }
  return out;
}

std::vector<uint8_t> flatten_key_row_major(const Key208x64& k) {
  std::vector<uint8_t> out(208 * 64);
  for (int r = 0; r < 208; ++r) {
    for (int c = 0; c < 64; ++c)
      out[size_t(r) * 64 + size_t(c)] = static_cast<uint8_t>(k[r][c]);
  }
  return out;
}

std::vector<uint8_t> flatten_keyt_row_major(const KeyT64x208& kt) {
  std::vector<uint8_t> out(64 * 208);
  for (int r = 0; r < 64; ++r) {
    for (int c = 0; c < 208; ++c)
      out[size_t(r) * 208 + size_t(c)] = static_cast<uint8_t>(kt[r][c]);
  }
  return out;
}

std::vector<uint8_t> read_binary_file(const std::string& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    TEST_FAIL("qkt_exact_state_replay", ("could not open " + path).c_str());
  return std::vector<uint8_t>(
      std::istreambuf_iterator<char>(stream),
      std::istreambuf_iterator<char>());
}

std::string read_text_file(const std::string& path) {
  std::ifstream stream(path);
  if (!stream)
    TEST_FAIL("qkt_exact_state_replay", ("could not open " + path).c_str());
  return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

int extract_json_int(const std::string& text, const std::string& key) {
  const std::string marker = "\"" + key + "\"";
  const size_t key_pos = text.find(marker);
  if (key_pos == std::string::npos)
    TEST_FAIL("qkt_exact_state_replay", ("missing metadata key " + key).c_str());
  const size_t colon = text.find(':', key_pos + marker.size());
  if (colon == std::string::npos)
    TEST_FAIL("qkt_exact_state_replay", ("malformed metadata for key " + key).c_str());
  size_t value_pos = colon + 1;
  while (value_pos < text.size() &&
         (text[value_pos] == ' ' || text[value_pos] == '\n' || text[value_pos] == '\r' || text[value_pos] == '\t'))
    ++value_pos;
  size_t value_end = value_pos;
  if (value_end < text.size() && text[value_end] == '-')
    ++value_end;
  while (value_end < text.size() && text[value_end] >= '0' && text[value_end] <= '9')
    ++value_end;
  if (value_end == value_pos)
    TEST_FAIL("qkt_exact_state_replay", ("metadata value is not an integer for key " + key).c_str());
  return std::stoi(text.substr(value_pos, value_end - value_pos));
}

std::vector<uint8_t> pad_i8_rows(
    const std::vector<uint8_t>& logical_bytes,
    int logical_rows,
    int logical_cols,
    int padded_rows,
    int padded_cols) {
  if (logical_cols > padded_cols || logical_rows > padded_rows)
    TEST_FAIL("qkt_helper_replay", "invalid INT8 pad dimensions");
  if (logical_bytes.size() != size_t(logical_rows * logical_cols))
    TEST_FAIL("qkt_helper_replay", "unexpected INT8 logical byte count");
  std::vector<uint8_t> padded(size_t(padded_rows) * size_t(padded_cols), uint8_t(0));
  for (int row = 0; row < logical_rows; ++row) {
    std::memcpy(
        padded.data() + size_t(row) * size_t(padded_cols),
        logical_bytes.data() + size_t(row) * size_t(logical_cols),
        size_t(logical_cols));
  }
  return padded;
}

void zero_i8_padded_rows(std::vector<uint8_t>& padded_bytes, int logical_rows, int padded_rows, int padded_cols) {
  if (logical_rows > padded_rows)
    TEST_FAIL("qkt_helper_replay", "invalid INT8 zero-pad dimensions");
  if (padded_bytes.size() != size_t(padded_rows * padded_cols))
    TEST_FAIL("qkt_helper_replay", "unexpected INT8 padded byte count");
  for (int row = logical_rows; row < padded_rows; ++row)
    std::memset(padded_bytes.data() + size_t(row) * size_t(padded_cols), 0, size_t(padded_cols));
}

std::vector<uint8_t> pad_i32_rows(
    const std::vector<uint8_t>& logical_bytes,
    int logical_rows,
    int logical_cols,
    int padded_rows,
    int padded_cols) {
  if (logical_cols > padded_cols || logical_rows > padded_rows)
    TEST_FAIL("qkt_helper_replay", "invalid INT32 pad dimensions");
  if (logical_bytes.size() != size_t(logical_rows * logical_cols * 4))
    TEST_FAIL("qkt_helper_replay", "unexpected INT32 logical byte count");
  std::vector<uint8_t> padded(size_t(padded_rows) * size_t(padded_cols) * 4u, uint8_t(0));
  for (int row = 0; row < logical_rows; ++row) {
    std::memcpy(
        padded.data() + size_t(row) * size_t(padded_cols) * 4u,
        logical_bytes.data() + size_t(row) * size_t(logical_cols) * 4u,
        size_t(logical_cols) * 4u);
  }
  return padded;
}

void expect_accum_i32_prefix(
    const char* name,
    Vtaccel_top* dut,
    const std::vector<uint8_t>& expected_bytes,
    int logical_rows,
    int logical_cols,
    const char* label) {
  const auto* expected = reinterpret_cast<const int32_t*>(expected_bytes.data());
  for (int row = 0; row < logical_rows; ++row) {
    for (int col = 0; col < logical_cols; ++col) {
      int32_t got = read_accum_wide(dut, 0, row, col, logical_cols);
      int32_t exp = expected[size_t(row) * size_t(logical_cols) + size_t(col)];
      if (got != exp) {
        std::fprintf(stderr,
                     "%s mismatch row=%d col=%d got=%d exp=%d\n",
                     label, row, col, got, exp);
        TEST_FAIL(name, "ACCUM replay mismatch");
      }
    }
  }
}

void expect_int8_matrix_prefix(
    const char* name,
    Vtaccel_top* dut,
    int buf_id,
    int offset_units,
    const std::vector<uint8_t>& expected_bytes,
    int logical_rows,
    int logical_cols,
    int padded_rows,
    int padded_cols,
    const char* label) {
  auto observed = sram_read_bytes(dut, buf_id, size_t(offset_units) * 16u, size_t(padded_rows) * size_t(padded_cols));
  auto expected_padded = pad_i8_rows(expected_bytes, logical_rows, logical_cols, padded_rows, padded_cols);
  for (int row = 0; row < logical_rows; ++row) {
    for (int col = 0; col < logical_cols; ++col) {
      uint8_t got = observed[size_t(row) * size_t(padded_cols) + size_t(col)];
      uint8_t exp = expected_padded[size_t(row) * size_t(padded_cols) + size_t(col)];
      if (got != exp) {
        std::fprintf(stderr,
                     "%s mismatch row=%d col=%d got=%d exp=%d\n",
                     label, row, col, int(int8_t(got)), int(int8_t(exp)));
        TEST_FAIL(name, "INT8 replay mismatch");
      }
    }
  }
}

bool microtrace_mode_enabled(const char* mode_name) {
  const char* out_path = std::getenv("RTL_QKT_MICROTRACE_OUT");
  const char* mode = std::getenv("RTL_QKT_MICROTRACE_MODE");
  if (out_path == nullptr || out_path[0] == '\0')
    return false;
  if (mode == nullptr || mode[0] == '\0')
    return false;
  return std::string(mode) == mode_name;
}

bool artifact_mode_allowed(const char* mode_name) {
  const char* mode = std::getenv("RTL_QKT_MICROTRACE_MODE");
  if (mode == nullptr || mode[0] == '\0')
    return true;
  return std::string(mode) == mode_name;
}

void maybe_write_microtrace(const char* mode_name, const tbutil::SystolicWindowTrace& trace) {
  if (!microtrace_mode_enabled(mode_name))
    return;
  const char* out_path = std::getenv("RTL_QKT_MICROTRACE_OUT");
  std::ofstream stream(out_path, std::ios::binary);
  if (!stream)
    TEST_FAIL(mode_name, ("could not open microtrace output " + std::string(out_path)).c_str());
  stream << tbutil::systolic_window_trace_to_json(trace);
}

void maybe_write_accum_write_log(const char* mode_name, const tbutil::AccumWriteLog& log) {
  const char* out_path = std::getenv("RTL_QKT_ACCUM_WRITE_LOG_OUT");
  if (out_path == nullptr || out_path[0] == '\0')
    return;
  if (!artifact_mode_allowed(mode_name))
    return;
  std::ofstream stream(out_path, std::ios::binary);
  if (!stream)
    TEST_FAIL(mode_name,
              ("could not open accum write log output " + std::string(out_path)).c_str());
  stream << tbutil::accum_write_log_to_json(log);
}

void maybe_write_sram_write_log(const char* mode_name, const tbutil::SramWriteLog& log) {
  const char* out_path = std::getenv("RTL_QKT_SRAM_WRITE_LOG_OUT");
  if (out_path == nullptr || out_path[0] == '\0')
    return;
  const char* mode = std::getenv("RTL_QKT_MICROTRACE_MODE");
  if (mode != nullptr && mode[0] != '\0' && std::string(mode) != mode_name)
    return;
  std::ofstream stream(out_path, std::ios::binary);
  if (!stream)
    TEST_FAIL(mode_name,
              ("could not open SRAM write log output " + std::string(out_path)).c_str());
  stream << tbutil::sram_write_log_to_json(log);
}

void maybe_write_hidden_snapshot(const char* mode_name, const tbutil::SystolicHiddenSnapshot& snapshot) {
  const char* out_path = std::getenv("RTL_QKT_HIDDEN_SNAPSHOT_OUT");
  if (out_path == nullptr || out_path[0] == '\0')
    return;
  if (!artifact_mode_allowed(mode_name))
    return;
  std::ofstream stream(out_path, std::ios::binary);
  if (!stream)
    TEST_FAIL(mode_name,
              ("could not open hidden snapshot output " + std::string(out_path)).c_str());
  stream << tbutil::hidden_snapshot_to_json(snapshot);
}

template <typename NegedgeObserver, typename CycleObserver>
void replay_start_with_debug(
    SimHarness& sim,
    NegedgeObserver&& observe_negedge,
    CycleObserver&& observe_cycle) {
  sim.dut->start = 1;
  tick_with_negedge_observer(
      sim.dut.get(),
      sim.dram,
      std::forward<NegedgeObserver>(observe_negedge));
  sim.dut->start = 0;
  observe_cycle();
}

template <typename NegedgeObserver, typename CycleObserver>
void replay_step_with_debug(
    SimHarness& sim,
    NegedgeObserver&& observe_negedge,
    CycleObserver&& observe_cycle) {
  tick_with_negedge_observer(
      sim.dut.get(),
      sim.dram,
      std::forward<NegedgeObserver>(observe_negedge));
  observe_cycle();
}

bool sram_log_contains_row(
    const tbutil::SramWriteLog& log,
    const char* writer_source,
    uint64_t issue_pc,
    const char* buf_name,
    uint32_t row) {
  for (const auto& rec : log.records) {
    if (rec.writer_source == writer_source &&
        rec.issue_pc == issue_pc &&
        rec.buf_name == buf_name &&
        rec.row == row) {
      return true;
    }
  }
  return false;
}

std::vector<int32_t> capture_accum_strip_i32(
    Vtaccel_top* dut,
    int dst_off_units,
    int rows,
    int cols,
    int mem_cols) {
  std::vector<int32_t> values(size_t(rows) * size_t(cols), 0);
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col)
      values[size_t(row) * size_t(cols) + size_t(col)] =
          read_accum_wide(dut, dst_off_units, row, col, mem_cols);
  }
  return values;
}

std::vector<int64_t> capture_buffer_strip_i8(
    Vtaccel_top* dut,
    int buf_id,
    int offset_units,
    int rows,
    int cols) {
  auto observed = sram_read_bytes(
      dut,
      buf_id,
      size_t(offset_units) * 16u,
      size_t(rows) * size_t(cols));
  std::vector<int64_t> values(size_t(rows) * size_t(cols), 0);
  for (size_t idx = 0; idx < observed.size(); ++idx)
    values[idx] = int64_t(int8_t(observed[idx]));
  return values;
}

std::vector<int64_t> capture_abuf_strip_i8(
    Vtaccel_top* dut,
    int offset_units,
    int rows,
    int cols) {
  return capture_buffer_strip_i8(dut, BUF_ABUF_ID, offset_units, rows, cols);
}

std::vector<int64_t> widen_i32_values(const std::vector<int32_t>& values) {
  std::vector<int64_t> widened(values.size(), 0);
  for (size_t idx = 0; idx < values.size(); ++idx)
    widened[idx] = int64_t(values[idx]);
  return widened;
}

struct MatrixCheckpoint {
  std::string key;
  std::string dtype;
  int rows = 0;
  int cols = 0;
  int row_start = 0;
  std::vector<int64_t> values;
};

void maybe_write_matrix_checkpoints(
    const char* mode_name,
    const std::string& node_prefix,
    int strip_row_start,
    const std::vector<MatrixCheckpoint>& checkpoints) {
  const char* out_path = std::getenv("RTL_QKT_CHECKPOINTS_OUT");
  if (out_path == nullptr || out_path[0] == '\0')
    return;
  if (!artifact_mode_allowed(mode_name))
    return;
  if (checkpoints.empty())
    return;

  auto write_matrix = [&](std::ofstream& stream,
                          const MatrixCheckpoint& checkpoint) {
    stream << "  \"" << checkpoint.key << "\": {\n";
    stream << "    \"dtype\": \"" << checkpoint.dtype << "\",\n";
    stream << "    \"shape\": [" << checkpoint.rows << ", " << checkpoint.cols << "],\n";
    stream << "    \"row_start\": " << checkpoint.row_start << ",\n";
    stream << "    \"values\": [\n";
    for (int row = 0; row < checkpoint.rows; ++row) {
      stream << "      [";
      for (int col = 0; col < checkpoint.cols; ++col) {
        if (col != 0)
          stream << ", ";
        stream << checkpoint.values[size_t(row) * size_t(checkpoint.cols) + size_t(col)];
      }
      stream << "]";
      if (row + 1 != checkpoint.rows)
        stream << ",";
      stream << "\n";
    }
    stream << "    ]\n";
    stream << "  }";
  };

  std::ofstream stream(out_path, std::ios::binary);
  if (!stream)
    TEST_FAIL(mode_name,
              ("could not open QK^T checkpoints output " + std::string(out_path)).c_str());
  stream << "{\n";
  stream << "  \"mode\": \"" << mode_name << "\",\n";
  stream << "  \"node_prefix\": \"" << node_prefix << "\",\n";
  stream << "  \"strip_row_start\": " << strip_row_start << ",\n";
  for (size_t idx = 0; idx < checkpoints.size(); ++idx) {
    write_matrix(stream, checkpoints[idx]);
    if (idx + 1 != checkpoints.size())
      stream << ",";
    stream << "\n";
  }
  stream << "\n}\n";
}

void maybe_write_qkt_checkpoints(
    const char* mode_name,
    const std::string& node_prefix,
    int strip_row_start,
    int cols,
    const std::vector<int32_t>& accum_pre_matmul,
    const std::vector<int32_t>& qkt_output) {
  maybe_write_matrix_checkpoints(
      mode_name,
      node_prefix,
      strip_row_start,
      {
          MatrixCheckpoint{
              "accum_pre_matmul",
              "int32",
              int(accum_pre_matmul.size() / size_t(cols)),
              cols,
              strip_row_start,
              widen_i32_values(accum_pre_matmul),
          },
          MatrixCheckpoint{
              "qkt_output",
              "int32",
              int(qkt_output.size() / size_t(cols)),
              cols,
              strip_row_start,
              widen_i32_values(qkt_output),
          },
      });
}

struct ProjectionReplayResult {
  bool exact_valid = false;
  bool exact_match = false;
  bool clean_valid = false;
  bool clean_match = false;
};

ProjectionReplayResult g_projection_replay_results[3];

int projection_result_index(const std::string& proj_name) {
  if (proj_name == "query")
    return 0;
  if (proj_name == "key")
    return 1;
  if (proj_name == "value")
    return 2;
  TEST_FAIL("qkt_projection_replay", ("unknown projection result key " + proj_name).c_str());
}

void maybe_write_projection_replay_results() {
  const char* out_path = std::getenv("RTL_QKT_PROJECTION_REPLAY_RESULTS_OUT");
  if (out_path == nullptr || out_path[0] == '\0')
    return;
  std::ofstream stream(out_path, std::ios::binary);
  if (!stream)
    TEST_FAIL("qkt_projection_replay",
              ("could not open projection replay results output " + std::string(out_path)).c_str());
  stream << "{\n";
  const char* proj_names[3] = {"query", "key", "value"};
  for (int i = 0; i < 3; ++i) {
    const auto& result = g_projection_replay_results[i];
    stream << "  \"" << proj_names[i] << "\": {";
    bool wrote = false;
    if (result.exact_valid) {
      stream << "\"exact_padded_match\": " << (result.exact_match ? "true" : "false");
      wrote = true;
    }
    if (result.clean_valid) {
      if (wrote)
        stream << ", ";
      stream << "\"clean_padded_match\": " << (result.clean_match ? "true" : "false");
    }
    stream << "}";
    if (i != 2)
      stream << ",";
    stream << "\n";
  }
  stream << "}\n";
}

void record_projection_replay_result(const char* proj_name, bool clean_mode, bool match) {
  int idx = projection_result_index(std::string(proj_name));
  if (clean_mode) {
    g_projection_replay_results[idx].clean_valid = true;
    g_projection_replay_results[idx].clean_match = match;
  } else {
    g_projection_replay_results[idx].exact_valid = true;
    g_projection_replay_results[idx].exact_match = match;
  }
  maybe_write_projection_replay_results();
}

struct Ln1ReplayFixture {
  std::string base;
  std::string metadata_text;
  std::vector<uint8_t> input_bytes;
  std::vector<uint8_t> output_bytes;
  std::vector<uint8_t> gamma_bytes;
  std::vector<uint8_t> beta_bytes;
  std::vector<uint8_t> gamma_beta_bytes;
  int input_off_units = 0;
  int output_off_units = 0;
  int gamma_beta_off_units = 0;
  int sreg_base = 0;
  int in_scale_fp16 = 0;
  int out_scale_fp16 = 0;
  int gamma_dram_offset = 0;
  int beta_dram_offset = 0;
  int rows = 0;
  int cols = 0;
  int m_tiles = 0;
  int n_tiles = 0;
};

Ln1ReplayFixture load_ln1_replay_fixture(const char* name) {
  const char* replay_dir = std::getenv("RTL_QKT_REPLAY_DIR");
  if (replay_dir == nullptr || replay_dir[0] == '\0') {
    std::printf("SKIP: %s (set RTL_QKT_REPLAY_DIR to enable)\n", name);
    return {};
  }

  Ln1ReplayFixture fixture;
  fixture.base = std::string(replay_dir);
  fixture.metadata_text = read_text_file(fixture.base + "/replay_metadata.json");
  fixture.input_bytes = read_binary_file(fixture.base + "/ln1_input_padded.raw");
  fixture.output_bytes = read_binary_file(fixture.base + "/ln1_output_padded.raw");
  fixture.gamma_bytes = read_binary_file(fixture.base + "/ln1_gamma.raw");
  fixture.beta_bytes = read_binary_file(fixture.base + "/ln1_beta.raw");
  fixture.gamma_beta_bytes = fixture.gamma_bytes;
  fixture.gamma_beta_bytes.insert(
      fixture.gamma_beta_bytes.end(),
      fixture.beta_bytes.begin(),
      fixture.beta_bytes.end());

  fixture.input_off_units = extract_json_int(fixture.metadata_text, "ln1_input_padded_offset_units");
  fixture.output_off_units = extract_json_int(fixture.metadata_text, "ln1_output_padded_offset_units");
  fixture.gamma_beta_off_units = extract_json_int(fixture.metadata_text, "ln1_gamma_beta_wbuf_offset_units");
  fixture.sreg_base = extract_json_int(fixture.metadata_text, "ln1_sreg_base");
  fixture.in_scale_fp16 = extract_json_int(fixture.metadata_text, "ln1_in_scale_fp16");
  fixture.out_scale_fp16 = extract_json_int(fixture.metadata_text, "ln1_out_scale_fp16");
  fixture.gamma_dram_offset = extract_json_int(fixture.metadata_text, "ln1_gamma_dram_offset");
  fixture.beta_dram_offset = extract_json_int(fixture.metadata_text, "ln1_beta_dram_offset");
  fixture.rows = extract_json_int(fixture.metadata_text, "ln1_input_padded_rows");
  fixture.cols = extract_json_int(fixture.metadata_text, "ln1_input_padded_cols");
  fixture.m_tiles = fixture.rows / 16;
  fixture.n_tiles = fixture.cols / 16;

  if (fixture.rows != 208 || fixture.cols != 192)
    TEST_FAIL(name, "unexpected LayerNorm replay shape");
  if (fixture.input_bytes.size() != size_t(fixture.rows * fixture.cols))
    TEST_FAIL(name, "unexpected ln1_input_padded.raw size");
  if (fixture.output_bytes.size() != size_t(fixture.rows * fixture.cols))
    TEST_FAIL(name, "unexpected ln1_output_padded.raw size");
  if (fixture.gamma_bytes.size() != size_t(fixture.cols * 2) ||
      fixture.beta_bytes.size() != size_t(fixture.cols * 2))
    TEST_FAIL(name, "unexpected LayerNorm gamma/beta payload size");
  if (fixture.gamma_beta_bytes.size() % 16u != 0u)
    TEST_FAIL(name, "unexpected packed LayerNorm gamma/beta alignment");

  return fixture;
}

int32_t read_accum_wide(Vtaccel_top* dut, int dst_off, int row_idx, int col_idx, int cols) {
  auto* root = dut->rootp;
  const int words_per_row = cols / 4;
  const int grp = col_idx / 4;
  const int lane = col_idx % 4;
  const int row = dst_off + row_idx * words_per_row + grp;
  uint32_t word = root->taccel_top__DOT__u_sram__DOT__u_accum__DOT__mem[row][lane];
  return static_cast<int32_t>(word);
}

int pad_dim16(int dim) {
  return (dim + 15) & ~15;
}

void run_projection_bias_replay(const char* name, const char* replay_dir, const char* proj_name) {
  const std::string base(replay_dir);
  const std::string proj(proj_name);
  auto metadata_text = read_text_file(base + "/replay_metadata.json");
  auto accum_pre_bias_bytes = read_binary_file(base + "/" + proj + "_accum_pre_bias.raw");
  auto bias_input_bytes = read_binary_file(base + "/" + proj + "_bias_input.raw");
  auto accum_post_bias_bytes = read_binary_file(base + "/" + proj + "_accum_post_bias.raw");

  const int rows = extract_json_int(metadata_text, proj + "_accum_pre_bias_rows");
  const int cols = extract_json_int(metadata_text, proj + "_accum_pre_bias_cols");
  const int bias_off_units = extract_json_int(metadata_text, proj + "_bias_input_offset_units");
  const int act_cols = extract_json_int(metadata_text, proj + "_act_input_cols");
  const int padded_rows = pad_dim16(rows);
  const int padded_cols = cols;

  SimHarness s;
  sram_write_bytes(
      s.dut.get(),
      BUF_ACCUM_ID,
      0,
      pad_i32_rows(accum_pre_bias_bytes, rows, cols, padded_rows, padded_cols));
  sram_write_bytes(s.dut.get(), BUF_WBUF_ID, size_t(bias_off_units) * 16u, bias_input_bytes);

  s.load({
      insn::CONFIG_TILE(padded_rows / 16, padded_cols / 16, act_cols / 16),
      insn::VADD(BUF_ACCUM_ID, 0, BUF_WBUF_ID, bias_off_units, BUF_ACCUM_ID, 0, 0, 0),
      insn::HALT(),
  });
  s.run(500000);
  expect_clean_halt(name, s.dut.get());
  expect_accum_i32_prefix(name, s.dut.get(), accum_post_bias_bytes, rows, cols, "projection bias replay");
  TEST_PASS(name);
}

void run_projection_requant_replay(const char* name, const char* replay_dir, const char* proj_name) {
  const std::string base(replay_dir);
  const std::string proj(proj_name);
  auto metadata_text = read_text_file(base + "/replay_metadata.json");
  auto accum_post_bias_bytes = read_binary_file(base + "/" + proj + "_accum_post_bias.raw");
  auto output_bytes = read_binary_file(base + "/" + proj + "_output.raw");

  const int rows = extract_json_int(metadata_text, proj + "_accum_post_bias_rows");
  const int cols = extract_json_int(metadata_text, proj + "_accum_post_bias_cols");
  const int output_off_units = extract_json_int(metadata_text, proj + "_output_offset_units");
  const int scale_fp16 = extract_json_int(metadata_text, proj + "_requant_scale_fp16");
  const int act_cols = extract_json_int(metadata_text, proj + "_act_input_cols");
  const int padded_rows = pad_dim16(rows);
  const int padded_cols = cols;

  SimHarness s;
  sram_write_bytes(
      s.dut.get(),
      BUF_ACCUM_ID,
      0,
      pad_i32_rows(accum_post_bias_bytes, rows, cols, padded_rows, padded_cols));

  s.load({
      insn::CONFIG_TILE(padded_rows / 16, padded_cols / 16, act_cols / 16),
      insn::SET_SCALE(0, uint16_t(scale_fp16), 0),
      insn::REQUANT(BUF_ACCUM_ID, 0, BUF_ABUF_ID, output_off_units, 0, 0),
      insn::HALT(),
  });
  s.run(500000);
  expect_clean_halt(name, s.dut.get());
  expect_int8_matrix_prefix(
      name,
      s.dut.get(),
      BUF_ABUF_ID,
      output_off_units,
      output_bytes,
      rows,
      cols,
      padded_rows,
      padded_cols,
      "projection requant replay");
  TEST_PASS(name);
}

void run_projection_padded_bias_replay(const char* name, const char* replay_dir, const char* proj_name) {
  const std::string base(replay_dir);
  const std::string proj(proj_name);
  auto metadata_text = read_text_file(base + "/replay_metadata.json");
  auto accum_pre_bias_bytes = read_binary_file(base + "/" + proj + "_accum_pre_bias_padded.raw");
  auto bias_input_bytes = read_binary_file(base + "/" + proj + "_bias_input.raw");
  auto accum_post_bias_bytes = read_binary_file(base + "/" + proj + "_accum_post_bias_padded.raw");

  const int rows = extract_json_int(metadata_text, proj + "_accum_pre_bias_padded_rows");
  const int cols = extract_json_int(metadata_text, proj + "_accum_pre_bias_padded_cols");
  const int bias_off_units = extract_json_int(metadata_text, proj + "_bias_input_offset_units");
  const int act_cols = extract_json_int(metadata_text, proj + "_act_input_cols");

  SimHarness s;
  sram_write_bytes(s.dut.get(), BUF_ACCUM_ID, 0, accum_pre_bias_bytes);
  sram_write_bytes(s.dut.get(), BUF_WBUF_ID, size_t(bias_off_units) * 16u, bias_input_bytes);

  s.load({
      insn::CONFIG_TILE(rows / 16, cols / 16, act_cols / 16),
      insn::VADD(BUF_ACCUM_ID, 0, BUF_WBUF_ID, bias_off_units, BUF_ACCUM_ID, 0, 0, 0),
      insn::HALT(),
  });
  s.run(500000);
  expect_clean_halt(name, s.dut.get());
  expect_accum_i32_prefix(name, s.dut.get(), accum_post_bias_bytes, rows, cols, "projection padded bias replay");
  TEST_PASS(name);
}

void run_projection_padded_requant_replay(const char* name, const char* replay_dir, const char* proj_name) {
  const std::string base(replay_dir);
  const std::string proj(proj_name);
  auto metadata_text = read_text_file(base + "/replay_metadata.json");
  auto accum_post_bias_bytes = read_binary_file(base + "/" + proj + "_accum_post_bias_padded.raw");
  auto output_bytes = read_binary_file(base + "/" + proj + "_output_padded.raw");

  const int rows = extract_json_int(metadata_text, proj + "_accum_post_bias_padded_rows");
  const int cols = extract_json_int(metadata_text, proj + "_accum_post_bias_padded_cols");
  const int output_off_units = extract_json_int(metadata_text, proj + "_output_offset_units");
  const int scale_fp16 = extract_json_int(metadata_text, proj + "_requant_scale_fp16");
  const int act_cols = extract_json_int(metadata_text, proj + "_act_input_cols");

  SimHarness s;
  sram_write_bytes(s.dut.get(), BUF_ACCUM_ID, 0, accum_post_bias_bytes);

  s.load({
      insn::CONFIG_TILE(rows / 16, cols / 16, act_cols / 16),
      insn::SET_SCALE(0, uint16_t(scale_fp16), 0),
      insn::REQUANT(BUF_ACCUM_ID, 0, BUF_ABUF_ID, output_off_units, 0, 0),
      insn::HALT(),
  });
  s.run(500000);
  expect_clean_halt(name, s.dut.get());
  expect_int8_matrix_prefix(
      name,
      s.dut.get(),
      BUF_ABUF_ID,
      output_off_units,
      output_bytes,
      rows,
      cols,
      rows,
      cols,
      "projection padded requant replay");
  TEST_PASS(name);
}

void run_projection_padded_matmul_replay(
    const char* name,
    const char* replay_dir,
    const char* proj_name,
    bool clean_padded_input) {
  const std::string base(replay_dir);
  const std::string proj(proj_name);
  auto metadata_text = read_text_file(base + "/replay_metadata.json");
  auto act_input_bytes = read_binary_file(base + "/" + proj + "_act_input_padded.raw");
  auto weight_input_bytes = read_binary_file(base + "/" + proj + "_projection_weight_input.raw");
  auto accum_pre_bias_bytes = read_binary_file(
      base + "/" + proj + (clean_padded_input ? "_accum_pre_bias_padded_golden.raw" : "_accum_pre_bias_padded.raw"));

  const int act_rows = extract_json_int(metadata_text, proj + "_act_input_padded_rows");
  const int act_cols = extract_json_int(metadata_text, proj + "_act_input_padded_cols");
  const int logical_rows = extract_json_int(metadata_text, proj + "_act_input_rows");
  const int weight_rows = extract_json_int(metadata_text, proj + "_weight_input_rows");
  const int weight_cols = extract_json_int(metadata_text, proj + "_weight_input_cols");
  const int act_off_units = extract_json_int(metadata_text, proj + "_act_input_offset_units");
  const int weight_off_units = extract_json_int(metadata_text, proj + "_weight_input_offset_units");
  const int accum_rows = extract_json_int(metadata_text, proj + "_accum_pre_bias_padded_rows");
  const int accum_cols = extract_json_int(metadata_text, proj + "_accum_pre_bias_padded_cols");

  if (act_rows != accum_rows)
    TEST_FAIL(name, "act-input and accum padded rows do not match");
  if (act_cols != weight_rows)
    TEST_FAIL(name, "act-input K dimension does not match weight rows");
  if (weight_cols != accum_cols)
    TEST_FAIL(name, "weight columns do not match accum columns");
  if (act_input_bytes.size() != size_t(act_rows * act_cols))
    TEST_FAIL(name, "unexpected padded act_input size");
  if (weight_input_bytes.size() != size_t(weight_rows * weight_cols))
    TEST_FAIL(name, "unexpected projection weight size");
  if (accum_pre_bias_bytes.size() != size_t(accum_rows * accum_cols * 4))
    TEST_FAIL(name, "unexpected padded accum_pre_bias size");

  if (clean_padded_input)
    zero_i8_padded_rows(act_input_bytes, logical_rows, act_rows, act_cols);

  SimHarness s;
  sram_write_bytes(s.dut.get(), BUF_ABUF_ID, size_t(act_off_units) * 16u, act_input_bytes);
  sram_write_bytes(s.dut.get(), BUF_WBUF_ID, size_t(weight_off_units) * 16u, weight_input_bytes);

  s.load({
      insn::CONFIG_TILE(act_rows / 16, accum_cols / 16, act_cols / 16),
      insn::MATMUL(BUF_ABUF_ID, act_off_units, BUF_WBUF_ID, weight_off_units, BUF_ACCUM_ID, 0, 0, 0),
      insn::SYNC(0b010),
      insn::HALT(),
  });
  s.run(2000000);
  expect_clean_halt(name, s.dut.get());
  expect_accum_i32_prefix(
      name,
      s.dut.get(),
      accum_pre_bias_bytes,
      accum_rows,
      accum_cols,
      clean_padded_input ? "projection clean-padded matmul replay" : "projection exact padded matmul replay");
  record_projection_replay_result(proj_name, clean_padded_input, true);
  TEST_PASS(name);
}

void matmul_ref_16x64x208(const QStrip16x64& q, const KeyT64x208& kt, int32_t (&acc)[16][208]) {
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 208; ++j) {
      int32_t sum = 0;
      for (int k = 0; k < 64; ++k)
        sum += int32_t(q[i][k]) * int32_t(kt[k][j]);
      acc[i][j] = sum;
    }
  }
}

void expect_clean_halt(const char* name, Vtaccel_top* dut) {
  if (!dut->done || dut->fault)
    TEST_FAIL(name, "did not halt cleanly");
}
}  // namespace

#endif  // RTL_VERILATOR_SYSTOLIC_QKT_UTILS_H
