// Focused native regressions for the QK^T debug split.

#include "Vtaccel_top.h"
#include "Vtaccel_top___024root.h"
#include "verilated.h"
#include "include/systolic_debug_artifacts.h"
#include "include/systolic_window_trace.h"
#include "include/testbench.h"

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

static int tests_run = 0;
static int tests_pass = 0;

#define TEST_PASS(name) do { \
  std::printf("PASS: %s\n", name); tests_pass++; tests_run++; \
} while (0)

#define TEST_FAIL(name, msg) do { \
  std::fprintf(stderr, "FAIL: %s - %s\n", name, msg); std::exit(1); \
} while (0)

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

void test_qkt_transpose_then_matmul_exact_replay() {
  const char* name = "qkt_transpose_then_matmul_exact_replay";
  const char* replay_dir = std::getenv("RTL_QKT_REPLAY_DIR");
  if (replay_dir == nullptr || replay_dir[0] == '\0') {
    std::printf("SKIP: %s (set RTL_QKT_REPLAY_DIR to enable)\n", name);
    return;
  }

  const std::string base(replay_dir);
  auto query_bytes = read_binary_file(base + "/query_input.raw");
  auto key_t_bytes = read_binary_file(base + "/key_transposed.raw");
  auto golden_qkt_bytes = read_binary_file(base + "/golden_qkt.raw");
  if (query_bytes.size() != 16u * 64u)
    TEST_FAIL(name, "unexpected query_input.raw size");
  if (key_t_bytes.size() != 64u * 208u)
    TEST_FAIL(name, "unexpected key_transposed.raw size");
  if (golden_qkt_bytes.size() != 16u * 197u * sizeof(int32_t))
    TEST_FAIL(name, "unexpected golden_qkt.raw size");

  std::vector<uint8_t> key_bytes(208u * 64u, uint8_t(0));
  for (int r = 0; r < 208; ++r) {
    for (int c = 0; c < 64; ++c)
      key_bytes[size_t(r) * 64u + size_t(c)] = key_t_bytes[size_t(c) * 208u + size_t(r)];
  }

  SimHarness s;
  constexpr int Q_OFF_UNITS = 4992;
  sram_write_bytes(s.dut.get(), BUF_ABUF_ID, 0, key_bytes);
  sram_write_bytes(s.dut.get(), BUF_ABUF_ID, size_t(Q_OFF_UNITS) * 16, query_bytes);

  s.load({
      insn::BUF_COPY(BUF_ABUF_ID, 0, BUF_WBUF_ID, 0, (208 * 64) / 16, 13, 1),
      insn::CONFIG_TILE(1, 13, 4),
      insn::MATMUL(BUF_ABUF_ID, Q_OFF_UNITS, BUF_WBUF_ID, 0, BUF_ACCUM_ID, 0, 0, 0),
      insn::SYNC(0b010),
      insn::HALT(),
  });
  s.run(2000000);
  expect_clean_halt(name, s.dut.get());

  const auto* golden = reinterpret_cast<const int32_t*>(golden_qkt_bytes.data());
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 197; ++j) {
      int32_t got = read_accum_wide(s.dut.get(), 0, i, j, 208);
      int32_t exp = golden[size_t(i) * 197 + size_t(j)];
      if (got != exp) {
        std::fprintf(stderr,
                     "transpose+matmul replay mismatch row=%d col=%d got=%d exp=%d",
                     i, j, got, exp);
        if (i == 1 && j == 0)
          std::fprintf(stderr, " [known baseline coordinate]");
        std::fprintf(stderr, "\n");
        TEST_FAIL(name, "combined transpose + QK^T replay mismatch");
      }
    }
  }

  TEST_PASS(name);
}

void test_qkt_prev_key_matmul_then_transpose_matmul_exact_replay() {
  const char* name = "qkt_prev_key_matmul_then_transpose_matmul_exact_replay";
  const char* replay_dir = std::getenv("RTL_QKT_REPLAY_DIR");
  if (replay_dir == nullptr || replay_dir[0] == '\0') {
    std::printf("SKIP: %s (set RTL_QKT_REPLAY_DIR to enable)\n", name);
    return;
  }

  const std::string base(replay_dir);
  auto query_bytes = read_binary_file(base + "/query_input.raw");
  auto key_padded_bytes = read_binary_file(base + "/key_padded_input.raw");
  auto key_act_bytes = read_binary_file(base + "/key_projection_act_input.raw");
  auto key_weight_bytes = read_binary_file(base + "/key_projection_weight_input.raw");
  auto golden_qkt_bytes = read_binary_file(base + "/golden_qkt.raw");
  auto metadata_text = read_text_file(base + "/replay_metadata.json");

  if (query_bytes.size() != 16u * 64u)
    TEST_FAIL(name, "unexpected query_input.raw size");
  if (key_padded_bytes.size() != 208u * 64u)
    TEST_FAIL(name, "unexpected key_padded_input.raw size");
  if (key_act_bytes.size() != 197u * 192u)
    TEST_FAIL(name, "unexpected key_projection_act_input.raw size");
  if (key_weight_bytes.size() != 192u * 64u)
    TEST_FAIL(name, "unexpected key_projection_weight_input.raw size");
  if (golden_qkt_bytes.size() != 16u * 197u * sizeof(int32_t))
    TEST_FAIL(name, "unexpected golden_qkt.raw size");

  const int query_off_units = extract_json_int(metadata_text, "query_input_offset_units");
  const int key_padded_off_units = extract_json_int(metadata_text, "key_padded_input_offset_units");
  const int key_act_off_units = extract_json_int(metadata_text, "key_projection_act_offset_units");
  const int key_weight_off_units = extract_json_int(metadata_text, "key_projection_weight_offset_units");

  SimHarness s;
  sram_write_bytes(s.dut.get(), BUF_ABUF_ID, size_t(key_act_off_units) * 16u, key_act_bytes);
  sram_write_bytes(s.dut.get(), BUF_WBUF_ID, size_t(key_weight_off_units) * 16u, key_weight_bytes);
  sram_write_bytes(s.dut.get(), BUF_ABUF_ID, size_t(key_padded_off_units) * 16u, key_padded_bytes);
  sram_write_bytes(s.dut.get(), BUF_ABUF_ID, size_t(query_off_units) * 16u, query_bytes);

  s.load({
      insn::CONFIG_TILE(13, 4, 12),
      insn::MATMUL(BUF_ABUF_ID, key_act_off_units, BUF_WBUF_ID, key_weight_off_units, BUF_ACCUM_ID, 0, 0, 0),
      insn::SYNC(0b010),
      insn::BUF_COPY(BUF_ABUF_ID, key_padded_off_units, BUF_WBUF_ID, 0, (208 * 64) / 16, 13, 1),
      insn::SYNC(0b001),
      insn::CONFIG_TILE(1, 13, 4),
      insn::MATMUL(BUF_ABUF_ID, query_off_units, BUF_WBUF_ID, 0, BUF_ACCUM_ID, 0, 0, 0),
      insn::SYNC(0b010),
      insn::HALT(),
  });
  s.run(5000000);
  expect_clean_halt(name, s.dut.get());

  const auto* golden = reinterpret_cast<const int32_t*>(golden_qkt_bytes.data());
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 197; ++j) {
      int32_t got = read_accum_wide(s.dut.get(), 0, i, j, 208);
      int32_t exp = golden[size_t(i) * 197 + size_t(j)];
      if (got != exp) {
        std::fprintf(stderr,
                     "prev-key-fragment replay mismatch row=%d col=%d got=%d exp=%d",
                     i, j, got, exp);
        if (i == 1 && j == 0)
          std::fprintf(stderr, " [known baseline coordinate]");
        std::fprintf(stderr,
                     " key_act_off=%d key_weight_off=%d key_padded_off=%d query_off=%d\n",
                     key_act_off_units, key_weight_off_units, key_padded_off_units, query_off_units);
        TEST_FAIL(name, "previous key MATMUL + QK^T fragment mismatch");
      }
    }
  }

  TEST_PASS(name);
}

void test_qkt_prev_key_fragment_sync_release() {
  const char* name = "qkt_prev_key_fragment_sync_release";
  const char* replay_dir = std::getenv("RTL_QKT_REPLAY_DIR");
  if (replay_dir == nullptr || replay_dir[0] == '\0') {
    std::printf("SKIP: %s (set RTL_QKT_REPLAY_DIR to enable)\n", name);
    return;
  }

  const std::string base(replay_dir);
  auto query_bytes = read_binary_file(base + "/query_input.raw");
  auto key_padded_bytes = read_binary_file(base + "/key_padded_input.raw");
  auto key_act_bytes = read_binary_file(base + "/key_projection_act_input.raw");
  auto key_weight_bytes = read_binary_file(base + "/key_projection_weight_input.raw");
  auto golden_qkt_bytes = read_binary_file(base + "/golden_qkt.raw");
  auto metadata_text = read_text_file(base + "/replay_metadata.json");

  const int query_off_units = extract_json_int(metadata_text, "query_input_offset_units");
  const int key_padded_off_units = extract_json_int(metadata_text, "key_padded_input_offset_units");
  const int key_act_off_units = extract_json_int(metadata_text, "key_projection_act_offset_units");
  const int key_weight_off_units = extract_json_int(metadata_text, "key_projection_weight_offset_units");

  SimHarness s;
  sram_write_bytes(s.dut.get(), BUF_ABUF_ID, size_t(key_act_off_units) * 16u, key_act_bytes);
  sram_write_bytes(s.dut.get(), BUF_WBUF_ID, size_t(key_weight_off_units) * 16u, key_weight_bytes);
  sram_write_bytes(s.dut.get(), BUF_ABUF_ID, size_t(key_padded_off_units) * 16u, key_padded_bytes);
  sram_write_bytes(s.dut.get(), BUF_ABUF_ID, size_t(query_off_units) * 16u, query_bytes);

  s.load({
      insn::CONFIG_TILE(13, 4, 12),
      insn::MATMUL(BUF_ABUF_ID, key_act_off_units, BUF_WBUF_ID, key_weight_off_units, BUF_ACCUM_ID, 0, 0, 0),
      insn::SYNC(0b010),
      insn::BUF_COPY(BUF_ABUF_ID, key_padded_off_units, BUF_WBUF_ID, 0, (208 * 64) / 16, 13, 1),
      insn::SYNC(0b001),
      insn::CONFIG_TILE(1, 13, 4),
      insn::MATMUL(BUF_ABUF_ID, query_off_units, BUF_WBUF_ID, 0, BUF_ACCUM_ID, 0, 0, 0),
      insn::SYNC(0b010),
      insn::HALT(),
  });

  s.start_once();
  auto* r = s.dut->rootp;
  tbutil::SystolicWindowCollector microtrace(5, 7);
  auto observe_microtrace = [&](bool retire_valid, uint64_t retire_pc, int retire_opcode) {
    if (microtrace_mode_enabled(name))
      microtrace.observe(r, retire_valid, retire_pc, retire_opcode);
  };
  observe_microtrace(
      r->taccel_top__DOT__obs_retire_pulse_w,
      r->taccel_top__DOT__obs_retire_pc_w,
      int(r->taccel_top__DOT__obs_retire_opcode_w));
  bool sync_released = false;
  for (int cycle = 0; cycle < 5000000; ++cycle) {
    if (r->taccel_top__DOT__u_ctrl__DOT__pc_reg == 8) {
      sync_released = true;
      break;
    }
    s.step();
    r = s.dut->rootp;
    observe_microtrace(
        r->taccel_top__DOT__obs_retire_pulse_w,
        r->taccel_top__DOT__obs_retire_pc_w,
        int(r->taccel_top__DOT__obs_retire_opcode_w));
    if (s.dut->done || s.dut->fault)
      break;
  }
  if (!sync_released)
    TEST_FAIL(name, "did not reach post-QK^T-SYNC checkpoint");

  const auto* golden = reinterpret_cast<const int32_t*>(golden_qkt_bytes.data());
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 197; ++j) {
      int32_t got = read_accum_wide(s.dut.get(), 0, i, j, 208);
      int32_t exp = golden[size_t(i) * 197 + size_t(j)];
      if (got != exp) {
        std::fprintf(stderr,
                     "sync-release mismatch row=%d col=%d got=%d exp=%d",
                     i, j, got, exp);
        if (i == 1 && j == 0)
          std::fprintf(stderr, " [known baseline coordinate]");
        std::fprintf(stderr, " pc=%d sys_state=%d lane=%d drain_row=%d drain_grp=%d\n",
                     int(r->taccel_top__DOT__u_ctrl__DOT__pc_reg),
                     int(r->taccel_top__DOT__u_systolic__DOT__state),
                     int(r->taccel_top__DOT__u_systolic__DOT__lane_q),
                     int(r->taccel_top__DOT__u_systolic__DOT__drain_row_q),
                     int(r->taccel_top__DOT__u_systolic__DOT__drain_grp_q));
        TEST_FAIL(name, "QK^T result not stable when SYNC released");
      }
    }
  }

  s.run(1000);
  expect_clean_halt(name, s.dut.get());
  maybe_write_microtrace(name, microtrace.finish());
  TEST_PASS(name);
}

void test_qkt_prev_qkv_matmuls_then_qkt_replay() {
  const char* name = "qkt_prev_qkv_matmuls_then_qkt_replay";
  const char* replay_dir = std::getenv("RTL_QKT_REPLAY_DIR");
  if (replay_dir == nullptr || replay_dir[0] == '\0') {
    std::printf("SKIP: %s (set RTL_QKT_REPLAY_DIR to enable)\n", name);
    return;
  }

  const std::string base(replay_dir);
  auto metadata_text = read_text_file(base + "/replay_metadata.json");
  auto query_accum_pre_bias_bytes = read_binary_file(base + "/query_accum_pre_bias.raw");
  auto query_bias_bytes = read_binary_file(base + "/query_bias_input.raw");
  auto key_act_bytes = read_binary_file(base + "/key_projection_act_input.raw");
  auto key_weight_bytes = read_binary_file(base + "/key_projection_weight_input.raw");
  auto key_bias_bytes = read_binary_file(base + "/key_bias_input.raw");
  auto value_act_bytes = read_binary_file(base + "/value_projection_act_input.raw");
  auto value_weight_bytes = read_binary_file(base + "/value_projection_weight_input.raw");
  auto value_bias_bytes = read_binary_file(base + "/value_bias_input.raw");
  auto golden_qkt_bytes = read_binary_file(base + "/golden_qkt.raw");

  const int query_bias_off_units = extract_json_int(metadata_text, "query_bias_input_offset_units");
  const int query_output_off_units = extract_json_int(metadata_text, "query_output_offset_units");
  const int query_off_units = extract_json_int(metadata_text, "query_input_offset_units");
  const int query_scale_fp16 = extract_json_int(metadata_text, "query_requant_scale_fp16");
  const int key_act_off_units = extract_json_int(metadata_text, "key_projection_act_offset_units");
  const int key_weight_off_units = extract_json_int(metadata_text, "key_projection_weight_offset_units");
  const int key_bias_off_units = extract_json_int(metadata_text, "key_bias_input_offset_units");
  const int key_output_off_units = extract_json_int(metadata_text, "key_output_offset_units");
  const int key_padded_off_units = extract_json_int(metadata_text, "key_padded_input_offset_units");
  const int key_scale_fp16 = extract_json_int(metadata_text, "key_requant_scale_fp16");
  const int value_act_off_units = extract_json_int(metadata_text, "value_projection_act_offset_units");
  const int value_weight_off_units = extract_json_int(metadata_text, "value_projection_weight_offset_units");
  const int value_bias_off_units = extract_json_int(metadata_text, "value_bias_input_offset_units");
  const int value_output_off_units = extract_json_int(metadata_text, "value_output_offset_units");
  const int value_scale_fp16 = extract_json_int(metadata_text, "value_requant_scale_fp16");
  const int key_t_off_units = extract_json_int(metadata_text, "key_transposed_offset_units");
  const int query_rows = extract_json_int(metadata_text, "query_accum_pre_bias_rows");
  const int query_cols = extract_json_int(metadata_text, "query_accum_pre_bias_cols");
  const int key_rows = extract_json_int(metadata_text, "key_output_rows");
  const int key_cols = extract_json_int(metadata_text, "key_output_cols");
  const int key_act_rows = extract_json_int(metadata_text, "key_projection_act_rows");
  const int key_act_cols = extract_json_int(metadata_text, "key_projection_act_cols");
  const int value_act_rows = extract_json_int(metadata_text, "value_projection_act_rows");
  const int value_act_cols = extract_json_int(metadata_text, "value_projection_act_cols");

  if (query_output_off_units != query_off_units)
    TEST_FAIL(name, "query output offset does not match QK^T query input offset");
  if (key_output_off_units != key_padded_off_units)
    TEST_FAIL(name, "key output offset does not match padded K input offset");

  const int proj_m_tiles = pad_dim16(key_rows) / 16;
  const int proj_n_tiles = key_cols / 16;
  const int proj_k_tiles = key_act_cols / 16;

  if (query_accum_pre_bias_bytes.size() != size_t(query_rows * query_cols * 4))
    TEST_FAIL(name, "unexpected query accum pre-bias size");
  if (query_bias_bytes.size() != size_t(query_cols * 4))
    TEST_FAIL(name, "unexpected query bias size");
  if (key_act_bytes.size() != size_t(key_act_rows * key_act_cols))
    TEST_FAIL(name, "unexpected key projection activation size");
  if (key_weight_bytes.size() != size_t(key_act_cols * key_cols))
    TEST_FAIL(name, "unexpected key projection weight size");
  if (key_bias_bytes.size() != size_t(key_cols * 4))
    TEST_FAIL(name, "unexpected key bias size");
  if (value_act_bytes.size() != size_t(value_act_rows * value_act_cols))
    TEST_FAIL(name, "unexpected value projection activation size");
  if (value_weight_bytes.size() != size_t(value_act_cols * key_cols))
    TEST_FAIL(name, "unexpected value projection weight size");
  if (value_bias_bytes.size() != size_t(key_cols * 4))
    TEST_FAIL(name, "unexpected value bias size");

  SimHarness s;
  sram_write_bytes(
      s.dut.get(),
      BUF_ACCUM_ID,
      0,
      pad_i32_rows(query_accum_pre_bias_bytes, query_rows, query_cols, pad_dim16(query_rows), query_cols));
  sram_write_bytes(s.dut.get(), BUF_WBUF_ID, size_t(query_bias_off_units) * 16u, query_bias_bytes);
  sram_write_bytes(
      s.dut.get(),
      BUF_ABUF_ID,
      size_t(key_act_off_units) * 16u,
      pad_i8_rows(key_act_bytes, key_act_rows, key_act_cols, pad_dim16(key_act_rows), key_act_cols));
  sram_write_bytes(
      s.dut.get(),
      BUF_ABUF_ID,
      size_t(value_act_off_units) * 16u,
      pad_i8_rows(value_act_bytes, value_act_rows, value_act_cols, pad_dim16(value_act_rows), value_act_cols));

  std::vector<uint64_t> prog(82, insn::NOP());
  prog[1] = insn::CONFIG_TILE(proj_m_tiles, proj_n_tiles, proj_k_tiles);
  prog[40] = insn::VADD(BUF_ACCUM_ID, 0, BUF_WBUF_ID, query_bias_off_units, BUF_ACCUM_ID, 0, 0, 0);
  prog[41] = insn::SET_SCALE(0, uint16_t(query_scale_fp16), 0);
  prog[42] = insn::REQUANT(BUF_ACCUM_ID, 0, BUF_ABUF_ID, query_output_off_units, 0, 0);
  prog[47] = insn::CONFIG_TILE(proj_m_tiles, proj_n_tiles, proj_k_tiles);
  prog[48] = insn::MATMUL(BUF_ABUF_ID, key_act_off_units, BUF_WBUF_ID, key_weight_off_units, BUF_ACCUM_ID, 0, 0, 0);
  prog[49] = insn::SYNC(0b010);
  prog[54] = insn::VADD(BUF_ACCUM_ID, 0, BUF_WBUF_ID, key_bias_off_units, BUF_ACCUM_ID, 0, 0, 0);
  prog[55] = insn::SET_SCALE(1, uint16_t(key_scale_fp16), 0);
  prog[56] = insn::REQUANT(BUF_ACCUM_ID, 0, BUF_ABUF_ID, key_output_off_units, 1, 0);
  prog[61] = insn::CONFIG_TILE(proj_m_tiles, proj_n_tiles, proj_k_tiles);
  prog[62] = insn::MATMUL(BUF_ABUF_ID, value_act_off_units, BUF_WBUF_ID, value_weight_off_units, BUF_ACCUM_ID, 0, 0, 0);
  prog[63] = insn::SYNC(0b010);
  prog[68] = insn::VADD(BUF_ACCUM_ID, 0, BUF_WBUF_ID, value_bias_off_units, BUF_ACCUM_ID, 0, 0, 0);
  prog[69] = insn::SET_SCALE(2, uint16_t(value_scale_fp16), 0);
  prog[70] = insn::REQUANT(BUF_ACCUM_ID, 0, BUF_ABUF_ID, value_output_off_units, 2, 0);
  prog[75] = insn::BUF_COPY(BUF_ABUF_ID, key_padded_off_units, BUF_WBUF_ID, key_t_off_units, (pad_dim16(key_rows) * key_cols) / 16, proj_m_tiles, 1);
  prog[76] = insn::SYNC(0b001);
  prog[77] = insn::CONFIG_TILE(1, 13, 4);
  prog[78] = insn::MATMUL(BUF_ABUF_ID, query_off_units, BUF_WBUF_ID, key_t_off_units, BUF_ACCUM_ID, 0, 0, 0);
  prog[79] = insn::SYNC(0b010);
  prog[81] = insn::HALT();
  s.load(prog);

  s.start_once();
  auto* r = s.dut->rootp;
  tbutil::SystolicWindowCollector microtrace(77, 79);
  tbutil::AccumWriteLogCollector accum_write_log(40, 80);
  tbutil::SystolicHiddenSnapshotCollector hidden_snapshot(77);
  auto observe_microtrace = [&](bool retire_valid, uint64_t retire_pc, int retire_opcode) {
    if (microtrace_mode_enabled(name))
      microtrace.observe(r, retire_valid, retire_pc, retire_opcode);
    accum_write_log.observe(r, retire_valid, retire_pc, retire_opcode);
    hidden_snapshot.observe(r, retire_valid, retire_pc, retire_opcode);
  };
  observe_microtrace(
      r->taccel_top__DOT__obs_retire_pulse_w,
      r->taccel_top__DOT__obs_retire_pc_w,
      int(r->taccel_top__DOT__obs_retire_opcode_w));
  bool loaded_key_weight = false;
  bool loaded_key_bias = false;
  bool loaded_value_weight = false;
  bool loaded_value_bias = false;
  for (int cycle = 0; cycle < 7000000; ++cycle) {
    const int pc = int(r->taccel_top__DOT__u_ctrl__DOT__pc_reg);
    if (pc == 47 && !loaded_key_weight) {
      sram_write_bytes(s.dut.get(), BUF_WBUF_ID, size_t(key_weight_off_units) * 16u, key_weight_bytes);
      loaded_key_weight = true;
    } else if (pc == 54 && !loaded_key_bias) {
      sram_write_bytes(s.dut.get(), BUF_WBUF_ID, size_t(key_bias_off_units) * 16u, key_bias_bytes);
      loaded_key_bias = true;
    } else if (pc == 61 && !loaded_value_weight) {
      sram_write_bytes(s.dut.get(), BUF_WBUF_ID, size_t(value_weight_off_units) * 16u, value_weight_bytes);
      loaded_value_weight = true;
    } else if (pc == 68 && !loaded_value_bias) {
      sram_write_bytes(s.dut.get(), BUF_WBUF_ID, size_t(value_bias_off_units) * 16u, value_bias_bytes);
      loaded_value_bias = true;
    }
    if (s.dut->done || s.dut->fault)
      break;
    s.step();
    r = s.dut->rootp;
    observe_microtrace(
        r->taccel_top__DOT__obs_retire_pulse_w,
        r->taccel_top__DOT__obs_retire_pc_w,
        int(r->taccel_top__DOT__obs_retire_opcode_w));
  }
  expect_clean_halt(name, s.dut.get());
  if (!loaded_key_weight || !loaded_key_bias || !loaded_value_weight || !loaded_value_bias)
    TEST_FAIL(name, "did not reach all helper-tail load checkpoints");

  const auto* golden = reinterpret_cast<const int32_t*>(golden_qkt_bytes.data());
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 197; ++j) {
      int32_t got = read_accum_wide(s.dut.get(), 0, i, j, 208);
      int32_t exp = golden[size_t(i) * 197 + size_t(j)];
      if (got != exp) {
        std::fprintf(stderr,
                     "prev-qkv replay mismatch row=%d col=%d got=%d exp=%d",
                     i, j, got, exp);
        if (i == 1 && j == 0)
          std::fprintf(stderr, " [known baseline coordinate]");
        std::fprintf(stderr,
                     " q_bias=%d k_w=%d k_b=%d v_w=%d v_b=%d pc=%d\n",
                     1,
                     int(loaded_key_weight),
                     int(loaded_key_bias),
                     int(loaded_value_weight),
                     int(loaded_value_bias),
                     int(r->taccel_top__DOT__u_ctrl__DOT__pc_reg));
        TEST_FAIL(name, "QKV history + QK^T fragment mismatch");
      }
    }
  }

  maybe_write_microtrace(name, microtrace.finish());
  maybe_write_accum_write_log(name, accum_write_log.finish());
  maybe_write_hidden_snapshot(name, hidden_snapshot.finish());
  TEST_PASS(name);
}

void test_qkt_prev_qkv_full_padded_history_then_qkt_replay() {
  const char* name = "qkt_prev_qkv_full_padded_history_then_qkt_replay";
  const char* replay_dir = std::getenv("RTL_QKT_REPLAY_DIR");
  if (replay_dir == nullptr || replay_dir[0] == '\0') {
    std::printf("SKIP: %s (set RTL_QKT_REPLAY_DIR to enable)\n", name);
    return;
  }

  const std::string base(replay_dir);
  auto metadata_text = read_text_file(base + "/replay_metadata.json");
  auto ln1_output_bytes = read_binary_file(base + "/ln1_output_padded.raw");
  auto query_weight_bytes = read_binary_file(base + "/query_projection_weight_input.raw");
  auto query_bias_bytes = read_binary_file(base + "/query_bias_input.raw");
  auto query_output_bytes = read_binary_file(base + "/query_output_padded.raw");
  auto key_weight_bytes = read_binary_file(base + "/key_projection_weight_input.raw");
  auto key_bias_bytes = read_binary_file(base + "/key_bias_input.raw");
  auto key_output_bytes = read_binary_file(base + "/key_output_padded.raw");
  auto key_padded_bytes = read_binary_file(base + "/key_padded_input.raw");
  auto value_weight_bytes = read_binary_file(base + "/value_projection_weight_input.raw");
  auto value_bias_bytes = read_binary_file(base + "/value_bias_input.raw");
  auto value_output_bytes = read_binary_file(base + "/value_output_padded.raw");
  auto golden_qkt_bytes = read_binary_file(base + "/golden_qkt.raw");

  const int ln1_output_off_units = extract_json_int(metadata_text, "ln1_output_padded_offset_units");
  const int query_act_off_units = extract_json_int(metadata_text, "query_act_input_padded_offset_units");
  const int key_act_off_units = extract_json_int(metadata_text, "key_act_input_padded_offset_units");
  const int value_act_off_units = extract_json_int(metadata_text, "value_act_input_padded_offset_units");
  const int query_weight_off_units = extract_json_int(metadata_text, "query_weight_input_offset_units");
  const int query_bias_off_units = extract_json_int(metadata_text, "query_bias_input_offset_units");
  const int query_output_off_units = extract_json_int(metadata_text, "query_output_padded_offset_units");
  const int query_input_off_units = extract_json_int(metadata_text, "query_input_offset_units");
  const int query_output_rows = extract_json_int(metadata_text, "query_output_padded_rows");
  const int query_output_cols = extract_json_int(metadata_text, "query_output_padded_cols");
  const int query_scale_fp16 = extract_json_int(metadata_text, "query_requant_scale_fp16");
  const int key_weight_off_units = extract_json_int(metadata_text, "key_weight_input_offset_units");
  const int key_bias_off_units = extract_json_int(metadata_text, "key_bias_input_offset_units");
  const int key_output_off_units = extract_json_int(metadata_text, "key_output_padded_offset_units");
  const int key_padded_off_units = extract_json_int(metadata_text, "key_padded_input_offset_units");
  const int key_output_rows = extract_json_int(metadata_text, "key_output_padded_rows");
  const int key_output_cols = extract_json_int(metadata_text, "key_output_padded_cols");
  const int key_scale_fp16 = extract_json_int(metadata_text, "key_requant_scale_fp16");
  const int value_weight_off_units = extract_json_int(metadata_text, "value_weight_input_offset_units");
  const int value_bias_off_units = extract_json_int(metadata_text, "value_bias_input_offset_units");
  const int value_output_off_units = extract_json_int(metadata_text, "value_output_padded_offset_units");
  const int value_output_rows = extract_json_int(metadata_text, "value_output_padded_rows");
  const int value_output_cols = extract_json_int(metadata_text, "value_output_padded_cols");
  const int value_scale_fp16 = extract_json_int(metadata_text, "value_requant_scale_fp16");
  const int key_t_off_units = extract_json_int(metadata_text, "key_transposed_offset_units");
  const int ln1_rows = query_output_rows;
  const int ln1_cols = extract_json_int(metadata_text, "query_act_input_padded_cols");

  if (ln1_rows != 208 || ln1_cols != 192)
    TEST_FAIL(name, "unexpected ln1_output_padded shape");
  if (ln1_output_bytes.size() != size_t(ln1_rows * ln1_cols))
    TEST_FAIL(name, "unexpected ln1_output_padded.raw size");
  if (query_weight_bytes.size() != size_t(ln1_cols * query_output_cols))
    TEST_FAIL(name, "unexpected query weight size");
  if (key_weight_bytes.size() != size_t(ln1_cols * key_output_cols))
    TEST_FAIL(name, "unexpected key weight size");
  if (value_weight_bytes.size() != size_t(ln1_cols * value_output_cols))
    TEST_FAIL(name, "unexpected value weight size");
  if (query_bias_bytes.size() != size_t(query_output_cols * 4))
    TEST_FAIL(name, "unexpected query bias size");
  if (key_bias_bytes.size() != size_t(key_output_cols * 4))
    TEST_FAIL(name, "unexpected key bias size");
  if (value_bias_bytes.size() != size_t(value_output_cols * 4))
    TEST_FAIL(name, "unexpected value bias size");
  if (query_output_bytes.size() != size_t(query_output_rows * query_output_cols))
    TEST_FAIL(name, "unexpected query_output_padded.raw size");
  if (key_output_bytes.size() != size_t(key_output_rows * key_output_cols))
    TEST_FAIL(name, "unexpected key_output_padded.raw size");
  if (key_padded_bytes.size() != size_t(key_output_rows * key_output_cols))
    TEST_FAIL(name, "unexpected key_padded_input.raw size");
  if (value_output_bytes.size() != size_t(value_output_rows * value_output_cols))
    TEST_FAIL(name, "unexpected value_output_padded.raw size");
  if (golden_qkt_bytes.size() != 16u * 197u * sizeof(int32_t))
    TEST_FAIL(name, "unexpected golden_qkt.raw size");

  if (query_act_off_units != ln1_output_off_units ||
      key_act_off_units != ln1_output_off_units ||
      value_act_off_units != ln1_output_off_units)
    TEST_FAIL(name, "projection activation offsets do not reuse ln1 output");
  if (query_output_off_units != query_input_off_units)
    TEST_FAIL(name, "query output offset does not match QK^T query input offset");
  if (key_output_off_units != key_padded_off_units)
    TEST_FAIL(name, "key output offset does not match padded K input offset");

  const int proj_m_tiles = query_output_rows / 16;
  const int proj_n_tiles = query_output_cols / 16;
  const int proj_k_tiles = ln1_cols / 16;
  const int key_padding_start_row = 197;

  SimHarness s;
  sram_write_bytes(s.dut.get(), BUF_ABUF_ID, size_t(ln1_output_off_units) * 16u, ln1_output_bytes);
  sram_write_bytes(s.dut.get(), BUF_WBUF_ID, size_t(query_weight_off_units) * 16u, query_weight_bytes);

  std::vector<uint64_t> prog(82, insn::NOP());
  prog[32] = insn::CONFIG_TILE(proj_m_tiles, proj_n_tiles, proj_k_tiles);
  prog[34] = insn::MATMUL(BUF_ABUF_ID, query_act_off_units, BUF_WBUF_ID, query_weight_off_units, BUF_ACCUM_ID, 0, 0, 0);
  prog[35] = insn::SYNC(0b010);
  prog[40] = insn::VADD(BUF_ACCUM_ID, 0, BUF_WBUF_ID, query_bias_off_units, BUF_ACCUM_ID, 0, 0, 0);
  prog[41] = insn::SET_SCALE(0, uint16_t(query_scale_fp16), 0);
  prog[42] = insn::REQUANT(BUF_ACCUM_ID, 0, BUF_ABUF_ID, query_output_off_units, 0, 0);
  prog[47] = insn::CONFIG_TILE(proj_m_tiles, proj_n_tiles, proj_k_tiles);
  prog[48] = insn::MATMUL(BUF_ABUF_ID, key_act_off_units, BUF_WBUF_ID, key_weight_off_units, BUF_ACCUM_ID, 0, 0, 0);
  prog[49] = insn::SYNC(0b010);
  prog[54] = insn::VADD(BUF_ACCUM_ID, 0, BUF_WBUF_ID, key_bias_off_units, BUF_ACCUM_ID, 0, 0, 0);
  prog[55] = insn::SET_SCALE(1, uint16_t(key_scale_fp16), 0);
  prog[56] = insn::REQUANT(BUF_ACCUM_ID, 0, BUF_ABUF_ID, key_output_off_units, 1, 0);
  prog[61] = insn::CONFIG_TILE(proj_m_tiles, proj_n_tiles, proj_k_tiles);
  prog[62] = insn::MATMUL(BUF_ABUF_ID, value_act_off_units, BUF_WBUF_ID, value_weight_off_units, BUF_ACCUM_ID, 0, 0, 0);
  prog[63] = insn::SYNC(0b010);
  prog[68] = insn::VADD(BUF_ACCUM_ID, 0, BUF_WBUF_ID, value_bias_off_units, BUF_ACCUM_ID, 0, 0, 0);
  prog[69] = insn::SET_SCALE(2, uint16_t(value_scale_fp16), 0);
  prog[70] = insn::REQUANT(BUF_ACCUM_ID, 0, BUF_ABUF_ID, value_output_off_units, 2, 0);
  prog[75] = insn::BUF_COPY(BUF_ABUF_ID, key_padded_off_units, BUF_WBUF_ID, key_t_off_units, (key_output_rows * key_output_cols) / 16, proj_m_tiles, 1);
  prog[76] = insn::SYNC(0b001);
  prog[77] = insn::CONFIG_TILE(1, 13, 4);
  prog[78] = insn::MATMUL(BUF_ABUF_ID, query_input_off_units, BUF_WBUF_ID, key_t_off_units, BUF_ACCUM_ID, 0, 0, 0);
  prog[79] = insn::SYNC(0b010);
  prog[81] = insn::HALT();
  s.load(prog);

  s.start_once();
  auto* r = s.dut->rootp;
  tbutil::SystolicWindowCollector microtrace(77, 79);
  tbutil::AccumWriteLogCollector accum_write_log(40, 80);
  tbutil::SystolicHiddenSnapshotCollector hidden_snapshot(77);
  auto observe_debug = [&](bool retire_valid, uint64_t retire_pc, int retire_opcode) {
    if (microtrace_mode_enabled(name))
      microtrace.observe(r, retire_valid, retire_pc, retire_opcode);
    accum_write_log.observe(r, retire_valid, retire_pc, retire_opcode);
    hidden_snapshot.observe(r, retire_valid, retire_pc, retire_opcode);
  };
  observe_debug(
      r->taccel_top__DOT__obs_retire_pulse_w,
      r->taccel_top__DOT__obs_retire_pc_w,
      int(r->taccel_top__DOT__obs_retire_opcode_w));

  bool loaded_query_bias = false;
  bool checked_query_output = false;
  bool loaded_key_weight = false;
  bool loaded_key_bias = false;
  bool checked_key_output = false;
  bool loaded_value_weight = false;
  bool loaded_value_bias = false;
  bool checked_value_output = false;
  bool applied_key_zero_mask = false;
  bool checked_key_padded = false;

  for (int cycle = 0; cycle < 7000000; ++cycle) {
    const int pc = int(r->taccel_top__DOT__u_ctrl__DOT__pc_reg);
    if (pc == 40 && !loaded_query_bias) {
      sram_write_bytes(s.dut.get(), BUF_WBUF_ID, size_t(query_bias_off_units) * 16u, query_bias_bytes);
      loaded_query_bias = true;
    } else if (pc == 47) {
      if (!checked_query_output) {
        expect_int8_matrix_prefix(
            name,
            s.dut.get(),
            BUF_ABUF_ID,
            query_output_off_units,
            query_output_bytes,
            query_output_rows,
            query_output_cols,
            query_output_rows,
            query_output_cols,
            "query padded output replay");
        checked_query_output = true;
      }
      if (!loaded_key_weight) {
        sram_write_bytes(s.dut.get(), BUF_WBUF_ID, size_t(key_weight_off_units) * 16u, key_weight_bytes);
        loaded_key_weight = true;
      }
    } else if (pc == 54 && !loaded_key_bias) {
      sram_write_bytes(s.dut.get(), BUF_WBUF_ID, size_t(key_bias_off_units) * 16u, key_bias_bytes);
      loaded_key_bias = true;
    } else if (pc == 61) {
      if (!checked_key_output) {
        expect_int8_matrix_prefix(
            name,
            s.dut.get(),
            BUF_ABUF_ID,
            key_output_off_units,
            key_output_bytes,
            key_output_rows,
            key_output_cols,
            key_output_rows,
            key_output_cols,
            "key padded output replay");
        checked_key_output = true;
      }
      if (!loaded_value_weight) {
        sram_write_bytes(s.dut.get(), BUF_WBUF_ID, size_t(value_weight_off_units) * 16u, value_weight_bytes);
        loaded_value_weight = true;
      }
    } else if (pc == 68 && !loaded_value_bias) {
      sram_write_bytes(s.dut.get(), BUF_WBUF_ID, size_t(value_bias_off_units) * 16u, value_bias_bytes);
      loaded_value_bias = true;
    } else if (pc == 75) {
      if (!checked_value_output) {
        expect_int8_matrix_prefix(
            name,
            s.dut.get(),
            BUF_ABUF_ID,
            value_output_off_units,
            value_output_bytes,
            value_output_rows,
            value_output_cols,
            value_output_rows,
            value_output_cols,
            "value padded output replay");
        checked_value_output = true;
      }
      if (!applied_key_zero_mask) {
        const size_t tail_offset = size_t(key_padding_start_row) * size_t(key_output_cols);
        sram_write_bytes(
            s.dut.get(),
            BUF_ABUF_ID,
            size_t(key_padded_off_units) * 16u + tail_offset,
            std::vector<uint8_t>(key_padded_bytes.begin() + std::ptrdiff_t(tail_offset), key_padded_bytes.end()));
        applied_key_zero_mask = true;
      }
      if (applied_key_zero_mask && !checked_key_padded) {
        expect_int8_matrix_prefix(
            name,
            s.dut.get(),
            BUF_ABUF_ID,
            key_padded_off_units,
            key_padded_bytes,
            key_output_rows,
            key_output_cols,
            key_output_rows,
            key_output_cols,
            "key zero-mask replay");
        checked_key_padded = true;
      }
    }

    if (s.dut->done || s.dut->fault)
      break;
    s.step();
    r = s.dut->rootp;
    observe_debug(
        r->taccel_top__DOT__obs_retire_pulse_w,
        r->taccel_top__DOT__obs_retire_pc_w,
        int(r->taccel_top__DOT__obs_retire_opcode_w));
  }

  expect_clean_halt(name, s.dut.get());
  if (!loaded_query_bias || !checked_query_output)
    TEST_FAIL(name, "did not complete query projection tail replay");
  if (!loaded_key_weight || !loaded_key_bias || !checked_key_output || !applied_key_zero_mask || !checked_key_padded)
    TEST_FAIL(name, "did not complete key projection tail replay");
  if (!loaded_value_weight || !loaded_value_bias || !checked_value_output)
    TEST_FAIL(name, "did not complete value projection tail replay");

  const auto* golden = reinterpret_cast<const int32_t*>(golden_qkt_bytes.data());
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 197; ++j) {
      int32_t got = read_accum_wide(s.dut.get(), 0, i, j, 208);
      int32_t exp = golden[size_t(i) * 197 + size_t(j)];
      if (got != exp) {
        std::fprintf(stderr,
                     "full padded-history replay mismatch row=%d col=%d got=%d exp=%d",
                     i, j, got, exp);
        if (i == 1 && j == 0)
          std::fprintf(stderr, " [known baseline coordinate]");
        std::fprintf(stderr,
                     " q_bias=%d q_out=%d k_w=%d k_b=%d k_out=%d k_mask=%d v_w=%d v_b=%d v_out=%d pc=%d\n",
                     int(loaded_query_bias),
                     int(checked_query_output),
                     int(loaded_key_weight),
                     int(loaded_key_bias),
                     int(checked_key_output),
                     int(checked_key_padded),
                     int(loaded_value_weight),
                     int(loaded_value_bias),
                     int(checked_value_output),
                     int(r->taccel_top__DOT__u_ctrl__DOT__pc_reg));
        TEST_FAIL(name, "full padded QKV history + QK^T fragment mismatch");
      }
    }
  }

  maybe_write_microtrace(name, microtrace.finish());
  maybe_write_accum_write_log(name, accum_write_log.finish());
  maybe_write_hidden_snapshot(name, hidden_snapshot.finish());
  TEST_PASS(name);
}

void test_ln1_preloaded_operands_replay() {
  const char* name = "ln1_preloaded_operands_replay";
  Ln1ReplayFixture fixture = load_ln1_replay_fixture(name);
  if (fixture.base.empty())
    return;

  SimHarness s;
  sram_write_bytes(
      s.dut.get(),
      BUF_ABUF_ID,
      size_t(fixture.input_off_units) * 16u,
      fixture.input_bytes);
  sram_write_bytes(
      s.dut.get(),
      BUF_WBUF_ID,
      size_t(fixture.gamma_beta_off_units) * 16u,
      fixture.gamma_beta_bytes);

  std::vector<uint64_t> prog(6, insn::NOP());
  prog[0] = insn::CONFIG_TILE(fixture.m_tiles, fixture.n_tiles, 1);
  prog[1] = insn::SET_SCALE(fixture.sreg_base, uint16_t(fixture.in_scale_fp16), 0);
  prog[2] = insn::SET_SCALE(fixture.sreg_base + 1, uint16_t(fixture.out_scale_fp16), 0);
  prog[3] = insn::LAYERNORM(
      BUF_ABUF_ID,
      fixture.input_off_units,
      BUF_WBUF_ID,
      fixture.gamma_beta_off_units,
      BUF_ABUF_ID,
      fixture.output_off_units,
      fixture.sreg_base,
      0);
  prog[4] = insn::SYNC(0b100);
  prog[5] = insn::HALT();
  s.load(prog);

  auto* r = s.dut->rootp;
  tbutil::SystolicWindowCollector microtrace(3, 4);
  tbutil::SramWriteLogCollector sram_write_log(0, 4);
  auto observe_debug = [&](bool retire_valid, uint64_t retire_pc, int retire_opcode) {
    if (microtrace_mode_enabled(name))
      microtrace.observe(r, retire_valid, retire_pc, retire_opcode);
    sram_write_log.observe(r, retire_valid, retire_pc, retire_opcode);
  };
  auto observe_cycle = [&]() {
    r = s.dut->rootp;
    observe_debug(
        r->taccel_top__DOT__obs_retire_pulse_w,
        r->taccel_top__DOT__obs_retire_pc_w,
        int(r->taccel_top__DOT__obs_retire_opcode_w));
  };
  auto observe_sram_negedge = [&]() {
    r = s.dut->rootp;
    sram_write_log.observe(
        r,
        r->taccel_top__DOT__obs_retire_pulse_w,
        r->taccel_top__DOT__obs_retire_pc_w,
        int(r->taccel_top__DOT__obs_retire_opcode_w));
  };
  auto fail_with_artifacts = [&](const char* msg) {
    maybe_write_microtrace(name, microtrace.finish());
    maybe_write_sram_write_log(name, sram_write_log.finish());
    TEST_FAIL(name, msg);
  };
  auto expect_output_or_fail = [&]() {
    auto observed = sram_read_bytes(
        s.dut.get(),
        BUF_ABUF_ID,
        size_t(fixture.output_off_units) * 16u,
        fixture.output_bytes.size());
    for (size_t idx = 0; idx < observed.size(); ++idx) {
      if (observed[idx] != fixture.output_bytes[idx]) {
        const int row = int(idx / size_t(fixture.cols));
        const int col = int(idx % size_t(fixture.cols));
        std::fprintf(
            stderr,
            "ln1 preloaded output mismatch row=%d col=%d got=%d exp=%d\n",
            row,
            col,
            int(int8_t(observed[idx])),
            int(int8_t(fixture.output_bytes[idx])));
        fail_with_artifacts("INT8 replay mismatch");
      }
    }
  };

  replay_start_with_debug(s, observe_sram_negedge, observe_cycle);
  for (int cycle = 0; cycle < 1000000; ++cycle) {
    if (s.dut->done || s.dut->fault)
      break;
    replay_step_with_debug(s, observe_sram_negedge, observe_cycle);
  }

  if (s.dut->fault)
    fail_with_artifacts("unexpected fault during LayerNorm preloaded replay");
  if (!s.dut->done)
    fail_with_artifacts("LayerNorm preloaded replay did not halt");
  expect_clean_halt(name, s.dut.get());
  expect_output_or_fail();
  maybe_write_microtrace(name, microtrace.finish());
  maybe_write_sram_write_log(name, sram_write_log.finish());
  TEST_PASS(name);
}

void test_ln1_dma_loaded_operands_replay() {
  const char* name = "ln1_dma_loaded_operands_replay";
  Ln1ReplayFixture fixture = load_ln1_replay_fixture(name);
  if (fixture.base.empty())
    return;

  auto addr_lo = [](int addr) -> int { return int(uint32_t(addr) & ((1u << 28) - 1u)); };
  auto addr_hi = [](int addr) -> int { return int(uint32_t(addr) >> 28); };
  const int gamma_rows = int(fixture.gamma_bytes.size() / 16u);
  const int beta_rows = int(fixture.beta_bytes.size() / 16u);

  SimHarness s;
  s.dram.write_bytes(uint64_t(fixture.gamma_dram_offset), fixture.gamma_bytes.data(), fixture.gamma_bytes.size());
  s.dram.write_bytes(uint64_t(fixture.beta_dram_offset), fixture.beta_bytes.data(), fixture.beta_bytes.size());
  sram_write_bytes(
      s.dut.get(),
      BUF_ABUF_ID,
      size_t(fixture.input_off_units) * 16u,
      fixture.input_bytes);

  std::vector<uint64_t> prog(14, insn::NOP());
  prog[0] = insn::CONFIG_TILE(fixture.m_tiles, fixture.n_tiles, 1);
  prog[1] = insn::SET_SCALE(fixture.sreg_base, uint16_t(fixture.in_scale_fp16), 0);
  prog[2] = insn::SET_SCALE(fixture.sreg_base + 1, uint16_t(fixture.out_scale_fp16), 0);
  prog[3] = insn::SET_ADDR_LO(1, addr_lo(fixture.gamma_dram_offset));
  prog[4] = insn::SET_ADDR_HI(1, addr_hi(fixture.gamma_dram_offset));
  prog[5] = insn::LOAD(BUF_WBUF_ID, fixture.gamma_beta_off_units, gamma_rows, 1, 0);
  prog[6] = insn::SYNC(0b001);
  prog[7] = insn::SET_ADDR_LO(1, addr_lo(fixture.beta_dram_offset));
  prog[8] = insn::SET_ADDR_HI(1, addr_hi(fixture.beta_dram_offset));
  prog[9] = insn::LOAD(BUF_WBUF_ID, fixture.gamma_beta_off_units + gamma_rows, beta_rows, 1, 0);
  prog[10] = insn::SYNC(0b001);
  prog[11] = insn::LAYERNORM(
      BUF_ABUF_ID,
      fixture.input_off_units,
      BUF_WBUF_ID,
      fixture.gamma_beta_off_units,
      BUF_ABUF_ID,
      fixture.output_off_units,
      fixture.sreg_base,
      0);
  prog[12] = insn::SYNC(0b100);
  prog[13] = insn::HALT();
  s.load(prog);

  tbutil::SystolicWindowCollector microtrace(3, 12);
  tbutil::SramWriteLogCollector sram_write_log(0, 12);
  auto* r = s.dut->rootp;
  auto observe_debug = [&](bool retire_valid, uint64_t retire_pc, int retire_opcode) {
    if (microtrace_mode_enabled(name))
      microtrace.observe(r, retire_valid, retire_pc, retire_opcode);
    sram_write_log.observe(r, retire_valid, retire_pc, retire_opcode);
  };
  auto observe_cycle = [&]() {
    r = s.dut->rootp;
    observe_debug(
        r->taccel_top__DOT__obs_retire_pulse_w,
        r->taccel_top__DOT__obs_retire_pc_w,
        int(r->taccel_top__DOT__obs_retire_opcode_w));
  };
  auto observe_sram_negedge = [&]() {
    r = s.dut->rootp;
    sram_write_log.observe(
        r,
        r->taccel_top__DOT__obs_retire_pulse_w,
        r->taccel_top__DOT__obs_retire_pc_w,
        int(r->taccel_top__DOT__obs_retire_opcode_w));
  };
  auto fail_with_artifacts = [&](const char* msg) {
    maybe_write_microtrace(name, microtrace.finish());
    maybe_write_sram_write_log(name, sram_write_log.finish());
    TEST_FAIL(name, msg);
  };
  auto expect_wbuf_rows_or_fail = [&]() {
    auto observed = sram_read_bytes(
        s.dut.get(),
        BUF_WBUF_ID,
        size_t(fixture.gamma_beta_off_units) * 16u,
        fixture.gamma_beta_bytes.size());
    for (size_t idx = 0; idx < observed.size(); ++idx) {
      if (observed[idx] != fixture.gamma_beta_bytes[idx]) {
        const int row = int(idx / 16u);
        const int byte_idx = int(idx % 16u);
        std::fprintf(
            stderr,
            "ln1 dma-loaded WBUF mismatch row=%d byte=%d got=0x%02x exp=0x%02x\n",
            row + fixture.gamma_beta_off_units,
            byte_idx,
            int(observed[idx]),
            int(fixture.gamma_beta_bytes[idx]));
        fail_with_artifacts("WBUF operand image mismatch");
      }
    }
  };
  auto expect_output_or_fail = [&]() {
    auto observed = sram_read_bytes(
        s.dut.get(),
        BUF_ABUF_ID,
        size_t(fixture.output_off_units) * 16u,
        fixture.output_bytes.size());
    for (size_t idx = 0; idx < observed.size(); ++idx) {
      if (observed[idx] != fixture.output_bytes[idx]) {
        const int row = int(idx / size_t(fixture.cols));
        const int col = int(idx % size_t(fixture.cols));
        std::fprintf(
            stderr,
            "ln1 dma-loaded output mismatch row=%d col=%d got=%d exp=%d\n",
            row,
            col,
            int(int8_t(observed[idx])),
            int(int8_t(fixture.output_bytes[idx])));
        fail_with_artifacts("INT8 replay mismatch");
      }
    }
  };

  replay_start_with_debug(s, observe_sram_negedge, observe_cycle);
  for (int cycle = 0; cycle < 1000000; ++cycle) {
    if (s.dut->done || s.dut->fault)
      break;
    replay_step_with_debug(s, observe_sram_negedge, observe_cycle);
  }

  if (s.dut->fault)
    fail_with_artifacts("unexpected fault during LayerNorm DMA replay");
  if (!s.dut->done)
    fail_with_artifacts("LayerNorm DMA replay did not halt");
  expect_clean_halt(name, s.dut.get());
  expect_wbuf_rows_or_fail();
  expect_output_or_fail();
  maybe_write_microtrace(name, microtrace.finish());
  maybe_write_sram_write_log(name, sram_write_log.finish());
  TEST_PASS(name);
}

void test_qkt_prev_ln1_qkv_full_history_then_qkt_replay() {
  const char* name = "qkt_prev_ln1_qkv_full_history_then_qkt_replay";
  const std::string node_prefix = "block0_head0_qkt";
  const char* replay_dir = std::getenv("RTL_QKT_REPLAY_DIR");
  if (replay_dir == nullptr || replay_dir[0] == '\0') {
    std::printf("SKIP: %s (set RTL_QKT_REPLAY_DIR to enable)\n", name);
    return;
  }

  const std::string base(replay_dir);
  auto metadata_text = read_text_file(base + "/replay_metadata.json");
  auto ln1_input_bytes = read_binary_file(base + "/ln1_input_padded.raw");
  auto ln1_output_bytes = read_binary_file(base + "/ln1_output_padded.raw");
  auto ln1_gamma_bytes = read_binary_file(base + "/ln1_gamma.raw");
  auto ln1_beta_bytes = read_binary_file(base + "/ln1_beta.raw");
  auto query_weight_bytes = read_binary_file(base + "/query_projection_weight_input.raw");
  auto query_bias_bytes = read_binary_file(base + "/query_bias_input.raw");
  auto query_output_bytes = read_binary_file(base + "/query_output_padded.raw");
  auto key_weight_bytes = read_binary_file(base + "/key_projection_weight_input.raw");
  auto key_bias_bytes = read_binary_file(base + "/key_bias_input.raw");
  auto key_output_bytes = read_binary_file(base + "/key_output_padded.raw");
  auto key_padded_bytes = read_binary_file(base + "/key_padded_input.raw");
  auto value_weight_bytes = read_binary_file(base + "/value_projection_weight_input.raw");
  auto value_bias_bytes = read_binary_file(base + "/value_bias_input.raw");
  auto value_output_bytes = read_binary_file(base + "/value_output_padded.raw");
  auto golden_qkt_bytes = read_binary_file(base + "/golden_qkt.raw");

  const int ln1_input_off_units = extract_json_int(metadata_text, "ln1_input_padded_offset_units");
  const int ln1_output_off_units = extract_json_int(metadata_text, "ln1_output_padded_offset_units");
  const int ln1_gb_off_units = extract_json_int(metadata_text, "ln1_gamma_beta_wbuf_offset_units");
  const int ln1_sreg_base = extract_json_int(metadata_text, "ln1_sreg_base");
  const int ln1_in_scale_fp16 = extract_json_int(metadata_text, "ln1_in_scale_fp16");
  const int ln1_out_scale_fp16 = extract_json_int(metadata_text, "ln1_out_scale_fp16");
  const int ln1_gamma_dram_offset = extract_json_int(metadata_text, "ln1_gamma_dram_offset");
  const int ln1_beta_dram_offset = extract_json_int(metadata_text, "ln1_beta_dram_offset");
  const int query_weight_dram_offset = extract_json_int(metadata_text, "query_weight_dram_offset");
  const int query_bias_dram_offset = extract_json_int(metadata_text, "query_bias_dram_offset");
  const int key_weight_dram_offset = extract_json_int(metadata_text, "key_weight_dram_offset");
  const int key_bias_dram_offset = extract_json_int(metadata_text, "key_bias_dram_offset");
  const int value_weight_dram_offset = extract_json_int(metadata_text, "value_weight_dram_offset");
  const int value_bias_dram_offset = extract_json_int(metadata_text, "value_bias_dram_offset");
  const int zero_pad_dram_offset = extract_json_int(metadata_text, "zero_pad_dram_offset");
  const int key_zero_pad_tail_bytes = extract_json_int(metadata_text, "key_zero_pad_tail_bytes");
  const int query_act_off_units = extract_json_int(metadata_text, "query_act_input_padded_offset_units");
  const int key_act_off_units = extract_json_int(metadata_text, "key_act_input_padded_offset_units");
  const int value_act_off_units = extract_json_int(metadata_text, "value_act_input_padded_offset_units");
  const int query_weight_off_units = extract_json_int(metadata_text, "query_weight_input_offset_units");
  const int query_bias_off_units = extract_json_int(metadata_text, "query_bias_input_offset_units");
  const int query_output_off_units = extract_json_int(metadata_text, "query_output_padded_offset_units");
  const int query_input_off_units = extract_json_int(metadata_text, "query_input_offset_units");
  const int query_output_rows = extract_json_int(metadata_text, "query_output_padded_rows");
  const int query_output_cols = extract_json_int(metadata_text, "query_output_padded_cols");
  const int query_scale_fp16 = extract_json_int(metadata_text, "query_requant_scale_fp16");
  const int key_weight_off_units = extract_json_int(metadata_text, "key_weight_input_offset_units");
  const int key_bias_off_units = extract_json_int(metadata_text, "key_bias_input_offset_units");
  const int key_output_off_units = extract_json_int(metadata_text, "key_output_padded_offset_units");
  const int key_padded_off_units = extract_json_int(metadata_text, "key_padded_input_offset_units");
  const int key_output_rows = extract_json_int(metadata_text, "key_output_padded_rows");
  const int key_output_cols = extract_json_int(metadata_text, "key_output_padded_cols");
  const int key_scale_fp16 = extract_json_int(metadata_text, "key_requant_scale_fp16");
  const int value_weight_off_units = extract_json_int(metadata_text, "value_weight_input_offset_units");
  const int value_bias_off_units = extract_json_int(metadata_text, "value_bias_input_offset_units");
  const int value_output_off_units = extract_json_int(metadata_text, "value_output_padded_offset_units");
  const int value_output_rows = extract_json_int(metadata_text, "value_output_padded_rows");
  const int value_output_cols = extract_json_int(metadata_text, "value_output_padded_cols");
  const int value_scale_fp16 = extract_json_int(metadata_text, "value_requant_scale_fp16");
  const int key_t_off_units = extract_json_int(metadata_text, "key_transposed_offset_units");
  const int ln1_rows = extract_json_int(metadata_text, "ln1_input_padded_rows");
  const int ln1_cols = extract_json_int(metadata_text, "query_act_input_padded_cols");

  if (query_act_off_units != ln1_output_off_units ||
      key_act_off_units != ln1_output_off_units ||
      value_act_off_units != ln1_output_off_units)
    TEST_FAIL(name, "projection activation offsets do not reuse ln1 output");
  if (query_output_off_units != query_input_off_units)
    TEST_FAIL(name, "query output offset does not match QK^T query input offset");
  if (key_output_off_units != key_padded_off_units)
    TEST_FAIL(name, "key output offset does not match padded K input offset");
  if (ln1_gamma_bytes.size() != size_t(ln1_cols * 2) || ln1_beta_bytes.size() != size_t(ln1_cols * 2))
    TEST_FAIL(name, "unexpected layernorm gamma/beta sizes");
  if (ln1_input_bytes.size() != size_t(ln1_rows * ln1_cols))
    TEST_FAIL(name, "unexpected ln1_input_padded.raw size");
  if (ln1_output_bytes.size() != size_t(ln1_rows * ln1_cols))
    TEST_FAIL(name, "unexpected ln1_output_padded.raw size");
  if (golden_qkt_bytes.size() != 16u * 197u * sizeof(int32_t))
    TEST_FAIL(name, "unexpected golden_qkt.raw size");

  const int proj_m_tiles = query_output_rows / 16;
  const int proj_n_tiles = query_output_cols / 16;
  const int proj_k_tiles = ln1_cols / 16;
  const int key_padding_start_row = 197;
  const int key_padding_tail_off_units = key_padded_off_units + (key_padding_start_row * key_output_cols) / 16;
  auto addr_lo = [](int addr) -> int { return int(uint32_t(addr) & ((1u << 28) - 1u)); };
  auto addr_hi = [](int addr) -> int { return int(uint32_t(addr) >> 28); };

  SimHarness s;
  s.dram.write_bytes(uint64_t(ln1_gamma_dram_offset), ln1_gamma_bytes.data(), ln1_gamma_bytes.size());
  s.dram.write_bytes(uint64_t(ln1_beta_dram_offset), ln1_beta_bytes.data(), ln1_beta_bytes.size());
  s.dram.write_bytes(uint64_t(query_weight_dram_offset), query_weight_bytes.data(), query_weight_bytes.size());
  s.dram.write_bytes(uint64_t(query_bias_dram_offset), query_bias_bytes.data(), query_bias_bytes.size());
  s.dram.write_bytes(uint64_t(key_weight_dram_offset), key_weight_bytes.data(), key_weight_bytes.size());
  s.dram.write_bytes(uint64_t(key_bias_dram_offset), key_bias_bytes.data(), key_bias_bytes.size());
  s.dram.write_bytes(uint64_t(value_weight_dram_offset), value_weight_bytes.data(), value_weight_bytes.size());
  s.dram.write_bytes(uint64_t(value_bias_dram_offset), value_bias_bytes.data(), value_bias_bytes.size());
  if (key_zero_pad_tail_bytes > 0) {
    std::vector<uint8_t> zero_pad(size_t(key_zero_pad_tail_bytes), uint8_t(0));
    s.dram.write_bytes(uint64_t(zero_pad_dram_offset), zero_pad.data(), zero_pad.size());
  }
  sram_write_bytes(s.dut.get(), BUF_ABUF_ID, size_t(ln1_input_off_units) * 16u, ln1_input_bytes);

  std::vector<uint64_t> prog(82, insn::NOP());
  prog[16] = insn::CONFIG_TILE(proj_m_tiles, proj_k_tiles, 1);
  prog[17] = insn::SET_SCALE(ln1_sreg_base, uint16_t(ln1_in_scale_fp16), 0);
  prog[18] = insn::SET_SCALE(ln1_sreg_base + 1, uint16_t(ln1_out_scale_fp16), 0);
  prog[19] = insn::SET_ADDR_LO(1, addr_lo(ln1_gamma_dram_offset));
  prog[20] = insn::SET_ADDR_HI(1, addr_hi(ln1_gamma_dram_offset));
  prog[21] = insn::LOAD(BUF_WBUF_ID, ln1_gb_off_units, int(ln1_gamma_bytes.size() / 16u), 1, 0);
  prog[22] = insn::SYNC(0b001);
  prog[23] = insn::SET_ADDR_LO(1, addr_lo(ln1_beta_dram_offset));
  prog[24] = insn::SET_ADDR_HI(1, addr_hi(ln1_beta_dram_offset));
  prog[25] = insn::LOAD(BUF_WBUF_ID, ln1_gb_off_units + int(ln1_gamma_bytes.size() / 16u), int(ln1_beta_bytes.size() / 16u), 1, 0);
  prog[26] = insn::SYNC(0b001);
  prog[27] = insn::LAYERNORM(BUF_ABUF_ID, ln1_input_off_units, BUF_WBUF_ID, ln1_gb_off_units, BUF_ABUF_ID, ln1_output_off_units, ln1_sreg_base, 0);
  prog[28] = insn::SYNC(0b100);
  prog[29] = insn::SET_ADDR_LO(0, addr_lo(query_weight_dram_offset));
  prog[30] = insn::SET_ADDR_HI(0, addr_hi(query_weight_dram_offset));
  prog[31] = insn::LOAD(BUF_WBUF_ID, query_weight_off_units, int(query_weight_bytes.size() / 16u), 0, 0);
  prog[32] = insn::SYNC(0b001);
  prog[33] = insn::CONFIG_TILE(proj_m_tiles, proj_n_tiles, proj_k_tiles);
  prog[34] = insn::MATMUL(BUF_ABUF_ID, query_act_off_units, BUF_WBUF_ID, query_weight_off_units, BUF_ACCUM_ID, 0, 0, 0);
  prog[35] = insn::SYNC(0b010);
  prog[36] = insn::SET_ADDR_LO(1, addr_lo(query_bias_dram_offset));
  prog[37] = insn::SET_ADDR_HI(1, addr_hi(query_bias_dram_offset));
  prog[38] = insn::LOAD(BUF_WBUF_ID, query_bias_off_units, int(query_bias_bytes.size() / 16u), 1, 0);
  prog[39] = insn::SYNC(0b001);
  prog[40] = insn::VADD(BUF_ACCUM_ID, 0, BUF_WBUF_ID, query_bias_off_units, BUF_ACCUM_ID, 0, 0, 0);
  prog[41] = insn::SET_SCALE(1, uint16_t(query_scale_fp16), 0);
  prog[42] = insn::REQUANT(BUF_ACCUM_ID, 0, BUF_ABUF_ID, query_output_off_units, 1, 0);
  prog[43] = insn::SET_ADDR_LO(0, addr_lo(key_weight_dram_offset));
  prog[44] = insn::SET_ADDR_HI(0, addr_hi(key_weight_dram_offset));
  prog[45] = insn::LOAD(BUF_WBUF_ID, key_weight_off_units, int(key_weight_bytes.size() / 16u), 0, 0);
  prog[46] = insn::SYNC(0b001);
  prog[47] = insn::CONFIG_TILE(proj_m_tiles, proj_n_tiles, proj_k_tiles);
  prog[48] = insn::MATMUL(BUF_ABUF_ID, key_act_off_units, BUF_WBUF_ID, key_weight_off_units, BUF_ACCUM_ID, 0, 0, 0);
  prog[49] = insn::SYNC(0b010);
  prog[50] = insn::SET_ADDR_LO(1, addr_lo(key_bias_dram_offset));
  prog[51] = insn::SET_ADDR_HI(1, addr_hi(key_bias_dram_offset));
  prog[52] = insn::LOAD(BUF_WBUF_ID, key_bias_off_units, int(key_bias_bytes.size() / 16u), 1, 0);
  prog[53] = insn::SYNC(0b001);
  prog[54] = insn::VADD(BUF_ACCUM_ID, 0, BUF_WBUF_ID, key_bias_off_units, BUF_ACCUM_ID, 0, 0, 0);
  prog[55] = insn::SET_SCALE(3, uint16_t(key_scale_fp16), 0);
  prog[56] = insn::REQUANT(BUF_ACCUM_ID, 0, BUF_ABUF_ID, key_output_off_units, 3, 0);
  prog[57] = insn::SET_ADDR_LO(0, addr_lo(value_weight_dram_offset));
  prog[58] = insn::SET_ADDR_HI(0, addr_hi(value_weight_dram_offset));
  prog[59] = insn::LOAD(BUF_WBUF_ID, value_weight_off_units, int(value_weight_bytes.size() / 16u), 0, 0);
  prog[60] = insn::SYNC(0b001);
  prog[61] = insn::CONFIG_TILE(proj_m_tiles, proj_n_tiles, proj_k_tiles);
  prog[62] = insn::MATMUL(BUF_ABUF_ID, value_act_off_units, BUF_WBUF_ID, value_weight_off_units, BUF_ACCUM_ID, 0, 0, 0);
  prog[63] = insn::SYNC(0b010);
  prog[64] = insn::SET_ADDR_LO(1, addr_lo(value_bias_dram_offset));
  prog[65] = insn::SET_ADDR_HI(1, addr_hi(value_bias_dram_offset));
  prog[66] = insn::LOAD(BUF_WBUF_ID, value_bias_off_units, int(value_bias_bytes.size() / 16u), 1, 0);
  prog[67] = insn::SYNC(0b001);
  prog[68] = insn::VADD(BUF_ACCUM_ID, 0, BUF_WBUF_ID, value_bias_off_units, BUF_ACCUM_ID, 0, 0, 0);
  prog[69] = insn::SET_SCALE(5, uint16_t(value_scale_fp16), 0);
  prog[70] = insn::REQUANT(BUF_ACCUM_ID, 0, BUF_ABUF_ID, value_output_off_units, 5, 0);
  prog[71] = insn::SET_ADDR_LO(3, addr_lo(zero_pad_dram_offset));
  prog[72] = insn::SET_ADDR_HI(3, addr_hi(zero_pad_dram_offset));
  prog[73] = insn::LOAD(BUF_ABUF_ID, key_padding_tail_off_units, key_zero_pad_tail_bytes / 16, 3, 0);
  prog[74] = insn::SYNC(0b001);
  prog[75] = insn::BUF_COPY(BUF_ABUF_ID, key_padded_off_units, BUF_WBUF_ID, key_t_off_units, (key_output_rows * key_output_cols) / 16, proj_m_tiles, 1);
  prog[76] = insn::SYNC(0b001);
  prog[77] = insn::CONFIG_TILE(1, 13, 4);
  prog[78] = insn::MATMUL(BUF_ABUF_ID, query_input_off_units, BUF_WBUF_ID, key_t_off_units, BUF_ACCUM_ID, 0, 0, 0);
  prog[79] = insn::SYNC(0b010);
  prog[81] = insn::HALT();
  s.load(prog);

  auto* r = s.dut->rootp;
  tbutil::SystolicWindowCollector microtrace(77, 79);
  tbutil::AccumWriteLogCollector accum_write_log(40, 80);
  tbutil::SramWriteLogCollector sram_write_log(26, 80);
  tbutil::SystolicHiddenSnapshotCollector hidden_snapshot(77);
  auto observe_debug = [&](bool retire_valid, uint64_t retire_pc, int retire_opcode) {
    if (microtrace_mode_enabled(name))
      microtrace.observe(r, retire_valid, retire_pc, retire_opcode);
    accum_write_log.observe(r, retire_valid, retire_pc, retire_opcode);
    sram_write_log.observe(r, retire_valid, retire_pc, retire_opcode);
    hidden_snapshot.observe(r, retire_valid, retire_pc, retire_opcode);
  };
  auto observe_cycle = [&]() {
    r = s.dut->rootp;
    observe_debug(
        r->taccel_top__DOT__obs_retire_pulse_w,
        r->taccel_top__DOT__obs_retire_pc_w,
        int(r->taccel_top__DOT__obs_retire_opcode_w));
  };
  auto observe_sram_negedge = [&]() {
    r = s.dut->rootp;
    sram_write_log.observe(
        r,
        r->taccel_top__DOT__obs_retire_pulse_w,
        r->taccel_top__DOT__obs_retire_pc_w,
        int(r->taccel_top__DOT__obs_retire_opcode_w));
  };
  auto fail_with_artifacts = [&](const char* msg) {
    maybe_write_microtrace(name, microtrace.finish());
    maybe_write_accum_write_log(name, accum_write_log.finish());
    maybe_write_sram_write_log(name, sram_write_log.finish());
    maybe_write_hidden_snapshot(name, hidden_snapshot.finish());
    TEST_FAIL(name, msg);
  };
  auto expect_int8_matrix_prefix_or_fail =
      [&](int buf_id,
          int offset_units,
          const std::vector<uint8_t>& expected_bytes,
          int logical_rows,
          int logical_cols,
          int padded_rows,
          int padded_cols,
          const char* label) {
        auto observed = sram_read_bytes(
            s.dut.get(),
            buf_id,
            size_t(offset_units) * 16u,
            size_t(padded_rows) * size_t(padded_cols));
        auto expected_padded =
            pad_i8_rows(expected_bytes, logical_rows, logical_cols, padded_rows, padded_cols);
        for (int row = 0; row < logical_rows; ++row) {
          for (int col = 0; col < logical_cols; ++col) {
            uint8_t got = observed[size_t(row) * size_t(padded_cols) + size_t(col)];
            uint8_t exp = expected_padded[size_t(row) * size_t(padded_cols) + size_t(col)];
            if (got != exp) {
              std::fprintf(stderr,
                           "%s mismatch row=%d col=%d got=%d exp=%d\n",
                           label,
                           row,
                           col,
                           int(int8_t(got)),
                           int(int8_t(exp)));
              fail_with_artifacts("INT8 replay mismatch");
            }
          }
        }
      };
  replay_start_with_debug(s, observe_sram_negedge, observe_cycle);

  bool checked_ln1_output = false;
  bool checked_query_output = false;
  bool checked_key_output = false;
  bool checked_value_output = false;
  bool checked_key_padded = false;
  bool captured_accum_pre_matmul = false;
  std::vector<int32_t> accum_pre_matmul_values;
  for (int cycle = 0; cycle < 7000000; ++cycle) {
    const int pc = int(r->taccel_top__DOT__u_ctrl__DOT__pc_reg);
    if (pc == 29 && !checked_ln1_output) {
      expect_int8_matrix_prefix_or_fail(
          BUF_ABUF_ID,
          ln1_output_off_units,
          ln1_output_bytes,
          query_output_rows,
          ln1_cols,
          query_output_rows,
          ln1_cols,
          "ln1 padded output replay");
      checked_ln1_output = true;
    } else if (pc == 43 && !checked_query_output) {
      expect_int8_matrix_prefix_or_fail(
          BUF_ABUF_ID,
          query_output_off_units,
          query_output_bytes,
          query_output_rows,
          query_output_cols,
          query_output_rows,
          query_output_cols,
          "query padded output replay");
      checked_query_output = true;
    } else if (pc == 57 && !checked_key_output) {
      expect_int8_matrix_prefix_or_fail(
          BUF_ABUF_ID,
          key_output_off_units,
          key_output_bytes,
          key_output_rows,
          key_output_cols,
          key_output_rows,
          key_output_cols,
          "key padded output replay");
      checked_key_output = true;
    } else if (pc == 71 && !checked_value_output) {
      expect_int8_matrix_prefix_or_fail(
          BUF_ABUF_ID,
          value_output_off_units,
          value_output_bytes,
          value_output_rows,
          value_output_cols,
          value_output_rows,
          value_output_cols,
          "value padded output replay");
      checked_value_output = true;
    } else if (pc == 75 && !checked_key_padded) {
      expect_int8_matrix_prefix_or_fail(
          BUF_ABUF_ID,
          key_padded_off_units,
          key_padded_bytes,
          key_output_rows,
          key_output_cols,
          key_output_rows,
          key_output_cols,
          "key zero-mask replay");
      checked_key_padded = true;
    } else if (pc == 78 && !captured_accum_pre_matmul) {
      accum_pre_matmul_values = capture_accum_strip_i32(s.dut.get(), 0, 16, 197, 208);
      captured_accum_pre_matmul = true;
    }
    if (s.dut->done || s.dut->fault)
      break;
    replay_step_with_debug(s, observe_sram_negedge, observe_cycle);
  }

  expect_clean_halt(name, s.dut.get());
  if (!checked_ln1_output || !checked_query_output || !checked_key_output || !checked_value_output || !checked_key_padded)
    fail_with_artifacts("did not complete ln1/qkv/qkt replay checks");
  if (!captured_accum_pre_matmul)
    fail_with_artifacts("did not capture pre-matmul ACCUM checkpoint");

  const auto* golden = reinterpret_cast<const int32_t*>(golden_qkt_bytes.data());
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 197; ++j) {
      int32_t got = read_accum_wide(s.dut.get(), 0, i, j, 208);
      int32_t exp = golden[size_t(i) * 197 + size_t(j)];
      if (got != exp) {
        std::fprintf(stderr,
                     "ln1->qkv->qkt replay mismatch row=%d col=%d got=%d exp=%d",
                     i, j, got, exp);
        if (i == 1 && j == 0)
          std::fprintf(stderr, " [known baseline coordinate]");
        std::fprintf(stderr,
                     " ln1=%d q=%d k=%d v=%d k_mask=%d pc=%d\n",
                     int(checked_ln1_output),
                     int(checked_query_output),
                     int(checked_key_output),
                     int(checked_value_output),
                     int(checked_key_padded),
                     int(r->taccel_top__DOT__u_ctrl__DOT__pc_reg));
        fail_with_artifacts("ln1 + full padded QKV history + QK^T fragment mismatch");
      }
    }
  }

  const auto qkt_output_values = capture_accum_strip_i32(s.dut.get(), 0, 16, 197, 208);
  maybe_write_microtrace(name, microtrace.finish());
  maybe_write_accum_write_log(name, accum_write_log.finish());
  maybe_write_sram_write_log(name, sram_write_log.finish());
  maybe_write_hidden_snapshot(name, hidden_snapshot.finish());
  maybe_write_qkt_checkpoints(
      name,
      node_prefix,
      0,
      197,
      accum_pre_matmul_values,
      qkt_output_values);
  TEST_PASS(name);
}

void test_qkt_prev_pos_embed_ln1_qkv_full_history_then_qkt_replay() {
  const char* name = "qkt_prev_pos_embed_ln1_qkv_full_history_then_qkt_replay";
  const std::string node_prefix = "block0_head0_qkt";
  const char* replay_dir = std::getenv("RTL_QKT_REPLAY_DIR");
  if (replay_dir == nullptr || replay_dir[0] == '\0') {
    std::printf("SKIP: %s (set RTL_QKT_REPLAY_DIR to enable)\n", name);
    return;
  }

  const std::string base(replay_dir);
  auto metadata_text = read_text_file(base + "/replay_metadata.json");
  auto pos_act_input_bytes = read_binary_file(base + "/pos_embed_add_act_input.raw");
  auto pos_pos_input_bytes = read_binary_file(base + "/pos_embed_add_pos_input.raw");
  auto pos_output_bytes = read_binary_file(base + "/pos_embed_add_output.raw");
  auto ln1_output_bytes = read_binary_file(base + "/ln1_output_padded.raw");
  auto ln1_gamma_bytes = read_binary_file(base + "/ln1_gamma.raw");
  auto ln1_beta_bytes = read_binary_file(base + "/ln1_beta.raw");
  auto query_weight_bytes = read_binary_file(base + "/query_projection_weight_input.raw");
  auto query_bias_bytes = read_binary_file(base + "/query_bias_input.raw");
  auto query_accum_pre_bias_bytes = read_binary_file(base + "/query_accum_pre_bias_padded.raw");
  auto query_output_bytes = read_binary_file(base + "/query_output_padded.raw");
  auto key_weight_bytes = read_binary_file(base + "/key_projection_weight_input.raw");
  auto key_bias_bytes = read_binary_file(base + "/key_bias_input.raw");
  auto key_output_bytes = read_binary_file(base + "/key_output_padded.raw");
  auto key_padded_bytes = read_binary_file(base + "/key_padded_input.raw");
  auto value_weight_bytes = read_binary_file(base + "/value_projection_weight_input.raw");
  auto value_bias_bytes = read_binary_file(base + "/value_bias_input.raw");
  auto value_output_bytes = read_binary_file(base + "/value_output_padded.raw");
  auto golden_qkt_bytes = read_binary_file(base + "/golden_qkt.raw");

  const int pos_act_off_units = extract_json_int(metadata_text, "pos_embed_add_act_input_offset_units");
  const int pos_pos_off_units = extract_json_int(metadata_text, "pos_embed_add_pos_input_offset_units");
  const int pos_output_off_units = extract_json_int(metadata_text, "pos_embed_add_output_offset_units");
  const int pos_rows = extract_json_int(metadata_text, "pos_embed_add_rows");
  const int pos_cols = extract_json_int(metadata_text, "pos_embed_add_cols");
  const int ln1_input_off_units = extract_json_int(metadata_text, "ln1_input_padded_offset_units");
  const int ln1_output_off_units = extract_json_int(metadata_text, "ln1_output_padded_offset_units");
  const int ln1_gb_off_units = extract_json_int(metadata_text, "ln1_gamma_beta_wbuf_offset_units");
  const int ln1_sreg_base = extract_json_int(metadata_text, "ln1_sreg_base");
  const int ln1_in_scale_fp16 = extract_json_int(metadata_text, "ln1_in_scale_fp16");
  const int ln1_out_scale_fp16 = extract_json_int(metadata_text, "ln1_out_scale_fp16");
  const int ln1_gamma_dram_offset = extract_json_int(metadata_text, "ln1_gamma_dram_offset");
  const int ln1_beta_dram_offset = extract_json_int(metadata_text, "ln1_beta_dram_offset");
  const int query_weight_dram_offset = extract_json_int(metadata_text, "query_weight_dram_offset");
  const int query_bias_dram_offset = extract_json_int(metadata_text, "query_bias_dram_offset");
  const int key_weight_dram_offset = extract_json_int(metadata_text, "key_weight_dram_offset");
  const int key_bias_dram_offset = extract_json_int(metadata_text, "key_bias_dram_offset");
  const int value_weight_dram_offset = extract_json_int(metadata_text, "value_weight_dram_offset");
  const int value_bias_dram_offset = extract_json_int(metadata_text, "value_bias_dram_offset");
  const int zero_pad_dram_offset = extract_json_int(metadata_text, "zero_pad_dram_offset");
  const int key_zero_pad_tail_bytes = extract_json_int(metadata_text, "key_zero_pad_tail_bytes");
  const int query_act_off_units = extract_json_int(metadata_text, "query_act_input_padded_offset_units");
  const int key_act_off_units = extract_json_int(metadata_text, "key_act_input_padded_offset_units");
  const int value_act_off_units = extract_json_int(metadata_text, "value_act_input_padded_offset_units");
  const int query_weight_off_units = extract_json_int(metadata_text, "query_weight_input_offset_units");
  const int query_bias_off_units = extract_json_int(metadata_text, "query_bias_input_offset_units");
  const int query_output_off_units = extract_json_int(metadata_text, "query_output_padded_offset_units");
  const int query_input_off_units = extract_json_int(metadata_text, "query_input_offset_units");
  const int query_output_rows = extract_json_int(metadata_text, "query_output_padded_rows");
  const int query_output_cols = extract_json_int(metadata_text, "query_output_padded_cols");
  const int query_scale_fp16 = extract_json_int(metadata_text, "query_requant_scale_fp16");
  const int key_weight_off_units = extract_json_int(metadata_text, "key_weight_input_offset_units");
  const int key_bias_off_units = extract_json_int(metadata_text, "key_bias_input_offset_units");
  const int key_output_off_units = extract_json_int(metadata_text, "key_output_padded_offset_units");
  const int key_padded_off_units = extract_json_int(metadata_text, "key_padded_input_offset_units");
  const int key_output_rows = extract_json_int(metadata_text, "key_output_padded_rows");
  const int key_output_cols = extract_json_int(metadata_text, "key_output_padded_cols");
  const int key_scale_fp16 = extract_json_int(metadata_text, "key_requant_scale_fp16");
  const int value_weight_off_units = extract_json_int(metadata_text, "value_weight_input_offset_units");
  const int value_bias_off_units = extract_json_int(metadata_text, "value_bias_input_offset_units");
  const int value_output_off_units = extract_json_int(metadata_text, "value_output_padded_offset_units");
  const int value_output_rows = extract_json_int(metadata_text, "value_output_padded_rows");
  const int value_output_cols = extract_json_int(metadata_text, "value_output_padded_cols");
  const int value_scale_fp16 = extract_json_int(metadata_text, "value_requant_scale_fp16");
  const int key_t_off_units = extract_json_int(metadata_text, "key_transposed_offset_units");
  const int ln1_rows = extract_json_int(metadata_text, "ln1_input_padded_rows");
  const int ln1_cols = extract_json_int(metadata_text, "query_act_input_padded_cols");

  if (query_act_off_units != ln1_output_off_units ||
      key_act_off_units != ln1_output_off_units ||
      value_act_off_units != ln1_output_off_units)
    TEST_FAIL(name, "projection activation offsets do not reuse ln1 output");
  if (query_output_off_units != query_input_off_units)
    TEST_FAIL(name, "query output offset does not match QK^T query input offset");
  if (key_output_off_units != key_padded_off_units)
    TEST_FAIL(name, "key output offset does not match padded K input offset");
  if (pos_output_off_units != ln1_output_off_units)
    TEST_FAIL(name, "pos_embed_add output offset does not match ln1 output offset");
  if (ln1_input_off_units != pos_act_off_units)
    TEST_FAIL(name, "pos_embed_add input offset does not match ln1 input offset");
  if (pos_act_input_bytes.size() != size_t(pos_rows * pos_cols) ||
      pos_pos_input_bytes.size() != size_t(pos_rows * pos_cols) ||
      pos_output_bytes.size() != size_t(pos_rows * pos_cols))
    TEST_FAIL(name, "unexpected pos_embed_add replay payload size");
  if (query_accum_pre_bias_bytes.size() != size_t(query_output_rows * query_output_cols * sizeof(int32_t)))
    TEST_FAIL(name, "unexpected query_accum_pre_bias_padded.raw size");
  if (ln1_gamma_bytes.size() != size_t(ln1_cols * 2) || ln1_beta_bytes.size() != size_t(ln1_cols * 2))
    TEST_FAIL(name, "unexpected layernorm gamma/beta sizes");
  if (ln1_output_bytes.size() != size_t(ln1_rows * ln1_cols))
    TEST_FAIL(name, "unexpected ln1_output_padded.raw size");
  if (golden_qkt_bytes.size() != 16u * 197u * sizeof(int32_t))
    TEST_FAIL(name, "unexpected golden_qkt.raw size");

  const int pos_m_tiles = (pos_rows + 15) / 16;
  const int pos_n_tiles = (pos_cols + 15) / 16;
  const int proj_m_tiles = query_output_rows / 16;
  const int proj_n_tiles = query_output_cols / 16;
  const int proj_k_tiles = ln1_cols / 16;
  const int key_padding_start_row = 197;
  const int key_padding_tail_off_units = key_padded_off_units + (key_padding_start_row * key_output_cols) / 16;
  auto addr_lo = [](int addr) -> int { return int(uint32_t(addr) & ((1u << 28) - 1u)); };
  auto addr_hi = [](int addr) -> int { return int(uint32_t(addr) >> 28); };

  SimHarness s;
  s.dram.write_bytes(uint64_t(ln1_gamma_dram_offset), ln1_gamma_bytes.data(), ln1_gamma_bytes.size());
  s.dram.write_bytes(uint64_t(ln1_beta_dram_offset), ln1_beta_bytes.data(), ln1_beta_bytes.size());
  s.dram.write_bytes(uint64_t(query_weight_dram_offset), query_weight_bytes.data(), query_weight_bytes.size());
  s.dram.write_bytes(uint64_t(query_bias_dram_offset), query_bias_bytes.data(), query_bias_bytes.size());
  s.dram.write_bytes(uint64_t(key_weight_dram_offset), key_weight_bytes.data(), key_weight_bytes.size());
  s.dram.write_bytes(uint64_t(key_bias_dram_offset), key_bias_bytes.data(), key_bias_bytes.size());
  s.dram.write_bytes(uint64_t(value_weight_dram_offset), value_weight_bytes.data(), value_weight_bytes.size());
  s.dram.write_bytes(uint64_t(value_bias_dram_offset), value_bias_bytes.data(), value_bias_bytes.size());
  if (key_zero_pad_tail_bytes > 0) {
    std::vector<uint8_t> zero_pad(size_t(key_zero_pad_tail_bytes), uint8_t(0));
    s.dram.write_bytes(uint64_t(zero_pad_dram_offset), zero_pad.data(), zero_pad.size());
  }
  sram_write_bytes(s.dut.get(), BUF_ABUF_ID, size_t(pos_act_off_units) * 16u, pos_act_input_bytes);
  sram_write_bytes(s.dut.get(), BUF_WBUF_ID, size_t(pos_pos_off_units) * 16u, pos_pos_input_bytes);

  std::vector<uint64_t> prog(82, insn::NOP());
  prog[12] = insn::CONFIG_TILE(pos_m_tiles, pos_n_tiles, 1);
  prog[13] = insn::VADD(BUF_ABUF_ID, pos_act_off_units, BUF_WBUF_ID, pos_pos_off_units, BUF_ABUF_ID, pos_output_off_units, 0, 0);
  prog[14] = insn::BUF_COPY(BUF_ABUF_ID, pos_output_off_units, BUF_ABUF_ID, ln1_input_off_units, (ln1_rows * ln1_cols) / 16, 0, 0);
  prog[15] = insn::SYNC(0b001);
  prog[16] = insn::CONFIG_TILE(proj_m_tiles, proj_k_tiles, 1);
  prog[17] = insn::SET_SCALE(ln1_sreg_base, uint16_t(ln1_in_scale_fp16), 0);
  prog[18] = insn::SET_SCALE(ln1_sreg_base + 1, uint16_t(ln1_out_scale_fp16), 0);
  prog[19] = insn::SET_ADDR_LO(1, addr_lo(ln1_gamma_dram_offset));
  prog[20] = insn::SET_ADDR_HI(1, addr_hi(ln1_gamma_dram_offset));
  prog[21] = insn::LOAD(BUF_WBUF_ID, ln1_gb_off_units, int(ln1_gamma_bytes.size() / 16u), 1, 0);
  prog[22] = insn::SYNC(0b001);
  prog[23] = insn::SET_ADDR_LO(1, addr_lo(ln1_beta_dram_offset));
  prog[24] = insn::SET_ADDR_HI(1, addr_hi(ln1_beta_dram_offset));
  prog[25] = insn::LOAD(BUF_WBUF_ID, ln1_gb_off_units + int(ln1_gamma_bytes.size() / 16u), int(ln1_beta_bytes.size() / 16u), 1, 0);
  prog[26] = insn::SYNC(0b001);
  prog[27] = insn::LAYERNORM(BUF_ABUF_ID, ln1_input_off_units, BUF_WBUF_ID, ln1_gb_off_units, BUF_ABUF_ID, ln1_output_off_units, ln1_sreg_base, 0);
  prog[28] = insn::SYNC(0b100);
  prog[29] = insn::SET_ADDR_LO(0, addr_lo(query_weight_dram_offset));
  prog[30] = insn::SET_ADDR_HI(0, addr_hi(query_weight_dram_offset));
  prog[31] = insn::LOAD(BUF_WBUF_ID, query_weight_off_units, int(query_weight_bytes.size() / 16u), 0, 0);
  prog[32] = insn::SYNC(0b001);
  prog[33] = insn::CONFIG_TILE(proj_m_tiles, proj_n_tiles, proj_k_tiles);
  prog[34] = insn::MATMUL(BUF_ABUF_ID, query_act_off_units, BUF_WBUF_ID, query_weight_off_units, BUF_ACCUM_ID, 0, 0, 0);
  prog[35] = insn::SYNC(0b010);
  prog[36] = insn::SET_ADDR_LO(1, addr_lo(query_bias_dram_offset));
  prog[37] = insn::SET_ADDR_HI(1, addr_hi(query_bias_dram_offset));
  prog[38] = insn::LOAD(BUF_WBUF_ID, query_bias_off_units, int(query_bias_bytes.size() / 16u), 1, 0);
  prog[39] = insn::SYNC(0b001);
  prog[40] = insn::VADD(BUF_ACCUM_ID, 0, BUF_WBUF_ID, query_bias_off_units, BUF_ACCUM_ID, 0, 0, 0);
  prog[41] = insn::SET_SCALE(1, uint16_t(query_scale_fp16), 0);
  prog[42] = insn::REQUANT(BUF_ACCUM_ID, 0, BUF_ABUF_ID, query_output_off_units, 1, 0);
  prog[43] = insn::SET_ADDR_LO(0, addr_lo(key_weight_dram_offset));
  prog[44] = insn::SET_ADDR_HI(0, addr_hi(key_weight_dram_offset));
  prog[45] = insn::LOAD(BUF_WBUF_ID, key_weight_off_units, int(key_weight_bytes.size() / 16u), 0, 0);
  prog[46] = insn::SYNC(0b001);
  prog[47] = insn::CONFIG_TILE(proj_m_tiles, proj_n_tiles, proj_k_tiles);
  prog[48] = insn::MATMUL(BUF_ABUF_ID, key_act_off_units, BUF_WBUF_ID, key_weight_off_units, BUF_ACCUM_ID, 0, 0, 0);
  prog[49] = insn::SYNC(0b010);
  prog[50] = insn::SET_ADDR_LO(1, addr_lo(key_bias_dram_offset));
  prog[51] = insn::SET_ADDR_HI(1, addr_hi(key_bias_dram_offset));
  prog[52] = insn::LOAD(BUF_WBUF_ID, key_bias_off_units, int(key_bias_bytes.size() / 16u), 1, 0);
  prog[53] = insn::SYNC(0b001);
  prog[54] = insn::VADD(BUF_ACCUM_ID, 0, BUF_WBUF_ID, key_bias_off_units, BUF_ACCUM_ID, 0, 0, 0);
  prog[55] = insn::SET_SCALE(3, uint16_t(key_scale_fp16), 0);
  prog[56] = insn::REQUANT(BUF_ACCUM_ID, 0, BUF_ABUF_ID, key_output_off_units, 3, 0);
  prog[57] = insn::SET_ADDR_LO(0, addr_lo(value_weight_dram_offset));
  prog[58] = insn::SET_ADDR_HI(0, addr_hi(value_weight_dram_offset));
  prog[59] = insn::LOAD(BUF_WBUF_ID, value_weight_off_units, int(value_weight_bytes.size() / 16u), 0, 0);
  prog[60] = insn::SYNC(0b001);
  prog[61] = insn::CONFIG_TILE(proj_m_tiles, proj_n_tiles, proj_k_tiles);
  prog[62] = insn::MATMUL(BUF_ABUF_ID, value_act_off_units, BUF_WBUF_ID, value_weight_off_units, BUF_ACCUM_ID, 0, 0, 0);
  prog[63] = insn::SYNC(0b010);
  prog[64] = insn::SET_ADDR_LO(1, addr_lo(value_bias_dram_offset));
  prog[65] = insn::SET_ADDR_HI(1, addr_hi(value_bias_dram_offset));
  prog[66] = insn::LOAD(BUF_WBUF_ID, value_bias_off_units, int(value_bias_bytes.size() / 16u), 1, 0);
  prog[67] = insn::SYNC(0b001);
  prog[68] = insn::VADD(BUF_ACCUM_ID, 0, BUF_WBUF_ID, value_bias_off_units, BUF_ACCUM_ID, 0, 0, 0);
  prog[69] = insn::SET_SCALE(5, uint16_t(value_scale_fp16), 0);
  prog[70] = insn::REQUANT(BUF_ACCUM_ID, 0, BUF_ABUF_ID, value_output_off_units, 5, 0);
  prog[71] = insn::SET_ADDR_LO(3, addr_lo(zero_pad_dram_offset));
  prog[72] = insn::SET_ADDR_HI(3, addr_hi(zero_pad_dram_offset));
  prog[73] = insn::LOAD(BUF_ABUF_ID, key_padding_tail_off_units, key_zero_pad_tail_bytes / 16, 3, 0);
  prog[74] = insn::SYNC(0b001);
  prog[75] = insn::BUF_COPY(BUF_ABUF_ID, key_padded_off_units, BUF_WBUF_ID, key_t_off_units, (key_output_rows * key_output_cols) / 16, proj_m_tiles, 1);
  prog[76] = insn::SYNC(0b001);
  prog[77] = insn::CONFIG_TILE(1, 13, 4);
  prog[78] = insn::MATMUL(BUF_ABUF_ID, query_input_off_units, BUF_WBUF_ID, key_t_off_units, BUF_ACCUM_ID, 0, 0, 0);
  prog[79] = insn::SYNC(0b010);
  prog[81] = insn::HALT();
  s.load(prog);

  auto* r = s.dut->rootp;
  tbutil::SystolicWindowCollector microtrace(12, 40);
  tbutil::AccumWriteLogCollector accum_write_log(40, 80);
  tbutil::SramWriteLogCollector sram_write_log(12, 40);
  tbutil::SystolicHiddenSnapshotCollector hidden_snapshot(77);
  auto observe_debug = [&](bool retire_valid, uint64_t retire_pc, int retire_opcode) {
    if (microtrace_mode_enabled(name))
      microtrace.observe(r, retire_valid, retire_pc, retire_opcode);
    accum_write_log.observe(r, retire_valid, retire_pc, retire_opcode);
    sram_write_log.observe(r, retire_valid, retire_pc, retire_opcode);
    hidden_snapshot.observe(r, retire_valid, retire_pc, retire_opcode);
  };
  auto observe_cycle = [&]() {
    r = s.dut->rootp;
    observe_debug(
        r->taccel_top__DOT__obs_retire_pulse_w,
        r->taccel_top__DOT__obs_retire_pc_w,
        int(r->taccel_top__DOT__obs_retire_opcode_w));
  };
  auto observe_sram_negedge = [&]() {
    r = s.dut->rootp;
    sram_write_log.observe(
        r,
        r->taccel_top__DOT__obs_retire_pulse_w,
        r->taccel_top__DOT__obs_retire_pc_w,
        int(r->taccel_top__DOT__obs_retire_opcode_w));
  };
  auto fail_with_artifacts = [&](const char* msg) {
    maybe_write_microtrace(name, microtrace.finish());
    maybe_write_accum_write_log(name, accum_write_log.finish());
    maybe_write_sram_write_log(name, sram_write_log.finish());
    maybe_write_hidden_snapshot(name, hidden_snapshot.finish());
    TEST_FAIL(name, msg);
  };
  auto finish_sram_log_or_fail = [&]() -> tbutil::SramWriteLog {
    const auto log = sram_write_log.finish();
    struct RequiredRow {
      uint64_t issue_pc;
      uint32_t row;
    };
    constexpr RequiredRow kRequiredRows[] = {
        {21, 0},
        {21, 23},
        {25, 24},
        {25, 47},
    };
    for (const auto& req : kRequiredRows) {
      if (!sram_log_contains_row(log, "dma", req.issue_pc, "wbuf", req.row)) {
        std::fprintf(
            stderr,
            "missing prefix SRAM log row issue_pc=%llu row=%u\n",
            static_cast<unsigned long long>(req.issue_pc),
            req.row);
        fail_with_artifacts("missing ln1 DMA burst edge row in SRAM log");
      }
    }
    return log;
  };
  auto expect_int8_matrix_prefix_or_fail =
      [&](int buf_id,
          int offset_units,
          const std::vector<uint8_t>& expected_bytes,
          int logical_rows,
          int logical_cols,
          int padded_rows,
          int padded_cols,
          const char* label) {
        auto observed = sram_read_bytes(
            s.dut.get(),
            buf_id,
            size_t(offset_units) * 16u,
            size_t(padded_rows) * size_t(padded_cols));
        auto expected_padded =
            pad_i8_rows(expected_bytes, logical_rows, logical_cols, padded_rows, padded_cols);
        for (int row = 0; row < logical_rows; ++row) {
          for (int col = 0; col < logical_cols; ++col) {
            uint8_t got = observed[size_t(row) * size_t(padded_cols) + size_t(col)];
            uint8_t exp = expected_padded[size_t(row) * size_t(padded_cols) + size_t(col)];
            if (got != exp) {
              std::fprintf(stderr,
                           "%s mismatch row=%d col=%d got=%d exp=%d\n",
                           label,
                           row,
                           col,
                           int(int8_t(got)),
                           int(int8_t(exp)));
              fail_with_artifacts("INT8 replay mismatch");
            }
          }
        }
      };
  replay_start_with_debug(s, observe_sram_negedge, observe_cycle);

  bool checked_pos_output = false;
  bool checked_ln1_output = false;
  bool captured_query_accum_pre_bias = false;
  bool checked_query_output = false;
  bool checked_key_output = false;
  bool checked_value_output = false;
  bool checked_key_padded = false;
  bool captured_accum_pre_matmul = false;
  std::vector<int64_t> pos_output_values;
  std::vector<int64_t> ln1_output_values;
  std::vector<int32_t> query_accum_pre_bias_values;
  std::vector<int32_t> accum_pre_matmul_values;
  for (int cycle = 0; cycle < 7000000; ++cycle) {
    const int pc = int(r->taccel_top__DOT__u_ctrl__DOT__pc_reg);
    if (pc == 16 && !checked_pos_output) {
      expect_int8_matrix_prefix_or_fail(
          BUF_ABUF_ID,
          pos_output_off_units,
          pos_output_bytes,
          pos_rows,
          pos_cols,
          pos_rows,
          pos_cols,
          "pos_embed_add output replay");
      pos_output_values = capture_abuf_strip_i8(s.dut.get(), pos_output_off_units, pos_rows, pos_cols);
      checked_pos_output = true;
    } else if (pc == 29 && !checked_ln1_output) {
      expect_int8_matrix_prefix_or_fail(
          BUF_ABUF_ID,
          ln1_output_off_units,
          ln1_output_bytes,
          ln1_rows,
          ln1_cols,
          ln1_rows,
          ln1_cols,
          "ln1 padded output replay");
      ln1_output_values = capture_abuf_strip_i8(s.dut.get(), ln1_output_off_units, ln1_rows, ln1_cols);
      checked_ln1_output = true;
    } else if (pc == 36 && !captured_query_accum_pre_bias) {
      expect_accum_i32_prefix(
          name,
          s.dut.get(),
          query_accum_pre_bias_bytes,
          query_output_rows,
          query_output_cols,
          "query padded accum pre-bias replay");
      query_accum_pre_bias_values =
          capture_accum_strip_i32(s.dut.get(), 0, query_output_rows, query_output_cols, query_output_cols);
      captured_query_accum_pre_bias = true;
    } else if (pc == 43 && !checked_query_output) {
      expect_int8_matrix_prefix_or_fail(
          BUF_ABUF_ID,
          query_output_off_units,
          query_output_bytes,
          query_output_rows,
          query_output_cols,
          query_output_rows,
          query_output_cols,
          "query padded output replay");
      checked_query_output = true;
    } else if (pc == 57 && !checked_key_output) {
      expect_int8_matrix_prefix_or_fail(
          BUF_ABUF_ID,
          key_output_off_units,
          key_output_bytes,
          key_output_rows,
          key_output_cols,
          key_output_rows,
          key_output_cols,
          "key padded output replay");
      checked_key_output = true;
    } else if (pc == 71 && !checked_value_output) {
      expect_int8_matrix_prefix_or_fail(
          BUF_ABUF_ID,
          value_output_off_units,
          value_output_bytes,
          value_output_rows,
          value_output_cols,
          value_output_rows,
          value_output_cols,
          "value padded output replay");
      checked_value_output = true;
    } else if (pc == 75 && !checked_key_padded) {
      expect_int8_matrix_prefix_or_fail(
          BUF_ABUF_ID,
          key_padded_off_units,
          key_padded_bytes,
          key_output_rows,
          key_output_cols,
          key_output_rows,
          key_output_cols,
          "key zero-mask replay");
      checked_key_padded = true;
    } else if (pc == 78 && !captured_accum_pre_matmul) {
      accum_pre_matmul_values = capture_accum_strip_i32(s.dut.get(), 0, 16, 197, 208);
      captured_accum_pre_matmul = true;
    }
    if (s.dut->done || s.dut->fault)
      break;
    replay_step_with_debug(s, observe_sram_negedge, observe_cycle);
  }

  expect_clean_halt(name, s.dut.get());
  if (!checked_pos_output || !checked_ln1_output || !captured_query_accum_pre_bias ||
      !checked_query_output || !checked_key_output || !checked_value_output || !checked_key_padded)
    fail_with_artifacts("did not complete pos_embed/ln1/qkv/qkt replay checks");
  if (!captured_accum_pre_matmul)
    fail_with_artifacts("did not capture pre-matmul ACCUM checkpoint");

  const auto* golden = reinterpret_cast<const int32_t*>(golden_qkt_bytes.data());
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 197; ++j) {
      int32_t got = read_accum_wide(s.dut.get(), 0, i, j, 208);
      int32_t exp = golden[size_t(i) * 197 + size_t(j)];
      if (got != exp) {
        std::fprintf(stderr,
                     "pos_embed->ln1->qkv->qkt replay mismatch row=%d col=%d got=%d exp=%d",
                     i, j, got, exp);
        if (i == 1 && j == 0)
          std::fprintf(stderr, " [known baseline coordinate]");
        std::fprintf(stderr,
                     " pos=%d ln1=%d q_pre=%d q=%d k=%d v=%d k_mask=%d pc=%d\n",
                     int(checked_pos_output),
                     int(checked_ln1_output),
                     int(captured_query_accum_pre_bias),
                     int(checked_query_output),
                     int(checked_key_output),
                     int(checked_value_output),
                     int(checked_key_padded),
                     int(r->taccel_top__DOT__u_ctrl__DOT__pc_reg));
        fail_with_artifacts("pos_embed + ln1 + full padded QKV history + QK^T fragment mismatch");
      }
    }
  }

  const auto qkt_output_values = capture_accum_strip_i32(s.dut.get(), 0, 16, 197, 208);
  const auto sram_log_result = finish_sram_log_or_fail();
  maybe_write_microtrace(name, microtrace.finish());
  maybe_write_accum_write_log(name, accum_write_log.finish());
  maybe_write_sram_write_log(name, sram_log_result);
  maybe_write_hidden_snapshot(name, hidden_snapshot.finish());
  maybe_write_matrix_checkpoints(
      name,
      node_prefix,
      0,
      {
          MatrixCheckpoint{"pos_embed_add_output", "int8", pos_rows, pos_cols, 0, pos_output_values},
          MatrixCheckpoint{"ln1_output", "int8", ln1_rows, ln1_cols, 0, ln1_output_values},
          MatrixCheckpoint{
              "query_accum_pre_bias_padded",
              "int32",
              query_output_rows,
              query_output_cols,
              0,
              widen_i32_values(query_accum_pre_bias_values),
          },
          MatrixCheckpoint{
              "accum_pre_matmul",
              "int32",
              16,
              197,
              0,
              widen_i32_values(accum_pre_matmul_values),
          },
          MatrixCheckpoint{
              "qkt_output",
              "int32",
              16,
              197,
              0,
              widen_i32_values(qkt_output_values),
          },
      });
  TEST_PASS(name);
}

void test_qkt_prev_program_entry_full_history_then_qkt_replay() {
  const char* name = "qkt_prev_program_entry_full_history_then_qkt_replay";
  const std::string node_prefix = "block0_head0_qkt";
  const char* replay_dir = std::getenv("RTL_QKT_REPLAY_DIR");
  if (replay_dir == nullptr || replay_dir[0] == '\0') {
    std::printf("SKIP: %s (set RTL_QKT_REPLAY_DIR to enable)\n", name);
    return;
  }

  const std::string base(replay_dir);
  auto metadata_text = read_text_file(base + "/replay_metadata.json");
  auto startup_cls_bytes = read_binary_file(base + "/startup_cls_token.raw");
  auto startup_patch_bytes = read_binary_file(base + "/startup_patch_input.raw");
  auto startup_pos_padded_bytes = read_binary_file(base + "/startup_pos_input_padded.raw");
  auto pos_act_input_bytes = read_binary_file(base + "/pos_embed_add_act_input.raw");
  auto pos_pos_input_bytes = read_binary_file(base + "/pos_embed_add_pos_input.raw");
  auto pos_output_bytes = read_binary_file(base + "/pos_embed_add_output.raw");
  auto ln1_output_bytes = read_binary_file(base + "/ln1_output_padded.raw");
  auto ln1_gamma_bytes = read_binary_file(base + "/ln1_gamma.raw");
  auto ln1_beta_bytes = read_binary_file(base + "/ln1_beta.raw");
  auto query_weight_bytes = read_binary_file(base + "/query_projection_weight_input.raw");
  auto query_bias_bytes = read_binary_file(base + "/query_bias_input.raw");
  auto query_accum_pre_bias_bytes = read_binary_file(base + "/query_accum_pre_bias_padded.raw");
  auto query_output_bytes = read_binary_file(base + "/query_output_padded.raw");
  auto key_weight_bytes = read_binary_file(base + "/key_projection_weight_input.raw");
  auto key_bias_bytes = read_binary_file(base + "/key_bias_input.raw");
  auto key_output_bytes = read_binary_file(base + "/key_output_padded.raw");
  auto key_padded_bytes = read_binary_file(base + "/key_padded_input.raw");
  auto value_weight_bytes = read_binary_file(base + "/value_projection_weight_input.raw");
  auto value_bias_bytes = read_binary_file(base + "/value_bias_input.raw");
  auto value_output_bytes = read_binary_file(base + "/value_output_padded.raw");
  auto golden_qkt_bytes = read_binary_file(base + "/golden_qkt.raw");

  const int startup_cls_dram_offset = extract_json_int(metadata_text, "startup_cls_dram_offset");
  const int startup_patch_dram_offset = extract_json_int(metadata_text, "startup_patch_dram_offset");
  const int startup_pos_dram_offset = extract_json_int(metadata_text, "startup_pos_dram_offset");
  const int startup_cls_dst_off_units = extract_json_int(metadata_text, "startup_cls_dst_offset_units");
  const int startup_patch_dst_off_units = extract_json_int(metadata_text, "startup_patch_dst_offset_units");
  const int startup_pos_wbuf_off_units = extract_json_int(metadata_text, "startup_pos_wbuf_offset_units");
  const int startup_patch_rows = extract_json_int(metadata_text, "startup_patch_rows");
  const int startup_cols = extract_json_int(metadata_text, "startup_cols");
  const int startup_pos_input_padded_rows = extract_json_int(metadata_text, "startup_pos_input_padded_rows");
  const int startup_pos_input_padded_cols = extract_json_int(metadata_text, "startup_pos_input_padded_cols");
  const int startup_pos_input_padded_row_units = extract_json_int(metadata_text, "startup_pos_input_padded_row_units");
  const int pos_act_off_units = extract_json_int(metadata_text, "pos_embed_add_act_input_offset_units");
  const int pos_pos_off_units = extract_json_int(metadata_text, "pos_embed_add_pos_input_offset_units");
  const int pos_output_off_units = extract_json_int(metadata_text, "pos_embed_add_output_offset_units");
  const int pos_rows = extract_json_int(metadata_text, "pos_embed_add_rows");
  const int pos_cols = extract_json_int(metadata_text, "pos_embed_add_cols");
  const int ln1_input_off_units = extract_json_int(metadata_text, "ln1_input_padded_offset_units");
  const int ln1_output_off_units = extract_json_int(metadata_text, "ln1_output_padded_offset_units");
  const int ln1_gb_off_units = extract_json_int(metadata_text, "ln1_gamma_beta_wbuf_offset_units");
  const int ln1_sreg_base = extract_json_int(metadata_text, "ln1_sreg_base");
  const int ln1_in_scale_fp16 = extract_json_int(metadata_text, "ln1_in_scale_fp16");
  const int ln1_out_scale_fp16 = extract_json_int(metadata_text, "ln1_out_scale_fp16");
  const int ln1_gamma_dram_offset = extract_json_int(metadata_text, "ln1_gamma_dram_offset");
  const int ln1_beta_dram_offset = extract_json_int(metadata_text, "ln1_beta_dram_offset");
  const int query_weight_dram_offset = extract_json_int(metadata_text, "query_weight_dram_offset");
  const int query_bias_dram_offset = extract_json_int(metadata_text, "query_bias_dram_offset");
  const int key_weight_dram_offset = extract_json_int(metadata_text, "key_weight_dram_offset");
  const int key_bias_dram_offset = extract_json_int(metadata_text, "key_bias_dram_offset");
  const int value_weight_dram_offset = extract_json_int(metadata_text, "value_weight_dram_offset");
  const int value_bias_dram_offset = extract_json_int(metadata_text, "value_bias_dram_offset");
  const int zero_pad_dram_offset = extract_json_int(metadata_text, "zero_pad_dram_offset");
  const int key_zero_pad_tail_bytes = extract_json_int(metadata_text, "key_zero_pad_tail_bytes");
  const int query_act_off_units = extract_json_int(metadata_text, "query_act_input_padded_offset_units");
  const int key_act_off_units = extract_json_int(metadata_text, "key_act_input_padded_offset_units");
  const int value_act_off_units = extract_json_int(metadata_text, "value_act_input_padded_offset_units");
  const int query_weight_off_units = extract_json_int(metadata_text, "query_weight_input_offset_units");
  const int query_bias_off_units = extract_json_int(metadata_text, "query_bias_input_offset_units");
  const int query_output_off_units = extract_json_int(metadata_text, "query_output_padded_offset_units");
  const int query_input_off_units = extract_json_int(metadata_text, "query_input_offset_units");
  const int query_output_rows = extract_json_int(metadata_text, "query_output_padded_rows");
  const int query_output_cols = extract_json_int(metadata_text, "query_output_padded_cols");
  const int query_scale_fp16 = extract_json_int(metadata_text, "query_requant_scale_fp16");
  const int key_weight_off_units = extract_json_int(metadata_text, "key_weight_input_offset_units");
  const int key_bias_off_units = extract_json_int(metadata_text, "key_bias_input_offset_units");
  const int key_output_off_units = extract_json_int(metadata_text, "key_output_padded_offset_units");
  const int key_padded_off_units = extract_json_int(metadata_text, "key_padded_input_offset_units");
  const int key_output_rows = extract_json_int(metadata_text, "key_output_padded_rows");
  const int key_output_cols = extract_json_int(metadata_text, "key_output_padded_cols");
  const int key_scale_fp16 = extract_json_int(metadata_text, "key_requant_scale_fp16");
  const int value_weight_off_units = extract_json_int(metadata_text, "value_weight_input_offset_units");
  const int value_bias_off_units = extract_json_int(metadata_text, "value_bias_input_offset_units");
  const int value_output_off_units = extract_json_int(metadata_text, "value_output_padded_offset_units");
  const int value_output_rows = extract_json_int(metadata_text, "value_output_padded_rows");
  const int value_output_cols = extract_json_int(metadata_text, "value_output_padded_cols");
  const int value_scale_fp16 = extract_json_int(metadata_text, "value_requant_scale_fp16");
  const int key_t_off_units = extract_json_int(metadata_text, "key_transposed_offset_units");
  const int ln1_rows = extract_json_int(metadata_text, "ln1_input_padded_rows");
  const int ln1_cols = extract_json_int(metadata_text, "query_act_input_padded_cols");

  if (startup_cols != pos_cols)
    TEST_FAIL(name, "startup cols do not match pos_embed_add cols");
  if (startup_patch_rows != 196)
    TEST_FAIL(name, "unexpected startup patch row count");
  if (pos_act_off_units != ln1_input_off_units)
    TEST_FAIL(name, "pos_embed_add input offset does not match ln1 input offset");
  if (query_act_off_units != ln1_output_off_units ||
      key_act_off_units != ln1_output_off_units ||
      value_act_off_units != ln1_output_off_units)
    TEST_FAIL(name, "projection activation offsets do not reuse ln1 output");
  if (query_output_off_units != query_input_off_units)
    TEST_FAIL(name, "query output offset does not match QK^T query input offset");
  if (key_output_off_units != key_padded_off_units)
    TEST_FAIL(name, "key output offset does not match padded K input offset");
  if (pos_output_off_units != ln1_output_off_units)
    TEST_FAIL(name, "pos_embed_add output offset does not match ln1 output offset");
  if (startup_cls_bytes.size() != size_t(startup_cols))
    TEST_FAIL(name, "unexpected startup_cls_token.raw size");
  if (startup_patch_bytes.size() != size_t(startup_patch_rows * startup_cols))
    TEST_FAIL(name, "unexpected startup_patch_input.raw size");
  if (startup_pos_padded_bytes.size() != size_t(startup_pos_input_padded_rows * startup_pos_input_padded_cols))
    TEST_FAIL(name, "unexpected startup_pos_input_padded.raw size");
  if (startup_pos_input_padded_cols != startup_cols)
    TEST_FAIL(name, "startup_pos_input_padded cols do not match startup cols");
  if (startup_pos_input_padded_row_units != int((startup_pos_padded_bytes.size() + 15u) / 16u))
    TEST_FAIL(name, "startup_pos_input_padded_row_units does not match payload size");
  if (pos_act_input_bytes.size() != size_t(pos_rows * pos_cols) ||
      pos_pos_input_bytes.size() != size_t(pos_rows * pos_cols) ||
      pos_output_bytes.size() != size_t(pos_rows * pos_cols))
    TEST_FAIL(name, "unexpected pos_embed_add replay payload size");
  if (query_accum_pre_bias_bytes.size() != size_t(query_output_rows * query_output_cols * sizeof(int32_t)))
    TEST_FAIL(name, "unexpected query_accum_pre_bias_padded.raw size");
  if (ln1_gamma_bytes.size() != size_t(ln1_cols * 2) || ln1_beta_bytes.size() != size_t(ln1_cols * 2))
    TEST_FAIL(name, "unexpected layernorm gamma/beta sizes");
  if (ln1_output_bytes.size() != size_t(ln1_rows * ln1_cols))
    TEST_FAIL(name, "unexpected ln1_output_padded.raw size");
  if (golden_qkt_bytes.size() != 16u * 197u * sizeof(int32_t))
    TEST_FAIL(name, "unexpected golden_qkt.raw size");

  const int pos_m_tiles = (pos_rows + 15) / 16;
  const int pos_n_tiles = (pos_cols + 15) / 16;
  const int proj_m_tiles = query_output_rows / 16;
  const int proj_n_tiles = query_output_cols / 16;
  const int proj_k_tiles = ln1_cols / 16;
  const int key_padding_start_row = 197;
  const int key_padding_tail_off_units = key_padded_off_units + (key_padding_start_row * key_output_cols) / 16;
  const uint32_t startup_cls_last_row =
      uint32_t(startup_cls_dst_off_units + int(startup_cls_bytes.size() / 16u) - 1);
  const uint32_t startup_patch_last_row =
      uint32_t(startup_patch_dst_off_units + int(startup_patch_bytes.size() / 16u) - 1);
  const uint32_t startup_pos_last_row =
      uint32_t(startup_pos_wbuf_off_units + startup_pos_input_padded_row_units - 1);
  const uint32_t ln1_gamma_last_row =
      uint32_t(ln1_gb_off_units + int(ln1_gamma_bytes.size() / 16u) - 1);
  const uint32_t ln1_beta_first_row = uint32_t(ln1_gb_off_units + int(ln1_gamma_bytes.size() / 16u));
  const uint32_t ln1_beta_last_row =
      uint32_t(ln1_beta_first_row + int(ln1_beta_bytes.size() / 16u) - 1);
  auto addr_lo = [](int addr) -> int { return int(uint32_t(addr) & ((1u << 28) - 1u)); };
  auto addr_hi = [](int addr) -> int { return int(uint32_t(addr) >> 28); };

  SimHarness s;
  s.dram.write_bytes(uint64_t(startup_cls_dram_offset), startup_cls_bytes.data(), startup_cls_bytes.size());
  s.dram.write_bytes(uint64_t(startup_patch_dram_offset), startup_patch_bytes.data(), startup_patch_bytes.size());
  s.dram.write_bytes(uint64_t(startup_pos_dram_offset), startup_pos_padded_bytes.data(), startup_pos_padded_bytes.size());
  s.dram.write_bytes(uint64_t(ln1_gamma_dram_offset), ln1_gamma_bytes.data(), ln1_gamma_bytes.size());
  s.dram.write_bytes(uint64_t(ln1_beta_dram_offset), ln1_beta_bytes.data(), ln1_beta_bytes.size());
  s.dram.write_bytes(uint64_t(query_weight_dram_offset), query_weight_bytes.data(), query_weight_bytes.size());
  s.dram.write_bytes(uint64_t(query_bias_dram_offset), query_bias_bytes.data(), query_bias_bytes.size());
  s.dram.write_bytes(uint64_t(key_weight_dram_offset), key_weight_bytes.data(), key_weight_bytes.size());
  s.dram.write_bytes(uint64_t(key_bias_dram_offset), key_bias_bytes.data(), key_bias_bytes.size());
  s.dram.write_bytes(uint64_t(value_weight_dram_offset), value_weight_bytes.data(), value_weight_bytes.size());
  s.dram.write_bytes(uint64_t(value_bias_dram_offset), value_bias_bytes.data(), value_bias_bytes.size());
  if (key_zero_pad_tail_bytes > 0) {
    std::vector<uint8_t> zero_pad(size_t(key_zero_pad_tail_bytes), uint8_t(0));
    s.dram.write_bytes(uint64_t(zero_pad_dram_offset), zero_pad.data(), zero_pad.size());
  }

  std::vector<uint64_t> prog(82, insn::NOP());
  prog[0] = insn::SET_ADDR_LO(0, addr_lo(startup_cls_dram_offset));
  prog[1] = insn::SET_ADDR_HI(0, addr_hi(startup_cls_dram_offset));
  prog[2] = insn::LOAD(BUF_ABUF_ID, startup_cls_dst_off_units, int(startup_cls_bytes.size() / 16u), 0, 0);
  prog[3] = insn::SYNC(0b001);
  prog[4] = insn::SET_ADDR_LO(1, addr_lo(startup_patch_dram_offset));
  prog[5] = insn::SET_ADDR_HI(1, addr_hi(startup_patch_dram_offset));
  prog[6] = insn::LOAD(BUF_ABUF_ID, startup_patch_dst_off_units, int(startup_patch_bytes.size() / 16u), 1, 0);
  prog[7] = insn::SYNC(0b001);
  prog[8] = insn::SET_ADDR_LO(0, addr_lo(startup_pos_dram_offset));
  prog[9] = insn::SET_ADDR_HI(0, addr_hi(startup_pos_dram_offset));
  prog[10] = insn::LOAD(BUF_WBUF_ID, startup_pos_wbuf_off_units, startup_pos_input_padded_row_units, 0, 0);
  prog[11] = insn::SYNC(0b001);
  prog[12] = insn::CONFIG_TILE(pos_m_tiles, pos_n_tiles, 1);
  prog[13] = insn::VADD(BUF_ABUF_ID, pos_act_off_units, BUF_WBUF_ID, pos_pos_off_units, BUF_ABUF_ID, pos_output_off_units, 0, 0);
  prog[14] = insn::BUF_COPY(BUF_ABUF_ID, pos_output_off_units, BUF_ABUF_ID, ln1_input_off_units, (ln1_rows * ln1_cols) / 16, 0, 0);
  prog[15] = insn::SYNC(0b001);
  prog[16] = insn::CONFIG_TILE(proj_m_tiles, proj_k_tiles, 1);
  prog[17] = insn::SET_SCALE(ln1_sreg_base, uint16_t(ln1_in_scale_fp16), 0);
  prog[18] = insn::SET_SCALE(ln1_sreg_base + 1, uint16_t(ln1_out_scale_fp16), 0);
  prog[19] = insn::SET_ADDR_LO(1, addr_lo(ln1_gamma_dram_offset));
  prog[20] = insn::SET_ADDR_HI(1, addr_hi(ln1_gamma_dram_offset));
  prog[21] = insn::LOAD(BUF_WBUF_ID, ln1_gb_off_units, int(ln1_gamma_bytes.size() / 16u), 1, 0);
  prog[22] = insn::SYNC(0b001);
  prog[23] = insn::SET_ADDR_LO(1, addr_lo(ln1_beta_dram_offset));
  prog[24] = insn::SET_ADDR_HI(1, addr_hi(ln1_beta_dram_offset));
  prog[25] = insn::LOAD(BUF_WBUF_ID, ln1_gb_off_units + int(ln1_gamma_bytes.size() / 16u), int(ln1_beta_bytes.size() / 16u), 1, 0);
  prog[26] = insn::SYNC(0b001);
  prog[27] = insn::LAYERNORM(BUF_ABUF_ID, ln1_input_off_units, BUF_WBUF_ID, ln1_gb_off_units, BUF_ABUF_ID, ln1_output_off_units, ln1_sreg_base, 0);
  prog[28] = insn::SYNC(0b100);
  prog[29] = insn::SET_ADDR_LO(0, addr_lo(query_weight_dram_offset));
  prog[30] = insn::SET_ADDR_HI(0, addr_hi(query_weight_dram_offset));
  prog[31] = insn::LOAD(BUF_WBUF_ID, query_weight_off_units, int(query_weight_bytes.size() / 16u), 0, 0);
  prog[32] = insn::SYNC(0b001);
  prog[33] = insn::CONFIG_TILE(proj_m_tiles, proj_n_tiles, proj_k_tiles);
  prog[34] = insn::MATMUL(BUF_ABUF_ID, query_act_off_units, BUF_WBUF_ID, query_weight_off_units, BUF_ACCUM_ID, 0, 0, 0);
  prog[35] = insn::SYNC(0b010);
  prog[36] = insn::SET_ADDR_LO(1, addr_lo(query_bias_dram_offset));
  prog[37] = insn::SET_ADDR_HI(1, addr_hi(query_bias_dram_offset));
  prog[38] = insn::LOAD(BUF_WBUF_ID, query_bias_off_units, int(query_bias_bytes.size() / 16u), 1, 0);
  prog[39] = insn::SYNC(0b001);
  prog[40] = insn::VADD(BUF_ACCUM_ID, 0, BUF_WBUF_ID, query_bias_off_units, BUF_ACCUM_ID, 0, 0, 0);
  prog[41] = insn::SET_SCALE(1, uint16_t(query_scale_fp16), 0);
  prog[42] = insn::REQUANT(BUF_ACCUM_ID, 0, BUF_ABUF_ID, query_output_off_units, 1, 0);
  prog[43] = insn::SET_ADDR_LO(0, addr_lo(key_weight_dram_offset));
  prog[44] = insn::SET_ADDR_HI(0, addr_hi(key_weight_dram_offset));
  prog[45] = insn::LOAD(BUF_WBUF_ID, key_weight_off_units, int(key_weight_bytes.size() / 16u), 0, 0);
  prog[46] = insn::SYNC(0b001);
  prog[47] = insn::CONFIG_TILE(proj_m_tiles, proj_n_tiles, proj_k_tiles);
  prog[48] = insn::MATMUL(BUF_ABUF_ID, key_act_off_units, BUF_WBUF_ID, key_weight_off_units, BUF_ACCUM_ID, 0, 0, 0);
  prog[49] = insn::SYNC(0b010);
  prog[50] = insn::SET_ADDR_LO(1, addr_lo(key_bias_dram_offset));
  prog[51] = insn::SET_ADDR_HI(1, addr_hi(key_bias_dram_offset));
  prog[52] = insn::LOAD(BUF_WBUF_ID, key_bias_off_units, int(key_bias_bytes.size() / 16u), 1, 0);
  prog[53] = insn::SYNC(0b001);
  prog[54] = insn::VADD(BUF_ACCUM_ID, 0, BUF_WBUF_ID, key_bias_off_units, BUF_ACCUM_ID, 0, 0, 0);
  prog[55] = insn::SET_SCALE(3, uint16_t(key_scale_fp16), 0);
  prog[56] = insn::REQUANT(BUF_ACCUM_ID, 0, BUF_ABUF_ID, key_output_off_units, 3, 0);
  prog[57] = insn::SET_ADDR_LO(0, addr_lo(value_weight_dram_offset));
  prog[58] = insn::SET_ADDR_HI(0, addr_hi(value_weight_dram_offset));
  prog[59] = insn::LOAD(BUF_WBUF_ID, value_weight_off_units, int(value_weight_bytes.size() / 16u), 0, 0);
  prog[60] = insn::SYNC(0b001);
  prog[61] = insn::CONFIG_TILE(proj_m_tiles, proj_n_tiles, proj_k_tiles);
  prog[62] = insn::MATMUL(BUF_ABUF_ID, value_act_off_units, BUF_WBUF_ID, value_weight_off_units, BUF_ACCUM_ID, 0, 0, 0);
  prog[63] = insn::SYNC(0b010);
  prog[64] = insn::SET_ADDR_LO(1, addr_lo(value_bias_dram_offset));
  prog[65] = insn::SET_ADDR_HI(1, addr_hi(value_bias_dram_offset));
  prog[66] = insn::LOAD(BUF_WBUF_ID, value_bias_off_units, int(value_bias_bytes.size() / 16u), 1, 0);
  prog[67] = insn::SYNC(0b001);
  prog[68] = insn::VADD(BUF_ACCUM_ID, 0, BUF_WBUF_ID, value_bias_off_units, BUF_ACCUM_ID, 0, 0, 0);
  prog[69] = insn::SET_SCALE(5, uint16_t(value_scale_fp16), 0);
  prog[70] = insn::REQUANT(BUF_ACCUM_ID, 0, BUF_ABUF_ID, value_output_off_units, 5, 0);
  prog[71] = insn::SET_ADDR_LO(3, addr_lo(zero_pad_dram_offset));
  prog[72] = insn::SET_ADDR_HI(3, addr_hi(zero_pad_dram_offset));
  prog[73] = insn::LOAD(BUF_ABUF_ID, key_padding_tail_off_units, key_zero_pad_tail_bytes / 16, 3, 0);
  prog[74] = insn::SYNC(0b001);
  prog[75] = insn::BUF_COPY(BUF_ABUF_ID, key_padded_off_units, BUF_WBUF_ID, key_t_off_units, (key_output_rows * key_output_cols) / 16, proj_m_tiles, 1);
  prog[76] = insn::SYNC(0b001);
  prog[77] = insn::CONFIG_TILE(1, 13, 4);
  prog[78] = insn::MATMUL(BUF_ABUF_ID, query_input_off_units, BUF_WBUF_ID, key_t_off_units, BUF_ACCUM_ID, 0, 0, 0);
  prog[79] = insn::SYNC(0b010);
  prog[81] = insn::HALT();
  s.load(prog);

  auto* r = s.dut->rootp;
  tbutil::SystolicWindowCollector microtrace(0, 40);
  tbutil::AccumWriteLogCollector accum_write_log(40, 80);
  tbutil::SramWriteLogCollector sram_write_log(0, 40);
  tbutil::SystolicHiddenSnapshotCollector hidden_snapshot(77);
  auto observe_debug = [&](bool retire_valid, uint64_t retire_pc, int retire_opcode) {
    if (microtrace_mode_enabled(name))
      microtrace.observe(r, retire_valid, retire_pc, retire_opcode);
    accum_write_log.observe(r, retire_valid, retire_pc, retire_opcode);
    sram_write_log.observe(r, retire_valid, retire_pc, retire_opcode);
    hidden_snapshot.observe(r, retire_valid, retire_pc, retire_opcode);
  };
  auto observe_cycle = [&]() {
    r = s.dut->rootp;
    observe_debug(
        r->taccel_top__DOT__obs_retire_pulse_w,
        r->taccel_top__DOT__obs_retire_pc_w,
        int(r->taccel_top__DOT__obs_retire_opcode_w));
  };
  auto observe_sram_negedge = [&]() {
    r = s.dut->rootp;
    sram_write_log.observe(
        r,
        r->taccel_top__DOT__obs_retire_pulse_w,
        r->taccel_top__DOT__obs_retire_pc_w,
        int(r->taccel_top__DOT__obs_retire_opcode_w));
  };
  auto fail_with_artifacts = [&](const char* msg) {
    maybe_write_microtrace(name, microtrace.finish());
    maybe_write_accum_write_log(name, accum_write_log.finish());
    maybe_write_sram_write_log(name, sram_write_log.finish());
    maybe_write_hidden_snapshot(name, hidden_snapshot.finish());
    TEST_FAIL(name, msg);
  };
  auto finish_sram_log_or_fail = [&]() -> tbutil::SramWriteLog {
    const auto log = sram_write_log.finish();
    struct RequiredRow {
      uint64_t issue_pc;
      const char* buf_name;
      uint32_t row;
    };
    const RequiredRow required_rows[] = {
        {2, "abuf", uint32_t(startup_cls_dst_off_units)},
        {2, "abuf", startup_cls_last_row},
        {6, "abuf", uint32_t(startup_patch_dst_off_units)},
        {6, "abuf", startup_patch_last_row},
        {10, "wbuf", uint32_t(startup_pos_wbuf_off_units)},
        {10, "wbuf", startup_pos_last_row},
        {21, "wbuf", uint32_t(ln1_gb_off_units)},
        {21, "wbuf", ln1_gamma_last_row},
        {25, "wbuf", ln1_beta_first_row},
        {25, "wbuf", ln1_beta_last_row},
    };
    for (const auto& req : required_rows) {
      if (!sram_log_contains_row(log, "dma", req.issue_pc, req.buf_name, req.row)) {
        std::fprintf(
            stderr,
            "missing startup SRAM log row issue_pc=%llu buf=%s row=%u\n",
            static_cast<unsigned long long>(req.issue_pc),
            req.buf_name,
            req.row);
        fail_with_artifacts("missing startup DMA burst edge row in SRAM log");
      }
    }
    return log;
  };
  auto expect_int8_matrix_prefix_or_fail =
      [&](int buf_id,
          int offset_units,
          const std::vector<uint8_t>& expected_bytes,
          int logical_rows,
          int logical_cols,
          int padded_rows,
          int padded_cols,
          const char* label) {
        auto observed = sram_read_bytes(
            s.dut.get(),
            buf_id,
            size_t(offset_units) * 16u,
            size_t(padded_rows) * size_t(padded_cols));
        auto expected_padded =
            pad_i8_rows(expected_bytes, logical_rows, logical_cols, padded_rows, padded_cols);
        for (int row = 0; row < logical_rows; ++row) {
          for (int col = 0; col < logical_cols; ++col) {
            uint8_t got = observed[size_t(row) * size_t(padded_cols) + size_t(col)];
            uint8_t exp = expected_padded[size_t(row) * size_t(padded_cols) + size_t(col)];
            if (got != exp) {
              std::fprintf(stderr,
                           "%s mismatch row=%d col=%d got=%d exp=%d\n",
                           label,
                           row,
                           col,
                           int(int8_t(got)),
                           int(int8_t(exp)));
              fail_with_artifacts("INT8 replay mismatch");
            }
          }
        }
      };

  replay_start_with_debug(s, observe_sram_negedge, observe_cycle);

  bool checked_pos_act_input = false;
  bool checked_pos_pos_input = false;
  bool checked_pos_output = false;
  bool checked_ln1_output = false;
  bool captured_query_accum_pre_bias = false;
  bool checked_query_output = false;
  bool checked_key_output = false;
  bool checked_value_output = false;
  bool checked_key_padded = false;
  bool captured_accum_pre_matmul = false;
  std::vector<int64_t> pos_act_input_values;
  std::vector<int64_t> pos_pos_input_values;
  std::vector<int64_t> pos_output_values;
  std::vector<int64_t> ln1_output_values;
  std::vector<int32_t> query_accum_pre_bias_values;
  std::vector<int32_t> accum_pre_matmul_values;
  for (int cycle = 0; cycle < 7000000; ++cycle) {
    const int pc = int(r->taccel_top__DOT__u_ctrl__DOT__pc_reg);
    if (pc == 12 && !checked_pos_act_input) {
      expect_int8_matrix_prefix_or_fail(
          BUF_ABUF_ID,
          pos_act_off_units,
          pos_act_input_bytes,
          pos_rows,
          pos_cols,
          pos_rows,
          pos_cols,
          "pos_embed_add act input replay");
      pos_act_input_values = capture_abuf_strip_i8(s.dut.get(), pos_act_off_units, pos_rows, pos_cols);
      checked_pos_act_input = true;
    }
    if (pc == 12 && !checked_pos_pos_input) {
      expect_int8_matrix_prefix_or_fail(
          BUF_WBUF_ID,
          pos_pos_off_units,
          pos_pos_input_bytes,
          pos_rows,
          pos_cols,
          pos_rows,
          pos_cols,
          "pos_embed_add pos input replay");
      pos_pos_input_values = capture_buffer_strip_i8(s.dut.get(), BUF_WBUF_ID, pos_pos_off_units, pos_rows, pos_cols);
      checked_pos_pos_input = true;
    } else if (pc == 16 && !checked_pos_output) {
      expect_int8_matrix_prefix_or_fail(
          BUF_ABUF_ID,
          pos_output_off_units,
          pos_output_bytes,
          pos_rows,
          pos_cols,
          pos_rows,
          pos_cols,
          "pos_embed_add output replay");
      pos_output_values = capture_abuf_strip_i8(s.dut.get(), pos_output_off_units, pos_rows, pos_cols);
      checked_pos_output = true;
    } else if (pc == 29 && !checked_ln1_output) {
      expect_int8_matrix_prefix_or_fail(
          BUF_ABUF_ID,
          ln1_output_off_units,
          ln1_output_bytes,
          ln1_rows,
          ln1_cols,
          ln1_rows,
          ln1_cols,
          "ln1 padded output replay");
      ln1_output_values = capture_abuf_strip_i8(s.dut.get(), ln1_output_off_units, ln1_rows, ln1_cols);
      checked_ln1_output = true;
    } else if (pc == 36 && !captured_query_accum_pre_bias) {
      expect_accum_i32_prefix(
          name,
          s.dut.get(),
          query_accum_pre_bias_bytes,
          query_output_rows,
          query_output_cols,
          "query padded accum pre-bias replay");
      query_accum_pre_bias_values =
          capture_accum_strip_i32(s.dut.get(), 0, query_output_rows, query_output_cols, query_output_cols);
      captured_query_accum_pre_bias = true;
    } else if (pc == 43 && !checked_query_output) {
      expect_int8_matrix_prefix_or_fail(
          BUF_ABUF_ID,
          query_output_off_units,
          query_output_bytes,
          query_output_rows,
          query_output_cols,
          query_output_rows,
          query_output_cols,
          "query padded output replay");
      checked_query_output = true;
    } else if (pc == 57 && !checked_key_output) {
      expect_int8_matrix_prefix_or_fail(
          BUF_ABUF_ID,
          key_output_off_units,
          key_output_bytes,
          key_output_rows,
          key_output_cols,
          key_output_rows,
          key_output_cols,
          "key padded output replay");
      checked_key_output = true;
    } else if (pc == 71 && !checked_value_output) {
      expect_int8_matrix_prefix_or_fail(
          BUF_ABUF_ID,
          value_output_off_units,
          value_output_bytes,
          value_output_rows,
          value_output_cols,
          value_output_rows,
          value_output_cols,
          "value padded output replay");
      checked_value_output = true;
    } else if (pc == 75 && !checked_key_padded) {
      expect_int8_matrix_prefix_or_fail(
          BUF_ABUF_ID,
          key_padded_off_units,
          key_padded_bytes,
          key_output_rows,
          key_output_cols,
          key_output_rows,
          key_output_cols,
          "key zero-mask replay");
      checked_key_padded = true;
    } else if (pc == 78 && !captured_accum_pre_matmul) {
      accum_pre_matmul_values = capture_accum_strip_i32(s.dut.get(), 0, 16, 197, 208);
      captured_accum_pre_matmul = true;
    }
    if (s.dut->done || s.dut->fault)
      break;
    replay_step_with_debug(s, observe_sram_negedge, observe_cycle);
  }

  expect_clean_halt(name, s.dut.get());
  if (!checked_pos_act_input || !checked_pos_pos_input || !checked_pos_output || !checked_ln1_output ||
      !captured_query_accum_pre_bias || !checked_query_output || !checked_key_output ||
      !checked_value_output || !checked_key_padded)
    fail_with_artifacts("did not complete program-entry/ln1/qkv/qkt replay checks");
  if (!captured_accum_pre_matmul)
    fail_with_artifacts("did not capture pre-matmul ACCUM checkpoint");

  const auto* golden = reinterpret_cast<const int32_t*>(golden_qkt_bytes.data());
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 197; ++j) {
      int32_t got = read_accum_wide(s.dut.get(), 0, i, j, 208);
      int32_t exp = golden[size_t(i) * 197 + size_t(j)];
      if (got != exp) {
        std::fprintf(stderr,
                     "program-entry->qkt replay mismatch row=%d col=%d got=%d exp=%d",
                     i, j, got, exp);
        if (i == 1 && j == 0)
          std::fprintf(stderr, " [known baseline coordinate]");
        std::fprintf(stderr,
                     " pos_act=%d pos_pos=%d pos=%d ln1=%d q_pre=%d q=%d k=%d v=%d k_mask=%d pc=%d\n",
                     int(checked_pos_act_input),
                     int(checked_pos_pos_input),
                     int(checked_pos_output),
                     int(checked_ln1_output),
                     int(captured_query_accum_pre_bias),
                     int(checked_query_output),
                     int(checked_key_output),
                     int(checked_value_output),
                     int(checked_key_padded),
                     int(r->taccel_top__DOT__u_ctrl__DOT__pc_reg));
        fail_with_artifacts("program-entry + full padded QKV history + QK^T fragment mismatch");
      }
    }
  }

  const auto qkt_output_values = capture_accum_strip_i32(s.dut.get(), 0, 16, 197, 208);
  const auto sram_log_result = finish_sram_log_or_fail();
  maybe_write_microtrace(name, microtrace.finish());
  maybe_write_accum_write_log(name, accum_write_log.finish());
  maybe_write_sram_write_log(name, sram_log_result);
  maybe_write_hidden_snapshot(name, hidden_snapshot.finish());
  maybe_write_matrix_checkpoints(
      name,
      node_prefix,
      0,
      {
          MatrixCheckpoint{"pos_embed_add_act_input", "int8", pos_rows, pos_cols, 0, pos_act_input_values},
          MatrixCheckpoint{"pos_embed_add_pos_input", "int8", pos_rows, pos_cols, 0, pos_pos_input_values},
          MatrixCheckpoint{"pos_embed_add_output", "int8", pos_rows, pos_cols, 0, pos_output_values},
          MatrixCheckpoint{"ln1_output", "int8", ln1_rows, ln1_cols, 0, ln1_output_values},
          MatrixCheckpoint{
              "query_accum_pre_bias_padded",
              "int32",
              query_output_rows,
              query_output_cols,
              0,
              widen_i32_values(query_accum_pre_bias_values),
          },
          MatrixCheckpoint{
              "accum_pre_matmul",
              "int32",
              16,
              197,
              0,
              widen_i32_values(accum_pre_matmul_values),
          },
          MatrixCheckpoint{
              "qkt_output",
              "int32",
              16,
              197,
              0,
              widen_i32_values(qkt_output_values),
          },
      });
  TEST_PASS(name);
}

void test_qkt_sequential_carryover_leak() {
  const char* name = "qkt_sequential_carryover_leak";
  const char* replay_dir = std::getenv("RTL_QKT_REPLAY_DIR");
  const char* enable = std::getenv("RTL_QKT_INTERDISPATCH_DEBUG");
  if (replay_dir == nullptr || replay_dir[0] == '\0' || enable == nullptr || enable[0] == '\0') {
    std::printf("SKIP: %s (set RTL_QKT_REPLAY_DIR and RTL_QKT_INTERDISPATCH_DEBUG)\n", name);
    return;
  }

  constexpr int Q_OFF_UNITS = 4992;
  constexpr int FIRST_W_OFF_UNITS = 832;
  constexpr int ACCUM_MEM_COLS = 208;

  auto query_bytes = read_binary_file(std::string(replay_dir) + "/query_input.raw");
  auto key_t_bytes = read_binary_file(std::string(replay_dir) + "/key_transposed.raw");
  auto golden_qkt_bytes = read_binary_file(std::string(replay_dir) + "/golden_qkt.raw");

  // The first MATMUL intentionally spans multiple M tiles so a broken
  // controller leaves a nonzero drain base behind for the following QK^T op.
  std::vector<uint8_t> first_query(48u * 32u, uint8_t(1));
  std::vector<uint8_t> first_weight(32u * 32u, uint8_t(1));
  if (golden_qkt_bytes.size() != 16u * 197u * sizeof(int32_t))
    TEST_FAIL(name, "unexpected golden_qkt.raw size");

  SimHarness s;
  sram_write_bytes(s.dut.get(), BUF_ABUF_ID, 0, first_query);
  sram_write_bytes(s.dut.get(), BUF_WBUF_ID, size_t(FIRST_W_OFF_UNITS) * 16, first_weight);
  sram_write_bytes(s.dut.get(), BUF_ABUF_ID, size_t(Q_OFF_UNITS) * 16, query_bytes);
  sram_write_bytes(s.dut.get(), BUF_WBUF_ID, 0, key_t_bytes);

  s.load({
      insn::CONFIG_TILE(2, 1, 1),
      insn::MATMUL(BUF_ABUF_ID, 0, BUF_WBUF_ID, FIRST_W_OFF_UNITS, BUF_ACCUM_ID, 0, 0, 0),
      insn::SYNC(0b010),
      insn::CONFIG_TILE(1, 13, 4),
      insn::MATMUL(BUF_ABUF_ID, Q_OFF_UNITS, BUF_WBUF_ID, 0, BUF_ACCUM_ID, 0, 0, 0),
      insn::SYNC(0b010),
      insn::HALT(),
  });

  s.start_once();
  auto* r = s.dut->rootp;
  bool reached_pre_second = false;
  for (int cycle = 0; cycle < 200000; ++cycle) {
    if (r->taccel_top__DOT__u_ctrl__DOT__pc_reg == 4) {
      reached_pre_second = true;
      break;
    }
    s.step();
    r = s.dut->rootp;
    if (s.dut->done || s.dut->fault)
      break;
  }
  if (!reached_pre_second)
    TEST_FAIL(name, "did not reach pre-second-MATMUL checkpoint");

  int32_t pre_second_acc = read_accum_wide(s.dut.get(), 0, 0, 0, ACCUM_MEM_COLS);
  std::vector<std::string> microtrace;
  bool saw_busy = false;
  bool reached_idle = false;
  for (int cycle = 0; cycle < 20000; ++cycle) {
    s.step();
    r = s.dut->rootp;
    int state = r->taccel_top__DOT__u_systolic__DOT__state;
    if (state != ST_IDLE || saw_busy) {
      saw_busy = true;
      std::ostringstream oss;
      oss << "cycle=" << cycle
          << " state=" << state
          << " mtile=" << int(r->taccel_top__DOT__u_systolic__DOT__mtile_q)
          << " ntile=" << int(r->taccel_top__DOT__u_systolic__DOT__ntile_q)
          << " ktile=" << int(r->taccel_top__DOT__u_systolic__DOT__ktile_q)
          << " lane=" << int(r->taccel_top__DOT__u_systolic__DOT__lane_q)
          << " a_load_row=" << int(r->taccel_top__DOT__u_systolic__DOT__a_load_row_q)
          << " drain_row=" << int(r->taccel_top__DOT__u_systolic__DOT__drain_row_q)
          << " drain_grp=" << int(r->taccel_top__DOT__u_systolic__DOT__drain_grp_q)
          << " clear_acc=" << int(r->taccel_top__DOT__u_systolic__DOT__clear_acc)
          << " step_en=" << int(r->taccel_top__DOT__u_systolic__DOT__step_en)
          << " sram_a_row=" << int(r->taccel_top__DOT__sys_sram_a_row)
          << " sram_b_row=" << int(r->taccel_top__DOT__sys_sram_b_row)
          << " tile_drain_base=" << int(r->taccel_top__DOT__u_systolic__DOT__tile_drain_base_q)
          << " drain_row_addr=" << int(r->taccel_top__DOT__u_systolic__DOT__drain_row_addr_q);
      microtrace.push_back(oss.str());
      if (state == ST_IDLE) {
        reached_idle = true;
        break;
      }
    }
    if (s.dut->done || s.dut->fault)
      break;
  }

  if (pre_second_acc != 0)
    std::printf("INFO: %s pre-second ACCUM[0,0]=%d\n", name, pre_second_acc);

  if (s.dut->fault)
    TEST_FAIL(name, "unexpected fault during sequential carryover probe");
  if (!reached_idle && !s.dut->done)
    TEST_FAIL(name, "second MATMUL did not finish within debug budget");

  const auto* golden = reinterpret_cast<const int32_t*>(golden_qkt_bytes.data());
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 197; ++j) {
      int32_t got = read_accum_wide(s.dut.get(), 0, i, j, ACCUM_MEM_COLS);
      int32_t exp = golden[size_t(i) * 197 + size_t(j)];
      if (got != exp) {
        std::fprintf(stderr,
                     "sequential carryover replay mismatch row=%d col=%d got=%d exp=%d",
                     i, j, got, exp);
        if (i == 1 && j == 0)
          std::fprintf(stderr, " [known baseline coordinate]");
        std::fprintf(stderr, " pre_second_acc=%d\n", pre_second_acc);
        for (const std::string& line : microtrace)
          std::fprintf(stderr, "%s\n", line.c_str());
        TEST_FAIL(name, "second MATMUL output mismatch after carryover probe");
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
  test_qkt_exact_state_replay();
  test_qkt_query_bias_replay();
  test_qkt_query_requant_replay();
  test_qkt_key_bias_replay();
  test_qkt_key_requant_replay();
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
  test_qkt_transpose_then_matmul_exact_replay();
  test_qkt_prev_key_matmul_then_transpose_matmul_exact_replay();
  test_qkt_prev_key_fragment_sync_release();
  test_qkt_prev_qkv_matmuls_then_qkt_replay();
  test_qkt_prev_qkv_full_padded_history_then_qkt_replay();
  test_ln1_preloaded_operands_replay();
  test_ln1_dma_loaded_operands_replay();
  test_qkt_prev_ln1_qkv_full_history_then_qkt_replay();
  test_qkt_prev_pos_embed_ln1_qkv_full_history_then_qkt_replay();
  test_qkt_prev_program_entry_full_history_then_qkt_replay();
  test_qkt_sequential_carryover_leak();

  std::printf("\n%d / %d tests passed\n", tests_pass, tests_run);
  return (tests_pass == tests_run) ? 0 : 1;
}
