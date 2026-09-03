// Shared helpers for test_systolic_qkt_{basic,replay,padded}.cpp.
//
// Each split includes this header and receives an internal-linkage copy of
// the helpers it uses.

#ifndef RTL_VERILATOR_SYSTOLIC_QKT_UTILS_H
#define RTL_VERILATOR_SYSTOLIC_QKT_UTILS_H

// Focused native regressions for the QK^T debug split.

#include "Vtaccel_top.h"
#include "Vtaccel_top___024root.h"
#include "verilated.h"
#include "testbench.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
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

int32_t read_accum_wide(Vtaccel_top* dut, int dst_off, int row_idx, int col_idx, int cols) {
  auto* root = dut->rootp;
  const int words_per_row = cols / 4;
  const int grp = col_idx / 4;
  const int lane = col_idx % 4;
  const int row = dst_off + row_idx * words_per_row + grp;
  uint32_t word = root->taccel_top__DOT__u_sram__DOT__u_accum__DOT__u_impl__DOT__mem[row][lane];
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
