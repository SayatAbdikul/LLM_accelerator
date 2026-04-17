#pragma once

#include "systolic_window_trace.h"

#include <array>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace tbutil {

constexpr int SYSTOLIC_DEBUG_BUF_ABUF = 0;
constexpr int SYSTOLIC_DEBUG_BUF_WBUF = 1;
constexpr int SYSTOLIC_DEBUG_BUF_ACCUM = 2;

inline const char* systolic_debug_buf_name(uint32_t buf) {
  switch (buf) {
    case SYSTOLIC_DEBUG_BUF_ABUF:
      return "abuf";
    case SYSTOLIC_DEBUG_BUF_WBUF:
      return "wbuf";
    case SYSTOLIC_DEBUG_BUF_ACCUM:
      return "accum";
    default:
      return "unknown";
  }
}

struct SramWriteLogRecord {
  uint64_t cycle = 0;
  uint64_t ctrl_pc = 0;
  bool retire_valid = false;
  uint64_t retire_pc = 0;
  int retire_opcode = 0;
  std::string writer_source;
  uint64_t issue_pc = 0;
  int issue_opcode = 0;
  uint32_t buf_id = 0;
  std::string buf_name;
  uint32_t row = 0;
  uint32_t first_word0 = 0;
  uint32_t first_word1 = 0;
  std::string row_hex;
};

struct SramWriteLog {
  uint64_t window_start_pc = 0;
  uint64_t window_end_pc = 0;
  bool window_reached = false;
  bool completed = false;
  std::string reason;
  std::vector<SramWriteLogRecord> records;
};

struct AccumWriteLogRecord {
  uint64_t cycle = 0;
  uint64_t ctrl_pc = 0;
  bool retire_valid = false;
  uint64_t retire_pc = 0;
  int retire_opcode = 0;
  std::string writer_source;
  uint64_t issue_pc = 0;
  int issue_opcode = 0;
  uint32_t row = 0;
  uint32_t first_word0 = 0;
  uint32_t first_word1 = 0;
  std::string row_hex;
};

struct AccumWriteLog {
  uint64_t window_start_pc = 0;
  uint64_t window_end_pc = 0;
  bool window_reached = false;
  bool completed = false;
  std::string reason;
  std::vector<AccumWriteLogRecord> records;
};

struct SystolicHiddenSnapshot {
  uint64_t requested_pc = 0;
  bool captured = false;
  std::string reason;
  uint64_t cycle = 0;
  uint64_t ctrl_pc = 0;
  bool retire_valid = false;
  uint64_t retire_pc = 0;
  int retire_opcode = 0;
  bool sys_busy = false;
  bool dma_busy = false;
  bool helper_busy = false;
  bool sfu_busy = false;
  bool sync_waiting_on_sys = false;
  int state = 0;
  int mtile_q = 0;
  int ntile_q = 0;
  int ktile_q = 0;
  int lane_q = 0;
  int a_load_row_q = 0;
  int drain_row_q = 0;
  int drain_grp_q = 0;
  int tile_drain_base_q = 0;
  int drain_row_addr_q = 0;
  bool clear_acc = false;
  bool step_en = false;
  bool dst_clear_active = false;
  uint32_t dst_clear_row_q = 0;
  uint32_t dst_clear_rows_total_q = 0;
  std::array<std::array<uint8_t, 16>, 16> a_tile_scratch{};
  std::array<std::array<uint8_t, 15>, 16> a_skew{};
  std::array<std::array<uint8_t, 15>, 16> b_skew{};
  std::array<std::array<int32_t, 16>, 16> pe_acc{};
};

inline bool systolic_debug_all_async_engines_idle(Vtaccel_top___024root* root) {
  return (root->taccel_top__DOT__u_systolic__DOT__state == SYSTOLIC_TRACE_ST_IDLE) &&
         (root->taccel_top__DOT__u_dma__DOT__state == 0) &&
         !root->taccel_top__DOT__helper_busy &&
         !root->taccel_top__DOT__sfu_busy;
}

inline std::string systolic_trace_hex128(const VlWide<4>& data) {
  std::ostringstream oss;
  oss << std::hex << std::setfill('0');
  for (int idx = 3; idx >= 0; --idx) {
    oss << std::setw(8) << static_cast<uint32_t>(data[idx]);
  }
  return oss.str();
}

inline bool capture_accum_write_log_record(
    Vtaccel_top___024root* root,
    bool retire_valid,
    uint64_t retire_pc,
    int retire_opcode,
    AccumWriteLogRecord* out) {
  if (out == nullptr) {
    return false;
  }

  const VlWide<4>* wdata = nullptr;
  const char* writer_source = nullptr;
  uint64_t issue_pc = 0;
  int issue_opcode = 0;
  bool selected_en = false;
  bool selected_we = false;

  if (root->taccel_top__DOT__helper_sram_a_en) {
    writer_source = "helper";
    issue_pc = root->taccel_top__DOT__obs_helper_issue_pc_q;
    issue_opcode = root->taccel_top__DOT__obs_helper_issue_opcode_q;
    selected_en = true;
    selected_we = root->taccel_top__DOT__helper_sram_a_we;
  } else if (root->taccel_top__DOT__sfu_sram_a_en) {
    writer_source = "sfu";
    issue_pc = root->taccel_top__DOT__obs_sfu_issue_pc_q;
    issue_opcode = root->taccel_top__DOT__obs_sfu_issue_opcode_q;
    selected_en = true;
    selected_we = root->taccel_top__DOT__sfu_sram_a_we;
  } else if (root->taccel_top__DOT__dma_sram_en) {
    writer_source = "dma";
    issue_pc = root->taccel_top__DOT__obs_dma_issue_pc_q;
    issue_opcode = root->taccel_top__DOT__obs_dma_issue_opcode_q;
    selected_en = true;
    selected_we = root->taccel_top__DOT__dma_sram_we;
  } else if (root->taccel_top__DOT__sys_sram_a_en) {
    writer_source = "sys";
    issue_pc = root->taccel_top__DOT__obs_sys_issue_pc_q;
    issue_opcode = root->taccel_top__DOT__obs_sys_issue_opcode_q;
    selected_en = true;
    selected_we = root->taccel_top__DOT__sys_sram_a_we;
  }

  wdata = &root->taccel_top__DOT__sram_a_wdata;
  const uint32_t row = root->taccel_top__DOT__sram_a_row;
  const uint32_t buf = root->taccel_top__DOT__sram_a_buf;

  if (!selected_en || !selected_we || buf != SYSTOLIC_DEBUG_BUF_ACCUM) {
    return false;
  }

  out->cycle = root->taccel_top__DOT__obs_cycle_count_q;
  out->ctrl_pc = root->taccel_top__DOT__u_ctrl__DOT__pc_reg;
  out->retire_valid = retire_valid;
  out->retire_pc = retire_pc;
  out->retire_opcode = retire_opcode;
  out->writer_source = writer_source;
  out->issue_pc = issue_pc;
  out->issue_opcode = issue_opcode;
  out->row = row;
  out->first_word0 = (*wdata)[0];
  out->first_word1 = (*wdata)[1];
  out->row_hex = systolic_trace_hex128(*wdata);
  return true;
}

inline bool capture_sram_write_log_record(
    Vtaccel_top___024root* root,
    bool retire_valid,
    uint64_t retire_pc,
    int retire_opcode,
    SramWriteLogRecord* out) {
  if (out == nullptr) {
    return false;
  }

  const VlWide<4>* wdata = nullptr;
  const char* writer_source = nullptr;
  uint64_t issue_pc = 0;
  int issue_opcode = 0;
  bool selected_en = false;
  bool selected_we = false;

  if (root->taccel_top__DOT__helper_sram_a_en) {
    writer_source = "helper";
    issue_pc = root->taccel_top__DOT__obs_helper_issue_pc_q;
    issue_opcode = root->taccel_top__DOT__obs_helper_issue_opcode_q;
    selected_en = true;
    selected_we = root->taccel_top__DOT__helper_sram_a_we;
  } else if (root->taccel_top__DOT__sfu_sram_a_en) {
    writer_source = "sfu";
    issue_pc = root->taccel_top__DOT__obs_sfu_issue_pc_q;
    issue_opcode = root->taccel_top__DOT__obs_sfu_issue_opcode_q;
    selected_en = true;
    selected_we = root->taccel_top__DOT__sfu_sram_a_we;
  } else if (root->taccel_top__DOT__dma_sram_en) {
    writer_source = "dma";
    issue_pc = root->taccel_top__DOT__obs_dma_issue_pc_q;
    issue_opcode = root->taccel_top__DOT__obs_dma_issue_opcode_q;
    selected_en = true;
    selected_we = root->taccel_top__DOT__dma_sram_we;
  } else if (root->taccel_top__DOT__sys_sram_a_en) {
    writer_source = "sys";
    issue_pc = root->taccel_top__DOT__obs_sys_issue_pc_q;
    issue_opcode = root->taccel_top__DOT__obs_sys_issue_opcode_q;
    selected_en = true;
    selected_we = root->taccel_top__DOT__sys_sram_a_we;
  }

  wdata = &root->taccel_top__DOT__sram_a_wdata;
  const uint32_t row = root->taccel_top__DOT__sram_a_row;
  const uint32_t buf = root->taccel_top__DOT__sram_a_buf;

  if (!selected_en || !selected_we) {
    return false;
  }
  if (buf != SYSTOLIC_DEBUG_BUF_ABUF &&
      buf != SYSTOLIC_DEBUG_BUF_WBUF &&
      buf != SYSTOLIC_DEBUG_BUF_ACCUM) {
    return false;
  }

  out->cycle = root->taccel_top__DOT__obs_cycle_count_q;
  out->ctrl_pc = root->taccel_top__DOT__u_ctrl__DOT__pc_reg;
  out->retire_valid = retire_valid;
  out->retire_pc = retire_pc;
  out->retire_opcode = retire_opcode;
  out->writer_source = writer_source;
  out->issue_pc = issue_pc;
  out->issue_opcode = issue_opcode;
  out->buf_id = buf;
  out->buf_name = systolic_debug_buf_name(buf);
  out->row = row;
  out->first_word0 = (*wdata)[0];
  out->first_word1 = (*wdata)[1];
  out->row_hex = systolic_trace_hex128(*wdata);
  return true;
}

class AccumWriteLogCollector {
 public:
  AccumWriteLogCollector(uint64_t start_pc, uint64_t end_pc)
      : start_pc_(start_pc), end_pc_(end_pc) {
    log_.window_start_pc = start_pc;
    log_.window_end_pc = end_pc;
  }

  void observe(Vtaccel_top___024root* root,
               bool retire_valid,
               uint64_t retire_pc,
               int retire_opcode) {
    if (completed_) {
      return;
    }

    const uint64_t ctrl_pc = root->taccel_top__DOT__u_ctrl__DOT__pc_reg;
    if (!active_ && ctrl_pc == start_pc_) {
      active_ = true;
      log_.window_reached = true;
    }
    if (!active_) {
      return;
    }

    AccumWriteLogRecord rec;
    if (capture_accum_write_log_record(root, retire_valid, retire_pc, retire_opcode, &rec)) {
      log_.records.push_back(rec);
    }

    if (retire_valid && retire_pc == end_pc_) {
      saw_end_retire_ = true;
    }
    if (saw_end_retire_ && systolic_debug_all_async_engines_idle(root)) {
      completed_ = true;
      active_ = false;
      log_.completed = true;
    }
  }

  AccumWriteLog finish() const {
    AccumWriteLog result = log_;
    if (!result.window_reached) {
      result.reason = "window_not_reached";
    } else if (!result.completed) {
      result.reason = "window_incomplete";
    }
    return result;
  }

 private:
  uint64_t start_pc_ = 0;
  uint64_t end_pc_ = 0;
  bool active_ = false;
  bool saw_end_retire_ = false;
  bool completed_ = false;
  AccumWriteLog log_;
};

class SramWriteLogCollector {
 public:
  SramWriteLogCollector(uint64_t start_pc, uint64_t end_pc)
      : start_pc_(start_pc), end_pc_(end_pc) {
    log_.window_start_pc = start_pc;
    log_.window_end_pc = end_pc;
  }

  void observe(Vtaccel_top___024root* root,
               bool retire_valid,
               uint64_t retire_pc,
               int retire_opcode) {
    if (completed_) {
      return;
    }

    SramWriteLogRecord rec;
    const bool has_record =
        capture_sram_write_log_record(root, retire_valid, retire_pc, retire_opcode, &rec);
    const bool in_issue_window =
        has_record && (rec.issue_pc >= start_pc_) && (rec.issue_pc <= end_pc_);
    const bool start_reached = retire_valid && retire_pc == start_pc_;

    if (!active_ && (in_issue_window || start_reached)) {
      active_ = true;
      log_.window_reached = true;
    }

    if (active_ && in_issue_window && !is_duplicate(rec)) {
      log_.records.push_back(rec);
    }

    if (retire_valid && retire_pc == end_pc_) {
      saw_end_retire_ = true;
    }
    if (saw_end_retire_ && systolic_debug_all_async_engines_idle(root)) {
      completed_ = true;
      active_ = false;
      log_.completed = true;
    }
  }

  SramWriteLog finish() const {
    SramWriteLog result = log_;
    if (!result.window_reached) {
      result.reason = "window_not_reached";
    } else if (!result.completed) {
      result.reason = "window_incomplete";
    }
    return result;
  }

 private:
  bool is_duplicate(const SramWriteLogRecord& rec) const {
    if (log_.records.empty()) {
      return false;
    }
    const auto& last = log_.records.back();
    return last.writer_source == rec.writer_source &&
           last.issue_pc == rec.issue_pc &&
           last.issue_opcode == rec.issue_opcode &&
           last.buf_id == rec.buf_id &&
           last.row == rec.row &&
           last.row_hex == rec.row_hex;
  }

  uint64_t start_pc_ = 0;
  uint64_t end_pc_ = 0;
  bool active_ = false;
  bool saw_end_retire_ = false;
  bool completed_ = false;
  SramWriteLog log_;
};

inline SystolicHiddenSnapshot capture_hidden_snapshot_record(
    Vtaccel_top___024root* root,
    uint64_t requested_pc,
    bool retire_valid,
    uint64_t retire_pc,
    int retire_opcode) {
  SystolicHiddenSnapshot snapshot;
  snapshot.requested_pc = requested_pc;
  snapshot.captured = true;
  snapshot.cycle = root->taccel_top__DOT__obs_cycle_count_q;
  snapshot.ctrl_pc = root->taccel_top__DOT__u_ctrl__DOT__pc_reg;
  snapshot.retire_valid = retire_valid;
  snapshot.retire_pc = retire_pc;
  snapshot.retire_opcode = retire_opcode;
  snapshot.sys_busy = (root->taccel_top__DOT__u_systolic__DOT__state != SYSTOLIC_TRACE_ST_IDLE);
  snapshot.dma_busy = (root->taccel_top__DOT__u_dma__DOT__state != 0);
  snapshot.helper_busy = root->taccel_top__DOT__helper_busy;
  snapshot.sfu_busy = root->taccel_top__DOT__sfu_busy;
  snapshot.sync_waiting_on_sys = root->taccel_top__DOT__obs_sync_wait_sys_w;
  snapshot.state = root->taccel_top__DOT__u_systolic__DOT__state;
  snapshot.mtile_q = root->taccel_top__DOT__u_systolic__DOT__mtile_q;
  snapshot.ntile_q = root->taccel_top__DOT__u_systolic__DOT__ntile_q;
  snapshot.ktile_q = root->taccel_top__DOT__u_systolic__DOT__ktile_q;
  snapshot.lane_q = root->taccel_top__DOT__u_systolic__DOT__lane_q;
  snapshot.a_load_row_q = root->taccel_top__DOT__u_systolic__DOT__a_load_row_q;
  snapshot.drain_row_q = root->taccel_top__DOT__u_systolic__DOT__drain_row_q;
  snapshot.drain_grp_q = root->taccel_top__DOT__u_systolic__DOT__drain_grp_q;
  snapshot.tile_drain_base_q = root->taccel_top__DOT__u_systolic__DOT__tile_drain_base_q;
  snapshot.drain_row_addr_q = root->taccel_top__DOT__u_systolic__DOT__drain_row_addr_q;
  snapshot.clear_acc = root->taccel_top__DOT__u_systolic__DOT__clear_acc;
  snapshot.step_en = root->taccel_top__DOT__u_systolic__DOT__step_en;
  snapshot.dst_clear_row_q = root->taccel_top__DOT__u_systolic__DOT__dst_clear_row_idx_q;
  snapshot.dst_clear_rows_total_q = root->taccel_top__DOT__u_systolic__DOT__dst_clear_total_rows_q;
  snapshot.dst_clear_active =
      (snapshot.state == SYSTOLIC_TRACE_ST_DST_CLEAR_PREP) ||
      (snapshot.state == SYSTOLIC_TRACE_ST_DST_CLEAR_WR);

  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 16; ++col) {
      snapshot.a_tile_scratch[row][col] =
          root->taccel_top__DOT__u_systolic__DOT__a_tile_scratch[row][col];
      snapshot.pe_acc[row][col] =
          static_cast<int32_t>(root->taccel_top__DOT__u_systolic__DOT__u_array__DOT__pe_acc[row][col]);
    }
    for (int skew = 0; skew < 15; ++skew) {
      snapshot.a_skew[row][skew] =
          root->taccel_top__DOT__u_systolic__DOT__u_array__DOT__a_skew[row][skew];
      snapshot.b_skew[row][skew] =
          root->taccel_top__DOT__u_systolic__DOT__u_array__DOT__b_skew[row][skew];
    }
  }
  return snapshot;
}

class SystolicHiddenSnapshotCollector {
 public:
  explicit SystolicHiddenSnapshotCollector(uint64_t requested_pc)
      : requested_pc_(requested_pc) {
    snapshot_.requested_pc = requested_pc;
  }

  void observe(Vtaccel_top___024root* root,
               bool retire_valid,
               uint64_t retire_pc,
               int retire_opcode) {
    if (snapshot_.captured) {
      return;
    }

    if (!armed_ && retire_valid && retire_pc == requested_pc_) {
      armed_ = true;
      return;
    }

    if (!armed_) {
      return;
    }

    if ((root->taccel_top__DOT__u_systolic__DOT__state != SYSTOLIC_TRACE_ST_IDLE) &&
        !root->taccel_top__DOT__u_systolic__DOT__step_en) {
      snapshot_ = capture_hidden_snapshot_record(root, requested_pc_, retire_valid, retire_pc, retire_opcode);
    }
  }

  SystolicHiddenSnapshot finish() const {
    SystolicHiddenSnapshot result = snapshot_;
    if (!result.captured) {
      result.reason = armed_ ? "capture_condition_not_met" : "trigger_not_reached";
    }
    return result;
  }

 private:
  uint64_t requested_pc_ = 0;
  bool armed_ = false;
  SystolicHiddenSnapshot snapshot_;
};

inline std::string accum_write_log_to_json(const AccumWriteLog& log) {
  std::ostringstream oss;
  oss << "{\n";
  oss << "  \"window_start_pc\": " << log.window_start_pc << ",\n";
  oss << "  \"window_end_pc\": " << log.window_end_pc << ",\n";
  oss << "  \"window_reached\": " << (log.window_reached ? "true" : "false") << ",\n";
  oss << "  \"completed\": " << (log.completed ? "true" : "false") << ",\n";
  if (log.reason.empty()) {
    oss << "  \"reason\": null,\n";
  } else {
    oss << "  \"reason\": \"" << systolic_trace_json_escape(log.reason) << "\",\n";
  }
  oss << "  \"records\": [\n";
  for (size_t i = 0; i < log.records.size(); ++i) {
    const auto& rec = log.records[i];
    oss << "    {\n";
    oss << "      \"cycle\": " << rec.cycle << ",\n";
    oss << "      \"ctrl_pc\": " << rec.ctrl_pc << ",\n";
    if (rec.retire_valid) {
      oss << "      \"retire_pc\": " << rec.retire_pc << ",\n";
      oss << "      \"retire_opcode\": " << rec.retire_opcode << ",\n";
    } else {
      oss << "      \"retire_pc\": null,\n";
      oss << "      \"retire_opcode\": null,\n";
    }
    oss << "      \"writer_source\": \"" << systolic_trace_json_escape(rec.writer_source) << "\",\n";
    oss << "      \"issue_pc\": " << rec.issue_pc << ",\n";
    oss << "      \"issue_opcode\": " << rec.issue_opcode << ",\n";
    oss << "      \"row\": " << rec.row << ",\n";
    oss << "      \"first_word0\": " << rec.first_word0 << ",\n";
    oss << "      \"first_word1\": " << rec.first_word1 << ",\n";
    oss << "      \"row_hex\": \"" << rec.row_hex << "\"\n";
    oss << "    }";
    if (i + 1 != log.records.size()) {
      oss << ",";
    }
    oss << "\n";
  }
  oss << "  ]\n";
  oss << "}\n";
  return oss.str();
}

inline std::string sram_write_log_to_json(const SramWriteLog& log) {
  std::ostringstream oss;
  oss << "{\n";
  oss << "  \"window_start_pc\": " << log.window_start_pc << ",\n";
  oss << "  \"window_end_pc\": " << log.window_end_pc << ",\n";
  oss << "  \"window_reached\": " << (log.window_reached ? "true" : "false") << ",\n";
  oss << "  \"completed\": " << (log.completed ? "true" : "false") << ",\n";
  if (log.reason.empty()) {
    oss << "  \"reason\": null,\n";
  } else {
    oss << "  \"reason\": \"" << systolic_trace_json_escape(log.reason) << "\",\n";
  }
  oss << "  \"records\": [\n";
  for (size_t i = 0; i < log.records.size(); ++i) {
    const auto& rec = log.records[i];
    oss << "    {\n";
    oss << "      \"cycle\": " << rec.cycle << ",\n";
    oss << "      \"ctrl_pc\": " << rec.ctrl_pc << ",\n";
    if (rec.retire_valid) {
      oss << "      \"retire_pc\": " << rec.retire_pc << ",\n";
      oss << "      \"retire_opcode\": " << rec.retire_opcode << ",\n";
    } else {
      oss << "      \"retire_pc\": null,\n";
      oss << "      \"retire_opcode\": null,\n";
    }
    oss << "      \"writer_source\": \"" << systolic_trace_json_escape(rec.writer_source) << "\",\n";
    oss << "      \"issue_pc\": " << rec.issue_pc << ",\n";
    oss << "      \"issue_opcode\": " << rec.issue_opcode << ",\n";
    oss << "      \"buf_id\": " << rec.buf_id << ",\n";
    oss << "      \"buf_name\": \"" << systolic_trace_json_escape(rec.buf_name) << "\",\n";
    oss << "      \"row\": " << rec.row << ",\n";
    oss << "      \"first_word0\": " << rec.first_word0 << ",\n";
    oss << "      \"first_word1\": " << rec.first_word1 << ",\n";
    oss << "      \"row_hex\": \"" << rec.row_hex << "\"\n";
    oss << "    }";
    if (i + 1 != log.records.size()) {
      oss << ",";
    }
    oss << "\n";
  }
  oss << "  ]\n";
  oss << "}\n";
  return oss.str();
}

inline std::string hidden_snapshot_to_json(const SystolicHiddenSnapshot& snapshot) {
  auto write_u8_matrix_16x16 = [](std::ostringstream& oss,
                                  const std::array<std::array<uint8_t, 16>, 16>& matrix) {
    oss << "[\n";
    for (size_t row = 0; row < matrix.size(); ++row) {
      oss << "    [";
      for (size_t col = 0; col < matrix[row].size(); ++col) {
        if (col != 0) {
          oss << ", ";
        }
        oss << int(matrix[row][col]);
      }
      oss << "]";
      if (row + 1 != matrix.size()) {
        oss << ",";
      }
      oss << "\n";
    }
    oss << "  ]";
  };

  auto write_u8_matrix_16x15 = [](std::ostringstream& oss,
                                  const std::array<std::array<uint8_t, 15>, 16>& matrix) {
    oss << "[\n";
    for (size_t row = 0; row < matrix.size(); ++row) {
      oss << "    [";
      for (size_t col = 0; col < matrix[row].size(); ++col) {
        if (col != 0) {
          oss << ", ";
        }
        oss << int(matrix[row][col]);
      }
      oss << "]";
      if (row + 1 != matrix.size()) {
        oss << ",";
      }
      oss << "\n";
    }
    oss << "  ]";
  };

  auto write_i32_matrix_16x16 = [](std::ostringstream& oss,
                                   const std::array<std::array<int32_t, 16>, 16>& matrix) {
    oss << "[\n";
    for (size_t row = 0; row < matrix.size(); ++row) {
      oss << "    [";
      for (size_t col = 0; col < matrix[row].size(); ++col) {
        if (col != 0) {
          oss << ", ";
        }
        oss << matrix[row][col];
      }
      oss << "]";
      if (row + 1 != matrix.size()) {
        oss << ",";
      }
      oss << "\n";
    }
    oss << "  ]";
  };

  std::ostringstream oss;
  oss << "{\n";
  oss << "  \"requested_pc\": " << snapshot.requested_pc << ",\n";
  oss << "  \"captured\": " << (snapshot.captured ? "true" : "false") << ",\n";
  if (snapshot.reason.empty()) {
    oss << "  \"reason\": null,\n";
  } else {
    oss << "  \"reason\": \"" << systolic_trace_json_escape(snapshot.reason) << "\",\n";
  }
  oss << "  \"cycle\": " << snapshot.cycle << ",\n";
  oss << "  \"ctrl_pc\": " << snapshot.ctrl_pc << ",\n";
  if (snapshot.retire_valid) {
    oss << "  \"retire_pc\": " << snapshot.retire_pc << ",\n";
    oss << "  \"retire_opcode\": " << snapshot.retire_opcode << ",\n";
  } else {
    oss << "  \"retire_pc\": null,\n";
    oss << "  \"retire_opcode\": null,\n";
  }
  oss << "  \"sys_busy\": " << (snapshot.sys_busy ? "true" : "false") << ",\n";
  oss << "  \"dma_busy\": " << (snapshot.dma_busy ? "true" : "false") << ",\n";
  oss << "  \"helper_busy\": " << (snapshot.helper_busy ? "true" : "false") << ",\n";
  oss << "  \"sfu_busy\": " << (snapshot.sfu_busy ? "true" : "false") << ",\n";
  oss << "  \"sync_waiting_on_sys\": " << (snapshot.sync_waiting_on_sys ? "true" : "false") << ",\n";
  oss << "  \"state\": " << snapshot.state << ",\n";
  oss << "  \"mtile_q\": " << snapshot.mtile_q << ",\n";
  oss << "  \"ntile_q\": " << snapshot.ntile_q << ",\n";
  oss << "  \"ktile_q\": " << snapshot.ktile_q << ",\n";
  oss << "  \"lane_q\": " << snapshot.lane_q << ",\n";
  oss << "  \"a_load_row_q\": " << snapshot.a_load_row_q << ",\n";
  oss << "  \"drain_row_q\": " << snapshot.drain_row_q << ",\n";
  oss << "  \"drain_grp_q\": " << snapshot.drain_grp_q << ",\n";
  oss << "  \"tile_drain_base_q\": " << snapshot.tile_drain_base_q << ",\n";
  oss << "  \"drain_row_addr_q\": " << snapshot.drain_row_addr_q << ",\n";
  oss << "  \"clear_acc\": " << (snapshot.clear_acc ? "true" : "false") << ",\n";
  oss << "  \"step_en\": " << (snapshot.step_en ? "true" : "false") << ",\n";
  oss << "  \"dst_clear_active\": " << (snapshot.dst_clear_active ? "true" : "false") << ",\n";
  oss << "  \"dst_clear_row_q\": " << snapshot.dst_clear_row_q << ",\n";
  oss << "  \"dst_clear_rows_total_q\": " << snapshot.dst_clear_rows_total_q << ",\n";
  oss << "  \"a_tile_scratch\": ";
  write_u8_matrix_16x16(oss, snapshot.a_tile_scratch);
  oss << ",\n";
  oss << "  \"a_skew\": ";
  write_u8_matrix_16x15(oss, snapshot.a_skew);
  oss << ",\n";
  oss << "  \"b_skew\": ";
  write_u8_matrix_16x15(oss, snapshot.b_skew);
  oss << ",\n";
  oss << "  \"pe_acc\": ";
  write_i32_matrix_16x16(oss, snapshot.pe_acc);
  oss << "\n";
  oss << "}\n";
  return oss.str();
}

}  // namespace tbutil
