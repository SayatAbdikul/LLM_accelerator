#pragma once

#include "Vtaccel_top___024root.h"

#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace tbutil {

constexpr int SYSTOLIC_TRACE_ST_IDLE = 0;
constexpr int SYSTOLIC_TRACE_ST_DST_CLEAR_PREP = 8;
constexpr int SYSTOLIC_TRACE_ST_DST_CLEAR_WR = 9;

struct SystolicWindowTraceRecord {
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
  int sys_sram_a_row = 0;
  int sys_sram_b_row = 0;
  bool dst_clear_active = false;
  uint32_t dst_clear_row_q = 0;
  uint32_t dst_clear_rows_total_q = 0;
};

struct SystolicWindowTrace {
  uint64_t window_start_pc = 0;
  uint64_t window_end_pc = 0;
  bool window_reached = false;
  bool completed = false;
  std::string reason;
  std::vector<SystolicWindowTraceRecord> records;
};

inline std::string systolic_trace_json_escape(const std::string& in) {
  std::ostringstream oss;
  for (char ch : in) {
    switch (ch) {
      case '\\': oss << "\\\\"; break;
      case '"': oss << "\\\""; break;
      case '\n': oss << "\\n"; break;
      case '\r': oss << "\\r"; break;
      case '\t': oss << "\\t"; break;
      default:
        if (static_cast<unsigned char>(ch) < 0x20) {
          oss << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << int(static_cast<unsigned char>(ch))
              << std::dec << std::setfill(' ');
        } else {
          oss << ch;
        }
        break;
    }
  }
  return oss.str();
}

inline SystolicWindowTraceRecord capture_systolic_window_record(
    Vtaccel_top___024root* root,
    bool retire_valid,
    uint64_t retire_pc,
    int retire_opcode) {
  SystolicWindowTraceRecord rec;
  rec.cycle = root->taccel_top__DOT__obs_cycle_count_q;
  rec.ctrl_pc = root->taccel_top__DOT__u_ctrl__DOT__pc_reg;
  rec.retire_valid = retire_valid;
  rec.retire_pc = retire_pc;
  rec.retire_opcode = retire_opcode;
  rec.sys_busy = (root->taccel_top__DOT__u_systolic__DOT__state != SYSTOLIC_TRACE_ST_IDLE);
  rec.dma_busy = (root->taccel_top__DOT__u_dma__DOT__state != 0);
  rec.helper_busy = root->taccel_top__DOT__helper_busy;
  rec.sfu_busy = root->taccel_top__DOT__sfu_busy;
  rec.sync_waiting_on_sys = root->taccel_top__DOT__obs_sync_wait_sys_w;
  rec.state = root->taccel_top__DOT__u_systolic__DOT__state;
  rec.mtile_q = root->taccel_top__DOT__u_systolic__DOT__mtile_q;
  rec.ntile_q = root->taccel_top__DOT__u_systolic__DOT__ntile_q;
  rec.ktile_q = root->taccel_top__DOT__u_systolic__DOT__ktile_q;
  rec.lane_q = root->taccel_top__DOT__u_systolic__DOT__lane_q;
  rec.a_load_row_q = root->taccel_top__DOT__u_systolic__DOT__a_load_row_q;
  rec.drain_row_q = root->taccel_top__DOT__u_systolic__DOT__drain_row_q;
  rec.drain_grp_q = root->taccel_top__DOT__u_systolic__DOT__drain_grp_q;
  rec.tile_drain_base_q = root->taccel_top__DOT__u_systolic__DOT__tile_drain_base_q;
  rec.drain_row_addr_q = root->taccel_top__DOT__u_systolic__DOT__drain_row_addr_q;
  rec.clear_acc = root->taccel_top__DOT__u_systolic__DOT__clear_acc;
  rec.step_en = root->taccel_top__DOT__u_systolic__DOT__step_en;
  rec.sys_sram_a_row = root->taccel_top__DOT__sys_sram_a_row;
  rec.sys_sram_b_row = root->taccel_top__DOT__sys_sram_b_row;
  rec.dst_clear_row_q = root->taccel_top__DOT__u_systolic__DOT__dst_clear_row_idx_q;
  rec.dst_clear_rows_total_q = root->taccel_top__DOT__u_systolic__DOT__dst_clear_total_rows_q;
  rec.dst_clear_active =
      (rec.state == SYSTOLIC_TRACE_ST_DST_CLEAR_PREP) ||
      (rec.state == SYSTOLIC_TRACE_ST_DST_CLEAR_WR);
  return rec;
}

class SystolicWindowCollector {
 public:
  SystolicWindowCollector(uint64_t start_pc, uint64_t end_pc)
      : start_pc_(start_pc), end_pc_(end_pc) {
    trace_.window_start_pc = start_pc;
    trace_.window_end_pc = end_pc;
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
      trace_.window_reached = true;
    }
    if (!active_) {
      return;
    }

    trace_.records.push_back(
        capture_systolic_window_record(root, retire_valid, retire_pc, retire_opcode));

    if (retire_valid && retire_pc == end_pc_) {
      saw_end_retire_ = true;
    }
    if (saw_end_retire_ &&
        root->taccel_top__DOT__u_systolic__DOT__state == SYSTOLIC_TRACE_ST_IDLE) {
      completed_ = true;
      active_ = false;
      trace_.completed = true;
    }
  }

  SystolicWindowTrace finish() const {
    SystolicWindowTrace result = trace_;
    if (!result.window_reached) {
      result.reason = "window_not_reached";
    } else if (!result.completed) {
      result.reason = "window_incomplete";
    }
    return result;
  }

  bool active() const { return active_; }
  bool completed() const { return completed_; }

 private:
  uint64_t start_pc_ = 0;
  uint64_t end_pc_ = 0;
  bool active_ = false;
  bool saw_end_retire_ = false;
  bool completed_ = false;
  SystolicWindowTrace trace_;
};

inline std::string systolic_window_trace_to_json(const SystolicWindowTrace& trace) {
  std::ostringstream oss;
  oss << "{\n";
  oss << "  \"window_start_pc\": " << trace.window_start_pc << ",\n";
  oss << "  \"window_end_pc\": " << trace.window_end_pc << ",\n";
  oss << "  \"window_reached\": " << (trace.window_reached ? "true" : "false") << ",\n";
  oss << "  \"completed\": " << (trace.completed ? "true" : "false") << ",\n";
  if (trace.reason.empty()) {
    oss << "  \"reason\": null,\n";
  } else {
    oss << "  \"reason\": \"" << systolic_trace_json_escape(trace.reason) << "\",\n";
  }
  oss << "  \"records\": [\n";
  for (size_t i = 0; i < trace.records.size(); ++i) {
    const auto& rec = trace.records[i];
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
    oss << "      \"sys_busy\": " << (rec.sys_busy ? "true" : "false") << ",\n";
    oss << "      \"dma_busy\": " << (rec.dma_busy ? "true" : "false") << ",\n";
    oss << "      \"helper_busy\": " << (rec.helper_busy ? "true" : "false") << ",\n";
    oss << "      \"sfu_busy\": " << (rec.sfu_busy ? "true" : "false") << ",\n";
    oss << "      \"sync_waiting_on_sys\": " << (rec.sync_waiting_on_sys ? "true" : "false") << ",\n";
    oss << "      \"state\": " << rec.state << ",\n";
    oss << "      \"mtile_q\": " << rec.mtile_q << ",\n";
    oss << "      \"ntile_q\": " << rec.ntile_q << ",\n";
    oss << "      \"ktile_q\": " << rec.ktile_q << ",\n";
    oss << "      \"lane_q\": " << rec.lane_q << ",\n";
    oss << "      \"a_load_row_q\": " << rec.a_load_row_q << ",\n";
    oss << "      \"drain_row_q\": " << rec.drain_row_q << ",\n";
    oss << "      \"drain_grp_q\": " << rec.drain_grp_q << ",\n";
    oss << "      \"tile_drain_base_q\": " << rec.tile_drain_base_q << ",\n";
    oss << "      \"drain_row_addr_q\": " << rec.drain_row_addr_q << ",\n";
    oss << "      \"clear_acc\": " << (rec.clear_acc ? "true" : "false") << ",\n";
    oss << "      \"step_en\": " << (rec.step_en ? "true" : "false") << ",\n";
    oss << "      \"sys_sram_a_row\": " << rec.sys_sram_a_row << ",\n";
    oss << "      \"sys_sram_b_row\": " << rec.sys_sram_b_row << ",\n";
    oss << "      \"dst_clear_active\": " << (rec.dst_clear_active ? "true" : "false") << ",\n";
    oss << "      \"dst_clear_row_q\": " << rec.dst_clear_row_q << ",\n";
    oss << "      \"dst_clear_rows_total_q\": " << rec.dst_clear_rows_total_q << "\n";
    oss << "    }";
    if (i + 1 != trace.records.size()) {
      oss << ",";
    }
    oss << "\n";
  }
  oss << "  ]\n";
  oss << "}\n";
  return oss.str();
}

}  // namespace tbutil
