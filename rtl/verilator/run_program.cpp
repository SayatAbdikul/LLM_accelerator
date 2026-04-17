#include "testbench.h"
#include "systolic_debug_artifacts.h"
#include "systolic_window_trace.h"

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr uint32_t kMagic = 0x54414343u;
constexpr uint16_t kLegacyVersion = 0x0001u;
constexpr uint16_t kRuntimeMetadataVersion = 0x0002u;
constexpr uint16_t kVersion = 0x0003u;
constexpr size_t kLegacyHeaderSize = 32;
constexpr size_t kHeaderSize = 64;

struct CliOptions {
    struct ReadInjection {
        uint64_t beat_idx = 0;
        int resp = 0;
        int force_last = -1;
    };

    struct BrespInjection {
        uint64_t resp_idx = 0;
        int resp = 0;
    };

    struct ReadAddrInjection {
        uint64_t addr = 0;
        int resp = 0;
        int force_last = -1;
    };

    std::string program_path;
    std::string json_out_path;
    std::string trace_json_out_path;
    std::string snapshot_request_path;
    std::string snapshot_manifest_out_path;
    std::string snapshot_data_out_path;
    std::string systolic_window_json_out_path;
    std::string accum_write_json_out_path;
    std::string sram_write_json_out_path;
    std::string systolic_hidden_snapshot_json_out_path;
    std::string patches_raw_path;
    std::string cls_raw_path;
    int patch_rows = 0;
    int patch_cols = 0;
    int num_classes = 1000;
    int max_cycles = 500000;
    int latency = 2;
    int systolic_window_start_pc = -1;
    int systolic_window_end_pc = -1;
    int accum_write_start_pc = -1;
    int accum_write_end_pc = -1;
    int sram_write_start_pc = -1;
    int sram_write_end_pc = -1;
    int systolic_hidden_snapshot_pc = -1;
    bool folded_pos_embed = false;
    int inject_next_rresp = -1;
    int inject_next_rlast = -1;
    int inject_next_bresp = -1;
    std::vector<ReadInjection> inject_rresp_at;
    std::vector<ReadAddrInjection> inject_rresp_addr;
    std::vector<BrespInjection> inject_bresp_at;
};

struct ProgramBinary {
    std::vector<uint8_t> instructions;
    std::vector<uint8_t> data;
    uint32_t insn_count = 0;
    uint32_t data_base = 0;
    uint32_t input_offset = 0;
    uint32_t pos_embed_patch_dram_offset = 0;
    uint32_t pos_embed_cls_dram_offset = 0;
    uint32_t cls_token_dram_offset = 0;

    std::vector<uint8_t> to_dram_image() const {
        const size_t pad_to = (data_base > 0)
            ? data_base
            : ((instructions.size() + 15u) & ~size_t(15u));
        if (pad_to < instructions.size()) {
            throw std::runtime_error("Program data_base is smaller than instruction payload");
        }
        std::vector<uint8_t> image = instructions;
        image.resize(pad_to, 0);
        image.insert(image.end(), data.begin(), data.end());
        return image;
    }
};

struct Summary {
    std::string status = "unknown";
    bool done = false;
    bool fault = false;
    int fault_code = 0;
    bool timeout = false;
    std::string parse_error;
    uint64_t cycles = 0;
    uint64_t retired_instructions = 0;
    uint64_t sync_wait_dma_cycles = 0;
    uint64_t sync_wait_sys_cycles = 0;
    uint64_t sync_wait_sfu_cycles = 0;
    uint64_t dma_burst_count = 0;
    uint64_t dma_beat_count = 0;
    uint64_t helper_busy_cycles = 0;
    uint64_t sfu_busy_cycles = 0;
    uint64_t systolic_busy_cycles = 0;
    bool forbidden_overlap_violation = false;
    bool fault_context_valid = false;
    uint64_t fault_pc = 0;
    int fault_opcode = 0;
    bool fault_opcode_valid = false;
    int fault_source = 0;
    int latched_fault_code = 0;
    std::vector<int32_t> logits;
    std::vector<std::string> violations;
};

struct RetireEvent {
    uint64_t cycle = 0;
    uint64_t pc = 0;
    int opcode = 0;
};

struct SnapshotRequest {
    uint64_t pc = 0;
    int event_index = 0;
    std::string node_name;
    int buf_id = 0;
    int offset_units = 0;
    int mem_rows = 0;
    int mem_cols = 0;
    int logical_rows = 0;
    int logical_cols = 0;
    int full_rows = 0;
    int full_cols = 0;
    int row_start = 0;
    std::string dtype;
    double scale = 0.0;
    std::string source;
    std::string capture_phase = "retire_cycle";
};

struct SnapshotCapture {
    SnapshotRequest req;
    std::string status;
    uint64_t cycle = 0;
    uint64_t byte_offset = 0;
    uint64_t byte_size = 0;
};

struct PendingSnapshotCapture {
    SnapshotRequest req;
    uint64_t due_cycle = 0;
};

uint16_t read_be16(const std::vector<uint8_t>& data, size_t off) {
    if (off + 2 > data.size()) {
        throw std::runtime_error("Unexpected EOF while reading BE16");
    }
    return (uint16_t(data[off]) << 8) | uint16_t(data[off + 1]);
}

uint32_t read_be32(const std::vector<uint8_t>& data, size_t off) {
    if (off + 4 > data.size()) {
        throw std::runtime_error("Unexpected EOF while reading BE32");
    }
    return (uint32_t(data[off]) << 24) |
           (uint32_t(data[off + 1]) << 16) |
           (uint32_t(data[off + 2]) << 8) |
           uint32_t(data[off + 3]);
}

std::vector<uint8_t> read_file_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(f),
                                std::istreambuf_iterator<char>());
}

void write_text_file(const std::string& path, const std::string& text) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("Failed to open output file: " + path);
    }
    f << text;
}

void write_binary_file(const std::string& path, const std::vector<uint8_t>& data) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("Failed to open output file: " + path);
    }
    if (!data.empty()) {
        f.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    }
}

std::string json_escape(const std::string& in) {
    std::ostringstream oss;
    for (char ch : in) {
        switch (ch) {
            case '\\': oss << "\\\\"; break;
            case '"':  oss << "\\\""; break;
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

std::string fault_source_name(int src) {
    switch (src) {
        case 0: return "none";
        case 1: return "fetch";
        case 2: return "dma";
        case 3: return "helper";
        case 4: return "sfu";
        case 5: return "sram";
        case 6: return "control";
        default: return "unknown";
    }
}

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    std::stringstream ss(line);
    while (std::getline(ss, field, ',')) {
        fields.push_back(field);
    }
    return fields;
}

std::vector<SnapshotRequest> load_snapshot_requests(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("Failed to open snapshot request file: " + path);
    }

    std::vector<SnapshotRequest> requests;
    std::string line;
    int lineno = 0;
    while (std::getline(f, line)) {
        lineno++;
        if (line.empty()) {
            continue;
        }
        const auto fields = split_csv_line(line);
        if (fields.size() != 15 && fields.size() != 16) {
            std::ostringstream oss;
            oss << "Bad snapshot request line " << lineno << ": expected 15 or 16 CSV fields";
            throw std::runtime_error(oss.str());
        }

        SnapshotRequest req;
        req.pc = std::stoull(fields[0]);
        req.event_index = std::stoi(fields[1]);
        req.node_name = fields[2];
        req.buf_id = std::stoi(fields[3]);
        req.offset_units = std::stoi(fields[4]);
        req.mem_rows = std::stoi(fields[5]);
        req.mem_cols = std::stoi(fields[6]);
        req.logical_rows = std::stoi(fields[7]);
        req.logical_cols = std::stoi(fields[8]);
        req.full_rows = std::stoi(fields[9]);
        req.full_cols = std::stoi(fields[10]);
        req.row_start = std::stoi(fields[11]);
        req.dtype = fields[12];
        req.scale = std::stod(fields[13]);
        req.source = fields[14];
        if (fields.size() == 16) {
            req.capture_phase = fields[15];
        }
        if (req.capture_phase != "retire_cycle" && req.capture_phase != "retire_plus_1") {
            std::ostringstream oss;
            oss << "Bad snapshot request line " << lineno
                << ": unsupported capture_phase " << req.capture_phase;
            throw std::runtime_error(oss.str());
        }
        requests.push_back(req);
    }

    std::sort(
        requests.begin(),
        requests.end(),
        [](const SnapshotRequest& lhs, const SnapshotRequest& rhs) {
            if (lhs.pc != rhs.pc) {
                return lhs.pc < rhs.pc;
            }
            return lhs.event_index < rhs.event_index;
        }
    );
    return requests;
}

std::string snapshot_manifest_to_json(const std::vector<SnapshotCapture>& captures) {
    std::ostringstream oss;
    oss << "{\n";
    oss << "  \"entries\": [\n";
    for (size_t i = 0; i < captures.size(); ++i) {
        const auto& cap = captures[i];
        oss << "    {\n";
        oss << "      \"pc\": " << cap.req.pc << ",\n";
        oss << "      \"event_index\": " << cap.req.event_index << ",\n";
        oss << "      \"node_name\": \"" << json_escape(cap.req.node_name) << "\",\n";
        oss << "      \"buf_id\": " << cap.req.buf_id << ",\n";
        oss << "      \"offset_units\": " << cap.req.offset_units << ",\n";
        oss << "      \"mem_rows\": " << cap.req.mem_rows << ",\n";
        oss << "      \"mem_cols\": " << cap.req.mem_cols << ",\n";
        oss << "      \"logical_rows\": " << cap.req.logical_rows << ",\n";
        oss << "      \"logical_cols\": " << cap.req.logical_cols << ",\n";
        oss << "      \"full_rows\": " << cap.req.full_rows << ",\n";
        oss << "      \"full_cols\": " << cap.req.full_cols << ",\n";
        oss << "      \"row_start\": " << cap.req.row_start << ",\n";
        oss << "      \"dtype\": \"" << json_escape(cap.req.dtype) << "\",\n";
        oss << "      \"scale\": " << cap.req.scale << ",\n";
        oss << "      \"source\": \"" << json_escape(cap.req.source) << "\",\n";
        oss << "      \"capture_phase\": \"" << json_escape(cap.req.capture_phase) << "\",\n";
        oss << "      \"status\": \"" << json_escape(cap.status) << "\",\n";
        oss << "      \"cycle\": " << cap.cycle << ",\n";
        oss << "      \"byte_offset\": " << cap.byte_offset << ",\n";
        oss << "      \"byte_size\": " << cap.byte_size << "\n";
        oss << "    }";
        if (i + 1 != captures.size()) {
            oss << ",";
        }
        oss << "\n";
    }
    oss << "  ]\n";
    oss << "}\n";
    return oss.str();
}

ProgramBinary parse_program_binary(const std::vector<uint8_t>& bytes) {
    if (bytes.size() < kLegacyHeaderSize) {
        throw std::runtime_error("Program too short for header");
    }

    const uint32_t magic = read_be32(bytes, 0);
    const uint16_t version = read_be16(bytes, 4);
    if (magic != kMagic) {
        throw std::runtime_error("Bad program magic");
    }

    ProgramBinary program;
    size_t header_size = 0;
    uint32_t data_offset = 0;
    uint32_t data_size = 0;

    if (version == kLegacyVersion) {
        header_size = kLegacyHeaderSize;
        program.insn_count = read_be32(bytes, 8);
        data_offset = read_be32(bytes, 12);
        data_size = read_be32(bytes, 16);
    } else if (version == kRuntimeMetadataVersion || version == kVersion) {
        if (bytes.size() < kHeaderSize) {
            throw std::runtime_error("Program too short for v2/v3 header");
        }
        header_size = kHeaderSize;
        program.insn_count = read_be32(bytes, 8);
        data_offset = read_be32(bytes, 12);
        data_size = read_be32(bytes, 16);
        program.data_base = read_be32(bytes, 24);
        program.input_offset = read_be32(bytes, 28);
        program.pos_embed_patch_dram_offset = read_be32(bytes, 32);
        program.pos_embed_cls_dram_offset = read_be32(bytes, 36);
        program.cls_token_dram_offset = read_be32(bytes, 40);
        if (version == kVersion) {
            const uint32_t metadata_offset = read_be32(bytes, 44);
            const uint32_t metadata_size = read_be32(bytes, 48);
            if ((metadata_offset != 0 || metadata_size != 0) &&
                (metadata_offset < data_offset + data_size ||
                 uint64_t(metadata_offset) + uint64_t(metadata_size) > bytes.size())) {
                throw std::runtime_error("Invalid metadata range in ProgramBinary");
            }
        }
    } else {
        throw std::runtime_error("Unsupported ProgramBinary version");
    }

    if (data_offset < header_size) {
        throw std::runtime_error("Program data offset overlaps header");
    }
    if (uint64_t(data_offset) + uint64_t(data_size) > bytes.size()) {
        throw std::runtime_error("Program data range exceeds file size");
    }

    program.instructions.assign(bytes.begin() + header_size, bytes.begin() + data_offset);
    program.data.assign(bytes.begin() + data_offset, bytes.begin() + data_offset + data_size);
    return program;
}

int pad16(int value) {
    return (value + 15) & ~15;
}

std::vector<uint8_t> prepare_runtime_patches(const std::vector<uint8_t>& raw,
                                             int rows, int cols) {
    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error("patch_rows and patch_cols must be positive");
    }
    const size_t expected = size_t(rows) * size_t(cols);
    if (raw.size() != expected) {
        std::ostringstream oss;
        oss << "Patch input size mismatch: expected " << expected
            << " bytes, got " << raw.size();
        throw std::runtime_error(oss.str());
    }

    const int cols_padded = pad16(cols);
    std::vector<uint8_t> out(size_t(rows) * size_t(cols_padded), 0);
    for (int r = 0; r < rows; ++r) {
        std::memcpy(out.data() + size_t(r) * size_t(cols_padded),
                    raw.data() + size_t(r) * size_t(cols),
                    size_t(cols));
    }
    return out;
}

void place_runtime_inputs(tbutil::SimHarness& sim,
                          const ProgramBinary& program,
                          const std::vector<uint8_t>& patch_bytes,
                          const std::vector<uint8_t>& cls_bytes,
                          bool folded_pos_embed) {
    if (!patch_bytes.empty()) {
        if (program.input_offset > 0) {
            sim.dram.write_bytes(program.input_offset, patch_bytes.data(), patch_bytes.size());
        } else {
            tbutil::sram_write_bytes(sim.dut.get(), tbutil::BUF_ABUF_ID, 0, patch_bytes);
        }
    }

    if (!cls_bytes.empty() && program.cls_token_dram_offset > 0) {
        sim.dram.write_bytes(program.cls_token_dram_offset, cls_bytes.data(), cls_bytes.size());
    }

    if (!folded_pos_embed) {
        return;
    }

    if (!patch_bytes.empty() && program.pos_embed_patch_dram_offset > 0) {
        std::vector<uint8_t> zeros(patch_bytes.size(), 0);
        sim.dram.write_bytes(program.pos_embed_patch_dram_offset, zeros.data(), zeros.size());
    }

    if (!cls_bytes.empty() && program.pos_embed_cls_dram_offset > 0) {
        std::vector<uint8_t> zeros(cls_bytes.size(), 0);
        sim.dram.write_bytes(program.pos_embed_cls_dram_offset, zeros.data(), zeros.size());
    }
}

size_t required_dram_size(const ProgramBinary& program,
                          const std::vector<uint8_t>& patch_bytes,
                          const std::vector<uint8_t>& cls_bytes) {
    size_t need = std::max<size_t>(16 * 1024 * 1024, program.to_dram_image().size() + 4096);
    if (!patch_bytes.empty()) {
        need = std::max(need, size_t(program.input_offset) + patch_bytes.size() + 4096);
        need = std::max(need, size_t(program.pos_embed_patch_dram_offset) + patch_bytes.size() + 4096);
    }
    if (!cls_bytes.empty()) {
        need = std::max(need, size_t(program.cls_token_dram_offset) + cls_bytes.size() + 4096);
        need = std::max(need, size_t(program.pos_embed_cls_dram_offset) + cls_bytes.size() + 4096);
    }
    return need;
}

Summary build_summary(Vtaccel_top* dut, int num_classes) {
    auto* root = dut->rootp;
    Summary summary;
    summary.done = dut->done;
    summary.fault = dut->fault;
    summary.fault_code = dut->fault_code;
    summary.cycles = root->taccel_top__DOT__obs_cycle_count_q;
    summary.retired_instructions = root->taccel_top__DOT__obs_retired_insn_count_q;
    summary.sync_wait_dma_cycles = root->taccel_top__DOT__obs_sync_wait_dma_cycles_q;
    summary.sync_wait_sys_cycles = root->taccel_top__DOT__obs_sync_wait_sys_cycles_q;
    summary.sync_wait_sfu_cycles = root->taccel_top__DOT__obs_sync_wait_sfu_cycles_q;
    summary.dma_burst_count = root->taccel_top__DOT__obs_dma_burst_count_q;
    summary.dma_beat_count = root->taccel_top__DOT__obs_dma_beat_count_q;
    summary.helper_busy_cycles = root->taccel_top__DOT__obs_helper_busy_cycles_q;
    summary.sfu_busy_cycles = root->taccel_top__DOT__obs_sfu_busy_cycles_q;
    summary.systolic_busy_cycles = root->taccel_top__DOT__obs_sys_busy_cycles_q;
    summary.forbidden_overlap_violation = root->taccel_top__DOT__obs_forbidden_overlap_violation_q;
    summary.fault_context_valid = root->taccel_top__DOT__obs_fault_valid_q;
    summary.fault_pc = root->taccel_top__DOT__obs_fault_pc_q;
    summary.fault_opcode = root->taccel_top__DOT__obs_fault_opcode_q;
    summary.fault_opcode_valid = root->taccel_top__DOT__obs_fault_opcode_valid_q;
    summary.fault_source = root->taccel_top__DOT__obs_fault_source_q;
    summary.latched_fault_code = root->taccel_top__DOT__obs_fault_code_q;

    if (summary.done) {
        summary.status = "halted";
    } else if (summary.fault) {
        summary.status = "fault";
    }

    const auto logits_bytes = tbutil::sram_read_bytes(dut, tbutil::BUF_ACCUM_ID, 0,
                                                      size_t(num_classes) * 4u);
    summary.logits = tbutil::unpack_i32_le(logits_bytes);

    if (summary.fault && summary.fault_code == 0) {
        summary.violations.push_back("terminal_fault_has_zero_fault_code");
    }
    if (summary.fault && summary.fault_context_valid &&
        summary.latched_fault_code == 0) {
        summary.violations.push_back("latched_fault_context_has_zero_fault_code");
    }
    if (summary.forbidden_overlap_violation) {
        summary.violations.push_back("forbidden_async_engine_overlap");
    }

    return summary;
}

std::string summary_to_json(const Summary& summary) {
    std::ostringstream oss;
    oss << "{\n";
    oss << "  \"status\": \"" << json_escape(summary.status) << "\",\n";
    oss << "  \"done\": " << (summary.done ? "true" : "false") << ",\n";
    oss << "  \"fault\": " << (summary.fault ? "true" : "false") << ",\n";
    oss << "  \"fault_code\": " << summary.fault_code << ",\n";
    oss << "  \"timeout\": " << (summary.timeout ? "true" : "false") << ",\n";
    if (!summary.parse_error.empty()) {
        oss << "  \"parse_error\": \"" << json_escape(summary.parse_error) << "\",\n";
    } else {
        oss << "  \"parse_error\": null,\n";
    }
    oss << "  \"cycles\": " << summary.cycles << ",\n";
    oss << "  \"retired_instructions\": " << summary.retired_instructions << ",\n";
    oss << "  \"sync_wait_cycles\": {\n";
    oss << "    \"dma\": " << summary.sync_wait_dma_cycles << ",\n";
    oss << "    \"systolic\": " << summary.sync_wait_sys_cycles << ",\n";
    oss << "    \"sfu\": " << summary.sync_wait_sfu_cycles << "\n";
    oss << "  },\n";
    oss << "  \"dma\": {\n";
    oss << "    \"burst_count\": " << summary.dma_burst_count << ",\n";
    oss << "    \"beat_count\": " << summary.dma_beat_count << "\n";
    oss << "  },\n";
    oss << "  \"busy_cycles\": {\n";
    oss << "    \"helper\": " << summary.helper_busy_cycles << ",\n";
    oss << "    \"sfu\": " << summary.sfu_busy_cycles << ",\n";
    oss << "    \"systolic\": " << summary.systolic_busy_cycles << "\n";
    oss << "  },\n";
    oss << "  \"forbidden_overlap_violation\": "
        << (summary.forbidden_overlap_violation ? "true" : "false") << ",\n";
    oss << "  \"fault_context\": {\n";
    oss << "    \"valid\": " << (summary.fault_context_valid ? "true" : "false") << ",\n";
    oss << "    \"pc\": " << summary.fault_pc << ",\n";
    oss << "    \"opcode\": " << summary.fault_opcode << ",\n";
    oss << "    \"opcode_valid\": " << (summary.fault_opcode_valid ? "true" : "false") << ",\n";
    oss << "    \"source\": " << summary.fault_source << ",\n";
    oss << "    \"source_name\": \"" << fault_source_name(summary.fault_source) << "\",\n";
    oss << "    \"fault_code\": " << summary.latched_fault_code << "\n";
    oss << "  },\n";
    oss << "  \"violations\": [";
    for (size_t i = 0; i < summary.violations.size(); ++i) {
        if (i != 0) {
            oss << ", ";
        }
        oss << "\"" << json_escape(summary.violations[i]) << "\"";
    }
    oss << "],\n";
    oss << "  \"logits\": [";
    for (size_t i = 0; i < summary.logits.size(); ++i) {
        if (i != 0) {
            oss << ", ";
        }
        oss << summary.logits[i];
    }
    oss << "]\n";
    oss << "}\n";
    return oss.str();
}

std::string trace_to_json(const std::vector<RetireEvent>& retire_events,
                          const Summary& summary) {
    std::ostringstream oss;
    oss << "{\n";
    oss << "  \"retire_events\": [\n";
    for (size_t i = 0; i < retire_events.size(); ++i) {
        const auto& ev = retire_events[i];
        oss << "    {\"cycle\": " << ev.cycle
            << ", \"pc\": " << ev.pc
            << ", \"opcode\": " << ev.opcode << "}";
        if (i + 1 != retire_events.size()) {
            oss << ",";
        }
        oss << "\n";
    }
    oss << "  ],\n";
    oss << "  \"final_fault_context\": {\n";
    oss << "    \"valid\": " << (summary.fault_context_valid ? "true" : "false") << ",\n";
    oss << "    \"pc\": " << summary.fault_pc << ",\n";
    oss << "    \"opcode\": " << summary.fault_opcode << ",\n";
    oss << "    \"opcode_valid\": " << (summary.fault_opcode_valid ? "true" : "false") << ",\n";
    oss << "    \"source\": " << summary.fault_source << ",\n";
    oss << "    \"fault_code\": " << summary.latched_fault_code << "\n";
    oss << "  },\n";
    oss << "  \"forbidden_overlap_violation\": "
        << (summary.forbidden_overlap_violation ? "true" : "false") << "\n";
    oss << "}\n";
    return oss.str();
}

void print_usage(const char* argv0) {
    std::cerr
        << "Usage: " << argv0 << " --program <program.bin> --json-out <summary.json> [options]\n"
        << "Options:\n"
        << "  --trace-json-out <path>      Optional retire/fault trace JSON\n"
        << "  --snapshot-request <path>    Optional CSV snapshot request list\n"
        << "  --snapshot-manifest-out <p>  Snapshot manifest JSON output\n"
        << "  --snapshot-data-out <path>   Snapshot raw-data binary output\n"
        << "  --systolic-window-start-pc <pc> Start PC for optional systolic microtrace window\n"
        << "  --systolic-window-end-pc <pc>   End PC for optional systolic microtrace window\n"
        << "  --systolic-window-json-out <p>  Optional systolic microtrace JSON output\n"
        << "  --accum-write-start-pc <pc>   Start PC for optional ACCUM write provenance window\n"
        << "  --accum-write-end-pc <pc>     End PC for optional ACCUM write provenance window\n"
        << "  --accum-write-json-out <path> Optional ACCUM write provenance JSON output\n"
        << "  --sram-write-start-pc <pc>    Start PC for optional generic SRAM write provenance window\n"
        << "  --sram-write-end-pc <pc>      End PC for optional generic SRAM write provenance window\n"
        << "  --sram-write-json-out <path>  Optional generic SRAM write provenance JSON output\n"
        << "  --systolic-hidden-snapshot-pc <pc> Tagged retire PC for optional hidden systolic snapshot\n"
        << "  --systolic-hidden-snapshot-json-out <path> Optional hidden systolic snapshot JSON output\n"
        << "  --patches-raw <path>         Raw INT8 patch rows\n"
        << "  --patch-rows <rows>          Patch row count for raw input\n"
        << "  --patch-cols <cols>          Patch column count before 16-byte padding\n"
        << "  --cls-raw <path>             Optional raw INT8 CLS row\n"
        << "  --folded-pos-embed           Zero folded position-embedding regions\n"
        << "  --num-classes <n>            Number of ACCUM logits to dump (default 1000)\n"
        << "  --max-cycles <n>             Timeout budget (default 500000)\n"
        << "  --latency <n>                AXI memory latency in cycles (default 2)\n"
        << "  --inject-next-rresp <0..3>   Inject next AXI read response code\n"
        << "  --inject-next-rlast <0|1>    Override next AXI read last flag\n"
        << "  --inject-next-bresp <0..3>   Inject next AXI write response code\n"
        << "  --inject-rresp-at <b:r[:l]>  Inject read beat b with resp r and optional last l\n"
        << "  --inject-rresp-addr <a:r[:l]> Inject read address a with resp r and optional last l\n"
        << "  --inject-bresp-at <b:r>      Inject write response b with resp r\n";
}

CliOptions::ReadInjection parse_read_injection_spec(const std::string& spec) {
    CliOptions::ReadInjection inj;
    size_t first = spec.find(':');
    if (first == std::string::npos) {
        throw std::runtime_error("Bad --inject-rresp-at format");
    }
    size_t second = spec.find(':', first + 1);
    inj.beat_idx = std::stoull(spec.substr(0, first));
    inj.resp = std::stoi(spec.substr(first + 1, second == std::string::npos
                                                  ? std::string::npos
                                                  : second - first - 1));
    if (second != std::string::npos) {
        inj.force_last = std::stoi(spec.substr(second + 1));
    }
    return inj;
}

CliOptions::BrespInjection parse_bresp_injection_spec(const std::string& spec) {
    CliOptions::BrespInjection inj;
    size_t first = spec.find(':');
    if (first == std::string::npos) {
        throw std::runtime_error("Bad --inject-bresp-at format");
    }
    inj.resp_idx = std::stoull(spec.substr(0, first));
    inj.resp = std::stoi(spec.substr(first + 1));
    return inj;
}

CliOptions::ReadAddrInjection parse_read_addr_injection_spec(const std::string& spec) {
    CliOptions::ReadAddrInjection inj;
    size_t first = spec.find(':');
    if (first == std::string::npos) {
        throw std::runtime_error("Bad --inject-rresp-addr format");
    }
    size_t second = spec.find(':', first + 1);
    inj.addr = std::stoull(spec.substr(0, first), nullptr, 0);
    inj.resp = std::stoi(spec.substr(first + 1, second == std::string::npos
                                                  ? std::string::npos
                                                  : second - first - 1));
    if (second != std::string::npos) {
        inj.force_last = std::stoi(spec.substr(second + 1));
    }
    return inj;
}

CliOptions parse_args(int argc, char** argv) {
    CliOptions opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("Missing value for ") + name);
            }
            return argv[++i];
        };

        if (arg == "--program") {
            opts.program_path = need_value("--program");
        } else if (arg == "--json-out") {
            opts.json_out_path = need_value("--json-out");
        } else if (arg == "--trace-json-out") {
            opts.trace_json_out_path = need_value("--trace-json-out");
        } else if (arg == "--snapshot-request") {
            opts.snapshot_request_path = need_value("--snapshot-request");
        } else if (arg == "--snapshot-manifest-out") {
            opts.snapshot_manifest_out_path = need_value("--snapshot-manifest-out");
        } else if (arg == "--snapshot-data-out") {
            opts.snapshot_data_out_path = need_value("--snapshot-data-out");
        } else if (arg == "--systolic-window-start-pc") {
            opts.systolic_window_start_pc = std::stoi(need_value("--systolic-window-start-pc"));
        } else if (arg == "--systolic-window-end-pc") {
            opts.systolic_window_end_pc = std::stoi(need_value("--systolic-window-end-pc"));
        } else if (arg == "--systolic-window-json-out") {
            opts.systolic_window_json_out_path = need_value("--systolic-window-json-out");
        } else if (arg == "--accum-write-start-pc") {
            opts.accum_write_start_pc = std::stoi(need_value("--accum-write-start-pc"));
        } else if (arg == "--accum-write-end-pc") {
            opts.accum_write_end_pc = std::stoi(need_value("--accum-write-end-pc"));
        } else if (arg == "--accum-write-json-out") {
            opts.accum_write_json_out_path = need_value("--accum-write-json-out");
        } else if (arg == "--sram-write-start-pc") {
            opts.sram_write_start_pc = std::stoi(need_value("--sram-write-start-pc"));
        } else if (arg == "--sram-write-end-pc") {
            opts.sram_write_end_pc = std::stoi(need_value("--sram-write-end-pc"));
        } else if (arg == "--sram-write-json-out") {
            opts.sram_write_json_out_path = need_value("--sram-write-json-out");
        } else if (arg == "--systolic-hidden-snapshot-pc") {
            opts.systolic_hidden_snapshot_pc = std::stoi(need_value("--systolic-hidden-snapshot-pc"));
        } else if (arg == "--systolic-hidden-snapshot-json-out") {
            opts.systolic_hidden_snapshot_json_out_path = need_value("--systolic-hidden-snapshot-json-out");
        } else if (arg == "--patches-raw") {
            opts.patches_raw_path = need_value("--patches-raw");
        } else if (arg == "--patch-rows") {
            opts.patch_rows = std::stoi(need_value("--patch-rows"));
        } else if (arg == "--patch-cols") {
            opts.patch_cols = std::stoi(need_value("--patch-cols"));
        } else if (arg == "--cls-raw") {
            opts.cls_raw_path = need_value("--cls-raw");
        } else if (arg == "--folded-pos-embed") {
            opts.folded_pos_embed = true;
        } else if (arg == "--num-classes") {
            opts.num_classes = std::stoi(need_value("--num-classes"));
        } else if (arg == "--max-cycles") {
            opts.max_cycles = std::stoi(need_value("--max-cycles"));
        } else if (arg == "--latency") {
            opts.latency = std::stoi(need_value("--latency"));
        } else if (arg == "--inject-next-rresp") {
            opts.inject_next_rresp = std::stoi(need_value("--inject-next-rresp"));
        } else if (arg == "--inject-next-rlast") {
            opts.inject_next_rlast = std::stoi(need_value("--inject-next-rlast"));
        } else if (arg == "--inject-next-bresp") {
            opts.inject_next_bresp = std::stoi(need_value("--inject-next-bresp"));
        } else if (arg == "--inject-rresp-at") {
            opts.inject_rresp_at.push_back(parse_read_injection_spec(need_value("--inject-rresp-at")));
        } else if (arg == "--inject-rresp-addr") {
            opts.inject_rresp_addr.push_back(parse_read_addr_injection_spec(need_value("--inject-rresp-addr")));
        } else if (arg == "--inject-bresp-at") {
            opts.inject_bresp_at.push_back(parse_bresp_injection_spec(need_value("--inject-bresp-at")));
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (opts.program_path.empty()) {
        throw std::runtime_error("--program is required");
    }
    if (opts.json_out_path.empty()) {
        throw std::runtime_error("--json-out is required");
    }
    if (!opts.patches_raw_path.empty() && (opts.patch_rows <= 0 || opts.patch_cols <= 0)) {
        throw std::runtime_error("--patch-rows and --patch-cols are required with --patches-raw");
    }
    const bool snapshot_enabled = !opts.snapshot_request_path.empty() ||
                                  !opts.snapshot_manifest_out_path.empty() ||
                                  !opts.snapshot_data_out_path.empty();
    if (snapshot_enabled &&
        (opts.snapshot_request_path.empty() ||
         opts.snapshot_manifest_out_path.empty() ||
         opts.snapshot_data_out_path.empty())) {
        throw std::runtime_error(
            "--snapshot-request, --snapshot-manifest-out, and --snapshot-data-out must be used together"
        );
    }
    const bool window_enabled = !opts.systolic_window_json_out_path.empty() ||
                                opts.systolic_window_start_pc >= 0 ||
                                opts.systolic_window_end_pc >= 0;
    if (window_enabled &&
        (opts.systolic_window_json_out_path.empty() ||
         opts.systolic_window_start_pc < 0 ||
         opts.systolic_window_end_pc < 0)) {
        throw std::runtime_error(
            "--systolic-window-start-pc, --systolic-window-end-pc, and --systolic-window-json-out must be used together"
        );
    }
    const bool accum_write_enabled = !opts.accum_write_json_out_path.empty() ||
                                     opts.accum_write_start_pc >= 0 ||
                                     opts.accum_write_end_pc >= 0;
    if (accum_write_enabled &&
        (opts.accum_write_json_out_path.empty() ||
         opts.accum_write_start_pc < 0 ||
         opts.accum_write_end_pc < 0)) {
        throw std::runtime_error(
            "--accum-write-start-pc, --accum-write-end-pc, and --accum-write-json-out must be used together"
        );
    }
    const bool sram_write_enabled = !opts.sram_write_json_out_path.empty() ||
                                    opts.sram_write_start_pc >= 0 ||
                                    opts.sram_write_end_pc >= 0;
    if (sram_write_enabled &&
        (opts.sram_write_json_out_path.empty() ||
         opts.sram_write_start_pc < 0 ||
         opts.sram_write_end_pc < 0)) {
        throw std::runtime_error(
            "--sram-write-start-pc, --sram-write-end-pc, and --sram-write-json-out must be used together"
        );
    }
    const bool hidden_snapshot_enabled = !opts.systolic_hidden_snapshot_json_out_path.empty() ||
                                         opts.systolic_hidden_snapshot_pc >= 0;
    if (hidden_snapshot_enabled &&
        (opts.systolic_hidden_snapshot_json_out_path.empty() ||
         opts.systolic_hidden_snapshot_pc < 0)) {
        throw std::runtime_error(
            "--systolic-hidden-snapshot-pc and --systolic-hidden-snapshot-json-out must be used together"
        );
    }
    return opts;
}

} // namespace

int main(int argc, char** argv) {
    CliOptions opts;
    try {
        opts = parse_args(argc, argv);
    } catch (const std::exception& ex) {
        print_usage(argv[0]);
        std::cerr << ex.what() << "\n";
        return 2;
    }

    Summary summary;
    std::vector<RetireEvent> retire_events;
    std::vector<SnapshotCapture> snapshot_captures;
    std::vector<uint8_t> snapshot_bytes;
    tbutil::SystolicWindowCollector window_collector(0, 0);
    bool window_trace_enabled = false;
    tbutil::AccumWriteLogCollector accum_write_collector(0, 0);
    bool accum_write_log_enabled = false;
    tbutil::SramWriteLogCollector sram_write_collector(0, 0);
    bool sram_write_log_enabled = false;
    tbutil::SystolicHiddenSnapshotCollector hidden_snapshot_collector(0);
    bool hidden_snapshot_enabled = false;

    try {
        const auto program_bytes = read_file_bytes(opts.program_path);
        const auto program = parse_program_binary(program_bytes);
        const auto snapshot_requests = opts.snapshot_request_path.empty()
            ? std::vector<SnapshotRequest>{}
            : load_snapshot_requests(opts.snapshot_request_path);
        const auto dram_image = program.to_dram_image();
        const auto patch_raw = opts.patches_raw_path.empty()
            ? std::vector<uint8_t>{}
            : read_file_bytes(opts.patches_raw_path);
        const auto cls_raw = opts.cls_raw_path.empty()
            ? std::vector<uint8_t>{}
            : read_file_bytes(opts.cls_raw_path);
        const auto patch_bytes = patch_raw.empty()
            ? std::vector<uint8_t>{}
            : prepare_runtime_patches(patch_raw, opts.patch_rows, opts.patch_cols);

        tbutil::SimHarness sim(required_dram_size(program, patch_bytes, cls_raw));
        sim.dram.write_bytes(0, dram_image.data(), dram_image.size());
        place_runtime_inputs(sim, program, patch_bytes, cls_raw, opts.folded_pos_embed);

        if (opts.inject_next_rresp >= 0 || opts.inject_next_rlast >= 0) {
            sim.dram.inject_next_read(
                (opts.inject_next_rresp >= 0) ? opts.inject_next_rresp : 0,
                opts.inject_next_rlast
            );
        }
        if (opts.inject_next_bresp >= 0) {
            sim.dram.inject_next_bresp(opts.inject_next_bresp);
        }
        for (const auto& inj : opts.inject_rresp_at) {
            sim.dram.inject_read_at(inj.beat_idx, inj.resp, inj.force_last);
        }
        for (const auto& inj : opts.inject_rresp_addr) {
            sim.dram.inject_read_addr(inj.addr, inj.resp, inj.force_last);
        }
        for (const auto& inj : opts.inject_bresp_at) {
            sim.dram.inject_bresp_at(inj.resp_idx, inj.resp);
        }

        sim.start_once(opts.latency);
        auto* root = sim.dut->rootp;
        window_trace_enabled = !opts.systolic_window_json_out_path.empty();
        if (window_trace_enabled) {
            window_collector = tbutil::SystolicWindowCollector(
                uint64_t(opts.systolic_window_start_pc),
                uint64_t(opts.systolic_window_end_pc)
            );
        }
        accum_write_log_enabled = !opts.accum_write_json_out_path.empty();
        if (accum_write_log_enabled) {
            accum_write_collector = tbutil::AccumWriteLogCollector(
                uint64_t(opts.accum_write_start_pc),
                uint64_t(opts.accum_write_end_pc)
            );
        }
        sram_write_log_enabled = !opts.sram_write_json_out_path.empty();
        if (sram_write_log_enabled) {
            sram_write_collector = tbutil::SramWriteLogCollector(
                uint64_t(opts.sram_write_start_pc),
                uint64_t(opts.sram_write_end_pc)
            );
        }
        hidden_snapshot_enabled = !opts.systolic_hidden_snapshot_json_out_path.empty();
        if (hidden_snapshot_enabled) {
            hidden_snapshot_collector = tbutil::SystolicHiddenSnapshotCollector(
                uint64_t(opts.systolic_hidden_snapshot_pc)
            );
        }
        size_t next_snapshot_req = 0;
        std::vector<PendingSnapshotCapture> pending_snapshot_reqs;
        const bool want_retire_trace = !opts.trace_json_out_path.empty();
        const bool want_snapshots = !snapshot_requests.empty();

        auto capture_missing_snapshot_requests_until = [&](uint64_t retired_pc) {
            while (next_snapshot_req < snapshot_requests.size() &&
                   snapshot_requests[next_snapshot_req].pc < retired_pc) {
                snapshot_captures.push_back(
                    SnapshotCapture{
                        snapshot_requests[next_snapshot_req],
                        "missing_retire",
                        0,
                        0,
                        0,
                    }
                );
                next_snapshot_req++;
            }
        };

        auto capture_snapshot_request = [&](const SnapshotRequest& req, uint64_t cycle) {
            SnapshotCapture cap;
            cap.req = req;
            cap.cycle = cycle;
            if (req.source == "virtual") {
                cap.status = "skipped_virtual";
            } else if (req.dtype == "int8") {
                cap.status = "captured";
                cap.byte_offset = snapshot_bytes.size();
                cap.byte_size = uint64_t(req.logical_rows) * uint64_t(req.logical_cols);
                // Use tile-layout reader: handles both padded (mem_cols > logical_cols)
                // and unpadded (mem_cols == logical_cols) cases correctly.
                const auto bytes = tbutil::sfu_read_logical_i8(
                    sim.dut.get(),
                    req.buf_id,
                    req.offset_units, req.mem_cols,
                    req.logical_rows, req.logical_cols
                );
                snapshot_bytes.insert(snapshot_bytes.end(), bytes.begin(), bytes.end());
            } else if (req.dtype == "int32") {
                cap.status = "captured";
                cap.byte_offset = snapshot_bytes.size();
                cap.byte_size = uint64_t(req.logical_rows) * uint64_t(req.logical_cols) * 4u;
                std::vector<uint8_t> bytes;
                if (req.buf_id == tbutil::BUF_ACCUM_ID) {
                    // ACCUM uses tile-major physical layout; convert to logical row-major.
                    bytes = tbutil::accum_read_logical_i32(
                        sim.dut.get(),
                        req.offset_units, req.mem_cols,
                        req.logical_rows, req.logical_cols
                    );
                } else {
                    bytes = tbutil::sram_read_bytes(
                        sim.dut.get(),
                        req.buf_id,
                        size_t(req.offset_units) * 16u,
                        size_t(cap.byte_size)
                    );
                }
                snapshot_bytes.insert(snapshot_bytes.end(), bytes.begin(), bytes.end());
            } else {
                cap.status = "unsupported_dtype";
            }
            snapshot_captures.push_back(cap);
        };

        auto capture_due_pending_snapshot_requests = [&](uint64_t cycle) {
            if (!want_snapshots || pending_snapshot_reqs.empty()) {
                return;
            }
            auto it = pending_snapshot_reqs.begin();
            while (it != pending_snapshot_reqs.end()) {
                if (it->due_cycle <= cycle) {
                    capture_snapshot_request(it->req, cycle);
                    it = pending_snapshot_reqs.erase(it);
                } else {
                    ++it;
                }
            }
        };

        auto capture_snapshot_requests_for_pc = [&](uint64_t cycle, uint64_t retired_pc) {
            if (!want_snapshots) {
                return;
            }
            capture_missing_snapshot_requests_until(retired_pc);
            while (next_snapshot_req < snapshot_requests.size() &&
                   snapshot_requests[next_snapshot_req].pc == retired_pc) {
                const auto& req = snapshot_requests[next_snapshot_req];
                if (req.capture_phase == "retire_plus_1") {
                    pending_snapshot_reqs.push_back(PendingSnapshotCapture{req, cycle + 1});
                } else {
                    capture_snapshot_request(req, cycle);
                }
                next_snapshot_req++;
            }
        };

        auto process_current_cycle = [&]() {
            const bool retire_valid = root->taccel_top__DOT__obs_retire_pulse_w;
            const uint64_t retire_cycle = root->taccel_top__DOT__obs_cycle_count_q;
            const uint64_t retire_pc = root->taccel_top__DOT__obs_retire_pc_w;
            const int retire_opcode = int(root->taccel_top__DOT__obs_retire_opcode_w);
            capture_due_pending_snapshot_requests(retire_cycle);
            if (retire_valid) {
                if (want_retire_trace) {
                    retire_events.push_back(RetireEvent{retire_cycle, retire_pc, retire_opcode});
                }
                capture_snapshot_requests_for_pc(retire_cycle, retire_pc);
            }
            if (window_trace_enabled) {
                window_collector.observe(root, retire_valid, retire_pc, retire_opcode);
            }
            if (accum_write_log_enabled) {
                accum_write_collector.observe(root, retire_valid, retire_pc, retire_opcode);
            }
            if (sram_write_log_enabled) {
                sram_write_collector.observe(root, retire_valid, retire_pc, retire_opcode);
            }
            if (hidden_snapshot_enabled) {
                hidden_snapshot_collector.observe(root, retire_valid, retire_pc, retire_opcode);
            }
        };

        auto process_negedge_sram_writes = [&]() {
            if (!sram_write_log_enabled) {
                return;
            }
            sram_write_collector.observe(
                root,
                root->taccel_top__DOT__obs_retire_pulse_w,
                root->taccel_top__DOT__obs_retire_pc_w,
                int(root->taccel_top__DOT__obs_retire_opcode_w));
        };

        bool terminated = sim.dut->done || sim.dut->fault;
        process_current_cycle();

        for (int i = 0; !terminated && i < opts.max_cycles; ++i) {
            tick_with_negedge_observer(sim.dut.get(), sim.dram, process_negedge_sram_writes, opts.latency);
            process_current_cycle();
            terminated = sim.dut->done || sim.dut->fault;
        }

        // The top-level observability counters/context latch submodule pulses
        // one cycle after the retiring/faulting event. Advance one extra cycle
        // after HALT/FAULT so the summary reflects the terminal instruction.
        if (terminated) {
            tick_with_negedge_observer(sim.dut.get(), sim.dram, process_negedge_sram_writes, opts.latency);
            process_current_cycle();
        }

        summary = build_summary(sim.dut.get(), opts.num_classes);
        while (next_snapshot_req < snapshot_requests.size()) {
            snapshot_captures.push_back(
                SnapshotCapture{
                    snapshot_requests[next_snapshot_req],
                    "missing_retire",
                    0,
                    0,
                    0,
                }
            );
            next_snapshot_req++;
        }
        for (const auto& pending : pending_snapshot_reqs) {
            snapshot_captures.push_back(
                SnapshotCapture{
                    pending.req,
                    "missing_capture_phase",
                    0,
                    0,
                    0,
                }
            );
        }
        if (!terminated) {
            summary.status = "timeout";
            summary.timeout = true;
            summary.violations.push_back("cycle_budget_exhausted");
        }
    } catch (const std::exception& ex) {
        summary.status = "parse_error";
        summary.parse_error = ex.what();
        write_text_file(opts.json_out_path, summary_to_json(summary));
        if (!opts.trace_json_out_path.empty()) {
            write_text_file(opts.trace_json_out_path, trace_to_json(retire_events, summary));
        }
        if (!opts.systolic_window_json_out_path.empty()) {
            tbutil::SystolicWindowTrace trace;
            trace.window_start_pc = uint64_t(opts.systolic_window_start_pc >= 0 ? opts.systolic_window_start_pc : 0);
            trace.window_end_pc = uint64_t(opts.systolic_window_end_pc >= 0 ? opts.systolic_window_end_pc : 0);
            trace.reason = "parse_error";
            write_text_file(opts.systolic_window_json_out_path, tbutil::systolic_window_trace_to_json(trace));
        }
        if (!opts.accum_write_json_out_path.empty()) {
            tbutil::AccumWriteLog log;
            log.window_start_pc = uint64_t(opts.accum_write_start_pc >= 0 ? opts.accum_write_start_pc : 0);
            log.window_end_pc = uint64_t(opts.accum_write_end_pc >= 0 ? opts.accum_write_end_pc : 0);
            log.reason = "parse_error";
            write_text_file(opts.accum_write_json_out_path, tbutil::accum_write_log_to_json(log));
        }
        if (!opts.sram_write_json_out_path.empty()) {
            tbutil::SramWriteLog log;
            log.window_start_pc = uint64_t(opts.sram_write_start_pc >= 0 ? opts.sram_write_start_pc : 0);
            log.window_end_pc = uint64_t(opts.sram_write_end_pc >= 0 ? opts.sram_write_end_pc : 0);
            log.reason = "parse_error";
            write_text_file(opts.sram_write_json_out_path, tbutil::sram_write_log_to_json(log));
        }
        if (!opts.systolic_hidden_snapshot_json_out_path.empty()) {
            tbutil::SystolicHiddenSnapshot snapshot;
            snapshot.requested_pc = uint64_t(opts.systolic_hidden_snapshot_pc >= 0 ? opts.systolic_hidden_snapshot_pc : 0);
            snapshot.reason = "parse_error";
            write_text_file(
                opts.systolic_hidden_snapshot_json_out_path,
                tbutil::hidden_snapshot_to_json(snapshot)
            );
        }
        return 2;
    }

    write_text_file(opts.json_out_path, summary_to_json(summary));
    if (!opts.trace_json_out_path.empty()) {
        write_text_file(opts.trace_json_out_path, trace_to_json(retire_events, summary));
    }
    if (!opts.snapshot_manifest_out_path.empty()) {
        write_text_file(opts.snapshot_manifest_out_path, snapshot_manifest_to_json(snapshot_captures));
        write_binary_file(opts.snapshot_data_out_path, snapshot_bytes);
    }
    if (!opts.systolic_window_json_out_path.empty()) {
        write_text_file(
            opts.systolic_window_json_out_path,
            tbutil::systolic_window_trace_to_json(window_collector.finish())
        );
    }
    if (!opts.accum_write_json_out_path.empty()) {
        write_text_file(
            opts.accum_write_json_out_path,
            tbutil::accum_write_log_to_json(accum_write_collector.finish())
        );
    }
    if (!opts.sram_write_json_out_path.empty()) {
        write_text_file(
            opts.sram_write_json_out_path,
            tbutil::sram_write_log_to_json(sram_write_collector.finish())
        );
    }
    if (!opts.systolic_hidden_snapshot_json_out_path.empty()) {
        write_text_file(
            opts.systolic_hidden_snapshot_json_out_path,
            tbutil::hidden_snapshot_to_json(hidden_snapshot_collector.finish())
        );
    }

    if (summary.timeout) {
        return 3;
    }
    if (!summary.violations.empty()) {
        return 4;
    }
    return 0;
}
