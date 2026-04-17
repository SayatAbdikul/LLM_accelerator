// Common Verilator testbench infrastructure for TACCEL Phase 1.
//
// Provides:
//   - ClockGen: drives clk + rst_n
//   - AXI4SlaveModel: behavioral DRAM with instruction-write helpers
//   - run_until_halt(): drives simulation until done or fault or timeout
//
// Usage pattern:
//   auto tb = std::make_unique<Vtaccel_top>();
//   AXI4SlaveModel dram(1 << 24);  // 16 MB
//   dram.write_insn(tb, 0, NOP_WORD);
//   dram.write_insn(tb, 1, HALT_WORD);
//   reset(tb);
//   run_until_halt(tb, dram, /*timeout=*/1000);
//   assert(tb->done == 1);

#pragma once

#include "Vtaccel_top.h"
#include "Vtaccel_top___024root.h"
#include "verilated.h"
#include "verilated_vcd_c.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>
#include <string>
#include <array>
#include <algorithm>
#include <utility>

inline int sfu_round_half_even_fp32_scalar(float x) {
    const long long floor_i = static_cast<long long>(std::floor(x));
    const float frac = x - static_cast<float>(floor_i);
    if (frac > 0.5f)
        return static_cast<int>(floor_i + 1);
    if (frac < 0.5f)
        return static_cast<int>(floor_i);
    if (floor_i & 1LL)
        return static_cast<int>(floor_i + 1);
    return static_cast<int>(floor_i);
}

extern "C" double sfu_fp32_round(double value_r) {
    return static_cast<double>(static_cast<float>(value_r));
}

extern "C" double sfu_fp32_add(double lhs_r, double rhs_r) {
    return static_cast<double>(static_cast<float>(static_cast<float>(lhs_r) + static_cast<float>(rhs_r)));
}

extern "C" double sfu_fp32_sub(double lhs_r, double rhs_r) {
    return static_cast<double>(static_cast<float>(static_cast<float>(lhs_r) - static_cast<float>(rhs_r)));
}

extern "C" double sfu_fp32_mul(double lhs_r, double rhs_r) {
    return static_cast<double>(static_cast<float>(static_cast<float>(lhs_r) * static_cast<float>(rhs_r)));
}

extern "C" double sfu_fp32_div(double lhs_r, double rhs_r) {
    return static_cast<double>(static_cast<float>(static_cast<float>(lhs_r) / static_cast<float>(rhs_r)));
}

extern "C" double sfu_fp32_exp(double value_r) {
    return static_cast<double>(static_cast<float>(std::exp(static_cast<float>(value_r))));
}

extern "C" double sfu_fp32_sqrt(double value_r) {
    return static_cast<double>(static_cast<float>(std::sqrt(static_cast<float>(value_r))));
}

extern "C" double sfu_fp32_gelu(double value_r) {
    const float x = static_cast<float>(value_r);
    const float inv_sqrt2 = 1.0f / std::sqrt(2.0f);
    const float y = x * 0.5f * (1.0f + std::erf(x * inv_sqrt2));
    return static_cast<double>(y);
}

extern "C" int sfu_fp32_quantize_i8(double value_r, double out_scale_r) {
    const float out_scale = static_cast<float>(out_scale_r);
    if (out_scale == 0.0f)
        return 0;
    int q = sfu_round_half_even_fp32_scalar(static_cast<float>(value_r) / out_scale);
    q = std::clamp(q, -128, 127);
    return q;
}

// ============================================================================
// ISA helper: build 64-bit big-endian instruction words
// (Matches software/taccel/isa/encoding.py exactly)
// ============================================================================
namespace insn {

constexpr int OPCODE_SHIFT      = 59;
constexpr int R_SRC1_BUF_SHIFT  = 57;
constexpr int R_SRC1_OFF_SHIFT  = 41;
constexpr int R_SRC2_BUF_SHIFT  = 39;
constexpr int R_SRC2_OFF_SHIFT  = 23;
constexpr int R_DST_BUF_SHIFT   = 21;
constexpr int R_DST_OFF_SHIFT   = 5;
constexpr int R_SREG_SHIFT      = 1;
constexpr int R_FLAGS_SHIFT     = 0;

constexpr int M_BUF_ID_SHIFT    = 57;
constexpr int M_SRAM_OFF_SHIFT  = 41;
constexpr int M_XFER_LEN_SHIFT  = 25;
constexpr int M_ADDR_REG_SHIFT  = 23;
constexpr int M_DRAM_OFF_SHIFT  = 7;

constexpr int B_SRC_BUF_SHIFT   = 57;
constexpr int B_SRC_OFF_SHIFT   = 41;
constexpr int B_DST_BUF_SHIFT   = 39;
constexpr int B_DST_OFF_SHIFT   = 23;
constexpr int B_LENGTH_SHIFT    = 7;
constexpr int B_SRC_ROWS_SHIFT  = 1;
constexpr int B_TRANSPOSE_SHIFT = 0;

constexpr int A_ADDR_REG_SHIFT  = 57;
constexpr int A_IMM28_SHIFT     = 29;

constexpr int C_M_SHIFT         = 49;
constexpr int C_N_SHIFT         = 39;
constexpr int C_K_SHIFT         = 29;

constexpr int SS_SREG_SHIFT     = 55;
constexpr int SS_SRC_MODE_SHIFT = 53;
constexpr int SS_IMM16_SHIFT    = 37;
constexpr int SYNC_MASK_SHIFT   = 56;

constexpr uint64_t NOP()         { return uint64_t(0x00) << OPCODE_SHIFT; }
constexpr uint64_t HALT()        { return uint64_t(0x01) << OPCODE_SHIFT; }
inline    uint64_t SYNC(uint8_t mask) {
    return (uint64_t(0x02) << OPCODE_SHIFT) |
           (uint64_t(mask & 0x7) << SYNC_MASK_SHIFT);
}
inline uint64_t CONFIG_TILE(int M, int N, int K) {
    assert(M >= 1 && M <= 1024 && N >= 1 && N <= 1024 && K >= 1 && K <= 1024);
    return (uint64_t(0x03) << OPCODE_SHIFT)   |
           (uint64_t(M-1)  << C_M_SHIFT)      |
           (uint64_t(N-1)  << C_N_SHIFT)      |
           (uint64_t(K-1)  << C_K_SHIFT);
}
inline uint64_t SET_SCALE(int sreg, uint16_t fp16_val, int src_mode = 0) {
    return (uint64_t(0x04)        << OPCODE_SHIFT)   |
           (uint64_t(sreg & 0xF)  << SS_SREG_SHIFT)  |
           (uint64_t(src_mode&3)  << SS_SRC_MODE_SHIFT) |
           (uint64_t(fp16_val)    << SS_IMM16_SHIFT);
}
inline uint64_t SET_ADDR_LO(int reg, uint32_t imm28) {
    return (uint64_t(0x05) << OPCODE_SHIFT) |
           (uint64_t(reg & 3) << A_ADDR_REG_SHIFT) |
           (uint64_t(imm28 & 0xFFFFFFF) << A_IMM28_SHIFT);
}
inline uint64_t SET_ADDR_HI(int reg, uint32_t imm28) {
    return (uint64_t(0x06) << OPCODE_SHIFT) |
           (uint64_t(reg & 3) << A_ADDR_REG_SHIFT) |
           (uint64_t(imm28 & 0xFFFFFFF) << A_IMM28_SHIFT);
}
inline uint64_t LOAD(int buf_id, int sram_off, int xfer_len,
                     int addr_reg, int dram_off) {
    return (uint64_t(0x07)       << OPCODE_SHIFT)  |
           (uint64_t(buf_id&3)   << M_BUF_ID_SHIFT)  |
           (uint64_t(sram_off)   << M_SRAM_OFF_SHIFT) |
           (uint64_t(xfer_len)   << M_XFER_LEN_SHIFT) |
           (uint64_t(addr_reg&3) << M_ADDR_REG_SHIFT) |
           (uint64_t(dram_off)   << M_DRAM_OFF_SHIFT);
}
inline uint64_t STORE(int buf_id, int sram_off, int xfer_len,
                      int addr_reg, int dram_off) {
    return (uint64_t(0x08)       << OPCODE_SHIFT)    |
           (uint64_t(buf_id&3)   << M_BUF_ID_SHIFT)  |
           (uint64_t(sram_off)   << M_SRAM_OFF_SHIFT) |
           (uint64_t(xfer_len)   << M_XFER_LEN_SHIFT) |
           (uint64_t(addr_reg&3) << M_ADDR_REG_SHIFT) |
           (uint64_t(dram_off)   << M_DRAM_OFF_SHIFT);
}
inline uint64_t BUF_COPY(int src_buf, int src_off, int dst_buf, int dst_off,
                         int length, int src_rows, int transpose = 0) {
    return (uint64_t(0x09)        << OPCODE_SHIFT)      |
           (uint64_t(src_buf & 3) << B_SRC_BUF_SHIFT)   |
           (uint64_t(src_off)     << B_SRC_OFF_SHIFT)   |
           (uint64_t(dst_buf & 3) << B_DST_BUF_SHIFT)   |
           (uint64_t(dst_off)     << B_DST_OFF_SHIFT)   |
           (uint64_t(length)      << B_LENGTH_SHIFT)    |
           (uint64_t(src_rows)    << B_SRC_ROWS_SHIFT)  |
           (uint64_t(transpose & 1) << B_TRANSPOSE_SHIFT);
}
inline uint64_t R_TYPE(int opcode, int src1_buf, int src1_off, int src2_buf,
                       int src2_off, int dst_buf, int dst_off, int sreg,
                       int flags = 0) {
    return (uint64_t(opcode & 0x1F) << OPCODE_SHIFT)   |
           (uint64_t(src1_buf & 3)  << R_SRC1_BUF_SHIFT) |
           (uint64_t(src1_off)      << R_SRC1_OFF_SHIFT) |
           (uint64_t(src2_buf & 3)  << R_SRC2_BUF_SHIFT) |
           (uint64_t(src2_off)      << R_SRC2_OFF_SHIFT) |
           (uint64_t(dst_buf & 3)   << R_DST_BUF_SHIFT) |
           (uint64_t(dst_off)       << R_DST_OFF_SHIFT) |
           (uint64_t(sreg & 0xF)    << R_SREG_SHIFT) |
           (uint64_t(flags & 1)     << R_FLAGS_SHIFT);
}
inline uint64_t MATMUL(int src1_buf, int src1_off, int src2_buf, int src2_off,
                       int dst_buf, int dst_off, int sreg, int flags = 0) {
    return R_TYPE(0x0A, src1_buf, src1_off, src2_buf, src2_off, dst_buf, dst_off, sreg, flags);
}
inline uint64_t REQUANT(int src1_buf, int src1_off, int dst_buf, int dst_off,
                        int sreg, int flags = 0) {
    return R_TYPE(0x0B, src1_buf, src1_off, 0, 0, dst_buf, dst_off, sreg, flags);
}
inline uint64_t REQUANT_PC(int src1_buf, int src1_off, int src2_buf, int src2_off,
                           int dst_buf, int dst_off, int sreg, int flags = 0) {
    return R_TYPE(0x11, src1_buf, src1_off, src2_buf, src2_off, dst_buf, dst_off, sreg, flags);
}
inline uint64_t SCALE_MUL(int src1_buf, int src1_off, int dst_buf, int dst_off,
                          int sreg, int flags = 0) {
    return R_TYPE(0x0C, src1_buf, src1_off, 0, 0, dst_buf, dst_off, sreg, flags);
}
inline uint64_t VADD(int src1_buf, int src1_off, int src2_buf, int src2_off,
                     int dst_buf, int dst_off, int sreg, int flags = 0) {
    return R_TYPE(0x0D, src1_buf, src1_off, src2_buf, src2_off, dst_buf, dst_off, sreg, flags);
}
inline uint64_t SOFTMAX(int src1_buf, int src1_off, int dst_buf, int dst_off,
                        int sreg, int flags = 0) {
    return R_TYPE(0x0E, src1_buf, src1_off, 0, 0, dst_buf, dst_off, sreg, flags);
}
inline uint64_t LAYERNORM(int src1_buf, int src1_off, int src2_buf, int src2_off,
                          int dst_buf, int dst_off, int sreg, int flags = 0) {
    return R_TYPE(0x0F, src1_buf, src1_off, src2_buf, src2_off, dst_buf, dst_off, sreg, flags);
}
inline uint64_t GELU(int src1_buf, int src1_off, int dst_buf, int dst_off,
                     int sreg, int flags = 0) {
    return R_TYPE(0x10, src1_buf, src1_off, 0, 0, dst_buf, dst_off, sreg, flags);
}
inline uint64_t SOFTMAX_ATTNV(int src1_buf, int src1_off, int src2_buf, int src2_off,
                              int dst_buf, int dst_off, int sreg, int flags = 0) {
    return R_TYPE(0x12, src1_buf, src1_off, src2_buf, src2_off, dst_buf, dst_off, sreg, flags);
}
inline uint64_t DEQUANT_ADD(int src1_buf, int src1_off, int src2_buf, int src2_off,
                            int dst_buf, int dst_off, int sreg, int flags = 0) {
    return R_TYPE(0x13, src1_buf, src1_off, src2_buf, src2_off, dst_buf, dst_off, sreg, flags);
}

// Illegal opcode for fault tests
constexpr uint64_t ILLEGAL_OP() { return uint64_t(0x14) << OPCODE_SHIFT; }

} // namespace insn

// ============================================================================
// AXI4 slave model: behavioral DRAM
// ============================================================================
class AXI4SlaveModel {
public:
    struct ReadInjection {
        uint64_t beat_idx;
        int resp;
        int force_last;
    };

    struct BrespInjection {
        uint64_t resp_idx;
        int resp;
    };

    struct ReadAddrInjection {
        uint64_t addr;
        int resp;
        int force_last;
    };

    explicit AXI4SlaveModel(size_t size_bytes)
        : mem_(size_bytes, 0), pending_valid_(false), pending_addr_(0),
          pending_timer_(0), r_valid_(false),
          aw_pending_(false), aw_addr_(0), aw_len_(0), w_beat_cnt_(0),
          b_pending_(false), next_r_resp_(0), next_r_last_override_(-1),
          next_b_resp_(0), active_b_resp_(0), read_beat_count_(0),
          read_emit_count_(0), bresp_emit_count_(0)
    {}

    // Write a big-endian 64-bit instruction word at instruction index pc_idx
    void write_insn(int pc_idx, uint64_t word) {
        size_t base = (size_t)pc_idx * 8;
        assert(base + 7 < mem_.size());
        mem_[base+0] = (word >> 56) & 0xFF;
        mem_[base+1] = (word >> 48) & 0xFF;
        mem_[base+2] = (word >> 40) & 0xFF;
        mem_[base+3] = (word >> 32) & 0xFF;
        mem_[base+4] = (word >> 24) & 0xFF;
        mem_[base+5] = (word >> 16) & 0xFF;
        mem_[base+6] = (word >>  8) & 0xFF;
        mem_[base+7] = (word >>  0) & 0xFF;
    }

    // Write a program sequence starting at instruction 0
    void write_program(const std::vector<uint64_t>& insns) {
        for (size_t i = 0; i < insns.size(); i++)
            write_insn((int)i, insns[i]);
    }

    void inject_next_read(int resp, int force_last = -1) {
        next_r_resp_ = resp & 0x3;
        next_r_last_override_ = force_last;
    }

    void inject_next_bresp(int resp) {
        next_b_resp_ = resp & 0x3;
    }

    void inject_read_at(uint64_t beat_idx, int resp, int force_last = -1) {
        scheduled_read_injections_.push_back(ReadInjection{beat_idx, resp & 0x3, force_last});
        std::sort(
            scheduled_read_injections_.begin(),
            scheduled_read_injections_.end(),
            [](const ReadInjection& lhs, const ReadInjection& rhs) {
                return lhs.beat_idx < rhs.beat_idx;
            }
        );
    }

    void inject_bresp_at(uint64_t resp_idx, int resp) {
        scheduled_bresp_injections_.push_back(BrespInjection{resp_idx, resp & 0x3});
        std::sort(
            scheduled_bresp_injections_.begin(),
            scheduled_bresp_injections_.end(),
            [](const BrespInjection& lhs, const BrespInjection& rhs) {
                return lhs.resp_idx < rhs.resp_idx;
            }
        );
    }

    void inject_read_addr(uint64_t addr, int resp, int force_last = -1) {
        scheduled_read_addr_injections_.push_back(ReadAddrInjection{addr, resp & 0x3, force_last});
    }

    // Write raw bytes at byte address
    void write_bytes(size_t addr, const uint8_t* data, size_t len) {
        assert(addr + len <= mem_.size());
        std::memcpy(mem_.data() + addr, data, len);
    }

    uint8_t read_byte(size_t addr) const {
        assert(addr < mem_.size());
        return mem_[addr];
    }

    const std::vector<uint64_t>& read_addr_log() const {
        return read_addr_log_;
    }

    const std::vector<int>& read_len_log() const {
        return read_len_log_;
    }

    uint64_t read_beat_count() const {
        return read_beat_count_;
    }

    // Drive AXI slave outputs given DUT's master outputs (call every cycle).
    // Handles AR/R (reads, single-beat model) and AW/W/B (writes, multi-beat).
    void tick(Vtaccel_top* dut, int latency = 2) {
        // ----------------------------------------------------------------
        // AR channel: accept one outstanding read request at a time
        // ----------------------------------------------------------------
        dut->m_axi_ar_ready = 1;
        if (dut->m_axi_ar_valid && !pending_valid_) {
            uint64_t aligned = (uint64_t)dut->m_axi_ar_addr & ~uint64_t(0xF);
            pending_addr_    = aligned;
            pending_ar_len_  = dut->m_axi_ar_len;   // burst beat count - 1
            pending_beat_    = 0;
            pending_timer_   = latency;
            pending_valid_   = true;
            read_addr_log_.push_back(aligned);
            read_len_log_.push_back(pending_ar_len_);
        }

        // ----------------------------------------------------------------
        // R channel: serve one beat per call when timer fires
        // ----------------------------------------------------------------
        if (r_valid_) {
            if (dut->m_axi_r_ready) {
                read_beat_count_++;
                r_valid_           = false;
                dut->m_axi_r_valid = 0;
                dut->m_axi_r_last  = 0;
                // If more beats remain in this burst, queue next beat
                if (pending_beat_ <= (int)pending_ar_len_) {
                    pending_timer_ = 1;  // next beat next cycle
                }
            }
        } else if (pending_valid_) {
            if (pending_timer_ > 0) {
                pending_timer_--;
            } else {
                // Build 16-byte response for current beat
                uint64_t beat_addr = pending_addr_ + (uint64_t)pending_beat_ * 16;
                uint64_t lo = 0, hi = 0;
                for (int b = 0; b < 8 && beat_addr+b < mem_.size(); b++)
                    lo |= (uint64_t)mem_[beat_addr+b] << (b*8);
                for (int b = 0; b < 8 && beat_addr+8+b < mem_.size(); b++)
                    hi |= (uint64_t)mem_[beat_addr+8+b] << (b*8);

                uint32_t* rdata = dut->m_axi_r_data.data();
                rdata[0] = lo & 0xFFFFFFFF;
                rdata[1] = (lo >> 32) & 0xFFFFFFFF;
                rdata[2] = hi & 0xFFFFFFFF;
                rdata[3] = (hi >> 32) & 0xFFFFFFFF;

                bool is_last = (pending_beat_ >= (int)pending_ar_len_);
                int resp = next_r_resp_;
                int force_last = next_r_last_override_;
                for (auto it = scheduled_read_addr_injections_.begin();
                     it != scheduled_read_addr_injections_.end(); ++it) {
                    if (it->addr == beat_addr) {
                        resp = it->resp;
                        force_last = it->force_last;
                        scheduled_read_addr_injections_.erase(it);
                        break;
                    }
                }
                if (!scheduled_read_injections_.empty() &&
                    scheduled_read_injections_.front().beat_idx == read_emit_count_) {
                    resp = scheduled_read_injections_.front().resp;
                    force_last = scheduled_read_injections_.front().force_last;
                    scheduled_read_injections_.erase(scheduled_read_injections_.begin());
                }
                if (force_last >= 0)
                    is_last = (force_last != 0);
                dut->m_axi_r_valid = 1;
                dut->m_axi_r_last  = is_last ? 1 : 0;
                dut->m_axi_r_resp  = resp;
                r_valid_           = true;
                pending_beat_++;
                read_emit_count_++;
                next_r_resp_ = 0;
                next_r_last_override_ = -1;
                if (is_last)
                    pending_valid_ = false;
            }
        }

        // ----------------------------------------------------------------
        // AW channel: accept write address
        // ----------------------------------------------------------------
        dut->m_axi_aw_ready = 1;
        if (dut->m_axi_aw_valid && !aw_pending_) {
            aw_addr_    = (uint64_t)dut->m_axi_aw_addr;
            aw_len_     = dut->m_axi_aw_len;
            w_beat_cnt_ = 0;
            aw_pending_ = true;
        }

        // ----------------------------------------------------------------
        // W channel: accept write data beats
        // ----------------------------------------------------------------
        if (aw_pending_) {
            dut->m_axi_w_ready = 1;
            if (dut->m_axi_w_valid) {
                // Write 16 bytes to memory
                uint64_t beat_addr = aw_addr_ + (uint64_t)w_beat_cnt_ * 16;
                uint32_t* wdata = dut->m_axi_w_data.data();
                for (int b = 0; b < 16 && beat_addr+b < mem_.size(); b++)
                    mem_[beat_addr+b] = (uint8_t)((wdata[b/4] >> ((b%4)*8)) & 0xFF);
                bool last = dut->m_axi_w_last || (w_beat_cnt_ >= (int)aw_len_);
                w_beat_cnt_++;
                if (last) {
                    aw_pending_ = false;
                    b_pending_  = true;
                    active_b_resp_ = next_b_resp_;
                    if (!scheduled_bresp_injections_.empty() &&
                        scheduled_bresp_injections_.front().resp_idx == bresp_emit_count_) {
                        active_b_resp_ = scheduled_bresp_injections_.front().resp;
                        scheduled_bresp_injections_.erase(scheduled_bresp_injections_.begin());
                    }
                    next_b_resp_ = 0;
                    bresp_emit_count_++;
                    // Keep w_ready=1 this cycle so DUT sees it on posedge;
                    // the else branch below will clear it next cycle.
                }
            }
        } else {
            dut->m_axi_w_ready = 0;
        }

        // ----------------------------------------------------------------
        // B channel: send write response
        // ----------------------------------------------------------------
        if (b_pending_) {
            dut->m_axi_b_valid = 1;
            dut->m_axi_b_resp  = active_b_resp_;
            if (dut->m_axi_b_ready) {
                b_pending_ = false;
                // Keep b_valid=1 this cycle so DUT sees it on posedge;
                // the else branch below will clear it next cycle.
            }
        } else {
            dut->m_axi_b_valid = 0;
            dut->m_axi_b_resp  = 0;
        }
    }

private:
    std::vector<uint8_t> mem_;
    // Read state
    bool     pending_valid_;
    uint64_t pending_addr_;
    int      pending_ar_len_;
    int      pending_beat_;
    int      pending_timer_;
    bool     r_valid_;
    // Write state
    bool     aw_pending_;
    uint64_t aw_addr_;
    int      aw_len_;
    int      w_beat_cnt_;
    bool     b_pending_;
    int      next_r_resp_;
    int      next_r_last_override_;
    int      next_b_resp_;
    int      active_b_resp_;
    uint64_t read_beat_count_;
    uint64_t read_emit_count_;
    uint64_t bresp_emit_count_;
    std::vector<uint64_t> read_addr_log_;
    std::vector<int> read_len_log_;
    std::vector<ReadInjection> scheduled_read_injections_;
    std::vector<ReadAddrInjection> scheduled_read_addr_injections_;
    std::vector<BrespInjection> scheduled_bresp_injections_;
};

// ============================================================================
// Simulation helpers
// ============================================================================

// Apply reset for n cycles
inline void do_reset(Vtaccel_top* dut, int cycles = 10) {
    dut->rst_n = 0;
    dut->start = 0;
    for (int i = 0; i < cycles; i++) {
        dut->clk = 0; dut->eval();
        dut->clk = 1; dut->eval();
    }
    dut->rst_n = 1;
    dut->clk = 0; dut->eval();
    dut->clk = 1; dut->eval();
}

// Tick one clock cycle, driving AXI slave
inline void tick(Vtaccel_top* dut, AXI4SlaveModel& dram, int latency = 2) {
    // Sample outputs on negedge (after posedge registered)
    dut->clk = 0;
    dut->eval();
    dram.tick(dut, latency);  // update slave combinational outputs
    dut->eval();

    dut->clk = 1;
    dut->eval();
}

template <typename NegedgeObserver>
inline void tick_with_negedge_observer(
    Vtaccel_top* dut,
    AXI4SlaveModel& dram,
    NegedgeObserver&& observe_negedge,
    int latency = 2) {
    dut->clk = 0;
    dut->eval();
    dram.tick(dut, latency);
    dut->eval();
    observe_negedge();

    dut->clk = 1;
    dut->eval();
}

// Run until done or fault; returns cycle count
inline int run_until_halt(Vtaccel_top* dut, AXI4SlaveModel& dram,
                           int timeout = 100000, int latency = 2) {
    for (int i = 0; i < timeout; i++) {
        tick(dut, dram, latency);
        if (dut->done || dut->fault)
            return i + 1;
    }
    throw std::runtime_error("Simulation timeout: DUT did not halt");
}

namespace tbutil {

constexpr int BUF_ABUF_ID  = 0;
constexpr int BUF_WBUF_ID  = 1;
constexpr int BUF_ACCUM_ID = 2;

struct SimHarness {
    std::unique_ptr<Vtaccel_top> dut;
    AXI4SlaveModel dram;

    explicit SimHarness(size_t dram_size = 16 * 1024 * 1024)
        : dut(std::make_unique<Vtaccel_top>()), dram(dram_size) {
        do_reset(dut.get());
    }

    void load(const std::vector<uint64_t>& prog) {
        dram.write_program(prog);
    }

    void start_once(int latency = 2) {
        dut->start = 1;
        tick(dut.get(), dram, latency);
        dut->start = 0;
    }

    void start(int latency = 2) {
        start_once(latency);
    }

    void step(int cycles = 1, int latency = 2) {
        for (int i = 0; i < cycles; ++i)
            tick(dut.get(), dram, latency);
    }

    int run(int timeout = 100000, int latency = 2) {
        start_once(latency);
        return run_until_halt(dut.get(), dram, timeout, latency);
    }
};

inline VlWide<4>* sram_row_ptr(Vtaccel_top* dut, int buf_id, int row) {
    auto* r = dut->rootp;
    switch (buf_id) {
        case BUF_ABUF_ID:  return &r->taccel_top__DOT__u_sram__DOT__u_abuf__DOT__mem[row];
        case BUF_WBUF_ID:  return &r->taccel_top__DOT__u_sram__DOT__u_wbuf__DOT__mem[row];
        case BUF_ACCUM_ID: return &r->taccel_top__DOT__u_sram__DOT__u_accum__DOT__mem[row];
        default: std::abort();
    }
}

inline const VlWide<4>* sram_row_ptr_const(Vtaccel_top* dut, int buf_id, int row) {
    return sram_row_ptr(dut, buf_id, row);
}

inline void sram_write_row(Vtaccel_top* dut, int buf_id, int row, const uint8_t data[16]) {
    VlWide<4>* mem = sram_row_ptr(dut, buf_id, row);
    for (int w = 0; w < 4; ++w) {
        (*mem)[w] = (uint32_t(data[w * 4 + 0])      ) |
                    (uint32_t(data[w * 4 + 1]) <<  8) |
                    (uint32_t(data[w * 4 + 2]) << 16) |
                    (uint32_t(data[w * 4 + 3]) << 24);
    }
}

inline void sram_read_row(Vtaccel_top* dut, int buf_id, int row, uint8_t out[16]) {
    const VlWide<4>* mem = sram_row_ptr_const(dut, buf_id, row);
    for (int w = 0; w < 4; ++w) {
        uint32_t word = (*mem)[w];
        out[w * 4 + 0] = uint8_t((word >> 0) & 0xFF);
        out[w * 4 + 1] = uint8_t((word >> 8) & 0xFF);
        out[w * 4 + 2] = uint8_t((word >> 16) & 0xFF);
        out[w * 4 + 3] = uint8_t((word >> 24) & 0xFF);
    }
}

inline void sram_write_bytes(Vtaccel_top* dut, int buf_id, size_t byte_off,
                             const std::vector<uint8_t>& data) {
    size_t pos = 0;
    while (pos < data.size()) {
        int row = int((byte_off + pos) / 16);
        int lane = int((byte_off + pos) % 16);
        int take = int(std::min<size_t>(16 - lane, data.size() - pos));
        uint8_t tmp[16];
        sram_read_row(dut, buf_id, row, tmp);
        for (int i = 0; i < take; ++i)
            tmp[lane + i] = data[pos + size_t(i)];
        sram_write_row(dut, buf_id, row, tmp);
        pos += size_t(take);
    }
}

inline std::vector<uint8_t> sram_read_bytes(Vtaccel_top* dut, int buf_id,
                                            size_t byte_off, size_t len) {
    std::vector<uint8_t> out(len);
    size_t pos = 0;
    while (pos < len) {
        int row = int((byte_off + pos) / 16);
        int lane = int((byte_off + pos) % 16);
        int take = int(std::min<size_t>(16 - lane, len - pos));
        uint8_t tmp[16];
        sram_read_row(dut, buf_id, row, tmp);
        for (int i = 0; i < take; ++i)
            out[pos + size_t(i)] = tmp[lane + i];
        pos += size_t(take);
    }
    return out;
}

// Read ACCUM buffer in logical row-major order.
//
// The RTL systolic drain stores results in a tile-major physical layout:
//   physical_row = m * M_stride + n_tile * 4 + col_group
// where M_stride = mem_cols / 4 = N_tiles * 4.
//
// This function reads all needed physical rows and rearranges them into the
// logical (rows, cols) row-major layout matching the golden model's flat int32
// array. Returns logical_rows * logical_cols * 4 bytes, little-endian int32.
inline std::vector<uint8_t> accum_read_logical_i32(
    Vtaccel_top* dut,
    int offset_units, int mem_cols,
    int logical_rows, int logical_cols
) {
    if (mem_cols < 16 || (mem_cols % 16) != 0) {
        throw std::runtime_error(
            "accum_read_logical_i32: mem_cols=" + std::to_string(mem_cols) +
            " must be a positive multiple of 16");
    }
    const int m_stride = mem_cols / 4;  // N_tiles * 4 physical rows per logical row
    const size_t total_phys_rows = size_t(logical_rows) * size_t(m_stride);
    const size_t phys_start = size_t(offset_units) * 16u;

    // One contiguous read covering all needed physical rows
    const auto phys = sram_read_bytes(dut, BUF_ACCUM_ID,
                                      phys_start,
                                      total_phys_rows * 16u);

    // Rearrange from tile-major to logical row-major
    const size_t out_bytes = size_t(logical_rows) * size_t(logical_cols) * 4u;
    std::vector<uint8_t> out(out_bytes, 0);

    for (int m = 0; m < logical_rows; m++) {
        for (int n = 0; n < logical_cols; n++) {
            const int n_tile      = n / 16;
            const int col_in_tile = n % 16;
            const int col_group   = col_in_tile / 4;
            const int col_in_grp  = col_in_tile % 4;
            const int phys_row    = m * m_stride + n_tile * 4 + col_group;
            const size_t phys_off = size_t(phys_row) * 16u + size_t(col_in_grp) * 4u;
            const size_t log_off  = (size_t(m) * size_t(logical_cols) + size_t(n)) * 4u;
            out[log_off + 0] = phys[phys_off + 0];
            out[log_off + 1] = phys[phys_off + 1];
            out[log_off + 2] = phys[phys_off + 2];
            out[log_off + 3] = phys[phys_off + 3];
        }
    }
    return out;
}

// Read a tile-padded int8 buffer (ABUF, WBUF) in logical row-major order.
//
// SFU operations (SOFTMAX, LAYERNORM, GELU) store results using a row-tile
// layout:  physical_row = m * n_tiles + n / 16,  byte_offset = n % 16
// where n_tiles = mem_cols / 16.  When logical_cols is not a multiple of 16
// the last tile of each row contains padding that must be skipped.
//
// This function reads all needed physical rows and extracts only the valid
// logical elements.  Returns logical_rows * logical_cols bytes.
// When mem_cols == logical_cols the result is identical to a plain flat read.
inline std::vector<uint8_t> sfu_read_logical_i8(
    Vtaccel_top* dut,
    int buf_id,
    int offset_units, int mem_cols,
    int logical_rows, int logical_cols
) {
    if (mem_cols < 16 || (mem_cols % 16) != 0) {
        throw std::runtime_error(
            "sfu_read_logical_i8: mem_cols=" + std::to_string(mem_cols) +
            " must be a positive multiple of 16");
    }
    const int n_tiles = mem_cols / 16;
    const size_t total_phys_rows = size_t(logical_rows) * size_t(n_tiles);
    const size_t phys_start = size_t(offset_units) * 16u;

    const auto phys = sram_read_bytes(dut, buf_id,
                                      phys_start,
                                      total_phys_rows * 16u);

    const size_t out_bytes = size_t(logical_rows) * size_t(logical_cols);
    std::vector<uint8_t> out(out_bytes, 0);

    for (int m = 0; m < logical_rows; m++) {
        for (int n = 0; n < logical_cols; n++) {
            const int phys_row = m * n_tiles + n / 16;
            const int byte_off = n % 16;
            const size_t phys_pos = size_t(phys_row) * 16u + size_t(byte_off);
            const size_t log_pos  = size_t(m) * size_t(logical_cols) + size_t(n);
            out[log_pos] = phys[phys_pos];
        }
    }
    return out;
}

inline std::vector<uint8_t> make_pattern(size_t size, uint8_t seed) {
    std::vector<uint8_t> data(size);
    for (size_t i = 0; i < size; ++i)
        data[i] = uint8_t((seed + (37 * i)) & 0xFF);
    return data;
}

inline std::vector<uint8_t> pack_i32_le(const std::vector<int32_t>& vals) {
    std::vector<uint8_t> out(vals.size() * 4);
    for (size_t i = 0; i < vals.size(); ++i) {
        uint32_t word = uint32_t(vals[i]);
        out[i * 4 + 0] = uint8_t((word >> 0) & 0xFF);
        out[i * 4 + 1] = uint8_t((word >> 8) & 0xFF);
        out[i * 4 + 2] = uint8_t((word >> 16) & 0xFF);
        out[i * 4 + 3] = uint8_t((word >> 24) & 0xFF);
    }
    return out;
}

inline std::vector<uint8_t> pack_u16_le(const std::vector<uint16_t>& vals) {
    std::vector<uint8_t> out(vals.size() * 2);
    for (size_t i = 0; i < vals.size(); ++i) {
        out[i * 2 + 0] = uint8_t(vals[i] & 0xFF);
        out[i * 2 + 1] = uint8_t((vals[i] >> 8) & 0xFF);
    }
    return out;
}

inline std::vector<int32_t> unpack_i32_le(const std::vector<uint8_t>& bytes) {
    std::vector<int32_t> out(bytes.size() / 4);
    for (size_t i = 0; i < out.size(); ++i) {
        uint32_t word = uint32_t(bytes[i * 4 + 0]) |
                        (uint32_t(bytes[i * 4 + 1]) << 8) |
                        (uint32_t(bytes[i * 4 + 2]) << 16) |
                        (uint32_t(bytes[i * 4 + 3]) << 24);
        out[i] = int32_t(word);
    }
    return out;
}

} // namespace tbutil

// ============================================================================
// Test assertion macro
// ============================================================================
#define EXPECT(cond, msg) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL: %s  (at %s:%d)\n", msg, __FILE__, __LINE__); \
            std::exit(1); \
        } \
    } while(0)

#define PASS(name) \
    fprintf(stdout, "PASS: %s\n", name)
