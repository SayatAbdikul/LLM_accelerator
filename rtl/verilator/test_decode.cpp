// Verilator unit tests for decode_unit via tb_decode_unit wrapper.
//
// Tests that every instruction format is decoded correctly — all field
// positions must match software/taccel/isa/encoding.py exactly.
//
// Purely combinational: no clock required.

#include "Vtb_decode_unit.h"
#include "verilated.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <memory>

// ============================================================================
// Bit-position constants (mirror software/taccel/isa/opcodes.py)
// ============================================================================
static const int OPCODE_SHIFT      = 59;
static const int R_SRC1_BUF_SHIFT  = 57;
static const int R_SRC1_OFF_SHIFT  = 41;
static const int R_SRC2_BUF_SHIFT  = 39;
static const int R_SRC2_OFF_SHIFT  = 23;
static const int R_DST_BUF_SHIFT   = 21;
static const int R_DST_OFF_SHIFT   = 5;
static const int R_SREG_SHIFT      = 1;
static const int R_FLAGS_SHIFT     = 0;
static const int M_BUF_ID_SHIFT    = 57;
static const int M_SRAM_OFF_SHIFT  = 41;
static const int M_XFER_LEN_SHIFT  = 25;
static const int M_ADDR_REG_SHIFT  = 23;
static const int M_DRAM_OFF_SHIFT  = 7;
static const int B_SRC_BUF_SHIFT   = 57;
static const int B_SRC_OFF_SHIFT   = 41;
static const int B_DST_BUF_SHIFT   = 39;
static const int B_DST_OFF_SHIFT   = 23;
static const int B_LENGTH_SHIFT    = 7;
static const int B_SRC_ROWS_SHIFT  = 1;
static const int B_TRANSPOSE_SHIFT = 0;
static const int A_ADDR_REG_SHIFT  = 57;
static const int A_IMM28_SHIFT     = 29;
static const int C_M_SHIFT         = 49;
static const int C_N_SHIFT         = 39;
static const int C_K_SHIFT         = 29;
static const int SS_SREG_SHIFT     = 55;
static const int SS_SRC_MODE_SHIFT = 53;
static const int SS_IMM16_SHIFT    = 37;
static const int SYNC_MASK_SHIFT   = 56;

// Apply instruction word and re-evaluate combinational logic
static Vtb_decode_unit* g_dut = nullptr;
static std::string g_test;
static int tests_run = 0, tests_pass = 0;

static void apply(uint64_t word) {
    g_dut->insn_data = word;
    g_dut->eval();
}

static void check_impl(const char* field, uint64_t got, uint64_t exp,
                        const char* file, int line) {
    if (got != exp) {
        fprintf(stderr, "FAIL [%s] field '%s': got 0x%llx, expected 0x%llx  (%s:%d)\n",
                g_test.c_str(), field,
                (unsigned long long)got, (unsigned long long)exp, file, line);
        std::exit(1);
    }
}
#define CHK(field, expected) check_impl(#field, (uint64_t)(g_dut->field), \
                                         (uint64_t)(expected), __FILE__, __LINE__)

static void begin_test(const char* name) {
    g_test = name;
    tests_run++;
}
static void end_test() {
    printf("PASS: %s\n", g_test.c_str());
    tests_pass++;
}

// ============================================================================
int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    g_dut = new Vtb_decode_unit;

    // -----------------------------------------------------------------------
    // NOP
    // -----------------------------------------------------------------------
    begin_test("decode_nop");
    apply(uint64_t(0x00) << OPCODE_SHIFT);
    CHK(opcode,  0x00);
    CHK(illegal, 0);
    end_test();

    // -----------------------------------------------------------------------
    // HALT
    // -----------------------------------------------------------------------
    begin_test("decode_halt");
    apply(uint64_t(0x01) << OPCODE_SHIFT);
    CHK(opcode,  0x01);
    CHK(illegal, 0);
    end_test();

    // -----------------------------------------------------------------------
    // SYNC resource_mask = 0b101
    // -----------------------------------------------------------------------
    begin_test("decode_sync_mask_101");
    apply((uint64_t(0x02) << OPCODE_SHIFT) | (uint64_t(0b101) << SYNC_MASK_SHIFT));
    CHK(opcode,    0x02);
    CHK(illegal,   0);
    CHK(sync_mask, 0b101);
    end_test();

    // -----------------------------------------------------------------------
    // CONFIG_TILE M=3, N=5, K=12  (stored 0-based)
    // -----------------------------------------------------------------------
    begin_test("decode_config_tile_3_5_12");
    apply((uint64_t(0x03) << OPCODE_SHIFT) |
          (uint64_t(2)    << C_M_SHIFT)    |   // M=3 encoded as 2
          (uint64_t(4)    << C_N_SHIFT)    |   // N=5 encoded as 4
          (uint64_t(11)   << C_K_SHIFT));       // K=12 encoded as 11
    CHK(opcode,   0x03);
    CHK(illegal,  0);
    CHK(c_tile_m, 2);
    CHK(c_tile_n, 4);
    CHK(c_tile_k, 11);
    end_test();

    // -----------------------------------------------------------------------
    // SET_SCALE S5, imm=0x3C00 (1.0 FP16)
    // -----------------------------------------------------------------------
    begin_test("decode_set_scale_s5_1_0");
    apply((uint64_t(0x04)   << OPCODE_SHIFT)      |
          (uint64_t(5)       << SS_SREG_SHIFT)     |
          (uint64_t(0)       << SS_SRC_MODE_SHIFT) |
          (uint64_t(0x3C00)  << SS_IMM16_SHIFT));
    CHK(opcode,    0x04);
    CHK(illegal,   0);
    CHK(s_sreg,    5);
    CHK(s_src_mode,0);
    CHK(s_imm16,   0x3C00);
    end_test();

    // -----------------------------------------------------------------------
    // SET_SCALE S0, src_mode=2 (from WBUF), offset=7
    // -----------------------------------------------------------------------
    begin_test("decode_set_scale_from_wbuf");
    apply((uint64_t(0x04) << OPCODE_SHIFT)      |
          (uint64_t(0)    << SS_SREG_SHIFT)     |
          (uint64_t(2)    << SS_SRC_MODE_SHIFT) |
          (uint64_t(7)    << SS_IMM16_SHIFT));
    CHK(s_src_mode, 2);
    CHK(s_imm16,    7);
    end_test();

    // -----------------------------------------------------------------------
    // SET_ADDR_LO R2, imm28=0x0ABCDEF0
    // -----------------------------------------------------------------------
    begin_test("decode_set_addr_lo_r2");
    apply((uint64_t(0x05)      << OPCODE_SHIFT)     |
          (uint64_t(2)          << A_ADDR_REG_SHIFT) |
          (uint64_t(0x0ABCDEF0) << A_IMM28_SHIFT));
    CHK(opcode,    0x05);
    CHK(illegal,   0);
    CHK(a_addr_reg, 2);
    CHK(a_imm28,   0x0ABCDEF0);
    end_test();

    // -----------------------------------------------------------------------
    // SET_ADDR_HI R3, imm28=0x0000123
    // -----------------------------------------------------------------------
    begin_test("decode_set_addr_hi_r3");
    apply((uint64_t(0x06)    << OPCODE_SHIFT)     |
          (uint64_t(3)        << A_ADDR_REG_SHIFT) |
          (uint64_t(0x0000123)<< A_IMM28_SHIFT));
    CHK(opcode,    0x06);
    CHK(a_addr_reg, 3);
    CHK(a_imm28,   0x0000123);
    end_test();

    // -----------------------------------------------------------------------
    // LOAD ABUF, sram_off=10, xfer_len=16, addr_reg=1, dram_off=4
    // -----------------------------------------------------------------------
    begin_test("decode_load_abuf");
    apply((uint64_t(0x07) << OPCODE_SHIFT)   |
          (uint64_t(0)    << M_BUF_ID_SHIFT)  |
          (uint64_t(10)   << M_SRAM_OFF_SHIFT)|
          (uint64_t(16)   << M_XFER_LEN_SHIFT)|
          (uint64_t(1)    << M_ADDR_REG_SHIFT)|
          (uint64_t(4)    << M_DRAM_OFF_SHIFT));
    CHK(opcode,    0x07);
    CHK(illegal,   0);
    CHK(m_buf_id,  0);    // BUF_ABUF
    CHK(m_sram_off,10);
    CHK(m_xfer_len,16);
    CHK(m_addr_reg,1);
    CHK(m_dram_off,4);
    end_test();

    // -----------------------------------------------------------------------
    // STORE WBUF, sram_off=0, xfer_len=8, addr_reg=0, dram_off=0
    // -----------------------------------------------------------------------
    begin_test("decode_store_wbuf");
    apply((uint64_t(0x08) << OPCODE_SHIFT)   |
          (uint64_t(1)    << M_BUF_ID_SHIFT)  |
          (uint64_t(0)    << M_SRAM_OFF_SHIFT)|
          (uint64_t(8)    << M_XFER_LEN_SHIFT)|
          (uint64_t(0)    << M_ADDR_REG_SHIFT)|
          (uint64_t(0)    << M_DRAM_OFF_SHIFT));
    CHK(opcode,    0x08);
    CHK(illegal,   0);
    CHK(m_buf_id,  1);    // BUF_WBUF
    CHK(m_xfer_len,8);
    end_test();

    // -----------------------------------------------------------------------
    // BUF_COPY ABUF→WBUF, src_off=0, dst_off=0, length=32, src_rows=2, transpose=1
    // -----------------------------------------------------------------------
    begin_test("decode_buf_copy_transpose");
    apply((uint64_t(0x09) << OPCODE_SHIFT)    |
          (uint64_t(0)    << B_SRC_BUF_SHIFT)  |
          (uint64_t(0)    << B_SRC_OFF_SHIFT)  |
          (uint64_t(1)    << B_DST_BUF_SHIFT)  |
          (uint64_t(0)    << B_DST_OFF_SHIFT)  |
          (uint64_t(32)   << B_LENGTH_SHIFT)   |
          (uint64_t(2)    << B_SRC_ROWS_SHIFT) |
          (uint64_t(1)    << B_TRANSPOSE_SHIFT));
    CHK(opcode,     0x09);
    CHK(illegal,    0);
    CHK(b_src_buf,  0);   // ABUF
    CHK(b_dst_buf,  1);   // WBUF
    CHK(b_length,   32);
    CHK(b_src_rows, 2);
    CHK(b_transpose,1);
    end_test();

    // -----------------------------------------------------------------------
    // MATMUL ABUF[0], WBUF[0] → ACCUM[0], sreg=3, flags=1 (accumulate)
    // -----------------------------------------------------------------------
    begin_test("decode_matmul_accumulate");
    apply((uint64_t(0x0A) << OPCODE_SHIFT)      |
          (uint64_t(0)    << R_SRC1_BUF_SHIFT)  |  // ABUF
          (uint64_t(0)    << R_SRC1_OFF_SHIFT)  |
          (uint64_t(1)    << R_SRC2_BUF_SHIFT)  |  // WBUF
          (uint64_t(0)    << R_SRC2_OFF_SHIFT)  |
          (uint64_t(2)    << R_DST_BUF_SHIFT)   |  // ACCUM
          (uint64_t(0)    << R_DST_OFF_SHIFT)   |
          (uint64_t(3)    << R_SREG_SHIFT)       |
          (uint64_t(1)    << R_FLAGS_SHIFT));
    CHK(opcode,  0x0A);
    CHK(illegal, 0);
    CHK(src1_buf,0);
    CHK(src2_buf,1);
    CHK(dst_buf, 2);
    CHK(sreg,    3);
    CHK(flags,   1);
    end_test();

    // -----------------------------------------------------------------------
    // REQUANT ACCUM[5] → ABUF[10], sreg=0
    // -----------------------------------------------------------------------
    begin_test("decode_requant");
    apply((uint64_t(0x0B) << OPCODE_SHIFT)     |
          (uint64_t(2)    << R_SRC1_BUF_SHIFT) |  // ACCUM
          (uint64_t(5)    << R_SRC1_OFF_SHIFT) |
          (uint64_t(0)    << R_DST_BUF_SHIFT)  |  // ABUF
          (uint64_t(10)   << R_DST_OFF_SHIFT)  |
          (uint64_t(0)    << R_SREG_SHIFT));
    CHK(opcode,  0x0B);
    CHK(illegal, 0);
    CHK(src1_buf,2);
    CHK(src1_off,5);
    CHK(dst_buf, 0);
    CHK(dst_off, 10);
    end_test();

    // -----------------------------------------------------------------------
    // SCALE_MUL (0x0C)
    // -----------------------------------------------------------------------
    begin_test("decode_scale_mul");
    apply((uint64_t(0x0C) << OPCODE_SHIFT)     |
          (uint64_t(2)    << R_SRC1_BUF_SHIFT) |  // ACCUM
          (uint64_t(0)    << R_SRC1_OFF_SHIFT) |
          (uint64_t(2)    << R_DST_BUF_SHIFT)  |  // ACCUM
          (uint64_t(0)    << R_DST_OFF_SHIFT)  |
          (uint64_t(1)    << R_SREG_SHIFT));
    CHK(opcode, 0x0C);
    CHK(illegal,0);
    CHK(sreg,   1);
    end_test();

    // -----------------------------------------------------------------------
    // VADD ABUF[0] + ABUF[16] → ABUF[32], sreg=0
    // -----------------------------------------------------------------------
    begin_test("decode_vadd");
    apply((uint64_t(0x0D) << OPCODE_SHIFT)     |
          (uint64_t(0)    << R_SRC1_BUF_SHIFT) |  // ABUF
          (uint64_t(0)    << R_SRC1_OFF_SHIFT) |
          (uint64_t(0)    << R_SRC2_BUF_SHIFT) |  // ABUF
          (uint64_t(16)   << R_SRC2_OFF_SHIFT) |
          (uint64_t(0)    << R_DST_BUF_SHIFT)  |
          (uint64_t(32)   << R_DST_OFF_SHIFT));
    CHK(opcode,  0x0D);
    CHK(illegal, 0);
    CHK(src2_off,16);
    CHK(dst_off, 32);
    end_test();

    // -----------------------------------------------------------------------
    // SOFTMAX, LAYERNORM, GELU — check opcodes only
    // -----------------------------------------------------------------------
    begin_test("decode_softmax_opcode");
    apply(uint64_t(0x0E) << OPCODE_SHIFT);
    CHK(opcode, 0x0E); CHK(illegal, 0);
    end_test();

    begin_test("decode_layernorm_opcode");
    apply(uint64_t(0x0F) << OPCODE_SHIFT);
    CHK(opcode, 0x0F); CHK(illegal, 0);
    end_test();

    begin_test("decode_gelu_opcode");
    apply((uint64_t(0x10) << OPCODE_SHIFT) |
          (uint64_t(7)    << R_SRC1_OFF_SHIFT) |
          (uint64_t(7)    << R_DST_OFF_SHIFT)  |
          (uint64_t(4)    << R_SREG_SHIFT));
    CHK(opcode,  0x10);
    CHK(illegal, 0);
    CHK(src1_off,7);
    CHK(dst_off, 7);
    CHK(sreg,    4);
    end_test();

    // -----------------------------------------------------------------------
    // REQUANT_PC / SOFTMAX_ATTNV / DEQUANT_ADD are legal in the Phase A ISA
    // -----------------------------------------------------------------------
    begin_test("decode_requant_pc");
    apply(uint64_t(0x11) << OPCODE_SHIFT);
    CHK(opcode,  0x11);
    CHK(illegal, 0);
    end_test();

    begin_test("decode_softmax_attnv");
    apply((uint64_t(0x12) << OPCODE_SHIFT)      |
          (uint64_t(2)    << R_SRC1_BUF_SHIFT)  |
          (uint64_t(3)    << R_SRC1_OFF_SHIFT)  |
          (uint64_t(0)    << R_SRC2_BUF_SHIFT)  |
          (uint64_t(4)    << R_SRC2_OFF_SHIFT)  |
          (uint64_t(1)    << R_DST_BUF_SHIFT)   |
          (uint64_t(5)    << R_DST_OFF_SHIFT)   |
          (uint64_t(6)    << R_SREG_SHIFT));
    CHK(opcode,  0x12);
    CHK(illegal, 0);
    CHK(src1_buf, 2);
    CHK(src2_buf, 0);
    CHK(dst_buf,  1);
    CHK(sreg,     6);
    end_test();

    begin_test("decode_dequant_add");
    apply((uint64_t(0x13) << OPCODE_SHIFT)      |
          (uint64_t(2)    << R_SRC1_BUF_SHIFT)  |
          (uint64_t(9)    << R_SRC1_OFF_SHIFT)  |
          (uint64_t(0)    << R_SRC2_BUF_SHIFT)  |
          (uint64_t(11)   << R_SRC2_OFF_SHIFT)  |
          (uint64_t(0)    << R_DST_BUF_SHIFT)   |
          (uint64_t(13)   << R_DST_OFF_SHIFT)   |
          (uint64_t(2)    << R_SREG_SHIFT));
    CHK(opcode,   0x13);
    CHK(illegal,  0);
    CHK(src1_off, 9);
    CHK(src2_off, 11);
    CHK(dst_off,  13);
    end_test();

    // -----------------------------------------------------------------------
    // Reserved opcode 0x14 → illegal=1
    // -----------------------------------------------------------------------
    begin_test("decode_illegal_opcode_0x14");
    apply(uint64_t(0x14) << OPCODE_SHIFT);
    CHK(opcode,  0x14);
    CHK(illegal, 1);
    end_test();

    begin_test("decode_illegal_opcode_0x1F");
    apply(uint64_t(0x1F) << OPCODE_SHIFT);
    CHK(illegal, 1);
    end_test();

    // -----------------------------------------------------------------------
    // R-type with dst_buf=0b11 → illegal=1
    // -----------------------------------------------------------------------
    begin_test("decode_illegal_buf_dst_0b11");
    apply((uint64_t(0x0A) << OPCODE_SHIFT)    |  // MATMUL
          (uint64_t(0)    << R_SRC1_BUF_SHIFT)|
          (uint64_t(1)    << R_SRC2_BUF_SHIFT)|
          (uint64_t(3)    << R_DST_BUF_SHIFT));   // 0b11 = reserved
    CHK(illegal, 1);
    end_test();

    // -----------------------------------------------------------------------
    // New legal R-type opcodes must still reject reserved buffer IDs
    // -----------------------------------------------------------------------
    begin_test("decode_illegal_buf_dequant_add_dst_0b11");
    apply((uint64_t(0x13) << OPCODE_SHIFT)    |
          (uint64_t(2)    << R_SRC1_BUF_SHIFT)|
          (uint64_t(0)    << R_SRC2_BUF_SHIFT)|
          (uint64_t(3)    << R_DST_BUF_SHIFT));
    CHK(illegal, 1);
    end_test();

    // -----------------------------------------------------------------------
    // M-type (LOAD) with buf_id=0b11 → illegal=1
    // -----------------------------------------------------------------------
    begin_test("decode_illegal_buf_m_type_0b11");
    apply((uint64_t(0x07) << OPCODE_SHIFT) |  // LOAD
          (uint64_t(3)    << M_BUF_ID_SHIFT)); // 0b11 = reserved
    CHK(illegal, 1);
    end_test();

    // -----------------------------------------------------------------------
    // B-type with src_buf=0b11 → illegal=1
    // -----------------------------------------------------------------------
    begin_test("decode_illegal_buf_b_src_0b11");
    apply((uint64_t(0x09) << OPCODE_SHIFT)   |  // BUF_COPY
          (uint64_t(3)    << B_SRC_BUF_SHIFT)); // 0b11 = reserved
    CHK(illegal, 1);
    end_test();

    // -----------------------------------------------------------------------
    // NOP has no payload — should not set illegal
    // -----------------------------------------------------------------------
    begin_test("decode_nop_all_zeros_body");
    apply(uint64_t(0x00) << OPCODE_SHIFT);
    CHK(illegal, 0);
    CHK(opcode,  0x00);
    end_test();

    // -----------------------------------------------------------------------
    printf("\n%d / %d tests passed\n", tests_pass, tests_run);
    delete g_dut;
    if (tests_pass != tests_run) std::exit(1);
    return 0;
}
