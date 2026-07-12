// Instruction decode unit -- purely combinational.
//
// Takes the raw 64-bit big-endian instruction word and extracts all
// format-specific fields in parallel.  The control_unit selects which
// fields to use based on insn.opcode.
//
// Bit positions match software/taccel/isa/encoding.py exactly.
// The single `insn.illegal` bit intentionally collapses two cases:
//   - reserved opcode
//   - reserved buffer ID inside a legal format
// The control unit maps those to the architectural fault code later.

`ifndef DECODE_UNIT_SV
`define DECODE_UNIT_SV

`include "taccel_pkg.sv"

module decode_unit
  import taccel_pkg::*;
(
  input  logic [63:0]    insn_data,
  output decoded_insn_t  insn
);

  // -------------------------------------------------------------------------
  // Opcode extraction
  // -------------------------------------------------------------------------
  logic [4:0] opcode_raw;
  assign opcode_raw = insn_data[63:59];

  // -------------------------------------------------------------------------
  // Illegal opcodes — 2026-05-23 ISA-reduction Phase B:
  //   Gen-1 SFU ops (0x0E SOFTMAX, 0x0F LAYERNORM, 0x10 GELU,
  //   0x12 SOFTMAX_ATTNV, 0x15 MASKED_SOFTMAX, 0x16 MASKED_SOFTMAX_ATTNV)
  //   stripped from RTL silicon. Plus pre-existing reserved 0x1C
  //   (non-causal SOFTMAX_FP32, no frontend consumer).
  //   The 5 gen-1 helper-engine ops (0x0B REQUANT, 0x0C SCALE_MUL,
  //   0x0D VADD, 0x11 REQUANT_PC, 0x13 DEQUANT_ADD) STAY LEGAL — the
  //   blocking_helper_engine still implements them; non-normative per
  //   freeze §3 but kept silicon-side for the W8A8 / debug paths.
  //   Legal: 0x00-0x0D infra+helper, 0x11/0x13 helper, 0x14 CONFIG_ATTN,
  //          gen-2 FP32 ops 0x17-0x1B / 0x1D-0x1F.
  // -------------------------------------------------------------------------
  logic illegal_opcode;
  assign illegal_opcode = (opcode_raw == 5'h0E) || (opcode_raw == 5'h0F) ||
                          (opcode_raw == 5'h10) || (opcode_raw == 5'h12) ||
                          (opcode_raw == 5'h15) || (opcode_raw == 5'h16) ||
                          (opcode_raw == 5'h1C);

  // CONFIG_ATTN has reserved bits [32:0], matching the software decoder.
  logic illegal_attn_reserved;
  assign illegal_attn_reserved = (opcode_raw == 5'h14) && (|insn_data[32:0]);

  // -------------------------------------------------------------------------
  // Illegal buffer ID check (R-type, M-type, B-type only).
  // Overlapping field positions are read directly from insn_data so the check
  // remains format-local and does not depend on downstream aliases.
  // -------------------------------------------------------------------------
  logic illegal_buf;
  always_comb begin
    illegal_buf = 1'b0;
    if (!illegal_opcode) begin
      case (opcode_raw)
        // R-type: check src1, src2, dst. 2026-05-23 Phase B: stripped
        // gen-1 SFU R-type ops (0x0E/0x0F/0x10/0x12/0x15/0x16) are trapped
        // by illegal_opcode above and never reach this case. The 5 gen-1
        // helper R-type ops (0x0B/0x0C/0x0D/0x11/0x13) STAY here — the
        // helper engine still handles them.
        5'h0A, 5'h0B, 5'h0C, 5'h0D, 5'h11, 5'h13,
        // gen-2 FP32 R-type ops (0x1C excluded — illegal_opcode handles it)
        5'h17, 5'h18, 5'h19, 5'h1A, 5'h1B,
        5'h1D, 5'h1E, 5'h1F: begin
          if (insn_data[58:57] == 2'b11 ||
              insn_data[40:39] == 2'b11 ||
              insn_data[22:21] == 2'b11)
            illegal_buf = 1'b1;
        end
        // M-type: check buf_id [58:57]
        5'h07, 5'h08: begin
          if (insn_data[58:57] == 2'b11)
            illegal_buf = 1'b1;
        end
        // B-type: check src_buf [58:57], dst_buf [40:39]
        5'h09: begin
          if (insn_data[58:57] == 2'b11 ||
              insn_data[40:39] == 2'b11)
            illegal_buf = 1'b1;
        end
        default: ;
      endcase
    end
  end

  // -------------------------------------------------------------------------
  // Parallel field extraction -- all formats decoded simultaneously.
  // Fields are only architecturally valid for their matching format.
  //
  // Bit positions:
  //  [63:59] opcode
  //
  //  R-TYPE:  [58:57] src1_buf  [56:41] src1_off  [40:39] src2_buf
  //           [38:23] src2_off  [22:21] dst_buf   [20:5]  dst_off
  //           [4:1]   sreg      [0]     flags
  //
  //  M-TYPE:  [58:57] buf_id    [56:41] sram_off  [40:25] xfer_len
  //           [24:23] addr_reg  [22:7]  dram_off  [6:3] cols_log2 [0] transpose
  //
  //  B-TYPE:  [58:57] src_buf   [56:41] src_off   [40:39] dst_buf
  //           [38:23] dst_off   [22:7]  length    [6:1]   src_rows
  //           [0]     transpose
  //
  //  A-TYPE:  [58:57] addr_reg  [56:29] imm28
  //
  //  C-TYPE:  [58:49] M         [48:39] N         [38:29] K
  //
  //  ATTN-TYPE: [58:47] query_row_base [46:35] valid_kv_len
  //             [34:33] mode           [32:0]  reserved zero
  //
  //  S-TYPE SET_SCALE: [58:55] sreg  [54:53] src_mode  [52:37] imm16
  //  S-TYPE SYNC:      [58:56] resource_mask
  // -------------------------------------------------------------------------
  always_comb begin
    // Common
    insn.opcode  = opcode_raw;
    insn.illegal = illegal_opcode || illegal_buf || illegal_attn_reserved;

    // R-type fields
    insn.src1_buf = insn_data[58:57];
    insn.src1_off = insn_data[56:41];
    insn.src2_buf = insn_data[40:39];
    insn.src2_off = insn_data[38:23];
    insn.dst_buf  = insn_data[22:21];
    insn.dst_off  = insn_data[20:5];
    insn.sreg     = insn_data[4:1];
    insn.flags    = insn_data[0];

    // M-type fields (buf_id/sram_off overlap with src1_buf/src1_off)
    insn.m_buf_id    = insn_data[58:57];
    insn.m_sram_off  = insn_data[56:41];
    insn.m_xfer_len  = insn_data[40:25];
    insn.m_addr_reg  = insn_data[24:23];
    insn.m_dram_off  = insn_data[22:7];
    // Lever D transpose-load geometry (reserved bits, 0 on plain loads/stores).
    insn.m_cols_log2 = insn_data[6:3];
    insn.m_transpose = insn_data[0];

    // B-type fields
    insn.b_src_buf   = insn_data[58:57];
    insn.b_src_off   = insn_data[56:41];
    insn.b_dst_buf   = insn_data[40:39];
    insn.b_dst_off   = insn_data[38:23];
    insn.b_length    = insn_data[22:7];
    insn.b_src_rows  = insn_data[6:1];
    insn.b_transpose = insn_data[0];

    // A-type fields
    insn.a_addr_reg = insn_data[58:57];
    insn.a_imm28    = insn_data[56:29];

    // C-type fields
    insn.c_tile_m = insn_data[58:49];
    insn.c_tile_n = insn_data[48:39];
    insn.c_tile_k = insn_data[38:29];
    // m_exact (freeze §6 rev 2026-07-10): exact SFU row count; 0 = full
    // tiles (legacy). Pre-existing bundles carry zeros in [27:16] (the
    // bits were ignored), so decode stays backward byte-compatible.
    insn.c_m_exact = insn_data[27:16];

    // ATTN-type fields
    insn.attn_query_row_base = insn_data[58:47];
    insn.attn_valid_kv_len   = insn_data[46:35];
    insn.attn_mode           = insn_data[34:33];

    // S-type SET_SCALE fields
    insn.s_sreg     = insn_data[58:55];
    insn.s_src_mode = insn_data[54:53];
    insn.s_imm16    = insn_data[52:37];

    // S-type SYNC fields
    insn.sync_mask = insn_data[58:56];
  end

endmodule

`endif // DECODE_UNIT_SV
