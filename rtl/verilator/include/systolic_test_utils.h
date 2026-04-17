#pragma once

#include "Vtaccel_top.h"
#include "Vtaccel_top___024root.h"
#include "testbench.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

namespace systolic_test {

constexpr int BUF_ABUF_ID  = 0;
constexpr int BUF_WBUF_ID  = 1;
constexpr int BUF_ACCUM_ID = 2;

constexpr int SYS_DIM = 16;

struct Sim {
  std::unique_ptr<Vtaccel_top> dut;
  AXI4SlaveModel dram;

  Sim() : dut(std::make_unique<Vtaccel_top>()), dram(16 * 1024 * 1024) {
    do_reset(dut.get());
  }

  void load_program(const std::vector<uint64_t>& prog) {
    dram.write_program(prog);
  }

  void run(int timeout = 300000) {
    dut->start = 1;
    tick(dut.get(), dram);
    dut->start = 0;
    run_until_halt(dut.get(), dram, timeout);
  }
};

inline void append_set_addr(std::vector<uint64_t>& prog, int reg, uint64_t addr) {
  prog.push_back(insn::SET_ADDR_LO(reg, static_cast<uint32_t>(addr & 0x0FFFFFFFULL)));
  prog.push_back(insn::SET_ADDR_HI(reg, static_cast<uint32_t>((addr >> 28) & 0x0FFFFFFFULL)));
}

inline void append_load_sync(std::vector<uint64_t>& prog, int reg, uint64_t addr,
                             int buf_id, int sram_off, int xfer_len) {
  append_set_addr(prog, reg, addr);
  prog.push_back(insn::LOAD(buf_id, sram_off, xfer_len, reg, 0));
  prog.push_back(insn::SYNC(0b001));
}

inline void append_prepare_a_tile(std::vector<uint64_t>& prog, int reg, uint64_t addr,
                                  int abuf_off, int xfer_len = (SYS_DIM * SYS_DIM) / 16) {
  append_load_sync(prog, reg, addr, BUF_ABUF_ID, abuf_off, xfer_len);
}

inline std::vector<uint8_t> flatten_16x16(const int8_t (&m)[16][16]) {
  std::vector<uint8_t> out(16 * 16);
  for (int r = 0; r < 16; ++r) {
    for (int c = 0; c < 16; ++c)
      out[r * 16 + c] = static_cast<uint8_t>(m[r][c]);
  }
  return out;
}

inline std::vector<uint8_t> flatten_tile_32x32(const int8_t (&m)[32][32], int row_base, int col_base) {
  std::vector<uint8_t> out(16 * 16);
  for (int r = 0; r < 16; ++r) {
    for (int c = 0; c < 16; ++c)
      out[r * 16 + c] = static_cast<uint8_t>(m[row_base + r][col_base + c]);
  }
  return out;
}

inline std::vector<uint8_t> flatten_tile_16x64(const int8_t (&m)[16][64], int col_base) {
  std::vector<uint8_t> out(16 * 16);
  for (int r = 0; r < 16; ++r) {
    for (int c = 0; c < 16; ++c)
      out[r * 16 + c] = static_cast<uint8_t>(m[r][col_base + c]);
  }
  return out;
}

inline std::vector<uint8_t> flatten_tile_64x16(const int8_t (&m)[64][16], int row_base) {
  std::vector<uint8_t> out(16 * 16);
  for (int r = 0; r < 16; ++r) {
    for (int c = 0; c < 16; ++c)
      out[r * 16 + c] = static_cast<uint8_t>(m[row_base + r][c]);
  }
  return out;
}

inline void write_dram_bytes(AXI4SlaveModel& dram, uint64_t addr, const std::vector<uint8_t>& bytes) {
  dram.write_bytes(addr, bytes.data(), bytes.size());
}

inline void prepare_logical_16x16(AXI4SlaveModel& dram, std::vector<uint64_t>& prog,
                                  const int8_t (&a)[16][16], const int8_t (&b)[16][16],
                                  uint64_t a_addr, uint64_t b_addr,
                                  int abuf_off = 0, int wbuf_off = 0) {
  write_dram_bytes(dram, a_addr, flatten_16x16(a));
  write_dram_bytes(dram, b_addr, flatten_16x16(b));
  append_load_sync(prog, 0, a_addr, BUF_ABUF_ID, abuf_off, (16 * 16) / 16);
  append_load_sync(prog, 1, b_addr, BUF_WBUF_ID, wbuf_off, (16 * 16) / 16);
}

inline void prepare_logical_32x32(AXI4SlaveModel& dram, std::vector<uint64_t>& prog,
                                  const int8_t (&a)[32][32], const int8_t (&b)[32][32],
                                  uint64_t a_base, uint64_t b_base,
                                  int abuf_off = 0, int wbuf_off = 0) {
  std::vector<uint8_t> a_bytes(32 * 32);
  std::vector<uint8_t> b_bytes(32 * 32);
  for (int r = 0; r < 32; ++r) {
    for (int c = 0; c < 32; ++c) {
      a_bytes[r * 32 + c] = static_cast<uint8_t>(a[r][c]);
      b_bytes[r * 32 + c] = static_cast<uint8_t>(b[r][c]);
    }
  }
  write_dram_bytes(dram, a_base, a_bytes);
  write_dram_bytes(dram, b_base, b_bytes);
  append_load_sync(prog, 0, a_base, BUF_ABUF_ID, abuf_off, (32 * 32) / 16);
  append_load_sync(prog, 1, b_base, BUF_WBUF_ID, wbuf_off, (32 * 32) / 16);
}

inline void prepare_logical_16x64x16(AXI4SlaveModel& dram, std::vector<uint64_t>& prog,
                                     const int8_t (&a)[16][64], const int8_t (&b)[64][16],
                                     uint64_t a_base, uint64_t b_base,
                                     int abuf_off = 0, int wbuf_off = 0) {
  std::vector<uint8_t> a_bytes(16 * 64);
  std::vector<uint8_t> b_bytes(64 * 16);
  for (int r = 0; r < 16; ++r)
    for (int c = 0; c < 64; ++c)
      a_bytes[r * 64 + c] = static_cast<uint8_t>(a[r][c]);

  for (int r = 0; r < 64; ++r)
    for (int c = 0; c < 16; ++c)
      b_bytes[r * 16 + c] = static_cast<uint8_t>(b[r][c]);

  write_dram_bytes(dram, a_base, a_bytes);
  write_dram_bytes(dram, b_base, b_bytes);
  append_load_sync(prog, 0, a_base, BUF_ABUF_ID, abuf_off, (16 * 64) / 16);
  append_load_sync(prog, 1, b_base, BUF_WBUF_ID, wbuf_off, (64 * 16) / 16);
}

inline int32_t read_accum_ij(Vtaccel_top* dut, int dst_off, int i, int j) {
  auto* r = dut->rootp;
  int grp = j / 4;
  int lane = j % 4;
  int row = dst_off + i * 4 + grp;
  uint32_t word = r->taccel_top__DOT__u_sram__DOT__u_accum__DOT__mem[row][lane];
  return static_cast<int32_t>(word);
}

inline int32_t read_accum_32x32(Vtaccel_top* dut, int off, int i, int j) {
  int grp = j / 4;
  int lane = j % 4;
  int row = off + i * 8 + grp;
  auto* r = dut->rootp;
  uint32_t word = r->taccel_top__DOT__u_sram__DOT__u_accum__DOT__mem[row][lane];
  return static_cast<int32_t>(word);
}

inline void matmul_ref(const int8_t (&a)[16][16], const int8_t (&b)[16][16], int32_t (&c)[16][16]) {
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 16; ++j) {
      int32_t acc = 0;
      for (int k = 0; k < 16; ++k)
        acc += int32_t(a[i][k]) * int32_t(b[k][j]);
      c[i][j] = acc;
    }
  }
}

inline void matmul_ref_32(const int8_t (&a)[32][32], const int8_t (&b)[32][32], int32_t (&c)[32][32]) {
  for (int i = 0; i < 32; ++i) {
    for (int j = 0; j < 32; ++j) {
      int32_t acc = 0;
      for (int k = 0; k < 32; ++k)
        acc += int32_t(a[i][k]) * int32_t(b[k][j]);
      c[i][j] = acc;
    }
  }
}

inline void matmul_ref_16x64x16(const int8_t (&a)[16][64], const int8_t (&b)[64][16],
                                int32_t (&c)[16][16]) {
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 16; ++j) {
      int32_t acc = 0;
      for (int k = 0; k < 64; ++k)
        acc += int32_t(a[i][k]) * int32_t(b[k][j]);
      c[i][j] = acc;
    }
  }
}

inline bool check_accum_16x16(Vtaccel_top* dut, int dst_off, const int32_t (&exp)[16][16],
                              const char* tag) {
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 16; ++j) {
      int32_t got = read_accum_ij(dut, dst_off, i, j);
      if (got != exp[i][j]) {
        std::fprintf(stderr, "%s mismatch i=%d j=%d got=%d exp=%d\n",
                     tag, i, j, got, exp[i][j]);
        return false;
      }
    }
  }
  return true;
}

inline bool check_accum_32x32(Vtaccel_top* dut, int dst_off, const int32_t (&exp)[32][32],
                              const char* tag) {
  for (int i = 0; i < 32; ++i) {
    for (int j = 0; j < 32; ++j) {
      int32_t got = read_accum_32x32(dut, dst_off, i, j);
      if (got != exp[i][j]) {
        std::fprintf(stderr, "%s mismatch i=%d j=%d got=%d exp=%d\n",
                     tag, i, j, got, exp[i][j]);
        return false;
      }
    }
  }
  return true;
}

}  // namespace systolic_test
