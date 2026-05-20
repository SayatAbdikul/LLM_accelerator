# Verilator build & test Makefile for TACCEL Phase 1
#
# Prerequisites:
#   verilator >= 5.0   (brew install verilator  OR  apt install verilator)
#   g++ / clang++ with C++17
#
# Usage:
#   make           — build + run all Phase 1 tests
#   make test_decode   — decode unit tests only
#   make test_control  — control unit / full-chip tests only
#   make clean     — remove generated objects

export LC_ALL := C
export LANG := C

VERILATOR     ?= verilator
CXX           ?= g++
CXXFLAGS      := -std=c++17 -O2 -Wall -Wno-unused-variable
VFLAGS        := --cc --exe --build --timing \
                 -Wall -Wno-UNUSED -Wno-UNDRIVEN \
                                  --x-assign fast --x-initial fast
SYSTOLIC_ARCH_MODE ?= 1
DRAM_SIZE ?= 16777216

SRC_DIR       := ../src
INCLUDE_DIR   := $(SRC_DIR)/include
SYSTOLIC_DIR  := $(SRC_DIR)/systolic
TB_DIR        := $(SRC_DIR)/tb
BUILD_DIR     := build

# Source file lists
PKG           := $(INCLUDE_DIR)/taccel_pkg.sv
DECODE_SV     := $(PKG) $(SRC_DIR)/decode_unit.sv $(TB_DIR)/tb_decode_unit.sv
CONTROL_SV    := $(PKG) \
                 $(SRC_DIR)/decode_unit.sv \
                 $(SRC_DIR)/fetch_unit.sv \
                 $(SRC_DIR)/control_unit.sv \
                 $(SRC_DIR)/blocking_helper_engine.sv \
                 $(SRC_DIR)/sfu_engine.sv \
                 $(SRC_DIR)/memory/register_file.sv \
                 $(SRC_DIR)/memory/sram_dp.sv \
                 $(SRC_DIR)/memory/sram_subsystem.sv \
				 $(SRC_DIR)/systolic/systolic_pe.sv \
				 $(SRC_DIR)/systolic/systolic_array.sv \
				 $(SRC_DIR)/systolic/systolic_controller.sv \
                 $(SRC_DIR)/dma_engine.sv \
                 $(SRC_DIR)/taccel_top.sv

.PHONY: all run_program test_decode test_control test_dma test_helpers test_sfu \
        test_systolic test_systolic_array_chained test_systolic_chained \
        test_systolic_query test_systolic_qkt test_addr_raw_hazard \
        test_accum_snapshot_readback test_fp32_add test_fp32_add_pipe \
        test_fp32_sub_pipe test_fp32_mul_pipe test_fp16_to_fp32_pipe \
        test_fp32_add_pipe_lat3 synth-check clean

FP32_DIR := $(SRC_DIR)/fp32

all: test_decode test_control test_dma test_helpers test_sfu test_systolic

# -----------------------------------------------------------------------
# run_program: build the full-program Verilator runner
# -----------------------------------------------------------------------
$(BUILD_DIR)/run_program/Vtaccel_top: $(CONTROL_SV) run_program.cpp
	mkdir -p $(BUILD_DIR)/run_program
	$(VERILATOR) $(VFLAGS) \
	  -GSYSTOLIC_ARCH_MODE=$(SYSTOLIC_ARCH_MODE) \
	  -GDRAM_SIZE=$(DRAM_SIZE) \
	  -I$(INCLUDE_DIR) -I$(SYSTOLIC_DIR) \
	  --top-module taccel_top \
	  --Mdir $(BUILD_DIR)/run_program \
	  $(CONTROL_SV) \
	  run_program.cpp \
	  -CFLAGS "-I$(CURDIR)/include $(CXXFLAGS)"
	$(MAKE) -C $(BUILD_DIR)/run_program -f Vtaccel_top.mk

run_program: $(BUILD_DIR)/run_program/Vtaccel_top

# -----------------------------------------------------------------------
# test_decode: verilate tb_decode_unit + compile + run
# -----------------------------------------------------------------------
$(BUILD_DIR)/test_decode/Vtb_decode_unit: $(DECODE_SV) test_decode.cpp
	mkdir -p $(BUILD_DIR)/test_decode
	$(VERILATOR) $(VFLAGS) \
	  -I$(INCLUDE_DIR) -I$(SYSTOLIC_DIR) \
	  --top-module tb_decode_unit \
	  --Mdir $(BUILD_DIR)/test_decode \
	  $(DECODE_SV) \
	  test_decode.cpp \
	  -CFLAGS "-I$(CURDIR)/include $(CXXFLAGS)"
	$(MAKE) -C $(BUILD_DIR)/test_decode -f Vtb_decode_unit.mk

test_decode: $(BUILD_DIR)/test_decode/Vtb_decode_unit
	@echo "--- Running decode unit tests ---"
	$(BUILD_DIR)/test_decode/Vtb_decode_unit

# -----------------------------------------------------------------------
# test_control: verilate taccel_top + compile + run
# -----------------------------------------------------------------------
$(BUILD_DIR)/test_control/Vtaccel_top: $(CONTROL_SV) test_control.cpp
	mkdir -p $(BUILD_DIR)/test_control
	$(VERILATOR) $(VFLAGS) \
	  -GSYSTOLIC_ARCH_MODE=$(SYSTOLIC_ARCH_MODE) \
	  -I$(INCLUDE_DIR) -I$(SYSTOLIC_DIR) \
	  --top-module taccel_top \
	  --Mdir $(BUILD_DIR)/test_control \
	  $(CONTROL_SV) \
	  test_control.cpp \
	  -CFLAGS "-I$(CURDIR)/include $(CXXFLAGS)"
	$(MAKE) -C $(BUILD_DIR)/test_control -f Vtaccel_top.mk

test_control: $(BUILD_DIR)/test_control/Vtaccel_top
	@echo "--- Running control unit tests ---"
	$(BUILD_DIR)/test_control/Vtaccel_top

# -----------------------------------------------------------------------
# test_dma: verilate taccel_top (with DMA engine) + compile + run
# -----------------------------------------------------------------------
$(BUILD_DIR)/test_dma/Vtaccel_top: $(CONTROL_SV) test_dma.cpp
	mkdir -p $(BUILD_DIR)/test_dma
	$(VERILATOR) $(VFLAGS) \
	  -GSYSTOLIC_ARCH_MODE=$(SYSTOLIC_ARCH_MODE) \
	  -I$(INCLUDE_DIR) -I$(SYSTOLIC_DIR) \
	  --top-module taccel_top \
	  --Mdir $(BUILD_DIR)/test_dma \
	  $(CONTROL_SV) \
	  test_dma.cpp \
	  -CFLAGS "-I$(CURDIR)/include $(CXXFLAGS)"
	$(MAKE) -C $(BUILD_DIR)/test_dma -f Vtaccel_top.mk

test_dma: $(BUILD_DIR)/test_dma/Vtaccel_top
	@echo "--- Running DMA tests ---"
	$(BUILD_DIR)/test_dma/Vtaccel_top

# -----------------------------------------------------------------------
# test_helpers: verilate taccel_top (with helper engine) + compile + run
# -----------------------------------------------------------------------
$(BUILD_DIR)/test_helpers/Vtaccel_top: $(CONTROL_SV) test_helpers.cpp
	mkdir -p $(BUILD_DIR)/test_helpers
	$(VERILATOR) $(VFLAGS) \
	  -GSYSTOLIC_ARCH_MODE=$(SYSTOLIC_ARCH_MODE) \
	  -I$(INCLUDE_DIR) -I$(SYSTOLIC_DIR) \
	  --top-module taccel_top \
	  --Mdir $(BUILD_DIR)/test_helpers \
	  $(CONTROL_SV) \
	  test_helpers.cpp \
	  -CFLAGS "-I$(CURDIR)/include $(CXXFLAGS)"
	$(MAKE) -C $(BUILD_DIR)/test_helpers -f Vtaccel_top.mk

test_helpers: $(BUILD_DIR)/test_helpers/Vtaccel_top
	@echo "--- Running helper engine tests ---"
	$(BUILD_DIR)/test_helpers/Vtaccel_top

# -----------------------------------------------------------------------
# test_sfu: verilate taccel_top (with Stage D SFU) + compile + run
# -----------------------------------------------------------------------
$(BUILD_DIR)/test_sfu/Vtaccel_top: $(CONTROL_SV) test_sfu.cpp
	mkdir -p $(BUILD_DIR)/test_sfu
	$(VERILATOR) $(VFLAGS) \
	  -GSYSTOLIC_ARCH_MODE=$(SYSTOLIC_ARCH_MODE) \
	  -I$(INCLUDE_DIR) -I$(SYSTOLIC_DIR) \
	  --top-module taccel_top \
	  --Mdir $(BUILD_DIR)/test_sfu \
	  $(CONTROL_SV) \
	  test_sfu.cpp \
	  -CFLAGS "-I$(CURDIR)/include $(CXXFLAGS)"
	$(MAKE) -C $(BUILD_DIR)/test_sfu -f Vtaccel_top.mk

test_sfu: $(BUILD_DIR)/test_sfu/Vtaccel_top
	@echo "--- Running SFU tests ---"
	$(BUILD_DIR)/test_sfu/Vtaccel_top

# -----------------------------------------------------------------------
# test_systolic: verilate taccel_top (with systolic) + compile + run
# -----------------------------------------------------------------------
$(BUILD_DIR)/test_systolic/Vtaccel_top: $(CONTROL_SV) test_systolic.cpp
	mkdir -p $(BUILD_DIR)/test_systolic
	$(VERILATOR) $(VFLAGS) \
	  -GSYSTOLIC_ARCH_MODE=$(SYSTOLIC_ARCH_MODE) \
	  -I$(INCLUDE_DIR) -I$(SYSTOLIC_DIR) \
	  --top-module taccel_top \
	  --Mdir $(BUILD_DIR)/test_systolic \
	  $(CONTROL_SV) \
	  test_systolic.cpp \
	  -CFLAGS "-I$(CURDIR)/include $(CXXFLAGS)"
	$(MAKE) -C $(BUILD_DIR)/test_systolic -f Vtaccel_top.mk

test_systolic: $(BUILD_DIR)/test_systolic/Vtaccel_top
	@echo "--- Running systolic tests ---"
	$(BUILD_DIR)/test_systolic/Vtaccel_top

# -----------------------------------------------------------------------
# test_systolic_array_chained: unit-test the chained mesh directly
# -----------------------------------------------------------------------
$(BUILD_DIR)/test_systolic_array_chained/Vsystolic_array: \
		$(PKG) $(SRC_DIR)/systolic/systolic_pe.sv $(SRC_DIR)/systolic/systolic_array.sv \
		test_systolic_array_chained.cpp
	mkdir -p $(BUILD_DIR)/test_systolic_array_chained
	$(VERILATOR) $(VFLAGS) \
	  -GSYSTOLIC_ARCH_MODE=1 \
	  -I$(INCLUDE_DIR) -I$(SYSTOLIC_DIR) \
	  --top-module systolic_array \
	  --Mdir $(BUILD_DIR)/test_systolic_array_chained \
	  $(PKG) \
	  $(SRC_DIR)/systolic/systolic_pe.sv \
	  $(SRC_DIR)/systolic/systolic_array.sv \
	  test_systolic_array_chained.cpp \
	  -CFLAGS "-I$(CURDIR)/include $(CXXFLAGS)"
	$(MAKE) -C $(BUILD_DIR)/test_systolic_array_chained -f Vsystolic_array.mk

test_systolic_array_chained: $(BUILD_DIR)/test_systolic_array_chained/Vsystolic_array
	@echo "--- Running chained systolic array tests ---"
	$(BUILD_DIR)/test_systolic_array_chained/Vsystolic_array

# -----------------------------------------------------------------------
# test_systolic_chained: top-level chained MATMUL verification
# -----------------------------------------------------------------------
$(BUILD_DIR)/test_systolic_chained/Vtaccel_top: $(CONTROL_SV) test_systolic_chained.cpp
	mkdir -p $(BUILD_DIR)/test_systolic_chained
	$(VERILATOR) $(VFLAGS) \
	  -GSYSTOLIC_ARCH_MODE=1 \
	  -I$(INCLUDE_DIR) -I$(SYSTOLIC_DIR) \
	  --top-module taccel_top \
	  --Mdir $(BUILD_DIR)/test_systolic_chained \
	  $(CONTROL_SV) \
	  test_systolic_chained.cpp \
	  -CFLAGS "-I$(CURDIR)/include $(CXXFLAGS)"
	$(MAKE) -C $(BUILD_DIR)/test_systolic_chained -f Vtaccel_top.mk

test_systolic_chained: $(BUILD_DIR)/test_systolic_chained/Vtaccel_top
	@echo "--- Running chained top-level systolic tests ---"
	$(BUILD_DIR)/test_systolic_chained/Vtaccel_top

# -----------------------------------------------------------------------
# test_accum_snapshot_readback: #114 ACCUM-snapshot readback reproducer
# -----------------------------------------------------------------------
$(BUILD_DIR)/test_accum_snapshot_readback/Vtaccel_top: $(CONTROL_SV) test_accum_snapshot_readback.cpp
	mkdir -p $(BUILD_DIR)/test_accum_snapshot_readback
	$(VERILATOR) $(VFLAGS) \
	  -GSYSTOLIC_ARCH_MODE=1 \
	  -I$(INCLUDE_DIR) -I$(SYSTOLIC_DIR) \
	  --top-module taccel_top \
	  --Mdir $(BUILD_DIR)/test_accum_snapshot_readback \
	  $(CONTROL_SV) \
	  test_accum_snapshot_readback.cpp \
	  -CFLAGS "-I$(CURDIR)/include $(CXXFLAGS)"
	$(MAKE) -C $(BUILD_DIR)/test_accum_snapshot_readback -f Vtaccel_top.mk

test_accum_snapshot_readback: $(BUILD_DIR)/test_accum_snapshot_readback/Vtaccel_top
	@echo "--- Running #114 ACCUM-snapshot readback reproducer ---"
	$(BUILD_DIR)/test_accum_snapshot_readback/Vtaccel_top

# -----------------------------------------------------------------------
# test_fp32_add: synthesizable IEEE-754 binary32 adder, bit-exact gate.
# Standalone — links ONLY fp32_add (no taccel_top); zero risk to the
# proven byte-exact datapath. First brick of the synthesizable SFU library.
# -----------------------------------------------------------------------
$(BUILD_DIR)/test_fp32_add/Vfp32_add: $(FP32_DIR)/fp32_add.sv test_fp32_add.cpp
	mkdir -p $(BUILD_DIR)/test_fp32_add
	$(VERILATOR) $(VFLAGS) \
	  --top-module fp32_add \
	  --Mdir $(BUILD_DIR)/test_fp32_add \
	  $(FP32_DIR)/fp32_add.sv \
	  test_fp32_add.cpp \
	  -CFLAGS "$(CXXFLAGS)"
	$(MAKE) -C $(BUILD_DIR)/test_fp32_add -f Vfp32_add.mk

test_fp32_add: $(BUILD_DIR)/test_fp32_add/Vfp32_add
	@echo "--- Running synthesizable fp32_add bit-exact gate ---"
	$(BUILD_DIR)/test_fp32_add/Vfp32_add

# -----------------------------------------------------------------------
# test_fp32_add_pipe: pipelined shell over the proven fp32_add core. Proves
# the uniform fixed-latency valid handshake (Phase 1 of the synthesizable
# SFU library). Standalone — links only the fp32 primitive files.
# -----------------------------------------------------------------------
$(BUILD_DIR)/test_fp32_add_pipe/Vfp32_add_pipe: $(FP32_DIR)/fp32_add.sv $(FP32_DIR)/fp32_add_pipe.sv test_fp32_add_pipe.cpp
	mkdir -p $(BUILD_DIR)/test_fp32_add_pipe
	$(VERILATOR) $(VFLAGS) \
	  --top-module fp32_add_pipe \
	  -I$(FP32_DIR) \
	  --Mdir $(BUILD_DIR)/test_fp32_add_pipe \
	  $(FP32_DIR)/fp32_add_pipe.sv \
	  test_fp32_add_pipe.cpp \
	  -CFLAGS "$(CXXFLAGS)"
	$(MAKE) -C $(BUILD_DIR)/test_fp32_add_pipe -f Vfp32_add_pipe.mk

test_fp32_add_pipe: $(BUILD_DIR)/test_fp32_add_pipe/Vfp32_add_pipe
	@echo "--- Running synthesizable fp32_add_pipe handshake+bit-exact gate ---"
	$(BUILD_DIR)/test_fp32_add_pipe/Vfp32_add_pipe

# -----------------------------------------------------------------------
# test_fp32_add_pipe_lat3: same gate built with LATENCY=3 (-GLATENCY=3).
# Proves the shell parameterization works at deeper LATENCY — every
# subsequent primitive's pipelined depth (div/sqrt/exp/tanh, all > 1) will
# inherit this parameter. Catches handshake/in_flight-queue bugs that
# LATENCY=1 happens to mask.
# -----------------------------------------------------------------------
$(BUILD_DIR)/test_fp32_add_pipe_lat3/Vfp32_add_pipe: $(FP32_DIR)/fp32_add.sv $(FP32_DIR)/fp32_add_pipe.sv test_fp32_add_pipe.cpp
	mkdir -p $(BUILD_DIR)/test_fp32_add_pipe_lat3
	$(VERILATOR) $(VFLAGS) \
	  --top-module fp32_add_pipe \
	  -GLATENCY=3 \
	  -I$(FP32_DIR) \
	  --Mdir $(BUILD_DIR)/test_fp32_add_pipe_lat3 \
	  $(FP32_DIR)/fp32_add_pipe.sv \
	  test_fp32_add_pipe.cpp \
	  -CFLAGS "-DPIPE_LATENCY=3 $(CXXFLAGS)"
	$(MAKE) -C $(BUILD_DIR)/test_fp32_add_pipe_lat3 -f Vfp32_add_pipe.mk

test_fp32_add_pipe_lat3: $(BUILD_DIR)/test_fp32_add_pipe_lat3/Vfp32_add_pipe
	@echo "--- Running fp32_add_pipe LATENCY=3 parameterization gate ---"
	$(BUILD_DIR)/test_fp32_add_pipe_lat3/Vfp32_add_pipe

# -----------------------------------------------------------------------
# test_fp32_sub_pipe: synthesizable fp32 subtract via fp32_add core
# (a - b == a + (-b); IEEE negate via sign-bit XOR). Bit-exact gate vs
# the sfu_fp32_sub DPI golden.
# -----------------------------------------------------------------------
$(BUILD_DIR)/test_fp32_sub_pipe/Vfp32_sub_pipe: $(FP32_DIR)/fp32_add.sv $(FP32_DIR)/fp32_add_pipe.sv $(FP32_DIR)/fp32_sub_pipe.sv test_fp32_sub_pipe.cpp
	mkdir -p $(BUILD_DIR)/test_fp32_sub_pipe
	$(VERILATOR) $(VFLAGS) \
	  --top-module fp32_sub_pipe \
	  -I$(FP32_DIR) \
	  --Mdir $(BUILD_DIR)/test_fp32_sub_pipe \
	  $(FP32_DIR)/fp32_sub_pipe.sv \
	  test_fp32_sub_pipe.cpp \
	  -CFLAGS "$(CXXFLAGS)"
	$(MAKE) -C $(BUILD_DIR)/test_fp32_sub_pipe -f Vfp32_sub_pipe.mk

test_fp32_sub_pipe: $(BUILD_DIR)/test_fp32_sub_pipe/Vfp32_sub_pipe
	@echo "--- Running synthesizable fp32_sub_pipe bit-exact gate ---"
	$(BUILD_DIR)/test_fp32_sub_pipe/Vfp32_sub_pipe

# -----------------------------------------------------------------------
# test_fp32_mul_pipe: synthesizable IEEE-754 binary32 multiply, bit-exact
# vs sfu_fp32_mul DPI golden (incl. inf*0 -> qNaN, subnormals, RNE).
# -----------------------------------------------------------------------
$(BUILD_DIR)/test_fp32_mul_pipe/Vfp32_mul_pipe: $(FP32_DIR)/fp32_mul.sv $(FP32_DIR)/fp32_mul_pipe.sv test_fp32_mul_pipe.cpp
	mkdir -p $(BUILD_DIR)/test_fp32_mul_pipe
	$(VERILATOR) $(VFLAGS) \
	  --top-module fp32_mul_pipe \
	  -I$(FP32_DIR) \
	  --Mdir $(BUILD_DIR)/test_fp32_mul_pipe \
	  $(FP32_DIR)/fp32_mul_pipe.sv \
	  test_fp32_mul_pipe.cpp \
	  -CFLAGS "$(CXXFLAGS)"
	$(MAKE) -C $(BUILD_DIR)/test_fp32_mul_pipe -f Vfp32_mul_pipe.mk

test_fp32_mul_pipe: $(BUILD_DIR)/test_fp32_mul_pipe/Vfp32_mul_pipe
	@echo "--- Running synthesizable fp32_mul_pipe bit-exact gate ---"
	$(BUILD_DIR)/test_fp32_mul_pipe/Vfp32_mul_pipe

# -----------------------------------------------------------------------
# test_fp16_to_fp32_pipe: exact (no rounding) fp16->fp32 converter, EXHAUSTIVE
# 65536-pattern gate vs the sfu_fp16_bits_to_fp32 DPI semantics.
# -----------------------------------------------------------------------
$(BUILD_DIR)/test_fp16_to_fp32_pipe/Vfp16_to_fp32_pipe: $(FP32_DIR)/fp16_to_fp32.sv $(FP32_DIR)/fp16_to_fp32_pipe.sv test_fp16_to_fp32_pipe.cpp
	mkdir -p $(BUILD_DIR)/test_fp16_to_fp32_pipe
	$(VERILATOR) $(VFLAGS) \
	  --top-module fp16_to_fp32_pipe \
	  -I$(FP32_DIR) \
	  --Mdir $(BUILD_DIR)/test_fp16_to_fp32_pipe \
	  $(FP32_DIR)/fp16_to_fp32_pipe.sv \
	  test_fp16_to_fp32_pipe.cpp \
	  -CFLAGS "$(CXXFLAGS)"
	$(MAKE) -C $(BUILD_DIR)/test_fp16_to_fp32_pipe -f Vfp16_to_fp32_pipe.mk

test_fp16_to_fp32_pipe: $(BUILD_DIR)/test_fp16_to_fp32_pipe/Vfp16_to_fp32_pipe
	@echo "--- Running EXHAUSTIVE fp16_to_fp32_pipe bit-exact gate ---"
	$(BUILD_DIR)/test_fp16_to_fp32_pipe/Vfp16_to_fp32_pipe

# -----------------------------------------------------------------------
# test_systolic_query: focused chained query-projection regression
# -----------------------------------------------------------------------
$(BUILD_DIR)/test_systolic_query/Vtaccel_top: $(CONTROL_SV) test_systolic_query.cpp
	mkdir -p $(BUILD_DIR)/test_systolic_query
	$(VERILATOR) $(VFLAGS) \
	  -GSYSTOLIC_ARCH_MODE=1 \
	  -I$(INCLUDE_DIR) -I$(SYSTOLIC_DIR) \
	  --top-module taccel_top \
	  --Mdir $(BUILD_DIR)/test_systolic_query \
	  $(CONTROL_SV) \
	  test_systolic_query.cpp \
	  -CFLAGS "-I$(CURDIR)/include $(CXXFLAGS)"
	$(MAKE) -C $(BUILD_DIR)/test_systolic_query -f Vtaccel_top.mk

test_systolic_query: $(BUILD_DIR)/test_systolic_query/Vtaccel_top
	@echo "--- Running focused chained query systolic regression ---"
	$(BUILD_DIR)/test_systolic_query/Vtaccel_top

# -----------------------------------------------------------------------
# test_systolic_qkt: focused QK^T prep + matmul regressions
# -----------------------------------------------------------------------
$(BUILD_DIR)/test_systolic_qkt/Vtaccel_top: $(CONTROL_SV) test_systolic_qkt.cpp
	mkdir -p $(BUILD_DIR)/test_systolic_qkt
	$(VERILATOR) $(VFLAGS) \
	  -GSYSTOLIC_ARCH_MODE=1 \
	  -I$(INCLUDE_DIR) -I$(SYSTOLIC_DIR) \
	  --top-module taccel_top \
	  --Mdir $(BUILD_DIR)/test_systolic_qkt \
	  $(CONTROL_SV) \
	  test_systolic_qkt.cpp \
	  -CFLAGS "-I$(CURDIR)/include $(CXXFLAGS)"
	$(MAKE) -C $(BUILD_DIR)/test_systolic_qkt -f Vtaccel_top.mk

test_systolic_qkt: $(BUILD_DIR)/test_systolic_qkt/Vtaccel_top
	@echo "--- Running focused QK^T regressions ---"
	$(BUILD_DIR)/test_systolic_qkt/Vtaccel_top

# -----------------------------------------------------------------------
# test_addr_raw_hazard: BUG1 reproducer — SET_ADDR_HI → LOAD RAW hazard.
# Requires 512 MB DRAM so both test regions (HI=0 and HI=1) are in-bounds.
# Pre-fix:  raw_hazard_consecutive FAILS (stale HI sampled → wrong address).
# Post-fix: all three sub-tests PASS.
# -----------------------------------------------------------------------
$(BUILD_DIR)/test_addr_raw_hazard/Vtaccel_top: $(CONTROL_SV) test_addr_raw_hazard.cpp
	mkdir -p $(BUILD_DIR)/test_addr_raw_hazard
	$(VERILATOR) $(VFLAGS) \
	  -GSYSTOLIC_ARCH_MODE=$(SYSTOLIC_ARCH_MODE) \
	  -GDRAM_SIZE=536870912 \
	  -I$(INCLUDE_DIR) -I$(SYSTOLIC_DIR) \
	  --top-module taccel_top \
	  --Mdir $(BUILD_DIR)/test_addr_raw_hazard \
	  $(CONTROL_SV) \
	  test_addr_raw_hazard.cpp \
	  -CFLAGS "-I$(CURDIR)/include $(CXXFLAGS)"
	$(MAKE) -C $(BUILD_DIR)/test_addr_raw_hazard -f Vtaccel_top.mk

test_addr_raw_hazard: $(BUILD_DIR)/test_addr_raw_hazard/Vtaccel_top
	@echo "--- Running addr RAW hazard tests ---"
	$(BUILD_DIR)/test_addr_raw_hazard/Vtaccel_top

# -----------------------------------------------------------------------
# synth-check: yosys generic-synthesis gate (no FPGA part). Proves the
# design elaborates to synthesizable logic with zero behavioral (real/DPI)
# constructs — the FPGA-demo roadmap Phase-2 definition of done. Single
# source of truth for the file list is CONTROL_SV (the taccel_top design
# set, tb/ excluded). Currently EXPECTED to fail on sfu_engine.sv +
# blocking_helper_engine.sv (real storage + DPI-C); that red is the work
# list. Goes green when the synthesizable-RTL effort is complete.
# -----------------------------------------------------------------------
SYNTH_DIR    := ../synth
SYNTH_SCRIPT := $(SYNTH_DIR)/synth_check.ys
SYNTH_BUILD  := $(BUILD_DIR)/synth
SV2V_OUT     := $(SYNTH_BUILD)/design_full.v

# sv2v adapts SystemVerilog (packages, enums, ANSI ports, import pkg::*) to
# Verilog-2005 that yosys's built-in frontend parses. yosys then runs the
# synth-check script (hierarchy -> synth -> check -> stat).
$(SV2V_OUT): $(CONTROL_SV)
	mkdir -p $(SYNTH_BUILD)
	sv2v -I$(INCLUDE_DIR) -I$(SYSTOLIC_DIR) $(CONTROL_SV) -w $(SV2V_OUT)

synth-check: $(SV2V_OUT)
	@echo "--- yosys generic synthesizability gate (taccel_top) ---"
	yosys -p "read_verilog $(SV2V_OUT); script $(SYNTH_SCRIPT)"

# -----------------------------------------------------------------------
clean:
	rm -rf $(BUILD_DIR)
