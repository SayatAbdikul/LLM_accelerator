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

SRC_DIR       := ../common/src
INCLUDE_DIR   := $(SRC_DIR)/include
SYSTOLIC_DIR  := $(SRC_DIR)/systolic
TB_DIR        := $(SRC_DIR)/tb
BUILD_DIR     := build

# R6 (2026-05-23): sfu_engine.sv was partitioned into sibling .svh files
# (sfu_dpi_helpers.svh, sfu_synth_datapath.svh, sfu_g1_compute.svh,
# sfu_attn.svh, sfu_g2_compute.svh) via `include directives. Verilator
# needs `-I$(SRC_DIR)` to resolve them since they are NOT in include/
# or systolic/. Added as a VFLAGS extension so every target picks it up.
VFLAGS += -I$(SRC_DIR)

# Source file lists
PKG           := $(INCLUDE_DIR)/taccel_pkg.sv
DECODE_SV     := $(PKG) $(SRC_DIR)/decode_unit.sv $(TB_DIR)/tb_decode_unit.sv

# Step C (2026-05-26 RTL restructure): single source of truth for the
# design source list lives at rtl/common/filelists/core.f (full design)
# and rtl/common/filelists/core_ctrl.f (control-plane subset, used by
# synth-check-ctrl). Each entry is a path relative to rtl/common/src/;
# the read_filelist function strips comments + blanks and the addprefix
# call prepends $(SRC_DIR) to produce a CWD-correct Make list.
CORE_F        := ../common/filelists/core.f
CORE_CTRL_F   := ../common/filelists/core_ctrl.f
read_filelist  = $(shell grep -E -v '^[[:space:]]*(\#|$$)' $(1))

CONTROL_SV    := $(addprefix $(SRC_DIR)/, $(call read_filelist,$(CORE_F)))
CTRL_SV       := $(addprefix $(SRC_DIR)/, $(call read_filelist,$(CORE_CTRL_F)))

.PHONY: all run_program test_decode test_control test_dma test_helpers test_sfu \
        test_systolic test_systolic_array_chained test_systolic_chained \
        test_systolic_query test_systolic_qkt \
        test_systolic_qkt_basic test_systolic_qkt_replay \
        test_systolic_qkt_padded test_systolic_qkt_history \
        test_addr_raw_hazard \
        test_accum_snapshot_readback test_fp32_add test_fp32_div test_fp32_div_p3 test_fp32_div_p4 test_fp32_div_p5 test_fp32_sqrt test_fp32_sqrt_p3 test_fp32_sqrt_p4 \
        test_sfu_synth test_helpers_synth \
        test_sfu_helper_synth synth-check synth-check-ctrl clean
# R7 (2026-05-23): test_fp32_*_pipe targets removed. The 12 fp32_*_pipe.sv
# wrappers and their 12 _pipe.cpp standalone gates were dead code from the
# abandoned "pipelined-first" direction. The combinational cores (fp32_add.sv,
# fp32_mul.sv, ...) remain in FP32_PRIMS and are exercised via test_sfu /
# test_sfu_synth integrations. If pipelining is later wanted, the wrappers
# are 60-line regenerable shells.

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
# test_sfu_synth: same SFU regressions with SFU_SYNTH_MODE=1. Migrated ops
# (currently: 0x19 VADD_FP32) take the synthesizable path through the
# fp32_add + fp32_to_fp16 primitives; unmigrated ops fall back to DPI.
# Used to gate each Phase-2 op migration; the canonical gate is the
# vadd_fp32 fixture under this build.
# -----------------------------------------------------------------------
$(BUILD_DIR)/test_sfu_synth/Vtaccel_top: $(CONTROL_SV) test_sfu.cpp
	mkdir -p $(BUILD_DIR)/test_sfu_synth
	$(VERILATOR) $(VFLAGS) \
	  -GSYSTOLIC_ARCH_MODE=$(SYSTOLIC_ARCH_MODE) \
	  -GSFU_SYNTH_MODE=1 \
	  -I$(INCLUDE_DIR) -I$(SYSTOLIC_DIR) \
	  --top-module taccel_top \
	  --Mdir $(BUILD_DIR)/test_sfu_synth \
	  $(CONTROL_SV) \
	  test_sfu.cpp \
	  -CFLAGS "-I$(CURDIR)/include $(CXXFLAGS)"
	$(MAKE) -C $(BUILD_DIR)/test_sfu_synth -f Vtaccel_top.mk

test_sfu_synth: $(BUILD_DIR)/test_sfu_synth/Vtaccel_top
	@echo "--- Running SFU tests with SFU_SYNTH_MODE=1 (Phase-2 synth path) ---"
	$(BUILD_DIR)/test_sfu_synth/Vtaccel_top

# -----------------------------------------------------------------------
# test_helpers_synth: helper regressions with HELPER_SYNTH_MODE=1. Currently
# migrates the dequant_add_pack op via 16-lane parallel fp32 primitive chain.
# -----------------------------------------------------------------------
$(BUILD_DIR)/test_helpers_synth/Vtaccel_top: $(CONTROL_SV) test_helpers.cpp
	mkdir -p $(BUILD_DIR)/test_helpers_synth
	$(VERILATOR) $(VFLAGS) \
	  -GSYSTOLIC_ARCH_MODE=$(SYSTOLIC_ARCH_MODE) \
	  -GHELPER_SYNTH_MODE=1 \
	  -I$(INCLUDE_DIR) -I$(SYSTOLIC_DIR) \
	  --top-module taccel_top \
	  --Mdir $(BUILD_DIR)/test_helpers_synth \
	  $(CONTROL_SV) \
	  test_helpers.cpp \
	  -CFLAGS "-I$(CURDIR)/include $(CXXFLAGS)"
	$(MAKE) -C $(BUILD_DIR)/test_helpers_synth -f Vtaccel_top.mk

test_helpers_synth: $(BUILD_DIR)/test_helpers_synth/Vtaccel_top
	@echo "--- Running helper tests with HELPER_SYNTH_MODE=1 (Phase-2 synth path) ---"
	$(BUILD_DIR)/test_helpers_synth/Vtaccel_top

# -----------------------------------------------------------------------
# test_sfu_helper_synth: BOTH SFU and HELPER on synth paths simultaneously.
# Catches cross-mode interactions; this is the Phase-2 "full synth path"
# gate for ops/state-machines that have been migrated.
# -----------------------------------------------------------------------
$(BUILD_DIR)/test_sfu_helper_synth/Vtaccel_top: $(CONTROL_SV) test_sfu.cpp
	mkdir -p $(BUILD_DIR)/test_sfu_helper_synth
	$(VERILATOR) $(VFLAGS) \
	  -GSYSTOLIC_ARCH_MODE=$(SYSTOLIC_ARCH_MODE) \
	  -GSFU_SYNTH_MODE=1 \
	  -GHELPER_SYNTH_MODE=1 \
	  -I$(INCLUDE_DIR) -I$(SYSTOLIC_DIR) \
	  --top-module taccel_top \
	  --Mdir $(BUILD_DIR)/test_sfu_helper_synth \
	  $(CONTROL_SV) \
	  test_sfu.cpp \
	  -CFLAGS "-I$(CURDIR)/include $(CXXFLAGS)"
	$(MAKE) -C $(BUILD_DIR)/test_sfu_helper_synth -f Vtaccel_top.mk

test_sfu_helper_synth: $(BUILD_DIR)/test_sfu_helper_synth/Vtaccel_top
	@echo "--- Running SFU tests with BOTH SFU_SYNTH_MODE=1 and HELPER_SYNTH_MODE=1 ---"
	$(BUILD_DIR)/test_sfu_helper_synth/Vtaccel_top

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
# test_fp32_div: 2-stage pipelined fp32_div_p2, bit-exact gate.
# Standalone — links ONLY fp32_div_p2 (no taccel_top). Proves the
# path-cutting divider is byte-identical to the host-float golden before
# any sfu_engine.sv integration.
# -----------------------------------------------------------------------
$(BUILD_DIR)/test_fp32_div/Vfp32_div_p2: $(FP32_DIR)/fp32_div_p2.sv test_fp32_div.cpp
	mkdir -p $(BUILD_DIR)/test_fp32_div
	$(VERILATOR) $(VFLAGS) \
	  --top-module fp32_div_p2 \
	  --Mdir $(BUILD_DIR)/test_fp32_div \
	  $(FP32_DIR)/fp32_div_p2.sv \
	  test_fp32_div.cpp \
	  -CFLAGS "$(CXXFLAGS)"
	$(MAKE) -C $(BUILD_DIR)/test_fp32_div -f Vfp32_div_p2.mk

test_fp32_div: $(BUILD_DIR)/test_fp32_div/Vfp32_div_p2
	@echo "--- Running pipelined fp32_div_p2 bit-exact gate ---"
	$(BUILD_DIR)/test_fp32_div/Vfp32_div_p2

# -----------------------------------------------------------------------
# test_fp32_div_p3: 3-stage pipelined fp32_div_p3, bit-exact gate.
# Standalone — links ONLY fp32_div_p3 (no taccel_top). Same byte-exact
# streaming harness as test_fp32_div (the in-flight FIFO auto-handles the
# LATENCY=3 depth).
# -----------------------------------------------------------------------
$(BUILD_DIR)/test_fp32_div_p3/Vfp32_div_p3: $(FP32_DIR)/fp32_div_p3.sv test_fp32_div_p3.cpp
	mkdir -p $(BUILD_DIR)/test_fp32_div_p3
	$(VERILATOR) $(VFLAGS) \
	  --top-module fp32_div_p3 \
	  --Mdir $(BUILD_DIR)/test_fp32_div_p3 \
	  $(FP32_DIR)/fp32_div_p3.sv \
	  test_fp32_div_p3.cpp \
	  -CFLAGS "$(CXXFLAGS)"
	$(MAKE) -C $(BUILD_DIR)/test_fp32_div_p3 -f Vfp32_div_p3.mk

test_fp32_div_p3: $(BUILD_DIR)/test_fp32_div_p3/Vfp32_div_p3
	@echo "--- Running pipelined fp32_div_p3 bit-exact gate ---"
	$(BUILD_DIR)/test_fp32_div_p3/Vfp32_div_p3

# -----------------------------------------------------------------------
# test_fp32_div_p4: 4-stage pipelined fp32_div_p4, bit-exact gate.
# Standalone — links ONLY fp32_div_p4 (no taccel_top). Same byte-exact
# streaming harness as test_fp32_div (the in-flight FIFO auto-handles the
# LATENCY=4 depth).
# -----------------------------------------------------------------------
$(BUILD_DIR)/test_fp32_div_p4/Vfp32_div_p4: $(FP32_DIR)/fp32_div_p4.sv test_fp32_div_p4.cpp
	mkdir -p $(BUILD_DIR)/test_fp32_div_p4
	$(VERILATOR) $(VFLAGS) \
	  --top-module fp32_div_p4 \
	  --Mdir $(BUILD_DIR)/test_fp32_div_p4 \
	  $(FP32_DIR)/fp32_div_p4.sv \
	  test_fp32_div_p4.cpp \
	  -CFLAGS "$(CXXFLAGS)"
	$(MAKE) -C $(BUILD_DIR)/test_fp32_div_p4 -f Vfp32_div_p4.mk

test_fp32_div_p4: $(BUILD_DIR)/test_fp32_div_p4/Vfp32_div_p4
	@echo "--- Running pipelined fp32_div_p4 bit-exact gate ---"
	$(BUILD_DIR)/test_fp32_div_p4/Vfp32_div_p4

# -----------------------------------------------------------------------
# test_fp32_div_p5: 5-stage pipelined fp32_div_p5, bit-exact gate.
# Standalone — links ONLY fp32_div_p5 (no taccel_top). Same byte-exact
# streaming harness as test_fp32_div (the in-flight FIFO auto-handles the
# LATENCY=5 depth).
# -----------------------------------------------------------------------
$(BUILD_DIR)/test_fp32_div_p5/Vfp32_div_p5: $(FP32_DIR)/fp32_div_p5.sv test_fp32_div_p5.cpp
	mkdir -p $(BUILD_DIR)/test_fp32_div_p5
	$(VERILATOR) $(VFLAGS) \
	  --top-module fp32_div_p5 \
	  --Mdir $(BUILD_DIR)/test_fp32_div_p5 \
	  $(FP32_DIR)/fp32_div_p5.sv \
	  test_fp32_div_p5.cpp \
	  -CFLAGS "$(CXXFLAGS)"
	$(MAKE) -C $(BUILD_DIR)/test_fp32_div_p5 -f Vfp32_div_p5.mk

test_fp32_div_p5: $(BUILD_DIR)/test_fp32_div_p5/Vfp32_div_p5
	@echo "--- Running pipelined fp32_div_p5 bit-exact gate ---"
	$(BUILD_DIR)/test_fp32_div_p5/Vfp32_div_p5

# -----------------------------------------------------------------------
# test_fp32_sqrt: 2-stage pipelined fp32_sqrt_p2, bit-exact gate.
# Standalone — links ONLY fp32_sqrt_p2 (no taccel_top). Proves the
# path-cutting sqrt is byte-identical to the host-float golden before
# any sfu_engine.sv integration.
# -----------------------------------------------------------------------
$(BUILD_DIR)/test_fp32_sqrt/Vfp32_sqrt_p2: $(FP32_DIR)/fp32_sqrt_p2.sv test_fp32_sqrt.cpp
	mkdir -p $(BUILD_DIR)/test_fp32_sqrt
	$(VERILATOR) $(VFLAGS) \
	  --top-module fp32_sqrt_p2 \
	  --Mdir $(BUILD_DIR)/test_fp32_sqrt \
	  $(FP32_DIR)/fp32_sqrt_p2.sv \
	  test_fp32_sqrt.cpp \
	  -CFLAGS "$(CXXFLAGS)"
	$(MAKE) -C $(BUILD_DIR)/test_fp32_sqrt -f Vfp32_sqrt_p2.mk

test_fp32_sqrt: $(BUILD_DIR)/test_fp32_sqrt/Vfp32_sqrt_p2
	@echo "--- Running pipelined fp32_sqrt_p2 bit-exact gate ---"
	$(BUILD_DIR)/test_fp32_sqrt/Vfp32_sqrt_p2

# -----------------------------------------------------------------------
# test_fp32_sqrt_p3: 3-stage pipelined fp32_sqrt_p3, bit-exact gate.
# Standalone — links ONLY fp32_sqrt_p3 (no taccel_top). Proves the
# 3-way path-cut sqrt is byte-identical to the host-float golden before
# any sfu_engine.sv integration.
# -----------------------------------------------------------------------
$(BUILD_DIR)/test_fp32_sqrt_p3/Vfp32_sqrt_p3: $(FP32_DIR)/fp32_sqrt_p3.sv test_fp32_sqrt_p3.cpp
	mkdir -p $(BUILD_DIR)/test_fp32_sqrt_p3
	$(VERILATOR) $(VFLAGS) \
	  --top-module fp32_sqrt_p3 \
	  --Mdir $(BUILD_DIR)/test_fp32_sqrt_p3 \
	  $(FP32_DIR)/fp32_sqrt_p3.sv \
	  test_fp32_sqrt_p3.cpp \
	  -CFLAGS "$(CXXFLAGS)"
	$(MAKE) -C $(BUILD_DIR)/test_fp32_sqrt_p3 -f Vfp32_sqrt_p3.mk

test_fp32_sqrt_p3: $(BUILD_DIR)/test_fp32_sqrt_p3/Vfp32_sqrt_p3
	@echo "--- Running pipelined fp32_sqrt_p3 bit-exact gate ---"
	$(BUILD_DIR)/test_fp32_sqrt_p3/Vfp32_sqrt_p3

# -----------------------------------------------------------------------
# test_fp32_sqrt_p4: 4-stage pipelined fp32_sqrt_p4, bit-exact gate.
# Standalone — links ONLY fp32_sqrt_p4 (no taccel_top). Proves the
# 4-way path-cut sqrt is byte-identical to the host-float golden before
# any sfu_engine.sv integration.
# -----------------------------------------------------------------------
$(BUILD_DIR)/test_fp32_sqrt_p4/Vfp32_sqrt_p4: $(FP32_DIR)/fp32_sqrt_p4.sv test_fp32_sqrt_p4.cpp
	mkdir -p $(BUILD_DIR)/test_fp32_sqrt_p4
	$(VERILATOR) $(VFLAGS) \
	  --top-module fp32_sqrt_p4 \
	  --Mdir $(BUILD_DIR)/test_fp32_sqrt_p4 \
	  $(FP32_DIR)/fp32_sqrt_p4.sv \
	  test_fp32_sqrt_p4.cpp \
	  -CFLAGS "$(CXXFLAGS)"
	$(MAKE) -C $(BUILD_DIR)/test_fp32_sqrt_p4 -f Vfp32_sqrt_p4.mk

test_fp32_sqrt_p4: $(BUILD_DIR)/test_fp32_sqrt_p4/Vfp32_sqrt_p4
	@echo "--- Running pipelined fp32_sqrt_p4 bit-exact gate ---"
	$(BUILD_DIR)/test_fp32_sqrt_p4/Vfp32_sqrt_p4

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
# test_systolic_qkt_{basic,replay,padded,history}: R4-split QK^T regressions.
# The pre-split monolithic test_systolic_qkt.cpp (3,672 LOC, 31 tests) was
# divided 2026-05-22 into four topic-based translation units. Shared helpers
# live in include/systolic_qkt_utils.h. Each split file builds its own
# Verilator-wrapped binary; the aggregate `test_systolic_qkt` target runs
# all four for backward-compat with scripts that invoked the old name.
# -----------------------------------------------------------------------
define QKT_SPLIT_RULES
$$(BUILD_DIR)/test_systolic_qkt_$(1)/Vtaccel_top: $$(CONTROL_SV) test_systolic_qkt_$(1).cpp include/systolic_qkt_utils.h
	mkdir -p $$(BUILD_DIR)/test_systolic_qkt_$(1)
	$$(VERILATOR) $$(VFLAGS) \
	  -GSYSTOLIC_ARCH_MODE=1 \
	  -I$$(INCLUDE_DIR) -I$$(SYSTOLIC_DIR) \
	  --top-module taccel_top \
	  --Mdir $$(BUILD_DIR)/test_systolic_qkt_$(1) \
	  $$(CONTROL_SV) \
	  test_systolic_qkt_$(1).cpp \
	  -CFLAGS "-I$$(CURDIR)/include $$(CXXFLAGS)"
	$$(MAKE) -C $$(BUILD_DIR)/test_systolic_qkt_$(1) -f Vtaccel_top.mk

test_systolic_qkt_$(1): $$(BUILD_DIR)/test_systolic_qkt_$(1)/Vtaccel_top
	@echo "--- Running QK^T $(1) regressions ---"
	$$(BUILD_DIR)/test_systolic_qkt_$(1)/Vtaccel_top
endef

$(eval $(call QKT_SPLIT_RULES,basic))
$(eval $(call QKT_SPLIT_RULES,replay))
$(eval $(call QKT_SPLIT_RULES,padded))
$(eval $(call QKT_SPLIT_RULES,history))

# Aggregate target — runs all four split binaries.
test_systolic_qkt: test_systolic_qkt_basic test_systolic_qkt_replay \
                   test_systolic_qkt_padded test_systolic_qkt_history

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
	sv2v -DSFU_SYNTH_NO_DPI -I$(INCLUDE_DIR) -I$(SYSTOLIC_DIR) $(CONTROL_SV) -w $(SV2V_OUT)

synth-check: $(SV2V_OUT)
	@echo "--- yosys generic synthesizability gate (taccel_top) ---"
	yosys -p "read_verilog $(SV2V_OUT); script $(SYNTH_SCRIPT)"

# -----------------------------------------------------------------------
# synth-check-ctrl: control-plane synthesizability gate (Phase-3).
# Reads CONTROL_SV MINUS sfu_engine.sv + blocking_helper_engine.sv, with
# those stubbed via rtl/synth/blackbox_stubs.v. Proves the entire control
# plane (decode/fetch/control/regfile/sram/systolic/dma/top) elaborates
# and structurally synthesizes through yosys. Lightweight pipeline
# (proc/flatten/opt-fast/check/stat) — skips ABC tech-mapping for speed.
# -----------------------------------------------------------------------
SV2V_CTRL := $(SYNTH_BUILD)/design_ctrl.v
# CTRL_SV is defined alongside CONTROL_SV at the top of this Makefile, both
# derived from rtl/common/filelists/ via read_filelist (Step C, 2026-05-26).

$(SV2V_CTRL): $(CTRL_SV)
	mkdir -p $(SYNTH_BUILD)
	sv2v -I$(INCLUDE_DIR) -I$(SYSTOLIC_DIR) $(CTRL_SV) -w $(SV2V_CTRL)

synth-check-ctrl: $(SV2V_CTRL)
	@echo "--- yosys control-plane synth-check (SFU/helper blackboxed) ---"
	yosys -p "read_verilog -lib $(SYNTH_DIR)/blackbox_stubs.v; read_verilog $(SV2V_CTRL); script $(SYNTH_DIR)/synth_check_ctrl.ys"

# -----------------------------------------------------------------------
clean:
	rm -rf $(BUILD_DIR)
