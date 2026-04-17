# cocotb test Makefile for TACCEL Phase 1
#
# Prerequisites:
#   pip install cocotb cocotb-bus
#   iverilog >= 11  OR  verilator >= 5.0  OR  cadence/synopsys VCS
#
# Usage:
#   make                  - run the default stable top-level cocotb suite
#   make SIM=verilator    - run with Verilator (sign-off path)
#   make test_dma SIM=verilator
#   make test_systolic_chained SIM=verilator
#   make clean

export LC_ALL := C
export LANG := C

TOPLEVEL_LANG ?= verilog
SIM           ?= icarus     # icarus | verilator | vcs | xsim
SYSTOLIC_ARCH_MODE ?= 1

TOPLEVEL      := taccel_top
COCOTB_TEST_MODULES := test_fetch_decode,test_dma,test_helpers,test_sfu,test_systolic
MODULE ?= $(COCOTB_TEST_MODULES)

SRC_DIR       := ../src
INCLUDE_DIR   := $(SRC_DIR)/include

# All SV source files for Phase 2 (order matters for Icarus)
VERILOG_SOURCES := \
  $(INCLUDE_DIR)/taccel_pkg.sv \
  $(SRC_DIR)/decode_unit.sv \
  $(SRC_DIR)/fetch_unit.sv \
  $(SRC_DIR)/control_unit.sv \
  $(SRC_DIR)/blocking_helper_engine.sv \
  $(SRC_DIR)/sfu_engine.sv \
  $(SRC_DIR)/memory/sram_dp.sv \
  $(SRC_DIR)/memory/register_file.sv \
  $(SRC_DIR)/memory/sram_subsystem.sv \
  $(SRC_DIR)/systolic/systolic_pe.sv \
  $(SRC_DIR)/systolic/systolic_array.sv \
  $(SRC_DIR)/systolic/systolic_controller.sv \
  $(SRC_DIR)/dma_engine.sv \
  $(SRC_DIR)/taccel_top.sv

# Include paths
COMPILE_ARGS += -I$(INCLUDE_DIR)
COMPILE_ARGS += -I$(SRC_DIR)/systolic

# Verilator-specific flags
ifeq ($(SIM), verilator)
  COMPILE_ARGS += --timing -Wall -Wno-UNUSED -Wno-UNDRIVEN
  COMPILE_ARGS += -GSYSTOLIC_ARCH_MODE=$(SYSTOLIC_ARCH_MODE)
endif

# Icarus-specific: use SystemVerilog mode
ifeq ($(SIM), icarus)
  COMPILE_ARGS += -g2012
endif

# cocotb makefile magic
include $(shell cocotb-config --makefiles)/Makefile.sim

.PHONY: clean test_all test_fetch_decode test_dma test_helpers test_sfu test_systolic test_systolic_chained

test_all:
	$(MAKE) SIM=$(SIM) MODULE=test_fetch_decode,test_dma,test_helpers,test_sfu,test_systolic COCOTB_TEST_MODULES=test_fetch_decode,test_dma,test_helpers,test_sfu,test_systolic sim

test_fetch_decode:
	$(MAKE) SIM=$(SIM) MODULE=test_fetch_decode COCOTB_TEST_MODULES=test_fetch_decode sim

test_dma:
	$(MAKE) SIM=$(SIM) MODULE=test_dma COCOTB_TEST_MODULES=test_dma sim

test_helpers:
	$(MAKE) SIM=$(SIM) MODULE=test_helpers COCOTB_TEST_MODULES=test_helpers sim

test_sfu:
	$(MAKE) SIM=$(SIM) MODULE=test_sfu COCOTB_TEST_MODULES=test_sfu sim

test_systolic:
	$(MAKE) SIM=$(SIM) MODULE=test_systolic COCOTB_TEST_MODULES=test_systolic sim

test_systolic_chained:
	$(MAKE) SIM=$(SIM) MODULE=test_systolic_chained COCOTB_TEST_MODULES=test_systolic_chained sim

clean::
	rm -rf sim_build __pycache__ results.xml *.vcd dump.* \
	       $(TOPLEVEL).vvp iverilog_dump.v
