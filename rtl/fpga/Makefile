# FPGA build entry.
#
# Step D (2026-05-26 RTL restructure): provides a `yosys-fpga` elaboration
# smoke gate. NO target FPGA part is picked yet, so full Vivado/Quartus/
# nextpnr integration is deferred. When a part is picked, add a `vivado`
# (or `quartus`/`nextpnr`) target here, point at the shared filelist below,
# and add a constraints file under constraints/.
#
# Usage:
#   make yosys-fpga   — elaboration smoke gate; must elaborate cleanly
#   make clean        — remove build artifacts

export LC_ALL := C
export LANG   := C

SRC_DIR      := ../common/src
INCLUDE_DIR  := $(SRC_DIR)/include
SYSTOLIC_DIR := $(SRC_DIR)/systolic
FPGA_DIR     := src
BUILD_DIR    := build

# Shared filelist — single source of truth for the core RTL. The same one
# Verilator and ASIC builds consume; per-target wrappers ride on top.
CORE_F        := ../common/filelists/core.f
read_filelist  = $(shell grep -E -v '^[[:space:]]*(\#|$$)' $(1))
CORE_SV       := $(addprefix $(SRC_DIR)/, $(call read_filelist,$(CORE_F)))

# FPGA target-specific wrappers (order: leaf stubs first, then DDR stub,
# then the top wrapper that instantiates them all).
FPGA_SV       := $(FPGA_DIR)/pll_stub.sv \
                 $(FPGA_DIR)/iobuf_stub.sv \
                 $(FPGA_DIR)/ddr_axi_stub.sv \
                 $(FPGA_DIR)/taccel_top_fpga.sv

# Target axis defines:
#   TARGET_FPGA          — selects FPGA bindings (currently FPGA falls
#                          through to sram_dp_inferred; this define
#                          is set for future explicit FPGA paths and
#                          for the `ifndef SFU_SYNTH_NO_DPI \`error`
#                          guard in the wrapper)
#   SFU_SYNTH_NO_DPI     — elides DPI-C imports; required for synth
#   SFU_SYNTH_MODE=1     — selects synthesizable fp32 SFU paths
#   HELPER_SYNTH_MODE=1  — selects synthesizable helper-engine chain
TARGET_DEFS  := -DTARGET_FPGA -DSFU_SYNTH_NO_DPI -DSFU_SYNTH_MODE=1 -DHELPER_SYNTH_MODE=1

SYNTH_BUILD  := $(BUILD_DIR)/synth
SV2V_OUT     := $(SYNTH_BUILD)/design_fpga.v

.PHONY: yosys-fpga clean

# sv2v: adapt SystemVerilog (packages, enums, ANSI ports) to Verilog-2005
# for yosys consumption. Includes paths are critical — package + sibling
# .svh files live in include/, systolic/, and the src/ root.
$(SV2V_OUT): $(CORE_SV) $(FPGA_SV)
	mkdir -p $(SYNTH_BUILD)
	sv2v $(TARGET_DEFS) -I$(INCLUDE_DIR) -I$(SYSTOLIC_DIR) -I$(SRC_DIR) \
	     $(CORE_SV) $(FPGA_SV) -w $(SV2V_OUT)

# Elaboration smoke gate. Definition of done: yosys resolves the full
# taccel_top_fpga hierarchy with zero unresolved modules. Aggressive
# synthesis (synth_xilinx etc.) is NOT run here — that closure happens
# in Vivado/nextpnr once a part is picked.
yosys-fpga: $(SV2V_OUT)
	@echo "--- yosys FPGA elaboration smoke gate (taccel_top_fpga) ---"
	yosys -p "read_verilog $(SV2V_OUT); hierarchy -check -top taccel_top_fpga; check; stat"

clean:
	rm -rf $(BUILD_DIR)
