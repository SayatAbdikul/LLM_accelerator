# ASIC build entry.
#
# Step E (2026-05-26 RTL restructure): provides a `yosys-asic` elaboration
# smoke gate. OpenLane integration is deferred — `make openlane` prints a
# pointer message and exits non-zero. Default PDK = sky130A; IHP130 path
# can be added by dropping a sibling `sram_dp_ihp130.sv` and switching
# PDK_SRAM_FILE.
#
# Usage:
#   make yosys-asic              — elaboration smoke gate
#   make yosys-asic PDK=sky130A  — same (default PDK)
#   make openlane                — stub: prints "not implemented yet"
#   make clean                   — remove build artifacts

export LC_ALL := C
export LANG   := C

# Default PDK. Override via `make PDK=...` when alternative PDK SRAM
# bindings are added (e.g. sram_dp_ihp130.sv → PDK=ihp130).
PDK ?= sky130A

SRC_DIR      := ../common/src
INCLUDE_DIR  := $(SRC_DIR)/include
SYSTOLIC_DIR := $(SRC_DIR)/systolic
ASIC_DIR     := src
BUILD_DIR    := build

# Shared filelist — single source of truth for the core RTL. Same one
# Verilator, cocotb, and the FPGA Makefile all consume.
CORE_F        := ../common/filelists/core.f
read_filelist  = $(shell grep -E -v '^[[:space:]]*(\#|$$)' $(1))
CORE_SV       := $(addprefix $(SRC_DIR)/, $(call read_filelist,$(CORE_F)))

# PDK-specific SRAM macro file. Resolved via Make's variable-name trick:
# PDK_SRAM_FILE_<pdk> selects the right .sv. Empty if PDK is unknown.
PDK_SRAM_FILE_sky130A := $(ASIC_DIR)/sram_dp_sky130.sv
PDK_SRAM_FILE         := $(PDK_SRAM_FILE_$(PDK))

ifeq ($(PDK_SRAM_FILE),)
  $(error Unknown PDK '$(PDK)' — supported: sky130A. Add a sram_dp_<pdk>.sv file and PDK_SRAM_FILE_<pdk> entry to extend.)
endif

# ASIC target wrappers (order: leaf SRAM macro + pad ring first, then the
# top wrapper that instantiates them).
ASIC_SV := $(PDK_SRAM_FILE) \
           $(ASIC_DIR)/pad_ring_stub.sv \
           $(ASIC_DIR)/taccel_top_asic.sv

# Target axis defines:
#   TARGET_ASIC          — selects sram_dp_macro binding in the common
#                          SRAM dispatch wrapper
#   SFU_SYNTH_NO_DPI     — elides DPI-C imports; required for synth
#   SFU_SYNTH_MODE=1     — selects synthesizable fp32 SFU paths
#   HELPER_SYNTH_MODE=1  — selects synthesizable helper-engine chain
TARGET_DEFS := -DTARGET_ASIC -DSFU_SYNTH_NO_DPI -DSFU_SYNTH_MODE=1 -DHELPER_SYNTH_MODE=1

SYNTH_BUILD := $(BUILD_DIR)/synth
SV2V_OUT    := $(SYNTH_BUILD)/design_asic.v

.PHONY: yosys-asic openlane clean

# sv2v: adapt SystemVerilog to Verilog-2005 with TARGET_ASIC defines.
$(SV2V_OUT): $(CORE_SV) $(ASIC_SV)
	mkdir -p $(SYNTH_BUILD)
	sv2v $(TARGET_DEFS) -I$(INCLUDE_DIR) -I$(SYSTOLIC_DIR) -I$(SRC_DIR) \
	     $(CORE_SV) $(ASIC_SV) -w $(SV2V_OUT)

# Elaboration smoke gate. Yosys must resolve the full taccel_top_asic
# hierarchy with zero unresolved modules. ABC/synth passes are NOT run
# — full physical closure happens in OpenLane once a PDK is installed.
yosys-asic: $(SV2V_OUT)
	@echo "--- yosys ASIC elaboration smoke gate (taccel_top_asic; PDK=$(PDK)) ---"
	yosys -p "read_verilog $(SV2V_OUT); hierarchy -check -top taccel_top_asic; check; stat"

# OpenLane integration is deferred until a PDK is installed locally
# (sky130A via volare, or IHP130 via OpenIHP). Stub prints a pointer
# message and exits non-zero so this target is not mistakenly green.
openlane:
	@echo "OpenLane integration is not implemented yet — pending PDK install."
	@echo "  Selected PDK: $(PDK)"
	@echo "  When ready, populate openlane/config.tcl and libs/<pdk>/ pointers,"
	@echo "  then replace this target with a real OpenLane runscript."
	@exit 1

clean:
	rm -rf $(BUILD_DIR)
