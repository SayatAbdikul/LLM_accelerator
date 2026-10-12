# ASIC build entry.
#
# Provides a `yosys-asic` wrapper-elaboration gate. `make openlane` is an
# intentionally failing placeholder: no production OpenLane configuration,
# macro set, or portable PDK binding has been integrated. `sky130A` currently
# selects only the behavioral SRAM compatibility module below.
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

# Shared filelist — single source of truth for the core RTL. The same one
# Verilator and the FPGA Makefile consume.
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
# Compute modes are RTL parameters passed explicitly by taccel_top_asic.
TARGET_DEFS := -DTARGET_ASIC -DSFU_SYNTH_NO_DPI

SYNTH_BUILD := $(BUILD_DIR)/synth
SV2V_OUT    := $(SYNTH_BUILD)/design_asic.v

.PHONY: yosys-asic openlane test-sram-bank clean

# Experimental macro adapter is not wired into the production SRAM contract.
# Fetch the pinned vendor simulation view into this ignored cache first.
SRAM_MODEL ?= cache/sky130_sram/sky130_sram_2kbyte_1rw1r_32x512_8.v
IVERILOG ?= iverilog
VVP ?= vvp
$(BUILD_DIR)/test-sram-bank.vvp: src/sram_dp_sky130_banked.sv tests/test_sram_dp_sky130_banked.sv ../common/src/memory/sram_dp_inferred.sv $(SRAM_MODEL) Makefile
	mkdir -p $(BUILD_DIR)
	$(IVERILOG) -g2012 -s test_sram_dp_sky130_banked -o $@ \
	  src/sram_dp_sky130_banked.sv tests/test_sram_dp_sky130_banked.sv \
	  ../common/src/memory/sram_dp_inferred.sv $(SRAM_MODEL)

test-sram-bank: $(BUILD_DIR)/test-sram-bank.vvp
	python3 tools/fetch_macro_views.py --verify-only
	$(VVP) $<

# sv2v: adapt SystemVerilog to Verilog-2005 with TARGET_ASIC defines.
$(SV2V_OUT): $(CORE_SV) $(wildcard $(SRC_DIR)/*.svh) $(ASIC_SV) $(CORE_F) Makefile
	mkdir -p $(SYNTH_BUILD)
	sv2v $(TARGET_DEFS) -I$(INCLUDE_DIR) -I$(SYSTOLIC_DIR) -I$(SRC_DIR) \
	     $(CORE_SV) $(ASIC_SV) -w $(SV2V_OUT)

# Elaboration smoke gate. Yosys must resolve the full taccel_top_asic
# hierarchy with zero unresolved modules. ABC/synth passes are NOT run
# — full physical closure requires a separately integrated physical flow.
# Raw-process check is diagnostic, not a fatal post-lowering structural proof.
yosys-asic: $(SV2V_OUT)
	@echo "--- yosys ASIC elaboration smoke gate (taccel_top_asic; PDK=$(PDK)) ---"
	yosys -p "read_verilog $(SV2V_OUT); hierarchy -check -top taccel_top_asic; check; stat"

# OpenLane integration is not implemented. The stub exits non-zero so this
# target cannot be mistaken for a successful physical-design run.
openlane:
	@echo "OpenLane integration is not implemented."
	@echo "  Selected PDK: $(PDK)"
	@echo "  Add a portable PDK/macro configuration and a real run script first."
	@exit 1

clean:
	rm -rf $(BUILD_DIR)
