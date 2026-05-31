# TACCEL design source list — single source of truth.
#
# Paths in this file are relative to rtl/common/src/. Each per-target
# Makefile (Verilator, cocotb, future FPGA/ASIC) prepends its own SRC_DIR
# before passing the list to its tool. Lines starting with `#` and blank
# lines are stripped.
#
# This keeps the filelist tool-agnostic and CWD-independent — the same
# list drives Verilator, sv2v + Yosys, eventual Vivado/OpenLane flows,
# etc. Adding a new source means adding one line here.
#
# Order matters for tools that lack ordered-elaboration discovery
# (Icarus, some sv2v paths). General convention:
#   1. package
#   2. leaf primitives (fp32)
#   3. execute units that may consume them
#   4. memory subsystem (sram_dp_inferred before sram_dp wrapper)
#   5. systolic array (pe → array → controller)
#   6. dma + top

include/taccel_pkg.sv

# FP32 primitives — referenced by SFU + helper engines when
# SFU_SYNTH_MODE=1 / HELPER_SYNTH_MODE=1.
fp32/fp32_add.sv
fp32/fp32_mul.sv
fp32/fp32_div.sv
fp32/fp32_div_p2.sv
fp32/fp32_div_p3.sv
fp32/fp32_div_p4.sv
fp32/fp32_sqrt.sv
fp32/fp32_sqrt_p2.sv
fp32/fp32_sqrt_p3.sv
fp32/fp32_sqrt_p4.sv
fp32/fp32_to_fp16.sv
fp32/fp16_to_fp32.sv
fp32/i32_to_fp32.sv
fp32/fp32_quantize_i8.sv
fp32/fp32_exp.sv
fp32/fp32_gelu_new.sv

# Execute units
decode_unit.sv
fetch_unit.sv
control_unit.sv
blocking_helper_engine.sv
sfu_engine.sv

# Memory subsystem (sram_dp_inferred is the body for SIM/FPGA targets;
# sram_dp is the target-dispatch wrapper — must be elaborated together).
memory/register_file.sv
memory/sram_dp_inferred.sv
memory/sram_dp.sv
memory/sram_subsystem.sv

# Systolic engine
systolic/systolic_pe.sv
systolic/systolic_array.sv
systolic/systolic_controller.sv

# DMA + top
dma_engine.sv
taccel_top.sv
