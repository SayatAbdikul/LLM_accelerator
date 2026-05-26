# TACCEL control-plane source subset.
#
# Same format as core.f. Drops sfu_engine.sv and blocking_helper_engine.sv —
# those are blackboxed via rtl/synth/blackbox_stubs.v for the synth-check-ctrl
# gate (a lightweight ~5s yosys pass on decode/fetch/control/memory/systolic/
# dma/top, skipping the heavyweight SFU + helper). Used by the Makefile rule
# `synth-check-ctrl`.

include/taccel_pkg.sv
decode_unit.sv
fetch_unit.sv
control_unit.sv
memory/register_file.sv
memory/sram_dp_inferred.sv
memory/sram_dp.sv
memory/sram_subsystem.sv
systolic/systolic_pe.sv
systolic/systolic_array.sv
systolic/systolic_controller.sv
dma_engine.sv
taccel_top.sv
