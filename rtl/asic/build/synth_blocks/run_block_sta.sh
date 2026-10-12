#!/bin/bash
# Synthesize <block> to sky130 cells with write_verilog, then run OpenSTA
# at a few clock periods and report fmax. Usage: run_block_sta.sh <block>
set -e

BLOCK="$1"
OUTDIR=rtl/asic/build/synth_blocks
LIB=${LIB:?Set LIB to the selected characterized Liberty file}
STA=${STA:-sta}
DRIVER_CELL=${DRIVER_CELL:?Set DRIVER_CELL to a cell from the selected library}
DRIVER_PIN=${DRIVER_PIN:?Set DRIVER_PIN to its output pin}
PARAMS=""
case "$BLOCK" in
  sfu_engine) PARAMS="chparam -set SFU_SYNTH_MODE 1 sfu_engine;" ;;
  blocking_helper_engine) PARAMS="chparam -set HELPER_SYNTH_MODE 1 blocking_helper_engine;" ;;
  taccel_top) PARAMS="chparam -set SFU_SYNTH_MODE 1 -set HELPER_SYNTH_MODE 1 taccel_top;" ;;
esac

# Synthesize + write the gate-level netlist (the standalone synth_block.sh
# did not write_verilog; replay with one extra step).
yosys -p "
  read_verilog -sv $OUTDIR/${BLOCK}.v;
  $PARAMS
  hierarchy -check -top $BLOCK;
  synth -top $BLOCK -flatten;
  dfflibmap -liberty $LIB;
  abc -liberty $LIB -D 2000;
  clean;
  write_verilog $OUTDIR/${BLOCK}.synth.v
" > "$OUTDIR/${BLOCK}.synth.log" 2>&1

# STA: sweep clock period to find the smallest that closes timing.
cat > "$OUTDIR/${BLOCK}_sta.tcl" <<TCLEOF
read_liberty $LIB
read_verilog $OUTDIR/${BLOCK}.synth.v
link_design $BLOCK
create_clock -period \$::env(PERIOD) -name clk [get_ports clk]
set_input_delay  -clock clk [expr \$::env(PERIOD) * 0.3] [all_inputs]
set_output_delay -clock clk [expr \$::env(PERIOD) * 0.3] [all_outputs]
set_driving_cell -lib_cell $DRIVER_CELL -pin $DRIVER_PIN [all_inputs]
set_load 0.05 [all_outputs]
report_wns
report_tns
report_clock_min_period
exit
TCLEOF

for P in 10.0 5.0 2.0 1.0 0.5 0.3; do
  PERIOD=$P "$STA" -no_splash -exit "$OUTDIR/${BLOCK}_sta.tcl" 2>&1 \
    | grep -E "wns max|period_min" \
    | xargs -I{} echo "  $BLOCK  P=${P}ns  {}"
done
