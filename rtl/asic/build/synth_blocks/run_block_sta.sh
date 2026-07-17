#!/bin/bash
# Synthesize <block> to sky130 cells with write_verilog, then run OpenSTA
# at a few clock periods and report fmax. Usage: run_block_sta.sh <block>
set -e

BLOCK="$1"
OUTDIR=rtl/asic/build/synth_blocks
LIB=/home/user/.ciel/ciel/sky130/versions/61a056e180dac7dcc6d4eb7529e2231f95105746/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib

# Synthesize + write the gate-level netlist (the standalone synth_block.sh
# did not write_verilog; replay with one extra step).
yosys -p "
  read_verilog -sv $OUTDIR/${BLOCK}.v;
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
set_driving_cell -lib_cell sky130_fd_sc_hd__inv_2 -pin Y [all_inputs]
set_load 0.05 [all_outputs]
report_wns
report_tns
report_clock_min_period
exit
TCLEOF

for P in 10.0 5.0 2.0 1.0 0.5 0.3; do
  PERIOD=$P /tmp/gcc-shim/sta -no_splash -exit "$OUTDIR/${BLOCK}_sta.tcl" 2>&1 \
    | grep -E "wns max|period_min" \
    | xargs -I{} echo "  $BLOCK  P=${P}ns  {}"
done
