#!/bin/bash
# Generic block STA driver: read netlist, sweep clock periods, report fmax.
# Usage: sta_one.sh <block_name> [<netlist_suffix>]
#   netlist_suffix defaults to "synth_sky.v" (the retimed netlist).
set -e
BLOCK="$1"
SUFFIX="${2:-synth_sky.v}"
OUTDIR=rtl/asic/build/synth_blocks
LIB=/home/user/.ciel/ciel/sky130/versions/61a056e180dac7dcc6d4eb7529e2231f95105746/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib

cat > "$OUTDIR/${BLOCK}_sweep.tcl" <<TCLEOF
read_liberty $LIB
read_verilog $OUTDIR/${BLOCK}.${SUFFIX}
link_design $BLOCK
create_clock -period \$::env(PERIOD) -name clk [get_ports clk]
set_input_delay  -clock clk 0.0 [all_inputs]
set_output_delay -clock clk 0.0 [all_outputs]
report_clock_min_period
exit
TCLEOF

for P in 10.0 5.0 2.0; do
  PERIOD=$P /tmp/gcc-shim/sta -no_splash -exit "$OUTDIR/${BLOCK}_sweep.tcl" 2>&1 \
    | grep -E "period_min" | head -1 | xargs -I{} echo "  $BLOCK  P=${P}ns  {}"
done
