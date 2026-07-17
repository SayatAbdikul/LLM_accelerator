#!/bin/bash
# Synthesize a given top with yosys' sky130-tuned macro flow (better adder
# mapping + multi-pass retiming + ABC9 vs default ABC).
set -e

TOP="$1"
OUTDIR=rtl/asic/build/synth_blocks
LIB=/home/user/.ciel/ciel/sky130/versions/61a056e180dac7dcc6d4eb7529e2231f95105746/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib

yosys -p "
  read_verilog -sv $OUTDIR/${TOP}.v;
  hierarchy -check -top $TOP;
  synth -top $TOP -flatten;
  dfflibmap -liberty $LIB;
  abc -liberty $LIB -dff -D 5000;
  clean;
  write_verilog $OUTDIR/${TOP}.synth_sky.v;
  stat -liberty $LIB
" > "$OUTDIR/${TOP}.synth_sky.log" 2>&1

CELLS=$(grep "Number of cells:" "$OUTDIR/${TOP}.synth_sky.log" | tail -1 | awk '{print $NF}')
AREA=$(grep "Chip area" "$OUTDIR/${TOP}.synth_sky.log" | tail -1 | awk '{print $NF}')
echo "$TOP  cells=$CELLS  area=${AREA} um^2"
