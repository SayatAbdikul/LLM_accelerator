#!/bin/bash
# Synthesize a given top module against sky130_fd_sc_hd using the project's
# shared filelist (rtl/common/filelists/core.f). Reports cells + area.
# usage: synth_full.sh <top_module> [<extra_files...>]
set -e

TOP="$1"; shift
OUTDIR=rtl/asic/build/synth_blocks
LIB=/home/user/.ciel/ciel/sky130/versions/61a056e180dac7dcc6d4eb7529e2231f95105746/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib

# Materialize core filelist as absolute paths.
CORE=$(grep -vE '^[[:space:]]*(#|$)' rtl/common/filelists/core.f | sed 's|^|rtl/common/src/|')

# sv2v: all core RTL + any extras (e.g. ASIC wrappers).
sv2v -DTARGET_ASIC -DSFU_SYNTH_NO_DPI -DSFU_SYNTH_MODE=1 -DHELPER_SYNTH_MODE=1 \
     -I rtl/common/src/include -I rtl/common/src/systolic -I rtl/common/src \
     $CORE "$@" -w "$OUTDIR/${TOP}.v"

# Single yosys pass: synth + ABC + report. No tightening sweep (slow on big designs).
yosys -p "
  read_verilog -sv $OUTDIR/${TOP}.v;
  hierarchy -check -top $TOP;
  synth -top $TOP -flatten;
  dfflibmap -liberty $LIB;
  abc -liberty $LIB -D 5000;
  clean;
  stat -liberty $LIB
" > "$OUTDIR/${TOP}.full.log" 2>&1

CELLS=$(grep "Number of cells:" "$OUTDIR/${TOP}.full.log" | tail -1 | awk '{print $NF}')
AREA=$(grep "Chip area" "$OUTDIR/${TOP}.full.log" | tail -1 | awk '{print $NF}')
WIRES=$(grep "Number of wires:" "$OUTDIR/${TOP}.full.log" | tail -1 | awk '{print $NF}')
SEQ_PCT=$(grep "used for sequential" "$OUTDIR/${TOP}.full.log" | tail -1 | grep -oE '\([0-9.]+%\)')
echo "$TOP  cells=$CELLS  wires=$WIRES  area=${AREA} um^2  seq=${SEQ_PCT}"
