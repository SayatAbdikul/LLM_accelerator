#!/bin/bash
# Synthesize a given top module against sky130_fd_sc_hd using the project's
# shared filelist (rtl/common/filelists/core.f). Reports cells + area.
# usage: synth_full.sh <top_module> [<extra_files...>]
set -e

TOP="$1"; shift
OUTDIR=rtl/asic/build/synth_blocks
LIB=${LIB:?Set LIB to the selected characterized Liberty file}
mkdir -p "$OUTDIR"
PARAMS=""
case "$TOP" in
  sfu_engine) PARAMS="chparam -set SFU_SYNTH_MODE 1 sfu_engine;" ;;
  blocking_helper_engine) PARAMS="chparam -set HELPER_SYNTH_MODE 1 blocking_helper_engine;" ;;
  taccel_top) PARAMS="chparam -set SFU_SYNTH_MODE 1 -set HELPER_SYNTH_MODE 1 taccel_top;" ;;
esac

# Materialize core filelist as absolute paths.
CORE=$(grep -vE '^[[:space:]]*(#|$)' rtl/common/filelists/core.f | sed 's|^|rtl/common/src/|')

# sv2v: all core RTL + any extras (e.g. ASIC wrappers).
sv2v -DTARGET_ASIC -DSFU_SYNTH_NO_DPI \
     -I rtl/common/src/include -I rtl/common/src/systolic -I rtl/common/src \
     $CORE "$@" -w "$OUTDIR/${TOP}.v"

# Single yosys pass: synth + ABC + report. No tightening sweep (slow on big designs).
yosys -p "
  read_verilog -sv $OUTDIR/${TOP}.v;
  $PARAMS
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
# Physical um^2 requires matching LEF verification in tools/block_flow.py.
echo "$TOP  cells=$CELLS  wires=$WIRES  area=${AREA} unverified_Liberty_units  seq=${SEQ_PCT}"
