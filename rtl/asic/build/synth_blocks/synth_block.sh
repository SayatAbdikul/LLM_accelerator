#!/bin/bash
# Synthesize one RTL block standalone against sky130_fd_sc_hd, tightening
# the ABC delay target until it stops shrinking. Reports area + final delay.
#
# usage: synth_block.sh <module_name> <sv2v-extra-files...>
set -e

BLOCK="$1"
shift
EXTRA_SV="$@"

OUTDIR=rtl/asic/build/synth_blocks
LIB=${LIB:?Set LIB to the selected characterized Liberty file}
mkdir -p "$OUTDIR"
PARAMS=""
case "$BLOCK" in
  sfu_engine) PARAMS="chparam -set SFU_SYNTH_MODE 1 sfu_engine;" ;;
  blocking_helper_engine) PARAMS="chparam -set HELPER_SYNTH_MODE 1 blocking_helper_engine;" ;;
  taccel_top) PARAMS="chparam -set SFU_SYNTH_MODE 1 -set HELPER_SYNTH_MODE 1 taccel_top;" ;;
esac

# Pick the .sv file for the module (search common/src and asic/src).
SRC=$(find rtl/common/src rtl/asic/src -name "${BLOCK}.sv" | head -1)
[ -z "$SRC" ] && { echo "no source found for $BLOCK"; exit 1; }

# Pre-process with sv2v (always include the package and any extras).
sv2v -DTARGET_ASIC -DSFU_SYNTH_NO_DPI \
     -I rtl/common/src/include -I rtl/common/src/systolic -I rtl/common/src \
     rtl/common/src/include/taccel_pkg.sv $EXTRA_SV "$SRC" \
     -w "$OUTDIR/${BLOCK}.v"

# Iteratively tighten ABC -D until it stops improving.
PREV_AREA=""
for D in 10000 5000 2000 1000 500 300 200 150 100; do
  yosys -p "
    read_verilog $OUTDIR/${BLOCK}.v;
    $PARAMS
    hierarchy -check -top $BLOCK;
    synth -top $BLOCK -flatten;
    dfflibmap -liberty $LIB;
    abc -liberty $LIB -D $D;
    clean;
    stat -liberty $LIB
  " > "$OUTDIR/${BLOCK}.D${D}.log" 2>&1 || { echo "  yosys failed; see ${BLOCK}.D${D}.log"; break; }
  AREA=$(grep "Chip area" "$OUTDIR/${BLOCK}.D${D}.log" | tail -1 | awk '{print $NF}')
  CELLS=$(grep "Number of cells:" "$OUTDIR/${BLOCK}.D${D}.log" | tail -1 | awk '{print $NF}')
  # Physical um^2 requires matching LEF verification in tools/block_flow.py.
  echo "$BLOCK  D=${D}ps  cells=$CELLS  area=${AREA} unverified_Liberty_units"
done
