# Minimal OpenROAD PNR + STA flow for blocking_helper_engine on sky130_fd_sc_hd.
# Goal: post-placement fmax measurement that includes real fanout buffering.

set PDK_ROOT /home/user/.ciel/ciel/sky130/versions/61a056e180dac7dcc6d4eb7529e2231f95105746/sky130A/libs.ref/sky130_fd_sc_hd
set TECH_LEF $PDK_ROOT/techlef/sky130_fd_sc_hd__nom.tlef
set CELL_LEF $PDK_ROOT/lef/sky130_fd_sc_hd.lef
set LIB      $PDK_ROOT/lib/sky130_fd_sc_hd__tt_025C_1v80.lib
set NETLIST  /home/user/LLM_accelerator/rtl/asic/build/synth_blocks/blocking_helper_engine.synth_sky.v
set TOP      blocking_helper_engine

# --- Tech setup ---
read_liberty $LIB
read_lef $TECH_LEF
read_lef $CELL_LEF
read_verilog $NETLIST
link_design $TOP

# --- Constraints ---
create_clock -period 6.0 -name clk [get_ports clk]
set_input_delay  -clock clk 0.0 [all_inputs]
set_output_delay -clock clk 0.0 [all_outputs]

# --- Wire RC up-front so repair_design and CTS see it ---
set_wire_rc -clock  -resistance 3.535e-04 -capacitance 7.69e-05
set_wire_rc -signal -resistance 1.68e-04  -capacitance 2.16e-04

# --- Floorplan ---
# Target ~50% utilization; aspect ratio 1.0.
initialize_floorplan -utilization 20 -aspect_ratio 1.0 -core_space 8.0 -site unithd

# Routing tracks (sky130 hd: from tracks.info)
make_tracks li1  -x_offset 0.23 -x_pitch 0.46 -y_offset 0.17 -y_pitch 0.34
make_tracks met1 -x_offset 0.17 -x_pitch 0.34 -y_offset 0.17 -y_pitch 0.34
make_tracks met2 -x_offset 0.23 -x_pitch 0.46 -y_offset 0.23 -y_pitch 0.46
make_tracks met3 -x_offset 0.34 -x_pitch 0.68 -y_offset 0.34 -y_pitch 0.68
make_tracks met4 -x_offset 0.46 -x_pitch 0.92 -y_offset 0.46 -y_pitch 0.92
make_tracks met5 -x_offset 1.70 -x_pitch 3.40 -y_offset 1.70 -y_pitch 3.40

# Place IOs
place_pins -hor_layers met3 -ver_layers met2 -corner_avoidance 0

# --- Power: declare global power/ground nets, skip full PDN (placement-only flow) ---
set block [ord::get_db_block]
odb::dbNet_create $block VPWR
odb::dbNet_create $block VGND
[$block findNet VPWR] setSpecial
[$block findNet VPWR] setSigType POWER
[$block findNet VGND] setSpecial
[$block findNet VGND] setSigType GROUND
add_global_connection -net VPWR -inst_pattern .* -pin_pattern VPWR -power
add_global_connection -net VGND -inst_pattern .* -pin_pattern VGND -ground
add_global_connection -net VPWR -inst_pattern .* -pin_pattern VPB -power
add_global_connection -net VGND -inst_pattern .* -pin_pattern VNB -ground
global_connect

# --- Tap cells ---
tapcell -distance 14 -tapcell_master sky130_fd_sc_hd__tapvpwrvgnd_1

# --- Placement ---
global_placement -density 0.6 -pad_left 1 -pad_right 1

# --- Repair after global placement (fanout, slew) ---
estimate_parasitics -placement
repair_design

# --- Clock tree synthesis ---
# Use a tree-friendly buffer
clock_tree_synthesis -buf_list sky130_fd_sc_hd__clkbuf_4 \
                     -root_buf sky130_fd_sc_hd__clkbuf_16 \
                     -sink_clustering_enable

# --- Detailed placement ---
detailed_placement

# --- Re-estimate after placement ---
estimate_parasitics -placement

# --- Report STA ---
puts "===== POST-PNR STA (placement-aware) ====="
report_clock_min_period
puts "\n===== worst path ====="
report_checks -path_delay max -endpoint_count 1 -unconstrained
puts "\n===== summary ====="
report_wns
report_tns
report_design_area

exit
