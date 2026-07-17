read_liberty /home/user/.ciel/ciel/sky130/versions/61a056e180dac7dcc6d4eb7529e2231f95105746/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib
read_verilog rtl/asic/build/synth_blocks/register_file.synth.v
link_design register_file
create_clock -period $::env(PERIOD) -name clk [get_ports clk]
set_input_delay  -clock clk [expr $::env(PERIOD) * 0.3] [all_inputs]
set_output_delay -clock clk [expr $::env(PERIOD) * 0.3] [all_outputs]
set_driving_cell -lib_cell sky130_fd_sc_hd__inv_2 -pin Y [all_inputs]
set_load 0.05 [all_outputs]
report_wns
report_tns
report_clock_min_period
exit
