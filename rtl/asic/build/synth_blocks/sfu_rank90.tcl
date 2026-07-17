read_liberty /home/user/.ciel/ciel/sky130/versions/61a056e180dac7dcc6d4eb7529e2231f95105746/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib
read_verilog rtl/asic/build/synth_blocks/sfu_engine.synth_sky.v
link_design sfu_engine
create_clock -period 100.0 -name clk [get_ports clk]
report_checks -path_delay max -group_count 90 -endpoint_count 1 -slack_max 1000 -format end
exit
