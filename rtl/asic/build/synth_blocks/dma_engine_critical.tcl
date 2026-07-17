## Worst-case path extraction for dma_engine on sky130_fd_sc_hd.
##
## Approach: clock at the design's measured period_min (7.49 ns) so the
## reg-to-reg worst path appears at slack ≈ 0 and `report_checks` shows
## full per-cell delay. Then dump the top 3 worst paths in full so we
## can name the actual logic gates and source RTL pins that need
## pipelining.
read_liberty /home/user/.ciel/ciel/sky130/versions/61a056e180dac7dcc6d4eb7529e2231f95105746/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib
read_verilog rtl/asic/build/synth_blocks/dma_engine.synth.v
link_design dma_engine

# Tight clock: 7.5 ns (slightly above the measured 7.49 ns floor).
create_clock -period 7.5 -name clk [get_ports clk]

# Don't constrain IO; we want the *internal* reg-to-reg critical path,
# not the input-delay-dominated one.
set_input_delay  -clock clk 0.0 [all_inputs]
set_output_delay -clock clk 0.0 [all_outputs]

puts "----- top-3 reg-to-reg max paths (full) -----"
report_checks -path_delay max -group_path_count 3 -path_group clk

puts "\n----- single worst path with all cells -----"
report_checks -path_delay max -group_path_count 1 -path_group clk

puts "\n----- summary -----"
report_wns
report_tns
report_clock_min_period
exit
