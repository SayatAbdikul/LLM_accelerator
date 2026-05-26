// Reserved for future FPGA-specific SRAM wrappers (explicit vendor BRAM
// macros, URAM, asymmetric-port memories, etc.) if explicit IP binding is
// later preferred over the inferred BRAM body.
//
// As of Step D (2026-05-26 RTL restructure), the FPGA target uses the
// shared `sram_dp_inferred` body via the target-dispatch wrapper in
// rtl/common/src/memory/sram_dp.sv (default branch when TARGET_ASIC is
// not defined). Vivado/Quartus consume the `(* ram_style = "block" *)`
// attribute on the inferred `mem` array — no explicit macro instantiation
// is needed for typical block-RAM mapping.
//
// Placeholder: no modules defined here. If modules are added later, also
// add them to rtl/fpga/Makefile's FPGA_SV source list.
