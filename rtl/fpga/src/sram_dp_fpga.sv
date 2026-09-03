// Reserved for future FPGA-specific SRAM wrappers (explicit vendor BRAM
// macros, URAM, asymmetric-port memories, etc.) if explicit IP binding is
// later preferred over the inferred BRAM body.
//
// The current FPGA target uses the shared `sram_dp_inferred` body via the
// target-dispatch wrapper in rtl/common/src/memory/sram_dp.sv. Whether that
// body maps successfully to device BRAM/URAM, with matching port semantics,
// has not been validated because no device or vendor flow is selected.
//
// Placeholder: no modules defined here. If modules are added later, also
// add them to rtl/fpga/Makefile's FPGA_SV source list.
