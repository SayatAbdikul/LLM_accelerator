# SFU storage and banking inventory

**Audited:** 2026-10-11  
**Scope:** active generation-2 instructions with `SFU_SYNTH_MODE=1`, FP16
storage flags, and the existing numerical operation order. This is an
implementation inventory and experiment plan, not a measured area reduction.

The [SFU](../rtl/common/src/sfu_engine.sv) declares 19 KiB of scratch storage:
four 1,024-element FP32 arrays, a 1,024-byte INT8 output array, and a
1,024-element FP16 output array. That is 155,648 declared bits, excluding
pipeline registers and scalar state. It is neither a mapped-flop count nor a
macro-area estimate. All six arrays currently receive asynchronous reset, and
variable-index readers form combinational muxes. These features must change
deliberately before assuming SRAM inference.

## Storage ownership and lifetime

| Array | Declared capacity | Live data and lifetime | Largest consumed access group |
|---|---:|---|---|
| `row_data_q` | 4 KiB | One source row; persists through all LayerNorm/softmax passes. Elementwise consumers trail a streamed load. Dequant stores converted INT32 inputs. | Up to 8 FP32 writes from FP16 load and 8 reads in a compute cycle; dequant loads 4 values/cycle |
| `attn_accum_q` | 4 KiB | VADD's second source row, or DEQUANT's per-column scale row. VADD load and compute overlap; scales precede dequant input loading. | Up to 8 writes and 8 reads/cycle |
| `gamma_q` | 4 KiB | LayerNorm gamma or scaled-dequant weight scales. Parameter capture precedes use; preserve values until the last matching collect/feed. | Load 8 values/cycle; scaled dequant consumes 8/cycle, LayerNorm consumes 1/cycle |
| `beta_q` | 4 KiB | LayerNorm beta or scaled-dequant bias. Bias must stay aligned through delayed arithmetic stages. | Same groups as gamma |
| `out_bytes_q` | 1 KiB | Quantized outputs between compute and packed writeback. Compute produces 8 values; writeback gathers a 16-value SRAM beat. | 8 writes and 16 reads in one cycle |
| `out_h_q` | 2 KiB | FP16 results between collect and writeback. Elementwise paths collect 8 values/cycle; reductions produce 1. | Up to 8 writes and 8 reads/cycle |

“Consumed” distinguishes a selected instruction's functional port demand from
all combinational expressions that happen to be elaborated. Parameter storage
must not be reclaimed based only on the main feed pointer: LayerNorm uses
`ln_coll_q`, scaled dequant delays beta, and GELU uses a delayed collect pointer.
Different arrays are live concurrently, even though instructions are serialized.

## Per-op demand

| Opcode | Read/storage phases | Concurrent activity and constraints |
|---|---|---|
| `0x17 DEQUANT_ACCUM_FP32` | Load scales into `attn_accum_q`, then stream INT32 source into `row_data_q`; compute FP32 and collect FP16 | 4 source values/cycle limit an 8-wide feed to available pairs of chunks; scale bits must survive the full row/instruction use |
| `0x18 QUANT_FP32_INT8` | Stream FP16 source into `row_data_q`; consume 8; pack 16 INT8 outputs | Source write and read overlap, as do output write and pack. A simple single-port array cannot support this schedule |
| `0x19 VADD_FP32` | Load source 1, then stream source 2 into `attn_accum_q` while adding and writing | Source 1 remains live until its corresponding add; source 2 needs concurrent capture/read |
| `0x1A LAYERNORM_FP32` | Load row and gamma/beta; reread row for sum, variance and normalization; collect output | Scalar ordered reductions; row cannot be discarded after its first pass. Gamma/beta collection address lags the row feed |
| `0x1B GELU_FP32` | Stream/load 8 inputs, feed 8 pipelines, collect 8 outputs after the fixed pipeline delay | Reclamation must follow consumed input/collected output validity, not elapsed wall-clock cycles or the issue pointer |
| `0x1D MASKED_SOFTMAX_FP32` | Load row; max, exp-sum and normalized-output passes, each honoring visibility | Scalar reads; retain row across repeated exp evaluation. Preserve reduction order and causal mask handling |
| `0x1E DEQUANT_ACCUM_FP32_SCALED` | Load weight scales/bias into gamma/beta, stream INT32 source, delayed FP16 collection | Up to 8 reads from row/gamma/beta each feed; retain side operands at the matching pipeline stages |
| `0x1F MAX_ABS_REDUCE_FP32` | Reduce 8 converted FP16 values directly from the SRAM return into a scalar max | Does not require a full source-row scratch array for the running reduction; outputs two scale-register values |

Source references: [streaming engines](../rtl/common/src/sfu_engine.sv),
[reduction sequencers](../rtl/common/src/sfu_g2_compute.svh), and
[parallel/collected operand readers](../rtl/common/src/sfu_synth_datapath.svh).

## Candidate implementation sequence

1. Make one array's read requests and return validity explicit. Eight
   consecutive element accesses map naturally to eight banks by `index % 8`,
   with bank row `index / 8`. Distinct load/feed positions require one write
   and one read per active bank. Register the SRAM response before arithmetic
   and update valid/collect alignment together.
2. Handle `out_bytes_q` separately. Eight banks with one read each cannot
   gather 16 bytes in one cycle. Options include 16 banks, wider packed words
   with byte write enables, or a small packing queue. Preserve any in-flight
   output chunks when writes and packing overlap.
3. Replace asynchronous bulk clears with bounded validity only after proving
   that every consumed element was loaded or produced. Cover short/masked
   rows, fault interruption, reset mid-operation, and reuse after different
   opcodes. Do not rely on Verilator's initial zero values.
4. Explore sharing `attn_accum_q` with gamma storage: their active consumers
   are mutually exclusive across these instructions. Keep beta separate where
   scaled dequant/LayerNorm need gamma and beta together. Prove lifetimes and
   parameter reload behavior before merging storage.
5. Evaluate streaming queues for elementwise output storage; retain row-sized
   input storage for LayerNorm/softmax's repeated passes. Confirm actual macro
   utilization: a bank smaller than a chosen physical macro still pays for
   the whole macro and its decoder/output logic.

Acceptance requires mode-1 per-op conformance, RTL-to-RTL preservation for the
storage change, full-step cycle reports, and real mapped bank/peripheral
statistics. Removing declared bits alone does not establish area, energy, or
throughput improvement.

## Separate main-memory macro experiment

The experimental [SKY130 bank adapter](../rtl/asic/src/sram_dp_sky130_banked.sv)
uses four 32-bit width lanes per logical 128-bit row and 512-row depth banks.
The representative 128×1,024 configuration uses eight
`sky130_sram_2kbyte_1rw1r_32x512_8` macros, or 16 KiB logical capacity.
It is not connected to the core/default ASIC wrapper.

It implements Port-A write-first bypass and retains outputs while ports are
disabled, because the upstream behavioral macro drives disabled outputs to X.
The upstream model captures requests on a rising edge and produces reads after
the falling edge plus its model delay; the test checks settling before the
following rising edge. Those arbitrary model delays are not physical timing.

The macro does not guarantee old Port-B data during a same-row Port-A write.
The adapter now preserves that logical behavior with a one-entry delayed-write
overlay. Each A write commits the previous entry and stages the new write;
A reads defer commit, while an idle A cycle commits and empties the entry.
Reads matching the pending row forward its pre-edge value, including B during
a simultaneous replacement write. The corresponding physical B read is disabled,
so undefined macro write/read collisions are never requested. Queue size remains
one because an A read cannot simultaneously add another write.

There is no reset or power-up initialization requirement beyond the inferred
memory's existing contract: contents and outputs are initially unspecified.
Arbitrary physical contents plus an arbitrary pending entry represent arbitrary
logical initial contents. Known values must be initialized through logical A
writes; external macro preloading alone is not a supported initialization method.
The first write establishes its logical row without a priming idle cycle, and
disabled outputs retain their last logical result.

Measured functional result: **96,804 cycle comparisons** against the actual
`sram_dp_inferred` pass with the unchanged pinned upstream model. The
[test](../rtl/asic/tests/test_sram_dp_sky130_banked.sv) runs eight geometry/startup
cases across 32×512, 64×1,024, 128×1,024 and 128×2,048, including 80,000
deterministic randomized cycles and 2,518 logical same-row write/read collisions.
It covers uninitialized/invalid/valid pending startup, first read/write/idle,
consecutive same-row writes, long dual-read deferral, holds, bank/lane boundaries,
and exhaustive final readback. Independent checks of actual macro pins observe
zero physical collisions. Negative-control copies with incorrect new-data B
forwarding or physical B collision suppression removed both fail as expected.
This is tested functional equivalence, not a formal or timing proof. Physical
area/timing closure, full-chip integration and DRC/LVS remain separate gates;
the experimental adapter is not promoted to the default SRAM binding.
