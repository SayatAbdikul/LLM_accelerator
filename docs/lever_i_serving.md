# Lever I — serving completeness: logits×N (I-a) and chunked multi-token prefill (I-b)

> **Status note (2026-07-16):** the I-b headline (TTFT **12.38×**, 128-token
> prompt 69.9 s → 5.6 s, byte-exact at any prompt length) is current. The decode
> rates quoted below (9.779/9.729) are pre-Port-A-fix; honest current b16 is
> 11.055. And "the only lever left that moves decode is F (multi-clock)" was
> overtaken twice: F was rejected in favor of single-domain T3, and T1 items 1+2
> moved decode 10.145 → 11.055 with zero RTL (`docs/perf_roadmap_2026-07-16.md`).

Two compiler-only, byte-exact levers. **Neither improves decode tok/s** — I-a makes
batched decode a real N-way serving path, and I-b attacks **TTFT** (time to first
token), which was by far the worst part of the user experience on this chip.

---

## I-a — logits×N (commit `5618577`)

Batched decode already **computed** every stream's logits (`lm_head` is
`M = n_streams` wide) but **stored only row 0**, so a B=16 bundle could not
actually serve 16 sequences. Now it stores all N rows.

- `decoder_bundle._copy_graph_with_logits_store` sets `store_rows = lm_head_shape[0]`
  on the **decode** stream's `logits_store` (prefill keeps its 1-row
  last-position default; b1 ⇒ `store_rows=1` ⇒ byte-identical).
- `tiny_fixture` scales `logits_size *= batch`; `TinyFixtureBundle` carries
  `n_streams`.
- **Landmine fixed:** the logits dtype was inferred as
  `logits_size // pad_dim(vocab)`. Once the region holds N rows that divisor is
  wrong; it must be `// (n_streams * pad_dim(vocab))`.
- `host_runner.run_decode_step_batch` returns the whole region (all N rows,
  contiguous row-major) — `out.reshape(n_streams, -1)` for per-stream logits.

`emit_logits_store` was **already** multi-row. No RTL / ISA / golden change: the
golden model is a pure ISA simulator that executes whatever DMA the compiler emits.

At 124M the 16-row store is 1.6 MB > ABUF, so it takes `emit_logits_store`'s
**chunked staging** path — correct because it copies the lm_head DRAM-temp output
linearly and the row stride is `pad_dim(vocab)`, exactly the store's `cols_pad`.

**Cost: +288,858 cycles = +0.51%** on the b16 step (9.779 → 9.729 tok/s), with
`sys_busy`/`sfu_busy` byte-invariant — precisely the "~free" price predicted.

---

## I-b — chunked multi-token prefill

### The waste

`M_pad = pad_dim(P)`. The prefill was decode-shaped — **one token per pass** — so
every pass occupied a full **16-row systolic m-tile and used one row of it**.
**15/16 of the mesh was thrown away on every prompt token.** That is the entire
lever. A P=16 chunk fills the tile.

It is the same effect that makes batched decode ~6× better per token than b1, and
the same reason B=32 buys nothing: past a multiple of 16 the mesh cost is linear
in m-tiles (see `docs/lever_h_b32.md`), so a **wider chunk buys almost nothing**.

### The right vehicle: a chunk is the DECODE graph with P query rows

The first attempt (`b566733`) built an in-graph **dense** prefill
(`inject_kv_cache_nodes(decode=False, seq_len=P)`). It is byte-exact and it works,
but it is the wrong shape:

- Its `(P, d_model)` embedding tiles pin ABUF — `2·P·768·2 > 128 KB` past
  **P=16** on 124M, so P=32/48/64 simply do not build.
- Its `kv_store` base is a **static relocation**, so the program cannot be re-run
  for a second chunk. A 128-token prompt still decoded 112 tokens one at a time,
  and TTFT barely moved.

The correct form (`67c7f46`) is
`inject_kv_cache_nodes(decode=True, seq_len=decode_key_len)` on a `seq_len=P`
graph — i.e. **the decode graph carrying P query rows instead of 1**:

- it **reads** the KV cache (`kv_load`),
- it **writes its P rows** back at a **runtime-patched** base, and
- crucially it has the **same ABUF profile as a 1-token decode** — every tile is
  `M_pad = pad_dim(16) = 16` either way.

So it re-runs per chunk and walks a prompt of **any** length at **no extra buffer
cost**:

```python
for c in range(0, len(prompt), P):
    logits = runner.run_prefill_chunk(prompt[c:c + P], c)
# `logits` is now the last prompt row's logits == the first generated token
```

The masked softmax keys its triangle off `query_row_base + row_idx`, so each row of
a chunk masks against its **global** position — the causal mask is already correct
for a chunk based anywhere.

### Measured — GPT-2 124M, mode-1 synth RTL, 34.41 MHz

Three RTL runs (P = 16, chunk measured mid-prompt at base 64):

```
1-token prefill              17,910,906 cyc
decode step @pos64           18,809,898 cyc
16-token prefill chunk @64   24,300,687 cyc   <-- 16 tokens for 1.29x the cost of ONE
```

**A 16-token chunk costs only 1.29× what a single token costs.** That is the
16-row m-tile getting *filled* instead of thrown away.

**Per prompt-token: 18,809,898 → 1,518,793 cycles = 12.38×.**

| prompt | sequential | **chunked I-b** | speedup |
|---:|---:|---:|---:|
| 32 tokens | 17.5 s | **1.4 s** | 12.4× |
| **128 tokens** | **69.9 s** | **5.6 s** | **12.38×** |
| 512 tokens | 280 s (4.7 min) | **22.6 s** | 12.4× |

The theoretical ceiling is 16× (perfect tile filling); the shortfall is the SFU,
which genuinely walks `m_exact` rows and so does scale with P, plus DMA.

### Measured (tiny, golden cycles) — the win is INDEPENDENT of prompt length

| prompt L | P | chunks | sequential | chunked | **TTFT** |
|---:|---:|---:|---:|---:|---:|
| 16 | 16 | 1 | 1,444,384 | 349,576 | **4.13×** |
| 64 | 16 | 4 | 5,782,576 | 1,398,304 | **4.14×** |
| 96 | 16 | 6 | 8,674,704 | 2,097,456 | **4.14×** |
| 64 | 32 | 2 | 5,782,576 | 1,365,932 | 4.23× |

The tiny model shows a smaller multiple (4.1×) than 124M (12.4×) simply because its
dimensions are small enough that the fixed per-pass overheads, not the systolic
mesh, dominate. 124M is the number that matters.

**P=16 captures essentially the whole win** (4.14 vs 4.23 at P=32 on tiny) — and it
is also the ABUF-safe choice on 124M (a wider in-graph chunk blows ABUF). Use P=16.

### Correctness is exact, not approximate

Chunked prefill produces **byte-identical** logits to the sequential path
`prefill(t0) + decode(t1..t_{L-1})`, at every prompt length — and the tokens
generated afterwards match, proving the chunks wrote the KV cache correctly and
decode continues from it. This must hold: matmuls are row-independent, LN/softmax
are per-row, and the KV quant scales are static. Anything else is a bug.

### Two real compiler bugs this exposed

Both come from **M_pad being a property of the CONSUMER node**, not of the weight:

1. **Stage-4 staging** keyed only on the weight's own size (`> WBUF`), but the
   **emitter** also tiles when `output`/`accum` exceed ABUF/ACCUM — and those
   scale with `M_pad`. A wide-M graph therefore tiled weights that staging never
   staged (`KeyError` on the tile symbol). Fixed with `stage4_forced_weights`,
   which mirrors the emitter's dispatch; `decoder_bundle` unions it across **both**
   streams so their data blobs stay identical.
2. **The Stage-4 tile plan** sized N-tiles from a hardcoded 16-row strip
   (`STAGE4_M_TILE`), under-counting ACCUM for `M_pad > 16`: it fit at `M_pad=32`
   only by luck (`32·512·4` = exactly `ACCUM_SIZE`) and overflowed 2× at 64. The
   plan now takes the bundle-wide max `M_pad`. Splitting an N-tile is
   **systolic-cost-neutral** (`mt·nt·(64+130+17(kt−1))` — halving `n_len` halves
   `nt`), so this is ~free. Default 16 ⇒ every pre-I-b bundle is byte-identical.
   *(This is also the fix B=64 would have needed.)*

### The plan's "shared-data-blob split" was unnecessary

Verified empirically: the multi-token prefill graph and the 1-token decode graph
produce **byte-identical** data blobs across every decode depth (N_pad
16/32/64/128). That constraint bites only the **batched** graph, which carries
per-stream scale vectors — hence `prefill_tokens > 1` requires `batch=1`.

### Gate

- `test_multi_token_prefill` **8/8**, incl. the RTL byte-match (mode-1 synth RTL ==
  mode-0 golden on the chunked prefill program).
- `test_batched_decode` still **12/12** (b16 + b32 RTL byte-match unchanged by the
  shared tile-plan / staging edits).
- 48 passed across the neighbouring suites, **zero new failures**.

---

## Where this leaves the chip

- **TTFT: 12.38× better on 124M**, at any prompt length. A 128-token prompt goes
  from **69.9 s → 5.6 s**. Before this lever, TTFT was by a wide margin the worst
  part of the user experience — a minute-plus of dead air before a chat response
  began streaming at ~10 tok/s. It is no longer the bottleneck.
- **Decode throughput** is untouched by lever I and is systolic-**compute**-bound.
  The only lever left that moves it is **F (multi-clock SFU island)**, which
  attacks the *clock* on that floor (~9.7 → ~21 tok/s).

End-to-end for a 128-token prompt + 100 generated tokens, at 34.41 MHz:

| | before | after lever I |
|---|---:|---:|
| TTFT | 69.9 s | **5.6 s** |
| generation (100 tok @ 9.73 tok/s) | 10.3 s | 10.3 s |
| **total** | **80.2 s** | **15.9 s** |

(At the post-lever-E fmax ≈ 37.3 MHz both columns scale by ~1.083.)
