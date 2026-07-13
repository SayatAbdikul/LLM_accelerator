"""Is the corruption GARBAGE, or just low-mantissa byte noise?

The A/B proved shipped != serialized on 99.78% of logits BYTES. But the logits are
fp16, so most bytes flipping could still be low-mantissa noise. And "serialized" was
only shown to DIFFER from shipped — not to be CORRECT.

Close both gaps in the metric this project actually judges 124M by (argmax):

  GOLDEN      mode-0 ISA simulator — models no concurrency at all, so it is the
              correct answer by construction.
  SERIALIZED  mode-1 RTL, SYS_DMA_OVERLAP=0 — cannot drop a write.
  SHIPPED     mode-1 RTL, SYS_DMA_OVERLAP=1 — drops 255,818 drain writes.

If the story holds: GOLDEN == SERIALIZED, and both != SHIPPED.
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/user/LLM_accelerator/software")
sys.path.insert(0, "/home/user/LLM_accelerator/software/tools")
from bench_decode_cycles import build_decode_bins                     # noqa: E402
from taccel.assembler.assembler import ProgramBinary                  # noqa: E402
from taccel.compiler.tiler import pad_dim                             # noqa: E402
from taccel.runtime.calibration import build_calibration_scales       # noqa: E402
from taccel.runtime.host_runner import HostRunner                     # noqa: E402
from taccel.runtime.tiny_fixture import build_stage3_tiny_decoder_bundle  # noqa: E402

ROOT = Path("/home/user/LLM_accelerator")
FIX = ROOT / "software/tests/fixtures/generated/gpt2_converted_nanogpt.pt"
OVL = ROOT / "rtl/verilator/build/run_program_synth/Vtaccel_top"
SER = ROOT / "rtl/verilator/build/run_program_noovl/Vtaccel_top"
BATCH, POS, VOCAB = 16, 510, 50257

print("loading payload ...", flush=True)
payload = torch.load(FIX, map_location="cpu")
scales = build_calibration_scales(payload)
tiny = build_stage3_tiny_decoder_bundle(
    payload, smoke_decode_steps=POS + 1, calibration_scales=scales,
    ptq_preset="weight_only_int8_quarot", batch=BATCH)
bundle = tiny.build.bundle
off, size = int(bundle.decode_logits_offset), int(bundle.logits_size)
pad_v = pad_dim(VOCAB)
elem = size // (BATCH * pad_v)
dtype = {1: np.int8, 2: np.float16, 4: np.float32}[elem]
print(f"logits: size={size:,} batch={BATCH} pad_vocab={pad_v} -> dtype={dtype.__name__}",
      flush=True)

TOK = [(7 * i + 11) % VOCAB for i in range(BATCH)]


def argmax_rows(flat):
    a = np.frombuffer(flat, dtype=dtype) if isinstance(flat, (bytes, bytearray)) \
        else np.asarray(flat, dtype=dtype)
    a = a[:BATCH * pad_v].reshape(BATCH, pad_v)[:, :VOCAB].astype(np.float32)
    a = np.nan_to_num(a, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    return a.argmax(axis=1)


# ---- GOLDEN (mode-0 ISA simulator; no concurrency modelled => correct) ----
print("running GOLDEN (mode-0 simulator) ...", flush=True)
g = HostRunner(bundle, logits_dtype=dtype).run_decode_step_batch(TOK, POS)
g_arg = argmax_rows(np.asarray(g))

# ---- RTL arms ----
td = Path(tempfile.mkdtemp(prefix="p0arg_"))
res = {}
try:
    p = HostRunner(bundle, simulator=None)
    p._patch_embeddings("decode", TOK, [POS] * BATCH)
    p._patch_kv_bases(POS)
    p._patch_decode_attention_context(POS)
    img = bundle.materialize(reset_runtime=False)
    db = int(bundle.data_base)
    dec = bytes(img[int(bundle.decode_instrs_offset):db])
    pb = ProgramBinary(instructions=dec, data=bytes(img[db:]), entry_point=0,
                       insn_count=len(dec) // 8, data_base=db, input_offset=0,
                       pos_embed_patch_dram_offset=0, pos_embed_cls_dram_offset=0,
                       cls_token_dram_offset=0, trace_manifest={}, compiler_manifest={})
    (td / "p.bin").write_bytes(pb.to_bytes())

    for tag, rtl in (("SHIPPED", OVL), ("SERIALIZED", SER)):
        print(f"running {tag} RTL ...", flush=True)
        cp = subprocess.run(
            [str(rtl), "--program", str(td / "p.bin"), "--json-out", str(td / "s.json"),
             "--fast-beats", "--max-cycles", "250000000",
             "--dram-dump-offset", str(off), "--dram-dump-size", str(size),
             "--dram-dump-out", str(td / "d.bin")],
            capture_output=True, text=True, timeout=28800)
        assert (td / "s.json").exists(), cp.stderr[-400:]
        s = json.loads((td / "s.json").read_text())
        assert s["status"] == "halted" and not s["fault"], s
        res[tag] = (argmax_rows((td / "d.bin").read_bytes()),
                    s["port_a"]["sys_lost_drain"])
        (td / "d.bin").unlink()
finally:
    shutil.rmtree(td, ignore_errors=True)

ship, ship_drop = res["SHIPPED"]
ser, ser_drop = res["SERIALIZED"]

print(f"\n=== predicted next token, per stream (124M b16 pos-{POS}) ===")
print(f"{'stream':>7} {'GOLDEN':>8} {'SERIALIZED':>11} {'SHIPPED':>9}   ")
for i in range(BATCH):
    mark = []
    if ser[i] != g[i] if False else ser[i] != g_arg[i]:
        mark.append("ser!=golden")
    if ship[i] != g_arg[i]:
        mark.append("SHIPPED WRONG")
    print(f"{i:>7} {g_arg[i]:>8} {ser[i]:>11} {ship[i]:>9}   {' '.join(mark)}")

ser_ok = int((ser == g_arg).sum())
ship_ok = int((ship == g_arg).sum())
print(f"\n  dropped drain writes:  SHIPPED {ship_drop:,}   SERIALIZED {ser_drop:,}")
print(f"  argmax agreement with GOLDEN (the correct answer):")
print(f"    SERIALIZED : {ser_ok:>2}/{BATCH} streams  ({100*ser_ok/BATCH:.0f}%)")
print(f"    SHIPPED    : {ship_ok:>2}/{BATCH} streams  ({100*ship_ok/BATCH:.0f}%)")

print("\n=== verdict ===")
if ser_ok == BATCH and ship_ok < BATCH:
    print(f"  CONFIRMED. The serialized machine reproduces the golden model EXACTLY")
    print(f"  ({ser_ok}/{BATCH}), and the shipped machine predicts the WRONG TOKEN on")
    print(f"  {BATCH - ship_ok}/{BATCH} streams. The corruption is not mantissa noise —")
    print(f"  the chip emits different text.")
elif ser_ok == BATCH and ship_ok == BATCH:
    print("  The dropped writes change the logits but NOT the argmax. Corruption is")
    print("  real but its quality impact is smaller than the byte-diff suggests —")
    print("  soften the 'computes garbage' claim and gate on perplexity instead.")
else:
    print(f"  UNEXPECTED: serialized agrees with golden on only {ser_ok}/{BATCH}.")
    print("  The serialized arm is NOT simply 'the correct machine' — investigate")
    print("  before making any claim about which build is right.")
