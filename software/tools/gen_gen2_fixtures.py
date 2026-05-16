"""Anti-drift golden fixture generator for the gen-2 FP32 opcodes.

Drives the FROZEN golden model (`taccel.golden_model.simulator._exec_*_fp32`)
directly on seeded inputs and dumps (input, scale_regs_pre, expected_output,
scale_regs_post) as raw little-endian bytes + a meta.json under
`rtl/verilator/fixtures/gen2/<op>/<case>/`.

The RTL per-op tests (P2-P5) LOAD and byte-compare these — the reference IS
the golden model, so the eps / rounding / clamp drift bug class (e.g. the
gen-1 LN_EPS 1e-6 vs golden 1e-5) is impossible by construction.

Pinned to `frozen_golden_sha` (the committed simulator.py baseline,
isa_generation_freeze.md §4.6). Re-run on any freeze revision.

Run from repo root:  PYTHONPATH=software python3 software/tools/gen_gen2_fixtures.py
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from taccel.golden_model import memory as mem
from taccel.golden_model.simulator import Simulator
from taccel.golden_model.state import MachineState
from taccel.isa.opcodes import BUF_ABUF, BUF_WBUF, BUF_ACCUM
from taccel.isa.instructions import (
    DequantAccumFp32Insn, QuantFp32Int8Insn, VaddFp32Insn, LayernormFp32Insn,
    GeluFp32Insn, MaskedSoftmaxFp32Insn, DequantAccumFp32ScaledInsn,
    MaxAbsReduceFp32Insn,
)

OUT = Path("rtl/verilator/fixtures/gen2")
M, N = 16, 64                       # 1 M-tile, 4 N-tiles
TILE_CONFIG = (M // 16 - 1, N // 16 - 1, 0)
# 16-byte-unit offsets, non-overlapping (fp16 tile = M*N*2 = 2048 B = 128 u).
OFF_SRC1, OFF_SRC2, OFF_DST = 0, 512, 1024


def _sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "UNKNOWN"


def _fresh():
    st = MachineState()
    st.tile_config = TILE_CONFIG
    return Simulator(st), st


def _dump(op: str, case: str, *, meta: dict, src1: bytes,
          src2: bytes | None, sregs_pre: np.ndarray, expected: bytes,
          sregs_post: np.ndarray) -> None:
    d = OUT / op / case
    d.mkdir(parents=True, exist_ok=True)
    (d / "input_src1.raw").write_bytes(src1)
    if src2 is not None:
        (d / "input_src2.raw").write_bytes(src2)
    (d / "scale_regs_pre.raw").write_bytes(
        sregs_pre.astype("<f2").tobytes())
    (d / "expected_out.raw").write_bytes(expected)
    (d / "scale_regs_post.raw").write_bytes(
        sregs_post.astype("<f2").tobytes())
    (d / "meta.json").write_text(json.dumps(meta, indent=2))


def _abuf(st, off_u, n_bytes):  # raw stored bytes at a 16-byte-unit offset
    b = off_u * mem.UNIT
    return bytes(st.abuf[b:b + n_bytes])


def gen(seed: int = 20260516) -> int:
    rng = np.random.default_rng(seed)
    OUT.mkdir(parents=True, exist_ok=True)
    sha = _sha()
    base = dict(M=M, N=N, src1_off=OFF_SRC1, src2_off=OFF_SRC2,
                dst_off=OFF_DST, frozen_golden_sha=sha)

    # ---- elementwise FP16 ABUF->ABUF: 0x19 VADD, 0x1A LN, 0x1B GELU ----
    def ew(op, opname, InsnCls, mk_inputs, extra_meta=None, case="std"):
        sim, st = _fresh()
        mk_inputs(st)
        pre = st.scale_regs.copy()
        insn = InsnCls(src1_buf=BUF_ABUF, src1_off=OFF_SRC1,
                       src2_buf=BUF_WBUF, src2_off=OFF_SRC2,
                       dst_buf=BUF_ABUF, dst_off=OFF_DST, sreg=0, flags=1)
        getattr(sim, op)(insn)
        meta = dict(base, op=opname, opcode=int(insn.opcode), sreg=0, flags=1,
                    src1_dtype="fp16", dst_dtype="fp16")
        if extra_meta:
            meta.update(extra_meta)
        _dump(opname, case, meta=meta,
              src1=_abuf(st, OFF_SRC1, M * N * 2),
              src2=(bytes(st.wbuf[OFF_SRC2 * mem.UNIT:
                                  OFF_SRC2 * mem.UNIT + 2 * N * 2])
                    if opname == "layernorm_fp32" else None),
              sregs_pre=pre, expected=_abuf(st, OFF_DST, M * N * 2),
              sregs_post=st.scale_regs.copy())

    def seed_src1(st, lo=-4.0, hi=4.0):
        x = rng.uniform(lo, hi, (M, N)).astype(np.float32)
        mem.write_fp16_tile(st, BUF_ABUF, OFF_SRC1, x)

    # 0x19 VADD_FP32: src1 + src2 (both ABUF fp16)
    sim, st = _fresh()
    a = rng.uniform(-3, 3, (M, N)).astype(np.float32)
    b = rng.uniform(-3, 3, (M, N)).astype(np.float32)
    mem.write_fp16_tile(st, BUF_ABUF, OFF_SRC1, a)
    mem.write_fp16_tile(st, BUF_ABUF, OFF_SRC2, b)
    pre = st.scale_regs.copy()
    insn = VaddFp32Insn(src1_buf=BUF_ABUF, src1_off=OFF_SRC1,
                        src2_buf=BUF_ABUF, src2_off=OFF_SRC2,
                        dst_buf=BUF_ABUF, dst_off=OFF_DST, sreg=0, flags=1)
    sim._exec_vadd_fp32(insn)
    _dump("vadd_fp32", "std",
          meta=dict(base, op="vadd_fp32", opcode=int(insn.opcode), sreg=0,
                    flags=1, src1_dtype="fp16", src2_dtype="fp16",
                    dst_dtype="fp16"),
          src1=_abuf(st, OFF_SRC1, M * N * 2),
          src2=_abuf(st, OFF_SRC2, M * N * 2),
          sregs_pre=pre, expected=_abuf(st, OFF_DST, M * N * 2),
          sregs_post=st.scale_regs.copy())

    # 0x1A LAYERNORM_FP32: src2 = 2N fp16 (gamma||beta) in WBUF; eps=1e-5
    def ln_inputs(st):
        seed_src1(st)
        gb = np.concatenate([
            rng.uniform(0.5, 1.5, N).astype(np.float32),   # gamma
            rng.uniform(-0.5, 0.5, N).astype(np.float32),  # beta
        ])
        mem.write_fp16_vector(st, BUF_WBUF, OFF_SRC2, gb)
    ew("_exec_layernorm_fp32", "layernorm_fp32", LayernormFp32Insn,
       ln_inputs, extra_meta=dict(eps=1e-5, src2_dtype="fp16_2N_gamma_beta"))

    # 0x1B GELU_FP32 (tanh gelu_new)
    ew("_exec_gelu_fp32", "gelu_fp32", GeluFp32Insn,
       lambda st: seed_src1(st, -6, 6))

    # ---- 0x17 DEQUANT_ACCUM_FP32: ACCUM int32 x N fp16 col-scales ----
    sim, st = _fresh()
    acc = rng.integers(-(1 << 20), 1 << 20, (M, N)).astype(np.int32)
    mem.write_int32_tile(st, BUF_ACCUM, 0, acc)
    cols = rng.uniform(1e-3, 5e-2, N).astype(np.float32)
    mem.write_fp16_vector(st, BUF_WBUF, OFF_SRC2, cols)
    pre = st.scale_regs.copy()
    insn = DequantAccumFp32Insn(src1_buf=BUF_ACCUM, src1_off=0,
                                src2_buf=BUF_WBUF, src2_off=OFF_SRC2,
                                dst_buf=BUF_ABUF, dst_off=OFF_DST,
                                sreg=0, flags=1)
    sim._exec_dequant_accum_fp32(insn)
    _dump("dequant_accum_fp32", "std",
          meta=dict(base, op="dequant_accum_fp32", opcode=int(insn.opcode),
                    sreg=0, flags=1, src1_off=0, src1_dtype="int32_accum",
                    src2_dtype="fp16_Ncolscales", dst_dtype="fp16"),
          src1=acc.astype("<i4").tobytes(),
          src2=cols.astype("<f2").tobytes(),
          sregs_pre=pre, expected=_abuf(st, OFF_DST, M * N * 2),
          sregs_post=st.scale_regs.copy())

    # ---- 0x18 QUANT_FP32_INT8 (+ round-half-even edge case) ----
    for case, x in (
        ("std", rng.uniform(-2, 2, (M, N)).astype(np.float32)),
        ("round_half_even",
         (np.arange(M * N).reshape(M, N).astype(np.float32) - 0.5)),
    ):
        sim, st = _fresh()
        mem.write_fp16_tile(st, BUF_ABUF, OFF_SRC1, x)
        st.scale_regs[3] = np.float16(1.0 if case == "round_half_even"
                                      else 37.0)
        pre = st.scale_regs.copy()
        insn = QuantFp32Int8Insn(src1_buf=BUF_ABUF, src1_off=OFF_SRC1,
                                 src2_buf=BUF_ABUF, src2_off=0,
                                 dst_buf=BUF_ABUF, dst_off=OFF_DST,
                                 sreg=3, flags=1)
        sim._exec_quant_fp32_int8(insn)
        _dump("quant_fp32_int8", case,
              meta=dict(base, op="quant_fp32_int8", opcode=int(insn.opcode),
                        sreg=3, flags=1, src1_dtype="fp16",
                        dst_dtype="int8", note="round-half-to-even"),
              src1=_abuf(st, OFF_SRC1, M * N * 2), src2=None,
              sregs_pre=pre, expected=_abuf(st, OFF_DST, M * N),
              sregs_post=st.scale_regs.copy())

    # ---- 0x1D MASKED_SOFTMAX_FP32 (causal; CONFIG_ATTN ctx) ----
    sim, st = _fresh()
    st.attn_context = {"query_row_base": 0, "valid_kv_len": N,
                       "mode": 1}
    x = rng.uniform(-8, 8, (M, N)).astype(np.float32)
    mem.write_fp16_tile(st, BUF_ABUF, OFF_SRC1, x)
    pre = st.scale_regs.copy()
    insn = MaskedSoftmaxFp32Insn(src1_buf=BUF_ABUF, src1_off=OFF_SRC1,
                                 src2_buf=BUF_ABUF, src2_off=0,
                                 dst_buf=BUF_ABUF, dst_off=OFF_DST,
                                 sreg=0, flags=1)
    sim._exec_masked_softmax_fp32(insn)
    _dump("masked_softmax_fp32", "std",
          meta=dict(base, op="masked_softmax_fp32", opcode=int(insn.opcode),
                    sreg=0, flags=1, src1_dtype="fp16", dst_dtype="fp16",
                    attn_query_row_base=0, attn_valid_kv_len=N),
          src1=_abuf(st, OFF_SRC1, M * N * 2), src2=None,
          sregs_pre=pre, expected=_abuf(st, OFF_DST, M * N * 2),
          sregs_post=st.scale_regs.copy())

    # ---- 0x1F MAX_ABS_REDUCE_FP32 (std + all-zero clamp edge) ----
    for case, x in (
        ("std", rng.uniform(-9, 9, (M, N)).astype(np.float32)),
        ("all_zero", np.zeros((M, N), np.float32)),
    ):
        sim, st = _fresh()
        mem.write_fp16_tile(st, BUF_ABUF, OFF_SRC1, x)
        pre = st.scale_regs.copy()
        insn = MaxAbsReduceFp32Insn(src1_buf=BUF_ABUF, src1_off=OFF_SRC1,
                                    src2_buf=BUF_ABUF, src2_off=0,
                                    dst_buf=BUF_ABUF, dst_off=0,
                                    sreg=5, flags=1)
        sim._exec_max_abs_reduce_fp32(insn)
        _dump("max_abs_reduce_fp32", case,
              meta=dict(base, op="max_abs_reduce_fp32",
                        opcode=int(insn.opcode), sreg=5, flags=1,
                        src1_dtype="fp16",
                        out="scale_regs[5]=127/maxabs, [6]=maxabs/127",
                        note="MAX_ABS_EPS=2**-9 clamp"),
              src1=_abuf(st, OFF_SRC1, M * N * 2), src2=None,
              sregs_pre=pre,
              expected=st.scale_regs.astype("<f2").tobytes(),
              sregs_post=st.scale_regs.copy())

    # ---- 0x1E DEQUANT_ACCUM_FP32_SCALED: int32 x wt_scale x act + bias ----
    sim, st = _fresh()
    acc = rng.integers(-(1 << 18), 1 << 18, (M, N)).astype(np.int32)
    mem.write_int32_tile(st, BUF_ACCUM, 0, acc)
    sb = np.concatenate([
        rng.uniform(1e-3, 3e-2, N).astype(np.float32),   # wt scales
        rng.uniform(-0.2, 0.2, N).astype(np.float32),     # bias
    ])
    mem.write_fp16_vector(st, BUF_WBUF, OFF_SRC2, sb)
    st.scale_regs[7] = np.float16(0.0123)                  # act scale
    pre = st.scale_regs.copy()
    insn = DequantAccumFp32ScaledInsn(src1_buf=BUF_ACCUM, src1_off=0,
                                      src2_buf=BUF_WBUF, src2_off=OFF_SRC2,
                                      dst_buf=BUF_ABUF, dst_off=OFF_DST,
                                      sreg=7, flags=1)
    sim._exec_dequant_accum_fp32_scaled(insn)
    _dump("dequant_accum_fp32_scaled", "std",
          meta=dict(base, op="dequant_accum_fp32_scaled",
                    opcode=int(insn.opcode), sreg=7, flags=1, src1_off=0,
                    src1_dtype="int32_accum", src2_dtype="fp16_2N_scale_bias",
                    dst_dtype="fp16"),
          src1=acc.astype("<i4").tobytes(),
          src2=sb.astype("<f2").tobytes(),
          sregs_pre=pre, expected=_abuf(st, OFF_DST, M * N * 2),
          sregs_post=st.scale_regs.copy())

    n = sum(1 for _ in OUT.rglob("meta.json"))
    print(f"wrote {n} gen-2 fixtures under {OUT}  (frozen_golden_sha={sha})")
    return 0


if __name__ == "__main__":
    raise SystemExit(gen())
