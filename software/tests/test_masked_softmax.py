import numpy as np
import pytest

from taccel.assembler import Assembler
from taccel.golden_model import Simulator
from taccel.golden_model.simulator import ConfigError
from taccel.golden_model.state import MachineState
from taccel.golden_model.sfu import execute_masked_softmax_attnv
from taccel.isa.instructions import MaskedSoftmaxAttnVInsn
from taccel.isa.opcodes import BUF_ACCUM, BUF_ABUF, BUF_WBUF


def _fp16_bits(value: float) -> int:
    fp16 = np.float16(value)
    return int(np.frombuffer(fp16.tobytes(), dtype=np.uint16)[0])


def _pad16(value: int) -> int:
    return ((value + 15) // 16) * 16


def _reference_masked_softmax(logits_i32, in_scale, out_scale, query_row_base, valid_kv_len, mode):
    in_scale = np.float32(np.float16(in_scale))
    out_scale = np.float32(np.float16(out_scale))
    x = logits_i32.astype(np.float32) * in_scale
    M, N = x.shape
    rows = np.arange(M, dtype=np.int32).reshape(M, 1)
    cols = np.arange(N, dtype=np.int32).reshape(1, N)
    visible = np.ones((M, N), dtype=bool)
    if mode & 0b10:
        visible &= cols <= (query_row_base + rows)
    if mode & 0b01:
        visible &= cols < valid_kv_len
    masked = np.where(visible, x, np.float32(-np.inf)).astype(np.float32)
    shifted = masked - masked.max(axis=-1, keepdims=True)
    exp_x = np.where(visible, np.exp(shifted).astype(np.float32), np.float32(0.0))
    probs = exp_x / exp_x.sum(axis=-1, keepdims=True)
    return np.clip(np.round(probs / out_scale), -128, 127).astype(np.int8)


def _run_masked_softmax(seq_len: int, mode: int, query_row_base: int = 0):
    key_cols = _pad16(seq_len)
    in_scale = 0.125
    out_scale = 1.0 / 127.0
    prog = Assembler().assemble(
        f"CONFIG_TILE M=1, N={key_cols // 16}, K=1\n"
        f"CONFIG_ATTN query_row_base={query_row_base}, valid_kv_len={seq_len}, mode=0b{mode:02b}\n"
        f"SET_SCALE S0, imm=0x{_fp16_bits(in_scale):04x}\n"
        f"SET_SCALE S1, imm=0x{_fp16_bits(out_scale):04x}\n"
        "MASKED_SOFTMAX src1=ACCUM[0], src2=ABUF[0], dst=ABUF[0], sreg=0, flags=0\n"
        "HALT"
    )
    sim = Simulator()
    sim.load_program(prog)
    logits = (np.arange(16 * key_cols, dtype=np.int32).reshape(16, key_cols) % 31) - 15
    sim.state.accum[:logits.size] = logits.reshape(-1)
    sim.run()
    got = np.frombuffer(bytes(sim.state.abuf[:logits.size]), dtype=np.int8).reshape(16, key_cols)
    expected = _reference_masked_softmax(logits, in_scale, out_scale, query_row_base, seq_len, mode)
    return got, expected


@pytest.mark.parametrize("seq_len", [7, 16, 17, 31, 64, 128, 255])
@pytest.mark.parametrize("mode", [0b01, 0b11])
def test_masked_softmax_matches_reference_for_padded_and_combined_masks(seq_len, mode):
    got, expected = _run_masked_softmax(seq_len, mode)
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("seq_len", [16, 64, 128])
def test_masked_softmax_matches_reference_for_pure_causal_exact_key_cols(seq_len):
    got, expected = _run_masked_softmax(seq_len, 0b10)
    np.testing.assert_array_equal(got, expected)


def test_masked_softmax_uses_query_row_base_for_second_strip():
    got, expected = _run_masked_softmax(31, 0b11, query_row_base=16)
    np.testing.assert_array_equal(got, expected)
    assert np.count_nonzero(got[0, :17]) > 1
    assert np.count_nonzero(got[0, 17:]) == 0


@pytest.mark.parametrize("source", [
    "CONFIG_TILE M=1, N=1, K=1\n"
    f"SET_SCALE S0, imm=0x{_fp16_bits(0.125):04x}\n"
    f"SET_SCALE S1, imm=0x{_fp16_bits(1.0 / 127.0):04x}\n"
    "MASKED_SOFTMAX src1=ACCUM[0], src2=ABUF[0], dst=ABUF[0], sreg=0, flags=0\n"
    "HALT",
    "CONFIG_TILE M=1, N=1, K=1\n"
    "CONFIG_ATTN query_row_base=0, valid_kv_len=16, mode=0b00\n"
    "HALT",
    "CONFIG_TILE M=1, N=1, K=1\n"
    "CONFIG_ATTN query_row_base=0, valid_kv_len=0, mode=0b11\n"
    "HALT",
    "CONFIG_TILE M=1, N=1, K=1\n"
    "CONFIG_ATTN query_row_base=0, valid_kv_len=17, mode=0b01\n"
    "MASKED_SOFTMAX src1=ACCUM[0], src2=ABUF[0], dst=ABUF[0], sreg=0, flags=0\n"
    "HALT",
    "CONFIG_TILE M=1, N=2, K=1\n"
    "CONFIG_ATTN query_row_base=0, valid_kv_len=31, mode=0b10\n"
    "MASKED_SOFTMAX src1=ACCUM[0], src2=ABUF[0], dst=ABUF[0], sreg=0, flags=0\n"
    "HALT",
])
def test_masked_softmax_faults_for_invalid_attention_context(source):
    sim = Simulator()
    sim.load_program(Assembler().assemble(source))
    with pytest.raises(ConfigError):
        sim.run()


def _run_masked_attnv(trace_scale):
    state = MachineState()
    state.tile_config = (0, 0, 1)  # M=16, N=16, K=32
    state.attn_context = {
        "is_valid": True,
        "query_row_base": 0,
        "valid_kv_len": 31,
        "mode": 0b11,
    }
    state.scale_regs[4] = np.float16(0.125)
    state.scale_regs[5] = np.float16(0.25)
    state.scale_regs[6] = np.float16(0.25)
    state.scale_regs[7] = np.float16(trace_scale)
    logits = (np.arange(16 * 32, dtype=np.int32).reshape(16, 32) % 13) - 6
    values = (np.arange(32 * 16, dtype=np.int16).reshape(32, 16) % 17 - 8).astype(np.int8)
    state.accum[:logits.size] = logits.reshape(-1)
    state.abuf[:values.size] = values.tobytes()
    payload = execute_masked_softmax_attnv(
        state,
        MaskedSoftmaxAttnVInsn(
            src1_buf=BUF_ACCUM, src1_off=0,
            src2_buf=BUF_ABUF, src2_off=0,
            dst_buf=BUF_WBUF, dst_off=0,
            sreg=4,
        ),
    )
    dest = np.frombuffer(bytes(state.wbuf[:16 * 16]), dtype=np.int8).reshape(16, 16).copy()
    return dest, payload["softmax"]["raw"]


def test_masked_softmax_attnv_trace_scale_does_not_change_destination():
    dest_a, trace_a = _run_masked_attnv(1.0 / 127.0)
    dest_b, trace_b = _run_masked_attnv(1.0 / 64.0)

    np.testing.assert_array_equal(dest_a, dest_b)
    assert not np.array_equal(trace_a, trace_b)
