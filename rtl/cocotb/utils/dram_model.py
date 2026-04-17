"""Behavioral DRAM model for cocotb testbenches.

Stores bytes in a Python bytearray.  Provides helpers to:
  - Write big-endian TACCEL instructions at a given PC index
  - Write raw bytes at a byte address
  - Drive an AXI4 read slave in a coroutine

TACCEL instruction byte layout in DRAM:
  DRAM[pc*8 + 0] = instruction MSByte (opcode, bits [63:56])
  DRAM[pc*8 + 7] = instruction LSByte (bits  [7:0])

AXI4 byte ordering:
  r_data[7:0]   = DRAM[aligned_addr + 0]
  r_data[15:8]  = DRAM[aligned_addr + 1]
  ...
  r_data[127:120] = DRAM[aligned_addr + 15]

The fetch_unit's bswap64() converts from AXI LE byte order back to the
big-endian instruction word.
"""

import struct
from typing import Optional
import cocotb
from cocotb.triggers import RisingEdge, Timer


class DramModel:
    """16 MB (default) byte-addressable DRAM model."""

    def __init__(self, size: int = 16 * 1024 * 1024):
        self.mem = bytearray(size)
        self.size = size
        self._next_r_resp = 0
        self._next_r_last_override = None
        self._next_b_resp_override = None
        self.ar_log = []
        self.read_beat_count = 0

    # -----------------------------------------------------------------------
    # Write helpers
    # -----------------------------------------------------------------------

    def write_insn(self, pc_idx: int, word: int) -> None:
        """Write a 64-bit big-endian instruction at instruction index pc_idx."""
        base = pc_idx * 8
        assert base + 7 < self.size, f"PC {pc_idx} out of DRAM bounds"
        self.mem[base:base + 8] = struct.pack(">Q", word & 0xFFFFFFFFFFFFFFFF)

    def write_program(self, insns: list) -> None:
        """Write a list of 64-bit instruction words starting at PC=0."""
        for i, w in enumerate(insns):
            self.write_insn(i, w)

    def write_bytes(self, addr: int, data: bytes) -> None:
        """Write raw bytes at a byte address."""
        assert addr + len(data) <= self.size
        self.mem[addr:addr + len(data)] = data

    def read_bytes(self, addr: int, length: int) -> bytes:
        return bytes(self.mem[addr:addr + length])

    def inject_next_read(self, resp: int = 0, force_last: Optional[int] = None) -> None:
        """Override the next AXI read beat's RRESP and/or RLAST."""
        self._next_r_resp = int(resp) & 0x3
        self._next_r_last_override = None if force_last is None else int(force_last)

    def inject_next_bresp(self, resp: int = 0) -> None:
        """Override the next AXI write response BRESP."""
        self._next_b_resp_override = int(resp) & 0x3

    @staticmethod
    def _sig_int(sig, default: int = 0) -> int:
        """Best-effort signal conversion that tolerates unresolved X/Z values."""
        try:
            v = sig.value
            return int(v) if v.is_resolvable else default
        except Exception:
            return default

    # -----------------------------------------------------------------------
    # AXI4 slave coroutine
    # -----------------------------------------------------------------------

    async def axi_slave(self, dut, latency: int = 2) -> None:
        """Coroutine: drive AXI4 read slave (AR/R channels only).

        Handles multi-beat AR/R bursts. Start axi_write_slave() separately
        for AW/W/B support.

        Run as background tasks:
            cocotb.start_soon(dram.axi_slave(dut))
            cocotb.start_soon(dram.axi_write_slave(dut))  # for STORE
        """
        dut.m_axi_ar_ready.value = 1
        dut.m_axi_r_valid.value  = 0
        dut.m_axi_r_data.value   = 0
        dut.m_axi_r_resp.value   = 0
        dut.m_axi_r_last.value   = 0

        # pending: list of {'addr': int, 'ar_len': int, 'beat': int, 'cd': int}
        pending = []

        while True:
            await RisingEdge(dut.clk)

            # Keep read channel quiescent during reset.
            if self._sig_int(dut.rst_n, default=0) == 0:
                pending.clear()
                dut.m_axi_ar_ready.value = 1
                dut.m_axi_r_valid.value  = 0
                dut.m_axi_r_last.value   = 0
                continue

            # Accept new AR transaction
            if self._sig_int(dut.m_axi_ar_valid, 0) and self._sig_int(dut.m_axi_ar_ready, 0):
                aligned  = int(dut.m_axi_ar_addr.value) & ~0xF
                ar_len   = int(dut.m_axi_ar_len.value)
                self.ar_log.append((aligned, ar_len))
                pending.append({'addr': aligned, 'ar_len': ar_len,
                                'beat': 0, 'cd': latency})

            # Decrement countdowns
            for req in pending:
                if req['cd'] > 0:
                    req['cd'] -= 1

            # Serve one beat for the oldest ready request
            for req in pending:
                if req['cd'] <= 0:
                    beat_addr = req['addr'] + req['beat'] * 16
                    chunk = self.mem[beat_addr:beat_addr + 16]
                    chunk = bytes(chunk).ljust(16, b'\x00')
                    val = int.from_bytes(chunk, 'little')
                    dut.m_axi_r_data.value  = val
                    dut.m_axi_r_valid.value = 1
                    dut.m_axi_r_resp.value  = self._next_r_resp
                    is_last = (req['beat'] >= req['ar_len'])
                    if self._next_r_last_override is not None:
                        is_last = bool(self._next_r_last_override)
                    dut.m_axi_r_last.value  = 1 if is_last else 0
                    self._next_r_resp = 0
                    self._next_r_last_override = None

                    # Gate ar_ready while waiting for this beat to be accepted,
                    # so a concurrent fetch AR isn't silently dropped.
                    dut.m_axi_ar_ready.value = 0

                    # Wait for master to accept this beat
                    while True:
                        await RisingEdge(dut.clk)
                        if self._sig_int(dut.m_axi_r_ready, 0):
                            self.read_beat_count += 1
                            break
                    dut.m_axi_r_valid.value  = 0
                    dut.m_axi_r_last.value   = 0
                    dut.m_axi_ar_ready.value = 1  # re-open AR channel

                    if is_last:
                        pending.remove(req)
                    else:
                        req['beat'] += 1
                        req['cd'] = 1   # next beat next cycle
                    break  # one beat per outer loop iteration

    async def axi_write_slave(
        self,
        dut,
        aw_stall_cycles: int = 0,
        w_stall_cycles: int = 0,
        b_valid_delay_cycles: int = 0,
        b_resp: int = 0,
    ) -> None:
        """Coroutine: drive AXI4 write slave (AW/W/B channels).

        Run as a background task alongside axi_slave():
            cocotb.start_soon(dram.axi_write_slave(dut))

        Args:
            aw_stall_cycles: Number of cycles to hold AWREADY low before each burst.
            w_stall_cycles: Number of cycles to hold WREADY low before each beat.
            b_valid_delay_cycles: Cycles to wait between last W and BVALID assertion.
            b_resp: BRESP value to return (0=OKAY, 2=SLVERR, 3=DECERR).
        """
        aw_stall_cfg = max(0, int(aw_stall_cycles))
        w_stall_cfg = max(0, int(w_stall_cycles))
        b_delay_cfg = max(0, int(b_valid_delay_cycles))
        b_resp_val = int(b_resp) & 0x3

        dut.m_axi_aw_ready.value = 1
        dut.m_axi_w_ready.value  = 0
        dut.m_axi_b_valid.value  = 0
        dut.m_axi_b_resp.value   = b_resp_val

        while True:
            await RisingEdge(dut.clk)

            # Keep channel outputs in a safe state while reset is asserted.
            if self._sig_int(dut.rst_n, default=0) == 0:
                dut.m_axi_aw_ready.value = 1
                dut.m_axi_w_ready.value  = 0
                dut.m_axi_b_valid.value  = 0
                dut.m_axi_b_resp.value   = b_resp_val
                continue

            # Optional AW backpressure before each burst.
            for _ in range(aw_stall_cfg):
                dut.m_axi_aw_ready.value = 0
                await RisingEdge(dut.clk)
                if self._sig_int(dut.rst_n, default=0) == 0:
                    break
            if self._sig_int(dut.rst_n, default=0) == 0:
                continue
            dut.m_axi_aw_ready.value = 1

            # Accept AW only on an edge-stable handshake. Do not use delta-only
            # acceptance; that can make the model think AW happened while the DUT
            # (which samples on clock edges) does not.
            aw_v = self._sig_int(dut.m_axi_aw_valid, 0)
            aw_r = self._sig_int(dut.m_axi_aw_ready, 0)
            if aw_v == 0 or aw_r == 0:
                continue

            aw_addr = int(dut.m_axi_aw_addr.value)
            aw_len = int(dut.m_axi_aw_len.value)
            beats_expected = aw_len + 1
            dut._log.info(f"[write_slave] AW: addr=0x{aw_addr:08x} len={aw_len}")

            # Block additional AW while this burst is in flight.
            dut.m_axi_aw_ready.value = 0
            dut.m_axi_w_ready.value = 1

            beat = 0
            stall_left = w_stall_cfg
            while beat < beats_expected:
                await RisingEdge(dut.clk)

                if self._sig_int(dut.rst_n, default=0) == 0:
                    dut.m_axi_w_ready.value = 0
                    dut.m_axi_b_valid.value = 0
                    dut.m_axi_aw_ready.value = 1
                    break

                # Optional per-beat W backpressure.
                if stall_left > 0:
                    dut.m_axi_w_ready.value = 0
                    stall_left -= 1
                    continue
                dut.m_axi_w_ready.value = 1

                # Accept W only on an edge-stable handshake.
                w_valid = self._sig_int(dut.m_axi_w_valid, 0)
                w_ready = self._sig_int(dut.m_axi_w_ready, 0)
                if not (w_valid and w_ready):
                    continue

                w_val = int(dut.m_axi_w_data.value)
                w_strb = int(dut.m_axi_w_strb.value)
                w_last = self._sig_int(dut.m_axi_w_last, 0)

                beat_addr = aw_addr + beat * 16
                for b in range(16):
                    if ((w_strb >> b) & 0x1) and (beat_addr + b < self.size):
                        self.mem[beat_addr + b] = (w_val >> (b * 8)) & 0xFF

                expected_last = 1 if beat == beats_expected - 1 else 0
                if w_last != expected_last:
                    dut._log.warning(
                        f"[write_slave] WLAST mismatch beat={beat} w_last={w_last} expected={expected_last}"
                    )

                beat += 1
                stall_left = w_stall_cfg

            # If reset hit during beat collection, restart in idle state.
            if beat < beats_expected:
                continue

            dut.m_axi_w_ready.value = 0

            # Send B response.
            for _ in range(b_delay_cfg):
                await RisingEdge(dut.clk)
                if self._sig_int(dut.rst_n, default=0) == 0:
                    break
            if self._sig_int(dut.rst_n, default=0) == 0:
                continue

            burst_b_resp = b_resp_val if self._next_b_resp_override is None else self._next_b_resp_override
            self._next_b_resp_override = None

            dut.m_axi_b_valid.value = 1
            dut.m_axi_b_resp.value = burst_b_resp
            while True:
                await RisingEdge(dut.clk)
                if self._sig_int(dut.rst_n, default=0) == 0:
                    dut.m_axi_b_valid.value = 0
                    break
                if self._sig_int(dut.m_axi_b_ready, 0):
                    dut.m_axi_b_valid.value = 0
                    break

            # Re-open AW channel for next transaction.
            dut.m_axi_aw_ready.value = 1
