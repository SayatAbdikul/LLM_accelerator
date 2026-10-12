// Standalone fetch/cache protocol tests. Build twice with FETCH_BUFFER_ENABLE
// and TEST_FETCH_BUFFER_ENABLE set to the same value (0 or 1).
#include "Vfetch_unit.h"
#include "verilated.h"

#include <cstdint>
#include <cstdio>
#include <stdexcept>

#ifndef TEST_FETCH_BUFFER_ENABLE
#error "Define TEST_FETCH_BUFFER_ENABLE to match the RTL parameter"
#endif

static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

struct FetchBench {
    Vfetch_unit dut;
    unsigned reads = 0;
    struct Sample {
        bool ar, ar_ready, valid, ready, fault;
        uint64_t addr, data;
        unsigned code;
    };

    Sample tick() {
        dut.clk = 0;
        dut.eval();
        Sample s{bool(dut.m_axi_ar_valid), bool(dut.m_axi_ar_ready),
                 bool(dut.insn_valid), bool(dut.m_axi_r_ready), bool(dut.fetch_fault),
                 uint64_t(dut.m_axi_ar_addr), uint64_t(dut.insn_data),
                 unsigned(dut.fetch_fault_code)};
        if (s.ar && s.ar_ready) ++reads;
        dut.clk = 1;
        dut.eval();
        return s;
    }

    FetchBench() { reset(); }
    void reset() {
        dut.rst_n = 0;
        dut.fetch_req = 0;
        dut.pc = 0;
        dut.invalidate = 0;
        dut.m_axi_ar_ready = 0;
        dut.m_axi_r_valid = 0;
        dut.m_axi_r_resp = 0;
        dut.m_axi_r_last = 0;
        for (int i = 0; i < 4; ++i) dut.m_axi_r_data[i] = 0;
        tick(); tick();
        dut.rst_n = 1;
        tick();
        reads = 0;
    }

    void beat(uint64_t even, uint64_t odd) {
        for (int word = 0; word < 4; ++word) dut.m_axi_r_data[word] = 0;
        for (unsigned byte = 0; byte < 16; ++byte) {
            const uint64_t insn = byte < 8 ? even : odd;
            const auto value = uint32_t((insn >> (56 - 8 * (byte % 8))) & 0xff);
            dut.m_axi_r_data[byte / 4] |= value << (8 * (byte % 4));
        }
    }

    // Keep the control-side request asserted until completion, matching the
    // real controller. AR stalls and R latency are independently exercised.
    uint64_t read(uint64_t pc, uint64_t even, uint64_t odd,
                  int ar_stalls = 3, int response_delay = 3) {
        dut.pc = pc;
        dut.fetch_req = 1;
        bool accepted = false;
        int age = 0;
        for (int cycle = 0; cycle < 100; ++cycle) {
            dut.m_axi_ar_ready = cycle >= ar_stalls;
            dut.m_axi_r_valid = accepted && age >= response_delay;
            dut.m_axi_r_last = 1;
            dut.m_axi_r_resp = 0;
            beat(even, odd);
            Sample s = tick();
            require(!s.fault, "unexpected fetch fault");
            if (s.ar) {
                require(s.addr == ((pc * 8) & ~uint64_t(15)), "wrong/stalled AR address");
                require(dut.m_axi_ar_len == 0 && dut.m_axi_ar_size == 4 &&
                        dut.m_axi_ar_burst == 1, "wrong fetch AXI attributes");
            }
            if (s.ar && s.ar_ready) accepted = true;
            if (accepted) ++age;
            if (s.valid) {
                dut.fetch_req = 0;
                dut.m_axi_r_valid = 0;
                tick();
                require(s.data == ((pc & 1) ? odd : even), "wrong instruction half/byte order");
                return s.data;
            }
        }
        throw std::runtime_error("fetch timeout");
    }

    void invalidate() {
        dut.invalidate = 1;
        tick();
        dut.invalidate = 0;
    }
};

static constexpr uint64_t EVEN = 0x0123456789abcdefULL;
static constexpr uint64_t ODD  = 0xfedcba9876543210ULL;

static void test_halves_and_tags() {
    FetchBench b;
    b.read(0, EVEN, ODD);
    b.read(1, EVEN, ODD);
    require(b.reads == (TEST_FETCH_BUFFER_ENABLE ? 1u : 2u), "instruction pair read count");
    b.read(2, EVEN ^ 1, ODD ^ 1);
    b.read(0, EVEN, ODD);
    require(b.reads == (TEST_FETCH_BUFFER_ENABLE ? 3u : 4u), "tag replacement/read count");
}

static void test_invalidation_and_reset() {
    FetchBench b;
    b.read(0, EVEN, ODD);
    b.invalidate(); // new run / completed host patch
    b.read(1, EVEN, ODD ^ 1);
    require(b.reads == 2, "invalidate retained stale instruction");
    b.dut.invalidate = 1; // DMA store active: bypass even across successful reads
    b.read(0, EVEN ^ 2, ODD ^ 2);
    b.read(1, EVEN ^ 3, ODD ^ 3);
    b.dut.invalidate = 0;
    b.read(0, EVEN ^ 4, ODD ^ 4);
    require(b.reads == 5, "store bypass incorrectly refilled cache");
    b.reset();
    b.read(1, EVEN, ODD ^ 5);
    require(b.reads == 1, "reset retained instruction");
}

static void test_pending_refill_invalidation(bool same_cycle) {
    FetchBench b;
    b.dut.pc = 8;
    b.dut.fetch_req = 1;
    b.dut.m_axi_ar_ready = 1;
    b.tick(); // latch miss
    require(b.tick().ar, "expected AR");
    if (!same_cycle) b.invalidate();
    b.dut.invalidate = same_cycle;
    b.dut.m_axi_r_valid = 1;
    b.dut.m_axi_r_last = 1;
    b.beat(EVEN, ODD);
    auto s = b.tick();
    require(s.valid && s.data == EVEN, "pending read must still complete");
    b.dut.fetch_req = 0;
    b.dut.m_axi_r_valid = 0;
    b.dut.invalidate = 0;
    b.tick();
    b.read(9, EVEN, ODD ^ 1);
    require(b.reads == 2, "invalidated pending response refilled cache");
}

static void test_latched_request_under_backpressure() {
    FetchBench b;
    b.dut.pc = 5;
    b.dut.fetch_req = 1;
    b.tick();
    b.dut.pc = 19; // upstream changes must not affect the accepted request
    for (int i = 0; i < 7; ++i) {
        auto s = b.tick();
        require(s.ar && s.addr == 32 && !s.valid, "AR changed while stalled");
    }
    b.dut.m_axi_ar_ready = 1;
    b.tick();
    b.dut.m_axi_r_valid = 1;
    b.dut.m_axi_r_last = 1;
    b.beat(EVEN, ODD);
    auto s = b.tick();
    require(s.valid && s.data == ODD, "PC lane changed during outstanding read");
}

static void test_fault(int response, bool last) {
    FetchBench b;
    b.dut.fetch_req = 1;
    b.dut.m_axi_ar_ready = 1;
    b.tick(); b.tick();
    b.dut.m_axi_r_valid = 1;
    b.dut.m_axi_r_resp = response;
    b.dut.m_axi_r_last = last;
    require(!b.tick().valid, "bad AXI response produced instruction");
    b.dut.fetch_req = 0;
    b.dut.m_axi_r_valid = 0;
    b.dut.invalidate = 1;
    for (int i = 0; i < 4; ++i) {
        auto s = b.tick();
        require(s.fault && s.code == 2 && !s.valid && !s.ar,
                "fault is not sticky or invalidate cleared it");
    }
    b.reset();
    b.read(0, EVEN, ODD);
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    try {
        test_halves_and_tags();
        test_invalidation_and_reset();
        test_pending_refill_invalidation(false);
        test_pending_refill_invalidation(true);
        test_latched_request_under_backpressure();
        test_fault(2, true);
        test_fault(0, false);
        std::printf("PASS: 7 fetch cases (buffer=%d)\n", TEST_FETCH_BUFFER_ENABLE);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
