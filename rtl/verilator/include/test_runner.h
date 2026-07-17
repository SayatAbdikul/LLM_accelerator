// Shared TEST_PASS / TEST_FAIL boilerplate for rtl/verilator/test_*.cpp.
//
// Refactor R3 (2026-05-22): every test_*.cpp file in this directory used to
// declare its own pair of file-static counters plus near-identical macros:
//
//     static int tests_run  = 0;
//     static int tests_pass = 0;
//     #define TEST_PASS(name) do {
//         printf("PASS: %s\n", name); tests_pass++; tests_run++;
//     } while (0)
//     #define TEST_FAIL(name, msg) do {
//         fprintf(stderr, "FAIL: %s — %s\n", name, msg); std::exit(1);
//     } while (0)
//
// This header centralizes both. `tests_run` and `tests_pass` are declared
// inline (C++17 — Verilator builds use `-std=c++17` per
// rtl/verilator/Makefile CXXFLAGS), so each test binary still sees a
// single shared counter even when its sources are split across multiple
// translation units (e.g. R4's test_systolic_qkt_{basic,replay,padded}).
//
// Output format preserved bit-for-bit against the prior majority style:
//   PASS:  "PASS: <name>\n"             via std::printf to stdout
//   FAIL:  "FAIL: <name> — <msg>\n"     via std::fprintf to stderr,
//                                       then std::exit(1)
// The em-dash separator matches all sites EXCEPT test_systolic_qkt.cpp,
// which previously used an ASCII hyphen; that file's FAIL string now
// matches the rest. PASS lines are unchanged everywhere.
//
// Each test_*.cpp now does:
//     #include "test_runner.h"
// in place of the duplicated declarations, and keeps its own summary
// line (e.g. `printf("\n%d / %d tests passed\n", tests_pass, tests_run);`)
// — the symbol names are unchanged, so summary code requires no edits.

#ifndef RTL_VERILATOR_TEST_RUNNER_H
#define RTL_VERILATOR_TEST_RUNNER_H

#include <cstdio>
#include <cstdlib>

inline int tests_run  = 0;
inline int tests_pass = 0;
// 2026-07-17: failure counter. `EXPECT` (include/testbench.h) used to
// std::exit(1) on the first failed assertion, so ONE stale expectation
// silently disabled every test after it in the same binary — test_control
// died at test 14 of 26 from 2026-05-23 (commit e7b3314, the gen-1 SFU
// opcode strip) until 2026-07-17, hiding 12 fault-path tests including
// test_systolic_sram_oob_fault. EXPECT now records and returns; each
// binary's `tests_pass != tests_run` summary check catches it.
inline int tests_fail = 0;

#define TEST_PASS(name) do {                                                \
    std::printf("PASS: %s\n", name);                                        \
    ++tests_pass;                                                            \
    ++tests_run;                                                             \
} while (0)

#define TEST_FAIL(name, msg) do {                                           \
    std::fprintf(stderr, "FAIL: %s \xe2\x80\x94 %s\n", name, msg);          \
    std::exit(1);                                                            \
} while (0)

#endif  // RTL_VERILATOR_TEST_RUNNER_H
