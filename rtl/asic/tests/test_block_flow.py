"""Fast report/CLI contract tests; never invoke synthesis, STA, or Docker.

Run with ``python3 -m unittest discover -s rtl/asic/tests -p 'test_*.py'``
or collect with pytest. Every output/library/source fixture is temporary.
"""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch


FLOW_PATH = Path(__file__).resolve().parents[1] / "tools" / "block_flow.py"
SPEC = importlib.util.spec_from_file_location("tested_block_flow", FLOW_PATH)
flow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(flow)
RUN_WITH_RESOURCES = flow.run


class BlockFlowTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.src = self.root / "rtl" / "common" / "src"
        self.src.mkdir(parents=True)
        (self.src / "pe.sv").write_text("module systolic_pe(input clk); endmodule\n")
        self.lib = self.root / "cells.lib"
        self.lef = self.root / "cells.lef"
        self.out = self.root / "results" / "run"
        self.lib.write_text(self.library())
        self.lef.write_text(self.physical_cells())
        self.mapping_log = "Chip area for module systolic_pe: 8.000000\n"
        self.netlist = (
            "module systolic_pe(input clk);\n"
            "  CELL_A _100_ (.A(clk));\n"
            "  CELL_B _101_ (.A(clk));\n"
            "endmodule\n"
        )
        self.cells = {"_100_": {"type": "CELL_A"}, "_101_": {"type": "CELL_B"}}
        self.sta_log = "REGISTER_TO_REGISTER_PATHS\nNo paths found.\n"
        self.top = "systolic_pe"
        self.fail_stage = None
        self.commands = []
        self.stack.enter_context(patch.object(flow, "ROOT", self.root))
        self.stack.enter_context(patch.object(flow, "SRC", self.src))
        self.stack.enter_context(patch.object(flow, "BLOCKS", {
            "systolic_pe": ["pe.sv"], "sram_dp_sky130_banked": []}))
        self.stack.enter_context(patch.object(flow, "version", return_value="mock tool 1.0"))
        self.stack.enter_context(patch.object(flow, "run", side_effect=self.fake_run))
        self.stack.enter_context(patch.object(flow.shutil, "disk_usage",
                                              return_value=SimpleNamespace(free=100 * 2**30)))
        # A new production-code subprocess path must be deliberately mocked;
        # these tests must never accidentally launch an installed heavy tool.
        for name in ("Popen", "run", "check_output"):
            self.stack.enter_context(patch.object(
                flow.subprocess, name, side_effect=AssertionError("unexpected real subprocess")))

    @staticmethod
    def library(time_unit='"1ns"', capacitance='1, "pf"'):
        return (
            "library(test) {\n"
            f"  time_unit : {time_unit};\n"
            f"  capacitive_load_unit ({capacitance});\n"
            '  cell ("CELL_A") { area : 2.0; }\n'
            "  cell (CELL_B) { area : 6.0; }\n"
            "}\n"
        )

    @staticmethod
    def physical_cells(b_size="3 BY 2"):
        return (
            "VERSION 5.8 ;\n"
            "MACRO CELL_A\n  SIZE 2 BY 1 ;\nEND CELL_A\n"
            f"MACRO CELL_B\n  SIZE {b_size} ;\nEND CELL_B\n"
            "END LIBRARY\n"
        )

    def fake_run(self, command, output, timeout):
        self.commands.append(command)
        tool = command[0]
        if tool == self.fail_stage:
            output.write_text("deliberate mocked tool failure\n")
            raise RuntimeError("deliberate mocked tool failure")
        if tool == "sv2v":
            Path(command[command.index("-w") + 1]).write_text(
                f"module {self.top}(input clk); endmodule\n")
            output.write_text("mock conversion\n")
        elif tool == "yosys":
            (output.parent / "mapped.v").write_text(self.netlist)
            (output.parent / "mapped.json").write_text(json.dumps(
                {"modules": {self.top: {"cells": self.cells}}}))
            output.write_text(self.mapping_log)
        elif tool == "sta":
            output.write_text(self.sta_log)
        else:
            self.fail(f"unexpected mocked tool: {command}")
        return {"command": command, "exit_code": 0, "wall_seconds": 0.001,
                "peak_rss_bytes": 1024}

    def invoke(self, *extra, base=True):
        args = [str(FLOW_PATH)]
        if base:
            args += ["--block", "systolic_pe", "--liberty", str(self.lib),
                     "--pdk", "test-pdk", "--corner", "tt", "--out", str(self.out)]
        args += list(extra)
        with patch("sys.argv", args), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return flow.main()

    def report(self):
        return json.loads((self.out / "report.json").read_text())

    def assert_rejected(self, *args, base=True):
        with self.assertRaises(SystemExit) as caught:
            self.invoke(*args, base=base)
        self.assertEqual(caught.exception.code, 2)
        self.assertFalse(self.commands, "invalid input reached a hardware tool")
        self.assertFalse((self.out / "report.json").exists())

    def test_path_quoting_preserves_spaces_and_unicode(self):
        value = Path("/tmp/design results/芯片.lib")
        self.assertEqual(flow.quoted(value), '"/tmp/design results/芯片.lib"')

    def test_path_quoting_rejects_interpreted_characters(self):
        for character in ('"', "\n", "\r", "$", "[", "]", "\\"):
            with self.subTest(character=repr(character)), self.assertRaises(ValueError):
                flow.quoted(Path("/tmp/a" + character + "b.lib"))

    def test_missing_required_arguments_are_rejected(self):
        self.assert_rejected(base=False)

    def test_missing_timing_option_values_are_rejected(self):
        for option in ("--clock-period-ns", "--abc-delay-ps", "--io-delay-fraction",
                       "--output-load-pf"):
            with self.subTest(option=option):
                self.assert_rejected(option)

    def test_nonfinite_timing_constraints_are_rejected(self):
        for option in ("--clock-period-ns", "--output-load-pf", "--io-delay-fraction"):
            for value in ("nan", "inf", "-inf"):
                with self.subTest(option=option, value=value):
                    self.assert_rejected(f"{option}={value}")

    def test_out_of_range_timing_constraints_are_rejected(self):
        for option, value in (("--clock-period-ns", "0"), ("--clock-period-ns", "-1"),
                              ("--abc-delay-ps", "0"), ("--output-load-pf", "-1"),
                              ("--io-delay-fraction", "-.01"), ("--io-delay-fraction", ".5")):
            with self.subTest(option=option, value=value):
                self.assert_rejected(f"{option}={value}")

    def test_quoted_and_unquoted_pf_units_are_accepted(self):
        for index, capacitance in enumerate(('1, "pf"', "1, pf", '1.000, "pf"')):
            with self.subTest(capacitance=capacitance):
                self.lib.write_text(self.library(capacitance=capacitance))
                self.out = self.root / f"unit-case-{index}"
                self.assertEqual(self.invoke(), 0)
                self.assertEqual(self.report()["constraints"]["clock_period_ns"], 30.0)

    def test_missing_unsupported_or_malformed_library_units_are_rejected(self):
        libraries = [
            "library(test) {}\n", self.library(time_unit='"1ps"'),
            self.library(capacitance='1, "ff"'), self.library(capacitance='10, "pf"'),
            self.library(capacitance='1, "pf'), self.library(capacitance='1, pf"'),
        ]
        for contents in libraries:
            with self.subTest(contents=contents):
                self.lib.write_text(contents)
                self.assert_rejected()

    def test_area_remains_unverified_without_physical_lef(self):
        self.assertEqual(self.invoke(), 0)
        report = self.report()
        self.assertEqual(report["standard_cell_area_liberty_units"], 8.0)
        self.assertFalse(report["area_units_verified"])
        self.assertNotIn("standard_cell_area_um2", report)
        self.assertNotIn("register_to_register_setup_bound_ns_zero_wire", report)

    def test_matching_physical_lef_verifies_area_units(self):
        self.assertEqual(self.invoke("--cell-lef", str(self.lef)), 0)
        report = self.report()
        self.assertTrue(report["area_units_verified"])
        self.assertEqual(report["standard_cell_area_um2"], 8.0)
        self.assertEqual(report["cell_lef"]["verified_cell_types"], 2)
        self.assertEqual(report["cell_lef"]["sha256"], flow.digest(self.lef))

    def test_missing_or_mismatched_lef_cells_fail_closed(self):
        for contents in (self.physical_cells("3 BY 3"), "MACRO CELL_A\nSIZE 2 BY 1 ;\nEND CELL_A\n"):
            with self.subTest(contents=contents):
                self.lef.write_text(contents)
                with self.assertRaisesRegex(RuntimeError, "missing/mismatched"):
                    self.invoke("--cell-lef", str(self.lef))
                self.assertFalse((self.out / "report.json").exists())

    def test_named_instances_cannot_escape_physical_area_verification(self):
        self.netlist = self.netlist.replace("CELL_B _101_", "CELL_B named_instance")
        self.cells["named_instance"] = self.cells.pop("_101_")
        self.lef.write_text(self.physical_cells("3 BY 3"))
        with self.assertRaisesRegex(RuntimeError, "missing/mismatched"):
            self.invoke("--cell-lef", str(self.lef))
        self.assertFalse((self.out / "report.json").exists())

    def test_empty_mapped_inventory_cannot_verify_area(self):
        self.netlist = "module systolic_pe(input clk); endmodule\n"
        self.cells = {}
        with self.assertRaisesRegex(RuntimeError, "no mapped"):
            self.invoke("--cell-lef", str(self.lef))

    def configure_bank(self, macro_count=8, nominal=12.0):
        self.top = "sram_dp_sky130_banked"
        self.lib.write_text(self.library().replace("library(test)",
            "library(sky130_fd_sc_hd__tt_025C_1v80)").replace("CELL_", "sky130_fd_sc_hd__cell_"))
        self.lef.write_text(self.physical_cells().replace("CELL_", "sky130_fd_sc_hd__cell_"))
        for cell in self.cells.values():
            cell["type"] = cell["type"].replace("CELL_", "sky130_fd_sc_hd__cell_")
        asic = self.root / "rtl/asic"
        (asic / "src").mkdir(parents=True)
        (asic / "src/sram_dp_sky130_banked.sv").write_text(
            f"module {self.top}(input clk); endmodule\n")
        self.macro_lib = self.root / "macro.lib"
        self.macro_lef = self.root / "macro.lef"
        self.macro_lib.write_text(f"library(mem) {{ cell (SRAM) {{ area : {nominal}; }} }}\n")
        self.macro_lef.write_text("MACRO SRAM\nSIZE 3 BY 4 ;\nEND SRAM\n")
        (asic / "macro_views.json").write_text(json.dumps({
            "macro": "SRAM", "revision": "a" * 40,
            "views": {"macro.lib": flow.digest(self.macro_lib),
                      "macro.lef": flow.digest(self.macro_lef)}}))
        self.cells.update({f"bank_{i}": {"type": "SRAM"} for i in range(macro_count)})
        self.mapping_log = f"Chip area for module {self.top}: {8.0 + macro_count * nominal:.6f}\n"
        return ("--block", self.top, "--pdk", "sky130hd", "--macro-liberty", str(self.macro_lib),
                "--macro-lef", str(self.macro_lef), "--cell-lef", str(self.lef))

    def test_bank_rejects_mixed_process_or_mislabeled_periphery_before_tools(self):
        args = self.configure_bank()
        self.assert_rejected(*args, "--pdk", "icsprout55")
        self.lib.write_text(self.library())
        self.assert_rejected(*args)

    def test_mapping_script_asserts_structural_checks(self):
        self.assertEqual(self.invoke(), 0)
        script = (self.out / "map.ys").read_text()
        self.assertIn("check -assert\n", script)
        self.assertLess(script.index("read_liberty -lib"), script.index("check -assert"))

    def test_bank_requires_explicit_macro_and_physical_inputs(self):
        self.assert_rejected("--block", "sram_dp_sky130_banked")

    def test_macro_options_do_not_apply_to_other_leaves(self):
        self.assert_rejected("--macro-liberty", str(self.lib))

    def test_bank_depth_and_unvalidated_timing_are_rejected(self):
        args = self.configure_bank()
        for extra in (("--sram-depth", "0"), ("--sram-depth", "768"),
                      ("--sram-depth", "32768"), ("--sta",)):
            with self.subTest(extra=extra):
                self.assert_rejected(*args, *extra)

    def test_macro_view_hash_mismatch_fails_before_tools(self):
        args = self.configure_bank()
        self.macro_lef.write_text("changed view\n")
        self.assert_rejected(*args)

    def test_bank_reports_macro_and_peripheral_area_separately(self):
        self.assertEqual(self.invoke(*self.configure_bank()), 0)
        report = self.report()
        self.assertEqual(report["standard_cell_area_um2"], 8.0)
        self.assertEqual(report["macro_area_um2"], 96.0)
        self.assertEqual(report["total_cell_and_macro_area_um2"], 104.0)
        self.assertEqual(report["macro"]["instances"], 8)
        self.assertEqual(report["macro"]["logical_capacity_bytes"], 16384)
        self.assertEqual(report["parameters"], {"DATA_W": 128, "DEPTH": 1024})
        self.assertTrue(report["macro"]["liberty_area_matches_lef"])
        self.assertIn("read_liberty -lib", (self.out / "map.ys").read_text())
        self.assertIn("-DSYNTHESIS", self.commands[0])

    def test_macro_liberty_area_discrepancy_is_explicit_not_silently_repaired(self):
        self.assertEqual(self.invoke(*self.configure_bank(nominal=11.999)), 0)
        report = self.report()
        self.assertFalse(report["macro"]["liberty_area_matches_lef"])
        self.assertAlmostEqual(report["macro"]["liberty_minus_lef_area_per_instance"], -.001)
        self.assertEqual(report["macro_area_um2"], 96.0)
        self.assertAlmostEqual(report["mapped_cell_and_macro_area_liberty_units"], 103.992)

    def test_wrong_macro_instance_count_cannot_pass_bank_gate(self):
        with self.assertRaisesRegex(RuntimeError, "bank structure mismatch"):
            self.invoke(*self.configure_bank(macro_count=7))
        self.assertFalse((self.out / "report.json").exists())

    def test_missing_mapped_area_does_not_create_success_report(self):
        self.mapping_log = "No area statistic available\n"
        with self.assertRaisesRegex(RuntimeError, "area unavailable"):
            self.invoke()
        self.assertFalse((self.out / "report.json").exists())

    def test_stale_report_refused_without_rerun(self):
        self.out.mkdir(parents=True)
        previous = '{"stage":"old_success"}\n'
        (self.out / "report.json").write_text(previous)
        with self.assertRaises(SystemExit) as caught:
            self.invoke()
        self.assertEqual(caught.exception.code, 2)
        self.assertFalse(self.commands)
        self.assertEqual((self.out / "report.json").read_text(), previous)

    def test_failed_rerun_invalidates_but_preserves_previous_report(self):
        self.out.mkdir(parents=True)
        previous = '{"stage":"old_success"}\n'
        (self.out / "report.json").write_text(previous)
        self.fail_stage = "sv2v"
        with self.assertRaisesRegex(RuntimeError, "deliberate mocked"):
            self.invoke("--rerun")
        self.assertFalse((self.out / "report.json").exists())
        archived = list(self.out.glob("report.previous-*.json"))
        self.assertEqual(len(archived), 1)
        self.assertEqual(archived[0].read_text(), previous)

    def test_successful_rerun_replaces_report_and_archives_previous(self):
        self.out.mkdir(parents=True)
        (self.out / "report.json").write_text('{"stage":"old_success"}\n')
        self.assertEqual(self.invoke("--rerun"), 0)
        self.assertEqual(self.report()["stage"], "mapped")
        self.assertEqual(len(list(self.out.glob("report.previous-*.json"))), 1)

    def test_missing_sta_timing_is_null_not_zero(self):
        self.assertEqual(self.invoke("--sta"), 0)
        report = self.report()
        self.assertIsNone(report["tool_reported_clock_min_period_ns"])
        self.assertIsNone(report["register_to_register_setup_bound_ns_zero_wire"])
        self.assertIn("register-to-register setup path", report["unavailable"])

    def test_missing_register_marker_does_not_reclassify_io_slack(self):
        self.sta_log = "Input-to-output path\n-3.000000 slack (VIOLATED)\n"
        self.assertEqual(self.invoke("--sta"), 0)
        self.assertIsNone(self.report()["register_to_register_setup_bound_ns_zero_wire"])

    def test_register_setup_bound_excludes_earlier_io_paths(self):
        self.sta_log = (
            "Input-to-output path\n-3.000000 slack (VIOLATED)\n"
            "REGISTER_TO_REGISTER_PATHS\n22.500000 slack (MET)\nperiod_min = 7.5\n"
        )
        self.assertEqual(self.invoke("--sta"), 0)
        report = self.report()
        self.assertEqual(report["register_to_register_setup_bound_ns_zero_wire"], 7.5)
        self.assertEqual(report["tool_reported_clock_min_period_ns"], 7.5)
        self.assertIn("post-layout achieved frequency", report["unavailable"])

    def test_sta_error_does_not_create_success_report(self):
        self.sta_log = "Error: failed to link cell\n"
        with self.assertRaisesRegex(RuntimeError, "OpenSTA reported errors"):
            self.invoke("--sta")
        self.assertFalse((self.out / "report.json").exists())

    def test_resource_rss_units_match_host_time_format(self):
        for host, resource_line, expected in (
            ("Darwin", "4096  maximum resident set size\n", 4096),
            ("Linux", "Maximum resident set size (kbytes): 4096\n", 4096 * 1024),
            ("Linux", "resource measurement unavailable\n", None),
        ):
            with self.subTest(host=host, resource_line=resource_line):
                process = SimpleNamespace(pid=12345, wait=Mock(return_value=0))

                def launch(command, *, stdout, **kwargs):
                    self.assertTrue(kwargs["start_new_session"])
                    stdout.write(resource_line)
                    return process

                with patch.object(flow.platform, "system", return_value=host), \
                     patch.object(flow.subprocess, "Popen", side_effect=launch):
                    metrics = RUN_WITH_RESOURCES(["mock-tool"], self.root / "resource.log", 1)
                self.assertEqual(metrics["peak_rss_bytes"], expected)
                self.assertEqual(metrics["exit_code"], 0)

    def test_nonzero_tool_exit_is_not_a_successful_measurement(self):
        process = SimpleNamespace(pid=12345, wait=Mock(return_value=7))
        with patch.object(flow.subprocess, "Popen", return_value=process):
            with self.assertRaisesRegex(RuntimeError, "command failed.*7"):
                RUN_WITH_RESOURCES(["mock-tool"], self.root / "failed.log", 1)

    def test_timeout_terminates_process_group_and_named_container(self):
        process = SimpleNamespace(pid=12345, wait=Mock(side_effect=[
            subprocess.TimeoutExpired("mock-tool", 1), 0]))
        command = ["docker", "run", "--name", "mock-container", "mock-image"]
        with patch.object(flow.subprocess, "Popen", return_value=process), \
             patch("os.killpg") as kill_group, \
             patch.object(flow.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as cleanup:
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                RUN_WITH_RESOURCES(command, self.root / "timeout.log", 1)
        kill_group.assert_any_call(12345, signal.SIGTERM)
        self.assertEqual(cleanup.call_args.args[0], ["docker", "rm", "-f", "mock-container"])

    def test_timeout_escalates_if_process_does_not_exit(self):
        process = SimpleNamespace(pid=12345, wait=Mock(side_effect=[
            subprocess.TimeoutExpired("mock-tool", 1),
            subprocess.TimeoutExpired("mock-tool", 5), 0]))
        with patch.object(flow.subprocess, "Popen", return_value=process), patch("os.killpg") as kill_group:
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                RUN_WITH_RESOURCES(["mock-tool"], self.root / "timeout-kill.log", 1)
        kill_group.assert_any_call(12345, signal.SIGTERM)
        kill_group.assert_any_call(12345, signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
