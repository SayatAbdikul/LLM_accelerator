#!/usr/bin/env python3
"""Portable, serial leaf/bank mapping and optional zero-wire leaf STA.

This is not full-chip PNR or a power estimate. Libraries are supplied by the
caller and hashed. No PDK/tool download, whole-SFU flattening, or hidden fallback
is performed. The isolated bank experiment separately accounts for its pinned
macro LEF footprint and standard-cell periphery; it does not offer STA. Run
from any directory; all outputs go to the requested run dir.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import platform
import re
import signal
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "rtl/common/src"
BLOCKS = {
    "systolic_pe": ["systolic/systolic_pe.sv"],
    "dma_engine": ["include/taccel_pkg.sv", "dma_engine.sv"],
    "fetch_unit": ["include/taccel_pkg.sv", "fetch_unit.sv"],
    "fp32_div_p6": ["fp32/fp32_div_p6.sv"],
    "fp32_sqrt_p6": ["fp32/fp32_sqrt_p6.sv"],
    "sram_dp_sky130_banked": [],
}


def liberty_areas(contents: str) -> dict:
    return {name: float(area) for name, area in re.findall(
        r'cell\s*\(\s*"?([^\s\)\"]+)"?\s*\)\s*\{.*?\barea\s*:\s*([0-9.]+)', contents, re.S)}


def lef_areas(contents: str) -> dict:
    return {name: float(x) * float(y) for name, x, y in re.findall(
        r'MACRO\s+(\S+)\s+.*?SIZE\s+([0-9.]+)\s+BY\s+([0-9.]+)', contents, re.S)}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def quoted(path: Path) -> str:
    # Yosys and Tcl accept double-quoted POSIX paths. Reject language metacharacters.
    value = str(path)
    if any(c in value for c in ('"', '\n', '\r', '$', '[', ']', '\\')):
        raise ValueError(f"unsupported path characters: {value}")
    return f'"{value}"'


def version(command: list[str]) -> str:
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=30, check=False)
    return result.stdout.strip()


def run(command: list[str], output: Path, timeout: int) -> dict:
    started = time.monotonic()
    timed = (["/usr/bin/time", "-l"] if platform.system() == "Darwin" else
             ["/usr/bin/time", "-v"])
    with output.open("w") as stream:
        process = subprocess.Popen(timed + command, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True, cwd=ROOT)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os_group = process.pid
            # Kill the whole mapping process group, including ABC descendants.
            # A disconnected Docker client also needs explicit container cleanup.
            import os
            os.killpg(os_group, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(os_group, signal.SIGKILL)
                process.wait()
            if command[:2] == ["docker", "run"] and "--name" in command:
                name = command[command.index("--name") + 1]
                subprocess.run(["docker", "rm", "-f", name], stdout=stream,
                               stderr=subprocess.STDOUT, check=False, timeout=30)
            raise RuntimeError(f"command timed out after {timeout}s; see {output}")
    log = output.read_text(errors="replace")
    if platform.system() == "Darwin":
        rss = re.search(r"(\d+)\s+maximum resident set size", log)
        rss_bytes = int(rss[1]) if rss else None
    else:
        rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", log)
        rss_bytes = int(rss[1]) * 1024 if rss else None
    metrics = {"command": command, "exit_code": code,
               "wall_seconds": time.monotonic() - started, "peak_rss_bytes": rss_bytes}
    if code:
        raise RuntimeError(f"command failed ({code}); see {output}")
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--block", choices=BLOCKS, required=True)
    ap.add_argument("--liberty", type=Path, required=True)
    ap.add_argument("--pdk", required=True, help="library/process label; no automatic assumptions")
    ap.add_argument("--corner", required=True)
    ap.add_argument("--cell-lef", type=Path, help="verify mapped-cell Liberty areas against physical LEF sizes")
    ap.add_argument("--macro-liberty", type=Path, help="pinned SKY130 bank experiment only")
    ap.add_argument("--macro-lef", type=Path, help="pinned SKY130 bank experiment only")
    ap.add_argument("--sram-depth", type=int, default=1024, help="logical 128-bit rows for the bank experiment")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rerun", action="store_true", help="invalidate an earlier generated report in this run directory")
    ap.add_argument("--clock-period-ns", type=float, default=30.0)
    ap.add_argument("--abc-delay-ps", type=int, default=5000)
    ap.add_argument("--io-delay-fraction", type=float, default=0.3)
    ap.add_argument("--output-load-pf", type=float, default=0.05)
    ap.add_argument("--fetch-buffer", type=int, choices=(0, 1), default=1)
    ap.add_argument("--timeout-seconds", type=int, default=1800)
    ap.add_argument("--min-free-gib", type=float, default=20)
    ap.add_argument("--sta", action="store_true")
    ap.add_argument("--docker-image", help="existing pinned image containing OpenSTA; never pulls")
    ap.add_argument("--docker-sta-path", default="/OpenROAD-flow-scripts/tools/install/OpenROAD/bin/sta")
    args = ap.parse_args()
    bank = args.block == "sram_dp_sky130_banked"
    if not bank and (args.macro_liberty or args.macro_lef or args.sram_depth != 1024):
        ap.error("macro inputs/depth apply only to sram_dp_sky130_banked")
    if bank and (not args.macro_liberty or not args.macro_lef or not args.cell_lef):
        ap.error("bank mapping requires --macro-liberty, --macro-lef and --cell-lef")
    if bank and args.pdk != "sky130hd":
        ap.error("pinned SKY130 macro banks require --pdk sky130hd; mixed-process totals are not realizable")
    if bank and (args.sram_depth < 512 or args.sram_depth > 16384 or
                 args.sram_depth & (args.sram_depth - 1)):
        ap.error("bank depth must be a power of two from 512 through 16384")
    if bank and args.sta:
        ap.error("bank experiment is mapping-only; macro timing collateral is not validated")
    lib = args.liberty.resolve(strict=True)
    lib_text = lib.read_text()
    if bank and not re.search(r'library\s*\(\s*"?sky130_fd_sc_hd__', lib_text):
        ap.error("bank periphery requires a SKY130HD standard-cell Liberty library")
    if not re.search(r'time_unit\s*:\s*"1ns"', lib_text) or not re.search(
            r'capacitive_load_unit\s*\(\s*1(?:\.0+)?\s*,\s*(?:"pf"|pf)\s*\)', lib_text):
        ap.error("supported library units are explicitly 1ns and 1pf; convert others before use")
    macro = None
    if bank:
        manifest = json.loads((ROOT / "rtl/asic/macro_views.json").read_text())
        macro_lib = args.macro_liberty.resolve(strict=True)
        macro_lef = args.macro_lef.resolve(strict=True)
        for path, suffix in ((macro_lib, ".lib"), (macro_lef, ".lef")):
            expected = [sha for name, sha in manifest["views"].items() if name.endswith(suffix)]
            if len(expected) != 1 or digest(path) != expected[0]:
                ap.error(f"macro {suffix} does not match the pinned manifest")
        name = manifest["macro"]
        nominal = liberty_areas(macro_lib.read_text()).get(name)
        physical = lef_areas(macro_lef.read_text()).get(name)
        if nominal is None or physical is None or min(nominal, physical) <= 0:
            ap.error("pinned macro is missing a positive Liberty/LEF area")
        macro = {"cell": name, "revision": manifest["revision"],
                 "liberty": {"path": str(macro_lib), "sha256": digest(macro_lib)},
                 "lef": {"path": str(macro_lef), "sha256": digest(macro_lef)},
                 "liberty_area_per_instance": nominal,
                 "lef_area_um2_per_instance": physical,
                 "liberty_minus_lef_area_per_instance": nominal - physical,
                 "liberty_area_matches_lef": abs(nominal - physical) <= max(1e-4, physical * 1e-5),
                 "area_basis": "physical LEF footprint; Liberty discrepancy is retained, not silently corrected",
                 "timing": "unvalidated macro input max_transition/characterization; no timing-closure claim"}
    out = args.out.resolve()
    if not all(math.isfinite(value) for value in (
            args.clock_period_ns, args.io_delay_fraction, args.output_load_pf, args.min_free_gib)):
        ap.error("timing and disk constraints must be finite")
    if args.min_free_gib < 0 or args.timeout_seconds <= 0:
        ap.error("disk guardrail must be nonnegative and timeout positive")
    if args.clock_period_ns <= 0 or args.abc_delay_ps <= 0 or args.output_load_pf < 0:
        ap.error("timing constraints must be positive (load may be zero)")
    if not 0 <= args.io_delay_fraction < 0.5:
        ap.error("IO-delay fraction must be in [0, 0.5)")
    disk_parent = out
    while not disk_parent.exists():
        disk_parent = disk_parent.parent
    if shutil.disk_usage(disk_parent).free < args.min_free_gib * 2**30:
        ap.error("free disk is below the configured guardrail")
    out.mkdir(parents=True, exist_ok=True)
    previous_report = out / "report.json"
    if previous_report.exists():
        if not args.rerun:
            ap.error("report already exists in --out; choose a new directory or explicitly --rerun")
        previous_report.rename(out / f"report.previous-{time.time_ns()}.json")
    sources = ([ROOT / "rtl/asic/src/sram_dp_sky130_banked.sv"] if bank else
               [SRC / item for item in BLOCKS[args.block]])
    report = {"schema_version": 1, "stage": "mapped", "block": args.block,
              "pdk": args.pdk, "corner": args.corner,
              "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources},
              "liberty": {"path": str(lib), "sha256": digest(lib)},
              "constraints": {"clock_period_ns": args.clock_period_ns,
                              "abc_delay_ps": args.abc_delay_ps,
                              "io_delay_fraction": args.io_delay_fraction,
                              "output_load_pf": args.output_load_pf,
                              "input_slew": "ideal (no driving cell constraint)"},
              "parameters": ({"FETCH_BUFFER_ENABLE": args.fetch_buffer} if args.block == "fetch_unit" else
                             {"DATA_W": 128, "DEPTH": args.sram_depth} if bank else {}),
              "tools": {"sv2v": version(["sv2v", "--version"]),
                        "yosys": version(["yosys", "-V"])},
              "unavailable": ["routed parasitics", "DRC/LVS", "power/activity",
                              "full-chip memory coverage", "post-layout achieved frequency"]}
    if args.docker_image:
        identity = json.loads(subprocess.check_output(
            ["docker", "image", "inspect", args.docker_image], timeout=30))[0]
        report["tools"]["sta_container_resolved_id"] = identity["Id"]
    converted = out / "source.v"
    report["sv2v"] = run(["sv2v", "-DSFU_SYNTH_NO_DPI", *(["-DSYNTHESIS"] if bank else []), f"--top={args.block}",
                          "-I", str(SRC / "include"), *map(str, sources),
                          "-w", str(converted)], out / "sv2v.log", args.timeout_seconds)
    params = (f"chparam -set FETCH_BUFFER_ENABLE {args.fetch_buffer} {args.block};"
              if args.block == "fetch_unit" else
              f"chparam -set DATA_W 128 -set DEPTH {args.sram_depth} {args.block};" if bank else "")
    script = out / "map.ys"
    macro_read = f"read_liberty -lib {quoted(macro_lib)}\n" if bank else ""
    macro_stat = f" -liberty {quoted(macro_lib)}" if bank else ""
    script.write_text(f"""{macro_read}read_verilog {quoted(converted)}
{params}
hierarchy -check -top {args.block}
synth -top {args.block} -flatten -noabc
dfflibmap -liberty {quoted(lib)}
abc -liberty {quoted(lib)} -D {args.abc_delay_ps}
clean
# Teach structural checks the mapped standard cells' port directions.
read_liberty -lib {quoted(lib)}
check -assert
select -assert-none t:$_*
stat -liberty {quoted(lib)}{macro_stat}
write_verilog -noattr -noexpr {quoted(out / 'mapped.v')}
write_json {quoted(out / 'mapped.json')}
""")
    report["mapping"] = run(["yosys", "-s", str(script)], out / "map.log", args.timeout_seconds)
    log = (out / "map.log").read_text()
    areas = re.findall(r"Chip area[^:]*:\s*([0-9.]+)", log)
    if not areas:
        raise RuntimeError("mapped cell area unavailable; inspect map.log")
    mapped = json.loads((out / "mapped.json").read_text())
    counts = Counter(cell["type"] for cell in mapped["modules"][args.block]["cells"].values())
    used = set(counts)
    report["mapped_instance_counts"] = dict(sorted(counts.items()))
    report["standard_cell_area_liberty_units"] = float(areas[-1])
    if bank:
        macro_count = counts.pop(macro["cell"], 0)
        expected_count = args.sram_depth // 512 * 4
        if macro_count != expected_count:
            raise RuntimeError(f"bank structure mismatch: {macro_count} macros, expected {expected_count}")
        used = set(counts)
        std_areas = liberty_areas(lib_text)
        if used - std_areas.keys():
            raise RuntimeError("unmapped/unknown bank peripheral cells")
        if any(not name.startswith("sky130_fd_sc_hd__") for name in used):
            raise RuntimeError("bank contains non-SKY130HD peripheral cells")
        std_area = sum(count * std_areas[name] for name, count in counts.items())
        nominal_total = std_area + macro_count * macro["liberty_area_per_instance"]
        if abs(float(areas[-1]) - nominal_total) > max(1e-3, nominal_total * 1e-6):
            raise RuntimeError("mapped area statistic does not match the cell/macro inventory")
        report["standard_cell_area_liberty_units"] = std_area
        report["mapped_cell_and_macro_area_liberty_units"] = float(areas[-1])
        macro.update(instances=macro_count, logical_capacity_bytes=args.sram_depth * 16)
        report["macro"] = macro
        report["macro_area_um2"] = macro_count * macro["lef_area_um2_per_instance"]
    report["area_units_verified"] = False
    if args.cell_lef:
        lef = args.cell_lef.resolve(strict=True)
        lef_sizes = lef_areas(lef.read_text())
        lib_areas = liberty_areas(lib_text)
        if not used:
            raise RuntimeError("no mapped library instances found for LEF area verification")
        for cell in used:
            if cell not in lef_sizes or cell not in lib_areas or abs(lef_sizes[cell] - lib_areas[cell]) > max(1e-4, lib_areas[cell] * 1e-5):
                raise RuntimeError(f"missing/mismatched physical LEF area for mapped cell {cell}")
        report["cell_lef"] = {"path": str(lef), "sha256": digest(lef), "verified_cell_types": len(used)}
        report["area_units_verified"] = True
        report["standard_cell_area_um2"] = report["standard_cell_area_liberty_units"]
        if bank:
            report["total_cell_and_macro_area_um2"] = (report["standard_cell_area_um2"] +
                                                       report["macro_area_um2"])
            report["area_note"] = "standard-cell instance area + macro LEF footprint; not floorplan or die area"
    report["netlist_sha256"] = digest(out / "mapped.v")
    if args.sta:
        if args.docker_image:
            # Bind only this project. Refuse silently inaccessible external libs/output.
            def container(p: Path) -> Path:
                return Path("/work") / p.relative_to(ROOT)
            slib, netlist = container(lib), container(out / "mapped.v")
        else:
            slib, netlist = lib, out / "mapped.v"
        sta = out / "sta.tcl"
        sta.write_text(f"""read_liberty {quoted(slib)}
read_verilog {quoted(netlist)}
link_design {args.block}
create_clock -period {args.clock_period_ns} -name clk [get_ports clk]
set_input_delay -clock clk {args.io_delay_fraction * args.clock_period_ns} [lsearch -inline -all -not -exact [all_inputs] [get_ports clk]]
set_output_delay -clock clk {args.io_delay_fraction * args.clock_period_ns} [all_outputs]
set_load {args.output_load_pf} [all_outputs]
report_clock_min_period
report_wns
report_tns
report_checks -path_delay max -format full_clock_expanded -group_path_count 3
report_checks -unconstrained
puts "REGISTER_TO_REGISTER_PATHS"
report_checks -from [all_registers -clock clk -output_pins] -to [all_registers -clock clk -data_pins] -path_delay max -group_path_count 1 -digits 6
report_clock_min_period
exit
""")
        if args.docker_image:
            command = ["docker", "run", "--rm", "--pull=never", "--platform", "linux/amd64",
                       "--name", "taccel-block-sta-" + hashlib.sha256(str(out).encode()).hexdigest()[:12],
                       "--memory", "8g", "--cpus", "2", "--mount", f"type=bind,source={ROOT},target=/work",
                       "--entrypoint", "/usr/bin/time", args.docker_image,
                       "-v", args.docker_sta_path, "-no_splash", "-exit", str(container(sta))]
            report["tools"]["sta_container_image"] = args.docker_image
        else:
            command = ["sta", "-no_splash", "-exit", str(sta)]
            report["tools"]["sta"] = version(["sta", "-version"])
        report["sta"] = run(command, out / "sta.log", args.timeout_seconds)
        sta_log = (out / "sta.log").read_text()
        if re.search(r"(^|\n)Error:", sta_log):
            raise RuntimeError("OpenSTA reported errors; see sta.log")
        periods = re.findall(r"period_min\s*=\s*([0-9.]+)", sta_log)
        reported_period = float(periods[-1]) if periods else None
        report["tool_reported_clock_min_period_ns"] = reported_period
        reg_paths = (sta_log.split("REGISTER_TO_REGISTER_PATHS", 1)[1]
                     if "REGISTER_TO_REGISTER_PATHS" in sta_log else "")
        slacks = re.findall(r"([-+0-9.]+)\s+slack \((?:MET|VIOLATED)\)", reg_paths)
        report["register_to_register_setup_bound_ns_zero_wire"] = (
            args.clock_period_ns - min(map(float, slacks)) if slacks else None)
        report["setup_bound_method"] = "single-clock period minus worst register-to-register setup slack; no IO/extracted-wire bound"
        if not slacks:
            report["unavailable"].append("register-to-register setup path")
        if args.docker_image:
            inner_rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", sta_log)
            report["sta"]["container_peak_rss_bytes"] = int(inner_rss[1]) * 1024 if inner_rss else None
            report["sta"]["launcher_peak_rss_bytes"] = report["sta"].pop("peak_rss_bytes")
        report["stage"] = "mapped_zero_wire_sta"
    report["disk_bytes"] = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
