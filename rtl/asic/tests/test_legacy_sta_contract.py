"""Bounded legacy STA replay contract; mocked tools never map RTL or run STA."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'build/synth_blocks/run_block_sta.sh'


class LegacyStaContractTests(unittest.TestCase):
    def check_replay(self, top, expected):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'rtl/asic/build/synth_blocks'
            output.mkdir(parents=True)
            (output / f'{top}.v').write_text(f'module {top}; endmodule\n')
            # Capture Yosys's actual -p script. No installed hardware tools
            # can be launched: both commands resolve to these local mocks.
            yosys = root / 'yosys'
            yosys.write_text('#!/bin/sh\nprintf "%s\\n" "$2" > "$MOCK_YOSYS_SCRIPT"\n')
            yosys.chmod(0o755)
            sta = root / 'sta'
            sta.write_text('#!/bin/sh\nprintf "wns max 0.0\\nperiod_min = 1.0\\n"\n')
            sta.chmod(0o755)
            captured = root / 'captured.ys'
            environment = dict(os.environ, PATH=str(root) + os.pathsep + os.environ['PATH'],
                               LIB=str(root / 'fixture.lib'), STA=str(sta),
                               DRIVER_CELL='fixture_driver', DRIVER_PIN='Y',
                               MOCK_YOSYS_SCRIPT=str(captured))
            result = subprocess.run(['bash', str(SCRIPT), top], cwd=root, env=environment,
                                    capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stderr)
            commands = captured.read_text()
            read = commands.index('read_verilog')
            hierarchy = commands.index(f'hierarchy -check -top {top}')
            if expected:
                override = commands.index(expected)
                self.assertLess(read, override)
                self.assertLess(override, hierarchy)
                self.assertEqual(commands.count('chparam '), 1)
            else:
                self.assertNotIn('chparam ', commands)
            self.assertIn('set_driving_cell -lib_cell fixture_driver -pin Y',
                          (output / f'{top}_sta.tcl').read_text())

    def test_sfu_replay_selects_mode_one(self):
        self.check_replay('sfu_engine', 'chparam -set SFU_SYNTH_MODE 1 sfu_engine;')

    def test_helper_replay_selects_mode_one(self):
        self.check_replay('blocking_helper_engine',
                          'chparam -set HELPER_SYNTH_MODE 1 blocking_helper_engine;')

    def test_core_replay_selects_both_mode_one_parameters(self):
        self.check_replay('taccel_top',
                          'chparam -set SFU_SYNTH_MODE 1 -set HELPER_SYNTH_MODE 1 taccel_top;')

    def test_ordinary_leaf_replay_does_not_inject_nonexistent_parameters(self):
        self.check_replay('fp32_div_p6', None)

    def test_legacy_area_labels_are_explicitly_unverified(self):
        for name in ('synth_block.sh', 'synth_full.sh'):
            with self.subTest(script=name):
                script = (SCRIPT.parent / name).read_text()
                area_prints = [line for line in script.splitlines()
                               if line.lstrip().startswith('echo ') and 'area=' in line]
                self.assertEqual(len(area_prints), 1)
                self.assertIn('area=${AREA} unverified_Liberty_units', area_prints[0])
                self.assertNotIn('um^2', area_prints[0])
                self.assertIn('Physical um^2 requires matching LEF verification', script)


if __name__ == '__main__':
    unittest.main()
