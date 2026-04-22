"""Smoke test for the Stage 4 nanoGPT runner."""
import importlib.util
import os
import subprocess
from pathlib import Path


TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "train_tiny_fixture.py"
RUNNER = Path(__file__).resolve().parents[1] / "run_nanogpt.py"


def _fixture_tool():
    spec = importlib.util.spec_from_file_location("train_tiny_fixture", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_nanogpt_stage4_fixture_smoke(tmp_path):
    tool = _fixture_tool()
    checkpoint = tmp_path / "nanogpt_shakespeare_char_d384_l6.pt"
    metadata = checkpoint.with_suffix(checkpoint.suffix + ".json")
    tool.write_stage4_fixture(checkpoint, metadata)

    env = os.environ.copy()
    env["PYTHONPATH"] = "software"
    result = subprocess.run(
        [
            "/tmp/llm_accelerator_stage3_venv/bin/python",
            str(RUNNER),
            str(checkpoint),
            "--prompt-ids",
            "0",
            "--max-new-tokens",
            "2",
            "--compare-fp32",
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "generated_ids" in result.stdout
    assert "min_cosine" in result.stdout
    assert "fp32_generated_ids" in result.stdout
    assert "fp32_min_top5_overlap" in result.stdout
