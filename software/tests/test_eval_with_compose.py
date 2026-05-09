"""Smoke test for the SmoothQuant → GPTQ → bias-correction composition driver.

Verifies that ``software/tools/eval_with_compose.py`` runs end-to-end on the
real GPT-2 nanoGPT fixture under a microscopic eval budget. The driver itself
just chains existing apply-functions; this test catches plumbing regressions
(e.g., import path drift, CLI flag breakage, state-dict layout assumptions)
before they slip into a Phase-2 long run.

Skipped when the converted GPT-2 fixture or tokenizer is missing, mirroring
the existing perplexity smoke in ``test_run_gpt2.py``.
"""
import json
import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_TOOL = REPO_ROOT / "software" / "tools" / "eval_with_compose.py"
GPT2_FIXTURE = REPO_ROOT / "software" / "tests" / "fixtures" / "generated" / "gpt2_converted_nanogpt.pt"
TOKENIZER_DIR = REPO_ROOT / "software" / "tests" / "fixtures" / "generated" / "hf_gpt2"


def _runner_env():
    env = os.environ.copy()
    env["PYTHONPATH"] = "software"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("KMP_INIT_AT_FORK", "FALSE")
    return env


def _run(args):
    result = subprocess.run(
        ["/home/user/LLM_accelerator/.venv/bin/python", str(COMPOSE_TOOL), *args],
        cwd=str(REPO_ROOT),
        env=_runner_env(),
        text=True,
        capture_output=True,
    )
    if result.returncode and "OMP: Error #179" in result.stderr:
        pytest.skip(f"local OpenMP shared-memory blocker: {result.stderr.strip()}")
    if result.returncode:
        # Surface tool stdout/stderr in the assertion to make CI failures
        # actionable.
        print("STDOUT:\n" + result.stdout)
        print("STDERR:\n" + result.stderr)
    result.check_returncode()
    return result


def _check_fixtures():
    if not GPT2_FIXTURE.exists() or not TOKENIZER_DIR.exists():
        pytest.skip(
            "converted GPT-2 fixture/tokenizer missing; run the HF converter "
            "before the eval_with_compose smoke"
        )


def _write_inputs(tmp_path):
    calibration = tmp_path / "calibration.txt"
    evaluation = tmp_path / "eval.txt"
    calibration.write_text("The quick brown fox jumps over the lazy dog.\n", encoding="utf-8")
    evaluation.write_text("The quick brown fox jumps again.\n", encoding="utf-8")
    return calibration, evaluation


def _common_args(calibration, evaluation):
    """Microscopic eval budget that touches every plumbing path without burning
    minutes on the small CI machine."""
    return [
        str(GPT2_FIXTURE),
        "--tokenizer-dir", str(TOKENIZER_DIR),
        "--calibration-text", str(calibration),
        "--eval-text", str(evaluation),
        "--max-eval-tokens", "3",
        "--context-len", "2",
        "--calibration-n-seqs", "1",
        "--calibration-seq-len", "8",
        "--ptq-preset", "control",
        "--json",
    ]


def _parse_compose_json(stdout: str) -> dict:
    """The driver prints log lines, then a single JSON document when --json
    is set. Find the first `{` (which begins the JSON object) and decode
    from there to end of stdout."""
    json_start = stdout.find("{")
    assert json_start != -1, f"no JSON found in stdout:\n{stdout}"
    return json.loads(stdout[json_start:])


def test_compose_smoke_no_steps(tmp_path):
    """Driver with all three flag groups empty must run an unmodified eval."""
    _check_fixtures()
    calibration, evaluation = _write_inputs(tmp_path)
    result = _run(_common_args(calibration, evaluation))
    payload = _parse_compose_json(result.stdout)
    assert "golden_perplexity" in payload
    assert "fake_quant_perplexity" in payload
    assert "relative_delta" in payload
    assert payload["compose"]["sq"] is None
    assert payload["compose"]["gptq"] is None
    assert payload["compose"]["bc"] is None


def test_compose_smoke_gptq_only(tmp_path):
    """GPTQ on block 11 mlp.c_proj must run end-to-end via the driver."""
    _check_fixtures()
    calibration, evaluation = _write_inputs(tmp_path)
    args = _common_args(calibration, evaluation) + [
        "--gptq-blocks", "11",
        "--gptq-types", "mlp.c_proj",
        "--gptq-n-seqs", "1",
        "--gptq-seq-len", "8",
        "--sequential",
    ]
    result = _run(args)
    payload = _parse_compose_json(result.stdout)
    assert "golden_perplexity" in payload
    assert payload["compose"]["gptq"]["blocks"] == [11]
    assert payload["compose"]["gptq"]["types"] == ["mlp.c_proj"]
    assert payload["compose"]["gptq"]["sequential"] is True


def test_compose_smoke_gptq_plus_bc(tmp_path):
    """GPTQ → bias-correction composition must run end-to-end via the driver."""
    _check_fixtures()
    calibration, evaluation = _write_inputs(tmp_path)
    args = _common_args(calibration, evaluation) + [
        "--gptq-blocks", "11",
        "--gptq-types", "mlp.c_proj",
        "--gptq-n-seqs", "1",
        "--gptq-seq-len", "8",
        "--sequential",
        "--bc-blocks", "11",
        "--bc-types", "mlp.c_proj",
        "--bc-search-n-seqs", "1",
        "--bc-search-seq-len", "8",
    ]
    result = _run(args)
    payload = _parse_compose_json(result.stdout)
    assert "golden_perplexity" in payload
    assert payload["compose"]["gptq"]["blocks"] == [11]
    assert payload["compose"]["bc"]["blocks"] == [11]
    assert len(payload["compose"]["bc"]["reports"]) == 1


def test_compose_no_gptq_preset_variant_registered():
    """A GPTQ-companion preset variant was *not* added — the 512-token
    confirmation flipped the 256-tok win (10690→6147 became 7693→9213). The
    composition driver itself ships with the default preset; this test pins
    that no GPTQ-specific preset name was registered, so a future reader
    won't be misled into thinking the recipe was promoted."""
    import sys
    sys.path.insert(0, str(REPO_ROOT / "software"))
    from taccel.runtime.stage5_ptq import STAGE5_PTQ_PRESETS
    assert "output_aware_mlp_lm_head_0_11_pc_full_compose_block11" not in STAGE5_PTQ_PRESETS, (
        "the GPTQ-block-11 preset was rolled back; see .gptq_runs/FINAL_RESULTS.md "
        "for the 512-tok flip that invalidated the 256-tok win"
    )
