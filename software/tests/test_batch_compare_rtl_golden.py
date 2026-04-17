import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BATCH_SCRIPT = REPO_ROOT / "software" / "tools" / "batch_compare_rtl_golden.py"
BATCH_SPEC = importlib.util.spec_from_file_location("batch_compare_rtl_golden", BATCH_SCRIPT)
batch_compare = importlib.util.module_from_spec(BATCH_SPEC)
BATCH_SPEC.loader.exec_module(batch_compare)


def _result(golden, rtl_logits, runner_rc=0, **rtl_overrides):
    rtl = {
        "status": "halted",
        "fault": False,
        "timeout": False,
        "violations": [],
        **rtl_overrides,
    }
    return batch_compare._compare_logits_with_execution(
        golden_logits=golden,
        rtl_logits=rtl_logits,
        runner_rc=runner_rc,
        rtl=rtl,
    )


def test_batch_compare_clean_runner_matching_logits_passes():
    result = _result([1, 5, 3], [1, 5, 3])

    assert result["execution_ok"] is True
    assert result["raw_logits_exact_match"] is True
    assert result["raw_top1_match"] is True
    assert result["logits_exact_match"] is True
    assert result["top1_match"] is True


def test_batch_compare_nonzero_exit_gates_matching_logits_to_failure():
    result = _result([1, 5, 3], [1, 5, 3], runner_rc=2)

    assert result["execution_ok"] is False
    assert result["raw_logits_exact_match"] is True
    assert result["raw_top1_match"] is True
    assert result["logits_exact_match"] is False
    assert result["top1_match"] is False


def test_batch_compare_fault_timeout_or_violation_gates_matching_logits_to_failure():
    cases = [
        {"fault": True},
        {"timeout": True},
        {"violations": ["cycle_budget_exhausted"]},
        {"status": "timeout"},
    ]

    for overrides in cases:
        result = _result([1, 5, 3], [1, 5, 3], **overrides)
        assert result["execution_ok"] is False
        assert result["raw_logits_exact_match"] is True
        assert result["logits_exact_match"] is False
        assert result["top1_match"] is False


def test_batch_compare_clean_runner_same_top1_mismatching_logits_keeps_top1_only():
    result = _result([1, 5, 3], [0, 5, 2])

    assert result["execution_ok"] is True
    assert result["raw_logits_exact_match"] is False
    assert result["raw_top1_match"] is True
    assert result["logits_exact_match"] is False
    assert result["top1_match"] is True
