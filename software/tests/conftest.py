"""Shared pytest fixtures for the software test suite."""
from __future__ import annotations

import pytest

from taccel.runtime.fake_quant_reference import clear_weight_component_cache


# This profile is intentionally independent of optional model downloads,
# perplexity campaigns, and trained checkpoints. Its tiny generated fixture
# and selected hardware runner are mandatory inputs, not successful skips.
REQUIRED_MODULES = {
    "test_assembler", "test_isa_encoding", "test_w8a16_isa", "test_golden_model",
    "test_w8a16_simulator", "test_host_runner", "test_program_bundle",
    "test_decoder_bundle", "test_embedding_patch_sites", "test_logits_store",
    "test_relocation_patching", "test_runtime_patch_sites", "test_batched_decode",
    "test_compare_rtl_golden", "test_stage5_ptq_presets", "test_tiny_decode_smoke",
    "test_tiny_decode_determinism", "test_verification_contracts",
    "test_compiler_optimizations", "test_optimization_matrix", "test_chat_gpt2",
    "test_batched_av_capacity",
    "test_numerical_quality_gate", "test_w8a16_numerical_reference",
    "test_w8a16_quality_cli", "test_w8a16_rtl_quality_cli",
}


def pytest_addoption(parser):
    parser.addoption("--verification-profile", choices=("developer", "required"),
                     default="developer", help="required selects the release core and fails skipped gates")


def pytest_configure(config):
    config.addinivalue_line("markers", "required: mandatory release-core verification")


def pytest_collection_modifyitems(config, items):
    selected, optional = [], []
    for item in items:
        required = item.path.stem in REQUIRED_MODULES and "124m" not in item.name.lower()
        if required:
            item.add_marker(pytest.mark.required)
        (selected if required else optional).append(item)
    if config.getoption("--verification-profile") == "required":
        config.hook.pytest_deselected(items=optional)
        items[:] = selected


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if (item.config.getoption("--verification-profile") == "required" and
            item.get_closest_marker("required") and report.skipped):
        report.outcome = "failed"
        report.longrepr = f"Mandatory verification gate skipped: {item.nodeid}\n{report.longrepr}"


@pytest.fixture(autouse=True)
def _clear_weight_component_cache():
    clear_weight_component_cache()
    yield
    clear_weight_component_cache()
