"""Shared pytest fixtures for the software test suite."""
from __future__ import annotations

import pytest

from taccel.runtime.fake_quant_reference import clear_weight_component_cache


@pytest.fixture(autouse=True)
def _clear_weight_component_cache():
    clear_weight_component_cache()
    yield
    clear_weight_component_cache()
