"""Compiler frontend registry."""
from dataclasses import dataclass
from typing import Callable, Dict

from ..ir import IRGraph
from ..model_config import ModelConfig


@dataclass
class FrontendResult:
    graph: IRGraph
    config: ModelConfig


FrontendFactory = Callable[..., FrontendResult]
_REGISTRY: Dict[str, FrontendFactory] = {}


def register_frontend(name: str, factory: FrontendFactory) -> None:
    if not name:
        raise ValueError("frontend name must be non-empty")
    _REGISTRY[name] = factory


def get_frontend(name: str) -> FrontendFactory:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(f"Unknown frontend '{name}'. Available: {available}") from exc


def load_frontend(name: str, *args, **kwargs) -> FrontendResult:
    return get_frontend(name)(*args, **kwargs)


def registered_frontends() -> Dict[str, FrontendFactory]:
    return dict(_REGISTRY)


from .deit_plugin import load_deit_tiny  # noqa: E402
from .nanogpt_adapter import load_nanogpt  # noqa: E402

register_frontend("deit_tiny", load_deit_tiny)
register_frontend("nanogpt", load_nanogpt)
