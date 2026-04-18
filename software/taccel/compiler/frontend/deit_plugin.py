"""DeiT frontend adapter."""
from . import FrontendResult
from ..graph_extract import extract_deit_tiny
from ..model_config import deit_tiny_config


def load_deit_tiny() -> FrontendResult:
    return FrontendResult(graph=extract_deit_tiny(), config=deit_tiny_config())
