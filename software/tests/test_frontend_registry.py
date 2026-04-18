from taccel.compiler.frontend import load_frontend, registered_frontends
from taccel.compiler.graph_extract import extract_deit_tiny, extract_deit_tiny_with_config
from taccel.compiler.ir import IRGraph


def test_registry_resolves_deit_frontend():
    assert "deit_tiny" in registered_frontends()

    result = load_frontend("deit_tiny")

    assert isinstance(result.graph, IRGraph)
    assert len(result.graph) > 0
    assert result.config.name == "deit_tiny_patch16_224"
    assert result.config.embedding_kind == "patch_cls"


def test_legacy_extract_deit_tiny_returns_graph_only():
    graph = extract_deit_tiny()

    assert isinstance(graph, IRGraph)
    assert len(graph) > 0


def test_deit_graph_with_config_helper():
    graph, config = extract_deit_tiny_with_config()

    assert isinstance(graph, IRGraph)
    assert config.n_layer == 12
    assert config.d_model == 192
