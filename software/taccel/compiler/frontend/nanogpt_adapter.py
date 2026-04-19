"""Direct Stage 1 frontend for Karpathy-style nanoGPT modules."""
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from . import FrontendResult
from ..ir import IRGraph, IRNode
from ..model_config import ModelConfig


@dataclass(frozen=True)
class NanoGPTShape:
    n_layer: int
    n_head: int
    d_model: int
    d_head: int
    mlp_dim: int
    vocab_size: int
    max_seq_len: int
    norm_epsilon: float
    bias: bool


def _config_value(config: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(config, Mapping) and name in config:
            return config[name]
        if hasattr(config, name):
            return getattr(config, name)
    if default is not None:
        return default
    raise ValueError(f"nanoGPT config is missing one of: {', '.join(names)}")


def _coerce_shape(config: Any) -> NanoGPTShape:
    n_layer = int(_config_value(config, "n_layer"))
    n_head = int(_config_value(config, "n_head"))
    d_model = int(_config_value(config, "n_embd", "d_model"))
    if d_model % n_head != 0:
        raise ValueError("nanoGPT n_embd/d_model must be divisible by n_head")
    d_head = d_model // n_head
    return NanoGPTShape(
        n_layer=n_layer,
        n_head=n_head,
        d_model=d_model,
        d_head=d_head,
        mlp_dim=int(_config_value(config, "mlp_dim", default=4 * d_model)),
        vocab_size=int(_config_value(config, "vocab_size")),
        max_seq_len=int(_config_value(config, "block_size", "max_seq_len")),
        norm_epsilon=float(_config_value(config, "layer_norm_epsilon", "norm_epsilon", default=1e-5)),
        bias=bool(_config_value(config, "bias", default=True)),
    )


def _model_config(shape: NanoGPTShape) -> ModelConfig:
    return ModelConfig(
        name="nanogpt",
        model_kind="decoder",
        n_layer=shape.n_layer,
        n_head=shape.n_head,
        d_model=shape.d_model,
        d_head=shape.d_head,
        mlp_dim=shape.mlp_dim,
        vocab_size=shape.vocab_size,
        max_seq_len=shape.max_seq_len,
        embedding_kind="token_pos",
        norm_epsilon=shape.norm_epsilon,
        activation_scale_policy="single_set_unified",
        weight_name_map={
            "token_embedding": "transformer.wte.weight",
            "position_embedding": "transformer.wpe.weight",
            "final_layernorm_weight": "transformer.ln_f.weight",
            "final_layernorm_bias": "transformer.ln_f.bias",
            "lm_head": "lm_head.weight",
        },
    )


def _add(graph: IRGraph, op: str, name: str, inputs, shape, **attrs) -> str:
    graph.add_node(IRNode(op=op, name=name, inputs=list(inputs), output_shape=tuple(shape), attrs=attrs))
    return name


def _emit_embeddings(graph: IRGraph, seq_len: int, shape: NanoGPTShape) -> str:
    tok = _add(
        graph,
        "embed_lookup",
        "tok_embed",
        [],
        (seq_len, shape.d_model),
        table="transformer.wte.weight",
        token_ids=[0] * seq_len,
        seq_len=seq_len,
    )
    pos = _add(
        graph,
        "pos_embed_lookup",
        "pos_embed",
        [],
        (seq_len, shape.d_model),
        table="transformer.wpe.weight",
        position_ids=list(range(seq_len)),
        position_start=0,
        seq_len=seq_len,
    )
    return _add(graph, "vadd", "tok_pos_add", [tok, pos], (seq_len, shape.d_model))


def _emit_mlp_block(graph: IRGraph, prev: str, block_idx: int, seq_len: int,
                    shape: NanoGPTShape, *, include_ln1: bool) -> str:
    current = prev
    if include_ln1:
        current = _add(
            graph,
            "layernorm",
            f"block{block_idx}_ln1",
            [current, f"transformer.h.{block_idx}.ln_1.weight", f"transformer.h.{block_idx}.ln_1.bias"],
            (seq_len, shape.d_model),
            block_idx=block_idx,
            epsilon=shape.norm_epsilon,
        )
    ln2 = _add(
        graph,
        "layernorm",
        f"block{block_idx}_ln2",
        [current, f"transformer.h.{block_idx}.ln_2.weight", f"transformer.h.{block_idx}.ln_2.bias"],
        (seq_len, shape.d_model),
        block_idx=block_idx,
        epsilon=shape.norm_epsilon,
    )
    fc1 = _add(
        graph,
        "matmul",
        f"block{block_idx}_fc1",
        [ln2, f"transformer.h.{block_idx}.mlp.c_fc.weight"],
        (seq_len, shape.mlp_dim),
        weight_name=f"transformer.h.{block_idx}.mlp.c_fc.weight",
        bias=f"transformer.h.{block_idx}.mlp.c_fc.bias" if shape.bias else None,
    )
    gelu = _add(graph, "gelu", f"block{block_idx}_gelu", [fc1], (seq_len, shape.mlp_dim), block_idx=block_idx)
    fc2 = _add(
        graph,
        "matmul",
        f"block{block_idx}_fc2",
        [gelu, f"transformer.h.{block_idx}.mlp.c_proj.weight"],
        (seq_len, shape.d_model),
        weight_name=f"transformer.h.{block_idx}.mlp.c_proj.weight",
        bias=f"transformer.h.{block_idx}.mlp.c_proj.bias" if shape.bias else None,
    )
    return _add(graph, "vadd", f"block{block_idx}_residual2", [prev, fc2], (seq_len, shape.d_model))


def _emit_attention_block(graph: IRGraph, prev: str, block_idx: int, seq_len: int,
                          shape: NanoGPTShape) -> str:
    ln1 = _add(
        graph,
        "layernorm",
        f"block{block_idx}_ln1",
        [prev, f"transformer.h.{block_idx}.ln_1.weight", f"transformer.h.{block_idx}.ln_1.bias"],
        (seq_len, shape.d_model),
        block_idx=block_idx,
        epsilon=shape.norm_epsilon,
    )
    head_outputs = []
    for head_idx in range(shape.n_head):
        q_weight = f"transformer.h.{block_idx}.attn.c_attn.weight_h{head_idx}_query"
        k_weight = f"transformer.h.{block_idx}.attn.c_attn.weight_h{head_idx}_key"
        v_weight = f"transformer.h.{block_idx}.attn.c_attn.weight_h{head_idx}_value"
        q = _add(
            graph,
            "matmul",
            f"block{block_idx}_head{head_idx}_query",
            [ln1, q_weight],
            (seq_len, shape.d_head),
            block_idx=block_idx,
            head_idx=head_idx,
            projection="query",
            weight_name=q_weight,
        )
        k = _add(
            graph,
            "matmul",
            f"block{block_idx}_head{head_idx}_key",
            [ln1, k_weight],
            (seq_len, shape.d_head),
            block_idx=block_idx,
            head_idx=head_idx,
            projection="key",
            weight_name=k_weight,
        )
        v = _add(
            graph,
            "matmul",
            f"block{block_idx}_head{head_idx}_value",
            [ln1, v_weight],
            (seq_len, shape.d_head),
            block_idx=block_idx,
            head_idx=head_idx,
            projection="value",
            weight_name=v_weight,
        )
        qkt = _add(
            graph,
            "matmul_qkt",
            f"block{block_idx}_head{head_idx}_qkt",
            [q, k],
            (seq_len, seq_len),
            block_idx=block_idx,
            head_idx=head_idx,
            masked=True,
        )
        scaled = _add(
            graph,
            "scale_mul",
            f"block{block_idx}_head{head_idx}_scale",
            [qkt],
            (seq_len, seq_len),
            scale=shape.d_head ** -0.5,
        )
        softmax = _add(
            graph,
            "softmax",
            f"block{block_idx}_head{head_idx}_softmax",
            [scaled],
            (seq_len, seq_len),
            causal_identity=(seq_len == 1),
        )
        head_outputs.append(
            _add(
                graph,
                "matmul_attn_v",
                f"block{block_idx}_head{head_idx}_attn_v",
                [softmax, v],
                (seq_len, shape.d_head),
                block_idx=block_idx,
                head_idx=head_idx,
            )
        )
    concat = _add(graph, "concat_heads", f"block{block_idx}_concat", head_outputs, (seq_len, shape.d_model))
    out_proj = _add(
        graph,
        "matmul",
        f"block{block_idx}_out_proj",
        [concat, f"transformer.h.{block_idx}.attn.c_proj.weight"],
        (seq_len, shape.d_model),
        weight_name=f"transformer.h.{block_idx}.attn.c_proj.weight",
        bias=f"transformer.h.{block_idx}.attn.c_proj.bias" if shape.bias else None,
    )
    return _add(graph, "vadd", f"block{block_idx}_residual1", [prev, out_proj], (seq_len, shape.d_model))


def _finish(graph: IRGraph, prev: str, seq_len: int, shape: NanoGPTShape) -> None:
    ln_f = _add(
        graph,
        "layernorm",
        "ln_f",
        [prev, "transformer.ln_f.weight", "transformer.ln_f.bias"],
        (seq_len, shape.d_model),
        epsilon=shape.norm_epsilon,
    )
    _add(
        graph,
        "matmul",
        "lm_head",
        [ln_f, "lm_head.weight"],
        (seq_len, shape.vocab_size),
        weight_name="lm_head.weight",
        tied_to="transformer.wte.weight",
    )


def _build_graph(shape: NanoGPTShape, variant: str) -> IRGraph:
    if variant == "forward_1token":
        seq_len = 1
        include_attention = True
    elif variant == "non_attention_seq16":
        seq_len = 16
        include_attention = False
    else:
        raise ValueError("variant must be 'forward_1token' or 'non_attention_seq16'")

    graph = IRGraph()
    prev = _emit_embeddings(graph, seq_len, shape)
    for block_idx in range(shape.n_layer):
        if include_attention:
            prev = _emit_attention_block(graph, prev, block_idx, seq_len, shape)
            prev = _emit_mlp_block(graph, prev, block_idx, seq_len, shape, include_ln1=False)
        else:
            prev = _emit_mlp_block(graph, prev, block_idx, seq_len, shape, include_ln1=True)
    _finish(graph, prev, seq_len, shape)
    return graph


def load_nanogpt(*, model: Optional[Any] = None, state_dict: Optional[Mapping[str, Any]] = None,
                 config: Optional[Any] = None, variant: str = "forward_1token") -> FrontendResult:
    """Return a Stage 1 nanoGPT IR graph and ModelConfig.

    The adapter intentionally walks the known nanoGPT config/state_dict shape
    instead of tracing HuggingFace GPT-2. Stage 1 only validates frontend shape
    and graph plumbing; full decoder codegen arrives in later stages.
    """
    del state_dict  # Weight validation is deferred until decoder codegen lands.
    if config is None and model is not None:
        config = getattr(model, "config", None)
    if config is None:
        raise ValueError("load_nanogpt requires a nanoGPT config or model.config")

    shape = _coerce_shape(config)
    return FrontendResult(graph=_build_graph(shape, variant), config=_model_config(shape))
