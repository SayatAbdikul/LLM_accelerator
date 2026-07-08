"""Direct Stage 1 frontend for Karpathy-style nanoGPT modules."""
from dataclasses import dataclass, replace as _dc_replace
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
    split_qkv_bias: bool
    has_lm_head_bias: bool = False  # True iff `lm_head.bias` is in state_dict
                                     # (created by `fold_layernorm_for_quarot`).


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
        split_qkv_bias=bool(_config_value(config, "split_qkv_bias", default=False)),
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
            bias=f"transformer.h.{block_idx}.attn.c_attn.bias_h{head_idx}_query"
            if shape.split_qkv_bias else None,
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
            bias=f"transformer.h.{block_idx}.attn.c_attn.bias_h{head_idx}_key"
            if shape.split_qkv_bias else None,
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
            bias=f"transformer.h.{block_idx}.attn.c_attn.bias_h{head_idx}_value"
            if shape.split_qkv_bias else None,
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
            scale=shape.d_head ** -0.5,
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
            # `masked` + `key_len` are the M3-C contract additions: in the
            # W8A32 path `emit_softmax_fp32` reads these to decide whether
            # to emit a CONFIG_ATTN before the masked-softmax instruction.
            # The INT8 path keys off `matmul_qkt`'s `masked` attr instead
            # (it emits CONFIG_ATTN per Q strip inside _emit_qkt) and is
            # unaffected by these softmax attrs.
            masked=True,
            key_len=seq_len,
            # `causal_identity` stays — it's a decode-step shortcut that
            # marks softmax of a 1-row Q strip as trivially the identity
            # vector; orthogonal to the masked/key_len attention context.
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
        # `lm_head.bias` is created by `fold_layernorm_for_quarot` (β-fold of
        # ln_f) when `quarot_enabled` is True. It is absent in standard GPT-2.
        # The matmul op accepts an optional `bias` kwarg; passing the key
        # unconditionally and letting the codegen treat missing keys as zero
        # would be cleanest, but the existing op API treats `bias=None` as
        # "no bias add" — so we conditionally wire it based on a hint from
        # the shape config (`shape.has_lm_head_bias`).
        bias=("lm_head.bias" if getattr(shape, "has_lm_head_bias", False) else None),
        tied_to="transformer.wte.weight",
    )


def _build_graph(shape: NanoGPTShape, variant: str) -> IRGraph:
    if variant == "forward_1token":
        seq_len = 1
        include_attention = True
    elif variant == "non_attention_seq16":
        seq_len = 16
        include_attention = False
    elif variant == "forward_batch16":
        # Phase 2 lockstep batched decode (B=16). The 16 query rows share the
        # non-attention path (embeddings/LN/FFN/quant/logits are all M-shaped
        # and batch for free). Attention is emitted in the ordinary *dense*
        # (16, key) form here so the decode graph references the same weights
        # as the single-token prefill graph (they must share one data blob).
        # Step 2b rewrites the KV/attention lowering to be per-stream
        # block-diagonal — each of the 16 streams owns its own K/V cache and
        # cannot share one dense QK^T tile; that transform lives in the KV
        # injection + attention emitters, not the frontend shape.
        seq_len = 16
        include_attention = True
    else:
        raise ValueError(
            "variant must be 'forward_1token', 'non_attention_seq16', "
            "or 'forward_batch16'"
        )

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
    if config is None and model is not None:
        config = getattr(model, "config", None)
    if config is None:
        raise ValueError("load_nanogpt requires a nanoGPT config or model.config")

    shape = _coerce_shape(config)
    # Detect QuaRot's lm_head.bias (created by `fold_layernorm_for_quarot`).
    # When present, the lm_head matmul wires it as its bias input so the
    # codegen produces a bundle that adds the per-vocab β-fold contribution
    # to the logits, matching what `_fp32_forward` and `NanoGPTFQReference`
    # do at the same step.
    has_lm_head_bias = state_dict is not None and "lm_head.bias" in state_dict
    if has_lm_head_bias and not shape.has_lm_head_bias:
        shape = _dc_replace(shape, has_lm_head_bias=True)
    return FrontendResult(graph=_build_graph(shape, variant), config=_model_config(shape))


def _emit_batched_attention_block(graph: IRGraph, prev: str, block_idx: int,
                                  n_streams: int, key_len: int,
                                  shape: NanoGPTShape) -> str:
    """Per-stream block-diagonal attention for lockstep batched decode.

    The Q/K/V projections stay batched (one matmul each over the `n_streams`
    query rows). Attention then runs per stream: each stream extracts its own
    query row, stores its new K/V into its own cache region, loads its own
    cache, and runs the ordinary single-token (query_len=1) QK^T / softmax /
    AV. The 16 per-stream outputs are gathered back into one (n_streams,
    d_head) tile per head for concat_heads. Emits kv_store/kv_load nodes
    directly (with the `stream`/`src_row` attrs) — this graph is complete and
    must NOT be run through `inject_kv_cache_nodes`.
    """
    inv_sqrt = shape.d_head ** -0.5
    ln1 = _add(
        graph, "layernorm", f"block{block_idx}_ln1",
        [prev, f"transformer.h.{block_idx}.ln_1.weight", f"transformer.h.{block_idx}.ln_1.bias"],
        (n_streams, shape.d_model), block_idx=block_idx, epsilon=shape.norm_epsilon,
    )
    head_outputs = []
    for head_idx in range(shape.n_head):
        q_weight = f"transformer.h.{block_idx}.attn.c_attn.weight_h{head_idx}_query"
        k_weight = f"transformer.h.{block_idx}.attn.c_attn.weight_h{head_idx}_key"
        v_weight = f"transformer.h.{block_idx}.attn.c_attn.weight_h{head_idx}_value"
        qb = _add(
            graph, "matmul", f"block{block_idx}_head{head_idx}_query",
            [ln1, q_weight], (n_streams, shape.d_head),
            block_idx=block_idx, head_idx=head_idx, projection="query", weight_name=q_weight,
            bias=f"transformer.h.{block_idx}.attn.c_attn.bias_h{head_idx}_query" if shape.split_qkv_bias else None,
        )
        kb = _add(
            graph, "matmul", f"block{block_idx}_head{head_idx}_key",
            [ln1, k_weight], (n_streams, shape.d_head),
            block_idx=block_idx, head_idx=head_idx, projection="key", weight_name=k_weight,
            bias=f"transformer.h.{block_idx}.attn.c_attn.bias_h{head_idx}_key" if shape.split_qkv_bias else None,
        )
        vb = _add(
            graph, "matmul", f"block{block_idx}_head{head_idx}_value",
            [ln1, v_weight], (n_streams, shape.d_head),
            block_idx=block_idx, head_idx=head_idx, projection="value", weight_name=v_weight,
            bias=f"transformer.h.{block_idx}.attn.c_attn.bias_h{head_idx}_value" if shape.split_qkv_bias else None,
        )
        # Store every stream's new K/V (row s of the batched projection) into
        # its own cache region FIRST, so the batched K/V projection tiles
        # (kb/vb) free before the memory-heavy per-stream attention loop (at
        # ctx-512 each stream's V + v_int8 is ~100 KB — every KB of headroom
        # matters).
        for s in range(n_streams):
            pfx = f"block{block_idx}_head{head_idx}_s{s}"
            _add(graph, "kv_store", f"{pfx}_kstore", [kb], (),
                 layer=block_idx, kind="key", head=head_idx, tokens=1,
                 stream=s, src_row=s, decode=True)
            _add(graph, "kv_store", f"{pfx}_vstore", [vb], (),
                 layer=block_idx, kind="value", head=head_idx, tokens=1,
                 stream=s, src_row=s, decode=True)
        stream_outs = []
        for s in range(n_streams):
            pfx = f"block{block_idx}_head{head_idx}_s{s}"
            # Extract stream s's single query row.
            q_s = _add(graph, "row_copy", f"{pfx}_q", [qb], (1, shape.d_head), src_row=s)
            # Load stream s's own K cache, then QK^T.
            k_s = _add(graph, "kv_load", f"{pfx}_kload", [], (key_len, shape.d_head),
                       layer=block_idx, kind="key", head=head_idx, tokens=key_len,
                       stream=s, decode=True)
            qkt = _add(graph, "matmul_qkt", f"{pfx}_qkt", [q_s, k_s], (1, key_len),
                       block_idx=block_idx, head_idx=head_idx, query_len=1, key_len=key_len,
                       masked=True, runtime_config_attn=True, scale=inv_sqrt)
            scaled = _add(graph, "scale_mul", f"{pfx}_scale", [qkt], (1, key_len),
                          query_len=1, key_len=key_len, scale=inv_sqrt)
            sm = _add(graph, "softmax", f"{pfx}_softmax", [scaled], (1, key_len),
                      query_len=1, key_len=key_len, masked=True, runtime_config_attn=True,
                      causal_identity=True)
            v_s = _add(graph, "kv_load", f"{pfx}_vload", [], (key_len, shape.d_head),
                       layer=block_idx, kind="value", head=head_idx, tokens=key_len,
                       stream=s, decode=True)
            attn_v = _add(graph, "matmul_attn_v", f"{pfx}_attnv", [sm, v_s], (1, shape.d_head),
                          block_idx=block_idx, head_idx=head_idx, query_len=1, key_len=key_len)
            stream_outs.append(attn_v)
        # Gather the 16 per-stream (1, d_head) outputs into one (n_streams, d_head) tile.
        head_outputs.append(_add(
            graph, "gather_rows", f"block{block_idx}_head{head_idx}_attn_out",
            stream_outs, (n_streams, shape.d_head),
        ))
    concat = _add(graph, "concat_heads", f"block{block_idx}_concat", head_outputs,
                  (n_streams, shape.d_model))
    out_proj = _add(
        graph, "matmul", f"block{block_idx}_out_proj",
        [concat, f"transformer.h.{block_idx}.attn.c_proj.weight"], (n_streams, shape.d_model),
        weight_name=f"transformer.h.{block_idx}.attn.c_proj.weight",
        bias=f"transformer.h.{block_idx}.attn.c_proj.bias" if shape.bias else None,
    )
    return _add(graph, "vadd", f"block{block_idx}_residual1", [prev, out_proj],
                (n_streams, shape.d_model))


def build_batched_decode_graph(shape: NanoGPTShape, *, key_len: int, n_streams: int = 16) -> IRGraph:
    """Full lockstep batched-decode IR graph (per-stream attention).

    Complete decode graph — it already carries its KV nodes and attention
    context, so it must NOT be passed through `inject_kv_cache_nodes`.
    """
    graph = IRGraph()
    prev = _emit_embeddings(graph, n_streams, shape)
    for block_idx in range(shape.n_layer):
        prev = _emit_batched_attention_block(graph, prev, block_idx, n_streams, key_len, shape)
        prev = _emit_mlp_block(graph, prev, block_idx, n_streams, shape, include_ln1=False)
    _finish(graph, prev, n_streams, shape)
    return graph


def load_nanogpt_batched_decode(*, config: Any, state_dict: Optional[Mapping[str, Any]] = None,
                                key_len: int, n_streams: int = 16) -> FrontendResult:
    """Return the lockstep batched-decode graph + ModelConfig (Phase 2)."""
    shape = _coerce_shape(config)
    has_lm_head_bias = state_dict is not None and "lm_head.bias" in state_dict
    if has_lm_head_bias and not shape.has_lm_head_bias:
        shape = _dc_replace(shape, has_lm_head_bias=True)
    graph = build_batched_decode_graph(shape, key_len=key_len, n_streams=n_streams)
    return FrontendResult(graph=graph, config=_model_config(shape))
