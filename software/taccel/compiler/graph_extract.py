"""Extract IR graph from DeiT-tiny architecture (hard-coded, not traced)."""
from .ir import IRNode, IRGraph

# DeiT-tiny constants
EMBED_DIM = 192
DEPTH = 12
NUM_HEADS = 3
HEAD_DIM = 64
MLP_RATIO = 4
MLP_DIM = EMBED_DIM * MLP_RATIO  # 768
SEQ_LEN = 197  # 196 patches + 1 CLS
PATCH_SIZE = 16
IMAGE_SIZE = 224
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 196
PATCH_DIM = 3 * PATCH_SIZE * PATCH_SIZE  # 768
NUM_CLASSES = 1000

# Padded dimensions (multiple of 16)
SEQ_LEN_PAD = 208  # 13 * 16


def extract_deit_tiny() -> IRGraph:
    """Build IR graph for DeiT-tiny-patch16-224."""
    graph = IRGraph()

    # Patch embedding is done by the host (CPU pre-processing) before invoking the accelerator.
    # The program starts with [196, 192] INT8 embedded patches already in ABUF[row 1..].
    # CLS prepend: load cls_token from DRAM → ABUF[0], embedded patches at ABUF[row 1]
    graph.add_node(IRNode(
        op="cls_prepend", name="cls_prepend",
        inputs=["embedded_patches", "vit.embeddings.cls_token"],
        output_shape=(SEQ_LEN, EMBED_DIM),
    ))

    # Add position embedding: [197, 192] + [197, 192]
    graph.add_node(IRNode(
        op="pos_embed_add", name="pos_embed_add",
        inputs=["cls_prepend", "vit.embeddings.position_embeddings"],
        output_shape=(SEQ_LEN, EMBED_DIM),
    ))

    prev_output = "pos_embed_add"

    # --- Transformer Blocks ---
    for block_idx in range(DEPTH):
        prefix = f"vit.encoder.layer.{block_idx}"
        b = f"block{block_idx}"

        # LayerNorm 1
        ln1_name = f"{b}_ln1"
        graph.add_node(IRNode(
            op="layernorm", name=ln1_name,
            inputs=[prev_output,
                    f"{prefix}.layernorm_before.weight",
                    f"{prefix}.layernorm_before.bias"],
            output_shape=(SEQ_LEN, EMBED_DIM),
        ))

        # Per-head Q, K, V projections interleaved with attention computation.
        # Compute Q/K/V for one head, run its attention, free, then next head.
        # This keeps only one head's Q/K/V live at a time in ABUF.
        for h in range(NUM_HEADS):
            # Q, K, V projections for head h
            for proj in ["query", "key", "value"]:
                graph.add_node(IRNode(
                    op="matmul", name=f"{b}_head{h}_{proj}",
                    inputs=[ln1_name,
                            f"{prefix}.attention.attention.{proj}.weight_h{h}"],
                    output_shape=(SEQ_LEN, HEAD_DIM),
                    attrs={"bias": f"{prefix}.attention.attention.{proj}.bias_h{h}"},
                ))

            # Q_h @ K_h^T → [197, 197]
            graph.add_node(IRNode(
                op="matmul_qkt", name=f"{b}_head{h}_qkt",
                inputs=[f"{b}_head{h}_query", f"{b}_head{h}_key"],
                output_shape=(SEQ_LEN, SEQ_LEN),
                attrs={"head_idx": h, "transpose_b": True},
            ))

            # Scale by 1/sqrt(d_head) = 0.125
            graph.add_node(IRNode(
                op="scale_mul", name=f"{b}_head{h}_scale",
                inputs=[f"{b}_head{h}_qkt"],
                output_shape=(SEQ_LEN, SEQ_LEN),
                attrs={"scale": 0.125},
            ))

            # Softmax
            graph.add_node(IRNode(
                op="softmax", name=f"{b}_head{h}_softmax",
                inputs=[f"{b}_head{h}_scale"],
                output_shape=(SEQ_LEN, SEQ_LEN),
            ))

            # Attn @ V_h → [197, 64]
            graph.add_node(IRNode(
                op="matmul_attn_v", name=f"{b}_head{h}_attn_v",
                inputs=[f"{b}_head{h}_softmax", f"{b}_head{h}_value"],
                output_shape=(SEQ_LEN, HEAD_DIM),
                attrs={"head_idx": h},
            ))

        # Concat heads: [3, 197, 64] → [197, 192]
        graph.add_node(IRNode(
            op="concat_heads", name=f"{b}_concat",
            inputs=[f"{b}_head{h}_attn_v" for h in range(NUM_HEADS)],
            output_shape=(SEQ_LEN, EMBED_DIM),
        ))

        # Output projection
        graph.add_node(IRNode(
            op="matmul", name=f"{b}_out_proj",
            inputs=[f"{b}_concat", f"{prefix}.attention.output.dense.weight"],
            output_shape=(SEQ_LEN, EMBED_DIM),
            attrs={"bias": f"{prefix}.attention.output.dense.bias"},
        ))

        # Residual add 1
        graph.add_node(IRNode(
            op="vadd", name=f"{b}_residual1",
            inputs=[f"{b}_out_proj", prev_output],
            output_shape=(SEQ_LEN, EMBED_DIM),
        ))

        # LayerNorm 2
        ln2_name = f"{b}_ln2"
        graph.add_node(IRNode(
            op="layernorm", name=ln2_name,
            inputs=[f"{b}_residual1",
                    f"{prefix}.layernorm_after.weight",
                    f"{prefix}.layernorm_after.bias"],
            output_shape=(SEQ_LEN, EMBED_DIM),
        ))

        # MLP: FC1 (strip-mined; GELU is applied inline per strip)
        graph.add_node(IRNode(
            op="matmul", name=f"{b}_fc1",
            inputs=[ln2_name, f"{prefix}.intermediate.dense.weight"],
            output_shape=(SEQ_LEN, MLP_DIM),
            attrs={"bias": f"{prefix}.intermediate.dense.bias",
                   "strip_mine": True,
                   "inline_gelu": f"{b}_gelu"},
        ))

        # GELU — no-op at codegen time (handled inline in FC1 strip loop)
        graph.add_node(IRNode(
            op="gelu", name=f"{b}_gelu",
            inputs=[f"{b}_fc1"],
            output_shape=(SEQ_LEN, MLP_DIM),
            attrs={"inline_with": f"{b}_fc1"},
        ))

        # MLP: FC2
        graph.add_node(IRNode(
            op="matmul", name=f"{b}_fc2",
            inputs=[f"{b}_gelu", f"{prefix}.output.dense.weight"],
            output_shape=(SEQ_LEN, EMBED_DIM),
            attrs={"bias": f"{prefix}.output.dense.bias",
                   "strip_mine": True},
        ))

        # Residual add 2
        graph.add_node(IRNode(
            op="vadd", name=f"{b}_residual2",
            inputs=[f"{b}_fc2", f"{b}_residual1"],
            output_shape=(SEQ_LEN, EMBED_DIM),
        ))

        prev_output = f"{b}_residual2"

    # --- Final LayerNorm ---
    graph.add_node(IRNode(
        op="layernorm", name="final_ln",
        inputs=[prev_output, "vit.layernorm.weight", "vit.layernorm.bias"],
        output_shape=(SEQ_LEN, EMBED_DIM),
    ))

    # --- CLS token extraction ---
    graph.add_node(IRNode(
        op="cls_extract", name="cls_extract",
        inputs=["final_ln"],
        output_shape=(1, EMBED_DIM),
        attrs={"comment": "Extract row 0 (CLS token)"},
    ))

    # --- Classifier head ---
    graph.add_node(IRNode(
        op="matmul", name="classifier",
        inputs=["cls_extract", "classifier.weight"],
        output_shape=(1, NUM_CLASSES),
        attrs={"bias": "classifier.bias"},
    ))

    return graph
