"""Model configuration shared by compiler frontends and codegen."""
from dataclasses import dataclass, field
from typing import Literal, Mapping


ModelKind = Literal["encoder", "decoder"]
EmbeddingKind = Literal["patch_cls", "token_pos"]
ActivationScalePolicy = Literal[
    "single_set_unified",
    "separate_prefill_decode",
    "per_head_kv",
]


@dataclass
class ModelConfig:
    """Shape and layout metadata for a model graph.

    Stage 1 uses this to remove the implicit DeiT-tiny shape contract from
    compiler/codegen call sites without claiming full decoder compilation yet.
    """

    model_kind: ModelKind
    n_layer: int
    n_head: int
    d_model: int
    d_head: int
    mlp_dim: int
    vocab_size: int
    max_seq_len: int
    embedding_kind: EmbeddingKind
    norm_epsilon: float = 1e-6
    weight_name_map: Mapping[str, str] = field(default_factory=dict)
    activation_scale_policy: ActivationScalePolicy = "single_set_unified"
    name: str = "model"

    def __post_init__(self):
        valid_model_kinds = {"encoder", "decoder"}
        if self.model_kind not in valid_model_kinds:
            raise ValueError(f"model_kind must be one of {sorted(valid_model_kinds)}")

        valid_embedding_kinds = {"patch_cls", "token_pos"}
        if self.embedding_kind not in valid_embedding_kinds:
            raise ValueError(f"embedding_kind must be one of {sorted(valid_embedding_kinds)}")

        valid_scale_policies = {
            "single_set_unified",
            "separate_prefill_decode",
            "per_head_kv",
        }
        if self.activation_scale_policy not in valid_scale_policies:
            raise ValueError(
                "activation_scale_policy must be one of "
                f"{sorted(valid_scale_policies)}"
            )
        if self.activation_scale_policy == "per_head_kv":
            raise NotImplementedError("per_head_kv activation scales are deferred")

        positive_fields = {
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "d_model": self.d_model,
            "d_head": self.d_head,
            "mlp_dim": self.mlp_dim,
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
        }
        for field_name, value in positive_fields.items():
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")

        if self.d_model % 16 != 0:
            raise ValueError("d_model must be a multiple of 16")
        if self.d_head % 16 != 0:
            raise ValueError("d_head must be a multiple of 16")
        if self.d_model != self.n_head * self.d_head:
            raise ValueError("d_model must equal n_head * d_head")
        if self.mlp_dim % 16 != 0:
            raise ValueError("mlp_dim must be a multiple of 16")
        if self.max_seq_len > 4095:
            raise ValueError("max_seq_len must be <= 4095 for ISA v1.1 CONFIG_ATTN")
        if self.model_kind == "decoder" and self.embedding_kind != "token_pos":
            raise ValueError("decoder models require token_pos embeddings")

        # Store a plain dict so downstream code can mutate local copies without
        # depending on the mapping implementation supplied by a frontend.
        if not isinstance(self.weight_name_map, dict):
            self.weight_name_map = dict(self.weight_name_map)


def deit_tiny_config() -> ModelConfig:
    """Return the ModelConfig equivalent of graph_extract's DeiT constants."""
    from . import graph_extract as deit

    return ModelConfig(
        name="deit_tiny_patch16_224",
        model_kind="encoder",
        n_layer=deit.DEPTH,
        n_head=deit.NUM_HEADS,
        d_model=deit.EMBED_DIM,
        d_head=deit.HEAD_DIM,
        mlp_dim=deit.MLP_DIM,
        vocab_size=deit.NUM_CLASSES,
        max_seq_len=deit.SEQ_LEN,
        embedding_kind="patch_cls",
        norm_epsilon=1e-6,
        weight_name_map={},
        activation_scale_policy="single_set_unified",
    )
