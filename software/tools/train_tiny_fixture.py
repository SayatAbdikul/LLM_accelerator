#!/usr/bin/env python3
"""Create and validate the ignored Stage 3 tiny nanoGPT fixture.

The generated checkpoint is intentionally local-only: software/.gitignore
ignores *.pt files, so tests skip unless this tool has created the fixture.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE = (
    REPO_ROOT
    / "software"
    / "tests"
    / "fixtures"
    / "generated"
    / "nanogpt_shakespeare_char_d128_l2.pt"
)
DEFAULT_METADATA = DEFAULT_FIXTURE.with_suffix(DEFAULT_FIXTURE.suffix + ".json")
DEFAULT_STAGE4_FIXTURE = (
    REPO_ROOT
    / "software"
    / "tests"
    / "fixtures"
    / "generated"
    / "nanogpt_shakespeare_char_d384_l6.pt"
)
DEFAULT_STAGE4_METADATA = DEFAULT_STAGE4_FIXTURE.with_suffix(DEFAULT_STAGE4_FIXTURE.suffix + ".json")
SOURCE_SNAPSHOT = "local-stage3-deterministic-export-v1"
STAGE4_SOURCE_SNAPSHOT = "local-stage4-d384-deterministic-export-v1"
SHAKESPEARE_EXCERPT = (
    "First Citizen: Before we proceed any further, hear me speak.\n"
    "All: Speak, speak.\n"
    "First Citizen: You are all resolved rather to die than to famish?\n"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def split_ranges(text: str) -> Dict[str, Tuple[int, int]]:
    n = len(text.encode("utf-8"))
    train_end = int(n * 0.90)
    calib_end = min(train_end, 256)
    eval_start = train_end
    eval_end = n
    return {
        "train_bytes": (0, train_end),
        "calibration_bytes": (0, calib_end),
        "evaluation_bytes": (eval_start, eval_end),
    }


def build_metadata(
    checkpoint_path: Path,
    checkpoint_sha256: str,
    *,
    source_snapshot: str = SOURCE_SNAPSHOT,
    n_layer: int = 2,
    n_head: int = 4,
    d_model: int = 128,
    block_size: int = 128,
    stage4_ready: bool = False,
) -> Dict[str, object]:
    alphabet = "".join(sorted(set(SHAKESPEARE_EXCERPT)))
    ranges = split_ranges(SHAKESPEARE_EXCERPT)
    d_head = d_model // n_head
    return {
        "schema_version": 1,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "source_snapshot": source_snapshot,
        "dataset": "embedded_shakespeare_char_excerpt",
        "tokenizer": {
            "kind": "character",
            "alphabet": alphabet,
            "vocab_size": len(alphabet),
        },
        "seed": 1337,
        "hyperparameters": {
            "n_layer": n_layer,
            "n_head": n_head,
            "d_model": d_model,
            "d_head": d_head,
            "block_size": block_size,
            "bias": True,
        },
        "ranges": {
            key: {"start": start, "end": end}
            for key, (start, end) in ranges.items()
        },
        "sample_count": {
            "calibration": max(1, ranges["calibration_bytes"][1] - ranges["calibration_bytes"][0]),
            "evaluation": max(1, ranges["evaluation_bytes"][1] - ranges["evaluation_bytes"][0]),
        },
        "sequence_length_policy": "contiguous character windows, max block_size tokens",
        "validation_loss": None,
        "stage3e_logits_smoke_ready": True,
        "stage3f_full_graph_smoke_ready": True,
        "stage3c_e2e_ready": True,
        "stage4_d384_ready": bool(stage4_ready),
    }


def _randn_torch(torch, generator, *shape, scale=0.02):
    return torch.randn(*shape, generator=generator, dtype=torch.float32) * float(scale)


def _make_checkpoint_payload(
    *,
    n_layer: int = 2,
    n_head: int = 4,
    d_model: int = 128,
    block_size: int = 128,
    source_snapshot: str = SOURCE_SNAPSHOT,
):
    """Build a deterministic nanoGPT-compatible checkpoint payload."""
    import torch

    vocab = sorted(set(SHAKESPEARE_EXCERPT))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}
    cfg = {
        "n_layer": int(n_layer),
        "n_head": int(n_head),
        "n_embd": int(d_model),
        "block_size": int(block_size),
        "vocab_size": len(vocab),
        "bias": True,
        "layer_norm_epsilon": 1e-5,
    }
    d_model = cfg["n_embd"]
    d_head = d_model // cfg["n_head"]
    mlp_dim = 4 * d_model
    generator = torch.Generator().manual_seed(1337)

    state = {
        "transformer.wte.weight": _randn_torch(torch, generator, cfg["vocab_size"], d_model),
        "transformer.wpe.weight": _randn_torch(torch, generator, cfg["block_size"], d_model),
        "transformer.ln_f.weight": torch.ones(d_model, dtype=torch.float32),
        "transformer.ln_f.bias": torch.zeros(d_model, dtype=torch.float32),
    }
    state["lm_head.weight"] = state["transformer.wte.weight"]

    for layer in range(cfg["n_layer"]):
        state[f"transformer.h.{layer}.ln_1.weight"] = torch.ones(d_model, dtype=torch.float32)
        state[f"transformer.h.{layer}.ln_1.bias"] = torch.zeros(d_model, dtype=torch.float32)
        state[f"transformer.h.{layer}.ln_2.weight"] = torch.ones(d_model, dtype=torch.float32)
        state[f"transformer.h.{layer}.ln_2.bias"] = torch.zeros(d_model, dtype=torch.float32)
        for head in range(cfg["n_head"]):
            for proj in ("query", "key", "value"):
                state[f"transformer.h.{layer}.attn.c_attn.weight_h{head}_{proj}"] = _randn_torch(
                    torch,
                    generator,
                    d_head,
                    d_model,
                )
        state[f"transformer.h.{layer}.attn.c_proj.weight"] = _randn_torch(
            torch, generator, d_model, d_model
        )
        state[f"transformer.h.{layer}.attn.c_proj.bias"] = torch.zeros(d_model, dtype=torch.float32)
        state[f"transformer.h.{layer}.mlp.c_fc.weight"] = _randn_torch(
            torch, generator, mlp_dim, d_model
        )
        state[f"transformer.h.{layer}.mlp.c_fc.bias"] = torch.zeros(mlp_dim, dtype=torch.float32)
        state[f"transformer.h.{layer}.mlp.c_proj.weight"] = _randn_torch(
            torch, generator, d_model, mlp_dim
        )
        state[f"transformer.h.{layer}.mlp.c_proj.bias"] = torch.zeros(d_model, dtype=torch.float32)

    return {
        "source_snapshot": source_snapshot,
        "seed": 1337,
        "model_args": cfg,
        "stoi": stoi,
        "itos": itos,
        "state_dict": state,
        "text": SHAKESPEARE_EXCERPT,
    }


def write_fixture(checkpoint_path: Path = DEFAULT_FIXTURE,
                  metadata_path: Path = DEFAULT_METADATA) -> Dict[str, object]:
    import torch

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_make_checkpoint_payload(), checkpoint_path)
    metadata = build_metadata(checkpoint_path, sha256_file(checkpoint_path))
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def write_stage4_fixture(checkpoint_path: Path = DEFAULT_STAGE4_FIXTURE,
                         metadata_path: Path = DEFAULT_STAGE4_METADATA) -> Dict[str, object]:
    import torch

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _make_checkpoint_payload(
        n_layer=6,
        n_head=6,
        d_model=384,
        block_size=256,
        source_snapshot=STAGE4_SOURCE_SNAPSHOT,
    )
    torch.save(payload, checkpoint_path)
    metadata = build_metadata(
        checkpoint_path,
        sha256_file(checkpoint_path),
        source_snapshot=STAGE4_SOURCE_SNAPSHOT,
        n_layer=6,
        n_head=6,
        d_model=384,
        block_size=256,
        stage4_ready=True,
    )
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def validate_fixture_metadata(checkpoint_path: Path = DEFAULT_FIXTURE,
                              metadata_path: Path = DEFAULT_METADATA) -> Dict[str, object]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    actual = sha256_file(checkpoint_path)
    expected = metadata.get("checkpoint_sha256")
    if actual != expected:
        raise ValueError(f"checkpoint SHA-256 mismatch: metadata={expected}, actual={actual}")
    required = [
        "source_snapshot",
        "seed",
        "hyperparameters",
        "tokenizer",
        "ranges",
        "sample_count",
        "sequence_length_policy",
        "checkpoint_sha256",
    ]
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(f"metadata missing required keys: {missing}")
    return metadata


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--stage4-d384", action="store_true")
    args = parser.parse_args(argv)

    output_path = args.output
    if args.stage4_d384 and output_path == DEFAULT_FIXTURE and args.metadata is None:
        output_path = DEFAULT_STAGE4_FIXTURE
    metadata_path = args.metadata or output_path.with_suffix(output_path.suffix + ".json")
    if args.validate_only:
        validate_fixture_metadata(output_path, metadata_path)
        print(f"fixture metadata valid: {output_path}")
        return 0

    metadata = (
        write_stage4_fixture(output_path, metadata_path)
        if args.stage4_d384 else
        write_fixture(output_path, metadata_path)
    )
    print(f"wrote fixture: {output_path}")
    print(f"wrote metadata: {metadata_path}")
    print(f"sha256: {metadata['checkpoint_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
