#!/usr/bin/env python3
"""Create and validate the ignored Stage 3 tiny nanoGPT fixture.

The generated checkpoint is intentionally local-only: software/.gitignore
ignores *.pt files, so tests skip unless this tool has created the fixture.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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
DEFAULT_TRAINED_D128_FIXTURE = (
    REPO_ROOT
    / "software"
    / "tests"
    / "fixtures"
    / "generated"
    / "nanogpt_shakespeare_char_d128_l2_trained.pt"
)
DEFAULT_TRAINED_D128_METADATA = DEFAULT_TRAINED_D128_FIXTURE.with_suffix(
    DEFAULT_TRAINED_D128_FIXTURE.suffix + ".json"
)
SOURCE_SNAPSHOT = "local-stage3-deterministic-export-v4-qkt-scale-stable-rank"
STAGE4_SOURCE_SNAPSHOT = "local-stage4-d384-deterministic-export-v3-stable-rank"
TRAINED_D128_SOURCE_SNAPSHOT = "local-stage45-trained-d128-v2-lr3e4-attn-quality"
TRAINED_D128_DEFAULT_LR = 3e-4
TRAINED_D128_EMBED_LR_MULTIPLIER = 1.0 / 30.0
TRAINED_QUALITY_MIN_LOSS_RATIO = 0.85
TRAINED_QUALITY_MIN_ENTROPY_DROP = 0.05
TRAINED_QUALITY_MIN_QKT_P95 = 1e-3
SHAKESPEARE_EXCERPT = (
    "First Citizen: Before we proceed any further, hear me speak.\n"
    "All: Speak, speak.\n"
    "First Citizen: You are all resolved rather to die than to famish?\n"
)
TRAINED_CORPUS_REPEATS = 256


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def split_ranges(text: str, *, calibration_limit: int = 256) -> Dict[str, Tuple[int, int]]:
    n = len(text.encode("utf-8"))
    train_end = int(n * 0.90)
    calib_end = min(train_end, calibration_limit)
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
    real_trained_ready: bool = False,
    text: str = SHAKESPEARE_EXCERPT,
    training: Dict[str, object] | None = None,
    validation_loss: float | None = None,
) -> Dict[str, object]:
    alphabet = "".join(sorted(set(text)))
    ranges = split_ranges(text, calibration_limit=2048 if real_trained_ready else 256)
    d_head = d_model // n_head
    metadata = {
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
        "validation_loss": validation_loss,
        "stage3e_logits_smoke_ready": True,
        "stage3f_full_graph_smoke_ready": True,
        "stage3c_e2e_ready": True,
        "stage4_d384_ready": bool(stage4_ready),
        "real_trained_d128_ready": bool(real_trained_ready),
    }
    if training is not None:
        metadata["training"] = training
        metadata["learning_rate"] = training.get("learning_rate")
        metadata["initial_train_loss"] = training.get("initial_train_loss")
        metadata["final_train_loss"] = training.get("final_train_loss", training.get("train_loss"))
        metadata["training_quality"] = training.get("training_quality")
    return metadata


def _randn_torch(torch, generator, *shape, scale=0.02):
    return torch.randn(*shape, generator=generator, dtype=torch.float32) * float(scale)


def _stable_token_embedding(torch, vocab_size: int, d_model: int):
    """Create tied embeddings with large rank margins for FP32-vs-INT8 checks."""
    weight = torch.zeros(vocab_size, d_model, dtype=torch.float32)
    for token_id in range(vocab_size):
        weight[token_id, token_id % d_model] = 1.0
        weight[token_id, d_model - 1] = float(token_id + 1) / float(1000 * vocab_size)
    return weight


def _make_checkpoint_payload(
    *,
    n_layer: int = 2,
    n_head: int = 4,
    d_model: int = 128,
    block_size: int = 128,
    source_snapshot: str = SOURCE_SNAPSHOT,
    weight_scale: float = 0.0,
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
        "transformer.wte.weight": _stable_token_embedding(torch, cfg["vocab_size"], d_model),
        "transformer.wpe.weight": torch.zeros(cfg["block_size"], d_model, dtype=torch.float32),
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
                    scale=weight_scale,
                )
        state[f"transformer.h.{layer}.attn.c_proj.weight"] = _randn_torch(
            torch, generator, d_model, d_model, scale=weight_scale
        )
        state[f"transformer.h.{layer}.attn.c_proj.bias"] = torch.zeros(d_model, dtype=torch.float32)
        state[f"transformer.h.{layer}.mlp.c_fc.weight"] = _randn_torch(
            torch, generator, mlp_dim, d_model, scale=weight_scale
        )
        state[f"transformer.h.{layer}.mlp.c_fc.bias"] = torch.zeros(mlp_dim, dtype=torch.float32)
        state[f"transformer.h.{layer}.mlp.c_proj.weight"] = _randn_torch(
            torch, generator, d_model, mlp_dim, scale=weight_scale
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


def _trained_text() -> str:
    return SHAKESPEARE_EXCERPT * TRAINED_CORPUS_REPEATS


def _tokenize_text(text: str, stoi: Dict[str, int]):
    return [stoi[ch] for ch in text]


def _trained_attention_quality(payload: Dict[str, object], *, seq_len: int = 64) -> Dict[str, object]:
    """Measure whether the trained fixture's attention is meaningfully active."""
    import torch
    import torch.nn.functional as F

    state = payload["state_dict"]
    cfg = payload["model_args"]
    text = str(payload["text"])
    stoi = payload["stoi"]
    ranges = split_ranges(text, calibration_limit=2048)
    eval_text = text.encode("utf-8")[ranges["evaluation_bytes"][0]: ranges["evaluation_bytes"][1]].decode("utf-8")
    token_ids = [int(stoi[ch]) for ch in eval_text if ch in stoi]
    if len(token_ids) < seq_len:
        repeats = (seq_len // max(len(token_ids), 1)) + 1
        token_ids = (token_ids * repeats)[:seq_len]
    else:
        token_ids = token_ids[:seq_len]

    n_layer = int(cfg["n_layer"])
    n_head = int(cfg["n_head"])
    d_model = int(cfg["n_embd"])
    d_head = d_model // n_head
    eps = float(cfg.get("layer_norm_epsilon", 1e-5))

    idx = torch.tensor(token_ids, dtype=torch.long)
    pos = torch.arange(len(token_ids), dtype=torch.long)
    x = state["transformer.wte.weight"][idx] + state["transformer.wpe.weight"][pos]
    x = x.to(dtype=torch.float32).unsqueeze(0)
    mask = torch.triu(torch.ones(len(token_ids), len(token_ids), dtype=torch.bool), diagonal=1)

    best_entropy_drop = -float("inf")
    best_entropy_ratio = float("inf")
    best_qkt_p95 = 0.0
    best_layer = 0
    best_head = 0
    for layer in range(n_layer):
        ln1 = F.layer_norm(
            x,
            (d_model,),
            state[f"transformer.h.{layer}.ln_1.weight"].to(dtype=torch.float32),
            state[f"transformer.h.{layer}.ln_1.bias"].to(dtype=torch.float32),
            eps,
        )
        heads = []
        for head in range(n_head):
            q = ln1 @ state[f"transformer.h.{layer}.attn.c_attn.weight_h{head}_query"].to(dtype=torch.float32).T
            k = ln1 @ state[f"transformer.h.{layer}.attn.c_attn.weight_h{head}_key"].to(dtype=torch.float32).T
            v = ln1 @ state[f"transformer.h.{layer}.attn.c_attn.weight_h{head}_value"].to(dtype=torch.float32).T
            qkt = (q @ k.transpose(-2, -1)) * (d_head ** -0.5)
            qkt_p95 = float(torch.quantile(qkt.abs().reshape(-1), 0.95).detach().cpu())
            masked = qkt.masked_fill(mask.view(1, len(token_ids), len(token_ids)), float("-inf"))
            probs = F.softmax(masked, dim=-1)
            row_entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=-1)[0]
            uniform_entropy = torch.log(torch.arange(1, len(token_ids) + 1, dtype=torch.float32))
            valid = uniform_entropy > 0
            entropy_ratio = float((row_entropy[valid] / uniform_entropy[valid]).mean().detach().cpu())
            entropy_drop = 1.0 - entropy_ratio
            if entropy_drop > best_entropy_drop:
                best_entropy_drop = entropy_drop
                best_entropy_ratio = entropy_ratio
                best_qkt_p95 = qkt_p95
                best_layer = layer
                best_head = head
            heads.append(probs @ v)
        attn_out = torch.cat(heads, dim=-1)
        x = x + attn_out @ state[f"transformer.h.{layer}.attn.c_proj.weight"].to(dtype=torch.float32).T
        ln2 = F.layer_norm(
            x,
            (d_model,),
            state[f"transformer.h.{layer}.ln_2.weight"].to(dtype=torch.float32),
            state[f"transformer.h.{layer}.ln_2.bias"].to(dtype=torch.float32),
            eps,
        )
        fc = ln2 @ state[f"transformer.h.{layer}.mlp.c_fc.weight"].to(dtype=torch.float32).T
        gelu = F.gelu(fc)
        x = x + gelu @ state[f"transformer.h.{layer}.mlp.c_proj.weight"].to(dtype=torch.float32).T

    return {
        "sequence_length": int(len(token_ids)),
        "best_layer": int(best_layer),
        "best_head": int(best_head),
        "best_entropy_ratio": float(best_entropy_ratio),
        "best_entropy_drop": float(best_entropy_drop),
        "best_qkt_p95_abs": float(best_qkt_p95),
        "entropy_drop_threshold": TRAINED_QUALITY_MIN_ENTROPY_DROP,
        "qkt_p95_abs_threshold": TRAINED_QUALITY_MIN_QKT_P95,
    }


def _train_d128_payload(
    *,
    steps: int = 600,
    batch_size: int = 32,
    train_seq_len: int = 64,
    learning_rate: float = TRAINED_D128_DEFAULT_LR,
    seed: int = 1337,
):
    """Train a tiny dense nanoGPT-style model and export split-head weights."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    if steps < 0:
        raise ValueError("steps must be non-negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if train_seq_len <= 0 or train_seq_len > 128:
        raise ValueError("train_seq_len must be in 1..128")

    text = _trained_text()
    vocab = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}
    tokens = torch.tensor(_tokenize_text(text, stoi), dtype=torch.long)
    ranges = split_ranges(text, calibration_limit=2048)
    train_tokens = tokens[ranges["train_bytes"][0]: ranges["train_bytes"][1]]
    eval_tokens = tokens[ranges["evaluation_bytes"][0]: ranges["evaluation_bytes"][1]]
    if len(eval_tokens) <= train_seq_len + 1:
        eval_tokens = tokens

    n_layer = 2
    n_head = 4
    d_model = 128
    d_head = 32
    block_size = 128
    mlp_dim = 4 * d_model
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            self.q = nn.ModuleList([nn.Linear(d_model, d_head, bias=False) for _ in range(n_head)])
            self.k = nn.ModuleList([nn.Linear(d_model, d_head, bias=False) for _ in range(n_head)])
            self.v = nn.ModuleList([nn.Linear(d_model, d_head, bias=False) for _ in range(n_head)])
            self.c_proj = nn.Linear(d_model, d_model, bias=False)
            self.ln2 = nn.LayerNorm(d_model)
            self.fc = nn.Linear(d_model, mlp_dim, bias=False)
            self.proj = nn.Linear(mlp_dim, d_model, bias=False)

        def forward(self, x):
            bsz, seq, _ = x.shape
            ln1 = self.ln1(x)
            mask = torch.triu(
                torch.ones(seq, seq, dtype=torch.bool, device=x.device),
                diagonal=1,
            )
            heads = []
            for head in range(n_head):
                q = self.q[head](ln1)
                k = self.k[head](ln1)
                v = self.v[head](ln1)
                attn = (q @ k.transpose(-2, -1)) * (d_head ** -0.5)
                attn = attn.masked_fill(mask.view(1, seq, seq), float("-inf"))
                probs = F.softmax(attn, dim=-1)
                heads.append(probs @ v)
            x = x + self.c_proj(torch.cat(heads, dim=-1))
            ln2 = self.ln2(x)
            x = x + self.proj(F.gelu(self.fc(ln2)))
            return x

    class TinyGPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.wte = nn.Embedding(len(vocab), d_model)
            self.wpe = nn.Embedding(block_size, d_model)
            self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])
            self.ln_f = nn.LayerNorm(d_model)

        def forward(self, idx, targets=None):
            bsz, seq = idx.shape
            pos = torch.arange(seq, dtype=torch.long, device=idx.device)
            x = self.wte(idx) + self.wpe(pos)[None, :, :]
            for block in self.blocks:
                x = block(x)
            logits = self.ln_f(x) @ self.wte.weight.T
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            return logits, loss

    def batch_from(source, *, deterministic: bool = False):
        if len(source) <= train_seq_len + 1:
            repeats = (train_seq_len + 2) // max(len(source), 1) + 1
            source = source.repeat(repeats)
        max_start = len(source) - train_seq_len - 1
        if deterministic:
            starts = torch.arange(batch_size, dtype=torch.long) % max_start
        else:
            starts = torch.randint(0, max_start, (batch_size,), generator=generator)
        x = torch.stack([source[i: i + train_seq_len] for i in starts])
        y = torch.stack([source[i + 1: i + train_seq_len + 1] for i in starts])
        return x, y

    model = TinyGPT()
    with torch.no_grad():
        model.wte.weight.copy_(_stable_token_embedding(torch, len(vocab), d_model))
        model.wpe.weight.normal_(mean=0.0, std=0.02, generator=generator)
        for block in model.blocks:
            for head in range(n_head):
                block.q[head].weight.normal_(mean=0.0, std=0.02, generator=generator)
                block.k[head].weight.normal_(mean=0.0, std=0.02, generator=generator)
                block.v[head].weight.normal_(mean=0.0, std=0.02, generator=generator)
            block.c_proj.weight.normal_(mean=0.0, std=0.02, generator=generator)
            block.fc.weight.normal_(mean=0.0, std=0.02, generator=generator)
            block.proj.weight.normal_(mean=0.0, std=0.02, generator=generator)
    embed_param_ids = {id(param) for param in model.wte.parameters()}
    non_embed_params = [
        param for param in model.parameters()
        if id(param) not in embed_param_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(model.wte.parameters()),
                "lr": learning_rate * TRAINED_D128_EMBED_LR_MULTIPLIER,
            },
            {
                "params": non_embed_params,
                "lr": learning_rate,
            },
        ],
        weight_decay=0.0,
    )
    train_loss = None
    with torch.no_grad():
        x_initial, y_initial = batch_from(train_tokens, deterministic=True)
        _, initial_loss = model(x_initial, y_initial)
        initial_train_loss = float(initial_loss.detach().cpu())
    for _ in range(steps):
        x, y = batch_from(train_tokens)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        train_loss = float(loss.detach().cpu())

    with torch.no_grad():
        x_val, y_val = batch_from(eval_tokens, deterministic=True)
        _, val_loss = model(x_val, y_val)
        x_train, y_train = batch_from(train_tokens, deterministic=True)
        _, train_eval_loss = model(x_train, y_train)

    state = {
        "transformer.wte.weight": model.wte.weight.detach().cpu().clone(),
        "transformer.wpe.weight": model.wpe.weight.detach().cpu().clone(),
        "transformer.ln_f.weight": model.ln_f.weight.detach().cpu().clone(),
        "transformer.ln_f.bias": model.ln_f.bias.detach().cpu().clone(),
    }
    state["lm_head.weight"] = state["transformer.wte.weight"]
    for layer_idx, block in enumerate(model.blocks):
        state[f"transformer.h.{layer_idx}.ln_1.weight"] = block.ln1.weight.detach().cpu().clone()
        state[f"transformer.h.{layer_idx}.ln_1.bias"] = block.ln1.bias.detach().cpu().clone()
        state[f"transformer.h.{layer_idx}.ln_2.weight"] = block.ln2.weight.detach().cpu().clone()
        state[f"transformer.h.{layer_idx}.ln_2.bias"] = block.ln2.bias.detach().cpu().clone()
        for head in range(n_head):
            state[f"transformer.h.{layer_idx}.attn.c_attn.weight_h{head}_query"] = (
                block.q[head].weight.detach().cpu().clone()
            )
            state[f"transformer.h.{layer_idx}.attn.c_attn.weight_h{head}_key"] = (
                block.k[head].weight.detach().cpu().clone()
            )
            state[f"transformer.h.{layer_idx}.attn.c_attn.weight_h{head}_value"] = (
                block.v[head].weight.detach().cpu().clone()
            )
        state[f"transformer.h.{layer_idx}.attn.c_proj.weight"] = block.c_proj.weight.detach().cpu().clone()
        state[f"transformer.h.{layer_idx}.attn.c_proj.bias"] = torch.zeros(d_model, dtype=torch.float32)
        state[f"transformer.h.{layer_idx}.mlp.c_fc.weight"] = block.fc.weight.detach().cpu().clone()
        state[f"transformer.h.{layer_idx}.mlp.c_fc.bias"] = torch.zeros(mlp_dim, dtype=torch.float32)
        state[f"transformer.h.{layer_idx}.mlp.c_proj.weight"] = block.proj.weight.detach().cpu().clone()
        state[f"transformer.h.{layer_idx}.mlp.c_proj.bias"] = torch.zeros(d_model, dtype=torch.float32)

    payload = {
        "source_snapshot": TRAINED_D128_SOURCE_SNAPSHOT,
        "seed": seed,
        "model_args": {
            "n_layer": n_layer,
            "n_head": n_head,
            "n_embd": d_model,
            "block_size": block_size,
            "vocab_size": len(vocab),
            "bias": True,
            "layer_norm_epsilon": 1e-5,
        },
        "stoi": stoi,
        "itos": itos,
        "state_dict": state,
        "text": text,
        "training": {
            "steps": int(steps),
            "batch_size": int(batch_size),
            "train_seq_len": int(train_seq_len),
            "learning_rate": float(learning_rate),
            "embedding_learning_rate": float(learning_rate * TRAINED_D128_EMBED_LR_MULTIPLIER),
            "embedding_learning_rate_multiplier": float(TRAINED_D128_EMBED_LR_MULTIPLIER),
            "initial_train_loss": float(initial_train_loss),
            "final_train_loss": float(train_loss if train_loss is not None else train_eval_loss.detach().cpu()),
            "train_loss": float(train_loss if train_loss is not None else train_eval_loss.detach().cpu()),
            "train_eval_loss": float(train_eval_loss.detach().cpu()),
            "validation_loss": float(val_loss.detach().cpu()),
        },
    }
    payload["training"]["training_quality"] = _trained_attention_quality(payload)
    return payload


def _required_state_keys(model_args: Dict[str, object]):
    n_layer = int(model_args["n_layer"])
    n_head = int(model_args["n_head"])
    keys = [
        "transformer.wte.weight",
        "transformer.wpe.weight",
        "transformer.ln_f.weight",
        "transformer.ln_f.bias",
        "lm_head.weight",
    ]
    for layer in range(n_layer):
        keys.extend([
            f"transformer.h.{layer}.ln_1.weight",
            f"transformer.h.{layer}.ln_1.bias",
            f"transformer.h.{layer}.ln_2.weight",
            f"transformer.h.{layer}.ln_2.bias",
            f"transformer.h.{layer}.attn.c_proj.weight",
            f"transformer.h.{layer}.attn.c_proj.bias",
            f"transformer.h.{layer}.mlp.c_fc.weight",
            f"transformer.h.{layer}.mlp.c_fc.bias",
            f"transformer.h.{layer}.mlp.c_proj.weight",
            f"transformer.h.{layer}.mlp.c_proj.bias",
        ])
        for head in range(n_head):
            for proj in ("query", "key", "value"):
                keys.append(f"transformer.h.{layer}.attn.c_attn.weight_h{head}_{proj}")
    return keys


def _validate_trained_quality(payload: Dict[str, object]) -> None:
    training = payload.get("training", {})
    missing = [
        key for key in (
            "initial_train_loss",
            "final_train_loss",
            "learning_rate",
            "training_quality",
        )
        if key not in training
    ]
    if missing:
        raise ValueError(f"trained payload missing quality fields: {missing}")
    initial = float(training["initial_train_loss"])
    final = float(training["final_train_loss"])
    if not math.isfinite(initial) or not math.isfinite(final) or initial <= 0.0:
        raise ValueError("trained payload has invalid train-loss values")
    if final > TRAINED_QUALITY_MIN_LOSS_RATIO * initial:
        raise ValueError(
            "trained payload loss did not improve enough: "
            f"initial={initial:.6f}, final={final:.6f}, "
            f"required_final<={TRAINED_QUALITY_MIN_LOSS_RATIO * initial:.6f}"
        )
    quality = dict(training["training_quality"])
    entropy_drop = float(quality.get("best_entropy_drop", 0.0))
    qkt_p95 = float(quality.get("best_qkt_p95_abs", 0.0))
    if entropy_drop < TRAINED_QUALITY_MIN_ENTROPY_DROP:
        raise ValueError(
            "trained payload attention is too close to uniform: "
            f"best_entropy_drop={entropy_drop:.6f}, "
            f"required>={TRAINED_QUALITY_MIN_ENTROPY_DROP:.6f}"
        )
    if qkt_p95 < TRAINED_QUALITY_MIN_QKT_P95:
        raise ValueError(
            "trained payload QKT logits are too small: "
            f"best_qkt_p95_abs={qkt_p95:.6g}, "
            f"required>={TRAINED_QUALITY_MIN_QKT_P95:.6g}"
        )


def validate_trained_payload(payload: Dict[str, object], *, require_quality: bool | None = None) -> None:
    import numpy as np

    model_args = payload.get("model_args", {})
    state = payload.get("state_dict", {})
    missing = [key for key in _required_state_keys(model_args) if key not in state]
    if missing:
        raise ValueError(f"trained payload missing required state keys: {missing[:5]}")
    if int(model_args.get("n_layer", 0)) != 2 or int(model_args.get("n_embd", 0)) != 128:
        raise ValueError("trained d128 payload must have n_layer=2 and n_embd=128")
    dense_names = [
        name for name in state
        if (
            "c_attn.weight_h" in name
            or name.endswith("attn.c_proj.weight")
            or name.endswith("mlp.c_fc.weight")
            or name.endswith("mlp.c_proj.weight")
        )
    ]
    if not dense_names:
        raise ValueError("trained payload has no dense transformer weights")
    for name in dense_names:
        arr = state[name].detach().cpu().numpy() if hasattr(state[name], "detach") else np.asarray(state[name])
        nonzero_fraction = float(np.count_nonzero(np.abs(arr) > 1e-8) / max(arr.size, 1))
        if nonzero_fraction < 0.95:
            raise ValueError(f"trained weight {name!r} is not dense enough: {nonzero_fraction:.3f}")
    for name in state:
        if name.endswith(".bias") and (".attn.c_proj." in name or ".mlp." in name):
            arr = state[name].detach().cpu().numpy() if hasattr(state[name], "detach") else np.asarray(state[name])
            if not np.allclose(arr, 0.0):
                raise ValueError(f"trained fixture matmul bias {name!r} must be zero")
    training = payload.get("training", {})
    if require_quality is None:
        require_quality = int(training.get("steps", 0)) >= 100
    if require_quality:
        _validate_trained_quality(payload)


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


def write_trained_d128_fixture(
    checkpoint_path: Path = DEFAULT_TRAINED_D128_FIXTURE,
    metadata_path: Path = DEFAULT_TRAINED_D128_METADATA,
    *,
    steps: int = 600,
    batch_size: int = 32,
    train_seq_len: int = 64,
    learning_rate: float = TRAINED_D128_DEFAULT_LR,
) -> Dict[str, object]:
    import torch

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _train_d128_payload(
        steps=steps,
        batch_size=batch_size,
        train_seq_len=train_seq_len,
        learning_rate=learning_rate,
    )
    validate_trained_payload(payload)
    torch.save(payload, checkpoint_path)
    training = dict(payload["training"])
    metadata = build_metadata(
        checkpoint_path,
        sha256_file(checkpoint_path),
        source_snapshot=TRAINED_D128_SOURCE_SNAPSHOT,
        n_layer=2,
        n_head=4,
        d_model=128,
        block_size=128,
        real_trained_ready=True,
        text=payload["text"],
        training=training,
        validation_loss=float(training["validation_loss"]),
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


def validate_trained_fixture_metadata(
    checkpoint_path: Path = DEFAULT_TRAINED_D128_FIXTURE,
    metadata_path: Path = DEFAULT_TRAINED_D128_METADATA,
) -> Dict[str, object]:
    metadata = validate_fixture_metadata(checkpoint_path, metadata_path)
    if metadata.get("real_trained_d128_ready") is not True:
        raise ValueError("trained fixture metadata must set real_trained_d128_ready=true")
    required_metadata = [
        "learning_rate",
        "initial_train_loss",
        "final_train_loss",
        "validation_loss",
        "training_quality",
    ]
    missing_metadata = [key for key in required_metadata if key not in metadata]
    if missing_metadata:
        raise ValueError(f"trained fixture metadata missing required fields: {missing_metadata}")
    import torch

    payload = torch.load(checkpoint_path, map_location="cpu")
    training_steps = int(metadata.get("training", {}).get("steps", 0))
    validate_trained_payload(payload, require_quality=training_steps >= 100)
    return metadata


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--stage4-d384", action="store_true")
    parser.add_argument("--trained-d128", action="store_true")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-seq-len", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=TRAINED_D128_DEFAULT_LR)
    args = parser.parse_args(argv)
    if args.stage4_d384 and args.trained_d128:
        raise ValueError("--stage4-d384 and --trained-d128 are mutually exclusive")

    output_path = args.output
    if args.stage4_d384 and output_path == DEFAULT_FIXTURE and args.metadata is None:
        output_path = DEFAULT_STAGE4_FIXTURE
    if args.trained_d128 and output_path == DEFAULT_FIXTURE and args.metadata is None:
        output_path = DEFAULT_TRAINED_D128_FIXTURE
    metadata_path = args.metadata or output_path.with_suffix(output_path.suffix + ".json")
    if args.validate_only:
        if args.trained_d128:
            validate_trained_fixture_metadata(output_path, metadata_path)
        else:
            validate_fixture_metadata(output_path, metadata_path)
        print(f"fixture metadata valid: {output_path}")
        return 0

    if args.trained_d128:
        metadata = write_trained_d128_fixture(
            output_path,
            metadata_path,
            steps=args.steps,
            batch_size=args.batch_size,
            train_seq_len=args.train_seq_len,
            learning_rate=args.learning_rate,
        )
    elif args.stage4_d384:
        metadata = write_stage4_fixture(output_path, metadata_path)
    else:
        metadata = write_fixture(output_path, metadata_path)
    print(f"wrote fixture: {output_path}")
    print(f"wrote metadata: {metadata_path}")
    print(f"sha256: {metadata['checkpoint_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
