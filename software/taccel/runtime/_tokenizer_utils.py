"""Tokenizer helpers shared by the top-level CLI entry scripts.

`run_gpt2.py`, `run_nanogpt.py`, and `chat_gpt2.py` all need to parse a
comma-separated `--prompt-ids` argument and (for char-level fixtures)
encode/decode through the checkpoint's `stoi`/`itos` maps. The logic is
the same in each script; keep it here so the entry scripts stay thin.

These helpers are storage-format-agnostic — they operate on plain
``dict``s for ``stoi``/``itos``. Each entry script is responsible for
extracting those dicts from its own checkpoint payload and raising the
domain-specific "no tokenizer found" message if the prompt is text but
the checkpoint has no embedded char map (e.g. converted GPT-2, which
uses BPE via a separate HF tokenizer).
"""
from __future__ import annotations

from typing import List, Mapping


def parse_prompt_ids(raw: str) -> List[int]:
    """Parse ``"1,42,7"`` → ``[1, 42, 7]``. Raises on empty input."""
    if not raw.strip():
        raise ValueError("prompt-ids must not be empty")
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def tokenize_char_prompt(prompt: str, stoi: Mapping) -> List[int]:
    """Char-level prompt encode using a ``stoi`` map (one int per character).

    Raises ``ValueError`` if any character is absent from ``stoi``. Callers
    that need a "no stoi found" check should test ``bool(stoi)`` first and
    surface their own domain-specific message (e.g. "GPT-2 uses BPE; pass
    --prompt-ids instead").
    """
    if not prompt:
        raise ValueError("prompt must not be empty")
    missing = sorted({ch for ch in prompt if ch not in stoi})
    if missing:
        raise ValueError(
            f"prompt contains characters absent from checkpoint tokenizer: {missing!r}"
        )
    return [int(stoi[ch]) for ch in prompt]


def decode_char_ids(token_ids: List[int], itos: Mapping) -> str:
    """Char-level token decode using an ``itos`` map. Returns ``""`` when
    ``itos`` is empty (matches the converted-GPT-2 case where decoding
    runs through HF). Unknown ids decode to ``"?"``."""
    if not itos:
        return ""
    out = []
    for tok in token_ids:
        key = str(int(tok))
        out.append(str(itos.get(key, itos.get(int(tok), "?"))))
    return "".join(out)
