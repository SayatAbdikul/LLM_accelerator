#!/usr/bin/env python3
"""Run the unmodified RTN baseline at 256 tokens — same defaults as eval_with_gptq
but without invoking apply_gptq, to anchor the GPTQ comparison.
"""
from pathlib import Path
import sys
import torch
sys.path.insert(0, "software")

from taccel.runtime.gpt2_perplexity import (
    CALIBRATION_N_SEQS_LARGE,
    CALIBRATION_PERCENTILE_DEFAULT,
    CALIBRATION_SEQ_LEN_LARGE,
    evaluate_gpt2_perplexity,
    file_sha256,
    tokenize_text_file,
)

CKPT = Path("software/tests/fixtures/generated/gpt2_converted_nanogpt.pt")
TOK = Path("software/tests/fixtures/generated/hf_gpt2")
CALIB = Path("software/tests/fixtures/generated/wikitext2_stage5_calibration.txt")
EVAL = Path("software/tests/fixtures/generated/wikitext2_stage5_eval.txt")

payload = torch.load(CKPT, map_location="cpu")
calib_ids = tokenize_text_file(TOK, CALIB)
eval_ids = tokenize_text_file(TOK, EVAL, max_tokens=256)

print(f"# Unmodified-state RTN baseline at 256 tokens, default preset")
result = evaluate_gpt2_perplexity(
    payload,
    calibration_token_ids=calib_ids,
    eval_token_ids=eval_ids,
    tokenizer_dir=TOK,
    calibration_sha256=file_sha256(CALIB),
    eval_sha256=file_sha256(EVAL),
    max_eval_tokens=256,
    context_len=256,
    calibration_seq_len=CALIBRATION_SEQ_LEN_LARGE,
    calibration_n_seqs=CALIBRATION_N_SEQS_LARGE,
    calibration_percentile=CALIBRATION_PERCENTILE_DEFAULT,
    ptq_preset=None,
)
print(f"golden_perplexity: {result.golden_perplexity:.6f}")
print(f"fake_quant_perplexity: {result.fake_quant_perplexity:.6f}")
print(f"relative_delta: {result.relative_delta:.6%}")
print(f"target_count: {result.target_count}")
