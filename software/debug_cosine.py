"""Quick cosine test for the improved reference."""
import sys
import numpy as np
sys.path.insert(0, "/Users/sayat/Documents/GitHub/LLM_accelerator/software")

import torch
from taccel.runtime.calibration import build_calibration_scales
from taccel.runtime.tiny_fixture import build_stage3_tiny_decoder_bundle, run_tiny_decode_trace
from taccel.runtime.fake_quant_reference import NanoGPTFQReference
from taccel.runtime.host_runner import HostRunner
from taccel.runtime.fake_quant import cosine_similarity

FIXTURE = "/Users/sayat/Documents/GitHub/LLM_accelerator/software/tests/fixtures/generated/nanogpt_shakespeare_char_d128_l2.pt"

payload = torch.load(FIXTURE, map_location="cpu")
scales = build_calibration_scales(payload)

MAX_NEW_TOKENS = 8
PROMPT_IDS = [0]

tiny = build_stage3_tiny_decoder_bundle(payload, smoke_decode_steps=MAX_NEW_TOKENS, calibration_scales=scales)
runner = HostRunner(tiny.build.bundle, logits_dtype=np.int8)

ref = NanoGPTFQReference(
    state_dict=payload["state_dict"],
    model_args=payload["model_args"],
    scales=scales,
)

vocab = tiny.config.vocab_size

# Prefill
golden_logits = runner.run_prefill(PROMPT_IDS)
ref_logits = ref.forward(PROMPT_IDS)

cos = cosine_similarity(golden_logits[:vocab].astype(np.float32), ref_logits[:vocab].astype(np.float32))
diff = np.abs(golden_logits[:vocab].astype(np.int32) - ref_logits[:vocab].astype(np.int32))
print(f"Step 0 (prefill): cosine={cos:.4f}, max_diff={diff.max()}, mean_diff={diff.mean():.2f}")
print(f"  golden: {golden_logits[:10]}")
print(f"  ref:    {ref_logits[:10]}")

# Decode steps
generated = list(PROMPT_IDS)
next_tok = int(np.argmax(golden_logits[:vocab]))
ref_next_tok = int(np.argmax(ref_logits[:vocab]))

for step in range(MAX_NEW_TOKENS):
    generated.append(next_tok)
    pos = len(generated) - 1
    golden_logits = runner.run_decode_step(next_tok, pos)
    ref_logits = ref.forward(generated)
    cos = cosine_similarity(golden_logits[:vocab].astype(np.float32), ref_logits[:vocab].astype(np.float32))
    diff = np.abs(golden_logits[:vocab].astype(np.int32) - ref_logits[:vocab].astype(np.int32))
    print(f"Step {step+1} (decode, tok={next_tok}): cosine={cos:.4f}, max_diff={diff.max()}, mean_diff={diff.mean():.2f}")
    next_tok = int(np.argmax(golden_logits[:vocab]))

print(f"\nMin cosine across all steps: measured above")
