#!/bin/bash
# Step B: SmoothQuant α-sweep on block 11 + GPTQ on block 11 c_proj at 256 tok.
# Runs sequentially within this script; the parent invokes this in the background.
set -u
cd /home/user/LLM_accelerator
export PYTHONPATH=software
export OMP_NUM_THREADS=4
PY=.venv/bin/python
CKPT=software/tests/fixtures/generated/gpt2_converted_nanogpt.pt
TOK=software/tests/fixtures/generated/hf_gpt2
CALIB=software/tests/fixtures/generated/wikitext2_stage5_calibration.txt
EVAL=software/tests/fixtures/generated/wikitext2_stage5_eval.txt

for a in "$@"; do
  echo "=== Step B α=$a ==="
  $PY software/tools/eval_with_compose.py "$CKPT" \
    --tokenizer-dir "$TOK" --calibration-text "$CALIB" --eval-text "$EVAL" \
    --max-eval-tokens 256 --context-len 256 \
    --sq-blocks 11 --sq-targets ln_2_fc1 --sq-alpha "$a" \
    --gptq-blocks 11 --gptq-types mlp.c_proj \
    --gptq-percdamp 0.01 --gptq-blocksize 128 \
    --gptq-n-seqs 128 --gptq-seq-len 64 --sequential \
    2>&1 | tee ".gptq_runs/followup/B_sq${a}_gptq.log"
done
