#!/bin/bash
# D4 extension: push n_seqs further to confirm convergence below 4800.
set -u
cd /home/user/LLM_accelerator
export PYTHONPATH=software
export OMP_NUM_THREADS=1
PY=.venv/bin/python
CKPT=software/tests/fixtures/generated/gpt2_converted_nanogpt.pt
TOK=software/tests/fixtures/generated/hf_gpt2
CALIB=software/tests/fixtures/generated/wikitext2_stage5_calibration.txt
EVAL=software/tests/fixtures/generated/wikitext2_stage5_eval.txt

for n in 256 512; do
  echo "=== D4 n_seqs=$n ==="
  $PY software/tools/eval_with_gptq.py "$CKPT" \
    --tokenizer-dir "$TOK" --calibration-text "$CALIB" --eval-text "$EVAL" \
    --max-eval-tokens 33 --context-len 32 \
    --target-blocks 11 --weight-types mlp.c_proj \
    --gptq-percdamp 0.01 --gptq-blocksize 128 \
    --weight-search-n-seqs "$n" --weight-search-seq-len 64 \
    2>&1 | tee ".gptq_runs/D4_n${n}.log"
  echo
done
echo "=== D4 EXTRA SUMMARY ==="
for n in 16 32 64 128 256 512; do
  log=".gptq_runs/D4_n${n}.log"
  ppl=$(grep -E "^fake_quant_perplexity" "$log" 2>/dev/null | awk '{print $2}')
  rel=$(grep -E "^relative_delta" "$log" 2>/dev/null | awk '{print $2}')
  echo "n=$n  fake_quant_perplexity=$ppl  relative_delta=$rel"
done
