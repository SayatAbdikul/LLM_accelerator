#!/bin/bash
# D5: percdamp sweep at n=128 on block 11 mlp.c_proj.
set -u
cd /home/user/LLM_accelerator
export PYTHONPATH=software
export OMP_NUM_THREADS=1
PY=.venv/bin/python
CKPT=software/tests/fixtures/generated/gpt2_converted_nanogpt.pt
TOK=software/tests/fixtures/generated/hf_gpt2
CALIB=software/tests/fixtures/generated/wikitext2_stage5_calibration.txt
EVAL=software/tests/fixtures/generated/wikitext2_stage5_eval.txt

for d in 0.005 0.02 0.05 0.1; do
  echo "=== D5 percdamp=$d ==="
  $PY software/tools/eval_with_gptq.py "$CKPT" \
    --tokenizer-dir "$TOK" --calibration-text "$CALIB" --eval-text "$EVAL" \
    --max-eval-tokens 33 --context-len 32 \
    --target-blocks 11 --weight-types mlp.c_proj \
    --gptq-percdamp "$d" --gptq-blocksize 128 \
    --weight-search-n-seqs 128 --weight-search-seq-len 64 \
    2>&1 | tee ".gptq_runs/D5_d${d}.log"
done
echo "=== D5 SUMMARY ==="
echo "n=128 percdamp=0.01 (from D4): fake_quant_perplexity=4921.55"
for d in 0.005 0.02 0.05 0.1; do
  log=".gptq_runs/D5_d${d}.log"
  ppl=$(grep -E "^fake_quant_perplexity" "$log" 2>/dev/null | awk '{print $2}')
  rel=$(grep -E "^relative_delta" "$log" 2>/dev/null | awk '{print $2}')
  echo "percdamp=$d  fake_quant_perplexity=$ppl  relative_delta=$rel"
done
