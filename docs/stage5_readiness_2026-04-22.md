# Stage 5 Readiness Baseline - 2026-04-22

This note records the stabilization pass completed before starting Stage 5
large-vocab/GPT-2 work.

## Summary

Stage 3, Stage 4, and the Stage 4.5 trained-nanoGPT verification gates are ready
enough to use as the baseline for Stage 5. The remaining broad-suite failures
are local image-dataset discovery failures in DeiT calibration tests, not decoder
runtime, nanoGPT, ISA v1.1, or Stage 5 prerequisite failures.

## nanoGPT Verification Baseline

Trained d128 nanoGPT fixture:

- Golden model vs compiler-matched fake-quant:
  - `min_cosine = 0.996278703212738`
  - `min_top5_overlap = 4`
  - generated token IDs match exactly
- Golden model vs true FP32 nanoGPT inference, evaluated on the same
  golden-generated prefixes:
  - `fp32_argmax_in_golden_top10_rate = 1.0` for all 5 prompts
  - `golden_argmax_in_fp32_top10_rate = 0.9696969696969697` to `1.0`
  - exact top-1 match rate = `0.9696969696969697` to `1.0`
  - minimum top-10 overlap across prompts = `4`

The FP32 comparison is rank/top-k based by design. Raw dequantized-logit cosine
remains diagnostic-only because INT8 quantization changes logit magnitudes.

## Regression Commands Run

Broad non-RTL software suite, excluding RTL comparison tests and the network /
dataset download test:

```bash
PYTHONPATH=software /tmp/llm_accelerator_stage3_venv/bin/python -m pytest software/tests \
  --ignore=software/tests/test_compare_rtl_golden.py \
  --ignore=software/tests/test_batch_compare_rtl_golden.py \
  --ignore=software/tests/test_download_imagenet_class.py -q
```

Result:

- `356 passed`
- `3 failed`
- runtime: `342.61s`

The 3 failures are all dataset availability failures:

- `software/tests/test_compare_golden_calibration.py::TestCalibrationScaleMapping::test_diagnostic_preset_cats_dogs_uses_all_local_samples`
- `software/tests/test_compare_golden_calibration.py::TestCalibrationScaleMapping::test_diagnostic_preset_imagenet_class0_uses_all_local_samples`
- `software/tests/test_compare_golden_calibration.py::TestCalibrationScaleMapping::test_discover_cats_dogs_samples_is_stable_and_labeled`

Observed cause: local cats/dogs and ImageNet class-0 sample discovery returned
empty sample lists. These are unrelated to the nanoGPT decoder path.

Focused Stage 4 / Stage 5-prerequisite commands:

```bash
PYTHONPATH=software /tmp/llm_accelerator_stage3_venv/bin/python -m pytest \
  software/tests/test_e2e_nanogpt_d384.py -q
```

Result: `1 passed in 228.04s`.

```bash
PYTHONPATH=software /tmp/llm_accelerator_stage3_venv/bin/python -m pytest \
  software/tests/test_run_nanogpt.py -q
```

Result: `1 passed in 6.08s`.

```bash
PYTHONPATH=software /tmp/llm_accelerator_stage3_venv/bin/python -m pytest \
  software/tests/test_weight_striping.py \
  software/tests/test_activation_spill.py \
  software/tests/test_mlp_strip_mining.py -q
```

Result: `11 passed in 1.73s`.

Previously verified trained-nanoGPT gates after the FP32 comparison fix:

```bash
PYTHONPATH=software /tmp/llm_accelerator_stage3_venv/bin/python -m pytest \
  software/tests/test_debug_fp32_comparison.py \
  software/tests/test_tiny_fixture_tooling.py \
  software/tests/test_e2e_trained_nanogpt_fake_quant.py \
  software/tests/test_e2e_trained_nanogpt_fp32.py -q
```

Result: `15 passed`.

Stable tiny decoder checks:

```bash
PYTHONPATH=software /tmp/llm_accelerator_stage3_venv/bin/python -m pytest \
  software/tests/test_e2e_tiny.py \
  software/tests/test_tiny_decode_smoke.py \
  software/tests/test_tiny_decode_determinism.py -q
```

Result: `7 passed`.

Stage 3 runtime foundation checks:

```bash
PYTHONPATH=software /tmp/llm_accelerator_stage3_venv/bin/python -m pytest \
  software/tests/test_program_bundle.py \
  software/tests/test_relocation_patching.py \
  software/tests/test_runtime_patch_sites.py \
  software/tests/test_simulator_bundle_api.py \
  software/tests/test_embedding_patch_sites.py \
  software/tests/test_kv_layout.py \
  software/tests/test_kv_cache_store_load.py \
  software/tests/test_kv_cache_scale.py \
  software/tests/test_kv_banking.py \
  software/tests/test_config_attn_runtime_patch.py \
  software/tests/test_host_runner.py \
  software/tests/test_logits_store.py \
  software/tests/test_decoder_bundle.py -q
```

Result: `41 passed`.

ISA / assembler / masked-softmax checks:

```bash
PYTHONPATH=software /tmp/llm_accelerator_stage3_venv/bin/python -m pytest \
  software/tests/test_isa_encoding.py \
  software/tests/test_assembler.py \
  software/tests/test_masked_softmax.py -q
```

Result: `89 passed`.

## Stage 5 Start Criteria

Stage 5 can start with these known constraints:

- The decoder golden model, ProgramBundle runtime, KV cache flow, masked
  attention, d384 striping path, and trained d128 FP32 rank gate are green.
- GPT-2/HF work should treat large-vocab logits striping, HF Conv1D conversion,
  tied embedding export, and DRAM sizing as the first new risk areas.
- Dataset-dependent DeiT calibration tests should not be used as blockers for
  Stage 5 unless the local cats/dogs and ImageNet sample fixtures are restored.
