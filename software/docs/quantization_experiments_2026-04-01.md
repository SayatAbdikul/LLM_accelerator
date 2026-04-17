# Quantization Experiment Report — 2026-04-01

## Scope

This document is the cumulative wrap-up of the quantization and accuracy experiments implemented in the software stack up to 2026-04-01.

It covers:

- benchmark and dataset support added to the repo
- diagnostic infrastructure added to `compare_golden.py` and friends
- compiler / simulator experiments
- PTQ-only weight, scale, and calibration experiments
- PTQ4ViT-inspired software and simulator experiments
- current winners and dead ends

This report complements, not replaces, the point-in-time rerun note in
[diagnostics_rerun_2026-03-29.md](/Users/sayat/Documents/GitHub/transformer_accelerator/software/docs/diagnostics_rerun_2026-03-29.md).

## Verification State

Latest full software test status at the end of this work:

- `venv/bin/python3 -m pytest tests -q`
- `226 passed`

## Benchmark Surfaces

The repo now has three distinct evaluation surfaces. Their numbers should not be compared as if they came from the same pipeline.

| Surface | Driver | Dataset | Purpose |
|---|---|---|---|
| Golden-model frozen local | `compare_golden.py` | frozen local validation image set | canonical accelerator PTQ benchmark |
| Golden-model ImageNet class-0 | `compare_golden.py` | 100 local `tench` images | main late-stage tuning benchmark |
| Software fake-quant runtime | `compare_accuracy.py` | 100 local `tench` images | PyTorch fake-quant sanity check |

Additional local-dataset support was also added for:

- 200-image cats/dogs flat folder benchmark
- generic local flat folders
- ImageNet class downloader and one-class benchmark flow

## Major Software Additions

The experiment campaign added the following capabilities to the codebase:

- richer compiler manifests and run-config capture in `compare_golden.py`
- reproducible diagnostic presets for frozen-local, cats/dogs, and one-class ImageNet
- local flat-folder dataset support
- trace diff, block impact, and late-attention diagnostics
- selective `SmoothQuant`
- selective `FC1 REQUANT_PC`
- selective `FC2 REQUANT_PC`
- selective `out_proj REQUANT_PC` plumbing
- `DEQUANT_ADD`
- experimental `GELU-from-ACCUM`
- experimental fused `SOFTMAX_ATTNV`
- output-aware clipping
- greedy AdaRound
- analytical bias correction
- node-specific percentile activation calibration
- PTQ4ViT-inspired Hessian-guided search objectives
- PTQ4ViT-inspired twin-uniform search path
- ISA-free simulator-side runtime twin-uniform emulation driven by compiler manifest

## Dataset / Tooling Validation

### Cats And Dogs Local Benchmark

The local flat-folder benchmark path was added and validated on:

- [software/images/cats and dogs](/Users/sayat/Documents/GitHub/transformer_accelerator/software/images/cats%20and%20dogs)

Golden-model baseline on the 200-image set:

- mean cosine: `0.8323`
- `p10`: `0.7529`
- min: `0.5899`
- top-1 agreement: `70.5%`

This dataset was useful for validating local benchmark support, but it was not the primary tuning target.

### ImageNet One-Class Download Flow

The ImageNet class downloader in
[download_imagenet_class.py](/Users/sayat/Documents/GitHub/transformer_accelerator/software/images/download_imagenet_class.py)
was fixed and used to download 100 images from class `0` (`tench`).

The one-class benchmark folder is:

- [000_tench_Tinca_tinca](/Users/sayat/Documents/GitHub/transformer_accelerator/software/images/imagenet_one_class/000_tench_Tinca_tinca)

This became the main benchmark for late-stage PTQ experimentation.

## Frozen-Local Golden Benchmark

### Canonical Baseline And Best Variant

| Variant | Mean | P10 | Min | Top-1 |
|---|---:|---:|---:|---:|
| `baseline_frozen_local` | `0.8291` | `0.7467` | `0.6823` | `70%` |
| `current_best_sq_ln2_fc1_b0_8_10` | `0.8314` | `0.7671` | `0.6816` | `90%` |

The frozen-local winner remained:

- `SmoothQuant`
- target: `LN2 -> FC1`
- `alpha = 0.50`
- blocks: `0,1,2,3,4,5,6,7,8,10`

### Major Frozen-Local Experiments

| Experiment | Mean | P10 | Min | Top-1 | Read |
|---|---:|---:|---:|---:|---|
| `GELU-from-ACCUM blocks 2-7,9` | `0.8159` | `0.6910` | `0.6592` | `75%` | clear regression |
| `SOFTMAX_ATTNV block11` | `0.8344` | `0.7723` | `0.6767` | `85%` | better mean / p10, worse min / top-1 |
| `SOFTMAX_ATTNV all blocks` | `0.8332` | `0.7469` | `0.6747` | `65%` | broad rollout not promotable |
| `DEQUANT_ADD residual1 block11` | `0.8314` | `0.7671` | `0.6816` | `90%` | accuracy-neutral, lower cycles |
| `DEQUANT_ADD residual1 scale blend 0.25` | `0.8299` | `0.7617` | `0.6672` | `75%` | regressed |
| `DEQUANT_ADD residual1 scale blend 0.50` | `0.8257` | `0.7552` | `0.6442` | `80%` | regressed |
| `DEQUANT_ADD residual1 scale percentile 99` | `0.8234` | `0.7278` | `0.6692` | `75%` | regressed |
| `FC1 REQUANT_PC blocks 2-7,9` | `0.8323` | `0.7524` | `0.7171` | `80%` | interesting but mixed |
| `FC1 REQUANT_PC blocks 5,7` | `0.8360` | `0.7681` | `0.6540` | `70%` | high mean, poor tail / top-1 |
| `FC1 REQUANT_PC block 9` | `0.8338` | `0.7755` | `0.6534` | `80%` | good mean / p10, poor tail |
| `FC1 REQUANT_PC blocks 2-5` | `0.8331` | `0.7410` | `0.7095` | `65%` | better min, poor p10 / top-1 |

### Frozen-Local Conclusion

The frozen-local benchmark never found a promoted winner beyond the narrow `SmoothQuant` stack.

Key takeaways:

- `GELU-from-ACCUM` moved the metric in the wrong direction
- fused attention had signal, but not a clean enough tradeoff
- `DEQUANT_ADD` was a useful structural / performance primitive, but not an accuracy win
- `FC1 REQUANT_PC` had real signal, but no frozen-local subset dominated the `SmoothQuant` winner

## One-Class ImageNet Golden Benchmark

### Early Controls

| Variant | Mean | P10 | Min | Top-1 | Read |
|---|---:|---:|---:|---:|---|
| Plain local baseline | `0.8039` | `0.6962` | `0.4895` | `94%` | starting point |
| Frozen-local best SQ transferred here | `0.7980` | `0.6719` | `0.4809` | `93%` | transfer failed |
| Fused `SOFTMAX_ATTNV block11` | `0.8015` | `0.6909` | `0.4927` | `94%` | not a winner here |

The ImageNet class-0 benchmark became the best signal source for late-MLP PTQ work.

## FC1-Led Late-MLP Tuning

### FC1 REQUANT_PC Block Search

| Variant | Mean | P10 | Min | Top-1 |
|---|---:|---:|---:|---:|
| `FC1 REQUANT_PC block9` | `0.8022` | `0.6962` | `0.5127` | `95%` |
| `FC1 REQUANT_PC blocks8,9` | `0.8042` | `0.7049` | `0.5148` | `94%` |
| `FC1 REQUANT_PC blocks9,10` | `0.8030` | `0.7011` | `0.5251` | `94%` |
| `FC1 REQUANT_PC block10` | `0.8038` | `0.6944` | `0.5173` | `94%` |
| `FC1 REQUANT_PC blocks8,9,10` | `0.8027` | `0.6995` | `0.5256` | `95%` |
| `FC1 REQUANT_PC blocks7,8,9` | `0.8035` | `0.6986` | `0.5149` | `94%` |

The best clean FC1 block subset was:

- `FC1 REQUANT_PC blocks 8,9`

### Late-Logit And GELU Search On Top Of FC1

| Variant | Mean | P10 | Min | Top-1 | Read |
|---|---:|---:|---:|---:|---|
| `final-logit search` on baseline | `0.8040` | `0.6956` | `0.4928` | `94%` | small gain |
| `final-logit search` on `FC1 8,9` | `0.8044` | `0.7043` | `0.5185` | `94%` | helpful |
| `block9 GELU search` on `FC1 8,9` | `0.8052` | `0.7127` | `0.4853` | `95%` | best mean / p10 at the time |
| `block10 GELU search` on `FC1 8,9` | `0.8030` | `0.7014` | `0.5326` | `94%` | best min at the time |
| `block9 GELU + final-logit search` | `0.8053` | `0.7149` | `0.4864` | `95%` | best average-fidelity FC1 stack |
| `block10 GELU + final-logit search` | `0.8032` | `0.7028` | `0.5339` | `94%` | best tail-focused FC1 stack |

### Output-Aware Clipping, AdaRound, And Percentile Calibration On FC1

Starting control for this phase:

- `FC1 REQUANT_PC blocks 8,9`
- mean `0.80418`
- `p10 0.70493`
- min `0.51477`
- top-1 `94%`

Key FC1 weight-tuning results:

| Variant | Mean | P10 | Min | Top-1 |
|---|---:|---:|---:|---:|
| `block8` clipping | `0.80188` | `0.70524` | `0.52251` | `96%` |
| `block9` clipping | `0.80557` | `0.69805` | `0.49144` | `96%` |
| `block9` AdaRound | `0.80427` | `0.70906` | `0.51790` | `94%` |
| `block9` AdaRound + final-logit search | `0.80436` | `0.70772` | `0.52130` | `94%` |

Key percentile activation runs on top of the FC1/AdaRound stack:

| Variant | Mean | P10 | Min | Top-1 |
|---|---:|---:|---:|---:|
| `final_ln:99.9` | `0.80448` | `0.70821` | `0.52213` | `94%` |
| `block9_ln2:99.5` | `0.80470` | `0.70080` | `0.52625` | `95%` |
| `final_ln:99.9 + block9_ln2:99.5` | `0.80475` | `0.70117` | `0.52703` | `95%` |

Local percentile sweep around the winner:

| Variant | Mean | P10 | Min | Top-1 | Read |
|---|---:|---:|---:|---:|---|
| `final_ln:99.7, block9_ln2:99.0` | `0.80583` | `0.70701` | `0.52006` | `95%` | best mean |
| `final_ln:99.8, block9_ln2:99.0` | `0.80579` | `0.70715` | `0.52050` | `95%` | best balanced; became control alias |
| `final_ln:99.9, block9_ln2:99.0` | `0.80577` | `0.70805` | `0.51890` | `95%` | best p10 |
| `final_ln:99.7, block9_ln2:99.5` | `0.80478` | `0.70101` | `0.52851` | `95%` | best min |

This produced the stable control preset:

- `imagenet_class0_ptq4vit_base`
- alias: `imagenet_class0_current_best_ptq`

Control metrics for the rest of the campaign:

- mean `0.8057905436`
- `p10 0.7071453333`
- min `0.5205034614`
- top-1 `95%`

## Bias Correction

### Early Classifier-Only Trial

Bias correction was implemented first as a narrow PTQ pass.

On the plain one-class ImageNet baseline:

- control: `0.803946 / 0.696208 / 0.489485 / 94%`
- classifier-only bias correction: `0.803947 / 0.696052 / 0.489442 / 94%`

Conclusion:

- bias correction was initially neutral
- it should be kept, but only re-tested after weight quantization changed

## PTQ4ViT-Inspired Search Experiments

### Search / Fake-Quant Style `compare_golden` Experiments

Control:

- `0.805791 / 0.707145 / 0.520503 / 95%`

| Variant | Mean | P10 | Min | Top-1 | Read |
|---|---:|---:|---:|---:|---|
| Hessian softmax | `0.804563` | `0.706881` | `0.524663` | `95%` | slight mean regression |
| Hessian GELU | `0.805735` | `0.696834` | `0.509617` | `95%` | worse p10 / min |
| Hessian both | `0.805121` | `0.696538` | `0.504318` | `94%` | regression |
| Twin softmax | `0.800775` | `0.707160` | `0.520987` | `95%` | mean regression |
| Twin softmax + Hessian | `0.800775` | `0.707160` | `0.520987` | `95%` | identical to twin softmax |
| Twin GELU | `0.805735` | `0.696834` | `0.509617` | `95%` | same as Hessian GELU path |
| Twin GELU + Hessian | `0.805735` | `0.696834` | `0.509617` | `95%` | no additional value |

Conclusion:

- PTQ4ViT-inspired search did not beat the current late-MLP control on `compare_golden`

### Software Fake-Quant Runtime Check

This was run through `compare_accuracy.py`, not `compare_golden.py`.

| Variant | Mean Cosine | Min | Top-1 |
|---|---:|---:|---:|
| `W8A8` | `0.915442` | `0.561653` | `96%` |
| `W8A8 + PTQ4ViT twin` | `0.913449` | `0.568205` | `97%` |

Interpretation:

- the twin runtime had a real effect in the fake-quant PyTorch path
- it slightly helped tail / top-1
- it lost mean cosine
- these numbers are not directly comparable with `compare_golden`

## Late-MLP Completion: FC2 Track

This was the final remaining meaningful golden-model PTQ lane, and it produced the best ImageNet result so far.

### FC2 REQUANT_PC Sweep

Control:

- `0.8057905436 / 0.7071453333 / 0.5205034614 / 95%`

| Variant | Mean | P10 | Min | Top-1 |
|---|---:|---:|---:|---:|
| `FC2 REQUANT_PC block9` | `0.803270` | `0.697581` | `0.532310` | `94%` |
| `FC2 REQUANT_PC block10` | `0.804853` | `0.696449` | `0.543397` | `95%` |
| `FC2 REQUANT_PC blocks9,10` | `0.803323` | `0.697807` | `0.531724` | `95%` |

### FC2 Clipping On Top Of FC2 REQUANT_PC

| Variant | Mean | P10 | Min | Top-1 | Read |
|---|---:|---:|---:|---:|---|
| `block9` | `0.805835` | `0.697303` | `0.543434` | `94%` | good mean / min, loses p10 / top-1 |
| `block10` | `0.806111` | `0.712958` | `0.546332` | `95%` | clear winner |
| `blocks9,10` | `0.803922` | `0.696071` | `0.545083` | `94%` | too mixed |

This established the best late-MLP parent:

- `FC2 REQUANT_PC block10`
- `FC2 output-aware clipping block10`

### FC2 AdaRound

| Variant | Mean | P10 | Min | Top-1 | Read |
|---|---:|---:|---:|---:|---|
| `block10 AdaRound` | `0.804950` | `0.707741` | `0.549993` | `95%` | best min, but loses mean / p10 vs clipped parent |

Conclusion:

- keep clipped `FC2 block10`
- do not promote FC2 AdaRound as the default

### Bias Correction Revisited After FC2 Weight Changes

| Variant | Mean | P10 | Min | Top-1 | Read |
|---|---:|---:|---:|---:|---|
| `classifier` | `0.806143` | `0.713498` | `0.546195` | `95%` | best overall winner |
| `late_fc2` | `0.804172` | `0.708301` | `0.517344` | `94%` | regression |
| `classifier + late_fc2` | `0.804206` | `0.708564` | `0.517526` | `94%` | regression |

Important result:

- bias correction only became useful after the FC2 weight path changed
- the winning bias pass was `classifier` only
- `late_fc2` correction remained harmful

### Final Narrow Activation Sweep On Top Of The FC2 Winner

Parent stack:

- `FC2 REQUANT_PC block10`
- `FC2 clipping block10`
- `classifier bias correction`

Sweep:

| Variant | Mean | P10 | Min | Top-1 |
|---|---:|---:|---:|---:|
| `block10_ln2:99.0` | `0.803603` | `0.706968` | `0.520680` | `95%` |
| `block10_ln2:99.5` | `0.801978` | `0.703682` | `0.526581` | `94%` |
| `block10_ln2:99.8` | `0.803431` | `0.705000` | `0.524773` | `95%` |

Conclusion:

- the additional `block10_ln2` percentile sweep hurt the winning stack
- the FC2 winner should be kept without this extra activation override

## True Simulator-Side Twin-Uniform Runtime Emulation

The twin-uniform runtime path was finally pushed into the golden-model simulator using compiler-manifest per-PC specs, without adding new ISA opcodes.

The following variants were run on the ImageNet class-0 control:

- softmax twin on `block11`
- GELU twin on `block9`
- GELU twin on `block10`
- softmax `block11` + GELU `block9`
- softmax `block11` + GELU `block10`

All five runs were exactly neutral on aggregate metrics:

- mean `0.8057905436`
- `p10 0.7071453333`
- min `0.5205034614`
- top-1 `95%`

Conclusion:

- the runtime twin path is now implemented correctly in the simulator
- but it does not buy anything on this benchmark
- no frozen-local rerun or ISA promotion is justified from this line

## Current Winners

### Frozen-Local Golden Benchmark

Best retained preset:

- `current_best_sq_ln2_fc1_b0_8_10`
- metrics: `0.8314 / 0.7671 / 0.6816 / 90%`

### ImageNet Class-0 Golden Benchmark

Best retained stack:

- `imagenet_class0_current_best_ptq`
- plus `FC2 REQUANT_PC block10`
- plus `FC2 output-aware clipping block10`
- plus `classifier bias correction`

Metrics:

- mean `0.8061434031`
- `p10 0.7134982347`
- min `0.5461951494`
- top-1 `95%`

This is the best golden-model result obtained on the one-class ImageNet benchmark during the campaign.

## Main Negative Results

These directions were implemented, tested, and are not recommended as promoted defaults:

- broad `SmoothQuant` transfer from frozen-local to ImageNet one-class
- current `GELU-from-ACCUM`
- `DEQUANT_ADD` as a standalone accuracy-improving primitive
- broad fused `SOFTMAX_ATTNV`
- broad or naive PTQ4ViT-inspired search
- simulator runtime twin-uniform on the current benchmark
- `late_fc2` bias correction
- extra `block10_ln2` percentile tuning on top of the FC2 winner

## Final Read

The cumulative picture is now clear:

- the most reliable gains came from late-MLP PTQ work, not from new nonlinearity runtime formats
- `softmax` and `GELU` are still the main error amplifiers, but directly changing their runtime handling did not outperform better upstream weight quantization on this accelerator stack
- the strongest golden-model improvements came from:
  - selective `SmoothQuant` on frozen-local
  - selective `FC1 REQUANT_PC` and late-node calibration on ImageNet class-0
  - final late-MLP completion with `FC2 REQUANT_PC + clipping + classifier bias correction`

As of 2026-04-01, the software codebase has solid benchmark coverage, a reproducible experiment surface, and one clear promoted winner per main golden-model benchmark.
