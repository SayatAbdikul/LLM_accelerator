#!/usr/bin/env python3
"""Compare FP32 PyTorch inference vs INT8 golden model (hardware simulator).

Loads a frozen local COCO benchmark image cache, runs both:
  1. FP32 DeiT-tiny via PyTorch (reference)
  2. INT8 compiled program via the TACCEL golden model simulator

Reports top-K predictions side-by-side and accuracy metrics.

Usage:
    python3 compare_golden.py [--max-images 5] [--top-k 5]
"""
import argparse
import copy
import sys
import os
import json
import time
import math
import re
import numpy as np
import torch
from PIL import Image
import requests
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoImageProcessor, AutoConfig, AutoModelForImageClassification

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from taccel.compiler.compiler import Compiler
from taccel.compiler.graph_extract import (
    DEPTH,
    EMBED_DIM,
    NUM_HEADS,
    NUM_PATCHES,
    SEQ_LEN,
    PATCH_SIZE,
    extract_deit_tiny,
)
from taccel.assembler.assembler import ProgramBinary
from taccel.golden_model import Simulator, MachineState
from taccel.quantizer.bias_correction import (
    compute_bias_corrections,
    resolve_bias_correction_targets,
)
from taccel.quantizer.quantize import quantize_tensor, quantize_weights
from taccel.quantizer.calibrate import calibrate_model, collect_layer_inputs
from taccel.quantizer.hessian_guided import (
    gelu_fc2_hessian_diag,
    softmax_attn_v_hessian_diag,
    weighted_quant_error_score,
)
from taccel.quantizer.smooth_quant import apply_smooth_quant, compute_smooth_factors
from taccel.quantizer.twin_uniform import (
    quantize_dequant_gelu_twin,
    quantize_dequant_softmax_twin,
)
from taccel.compiler.tiler import pad_dim

MODEL_NAME = "facebook/deit-tiny-patch16-224"
WEIGHTS_PATH = "pytorch_model.bin"

COCO_VAL_IDS = [
    39769, 139, 285, 632, 724, 776, 785, 872, 1000, 1296,
    1353, 1503, 1761, 2006, 2153, 2473, 2685, 3501, 3845, 5037,
    6209, 6673, 7406, 8031, 9156, 10185, 11325, 12661, 13765, 14829,
    15991, 17097, 18304, 19492, 20750, 21873, 22988, 24139, 25376, 26612,
    27855, 28991, 30127, 31364, 32540, 33784, 34906, 36071, 37299, 38542,
    39783, 40915, 42067, 43298, 44531, 45772, 46995, 48126, 49370, 50591,
    51744, 52980, 54112, 55371, 56590, 57822, 58994, 60235, 61473, 62691,
    63824, 65067, 66299, 67514, 68760, 69982, 71119, 72354, 73590, 74821,
    76044, 77295, 78431, 79663, 80890, 82115, 83379, 84597, 85731, 86980,
    88122, 89354, 90591, 91730, 92975, 94108, 95344, 96592, 97731, 98970,
]
COCO_BASE = "http://images.cocodataset.org/val2017/{:012d}.jpg"
FROZEN_EVAL_IMAGE_IDS = [
    139, 285, 632, 724, 776, 785, 872, 1000, 1296, 1353,
    1503, 1761, 2006, 2153, 2473, 2685, 3501, 3845, 5037, 39769,
]
FROZEN_CALIBRATION_IMAGE_IDS = [
    397133, 37777, 252219, 87038, 174482, 403385, 6818, 480985, 458054, 331352,
    296649, 386912, 502136, 491497, 184791, 348881, 289393, 522713, 181666, 17627,
]
QKV_PROJECTIONS = ("query", "key", "value")
LOCAL_FROZEN_IMAGE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "images",
    "frozen_benchmark",
)
CATS_DOGS_IMAGE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "images",
    "cats and dogs",
)
IMAGENET_ONE_CLASS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "images",
    "imagenet_one_class",
)
IMAGENET_CLASS0_IMAGE_DIR = os.path.join(
    IMAGENET_ONE_CLASS_DIR,
    "000_tench_Tinca_tinca",
)
LOCAL_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _cats_dogs_sort_key(name: str):
    stem = os.path.splitext(os.path.basename(name))[0].lower()
    match = re.fullmatch(r"(cat|dog)(\d+)", stem)
    if match:
        label_order = 0 if match.group(1) == "cat" else 1
        return (label_order, int(match.group(2)), stem)
    return (2, 0, stem)


def _discover_local_flat_sample_ids(image_root: str, sort_key=None):
    if not os.path.isdir(image_root):
        return tuple()
    names = [
        entry
        for entry in os.listdir(image_root)
        if os.path.isfile(os.path.join(image_root, entry))
        and os.path.splitext(entry)[1].lower() in LOCAL_IMAGE_EXTENSIONS
    ]
    key_fn = sort_key or (lambda name: os.path.basename(name).lower())
    return tuple(os.path.splitext(name)[0] for name in sorted(names, key=key_fn))


def _discover_cats_dogs_sample_ids(image_root: str = CATS_DOGS_IMAGE_DIR):
    return _discover_local_flat_sample_ids(image_root=image_root, sort_key=_cats_dogs_sort_key)


# Keep separate frozen evaluation and calibration sets locally so benchmark
# wins do not depend on evaluating over the same images used to calibrate.
LOCAL_FROZEN_EVAL_IMAGE_IDS = tuple(FROZEN_EVAL_IMAGE_IDS)
LOCAL_FROZEN_CALIBRATION_IMAGE_IDS = tuple(FROZEN_CALIBRATION_IMAGE_IDS)
CATS_DOGS_SAMPLE_IDS = _discover_cats_dogs_sample_ids()
IMAGENET_CLASS0_SAMPLE_IDS = _discover_local_flat_sample_ids(IMAGENET_CLASS0_IMAGE_DIR)
CURRENT_BEST_SMOOTHQUANT_BLOCKS = tuple([0, 1, 2, 3, 4, 5, 6, 7, 8, 10])
DIAGNOSTIC_PRESETS = {
    "baseline_frozen_local": {
        "description": "Frozen local baseline with the canonical eval/calibration split",
        "benchmark": {
            "benchmark_dataset": "frozen_coco",
            "benchmark_image_source": "local",
            "local_benchmark_image_dir": LOCAL_FROZEN_IMAGE_DIR,
            "eval_image_ids": list(LOCAL_FROZEN_EVAL_IMAGE_IDS),
            "calibration_image_ids": list(LOCAL_FROZEN_CALIBRATION_IMAGE_IDS),
        },
        "compile_args": {
            "softmax_calibration": "max",
            "softmax_percentile": 99.0,
            "softmax_min_prob": 1e-4,
            "softmax_max_prob": 1.0,
            "final_logit_calibration": "off",
            "softmax_search_heads": "",
            "softmax_search_objective": "local_prob",
            "attn_v_calibration": "off",
            "attn_v_percentile": 99.0,
            "attn_v_safety_margin": 1.10,
            "attn_v_search_blocks": "",
            "attn_v_search_objective": "local_attn_v",
            "gelu_from_accum": False,
            "gelu_from_accum_blocks": "",
            "dequant_add_residual1_blocks": "",
            "dequant_add_residual1_scale_mode": "max",
            "dequant_add_residual1_scale_percentile": 99.9,
            "dequant_add_residual1_scale_alpha": 1.0,
            "fused_softmax_attnv": False,
            "fused_softmax_attnv_blocks": "",
            "fused_softmax_attnv_accum_out_proj": False,
            "per_head_qkv_calibration": False,
            "value_head_calibration": "off",
            "gelu_output_calibration": "off",
            "gelu_search_blocks": "9,10,11",
            "smoothquant_targets": "off",
            "smoothquant_alpha": 0.5,
            "smoothquant_blocks": "",
            "requant_pc_qkv": False,
            "requant_pc_qkv_blocks": "",
            "requant_pc_qkv_heads": "",
            "requant_pc_qkv_projections": "all",
            "requant_pc_qkv_exclude": "",
            "requant_pc_fc1": False,
            "requant_pc_fc1_blocks": "",
            "requant_pc_out_proj": False,
            "fold_cls_pos_embed": False,
        },
        "trace": {
            "trace_worst_k": 5,
            "trace_image_ids": "",
        },
    },
    "current_best_sq_ln2_fc1_b0_8_10": {
        "description": "Current best SmoothQuant variant on the frozen local split",
        "benchmark": {
            "benchmark_dataset": "frozen_coco",
            "benchmark_image_source": "local",
            "local_benchmark_image_dir": LOCAL_FROZEN_IMAGE_DIR,
            "eval_image_ids": list(LOCAL_FROZEN_EVAL_IMAGE_IDS),
            "calibration_image_ids": list(LOCAL_FROZEN_CALIBRATION_IMAGE_IDS),
        },
        "compile_args": {
            "softmax_calibration": "max",
            "softmax_percentile": 99.0,
            "softmax_min_prob": 1e-4,
            "softmax_max_prob": 1.0,
            "final_logit_calibration": "off",
            "softmax_search_heads": "",
            "softmax_search_objective": "local_prob",
            "attn_v_calibration": "off",
            "attn_v_percentile": 99.0,
            "attn_v_safety_margin": 1.10,
            "attn_v_search_blocks": "",
            "attn_v_search_objective": "local_attn_v",
            "gelu_from_accum": False,
            "gelu_from_accum_blocks": "",
            "dequant_add_residual1_blocks": "",
            "dequant_add_residual1_scale_mode": "max",
            "dequant_add_residual1_scale_percentile": 99.9,
            "dequant_add_residual1_scale_alpha": 1.0,
            "fused_softmax_attnv": False,
            "fused_softmax_attnv_blocks": "",
            "fused_softmax_attnv_accum_out_proj": False,
            "per_head_qkv_calibration": False,
            "value_head_calibration": "off",
            "gelu_output_calibration": "off",
            "gelu_search_blocks": "9,10,11",
            "smoothquant_targets": "ln2_fc1",
            "smoothquant_alpha": 0.5,
            "smoothquant_blocks": ",".join(str(block_idx) for block_idx in CURRENT_BEST_SMOOTHQUANT_BLOCKS),
            "requant_pc_qkv": False,
            "requant_pc_qkv_blocks": "",
            "requant_pc_qkv_heads": "",
            "requant_pc_qkv_projections": "all",
            "requant_pc_qkv_exclude": "",
            "requant_pc_fc1": False,
            "requant_pc_fc1_blocks": "",
            "requant_pc_out_proj": False,
            "fold_cls_pos_embed": False,
        },
        "trace": {
            "trace_worst_k": 5,
            "trace_image_ids": "",
        },
    },
    "cats_dogs_local_all": {
        "description": "Local cats/dogs benchmark using all 200 images for both calibration and evaluation",
        "benchmark": {
            "benchmark_dataset": "cats_dogs_local",
            "benchmark_image_source": "local",
            "local_benchmark_image_dir": CATS_DOGS_IMAGE_DIR,
            "eval_image_ids": list(CATS_DOGS_SAMPLE_IDS),
            "calibration_image_ids": list(CATS_DOGS_SAMPLE_IDS),
        },
        "compile_args": {
            "softmax_calibration": "max",
            "softmax_percentile": 99.0,
            "softmax_min_prob": 1e-4,
            "softmax_max_prob": 1.0,
            "final_logit_calibration": "off",
            "softmax_search_heads": "",
            "softmax_search_objective": "local_prob",
            "attn_v_calibration": "off",
            "attn_v_percentile": 99.0,
            "attn_v_safety_margin": 1.10,
            "attn_v_search_blocks": "",
            "attn_v_search_objective": "local_attn_v",
            "gelu_from_accum": False,
            "gelu_from_accum_blocks": "",
            "dequant_add_residual1_blocks": "",
            "dequant_add_residual1_scale_mode": "max",
            "dequant_add_residual1_scale_percentile": 99.9,
            "dequant_add_residual1_scale_alpha": 1.0,
            "fused_softmax_attnv": False,
            "fused_softmax_attnv_blocks": "",
            "fused_softmax_attnv_accum_out_proj": False,
            "per_head_qkv_calibration": False,
            "value_head_calibration": "off",
            "gelu_output_calibration": "off",
            "gelu_search_blocks": "9,10,11",
            "smoothquant_targets": "off",
            "smoothquant_alpha": 0.5,
            "smoothquant_blocks": "",
            "requant_pc_qkv": False,
            "requant_pc_qkv_blocks": "",
            "requant_pc_qkv_heads": "",
            "requant_pc_qkv_projections": "all",
            "requant_pc_qkv_exclude": "",
            "requant_pc_fc1": False,
            "requant_pc_fc1_blocks": "",
            "requant_pc_out_proj": False,
            "fold_cls_pos_embed": False,
        },
        "trace": {
            "trace_worst_k": 5,
            "trace_image_ids": "",
        },
    },
    "imagenet_class0_local_all": {
        "description": "Local one-class ImageNet benchmark on tench images using all samples for calibration and evaluation",
        "benchmark": {
            "benchmark_dataset": "local_flat",
            "benchmark_image_source": "local",
            "local_benchmark_image_dir": IMAGENET_CLASS0_IMAGE_DIR,
            "eval_image_ids": list(IMAGENET_CLASS0_SAMPLE_IDS),
            "calibration_image_ids": list(IMAGENET_CLASS0_SAMPLE_IDS),
        },
        "compile_args": {
            "softmax_calibration": "max",
            "softmax_percentile": 99.0,
            "softmax_min_prob": 1e-4,
            "softmax_max_prob": 1.0,
            "final_logit_calibration": "off",
            "softmax_search_heads": "",
            "softmax_search_objective": "local_prob",
            "attn_v_calibration": "off",
            "attn_v_percentile": 99.0,
            "attn_v_safety_margin": 1.10,
            "attn_v_search_blocks": "",
            "attn_v_search_objective": "local_attn_v",
            "gelu_from_accum": False,
            "gelu_from_accum_blocks": "",
            "dequant_add_residual1_blocks": "",
            "dequant_add_residual1_scale_mode": "max",
            "dequant_add_residual1_scale_percentile": 99.9,
            "dequant_add_residual1_scale_alpha": 1.0,
            "fused_softmax_attnv": False,
            "fused_softmax_attnv_blocks": "",
            "fused_softmax_attnv_accum_out_proj": False,
            "per_head_qkv_calibration": False,
            "value_head_calibration": "off",
            "gelu_output_calibration": "off",
            "gelu_search_blocks": "9,10,11",
            "smoothquant_targets": "off",
            "smoothquant_alpha": 0.5,
            "smoothquant_blocks": "",
            "requant_pc_qkv": False,
            "requant_pc_qkv_blocks": "",
            "requant_pc_qkv_heads": "",
            "requant_pc_qkv_projections": "all",
            "requant_pc_qkv_exclude": "",
            "requant_pc_fc1": False,
            "requant_pc_fc1_blocks": "",
            "requant_pc_out_proj": False,
            "fold_cls_pos_embed": False,
        },
        "trace": {
            "trace_worst_k": 5,
            "trace_image_ids": "",
        },
    },
    "imagenet_class0_current_best_sq_ln2_fc1_b0_8_10": {
        "description": "Current best SmoothQuant variant on the local one-class ImageNet tench benchmark",
        "benchmark": {
            "benchmark_dataset": "local_flat",
            "benchmark_image_source": "local",
            "local_benchmark_image_dir": IMAGENET_CLASS0_IMAGE_DIR,
            "eval_image_ids": list(IMAGENET_CLASS0_SAMPLE_IDS),
            "calibration_image_ids": list(IMAGENET_CLASS0_SAMPLE_IDS),
        },
        "compile_args": {
            "softmax_calibration": "max",
            "softmax_percentile": 99.0,
            "softmax_min_prob": 1e-4,
            "softmax_max_prob": 1.0,
            "final_logit_calibration": "off",
            "softmax_search_heads": "",
            "softmax_search_objective": "local_prob",
            "attn_v_calibration": "off",
            "attn_v_percentile": 99.0,
            "attn_v_safety_margin": 1.10,
            "attn_v_search_blocks": "",
            "attn_v_search_objective": "local_attn_v",
            "gelu_from_accum": False,
            "gelu_from_accum_blocks": "",
            "dequant_add_residual1_blocks": "",
            "dequant_add_residual1_scale_mode": "max",
            "dequant_add_residual1_scale_percentile": 99.9,
            "dequant_add_residual1_scale_alpha": 1.0,
            "fused_softmax_attnv": False,
            "fused_softmax_attnv_blocks": "",
            "fused_softmax_attnv_accum_out_proj": False,
            "per_head_qkv_calibration": False,
            "value_head_calibration": "off",
            "gelu_output_calibration": "off",
            "gelu_search_blocks": "9,10,11",
            "smoothquant_targets": "ln2_fc1",
            "smoothquant_alpha": 0.5,
            "smoothquant_blocks": ",".join(str(block_idx) for block_idx in CURRENT_BEST_SMOOTHQUANT_BLOCKS),
            "requant_pc_qkv": False,
            "requant_pc_qkv_blocks": "",
            "requant_pc_qkv_heads": "",
            "requant_pc_qkv_projections": "all",
            "requant_pc_qkv_exclude": "",
            "requant_pc_fc1": False,
            "requant_pc_fc1_blocks": "",
            "requant_pc_out_proj": False,
            "fold_cls_pos_embed": False,
        },
        "trace": {
            "trace_worst_k": 5,
            "trace_image_ids": "",
        },
    },
    "imagenet_class0_ptq4vit_base": {
        "description": "Current best late-MLP ImageNet class-0 control used for PTQ4ViT-inspired experiments",
        "benchmark": {
            "benchmark_dataset": "local_flat",
            "benchmark_image_source": "local",
            "local_benchmark_image_dir": IMAGENET_CLASS0_IMAGE_DIR,
            "eval_image_ids": list(IMAGENET_CLASS0_SAMPLE_IDS),
            "calibration_image_ids": list(IMAGENET_CLASS0_SAMPLE_IDS),
        },
        "compile_args": {
            "softmax_calibration": "max",
            "softmax_percentile": 99.0,
            "softmax_min_prob": 1e-4,
            "softmax_max_prob": 1.0,
            "final_logit_calibration": "off",
            "bias_correction": False,
            "bias_correction_layers": "classifier",
            "act_percentile_nodes": "final_ln:99.8,block9_ln2:99.0",
            "output_aware_clipping_fc1_blocks": "9",
            "output_aware_clipping_fc2_blocks": "",
            "output_aware_clipping_classifier": False,
            "output_aware_clipping_candidates": 31,
            "output_aware_clipping_alpha_min": 0.5,
            "adaround_fc1_blocks": "9",
            "adaround_fc2_blocks": "",
            "softmax_search_heads": "",
            "softmax_search_objective": "local_prob",
            "attn_v_calibration": "off",
            "attn_v_percentile": 99.0,
            "attn_v_safety_margin": 1.10,
            "attn_v_search_blocks": "",
            "attn_v_search_objective": "local_attn_v",
            "gelu_output_calibration": "off",
            "gelu_search_blocks": "9,10,11",
            "gelu_search_objective": "downstream_residual2",
            "hessian_calibration_images": 100,
            "hessian_target_nodes": "",
            "twin_uniform_softmax_blocks": "",
            "twin_uniform_gelu_blocks": "",
            "twin_uniform_mode": "off",
            "twin_uniform_disable_hessian": False,
            "gelu_from_accum": False,
            "gelu_from_accum_blocks": "",
            "dequant_add_residual1_blocks": "",
            "dequant_add_residual1_scale_mode": "max",
            "dequant_add_residual1_scale_percentile": 99.9,
            "dequant_add_residual1_scale_alpha": 1.0,
            "fused_softmax_attnv": False,
            "fused_softmax_attnv_blocks": "",
            "fused_softmax_attnv_accum_out_proj": False,
            "per_head_qkv_calibration": False,
            "value_head_calibration": "off",
            "smoothquant_targets": "off",
            "smoothquant_alpha": 0.5,
            "smoothquant_blocks": "",
            "requant_pc_qkv": False,
            "requant_pc_qkv_blocks": "",
            "requant_pc_qkv_heads": "",
            "requant_pc_qkv_projections": "all",
            "requant_pc_qkv_exclude": "",
            "requant_pc_fc1": True,
            "requant_pc_fc1_blocks": "8,9",
            "requant_pc_fc2": False,
            "requant_pc_fc2_blocks": "",
            "requant_pc_out_proj": False,
            "requant_pc_out_proj_blocks": "",
            "fold_cls_pos_embed": False,
        },
        "trace": {
            "trace_worst_k": 5,
            "trace_image_ids": "",
        },
    },
    "imagenet_class0_current_best_ptq": {
        "description": "Alias for the canonical ImageNet class-0 PTQ control",
        "benchmark": {
            "benchmark_dataset": "local_flat",
            "benchmark_image_source": "local",
            "local_benchmark_image_dir": IMAGENET_CLASS0_IMAGE_DIR,
            "eval_image_ids": list(IMAGENET_CLASS0_SAMPLE_IDS),
            "calibration_image_ids": list(IMAGENET_CLASS0_SAMPLE_IDS),
        },
        "compile_args": {
            "softmax_calibration": "max",
            "softmax_percentile": 99.0,
            "softmax_min_prob": 1e-4,
            "softmax_max_prob": 1.0,
            "final_logit_calibration": "off",
            "bias_correction": False,
            "bias_correction_layers": "classifier",
            "act_percentile_nodes": "final_ln:99.8,block9_ln2:99.0",
            "output_aware_clipping_fc1_blocks": "9",
            "output_aware_clipping_fc2_blocks": "",
            "output_aware_clipping_classifier": False,
            "output_aware_clipping_candidates": 31,
            "output_aware_clipping_alpha_min": 0.5,
            "adaround_fc1_blocks": "9",
            "adaround_fc2_blocks": "",
            "softmax_search_heads": "",
            "softmax_search_objective": "local_prob",
            "attn_v_calibration": "off",
            "attn_v_percentile": 99.0,
            "attn_v_safety_margin": 1.10,
            "attn_v_search_blocks": "",
            "attn_v_search_objective": "local_attn_v",
            "gelu_output_calibration": "off",
            "gelu_search_blocks": "9,10,11",
            "gelu_search_objective": "downstream_residual2",
            "hessian_calibration_images": 100,
            "hessian_target_nodes": "",
            "twin_uniform_softmax_blocks": "",
            "twin_uniform_gelu_blocks": "",
            "twin_uniform_mode": "off",
            "twin_uniform_disable_hessian": False,
            "gelu_from_accum": False,
            "gelu_from_accum_blocks": "",
            "dequant_add_residual1_blocks": "",
            "dequant_add_residual1_scale_mode": "max",
            "dequant_add_residual1_scale_percentile": 99.9,
            "dequant_add_residual1_scale_alpha": 1.0,
            "fused_softmax_attnv": False,
            "fused_softmax_attnv_blocks": "",
            "fused_softmax_attnv_accum_out_proj": False,
            "per_head_qkv_calibration": False,
            "value_head_calibration": "off",
            "smoothquant_targets": "off",
            "smoothquant_alpha": 0.5,
            "smoothquant_blocks": "",
            "requant_pc_qkv": False,
            "requant_pc_qkv_blocks": "",
            "requant_pc_qkv_heads": "",
            "requant_pc_qkv_projections": "all",
            "requant_pc_qkv_exclude": "",
            "requant_pc_fc1": True,
            "requant_pc_fc1_blocks": "8,9",
            "requant_pc_fc2": False,
            "requant_pc_fc2_blocks": "",
            "requant_pc_out_proj": False,
            "requant_pc_out_proj_blocks": "",
            "fold_cls_pos_embed": False,
        },
        "trace": {
            "trace_worst_k": 5,
            "trace_image_ids": "",
        },
    },
}


def get_diagnostic_preset(name: str):
    if not name:
        return None
    if name not in DIAGNOSTIC_PRESETS:
        raise KeyError(f"Unknown diagnostic preset '{name}'")
    return copy.deepcopy(DIAGNOSTIC_PRESETS[name])


def explicit_cli_dest_overrides(argv=None):
    """Return argparse dest names explicitly set on the CLI."""
    argv = list(sys.argv[1:] if argv is None else argv)
    overrides = set()
    for token in argv:
        if not token.startswith("--") or token == "--":
            continue
        option = token.split("=", 1)[0]
        overrides.add(option[2:].replace("-", "_"))
    return overrides


def apply_diagnostic_preset(args, explicit_overrides=None):
    """Apply a named preset directly onto an argparse namespace."""
    if not getattr(args, "diagnostic_preset", ""):
        return None
    if explicit_overrides is None:
        explicit_overrides = explicit_cli_dest_overrides()
    preset = get_diagnostic_preset(args.diagnostic_preset)
    if "max_images" not in explicit_overrides:
        args.max_images = len(preset["benchmark"]["eval_image_ids"])
    if "calibration_images" not in explicit_overrides:
        args.calibration_images = len(preset["benchmark"]["calibration_image_ids"])
    for field, value in preset["benchmark"].items():
        if field in {"eval_image_ids", "calibration_image_ids"}:
            continue
        if field not in explicit_overrides:
            setattr(args, field, copy.deepcopy(value))
    for field, value in preset["compile_args"].items():
        if field not in explicit_overrides:
            setattr(args, field, copy.deepcopy(value))
    for field, value in preset.get("trace", {}).items():
        if field not in explicit_overrides:
            setattr(args, field, copy.deepcopy(value))
    return preset


def build_run_config(
    args,
    *,
    eval_image_ids,
    calibration_image_ids,
    smoothquant_blocks,
    requant_pc_qkv_selection,
):
    """Capture the exact benchmark + compiler configuration used for one run."""
    return {
        "diagnostic_preset": getattr(args, "diagnostic_preset", "") or None,
        "benchmark": {
            "dataset": args.benchmark_dataset,
            "image_source": args.benchmark_image_source,
            "local_image_dir": args.local_benchmark_image_dir,
            "eval_image_ids": list(eval_image_ids),
            "calibration_image_ids": list(calibration_image_ids),
            "populate_local_benchmark_cache": bool(args.populate_local_benchmark_cache),
        },
        "compiler_flags": {
            "softmax_calibration": args.softmax_calibration,
            "softmax_percentile": float(args.softmax_percentile),
            "softmax_min_prob": float(args.softmax_min_prob),
            "softmax_max_prob": float(args.softmax_max_prob),
            "final_logit_calibration": args.final_logit_calibration,
            "bias_correction": bool(args.bias_correction),
            "bias_correction_layers": args.bias_correction_layers,
            "act_percentile_nodes": args.act_percentile_nodes,
            "output_aware_clipping_fc1_blocks": args.output_aware_clipping_fc1_blocks,
            "output_aware_clipping_fc2_blocks": args.output_aware_clipping_fc2_blocks,
            "output_aware_clipping_classifier": bool(args.output_aware_clipping_classifier),
            "output_aware_clipping_candidates": int(args.output_aware_clipping_candidates),
            "output_aware_clipping_alpha_min": float(args.output_aware_clipping_alpha_min),
            "adaround_fc1_blocks": args.adaround_fc1_blocks,
            "adaround_fc2_blocks": args.adaround_fc2_blocks,
            "softmax_search_heads": args.softmax_search_heads,
            "softmax_search_objective": args.softmax_search_objective,
            "attn_v_calibration": args.attn_v_calibration,
            "attn_v_percentile": float(args.attn_v_percentile),
            "attn_v_safety_margin": float(args.attn_v_safety_margin),
            "attn_v_search_blocks": args.attn_v_search_blocks,
            "attn_v_search_objective": args.attn_v_search_objective,
            "gelu_from_accum": bool(args.gelu_from_accum),
            "gelu_from_accum_blocks": args.gelu_from_accum_blocks,
            "dequant_add_residual1_blocks": args.dequant_add_residual1_blocks,
            "dequant_add_residual1_scale_mode": args.dequant_add_residual1_scale_mode,
            "dequant_add_residual1_scale_percentile": float(args.dequant_add_residual1_scale_percentile),
            "dequant_add_residual1_scale_alpha": float(args.dequant_add_residual1_scale_alpha),
            "fused_softmax_attnv": bool(args.fused_softmax_attnv),
            "fused_softmax_attnv_blocks": args.fused_softmax_attnv_blocks,
            "fused_softmax_attnv_accum_out_proj": bool(args.fused_softmax_attnv_accum_out_proj),
            "per_head_qkv_calibration": bool(args.per_head_qkv_calibration),
            "value_head_calibration": args.value_head_calibration,
            "gelu_output_calibration": args.gelu_output_calibration,
            "gelu_search_blocks": args.gelu_search_blocks,
            "gelu_search_objective": args.gelu_search_objective,
            "hessian_calibration_images": int(args.hessian_calibration_images),
            "hessian_target_nodes": args.hessian_target_nodes,
            "twin_uniform_softmax_blocks": args.twin_uniform_softmax_blocks,
            "twin_uniform_gelu_blocks": args.twin_uniform_gelu_blocks,
            "twin_uniform_mode": args.twin_uniform_mode,
            "twin_uniform_disable_hessian": bool(args.twin_uniform_disable_hessian),
            "smoothquant_targets": args.smoothquant_targets,
            "smoothquant_alpha": float(args.smoothquant_alpha),
            "smoothquant_blocks": (
                sorted(int(block_idx) for block_idx in smoothquant_blocks)
                if smoothquant_blocks is not None else None
            ),
            "requant_pc_qkv": bool(args.requant_pc_qkv),
            "requant_pc_qkv_blocks": args.requant_pc_qkv_blocks,
            "requant_pc_qkv_heads": args.requant_pc_qkv_heads,
            "requant_pc_qkv_projections": args.requant_pc_qkv_projections,
            "requant_pc_qkv_exclude": args.requant_pc_qkv_exclude,
            "requant_pc_fc1": bool(args.requant_pc_fc1),
            "requant_pc_fc1_blocks": args.requant_pc_fc1_blocks,
            "requant_pc_fc2": bool(args.requant_pc_fc2),
            "requant_pc_fc2_blocks": args.requant_pc_fc2_blocks,
            "requant_pc_qkv_selection": (
                [
                    {"block": int(block), "projection": proj, "head": int(head)}
                    for block, proj, head in sorted(requant_pc_qkv_selection)
                ]
                if requant_pc_qkv_selection is not None else None
            ),
            "requant_pc_out_proj": bool(args.requant_pc_out_proj),
            "requant_pc_out_proj_blocks": args.requant_pc_out_proj_blocks,
            "fold_cls_pos_embed": bool(args.fold_cls_pos_embed),
        },
        "trace_flags": {
            "trace_worst_k": int(args.trace_worst_k),
            "trace_image_ids": parse_csv_token_list(args.trace_image_ids),
            "trace_output": args.trace_output,
            "replay_early_attn": bool(args.replay_early_attn),
            "replay_blocks": args.replay_blocks,
            "replay_late_attn": bool(args.replay_late_attn),
            "replay_attn_blocks": args.replay_attn_blocks,
            "replay_late_mlp": bool(args.replay_late_mlp),
            "replay_mlp_blocks": args.replay_mlp_blocks,
        },
    }


def preset_compile_kwargs(preset):
    """Translate a diagnostics preset into compile_model kwargs."""
    if preset is None:
        return {}
    compile_args = preset["compile_args"]
    return {
        "softmax_mode": compile_args["softmax_calibration"],
        "softmax_percentile": compile_args["softmax_percentile"],
        "softmax_min_prob": compile_args["softmax_min_prob"],
        "softmax_max_prob": compile_args["softmax_max_prob"],
        "final_logit_mode": compile_args["final_logit_calibration"],
        "bias_correction": compile_args.get("bias_correction", False),
        "bias_correction_layers": compile_args.get("bias_correction_layers", ""),
        "activation_percentile_nodes": parse_activation_percentile_overrides(
            compile_args.get("act_percentile_nodes", "")
        ),
        "output_aware_clipping_fc1_blocks": parse_csv_int_set(
            compile_args.get("output_aware_clipping_fc1_blocks", "")
        ),
        "output_aware_clipping_fc2_blocks": parse_csv_int_set(
            compile_args.get("output_aware_clipping_fc2_blocks", "")
        ),
        "output_aware_clipping_classifier": bool(
            compile_args.get("output_aware_clipping_classifier", False)
        ),
        "output_aware_clipping_candidates": int(compile_args.get("output_aware_clipping_candidates", 25)),
        "output_aware_clipping_alpha_min": float(compile_args.get("output_aware_clipping_alpha_min", 0.5)),
        "adaround_fc1_blocks": parse_csv_int_set(
            compile_args.get("adaround_fc1_blocks", "")
        ),
        "adaround_fc2_blocks": parse_csv_int_set(
            compile_args.get("adaround_fc2_blocks", "")
        ),
        "softmax_search_heads": (
            {
                tuple(int(part) for part in item.split(":", 1))
                for item in compile_args["softmax_search_heads"].split(",")
                if item.strip()
            }
            if compile_args["softmax_search_heads"] else None
        ),
        "softmax_search_objective": compile_args["softmax_search_objective"],
        "attn_v_mode": compile_args["attn_v_calibration"],
        "attn_v_percentile": compile_args["attn_v_percentile"],
        "attn_v_safety_margin": compile_args["attn_v_safety_margin"],
        "attn_v_search_blocks": (
            {int(part) for part in compile_args["attn_v_search_blocks"].split(",") if part.strip()}
            if compile_args["attn_v_search_blocks"] else None
        ),
        "attn_v_search_objective": compile_args["attn_v_search_objective"],
        "gelu_output_mode": compile_args["gelu_output_calibration"],
        "gelu_search_blocks": (
            {int(part) for part in compile_args["gelu_search_blocks"].split(",") if part.strip()}
            if compile_args["gelu_search_blocks"] else None
        ),
        "gelu_search_objective": compile_args.get("gelu_search_objective", "downstream_residual2"),
        "hessian_calibration_images": int(compile_args.get("hessian_calibration_images", 0) or 0),
        "hessian_target_nodes": compile_args.get("hessian_target_nodes", ""),
        "twin_uniform_softmax_blocks": parse_csv_int_set(
            compile_args.get("twin_uniform_softmax_blocks", "")
        ),
        "twin_uniform_gelu_blocks": parse_csv_int_set(
            compile_args.get("twin_uniform_gelu_blocks", "")
        ),
        "twin_uniform_mode": compile_args.get("twin_uniform_mode", "off"),
        "twin_uniform_disable_hessian": compile_args.get("twin_uniform_disable_hessian", False),
        "value_head_mode": compile_args["value_head_calibration"],
        "per_head_qkv_calibration": compile_args["per_head_qkv_calibration"],
        "gelu_from_accum": compile_args["gelu_from_accum"],
        "gelu_from_accum_blocks": parse_csv_int_set(compile_args["gelu_from_accum_blocks"]),
        "dequant_add_residual1_blocks": parse_csv_int_set(compile_args["dequant_add_residual1_blocks"]),
        "dequant_add_residual1_scale_mode": compile_args["dequant_add_residual1_scale_mode"],
        "dequant_add_residual1_scale_percentile": compile_args["dequant_add_residual1_scale_percentile"],
        "dequant_add_residual1_scale_alpha": compile_args["dequant_add_residual1_scale_alpha"],
        "fused_softmax_attnv_blocks": (
            set(range(DEPTH))
            if compile_args.get("fused_softmax_attnv", False)
            else parse_csv_int_set(compile_args["fused_softmax_attnv_blocks"])
        ),
        "fused_softmax_attnv_accum_out_proj": compile_args.get("fused_softmax_attnv_accum_out_proj", False),
        "smoothquant_targets": compile_args["smoothquant_targets"],
        "smoothquant_alpha": compile_args["smoothquant_alpha"],
        "smoothquant_blocks": parse_csv_int_set(compile_args["smoothquant_blocks"]),
        "requant_pc_qkv": compile_args["requant_pc_qkv"],
        "requant_pc_qkv_selection": build_requant_pc_qkv_selection(
            blocks_text=compile_args["requant_pc_qkv_blocks"],
            heads_text=compile_args["requant_pc_qkv_heads"],
            projections_text=compile_args["requant_pc_qkv_projections"],
            exclude_text=compile_args["requant_pc_qkv_exclude"],
        ),
        "requant_pc_fc1": compile_args["requant_pc_fc1"],
        "requant_pc_fc1_blocks": parse_csv_int_set(compile_args["requant_pc_fc1_blocks"]),
        "requant_pc_fc2": compile_args.get("requant_pc_fc2", False),
        "requant_pc_fc2_blocks": parse_csv_int_set(compile_args.get("requant_pc_fc2_blocks", "")),
        "requant_pc_out_proj": compile_args["requant_pc_out_proj"],
        "requant_pc_out_proj_blocks": parse_csv_int_set(compile_args.get("requant_pc_out_proj_blocks", "")),
    }


def load_model():
    state_dict = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
    config = AutoConfig.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_config(config)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, state_dict


def fetch_image(img_id: int):
    url = COCO_BASE.format(img_id)
    try:
        r = requests.get(url, timeout=15, stream=True)
        if r.status_code == 200:
            return img_id, Image.open(BytesIO(r.content)).convert("RGB")
    except Exception:
        pass
    return img_id, None


def local_frozen_image_path(img_id: int, image_root: str = LOCAL_FROZEN_IMAGE_DIR) -> str:
    return os.path.join(image_root, f"{img_id:012d}.jpg")


def populate_local_image_cache(image_ids, label: str, image_root: str = LOCAL_FROZEN_IMAGE_DIR):
    """Download any missing frozen benchmark images into the local cache."""
    os.makedirs(image_root, exist_ok=True)
    missing_ids = [
        img_id for img_id in image_ids
        if not os.path.exists(local_frozen_image_path(img_id, image_root=image_root))
    ]
    if not missing_ids:
        print(f"  Local frozen image cache already has all {len(image_ids)} {label} images")
        return

    print(f"  Populating local frozen cache with {len(missing_ids)} missing {label} images...")
    collected = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch_image, img_id): img_id for img_id in missing_ids}
        for future in as_completed(futures):
            img_id, img = future.result()
            if img is not None:
                collected[img_id] = img
    missing = [img_id for img_id in missing_ids if img_id not in collected]
    if missing:
        raise RuntimeError(f"Missing frozen benchmark image downloads for COCO ids: {missing}")

    for img_id in missing_ids:
        path = local_frozen_image_path(img_id, image_root=image_root)
        collected[img_id].save(path, format="JPEG", quality=95)


def load_local_images(image_ids, label: str, image_root: str = LOCAL_FROZEN_IMAGE_DIR):
    """Load frozen benchmark images from the local cache only."""
    print(f"  Loading {len(image_ids)} frozen {label} images from local cache...")
    loaded = []
    missing = []
    for img_id in image_ids:
        path = local_frozen_image_path(img_id, image_root=image_root)
        if not os.path.exists(path):
            missing.append(img_id)
            continue
        with Image.open(path) as img:
            loaded.append((img_id, img.convert("RGB")))
    if missing:
        raise RuntimeError(
            "Missing local frozen benchmark images for COCO ids: "
            f"{missing}. Run with --populate-local-benchmark-cache first."
        )
    return loaded


def collect_images(image_ids, label: str):
    print(f"  Downloading {len(image_ids)} fixed COCO val2017 {label} images...")
    collected = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch_image, img_id): img_id for img_id in image_ids}
        for future in as_completed(futures):
            img_id, img = future.result()
            if img is not None:
                collected[img_id] = img
    missing = [img_id for img_id in image_ids if img_id not in collected]
    if missing:
        raise RuntimeError(f"Missing {label} image downloads for COCO ids: {missing}")
    return [(img_id, collected[img_id]) for img_id in image_ids]


def infer_cats_dogs_label(sample_id: str):
    sample_id = str(sample_id).lower()
    if sample_id.startswith("cat"):
        return "cat"
    if sample_id.startswith("dog"):
        return "dog"
    return None


def discover_local_flat_samples(image_root: str, *, label_fn=None, sort_key=None):
    """Return stable metadata records for a flat local image folder."""
    sample_ids = _discover_local_flat_sample_ids(image_root=image_root, sort_key=sort_key)
    samples = []
    for sample_id in sample_ids:
        ext = None
        for suffix in LOCAL_IMAGE_EXTENSIONS:
            candidate = os.path.join(image_root, f"{sample_id}{suffix}")
            if os.path.exists(candidate):
                ext = suffix
                break
        if ext is None:
            continue
        samples.append({
            "sample_id": sample_id,
            "image_path": os.path.join(image_root, f"{sample_id}{ext}"),
            "dataset_label": label_fn(sample_id) if label_fn is not None else None,
        })
    return samples


def discover_cats_dogs_samples(image_root: str = CATS_DOGS_IMAGE_DIR):
    """Return stable metadata records for the local cats/dogs image folder."""
    return discover_local_flat_samples(
        image_root=image_root,
        label_fn=infer_cats_dogs_label,
        sort_key=_cats_dogs_sort_key,
    )


def load_flat_local_images(sample_ids, label: str, image_root: str):
    """Load images from a flat local directory keyed by filename stem."""
    print(f"  Loading {len(sample_ids)} local {label} images from {image_root}...")
    sample_map = {
        sample["sample_id"]: sample
        for sample in discover_local_flat_samples(image_root=image_root)
    }
    loaded = []
    missing = []
    for sample_id in sample_ids:
        record = sample_map.get(sample_id)
        if record is None:
            missing.append(sample_id)
            continue
        with Image.open(record["image_path"]) as img:
            loaded.append({
                "sample_id": record["sample_id"],
                "dataset_label": record["dataset_label"],
                "image_path": record["image_path"],
                "image": img.convert("RGB"),
            })
    if missing:
        raise RuntimeError(
            "Missing local dataset images for sample ids: "
            + ", ".join(str(sample_id) for sample_id in missing)
        )
    return loaded


def parse_csv_int_set(text: str):
    """Parse a comma-separated integer list.

    Returns None when the input is empty so callers can distinguish "all" from
    an explicit filter.
    """
    if not text or not text.strip():
        return None
    return {int(part.strip()) for part in text.split(",") if part.strip()}


def parse_csv_int_list(text: str):
    """Parse a comma-separated integer list while preserving order."""
    if not text or not text.strip():
        return []
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_csv_token_list(text: str):
    """Parse a comma-separated token list while preserving order."""
    if not text or not text.strip():
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_activation_percentile_overrides(text: str):
    """Parse a comma-separated node:percentile list."""
    if not text or not text.strip():
        return None
    overrides = {}
    for item in text.split(","):
        token = item.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(
                "Activation percentile overrides must use node:percentile entries"
            )
        node_name, percentile_text = token.split(":", 1)
        node_name = node_name.strip()
        if node_name == "cls_extract":
            node_name = "final_ln"
        try:
            percentile = float(percentile_text)
        except ValueError as exc:
            raise ValueError(f"Invalid activation percentile override '{token}'") from exc
        if not (0.0 < percentile <= 100.0):
            raise ValueError(f"Percentile for '{node_name}' must be in (0, 100]")
        overrides[node_name] = percentile
    return overrides or None


def resolve_activation_percentile_targets(node_percentiles):
    """Map IR node percentile overrides to calibration module names."""
    if not node_percentiles:
        return {}
    resolved = {}
    for node_name, percentile in sorted(node_percentiles.items()):
        if node_name == "final_ln":
            resolved[node_name] = {
                "module_name": "vit.layernorm",
                "scale_keys": ["final_ln", "cls_extract"],
                "percentile": float(percentile),
            }
            continue
        match = re.fullmatch(r"block(\d+)_ln([12])", node_name)
        if match:
            block_idx = int(match.group(1))
            if not (0 <= block_idx < DEPTH):
                raise ValueError(f"Activation percentile block out of range: {node_name}")
            module_name = (
                f"vit.encoder.layer.{block_idx}.layernorm_before"
                if match.group(2) == "1"
                else f"vit.encoder.layer.{block_idx}.layernorm_after"
            )
            resolved[node_name] = {
                "module_name": module_name,
                "scale_keys": [node_name],
                "percentile": float(percentile),
            }
            continue
        raise ValueError(
            f"Unsupported activation percentile node '{node_name}'. "
            "Supported nodes: final_ln, cls_extract, blockN_ln1, blockN_ln2."
        )
    return resolved


def resolve_explicit_sample_ids(explicit_ids, available_sample_ids):
    """Resolve CLI string sample ids against the typed ids used in one run."""
    available_map = {str(sample_id): sample_id for sample_id in available_sample_ids}
    resolved = []
    invalid = []
    for token in explicit_ids or []:
        sample_id = available_map.get(str(token))
        if sample_id is None:
            invalid.append(token)
            continue
        resolved.append(sample_id)
    return resolved, invalid


def select_trace_image_ids(results, explicit_ids=None, trace_worst_k: int = 0):
    """Build the ordered list of evaluation sample ids to trace."""
    ordered = []
    seen = set()

    for img_id in explicit_ids or []:
        if img_id not in seen:
            ordered.append(img_id)
            seen.add(img_id)

    if trace_worst_k > 0:
        ranked = sorted(results, key=lambda item: item["cosine_sim"])
        for item in ranked[:trace_worst_k]:
            img_id = item.get("sample_id", item["img_id"])
            if img_id not in seen:
                ordered.append(img_id)
                seen.add(img_id)

    return ordered


def parse_qkv_projection_set(text: str):
    """Parse comma-separated Q/K/V projection names."""
    if not text or not text.strip() or text.strip().lower() == "all":
        return None
    projections = {part.strip().lower() for part in text.split(",") if part.strip()}
    invalid = sorted(projections.difference(QKV_PROJECTIONS))
    if invalid:
        raise ValueError(
            "Unknown Q/K/V projection(s): "
            + ", ".join(invalid)
            + f" (expected one of {', '.join(QKV_PROJECTIONS)})"
        )
    return projections


def parse_qkv_triplet_set(text: str):
    """Parse block:projection:head triplets for selective REQUANT_PC rollout."""
    specs = set()
    if not text or not text.strip():
        return specs
    for item in text.split(","):
        spec = item.strip()
        if not spec:
            continue
        parts = [part.strip() for part in spec.split(":")]
        if len(parts) != 3:
            raise ValueError(
                f"Invalid Q/K/V selection triplet '{spec}' (expected block:projection:head)"
            )
        block_idx = int(parts[0])
        projection = parts[1].lower()
        head_idx = int(parts[2])
        if projection not in QKV_PROJECTIONS:
            raise ValueError(
                f"Unknown Q/K/V projection '{projection}' in '{spec}' "
                f"(expected one of {', '.join(QKV_PROJECTIONS)})"
            )
        if not (0 <= block_idx < DEPTH):
            raise ValueError(f"Q/K/V block index out of range in '{spec}'")
        if not (0 <= head_idx < NUM_HEADS):
            raise ValueError(f"Q/K/V head index out of range in '{spec}'")
        specs.add((block_idx, projection, head_idx))
    return specs


def build_requant_pc_qkv_selection(
    blocks_text: str = "",
    heads_text: str = "",
    projections_text: str = "",
    exclude_text: str = "",
):
    """Build an explicit (block, projection, head) selection set or None for all."""
    blocks = parse_csv_int_set(blocks_text)
    heads = parse_csv_int_set(heads_text)
    projections = parse_qkv_projection_set(projections_text)
    exclude = parse_qkv_triplet_set(exclude_text)

    if blocks is not None:
        invalid_blocks = sorted(block for block in blocks if not (0 <= block < DEPTH))
        if invalid_blocks:
            raise ValueError(f"Q/K/V block indices out of range: {invalid_blocks}")
    if heads is not None:
        invalid_heads = sorted(head for head in heads if not (0 <= head < NUM_HEADS))
        if invalid_heads:
            raise ValueError(f"Q/K/V head indices out of range: {invalid_heads}")

    if blocks is None and heads is None and projections is None and not exclude:
        return None

    if blocks is None:
        blocks = set(range(DEPTH))
    if heads is None:
        heads = set(range(NUM_HEADS))
    if projections is None:
        projections = set(QKV_PROJECTIONS)

    selection = {
        (block_idx, projection, head_idx)
        for block_idx in blocks
        for projection in projections
        for head_idx in heads
    }
    selection.difference_update(exclude)
    return selection


def fp32_inference(model, processor, image):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits.squeeze(0).cpu().numpy()


def build_fc1_weight_quantization_overrides(
    model,
    sample_inputs: list,
    clip_block_indices,
    *,
    adaround_block_indices=None,
    requant_pc_fc1: bool,
    requant_pc_fc1_blocks,
    n_candidates: int,
    alpha_min: float,
):
    """Collect activation samples and build FC1 quantization overrides."""
    return build_block_dense_weight_quantization_overrides(
        model,
        sample_inputs,
        clip_block_indices,
        module_suffix="intermediate.dense",
        requant_pc_enabled=requant_pc_fc1,
        requant_pc_blocks=requant_pc_fc1_blocks,
        adaround_block_indices=adaround_block_indices,
        n_candidates=n_candidates,
        alpha_min=alpha_min,
    )


def build_block_dense_weight_quantization_overrides(
    model,
    sample_inputs: list,
    clip_block_indices,
    *,
    module_suffix: str,
    requant_pc_enabled: bool,
    requant_pc_blocks,
    adaround_block_indices=None,
    n_candidates: int,
    alpha_min: float,
):
    """Collect activation samples and build block-dense quantization overrides."""
    clip_blocks = set(clip_block_indices or [])
    adaround_blocks = set(adaround_block_indices or [])
    target_blocks = sorted(clip_blocks.union(adaround_blocks))
    if not target_blocks:
        return {}
    module_names = [f"vit.encoder.layer.{block_idx}.{module_suffix}" for block_idx in target_blocks]
    inputs_by_module = collect_layer_inputs(model, sample_inputs, module_names)
    overrides = {}
    for block_idx in target_blocks:
        module_name = f"vit.encoder.layer.{block_idx}.{module_suffix}"
        overrides[f"{module_name}.weight"] = {
            "mode": "output_aware_clipping",
            "per_channel": bool(
                requant_pc_enabled and (
                    requant_pc_blocks is None or block_idx in requant_pc_blocks
                )
            ),
            "n_candidates": int(n_candidates if block_idx in clip_blocks else 1),
            "alpha_min": float(alpha_min if block_idx in clip_blocks else 1.0),
            "calibration_inputs": inputs_by_module[module_name],
            "adaround": bool(block_idx in adaround_blocks),
        }
    return overrides


def build_classifier_weight_quantization_override(
    model,
    sample_inputs: list,
    *,
    enabled: bool,
    n_candidates: int,
    alpha_min: float,
):
    """Collect classifier inputs and build a per-tensor clipping override."""
    if not enabled:
        return {}
    inputs_by_module = collect_layer_inputs(model, sample_inputs, ["classifier"])
    return {
        "classifier.weight": {
            "mode": "output_aware_clipping",
            "per_channel": False,
            "n_candidates": int(n_candidates),
            "alpha_min": float(alpha_min),
            "calibration_inputs": inputs_by_module["classifier"],
            "adaround": False,
        }
    }


def fold_pos_embed_int8(base_rows_fp32: np.ndarray, pos_rows_fp32: np.ndarray, act_scale: float) -> np.ndarray:
    """Fold position embeddings into activations in FP32, then quantize once."""
    combined_fp32 = base_rows_fp32.astype(np.float32) + pos_rows_fp32.astype(np.float32)
    return np.clip(np.round(combined_fp32 / act_scale), -128, 127).astype(np.int8)


def patch_embed_int8(model, processor, image, act_scale=None, fold_cls_pos_embed: bool = False):
    """Run patch embedding on CPU and quantize to INT8 for the accelerator.

    Returns `(patches_int8, cls_int8, scale)`.
    `patches_int8` is [NUM_PATCHES, EMBED_DIM] and `cls_int8` is [1, EMBED_DIM]
    when `fold_cls_pos_embed=True`, else `None`.

    B3 optimisation: patch position embeddings are added in FP32 before
    quantisation (one INT8 quantisation step instead of two).  The compiled
    program must have the corresponding DRAM patch-pos-embed rows zeroed out
    so the in-program VADD becomes a no-op for those rows (see golden_inference).

    act_scale must match the embedding scale used in the compiled program for
    CLS token and position embeddings (cal_scales["pos_embed_add"]).
    """
    if act_scale is None:
        act_scale = 14.0 / 127.0  # safe default: covers patch+pos_embed sum ~13.8
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]  # [1, 3, 224, 224]

    with torch.no_grad():
        # Extract the patch embedding layer
        patch_embed = model.vit.embeddings.patch_embeddings
        # Conv2d: [1, 3, 224, 224] -> [1, 192, 14, 14]
        embedded = patch_embed.projection(pixel_values)
        # Reshape to [196, 192]
        embedded = embedded.flatten(2).transpose(1, 2).squeeze(0)  # [196, 192]

        # B3: fold patch position embeddings into host preprocessing.
        # pos_embed shape: [1, 197, 192]; rows 1:197 are the patch positions.
        pos_embed = model.vit.embeddings.position_embeddings  # [1, 197, 192]
        cls_token = model.vit.embeddings.cls_token  # [1, 1, 192]
        cls_pos = pos_embed[0, :1].detach().numpy().astype(np.float32)  # [1, 192]
        patch_pos = pos_embed[0, 1:NUM_PATCHES + 1].detach().numpy().astype(np.float32)  # [196, 192]

    patches_fp32 = embedded.numpy().astype(np.float32)
    patches_int8 = fold_pos_embed_int8(patches_fp32, patch_pos, act_scale)
    cls_int8 = None
    if fold_cls_pos_embed:
        cls_fp32 = cls_token[0].detach().numpy().astype(np.float32)  # [1, 192]
        cls_int8 = fold_pos_embed_int8(cls_fp32, cls_pos, act_scale)
    return patches_int8, cls_int8, act_scale


def golden_inference(program, patches_int8, cls_int8=None, num_classes=1000, trace_nodes=None):
    """Run the golden model simulator and return INT32 logits plus optional traces."""
    state = MachineState()
    sim = Simulator(state)
    sim.load_program(program)
    if trace_nodes:
        sim.enable_trace(trace_nodes)

    # Write embedded patches to DRAM at program.input_offset.
    # The program's DMA instructions load them from there to ABUF during execution.
    M, N = patches_int8.shape  # [196, 192]
    N_pad = pad_dim(N)
    if N < N_pad:
        patches_padded = np.zeros((M, N_pad), dtype=np.int8)
        patches_padded[:M, :N] = patches_int8
    else:
        patches_padded = patches_int8

    patch_bytes = patches_padded.tobytes()
    dram_off = program.input_offset
    state.dram[dram_off:dram_off + len(patch_bytes)] = patch_bytes

    # Host-side CLS folding: overwrite the stored cls_token row if the program
    # exposes its DRAM offset. Legacy binaries leave this metadata at 0.
    if cls_int8 is not None and getattr(program, "cls_token_dram_offset", 0) > 0:
        cls_row = np.asarray(cls_int8, dtype=np.int8)
        if cls_row.ndim == 1:
            cls_row = cls_row.reshape(1, -1)
        cls_bytes = cls_row[0, :EMBED_DIM].tobytes()
        cls_off = program.cls_token_dram_offset
        state.dram[cls_off:cls_off + len(cls_bytes)] = cls_bytes

    # If CLS pos_embed was folded into the host-side row, zero row 0 so the
    # in-program VADD becomes a no-op for CLS too.
    if cls_int8 is not None and getattr(program, "pos_embed_cls_dram_offset", 0) > 0:
        off = program.pos_embed_cls_dram_offset
        state.dram[off:off + EMBED_DIM] = bytes(EMBED_DIM)

    # B3: patch pos_embed was folded into patches on the host side; zero out
    # those DRAM rows so the in-program VADD becomes a no-op for patch rows
    # (prevents double-counting pos_embed).
    if program.pos_embed_patch_dram_offset > 0:
        patch_pos_size = NUM_PATCHES * pad_dim(EMBED_DIM)
        off = program.pos_embed_patch_dram_offset
        state.dram[off:off + patch_pos_size] = bytes(patch_pos_size)

    # Run simulation
    count = sim.run()

    # Extract logits from ACCUM buffer (INT32)
    logits_int32 = state.accum[:num_classes].copy()
    trace_payload = sim.get_trace_payload() if trace_nodes else None
    return logits_int32.astype(np.float32), count, state.cycle_count, trace_payload


def default_trace_node_order():
    """Return traced node names in execution order."""
    supported_ops = {
        "pos_embed_add",
        "layernorm",
        "matmul",
        "matmul_qkt",
        "softmax",
        "matmul_attn_v",
        "concat_heads",
        "vadd",
        "gelu",
    }
    graph = extract_deit_tiny()
    return [node.name for node in graph if node.op in supported_ops]


def _as_f32(tensor):
    return tensor.detach().cpu().numpy().astype(np.float32)


def fp32_trace(model, processor, image):
    """Run a manual DeiT forward pass and capture tensors matching the IR node names."""
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]
    traces = {}

    with torch.no_grad():
        x = model.vit.embeddings(pixel_values)
        traces["pos_embed_add"] = _as_f32(x[0])

        prev = x
        for block_idx, layer in enumerate(model.vit.encoder.layer):
            b = f"block{block_idx}"
            attn = layer.attention.attention

            ln1 = layer.layernorm_before(prev)
            traces[f"{b}_ln1"] = _as_f32(ln1[0])

            q = _reshape_heads(attn.query(ln1), attn)
            k = _reshape_heads(attn.key(ln1), attn)
            v = _reshape_heads(attn.value(ln1), attn)
            qkt = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(attn.attention_head_size)
            probs = torch.softmax(qkt, dim=-1)
            context = torch.matmul(probs, v)

            for h in range(NUM_HEADS):
                traces[f"{b}_head{h}_query"] = _as_f32(q[0, h])
                traces[f"{b}_head{h}_key"] = _as_f32(k[0, h])
                traces[f"{b}_head{h}_value"] = _as_f32(v[0, h])
                traces[f"{b}_head{h}_qkt"] = _as_f32(qkt[0, h])
                traces[f"{b}_head{h}_softmax"] = _as_f32(probs[0, h])
                traces[f"{b}_head{h}_attn_v"] = _as_f32(context[0, h])

            concat = context.permute(0, 2, 1, 3).reshape(1, SEQ_LEN, EMBED_DIM)
            traces[f"{b}_concat"] = _as_f32(concat[0])

            out_proj = layer.attention.output.dense(concat)
            traces[f"{b}_out_proj"] = _as_f32(out_proj[0])

            residual1 = out_proj + prev
            traces[f"{b}_residual1"] = _as_f32(residual1[0])

            ln2 = layer.layernorm_after(residual1)
            traces[f"{b}_ln2"] = _as_f32(ln2[0])

            fc1 = layer.intermediate.dense(ln2)
            traces[f"{b}_fc1"] = _as_f32(fc1[0])

            gelu = layer.intermediate.intermediate_act_fn(fc1)
            traces[f"{b}_gelu"] = _as_f32(gelu[0])

            fc2 = layer.output.dense(gelu)
            traces[f"{b}_fc2"] = _as_f32(fc2[0])

            residual2 = fc2 + residual1
            traces[f"{b}_residual2"] = _as_f32(residual2[0])
            prev = residual2

        final_ln = model.vit.layernorm(prev)
        traces["final_ln"] = _as_f32(final_ln[0])

        cls = final_ln[:, :1, :]
        traces["cls_extract"] = _as_f32(cls[0])

        logits = model.classifier(cls[:, 0, :])
        traces["classifier"] = _as_f32(logits)

    return logits.squeeze(0).cpu().numpy(), traces


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for arbitrary-shaped tensors."""
    a_flat = a.reshape(-1).astype(np.float32)
    b_flat = b.reshape(-1).astype(np.float32)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))


def summarize_results(results):
    """Compute stable benchmark summary metrics."""
    cosines = np.array([r["cosine_sim"] for r in results], dtype=np.float32)
    cycles = np.array([r["cycles"] for r in results], dtype=np.float32)
    top1 = np.mean([r["top1_match"] for r in results]) if results else 0.0
    top5 = np.mean([r["top5_overlap"] for r in results]) if results else 0.0
    return {
        "n_images": len(results),
        "top1_agreement": float(top1),
        "top5_overlap_avg": float(top5),
        "cosine_sim_avg": float(np.mean(cosines)) if len(cosines) else 0.0,
        "cosine_sim_p10": float(np.percentile(cosines, 10)) if len(cosines) else 0.0,
        "cosine_sim_min": float(np.min(cosines)) if len(cosines) else 0.0,
        "avg_cycles": float(np.mean(cycles)) if len(cycles) else 0.0,
    }


def quantize_dequant_tensor(tensor: np.ndarray, scale: float) -> np.ndarray:
    """Apply symmetric INT8 quantize-dequant with the given scale."""
    if scale <= 0:
        return np.zeros_like(tensor, dtype=np.float32)
    q = np.clip(np.round(tensor / scale), -128, 127).astype(np.int8)
    return q.astype(np.float32) * np.float32(scale)


def quantize_dequant_softmax_candidate(
    tensor: np.ndarray,
    max_prob: float,
    *,
    twin_uniform_mode: str = "off",
):
    """Quantize-dequant a softmax tensor with either uniform or twin-uniform."""
    if twin_uniform_mode == "paper_exact":
        qdq, meta = quantize_dequant_softmax_twin(
            tensor,
            max_prob,
            return_metadata=True,
        )
        return qdq.astype(np.float32), meta
    scale = max(float(max_prob), 1e-8) / 127.0
    q = np.clip(np.round(np.asarray(tensor, dtype=np.float32) / scale), 0, 127).astype(np.int8)
    meta = {
        "mode": "uniform",
        "scale": float(scale),
        "saturation_rate": float(np.mean(q == 127)) if q.size else 0.0,
        "zero_fraction": float(np.mean(q == 0)) if q.size else 1.0,
    }
    return q.astype(np.float32) * np.float32(scale), meta


def quantize_dequant_gelu_candidate(
    tensor: np.ndarray,
    positive_range_max: float,
    *,
    twin_uniform_mode: str = "off",
    negative_extent: float | None = None,
):
    """Quantize-dequant a GELU tensor with either uniform or twin-uniform."""
    if twin_uniform_mode == "paper_exact":
        qdq, meta = quantize_dequant_gelu_twin(
            tensor,
            positive_range_max,
            negative_extent=negative_extent,
            return_metadata=True,
        )
        return qdq.astype(np.float32), meta
    scale = max(float(positive_range_max), 1e-8) / 127.0
    qdq = quantize_dequant_tensor(tensor, scale)
    meta = quantization_diagnostics(tensor, scale)
    meta["mode"] = "uniform"
    meta["scale"] = float(scale)
    return qdq.astype(np.float32), meta


def quantization_diagnostics(tensor: np.ndarray, scale: float) -> dict:
    """Measure local INT8 quantize-dequant error for a tensor."""
    if scale <= 0:
        return {
            "qdq_cosine_sim": 0.0,
            "qdq_max_abs_error": float(np.max(np.abs(tensor))) if tensor.size else 0.0,
            "qdq_zero_fraction": 1.0,
            "qdq_saturation_rate": 0.0,
        }

    q = np.clip(np.round(tensor / scale), -128, 127).astype(np.int8)
    dq = q.astype(np.float32) * np.float32(scale)
    return {
        "qdq_cosine_sim": cosine_similarity(tensor, dq),
        "qdq_max_abs_error": float(np.max(np.abs(tensor - dq))),
        "qdq_zero_fraction": float(np.mean(q == 0)),
        "qdq_saturation_rate": float(np.mean((q == 127) | (q == -128))),
    }


def tensor_error_metrics(target: np.ndarray, candidate: np.ndarray) -> dict:
    """Summarize tensor reconstruction quality against a target tensor."""
    rows = min(target.shape[0], candidate.shape[0])
    cols = min(target.shape[1], candidate.shape[1])
    target_crop = target[:rows, :cols].astype(np.float32)
    candidate_crop = candidate[:rows, :cols].astype(np.float32)
    diff = target_crop - candidate_crop
    return {
        "cosine_sim": cosine_similarity(target_crop, candidate_crop),
        "max_abs_error": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "mean_abs_error": float(np.mean(np.abs(diff))) if diff.size else 0.0,
        "mse": float(np.mean(diff * diff)) if diff.size else 0.0,
    }


def replay_attention_head_variants(
    softmax: np.ndarray,
    value: np.ndarray,
    target_attn_v: np.ndarray,
    softmax_scale: float,
    value_scale: float,
    attn_v_scale: float,
):
    """Replay a single attention head with isolated input/output quantization paths."""
    variant_inputs = {
        "fp32_fp32": (
            softmax.astype(np.float32),
            value.astype(np.float32),
        ),
        "qdq_softmax": (
            quantize_dequant_tensor(softmax, softmax_scale),
            value.astype(np.float32),
        ),
        "qdq_value": (
            softmax.astype(np.float32),
            quantize_dequant_tensor(value, value_scale),
        ),
        "qdq_softmax_value": (
            quantize_dequant_tensor(softmax, softmax_scale),
            quantize_dequant_tensor(value, value_scale),
        ),
    }

    reports = {}
    tensors = {}
    for name, (softmax_in, value_in) in variant_inputs.items():
        raw = (softmax_in @ value_in).astype(np.float32)
        attn_v_qdq = quantize_dequant_tensor(raw, attn_v_scale)
        reports[name] = {
            "raw_metrics": tensor_error_metrics(target_attn_v, raw),
            "attn_v_qdq_metrics": tensor_error_metrics(target_attn_v, attn_v_qdq),
        }
        tensors[name] = {
            "raw": raw,
            "attn_v_qdq": attn_v_qdq,
        }

    return reports, tensors


def gelu_activation_fp32(x: np.ndarray) -> np.ndarray:
    """Reference GELU in FP32 matching the simulator path."""
    from scipy.special import erf

    x = x.astype(np.float32)
    sqrt2 = np.float32(np.sqrt(np.float32(2.0)))
    return x * np.float32(0.5) * (np.float32(1.0) + erf(x / sqrt2).astype(np.float32))


def replay_block_downstream_variants(
    variant_head_outputs: dict,
    concat_target: np.ndarray,
    out_proj_target: np.ndarray,
    out_proj_weight: np.ndarray,
    out_proj_bias: np.ndarray,
    out_proj_scale: float,
):
    """Replay concat and out_proj from per-head attn@V variant tensors."""
    reports = {}
    for variant_name, heads in variant_head_outputs.items():
        concat = np.concatenate(heads, axis=-1).astype(np.float32)
        out_proj_raw = (concat @ out_proj_weight.T + out_proj_bias).astype(np.float32)
        out_proj_qdq = quantize_dequant_tensor(out_proj_raw, out_proj_scale)
        reports[variant_name] = {
            "concat_metrics": tensor_error_metrics(concat_target, concat),
            "out_proj_raw_metrics": tensor_error_metrics(out_proj_target, out_proj_raw),
            "out_proj_qdq_metrics": tensor_error_metrics(out_proj_target, out_proj_qdq),
        }
    return reports


def replay_mlp_block_variants(
    fc1: np.ndarray,
    gelu_target: np.ndarray,
    fc2_target: np.ndarray,
    residual1_target: np.ndarray,
    residual2_target: np.ndarray,
    fc1_scale: float,
    gelu_scale: float,
    fc2_scale: float,
    fc2_weight: np.ndarray,
    fc2_bias: np.ndarray,
):
    """Replay MLP variants through GELU, FC2, and residual2."""
    fc1_qdq = quantize_dequant_tensor(fc1, fc1_scale)
    gelu_from_fc1_qdq = gelu_activation_fp32(fc1_qdq)

    variant_gelu = {
        "fp32_fp32": gelu_target.astype(np.float32),
        "qdq_fc1": gelu_from_fc1_qdq,
        "qdq_gelu_out": quantize_dequant_tensor(gelu_target, gelu_scale),
        "qdq_fc1_gelu_out": quantize_dequant_tensor(gelu_from_fc1_qdq, gelu_scale),
    }

    reports = {}
    for variant_name, gelu_variant in variant_gelu.items():
        fc2_raw = (gelu_variant @ fc2_weight.T + fc2_bias).astype(np.float32)
        fc2_qdq = quantize_dequant_tensor(fc2_raw, fc2_scale)
        residual2 = (residual1_target.astype(np.float32) + fc2_qdq).astype(np.float32)
        reports[variant_name] = {
            "gelu_metrics": tensor_error_metrics(gelu_target, gelu_variant),
            "fc2_qdq_metrics": tensor_error_metrics(fc2_target, fc2_qdq),
            "residual2_metrics": tensor_error_metrics(residual2_target, residual2),
            "gelu_saturation_rate": quantization_diagnostics(gelu_variant, gelu_scale)["qdq_saturation_rate"]
            if variant_name in {"qdq_gelu_out", "qdq_fc1_gelu_out"} else 0.0,
        }
    return reports


def collect_block_replay_tensors(model, sample_inputs: list, block_indices) -> dict:
    """Collect FP32 attention and MLP tensors for selected blocks."""
    block_set = set(block_indices)
    bundles = {block_idx: [] for block_idx in block_indices}

    model.eval()
    with torch.no_grad():
        for inp in sample_inputs:
            pixel_values = inp["pixel_values"] if hasattr(inp, "__getitem__") else inp
            x = model.vit.embeddings(pixel_values)
            prev = x
            for block_idx, layer in enumerate(model.vit.encoder.layer):
                attn = layer.attention.attention
                ln1 = layer.layernorm_before(prev)
                q = _reshape_heads(attn.query(ln1), attn)
                k = _reshape_heads(attn.key(ln1), attn)
                v = _reshape_heads(attn.value(ln1), attn)
                qkt = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(attn.attention_head_size)
                probs = torch.softmax(qkt, dim=-1)
                context = torch.matmul(probs, v)
                concat = context.permute(0, 2, 1, 3).reshape(1, SEQ_LEN, EMBED_DIM)
                out_proj = layer.attention.output.dense(concat)
                residual1 = out_proj + prev
                ln2 = layer.layernorm_after(residual1)
                fc1 = layer.intermediate.dense(ln2)
                gelu = layer.intermediate.intermediate_act_fn(fc1)
                fc2 = layer.output.dense(gelu)
                residual2 = fc2 + residual1

                if block_idx in block_set:
                    bundles[block_idx].append({
                        "heads": [
                            {
                                "softmax": _as_f32(probs[0, h]),
                                "value": _as_f32(v[0, h]),
                                "attn_v": _as_f32(context[0, h]),
                            }
                            for h in range(NUM_HEADS)
                        ],
                        "concat": _as_f32(concat[0]),
                        "out_proj": _as_f32(out_proj[0]),
                        "residual1": _as_f32(residual1[0]),
                        "fc1": _as_f32(fc1[0]),
                        "gelu": _as_f32(gelu[0]),
                        "fc2": _as_f32(fc2[0]),
                        "residual2": _as_f32(residual2[0]),
                    })

                prev = residual2

    return bundles


def calibrate_residual1_block_scales(
    block_replay_samples: dict,
    block_indices,
    *,
    mode: str = "max",
    default_scale: float | None = None,
    percentile: float = 99.9,
    blend_alpha: float = 1.0,
) -> dict:
    """Choose per-block residual1 output scales from FP32 replay tensors."""
    scales = {}
    for block_idx in sorted(set(block_indices or [])):
        samples = block_replay_samples.get(block_idx, [])
        if mode == "default":
            if default_scale is None:
                raise ValueError("default_scale is required for residual1 default mode")
            scales[block_idx] = float(default_scale)
            continue
        if not samples:
            continue
        residual_values = np.concatenate(
            [np.abs(sample["residual1"]).reshape(-1) for sample in samples]
        ).astype(np.float32)
        max_abs = float(np.max(residual_values)) if residual_values.size else 0.0
        max_scale = max(max_abs, 1e-8) / 127.0
        if mode == "max":
            scales[block_idx] = max_scale
            continue
        if mode == "percentile":
            pct_abs = float(np.percentile(residual_values, percentile)) if residual_values.size else 0.0
            scales[block_idx] = max(pct_abs, 1e-8) / 127.0
            continue
        if mode == "blend":
            if default_scale is None:
                raise ValueError("default_scale is required for residual1 blend mode")
            alpha = float(np.clip(blend_alpha, 0.0, 1.0))
            scales[block_idx] = (1.0 - alpha) * float(default_scale) + alpha * max_scale
            continue
        raise ValueError(f"Unsupported residual1 scale mode: {mode}")
    return scales


def replay_early_attention(
    model,
    fp32_traces: dict,
    golden_trace: dict,
    cal_scales: dict,
    block_indices,
):
    """Replay early attention blocks to attribute error to softmax, value, or output scale."""
    golden_tensors = (golden_trace or {}).get("tensors", {})
    blocks = []

    for block_idx in block_indices:
        b = f"block{block_idx}"
        layer = model.vit.encoder.layer[block_idx].attention.output.dense
        out_proj_weight = layer.weight.detach().cpu().numpy().astype(np.float32)
        out_proj_bias = layer.bias.detach().cpu().numpy().astype(np.float32)
        out_proj_scale = cal_scales.get(f"{b}_out_proj", 0.0)

        head_reports = []
        variant_head_outputs = {
            "fp32_fp32": [],
            "qdq_softmax": [],
            "qdq_value": [],
            "qdq_softmax_value": [],
        }

        for head_idx in range(NUM_HEADS):
            softmax = fp32_traces[f"{b}_head{head_idx}_softmax"]
            value = fp32_traces[f"{b}_head{head_idx}_value"]
            target_attn_v = fp32_traces[f"{b}_head{head_idx}_attn_v"]
            softmax_scale = cal_scales.get(f"{b}_head{head_idx}_softmax", 0.0)
            value_scale = cal_scales.get(f"{b}_head{head_idx}_value", 0.0)
            attn_v_scale = cal_scales.get(f"{b}_head{head_idx}_attn_v", 0.0)

            variant_reports, variant_tensors = replay_attention_head_variants(
                softmax,
                value,
                target_attn_v,
                softmax_scale,
                value_scale,
                attn_v_scale,
            )
            for variant_name in variant_head_outputs:
                variant_head_outputs[variant_name].append(variant_tensors[variant_name]["attn_v_qdq"])

            golden_metrics = None
            golden_attn_v = golden_tensors.get(f"{b}_head{head_idx}_attn_v")
            if golden_attn_v is not None:
                golden_metrics = tensor_error_metrics(target_attn_v, golden_attn_v)

            head_reports.append({
                "head_idx": head_idx,
                "softmax_scale": float(softmax_scale),
                "value_scale": float(value_scale),
                "attn_v_scale": float(attn_v_scale),
                "golden_attn_v_metrics": golden_metrics,
                "variants": variant_reports,
            })

        block_variants = replay_block_downstream_variants(
            variant_head_outputs,
            fp32_traces[f"{b}_concat"],
            fp32_traces[f"{b}_out_proj"],
            out_proj_weight,
            out_proj_bias,
            out_proj_scale,
        )

        golden_block_metrics = {}
        golden_concat = golden_tensors.get(f"{b}_concat")
        if golden_concat is not None:
            golden_block_metrics["concat_metrics"] = tensor_error_metrics(
                fp32_traces[f"{b}_concat"],
                golden_concat,
            )
        golden_out_proj = golden_tensors.get(f"{b}_out_proj")
        if golden_out_proj is not None:
            golden_block_metrics["out_proj_metrics"] = tensor_error_metrics(
                fp32_traces[f"{b}_out_proj"],
                golden_out_proj,
            )

        worst_head = min(
            head_reports,
            key=lambda item: (
                item["golden_attn_v_metrics"]["cosine_sim"]
                if item["golden_attn_v_metrics"] is not None else 1.0
            ),
        )

        blocks.append({
            "block_idx": block_idx,
            "out_proj_scale": float(out_proj_scale),
            "golden_block_metrics": golden_block_metrics,
            "worst_head_idx": int(worst_head["head_idx"]),
            "heads": head_reports,
            "block_variants": block_variants,
        })

    return {"blocks": blocks}


def summarize_early_attention_replay(replay_reports: list) -> list:
    """Aggregate early-attention replay metrics across traced images."""
    aggregate = {}
    for report in replay_reports:
        for block in report.get("blocks", []):
            bucket = aggregate.setdefault(block["block_idx"], {
                "golden_out_proj_cosines": [],
                "variant_out_proj_qdq_cosines": {},
                "golden_worst_head_cosines": [],
                "variant_worst_head_attn_v_qdq_cosines": {},
                "worst_head_counts": {},
            })

            golden_out_proj = block.get("golden_block_metrics", {}).get("out_proj_metrics")
            if golden_out_proj:
                bucket["golden_out_proj_cosines"].append(golden_out_proj["cosine_sim"])

            worst_head_idx = block["worst_head_idx"]
            bucket["worst_head_counts"][worst_head_idx] = bucket["worst_head_counts"].get(worst_head_idx, 0) + 1
            worst_head = next(head for head in block["heads"] if head["head_idx"] == worst_head_idx)
            if worst_head.get("golden_attn_v_metrics"):
                bucket["golden_worst_head_cosines"].append(
                    worst_head["golden_attn_v_metrics"]["cosine_sim"]
                )

            for variant_name, metrics in block["block_variants"].items():
                bucket["variant_out_proj_qdq_cosines"].setdefault(variant_name, []).append(
                    metrics["out_proj_qdq_metrics"]["cosine_sim"]
                )
                bucket["variant_worst_head_attn_v_qdq_cosines"].setdefault(variant_name, []).append(
                    worst_head["variants"][variant_name]["attn_v_qdq_metrics"]["cosine_sim"]
                )

    summaries = []
    for block_idx, bucket in sorted(aggregate.items()):
        summaries.append({
            "block_idx": block_idx,
            "mean_golden_out_proj_cosine": float(np.mean(bucket["golden_out_proj_cosines"]))
            if bucket["golden_out_proj_cosines"] else 0.0,
            "mean_golden_worst_head_attn_v_cosine": float(np.mean(bucket["golden_worst_head_cosines"]))
            if bucket["golden_worst_head_cosines"] else 0.0,
            "variant_mean_out_proj_qdq_cosine": {
                name: float(np.mean(vals))
                for name, vals in sorted(bucket["variant_out_proj_qdq_cosines"].items())
            },
            "variant_mean_worst_head_attn_v_qdq_cosine": {
                name: float(np.mean(vals))
                for name, vals in sorted(bucket["variant_worst_head_attn_v_qdq_cosines"].items())
            },
            "worst_head_counts": {
                f"head{head_idx}": int(count)
                for head_idx, count in sorted(bucket["worst_head_counts"].items())
            },
        })
    return summaries


def replay_late_attention(
    model,
    fp32_traces: dict,
    golden_trace: dict,
    cal_scales: dict,
    block_indices,
):
    """Replay late attention blocks with per-head downstream attribution."""
    golden_tensors = (golden_trace or {}).get("tensors", {})
    blocks = []

    for block_idx in block_indices:
        b = f"block{block_idx}"
        layer = model.vit.encoder.layer[block_idx].attention.output.dense
        out_proj_weight = layer.weight.detach().cpu().numpy().astype(np.float32)
        out_proj_bias = layer.bias.detach().cpu().numpy().astype(np.float32)
        out_proj_scale = cal_scales.get(f"{b}_out_proj", 0.0)

        head_reports = []
        head_variant_tensors = []
        all_head_variant_outputs = {
            "fp32_fp32": [],
            "qdq_softmax": [],
            "qdq_value": [],
            "qdq_softmax_value": [],
        }

        for head_idx in range(NUM_HEADS):
            softmax = fp32_traces[f"{b}_head{head_idx}_softmax"]
            value = fp32_traces[f"{b}_head{head_idx}_value"]
            target_attn_v = fp32_traces[f"{b}_head{head_idx}_attn_v"]
            softmax_scale = cal_scales.get(f"{b}_head{head_idx}_softmax", 0.0)
            value_scale = cal_scales.get(f"{b}_head{head_idx}_value", 0.0)
            attn_v_scale = cal_scales.get(f"{b}_head{head_idx}_attn_v", 0.0)

            variant_reports, variant_tensors = replay_attention_head_variants(
                softmax,
                value,
                target_attn_v,
                softmax_scale,
                value_scale,
                attn_v_scale,
            )
            head_variant_tensors.append(variant_tensors)
            for variant_name in all_head_variant_outputs:
                all_head_variant_outputs[variant_name].append(
                    variant_tensors[variant_name]["attn_v_qdq"]
                )

            golden_metrics = None
            golden_attn_v = golden_tensors.get(f"{b}_head{head_idx}_attn_v")
            if golden_attn_v is not None:
                golden_metrics = tensor_error_metrics(target_attn_v, golden_attn_v)

            head_reports.append({
                "head_idx": head_idx,
                "softmax_scale": float(softmax_scale),
                "value_scale": float(value_scale),
                "attn_v_scale": float(attn_v_scale),
                "golden_attn_v_metrics": golden_metrics,
                "variants": variant_reports,
            })

        block_variants = replay_block_downstream_variants(
            all_head_variant_outputs,
            fp32_traces[f"{b}_concat"],
            fp32_traces[f"{b}_out_proj"],
            out_proj_weight,
            out_proj_bias,
            out_proj_scale,
        )

        baseline_outputs = [
            variant_tensors["qdq_softmax_value"]["attn_v_qdq"]
            for variant_tensors in head_variant_tensors
        ]
        for head_idx, head_report in enumerate(head_reports):
            isolated = {}
            for variant_name in all_head_variant_outputs:
                candidate_outputs = list(baseline_outputs)
                candidate_outputs[head_idx] = head_variant_tensors[head_idx][variant_name]["attn_v_qdq"]
                isolated[variant_name] = replay_block_downstream_variants(
                    {"candidate": candidate_outputs},
                    fp32_traces[f"{b}_concat"],
                    fp32_traces[f"{b}_out_proj"],
                    out_proj_weight,
                    out_proj_bias,
                    out_proj_scale,
                )["candidate"]
            head_report["isolated_block_variants"] = isolated

        golden_block_metrics = {}
        golden_concat = golden_tensors.get(f"{b}_concat")
        if golden_concat is not None:
            golden_block_metrics["concat_metrics"] = tensor_error_metrics(
                fp32_traces[f"{b}_concat"],
                golden_concat,
            )
        golden_out_proj = golden_tensors.get(f"{b}_out_proj")
        if golden_out_proj is not None:
            golden_block_metrics["out_proj_metrics"] = tensor_error_metrics(
                fp32_traces[f"{b}_out_proj"],
                golden_out_proj,
            )

        blocks.append({
            "block_idx": block_idx,
            "golden_block_metrics": golden_block_metrics,
            "block_variants": block_variants,
            "heads": head_reports,
        })

    return {"blocks": blocks}


def summarize_late_attention_replay(replay_reports: list) -> list:
    """Aggregate late-attention replay metrics across traced images."""
    aggregate = {}
    for report in replay_reports:
        for block in report.get("blocks", []):
            bucket = aggregate.setdefault(block["block_idx"], {
                "golden_out_proj_cosines": [],
                "golden_concat_cosines": [],
                "variant_block_out_proj_qdq": {},
                "heads": {},
            })
            golden = block.get("golden_block_metrics", {})
            if golden.get("concat_metrics"):
                bucket["golden_concat_cosines"].append(golden["concat_metrics"]["cosine_sim"])
            if golden.get("out_proj_metrics"):
                bucket["golden_out_proj_cosines"].append(golden["out_proj_metrics"]["cosine_sim"])

            for variant_name, metrics in block["block_variants"].items():
                bucket["variant_block_out_proj_qdq"].setdefault(variant_name, []).append(
                    metrics["out_proj_qdq_metrics"]["cosine_sim"]
                )

            for head in block["heads"]:
                head_bucket = bucket["heads"].setdefault(head["head_idx"], {
                    "golden_attn_v": [],
                    "variant_attn_v_qdq": {},
                    "variant_isolated_out_proj_qdq": {},
                })
                if head.get("golden_attn_v_metrics"):
                    head_bucket["golden_attn_v"].append(head["golden_attn_v_metrics"]["cosine_sim"])
                for variant_name, metrics in head["variants"].items():
                    head_bucket["variant_attn_v_qdq"].setdefault(variant_name, []).append(
                        metrics["attn_v_qdq_metrics"]["cosine_sim"]
                    )
                for variant_name, metrics in head["isolated_block_variants"].items():
                    head_bucket["variant_isolated_out_proj_qdq"].setdefault(variant_name, []).append(
                        metrics["out_proj_qdq_metrics"]["cosine_sim"]
                    )

    summaries = []
    for block_idx, bucket in sorted(aggregate.items()):
        summaries.append({
            "block_idx": block_idx,
            "mean_golden_concat_cosine": float(np.mean(bucket["golden_concat_cosines"]))
            if bucket["golden_concat_cosines"] else 0.0,
            "mean_golden_out_proj_cosine": float(np.mean(bucket["golden_out_proj_cosines"]))
            if bucket["golden_out_proj_cosines"] else 0.0,
            "variant_mean_block_out_proj_qdq_cosine": {
                name: float(np.mean(vals))
                for name, vals in sorted(bucket["variant_block_out_proj_qdq"].items())
            },
            "per_head": [
                {
                    "head_idx": head_idx,
                    "mean_golden_attn_v_cosine": float(np.mean(head_bucket["golden_attn_v"]))
                    if head_bucket["golden_attn_v"] else 0.0,
                    "variant_mean_attn_v_qdq_cosine": {
                        name: float(np.mean(vals))
                        for name, vals in sorted(head_bucket["variant_attn_v_qdq"].items())
                    },
                    "variant_mean_isolated_out_proj_qdq_cosine": {
                        name: float(np.mean(vals))
                        for name, vals in sorted(head_bucket["variant_isolated_out_proj_qdq"].items())
                    },
                }
                for head_idx, head_bucket in sorted(bucket["heads"].items())
            ],
        })
    return summaries


def replay_late_mlp(
    model,
    fp32_traces: dict,
    golden_trace: dict,
    cal_scales: dict,
    block_indices,
):
    """Replay late MLP blocks through GELU, FC2, and residual2."""
    golden_tensors = (golden_trace or {}).get("tensors", {})
    blocks = []

    for block_idx in block_indices:
        b = f"block{block_idx}"
        layer = model.vit.encoder.layer[block_idx].output.dense
        fc2_weight = layer.weight.detach().cpu().numpy().astype(np.float32)
        fc2_bias = layer.bias.detach().cpu().numpy().astype(np.float32)
        variants = replay_mlp_block_variants(
            fp32_traces[f"{b}_fc1"],
            fp32_traces[f"{b}_gelu"],
            fp32_traces[f"{b}_fc2"],
            fp32_traces[f"{b}_residual1"],
            fp32_traces[f"{b}_residual2"],
            cal_scales.get(f"{b}_fc1", 0.0),
            cal_scales.get(f"{b}_gelu", 0.0),
            cal_scales.get(f"{b}_fc2", 0.0),
            fc2_weight,
            fc2_bias,
        )
        golden_metrics = {}
        for node_name in ("gelu", "fc2", "residual2"):
            golden_tensor = golden_tensors.get(f"{b}_{node_name}")
            if golden_tensor is not None:
                golden_metrics[f"{node_name}_metrics"] = tensor_error_metrics(
                    fp32_traces[f"{b}_{node_name}"],
                    golden_tensor,
                )

        blocks.append({
            "block_idx": block_idx,
            "golden_metrics": golden_metrics,
            "variants": variants,
        })

    return {"blocks": blocks}


def summarize_late_mlp_replay(replay_reports: list) -> list:
    """Aggregate late-MLP replay metrics across traced images."""
    aggregate = {}
    for report in replay_reports:
        for block in report.get("blocks", []):
            bucket = aggregate.setdefault(block["block_idx"], {
                "golden_gelu": [],
                "golden_fc2": [],
                "golden_residual2": [],
                "variant_gelu": {},
                "variant_fc2": {},
                "variant_residual2": {},
            })
            golden = block.get("golden_metrics", {})
            if golden.get("gelu_metrics"):
                bucket["golden_gelu"].append(golden["gelu_metrics"]["cosine_sim"])
            if golden.get("fc2_metrics"):
                bucket["golden_fc2"].append(golden["fc2_metrics"]["cosine_sim"])
            if golden.get("residual2_metrics"):
                bucket["golden_residual2"].append(golden["residual2_metrics"]["cosine_sim"])

            for variant_name, metrics in block["variants"].items():
                bucket["variant_gelu"].setdefault(variant_name, []).append(
                    metrics["gelu_metrics"]["cosine_sim"]
                )
                bucket["variant_fc2"].setdefault(variant_name, []).append(
                    metrics["fc2_qdq_metrics"]["cosine_sim"]
                )
                bucket["variant_residual2"].setdefault(variant_name, []).append(
                    metrics["residual2_metrics"]["cosine_sim"]
                )

    summaries = []
    for block_idx, bucket in sorted(aggregate.items()):
        summaries.append({
            "block_idx": block_idx,
            "mean_golden_gelu_cosine": float(np.mean(bucket["golden_gelu"])) if bucket["golden_gelu"] else 0.0,
            "mean_golden_fc2_cosine": float(np.mean(bucket["golden_fc2"])) if bucket["golden_fc2"] else 0.0,
            "mean_golden_residual2_cosine": float(np.mean(bucket["golden_residual2"]))
            if bucket["golden_residual2"] else 0.0,
            "variant_mean_gelu_cosine": {
                name: float(np.mean(vals))
                for name, vals in sorted(bucket["variant_gelu"].items())
            },
            "variant_mean_fc2_cosine": {
                name: float(np.mean(vals))
                for name, vals in sorted(bucket["variant_fc2"].items())
            },
            "variant_mean_residual2_cosine": {
                name: float(np.mean(vals))
                for name, vals in sorted(bucket["variant_residual2"].items())
            },
        })
    return summaries


def compare_trace_tensors(fp32_traces: dict, golden_trace: dict, node_order: list):
    """Compare FP32 and dequantized golden tensors node by node."""
    golden_tensors = golden_trace.get("tensors", {})
    trace_stats = golden_trace.get("stats", {})
    trace_meta = golden_trace.get("meta", {})
    node_metrics = []
    prev_cos = None

    for node_name in node_order:
        fp32_tensor = fp32_traces.get(node_name)
        golden_tensor = golden_tensors.get(node_name)
        if fp32_tensor is None or golden_tensor is None:
            continue

        rows = min(fp32_tensor.shape[0], golden_tensor.shape[0])
        cols = min(fp32_tensor.shape[1], golden_tensor.shape[1])
        fp32_crop = fp32_tensor[:rows, :cols]
        golden_crop = golden_tensor[:rows, :cols]
        cos = cosine_similarity(fp32_crop, golden_crop)
        max_abs_err = float(np.max(np.abs(fp32_crop - golden_crop)))
        delta = None if prev_cos is None else float(cos - prev_cos)
        meta = trace_meta.get(node_name, {})
        stats = trace_stats.get(node_name, {})
        qdq = {}
        if meta.get("dtype") == "int8":
            qdq = quantization_diagnostics(fp32_crop, float(meta.get("scale", 0.0)))
        node_metrics.append({
            "node": node_name,
            "cosine_sim": cos,
            "max_abs_error": max_abs_err,
            "saturation_rate": float(stats.get("saturation_rate", 0.0)),
            "zero_fraction": float(stats.get("zero_fraction", 0.0)),
            "trace_scale": float(meta.get("scale", 0.0)),
            "trace_dtype": meta.get("dtype", ""),
            "quant_step": float(meta.get("scale", 0.0)),
            "qdq_cosine_sim": float(qdq.get("qdq_cosine_sim", 0.0)),
            "qdq_max_abs_error": float(qdq.get("qdq_max_abs_error", 0.0)),
            "qdq_zero_fraction": float(qdq.get("qdq_zero_fraction", 0.0)),
            "qdq_saturation_rate": float(qdq.get("qdq_saturation_rate", 0.0)),
            "delta_from_prev": delta,
        })
        prev_cos = cos

    return node_metrics


def first_major_trace_drop(node_metrics, drop_threshold=0.02, cosine_floor=0.97):
    """Return the first node where cosine takes a meaningful step down."""
    best = None
    best_drop = -1.0
    for metric in node_metrics:
        delta = metric["delta_from_prev"]
        if delta is None:
            continue
        drop = -delta
        if metric["cosine_sim"] <= cosine_floor and drop >= drop_threshold:
            return metric
        if drop > best_drop:
            best = metric
            best_drop = drop
    return best


def _reshape_heads(tensor, module):
    """Reshape [B, S, D] -> [B, H, S, Dh] in a HF-version-agnostic way."""
    bsz, seq_len, _ = tensor.shape
    nh = module.num_attention_heads
    hd = module.attention_head_size
    return tensor.view(bsz, seq_len, nh, hd).permute(0, 2, 1, 3)


def _reduce_observations(values, mode: str, percentile: float) -> float:
    arr = np.array(values, dtype=np.float32)
    if mode == "percentile":
        return float(np.percentile(arr, percentile))
    return float(np.max(arr))


def _uses_downstream_softmax_objective(objective: str) -> bool:
    return objective in {"downstream_out_proj", "tail_out_proj", "tail_attn_v"}


def _uses_downstream_attn_v_objective(objective: str) -> bool:
    return objective in {"downstream_out_proj", "tail_out_proj", "tail_attn_v"}


def _uses_replay_softmax_objective(objective: str) -> bool:
    return objective in {"hessian_prob"} or _uses_downstream_softmax_objective(objective)


def calibrate_softmax_scales(
    model,
    sample_inputs: list,
    mode: str = "max",
    percentile: float = 99.0,
    min_prob: float = 1e-4,
    max_prob: float = 1.0,
    search_heads=None,
    search_objective: str = "local_prob",
    downstream_contexts: dict = None,
) -> tuple:
    """Capture per-head attention probabilities and QKT max_abs for calibration.

    mode="max": uses the largest observed per-image max for each (layer, head).
    mode="percentile": uses a percentile of per-image maxima to reduce outlier impact.

    Returns (softmax_probs, qkt_max_abs):
      softmax_probs: {(layer_idx, head_idx): calibrated_prob}
      qkt_max_abs:   {(layer_idx, head_idx): max_abs of Q@K^T/sqrt(d) over samples}
    """
    if mode == "search":
        search_heads = set(search_heads) if search_heads is not None else None
        softmax_samples, qkt_samples = collect_softmax_prob_tensors(model, sample_inputs)
        downstream_contexts = downstream_contexts or {}

        softmax_probs = {}
        debug = {}
        for key, tensors in softmax_samples.items():
            max_vals = [float(np.max(tensor)) for tensor in tensors]
            default_prob = float(np.clip(np.max(max_vals), min_prob, max_prob))
            if search_heads is not None and key not in search_heads:
                softmax_probs[key] = default_prob
                debug[key] = {"label": "default_skip", "max_prob": default_prob}
                continue
            if _uses_replay_softmax_objective(search_objective):
                context = downstream_contexts.get(key)
                if context is None:
                    best_prob, best_debug = default_prob, {
                        "label": "default_missing_context",
                        "max_prob": default_prob,
                    }
                else:
                    layer_idx, _ = key
                    default_head_probs = {
                        h: float(np.clip(
                            np.max([
                                float(np.max(tensor))
                                for tensor in softmax_samples[(layer_idx, h)]
                            ]),
                            min_prob,
                            max_prob,
                        ))
                        for h in range(NUM_HEADS)
                    }
                    best_prob, best_debug = select_best_softmax_prob_downstream(
                        context["block_samples"],
                        head_idx=key[1],
                        default_prob=default_prob,
                        default_head_probs=default_head_probs,
                        value_scales=context["value_scales"],
                        attn_v_scale=context["attn_v_scale"],
                        out_proj_scale=context["out_proj_scale"],
                        out_proj_weight=context["out_proj_weight"],
                        out_proj_bias=context["out_proj_bias"],
                        min_prob=min_prob,
                        max_prob=max_prob,
                        search_objective=search_objective,
                        twin_uniform_mode=context.get("twin_uniform_mode", "off"),
                    )
                    best_debug["objective"] = search_objective
            else:
                best_prob, best_debug = select_best_softmax_prob(
                    tensors,
                    default_prob=default_prob,
                    min_prob=min_prob,
                    max_prob=max_prob,
                )
                best_debug["objective"] = search_objective
            softmax_probs[key] = best_prob
            debug[key] = best_debug

        qkt_max_abs = {}
        for key, vals in qkt_samples.items():
            qkt_max_abs[key] = _reduce_observations(vals, "max", percentile)
        return softmax_probs, qkt_max_abs, debug

    import math as _math
    attn_samples = {}  # {(layer_idx, head_idx): [max_prob_per_image]}
    qkt_samples = {}   # {(layer_idx, head_idx): [max_abs_qkt_per_image]}

    # Build a temporary copy with eager attn + output_attentions enabled.
    # output_attentions requires attn_implementation='eager' (SDPA doesn't support it).
    cfg = model.config.__class__.from_dict(model.config.to_dict())
    cfg.output_attentions = True
    attn_model = type(model)(cfg)
    attn_model.load_state_dict(model.state_dict())
    attn_model.eval()

    handles = []
    for layer_idx in range(DEPTH):
        def _make_qkt_hook(l_idx):
            def hook(module, inputs, output):
                hidden = inputs[0]
                with torch.no_grad():
                    q = _reshape_heads(module.query(hidden), module)
                    k = _reshape_heads(module.key(hidden), module)
                    qkt = torch.matmul(q, k.transpose(-1, -2)) / _math.sqrt(
                        module.attention_head_size
                    )
                    for h in range(qkt.shape[1]):
                        key = (l_idx, h)
                        qkt_samples.setdefault(key, []).append(float(qkt[0, h].abs().max().item()))
            return hook

        h = attn_model.vit.encoder.layer[layer_idx].attention.attention.register_forward_hook(
            _make_qkt_hook(layer_idx)
        )
        handles.append(h)

    with torch.no_grad():
        for inp in sample_inputs:
            inp_dict = dict(inp.items()) if hasattr(inp, "items") else {"pixel_values": inp}
            outputs = attn_model(**inp_dict)
            if outputs.attentions:
                for layer_idx, attn_weights in enumerate(outputs.attentions):
                    for h in range(attn_weights.shape[1]):
                        key = (layer_idx, h)
                        attn_samples.setdefault(key, []).append(float(attn_weights[0, h].max().item()))

    for h in handles:
        h.remove()
    del attn_model

    softmax_probs = {}
    debug = {}
    for key, vals in attn_samples.items():
        raw = _reduce_observations(vals, mode, percentile)
        softmax_probs[key] = float(np.clip(raw, min_prob, max_prob))
        debug[key] = {"label": mode, "max_prob": softmax_probs[key]}

    qkt_max_abs = {}
    for key, vals in qkt_samples.items():
        qkt_max_abs[key] = _reduce_observations(vals, mode, percentile)

    return softmax_probs, qkt_max_abs, debug


def collect_softmax_prob_tensors(model, sample_inputs: list) -> tuple:
    """Collect per-head softmax probability tensors and QKT max-abs samples."""
    softmax_samples = {}  # {(layer_idx, head_idx): [tensor_per_image]}
    qkt_samples = {}      # {(layer_idx, head_idx): [max_abs_qkt_per_image]}
    handles = []

    for layer_idx in range(DEPTH):
        def _make_hook(l_idx):
            def hook(module, inputs, output):
                hidden = inputs[0]
                with torch.no_grad():
                    q = _reshape_heads(module.query(hidden), module)
                    k = _reshape_heads(module.key(hidden), module)
                    qkt = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(module.attention_head_size)
                    probs = torch.softmax(qkt, dim=-1)
                    for h in range(probs.shape[1]):
                        key = (l_idx, h)
                        softmax_samples.setdefault(key, []).append(_as_f32(probs[0, h]))
                        qkt_samples.setdefault(key, []).append(float(qkt[0, h].abs().max().item()))
            return hook

        h = model.vit.encoder.layer[layer_idx].attention.attention.register_forward_hook(
            _make_hook(layer_idx)
        )
        handles.append(h)

    model.eval()
    with torch.no_grad():
        for inp in sample_inputs:
            inp_dict = dict(inp.items()) if hasattr(inp, "items") else {"pixel_values": inp}
            model(**inp_dict)

    for h in handles:
        h.remove()

    return softmax_samples, qkt_samples


def calibrate_attn_v_scales(
    model,
    sample_inputs: list,
    mode: str = "max",
    percentile: float = 99.0,
    safety_margin: float = 1.10,
    default_block_scales: dict = None,
    search_blocks=None,
    search_objective: str = "local_attn_v",
    downstream_contexts: dict = None,
) -> tuple:
    """Capture per-block common scales for attn_probs @ V outputs."""
    attn_v_samples = collect_attn_context_tensors(model, sample_inputs)
    default_block_scales = default_block_scales or {}
    search_blocks = set(search_blocks) if search_blocks is not None else None
    downstream_contexts = downstream_contexts or {}

    block_scales = {}
    debug = {}
    for layer_idx in range(DEPTH):
        head_tensors = []
        head_maxes = []
        for h in range(NUM_HEADS):
            tensors = attn_v_samples.get((layer_idx, h), [])
            if not tensors:
                continue
            head_tensors.extend(tensors)
            head_maxes.extend(float(np.max(np.abs(tensor))) for tensor in tensors)
        if not head_tensors:
            continue

        default_scale = default_block_scales.get(layer_idx)
        if mode == "search":
            if search_blocks is not None and layer_idx not in search_blocks:
                block_scales[layer_idx] = default_scale
                debug[layer_idx] = {"label": "default_skip", "scale": default_scale}
                continue
            if _uses_downstream_attn_v_objective(search_objective):
                context = downstream_contexts.get(layer_idx)
                if context is None:
                    best_scale, best_debug = default_scale, {
                        "label": "default_missing_context",
                        "scale": default_scale,
                    }
                else:
                    best_scale, best_debug = select_best_attn_v_scale_downstream(
                        context["block_samples"],
                        default_scale=default_scale,
                        default_head_probs=context["default_head_probs"],
                        value_scales=context["value_scales"],
                        out_proj_scale=context["out_proj_scale"],
                        out_proj_weight=context["out_proj_weight"],
                        out_proj_bias=context["out_proj_bias"],
                        search_objective=search_objective,
                    )
                    best_debug["objective"] = search_objective
            else:
                best_scale, best_debug = select_best_attn_v_scale(
                    head_tensors,
                    default_scale=default_scale,
                )
                best_debug["objective"] = search_objective
            block_scales[layer_idx] = best_scale
            debug[layer_idx] = best_debug
            continue

        raw = _reduce_observations(head_maxes, mode, percentile)
        block_scales[layer_idx] = max(raw * safety_margin, 1e-8) / 127.0
        debug[layer_idx] = {"label": mode, "scale": block_scales[layer_idx]}

    return block_scales, debug


def collect_attn_context_tensors(model, sample_inputs: list) -> dict:
    """Collect per-head attn_probs @ V tensors for calibration."""
    samples = {}  # {(layer_idx, head_idx): [tensor_per_image]}
    handles = []

    for layer_idx in range(DEPTH):
        def _make_hook(l_idx):
            def hook(module, inputs, output):
                hidden = inputs[0]
                with torch.no_grad():
                    q = _reshape_heads(module.query(hidden), module)
                    k = _reshape_heads(module.key(hidden), module)
                    v = _reshape_heads(module.value(hidden), module)
                    qkt = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(
                        module.attention_head_size
                    )
                    attn_probs = torch.softmax(qkt, dim=-1)
                    context = torch.matmul(attn_probs, v)
                    for h in range(context.shape[1]):
                        samples.setdefault((l_idx, h), []).append(_as_f32(context[0, h]))
            return hook

        h = model.vit.encoder.layer[layer_idx].attention.attention.register_forward_hook(
            _make_hook(layer_idx)
        )
        handles.append(h)

    model.eval()
    with torch.no_grad():
        for inp in sample_inputs:
            inp_dict = dict(inp.items()) if hasattr(inp, "items") else {"pixel_values": inp}
            model(**inp_dict)

    for h in handles:
        h.remove()

    return samples


def collect_head_projection_tensors(model, sample_inputs: list, proj_names=("query", "key", "value")) -> dict:
    """Collect per-head projection tensors for calibration."""
    samples = {}  # {(layer_idx, proj_name, head_idx): [tensor_per_image]}
    handles = []

    for layer_idx in range(DEPTH):
        def _make_hook(l_idx):
            def hook(module, inputs, output):
                hidden = inputs[0]
                with torch.no_grad():
                    for proj_name in proj_names:
                        projected = _reshape_heads(getattr(module, proj_name)(hidden), module)
                        for h in range(projected.shape[1]):
                            key = (l_idx, proj_name, h)
                            samples.setdefault(key, []).append(_as_f32(projected[0, h]))
            return hook

        h = model.vit.encoder.layer[layer_idx].attention.attention.register_forward_hook(
            _make_hook(layer_idx)
        )
        handles.append(h)

    model.eval()
    with torch.no_grad():
        for inp in sample_inputs:
            inp_dict = dict(inp.items()) if hasattr(inp, "items") else {"pixel_values": inp}
            model(**inp_dict)

    for h in handles:
        h.remove()

    return samples


def calibrate_qkv_head_scales(
    model,
    sample_inputs: list,
    mode: str = "max",
    percentile: float = 99.0,
) -> dict:
    """Capture per-head Q/K/V activation scales for each encoder block."""
    samples = collect_head_projection_tensors(model, sample_inputs)

    head_scales = {}
    for (layer_idx, proj_name, head_idx), tensors in samples.items():
        vals = [float(np.max(np.abs(tensor))) for tensor in tensors]
        head_scales[(layer_idx, proj_name, head_idx)] = max(
            _reduce_observations(vals, mode, percentile),
            1e-8,
        ) / 127.0

    return head_scales


def select_best_value_scale(value_tensors: list) -> tuple:
    """Choose a value scale by minimizing local INT8 reconstruction error."""
    flat_abs = np.concatenate([np.abs(tensor).reshape(-1) for tensor in value_tensors]).astype(np.float32)
    max_abs = float(np.max(flat_abs)) if flat_abs.size else 1e-8

    candidate_specs = [
        ("max", max_abs),
        ("p99.9", float(np.percentile(flat_abs, 99.9))),
        ("p99.5", float(np.percentile(flat_abs, 99.5))),
        ("p99.0", float(np.percentile(flat_abs, 99.0))),
    ]
    for mult in (1.00, 0.95, 0.90, 0.85, 0.80):
        candidate_specs.append((f"mse_x{mult:.2f}", max_abs * mult))

    best = None
    for label, raw in candidate_specs:
        scale = max(raw, 1e-8) / 127.0
        mse_vals = []
        cos_vals = []
        for tensor in value_tensors:
            dq = quantize_dequant_tensor(tensor, scale)
            mse_vals.append(float(np.mean((tensor - dq) ** 2)))
            cos_vals.append(cosine_similarity(tensor, dq))
        score = (float(np.mean(mse_vals)), -float(np.mean(cos_vals)))
        candidate = {
            "label": label,
            "scale": scale,
            "mean_mse": float(np.mean(mse_vals)),
            "mean_cosine": float(np.mean(cos_vals)),
        }
        if best is None or score < (best["mean_mse"], -best["mean_cosine"]):
            best = candidate
    return best["scale"], best


def select_best_softmax_prob(
    prob_tensors: list,
    default_prob: float = None,
    min_prob: float = 1e-4,
    max_prob: float = 1.0,
) -> tuple:
    """Choose a softmax max-prob target by minimizing local INT8 reconstruction error."""
    flat = np.concatenate([tensor.reshape(-1) for tensor in prob_tensors]).astype(np.float32)
    observed_max = float(np.max(flat)) if flat.size else min_prob

    candidate_specs = []
    if default_prob and default_prob > 0:
        candidate_specs.extend([
            ("default", default_prob),
            ("default_x0.98", default_prob * 0.98),
            ("default_x0.95", default_prob * 0.95),
            ("default_x0.90", default_prob * 0.90),
        ])
    candidate_specs.extend([
        ("p99.99", float(np.percentile(flat, 99.99))),
        ("p99.9", float(np.percentile(flat, 99.9))),
        ("p99.5", float(np.percentile(flat, 99.5))),
        ("max", observed_max),
    ])

    best = None
    for label, raw in candidate_specs:
        target_prob = float(np.clip(raw, min_prob, max_prob))
        scale = target_prob / 127.0
        mse_vals = []
        cos_vals = []
        sat_vals = []
        for tensor in prob_tensors:
            q = np.clip(np.round(tensor / scale), 0, 127).astype(np.int8)
            dq = q.astype(np.float32) * np.float32(scale)
            mse_vals.append(float(np.mean((tensor - dq) ** 2)))
            cos_vals.append(cosine_similarity(tensor, dq))
            sat_vals.append(float(np.mean(q == 127)))
        score = (float(np.mean(mse_vals)), float(np.mean(sat_vals)), -float(np.mean(cos_vals)))
        candidate = {
            "label": label,
            "max_prob": target_prob,
            "mean_mse": float(np.mean(mse_vals)),
            "mean_cosine": float(np.mean(cos_vals)),
            "mean_saturation_rate": float(np.mean(sat_vals)),
        }
        if best is None or score < (
            best["mean_mse"],
            best["mean_saturation_rate"],
            -best["mean_cosine"],
        ):
            best = candidate
    return best["max_prob"], best


def default_value_head_scales(calibration, qkv_head_scales=None, value_head_scales=None) -> dict:
    """Return the default per-head value scales used by the current compiler path."""
    cal = calibration.scales
    qkv_head_scales = qkv_head_scales or {}
    value_head_scales = value_head_scales or {}
    head_scales = {}
    for block_idx in range(DEPTH):
        p = f"vit.encoder.layer.{block_idx}"
        fallback = cal.get(f"{p}.attention.attention.value", 6.0 / 127.0)
        for head_idx in range(NUM_HEADS):
            head_scales[(block_idx, head_idx)] = value_head_scales.get(
                (block_idx, head_idx),
                qkv_head_scales.get((block_idx, "value", head_idx), fallback),
            )
    return head_scales


def default_block_input_scale(calibration) -> float:
    """Return the shared residual/out-proj/fc2 target scale."""
    cal = calibration.scales
    return cal.get("vit.embeddings.dropout", cal.get("vit.embeddings", 6.0 / 127.0))


def select_best_softmax_prob_downstream(
    block_samples: list,
    head_idx: int,
    default_prob: float,
    default_head_probs: dict,
    value_scales: dict,
    attn_v_scale: float,
    out_proj_scale: float,
    out_proj_weight: np.ndarray,
    out_proj_bias: np.ndarray,
    min_prob: float = 1e-4,
    max_prob: float = 1.0,
    search_objective: str = "downstream_out_proj",
    twin_uniform_mode: str = "off",
) -> tuple:
    """Choose a softmax target by maximizing downstream block out_proj cosine."""
    flat = np.concatenate([sample["heads"][head_idx]["softmax"].reshape(-1) for sample in block_samples]).astype(np.float32)
    candidate_specs = [
        ("default", default_prob),
        ("default_x0.98", default_prob * 0.98),
        ("default_x0.95", default_prob * 0.95),
        ("default_x0.90", default_prob * 0.90),
        ("p99.99", float(np.percentile(flat, 99.99))),
        ("p99.9", float(np.percentile(flat, 99.9))),
        ("p99.5", float(np.percentile(flat, 99.5))),
    ]

    best = None
    for label, raw in candidate_specs:
        target_prob = float(np.clip(raw, min_prob, max_prob))
        out_proj_cosines = []
        attn_v_cosines = []
        sat_rates = []
        hessian_scores = []
        for sample in block_samples:
            head_outputs = []
            for h in range(NUM_HEADS):
                value_scale = value_scales[h]
                value_qdq = quantize_dequant_tensor(sample["heads"][h]["value"], value_scale)
                if h == head_idx:
                    softmax_qdq, soft_meta = quantize_dequant_softmax_candidate(
                        sample["heads"][h]["softmax"],
                        target_prob,
                        twin_uniform_mode=twin_uniform_mode,
                    )
                else:
                    softmax_qdq, _ = quantize_dequant_softmax_candidate(
                        sample["heads"][h]["softmax"],
                        float(np.clip(default_head_probs[h], min_prob, max_prob)),
                        twin_uniform_mode="off",
                    )
                    soft_meta = None
                attn_v_raw = (softmax_qdq @ value_qdq).astype(np.float32)
                attn_v_qdq = quantize_dequant_tensor(attn_v_raw, attn_v_scale)
                head_outputs.append(attn_v_qdq)
                if h == head_idx:
                    hessian_scores.append(
                        weighted_quant_error_score(
                            sample["heads"][h]["softmax"],
                            softmax_qdq,
                            softmax_attn_v_hessian_diag(sample["heads"][h]["softmax"], value_qdq),
                        )
                    )
                    attn_v_cosines.append(
                        tensor_error_metrics(
                            sample["heads"][h]["attn_v"],
                            attn_v_qdq,
                        )["cosine_sim"]
                    )
                    sat_rates.append(float(soft_meta.get("saturation_rate", 0.0)) if soft_meta else 0.0)

            out_proj_metrics = replay_block_downstream_variants(
                {"candidate": head_outputs},
                sample["concat"],
                sample["out_proj"],
                out_proj_weight,
                out_proj_bias,
                out_proj_scale,
            )["candidate"]["out_proj_qdq_metrics"]
            out_proj_cosines.append(out_proj_metrics["cosine_sim"])

        candidate = {
            "label": label,
            "max_prob": target_prob,
            "min_out_proj_cosine": float(np.min(out_proj_cosines)),
            "mean_out_proj_cosine": float(np.mean(out_proj_cosines)),
            "min_attn_v_cosine": float(np.min(attn_v_cosines)),
            "mean_saturation_rate": float(np.mean(sat_rates)),
            "mean_attn_v_cosine": float(np.mean(attn_v_cosines)),
            "mean_hessian_score": float(np.mean(hessian_scores)),
        }
        if search_objective == "hessian_prob":
            score = (
                candidate["mean_hessian_score"],
                candidate["mean_saturation_rate"],
                -candidate["mean_attn_v_cosine"],
                -candidate["mean_out_proj_cosine"],
            )
            best_score = None if best is None else (
                best["mean_hessian_score"],
                best["mean_saturation_rate"],
                -best["mean_attn_v_cosine"],
                -best["mean_out_proj_cosine"],
            )
        elif search_objective == "tail_out_proj":
            score = (
                -candidate["min_out_proj_cosine"],
                -candidate["mean_out_proj_cosine"],
                candidate["mean_saturation_rate"],
                -candidate["mean_attn_v_cosine"],
            )
            best_score = None if best is None else (
                -best["min_out_proj_cosine"],
                -best["mean_out_proj_cosine"],
                best["mean_saturation_rate"],
                -best["mean_attn_v_cosine"],
            )
        elif search_objective == "tail_attn_v":
            score = (
                -candidate["min_attn_v_cosine"],
                -candidate["mean_attn_v_cosine"],
                candidate["mean_saturation_rate"],
                -candidate["mean_out_proj_cosine"],
            )
            best_score = None if best is None else (
                -best["min_attn_v_cosine"],
                -best["mean_attn_v_cosine"],
                best["mean_saturation_rate"],
                -best["mean_out_proj_cosine"],
            )
        else:
            score = (
                -candidate["mean_out_proj_cosine"],
                candidate["mean_saturation_rate"],
                -candidate["mean_attn_v_cosine"],
            )
            best_score = None if best is None else (
                -best["mean_out_proj_cosine"],
                best["mean_saturation_rate"],
                -best["mean_attn_v_cosine"],
            )
        if best is None or score < best_score:
            best = candidate

    return best["max_prob"], best


def select_best_attn_v_scale(attn_v_tensors: list, default_scale: float = None) -> tuple:
    """Choose a shared attn@V scale by minimizing local INT8 reconstruction error."""
    flat_abs = np.concatenate([np.abs(tensor).reshape(-1) for tensor in attn_v_tensors]).astype(np.float32)
    max_abs = float(np.max(flat_abs)) if flat_abs.size else 1e-8

    candidate_specs = []
    if default_scale and default_scale > 0:
        default_raw = default_scale * 127.0
        candidate_specs.extend([
            ("default", default_raw),
            ("default_x0.95", default_raw * 0.95),
            ("default_x0.90", default_raw * 0.90),
        ])
    candidate_specs.extend([
        ("max", max_abs),
        ("p99.9", float(np.percentile(flat_abs, 99.9))),
        ("p99.5", float(np.percentile(flat_abs, 99.5))),
        ("p99.0", float(np.percentile(flat_abs, 99.0))),
    ])

    best = None
    for label, raw in candidate_specs:
        scale = max(float(raw), 1e-8) / 127.0
        mse_vals = []
        cos_vals = []
        sat_vals = []
        for tensor in attn_v_tensors:
            q = np.clip(np.round(tensor / scale), -128, 127).astype(np.int8)
            dq = q.astype(np.float32) * np.float32(scale)
            mse_vals.append(float(np.mean((tensor - dq) ** 2)))
            cos_vals.append(cosine_similarity(tensor, dq))
            sat_vals.append(float(np.mean((q == 127) | (q == -128))))
        score = (float(np.mean(mse_vals)), float(np.mean(sat_vals)), -float(np.mean(cos_vals)))
        candidate = {
            "label": label,
            "scale": scale,
            "mean_mse": float(np.mean(mse_vals)),
            "mean_cosine": float(np.mean(cos_vals)),
            "mean_saturation_rate": float(np.mean(sat_vals)),
        }
        if best is None or score < (
            best["mean_mse"],
            best["mean_saturation_rate"],
            -best["mean_cosine"],
        ):
            best = candidate
    return best["scale"], best


def select_best_attn_v_scale_downstream(
    block_samples: list,
    default_scale: float,
    default_head_probs: dict,
    value_scales: dict,
    out_proj_scale: float,
    out_proj_weight: np.ndarray,
    out_proj_bias: np.ndarray,
    search_objective: str = "downstream_out_proj",
) -> tuple:
    """Choose a shared attn@V scale by maximizing downstream block metrics."""
    flat_abs = np.concatenate(
        [
            np.abs(sample["heads"][head_idx]["attn_v"]).reshape(-1)
            for sample in block_samples
            for head_idx in range(NUM_HEADS)
        ]
    ).astype(np.float32)

    candidate_specs = [
        ("default", default_scale * 127.0),
        ("default_x0.95", default_scale * 127.0 * 0.95),
        ("default_x0.90", default_scale * 127.0 * 0.90),
        ("p99.9", float(np.percentile(flat_abs, 99.9))),
        ("p99.5", float(np.percentile(flat_abs, 99.5))),
        ("p99.0", float(np.percentile(flat_abs, 99.0))),
        ("max", float(np.max(flat_abs)) if flat_abs.size else 1e-8),
    ]

    best = None
    for label, raw in candidate_specs:
        scale = max(float(raw), 1e-8) / 127.0
        out_proj_cosines = []
        attn_v_cosines = []
        sat_rates = []
        for sample in block_samples:
            head_outputs = []
            for head_idx in range(NUM_HEADS):
                softmax_scale = max(float(default_head_probs[head_idx]), 1e-8) / 127.0
                value_scale = value_scales[head_idx]
                _, variant_tensors = replay_attention_head_variants(
                    sample["heads"][head_idx]["softmax"],
                    sample["heads"][head_idx]["value"],
                    sample["heads"][head_idx]["attn_v"],
                    softmax_scale,
                    value_scale,
                    scale,
                )
                attn_v_qdq = variant_tensors["qdq_softmax_value"]["attn_v_qdq"]
                head_outputs.append(attn_v_qdq)
                attn_v_cosines.append(
                    tensor_error_metrics(sample["heads"][head_idx]["attn_v"], attn_v_qdq)["cosine_sim"]
                )
                sat_rates.append(
                    quantization_diagnostics(sample["heads"][head_idx]["attn_v"], scale)["qdq_saturation_rate"]
                )

            out_proj_metrics = replay_block_downstream_variants(
                {"candidate": head_outputs},
                sample["concat"],
                sample["out_proj"],
                out_proj_weight,
                out_proj_bias,
                out_proj_scale,
            )["candidate"]["out_proj_qdq_metrics"]
            out_proj_cosines.append(out_proj_metrics["cosine_sim"])

        candidate = {
            "label": label,
            "scale": scale,
            "min_out_proj_cosine": float(np.min(out_proj_cosines)),
            "mean_out_proj_cosine": float(np.mean(out_proj_cosines)),
            "min_attn_v_cosine": float(np.min(attn_v_cosines)),
            "mean_attn_v_cosine": float(np.mean(attn_v_cosines)),
            "mean_saturation_rate": float(np.mean(sat_rates)),
        }
        if search_objective == "tail_out_proj":
            score = (
                -candidate["min_out_proj_cosine"],
                -candidate["mean_out_proj_cosine"],
                candidate["mean_saturation_rate"],
                -candidate["mean_attn_v_cosine"],
            )
            best_score = None if best is None else (
                -best["min_out_proj_cosine"],
                -best["mean_out_proj_cosine"],
                best["mean_saturation_rate"],
                -best["mean_attn_v_cosine"],
            )
        elif search_objective == "tail_attn_v":
            score = (
                -candidate["min_attn_v_cosine"],
                -candidate["mean_attn_v_cosine"],
                candidate["mean_saturation_rate"],
                -candidate["mean_out_proj_cosine"],
            )
            best_score = None if best is None else (
                -best["min_attn_v_cosine"],
                -best["mean_attn_v_cosine"],
                best["mean_saturation_rate"],
                -best["mean_out_proj_cosine"],
            )
        else:
            score = (
                -candidate["mean_out_proj_cosine"],
                candidate["mean_saturation_rate"],
                -candidate["mean_attn_v_cosine"],
            )
            best_score = None if best is None else (
                -best["mean_out_proj_cosine"],
                best["mean_saturation_rate"],
                -best["mean_attn_v_cosine"],
            )
        if best is None or score < best_score:
            best = candidate

    return best["scale"], best


def select_best_attn_v_head_scale_downstream(
    block_samples: list,
    head_idx: int,
    default_head_scales: dict,
    default_head_probs: dict,
    value_scales: dict,
    out_proj_scale: float,
    out_proj_weight: np.ndarray,
    out_proj_bias: np.ndarray,
    search_objective: str = "downstream_out_proj",
) -> tuple:
    """Choose a per-head attn@V output scale by maximizing downstream out_proj quality."""
    flat_abs = np.concatenate(
        [np.abs(sample["heads"][head_idx]["attn_v"]).reshape(-1) for sample in block_samples]
    ).astype(np.float32)
    default_scale = float(default_head_scales[head_idx])
    candidate_specs = [
        ("default", default_scale * 127.0),
        ("default_x0.95", default_scale * 127.0 * 0.95),
        ("default_x0.90", default_scale * 127.0 * 0.90),
        ("p99.9", float(np.percentile(flat_abs, 99.9))),
        ("p99.5", float(np.percentile(flat_abs, 99.5))),
        ("p99.0", float(np.percentile(flat_abs, 99.0))),
        ("max", float(np.max(flat_abs)) if flat_abs.size else 1e-8),
    ]

    best = None
    for label, raw in candidate_specs:
        scale = max(float(raw), 1e-8) / 127.0
        out_proj_cosines = []
        attn_v_cosines = []
        sat_rates = []
        for sample in block_samples:
            head_outputs = []
            for h in range(NUM_HEADS):
                softmax_scale = max(float(default_head_probs[h]), 1e-8) / 127.0
                value_scale = value_scales[h]
                attn_scale = scale if h == head_idx else float(default_head_scales[h])
                _, variant_tensors = replay_attention_head_variants(
                    sample["heads"][h]["softmax"],
                    sample["heads"][h]["value"],
                    sample["heads"][h]["attn_v"],
                    softmax_scale,
                    value_scale,
                    attn_scale,
                )
                attn_v_qdq = variant_tensors["qdq_softmax_value"]["attn_v_qdq"]
                head_outputs.append(attn_v_qdq)
                if h == head_idx:
                    attn_v_cosines.append(
                        tensor_error_metrics(sample["heads"][h]["attn_v"], attn_v_qdq)["cosine_sim"]
                    )
                    sat_rates.append(
                        quantization_diagnostics(sample["heads"][h]["attn_v"], attn_scale)["qdq_saturation_rate"]
                    )

            out_proj_metrics = replay_block_downstream_variants(
                {"candidate": head_outputs},
                sample["concat"],
                sample["out_proj"],
                out_proj_weight,
                out_proj_bias,
                out_proj_scale,
            )["candidate"]["out_proj_qdq_metrics"]
            out_proj_cosines.append(out_proj_metrics["cosine_sim"])

        candidate = {
            "label": label,
            "scale": scale,
            "min_out_proj_cosine": float(np.min(out_proj_cosines)),
            "mean_out_proj_cosine": float(np.mean(out_proj_cosines)),
            "min_attn_v_cosine": float(np.min(attn_v_cosines)),
            "mean_attn_v_cosine": float(np.mean(attn_v_cosines)),
            "mean_saturation_rate": float(np.mean(sat_rates)),
        }
        if search_objective == "tail_out_proj":
            score = (
                -candidate["min_out_proj_cosine"],
                -candidate["mean_out_proj_cosine"],
                candidate["mean_saturation_rate"],
                -candidate["mean_attn_v_cosine"],
            )
            best_score = None if best is None else (
                -best["min_out_proj_cosine"],
                -best["mean_out_proj_cosine"],
                best["mean_saturation_rate"],
                -best["mean_attn_v_cosine"],
            )
        elif search_objective == "tail_attn_v":
            score = (
                -candidate["min_attn_v_cosine"],
                -candidate["mean_attn_v_cosine"],
                candidate["mean_saturation_rate"],
                -candidate["mean_out_proj_cosine"],
            )
            best_score = None if best is None else (
                -best["min_attn_v_cosine"],
                -best["mean_attn_v_cosine"],
                best["mean_saturation_rate"],
                -best["mean_out_proj_cosine"],
            )
        else:
            score = (
                -candidate["mean_out_proj_cosine"],
                candidate["mean_saturation_rate"],
                -candidate["mean_attn_v_cosine"],
            )
            best_score = None if best is None else (
                -best["mean_out_proj_cosine"],
                best["mean_saturation_rate"],
                -best["mean_attn_v_cosine"],
            )
        if best is None or score < best_score:
            best = candidate

    return best["scale"], best


def calibrate_fused_attn_v_head_scales(
    block_replay_samples: dict,
    block_contexts: dict,
    *,
    search_objective: str = "downstream_out_proj",
) -> tuple:
    """Choose per-head attn@V scales for fused attention blocks."""
    head_scales = {}
    debug = {}
    for block_idx, context in sorted(block_contexts.items()):
        default_head_scales = {
            head_idx: float(context["default_attn_v_scale"])
            for head_idx in range(NUM_HEADS)
        }
        for head_idx in range(NUM_HEADS):
            best_scale, best_debug = select_best_attn_v_head_scale_downstream(
                block_samples=block_replay_samples[block_idx],
                head_idx=head_idx,
                default_head_scales=default_head_scales,
                default_head_probs=context["default_head_probs"],
                value_scales=context["value_scales"],
                out_proj_scale=context["out_proj_scale"],
                out_proj_weight=context["out_proj_weight"],
                out_proj_bias=context["out_proj_bias"],
                search_objective=search_objective,
            )
            default_head_scales[head_idx] = best_scale
            head_scales[(block_idx, head_idx)] = best_scale
            debug[(block_idx, head_idx)] = best_debug
    return head_scales, debug


def default_attn_v_block_scales(calibration, qkv_head_scales=None, value_head_scales=None) -> dict:
    """Return the current per-block attn@V defaults from value projection scales."""
    cal = calibration.scales
    qkv_head_scales = qkv_head_scales or {}
    value_head_scales = value_head_scales or {}
    block_scales = {}
    for i in range(DEPTH):
        p = f"vit.encoder.layer.{i}"
        v_fallback = cal.get(f"{p}.attention.attention.value", 6.0 / 127.0)
        block_scales[i] = max(
            value_head_scales.get(
                (i, h),
                qkv_head_scales.get((i, "value", h), v_fallback),
            )
            for h in range(NUM_HEADS)
        )
    return block_scales


def select_best_gelu_output_scale(
    block_samples: list,
    default_scale: float,
    fc2_scale: float,
    fc2_weight: np.ndarray,
    fc2_bias: np.ndarray,
    *,
    search_objective: str = "downstream_residual2",
    twin_uniform_mode: str = "off",
) -> tuple:
    """Choose a GELU output scale by maximizing downstream residual2 cosine."""
    flat = np.concatenate([np.abs(sample["gelu"]).reshape(-1) for sample in block_samples]).astype(np.float32)
    negative_extent = max(
        float(max(-np.min(sample["gelu"]), 0.0))
        for sample in block_samples
    ) if block_samples else 1e-8
    candidate_specs = [
        ("default", default_scale * 127.0),
        ("default_x0.95", default_scale * 127.0 * 0.95),
        ("default_x0.90", default_scale * 127.0 * 0.90),
        ("p99.9", float(np.percentile(flat, 99.9))),
        ("p99.5", float(np.percentile(flat, 99.5))),
        ("p99.0", float(np.percentile(flat, 99.0))),
    ]

    best = None
    for label, raw in candidate_specs:
        scale = max(float(raw), 1e-8) / 127.0
        residual2_cosines = []
        fc2_cosines = []
        sat_rates = []
        hessian_scores = []
        for sample in block_samples:
            gelu_qdq, gelu_meta = quantize_dequant_gelu_candidate(
                sample["gelu"],
                scale * 127.0,
                twin_uniform_mode=twin_uniform_mode,
                negative_extent=negative_extent,
            )
            fc2_raw = (gelu_qdq @ fc2_weight.T + fc2_bias).astype(np.float32)
            fc2_qdq = quantize_dequant_tensor(fc2_raw, fc2_scale)
            residual2 = sample["residual1"].astype(np.float32) + fc2_qdq
            residual2_cosines.append(tensor_error_metrics(sample["residual2"], residual2)["cosine_sim"])
            fc2_cosines.append(tensor_error_metrics(sample["fc2"], fc2_qdq)["cosine_sim"])
            sat_rates.append(float(gelu_meta.get("saturation_rate", 0.0)))
            hessian_scores.append(
                weighted_quant_error_score(
                    sample["gelu"],
                    gelu_qdq,
                    gelu_fc2_hessian_diag(sample["gelu"], fc2_weight),
                )
            )

        candidate = {
            "label": label,
            "scale": scale,
            "mean_residual2_cosine": float(np.mean(residual2_cosines)),
            "mean_fc2_cosine": float(np.mean(fc2_cosines)),
            "mean_saturation_rate": float(np.mean(sat_rates)),
            "mean_hessian_score": float(np.mean(hessian_scores)),
        }
        if search_objective == "hessian_output":
            score = (
                candidate["mean_hessian_score"],
                candidate["mean_saturation_rate"],
                -candidate["mean_fc2_cosine"],
                -candidate["mean_residual2_cosine"],
            )
            best_score = None if best is None else (
                best["mean_hessian_score"],
                best["mean_saturation_rate"],
                -best["mean_fc2_cosine"],
                -best["mean_residual2_cosine"],
            )
        else:
            score = (
                -candidate["mean_residual2_cosine"],
                candidate["mean_saturation_rate"],
                -candidate["mean_fc2_cosine"],
            )
            best_score = None if best is None else (
                -best["mean_residual2_cosine"],
                best["mean_saturation_rate"],
                -best["mean_fc2_cosine"],
            )
        if best is None or score < best_score:
            best = candidate
    return best["scale"], best


def collect_final_logit_tensors(model, processor, sample_images: list) -> list:
    """Collect FP32 CLS vectors and classifier logits for late-logit calibration."""
    samples = []
    hook_state = {}

    def _capture_final_ln(_module, _inputs, output):
        hook_state["final_ln"] = output.detach().cpu().numpy().astype(np.float32)

    handle = model.vit.layernorm.register_forward_hook(_capture_final_ln)
    try:
        with torch.no_grad():
            for image in sample_images:
                hook_state.clear()
                inputs = processor(images=image, return_tensors="pt")
                logits = model(**inputs).logits.detach().cpu().numpy().astype(np.float32)
                final_ln = hook_state.get("final_ln")
                if final_ln is None:
                    raise RuntimeError("Failed to capture final_ln activations for late-logit calibration")
                samples.append({
                    "cls_extract": final_ln[0, 0].astype(np.float32),
                    "classifier": logits[0].astype(np.float32),
                })
    finally:
        handle.remove()
    return samples


def select_best_final_logit_scale(
    final_logit_samples: list,
    default_scale: float,
    classifier_weight: np.ndarray,
    classifier_bias: np.ndarray,
) -> tuple:
    """Choose a CLS/final_ln activation scale by maximizing final logit cosine."""
    if not final_logit_samples:
        fallback = max(float(default_scale), 1e-8)
        return fallback, {
            "label": "default",
            "scale": fallback,
            "mean_classifier_cosine": 0.0,
            "p10_classifier_cosine": 0.0,
            "min_classifier_cosine": 0.0,
            "top1_agreement": 0.0,
            "mean_cls_qdq_cosine": 0.0,
            "mean_saturation_rate": 0.0,
        }

    cls_abs = np.concatenate(
        [np.abs(sample["cls_extract"]).reshape(-1) for sample in final_logit_samples]
    ).astype(np.float32)
    candidate_specs = [
        ("default", default_scale * 127.0),
        ("default_x1.05", default_scale * 127.0 * 1.05),
        ("default_x1.02", default_scale * 127.0 * 1.02),
        ("default_x0.98", default_scale * 127.0 * 0.98),
        ("default_x0.95", default_scale * 127.0 * 0.95),
        ("default_x0.90", default_scale * 127.0 * 0.90),
        ("p100", float(np.max(cls_abs))),
        ("p99.9", float(np.percentile(cls_abs, 99.9))),
        ("p99.5", float(np.percentile(cls_abs, 99.5))),
        ("p99.0", float(np.percentile(cls_abs, 99.0))),
        ("p98.0", float(np.percentile(cls_abs, 98.0))),
    ]

    weight_q, weight_scales = quantize_tensor(classifier_weight, per_channel=False)
    weight_i32 = weight_q.astype(np.int32)
    weight_scale = float(weight_scales.astype(np.float32)[0])
    bias_fp32 = classifier_bias.astype(np.float32)
    seen_scales = set()
    best = None

    for label, raw in candidate_specs:
        scale = max(float(raw), 1e-8) / 127.0
        scale_key = round(scale, 12)
        if scale_key in seen_scales:
            continue
        seen_scales.add(scale_key)

        denom = max(abs(scale * weight_scale), 1e-10)
        bias_i32 = np.round(bias_fp32 / denom).astype(np.int32)
        classifier_cosines = []
        cls_qdq_cosines = []
        sat_rates = []
        top1_matches = []

        for sample in final_logit_samples:
            cls_fp32 = sample["cls_extract"].astype(np.float32)
            logits_fp32 = sample["classifier"].astype(np.float32)
            cls_q = np.clip(np.round(cls_fp32 / scale), -128, 127).astype(np.int32)
            logits_i32 = (cls_q.reshape(1, -1) @ weight_i32.T).reshape(-1) + bias_i32
            cls_qdq = cls_q.astype(np.float32) * np.float32(scale)
            classifier_cosines.append(cosine_similarity(logits_fp32, logits_i32.astype(np.float32)))
            cls_qdq_cosines.append(cosine_similarity(cls_fp32, cls_qdq))
            sat_rates.append(float(np.mean((cls_q == 127) | (cls_q == -128))))
            top1_matches.append(int(np.argmax(logits_fp32) == int(np.argmax(logits_i32))))

        classifier_cosines = np.array(classifier_cosines, dtype=np.float32)
        candidate = {
            "label": label,
            "scale": scale,
            "mean_classifier_cosine": float(np.mean(classifier_cosines)),
            "p10_classifier_cosine": float(np.percentile(classifier_cosines, 10)),
            "min_classifier_cosine": float(np.min(classifier_cosines)),
            "top1_agreement": float(np.mean(top1_matches)),
            "mean_cls_qdq_cosine": float(np.mean(cls_qdq_cosines)),
            "mean_saturation_rate": float(np.mean(sat_rates)),
        }
        score = (
            -candidate["mean_classifier_cosine"],
            -candidate["p10_classifier_cosine"],
            -candidate["min_classifier_cosine"],
            -candidate["top1_agreement"],
            candidate["mean_saturation_rate"],
        )
        best_score = None if best is None else (
            -best["mean_classifier_cosine"],
            -best["p10_classifier_cosine"],
            -best["min_classifier_cosine"],
            -best["top1_agreement"],
            best["mean_saturation_rate"],
        )
        if best is None or score < best_score:
            best = candidate

    return best["scale"], best


def calibrate_value_head_scales(model, sample_inputs: list, mode: str = "search") -> tuple:
    """Calibrate only the per-head value projection scales."""
    samples = collect_head_projection_tensors(model, sample_inputs, proj_names=("value",))
    scales = {}
    debug = {}

    for (layer_idx, proj_name, head_idx), tensors in samples.items():
        if mode == "max":
            best_scale = max(float(np.max(np.abs(tensor))) for tensor in tensors) / 127.0
            best_debug = {"label": "max", "scale": best_scale}
        else:
            best_scale, best_debug = select_best_value_scale(tensors)
        scales[(layer_idx, head_idx)] = best_scale
        debug[(layer_idx, head_idx)] = best_debug

    return scales, debug


def build_calibration_scales(calibration, softmax_max_probs: dict = None,
                             qkt_max_abs: dict = None,
                             attn_v_block_scales: dict = None,
                             fused_attn_v_head_scales: dict = None,
                             qkv_head_scales: dict = None,
                             value_head_scales: dict = None,
                             gelu_block_scales: dict = None,
                             residual1_block_scales: dict = None,
                             final_ln_scale_override: float = None,
                             activation_scale_overrides: dict = None) -> dict:
    """Map calibration module output scales to IR node names used by the codegen."""
    cal = calibration.scales  # module_name → float scale

    def get(module_name, default=6.0 / 127.0):
        return cal.get(module_name, default)

    scales = {}
    attn_v_block_scales = attn_v_block_scales or {}
    fused_attn_v_head_scales = fused_attn_v_head_scales or {}
    qkv_head_scales = qkv_head_scales or {}
    value_head_scales = value_head_scales or {}
    gelu_block_scales = gelu_block_scales or {}
    residual1_block_scales = residual1_block_scales or {}
    activation_scale_overrides = activation_scale_overrides or {}
    # Positional embedding add output: similar to the embedding dropout output
    emb_scale = get("vit.embeddings.dropout", get("vit.embeddings", 6.0 / 127.0))
    scales["pos_embed_add"] = emb_scale

    # Propagate the block-input scale through all 12 encoder blocks.
    # The INT8 residual VADD (x + branch_output) requires BOTH operands to be at
    # the SAME scale, otherwise each INT8 unit has different weight and the sum is
    # wrong.  We enforce this by forcing out_proj and fc2 to REQUANT their output
    # to the current block_input_scale, so the VADD operands are always compatible.
    # block_input_scale starts as the pos_embed_add output scale and stays constant
    # because every residual output is forced back to the same scale.
    block_input_scale = emb_scale

    for i in range(DEPTH):
        b = f"block{i}"
        p = f"vit.encoder.layer.{i}"

        ln1_scale = get(f"{p}.layernorm_before")
        scales[f"{b}_ln1"] = ln1_scale

        for h in range(NUM_HEADS):
            q_scale = qkv_head_scales.get((i, "query", h), get(f"{p}.attention.attention.query"))
            k_scale = qkv_head_scales.get((i, "key", h), get(f"{p}.attention.attention.key"))
            v_scale = value_head_scales.get(
                (i, h),
                qkv_head_scales.get((i, "value", h), get(f"{p}.attention.attention.value")),
            )
            scales[f"{b}_head{h}_query"] = q_scale
            scales[f"{b}_head{h}_key"] = k_scale
            scales[f"{b}_head{h}_value"] = v_scale
            # C1: QKT scale is the raw INT32 accumulator dequant scale used by
            # strip-mined SOFTMAX from ACCUM: q_scale * k_scale * (1/sqrt(d_head)).
            qkt_scale = q_scale * k_scale * 0.125
            scales[f"{b}_head{h}_qkt"] = qkt_scale
            # scale_mul is metadata-only in C1, so it carries the same scale.
            scales[f"{b}_head{h}_scale"] = qkt_scale
            # Per-head softmax scale: calibrated to each head's max attention probability.
            # Heads with sharp CLS self-attention (≈99%) get a coarse scale;
            # heads with diffuse attention (≈10%) get a fine scale preserving variation.
            if softmax_max_probs and (i, h) in softmax_max_probs:
                max_prob = softmax_max_probs[(i, h)]
            else:
                max_prob = 0.20
            scales[f"{b}_head{h}_softmax"] = max(max_prob, 1e-4) / 127.0

        default_attn_v_scale = max(
            [scales[f"{b}_head{h}_value"] for h in range(NUM_HEADS)] or [get(f"{p}.attention.attention.value")]
        )
        attn_v_scale = attn_v_block_scales.get(i, default_attn_v_scale)
        per_head_attn_v_scales = []
        for h in range(NUM_HEADS):
            head_attn_v_scale = fused_attn_v_head_scales.get((i, h), attn_v_scale)
            scales[f"{b}_head{h}_attn_v"] = head_attn_v_scale
            per_head_attn_v_scales.append(head_attn_v_scale)

        # Concatenated head outputs feed into out_proj
        scales[f"{b}_concat"] = max(per_head_attn_v_scales or [attn_v_scale])

        # Force out_proj REQUANT to block_input_scale so that the residual1 VADD
        # (block_input + out_proj_output) has both operands at the same scale.
        residual1_scale = residual1_block_scales.get(i, block_input_scale)
        scales[f"{b}_out_proj"] = residual1_scale
        scales[f"{b}_residual1"] = residual1_scale

        ln2_scale = get(f"{p}.layernorm_after")
        scales[f"{b}_ln2"] = ln2_scale

        fc1_scale = get(f"{p}.intermediate.dense")
        scales[f"{b}_fc1"] = fc1_scale
        # Post-GELU range is 2-3.5× smaller than pre-GELU (negatives become ~0).
        # GELUActivation is a leaf module so calibration captures its output directly.
        scales[f"{b}_gelu"] = gelu_block_scales.get(
            i,
            get(f"{p}.intermediate.intermediate_act_fn", fc1_scale),
        )

        # Force fc2 REQUANT to block_input_scale so that the residual2 VADD
        # (residual1_output + fc2_output) has both operands at the same scale.
        # residual1 output is already at the selected residual1 scale from the fix above.
        scales[f"{b}_fc2"] = residual1_scale
        scales[f"{b}_residual2"] = residual1_scale

    final_ln_scale = (
        max(float(final_ln_scale_override), 1e-8)
        if final_ln_scale_override is not None
        else get("vit.layernorm")
    )
    scales["final_ln"] = final_ln_scale
    scales["cls_extract"] = final_ln_scale
    scales["classifier"] = get("classifier", 1.0)
    for node_name, scale in activation_scale_overrides.items():
        scales[node_name] = max(float(scale), 1e-8)
    return scales


def build_runtime_twin_uniform_manifest(
    program,
    *,
    softmax_max_probs: dict,
    cal_scales: dict,
    twin_uniform_softmax_blocks,
    twin_uniform_gelu_blocks,
    twin_uniform_mode: str,
    block_replay_samples: dict,
):
    """Build per-PC twin-uniform specs from compare_golden selections."""
    if twin_uniform_mode == "off":
        return {"mode": "off", "softmax": {}, "gelu": {}}

    softmax_blocks = set(twin_uniform_softmax_blocks or [])
    gelu_blocks = set(twin_uniform_gelu_blocks or [])
    if not softmax_blocks and not gelu_blocks:
        return {"mode": twin_uniform_mode, "softmax": {}, "gelu": {}}

    softmax_specs = {}
    gelu_specs = {}

    for pc, events in sorted((program.trace_manifest or {}).items()):
        for event in events:
            node_name = str(event.get("node_name", ""))
            softmax_match = re.match(r"block(\d+)_head(\d+)_softmax$", node_name)
            if softmax_match:
                block_idx = int(softmax_match.group(1))
                head_idx = int(softmax_match.group(2))
                if block_idx in softmax_blocks and (block_idx, head_idx) in softmax_max_probs:
                    softmax_specs[str(pc)] = {
                        "mode": twin_uniform_mode,
                        "range1_max": float(softmax_max_probs[(block_idx, head_idx)]),
                        "block": block_idx,
                        "head": head_idx,
                        "node_name": node_name,
                    }
                continue

            gelu_match = re.match(r"block(\d+)_gelu$", node_name)
            if gelu_match:
                block_idx = int(gelu_match.group(1))
                if block_idx not in gelu_blocks:
                    continue
                block_samples = block_replay_samples.get(block_idx, [])
                negative_extent = max(
                    [float(max(-np.min(sample["gelu"]), 0.0)) for sample in block_samples] or [1e-8]
                )
                gelu_specs[str(pc)] = {
                    "mode": twin_uniform_mode,
                    "positive_range_max": float(cal_scales.get(node_name, 1.0 / 127.0) * 127.0),
                    "negative_extent": float(max(negative_extent, 1e-8)),
                    "block": block_idx,
                    "node_name": node_name,
                }

    return {
        "mode": twin_uniform_mode,
        "softmax": softmax_specs,
        "gelu": gelu_specs,
    }


def compile_model(
    model,
    state_dict,
    sample_images,
    processor,
    softmax_mode="max",
    softmax_percentile=99.0,
    softmax_min_prob=1e-4,
    softmax_max_prob=1.0,
    final_logit_mode="off",
    bias_correction=False,
    bias_correction_layers="",
    activation_percentile_nodes=None,
    output_aware_clipping_fc1_blocks=None,
    output_aware_clipping_fc2_blocks=None,
    output_aware_clipping_classifier=False,
    output_aware_clipping_candidates=25,
    output_aware_clipping_alpha_min=0.5,
    adaround_fc1_blocks=None,
    adaround_fc2_blocks=None,
    softmax_search_heads=None,
    softmax_search_objective="local_prob",
    attn_v_mode="max",
    attn_v_percentile=99.0,
    attn_v_safety_margin=1.10,
    attn_v_search_blocks=None,
    attn_v_search_objective="local_attn_v",
    gelu_output_mode="off",
    gelu_search_blocks=None,
    gelu_search_objective="downstream_residual2",
    hessian_calibration_images=0,
    hessian_target_nodes="",
    twin_uniform_softmax_blocks=None,
    twin_uniform_gelu_blocks=None,
    twin_uniform_mode="off",
    twin_uniform_disable_hessian=False,
    value_head_mode="off",
    per_head_qkv_calibration=False,
    gelu_from_accum=False,
    gelu_from_accum_blocks=None,
    dequant_add_residual1_blocks=None,
    dequant_add_residual1_scale_mode="max",
    dequant_add_residual1_scale_percentile=99.9,
    dequant_add_residual1_scale_alpha=1.0,
    fused_softmax_attnv_blocks=None,
    fused_softmax_attnv_accum_out_proj=False,
    smoothquant_targets="off",
    smoothquant_alpha=0.5,
    smoothquant_blocks=None,
    requant_pc_qkv=False,
    requant_pc_qkv_selection=None,
    requant_pc_fc1=False,
    requant_pc_fc1_blocks=None,
    requant_pc_fc2=False,
    requant_pc_fc2_blocks=None,
    requant_pc_out_proj=False,
    requant_pc_out_proj_blocks=None,
):
    """Compile the model to a ProgramBinary using calibration from sample images.

    Returns (program, cal_scales) so callers can pass the embedding scale to
    patch_embed_int8() for consistent INT8 quantization.
    """
    print("  Calibrating activation scales...")
    sample_inputs = [processor(images=img, return_tensors="pt") for img in sample_images]
    compile_state_dict = {
        name: tensor.detach().clone() if torch.is_tensor(tensor) else copy.deepcopy(tensor)
        for name, tensor in state_dict.items()
    }
    compile_model_ref = model
    smoothquant_factors = {}
    if smoothquant_targets != "off":
        shown_blocks = (
            ",".join(str(block_idx) for block_idx in sorted(smoothquant_blocks))
            if smoothquant_blocks is not None else "all"
        )
        print(
            f"  SmoothQuant: targets={smoothquant_targets} alpha={smoothquant_alpha:.2f} "
            f"blocks={shown_blocks} "
            "(experimental)"
        )
        smoothquant_factors = compute_smooth_factors(
            model,
            sample_inputs,
            alpha=smoothquant_alpha,
            targets=smoothquant_targets,
            blocks=smoothquant_blocks,
        )
        compile_state_dict = apply_smooth_quant(compile_state_dict, smoothquant_factors)
        compile_model_ref = copy.deepcopy(model)
        compile_model_ref.load_state_dict(compile_state_dict, strict=False)
    activation_percentile_targets = resolve_activation_percentile_targets(activation_percentile_nodes)
    calibration = calibrate_model(
        compile_model_ref,
        sample_inputs,
        percentile_module_names={
            target["module_name"] for target in activation_percentile_targets.values()
        },
    )
    value_head_scales = {}
    value_head_debug = {}
    if value_head_mode != "off":
        value_head_scales, value_head_debug = calibrate_value_head_scales(
            compile_model_ref,
            sample_inputs,
            mode=value_head_mode,
        )
    qkv_head_scales = {}
    if per_head_qkv_calibration:
        qkv_head_scales = calibrate_qkv_head_scales(compile_model_ref, sample_inputs)
    if softmax_mode == "search" and _uses_replay_softmax_objective(softmax_search_objective) and not softmax_search_heads:
        softmax_search_heads = (
            {(11, 0), (11, 1), (11, 2)}
            if softmax_search_objective == "hessian_prob"
            else {(11, 1), (11, 2)}
        )
    if attn_v_mode == "search" and _uses_downstream_attn_v_objective(attn_v_search_objective) and attn_v_search_blocks is None:
        attn_v_search_blocks = {11}
    if gelu_output_mode == "search" and gelu_search_blocks is None:
        gelu_search_blocks = {9, 10} if gelu_search_objective == "hessian_output" else {9, 10, 11}
    if twin_uniform_mode != "off":
        if twin_uniform_softmax_blocks is None:
            twin_uniform_softmax_blocks = {11}
        if twin_uniform_gelu_blocks is None:
            twin_uniform_gelu_blocks = {9, 10}
    if hessian_calibration_images and hessian_calibration_images > 0:
        replay_sample_inputs = sample_inputs[: min(len(sample_inputs), int(hessian_calibration_images))]
    else:
        replay_sample_inputs = sample_inputs

    downstream_contexts = {}
    block_replay_samples = {}
    needed_blocks = set()
    if softmax_mode == "search" and _uses_replay_softmax_objective(softmax_search_objective):
        needed_blocks.update(layer_idx for layer_idx, _ in softmax_search_heads)
    if attn_v_mode == "search" and _uses_downstream_attn_v_objective(attn_v_search_objective):
        needed_blocks.update(attn_v_search_blocks)
    if gelu_output_mode == "search":
        needed_blocks.update(gelu_search_blocks)
    if twin_uniform_mode != "off" and twin_uniform_gelu_blocks is not None:
        needed_blocks.update(twin_uniform_gelu_blocks)
    if dequant_add_residual1_blocks is not None and dequant_add_residual1_scale_mode != "default":
        needed_blocks.update(dequant_add_residual1_blocks)
    if fused_softmax_attnv_accum_out_proj and fused_softmax_attnv_blocks is not None:
        needed_blocks.update(fused_softmax_attnv_blocks)
    if needed_blocks:
        block_replay_samples = collect_block_replay_tensors(
            compile_model_ref,
            replay_sample_inputs,
            sorted(needed_blocks),
        )

    if softmax_mode == "search" and _uses_replay_softmax_objective(softmax_search_objective):
        value_scales_by_head = default_value_head_scales(
            calibration,
            qkv_head_scales=qkv_head_scales,
            value_head_scales=value_head_scales,
        )
        out_proj_scale = default_block_input_scale(calibration)
        for layer_idx, head_idx in softmax_search_heads:
            layer = compile_model_ref.vit.encoder.layer[layer_idx].attention.output.dense
            per_head_value_scales = {
                h: value_scales_by_head[(layer_idx, h)]
                for h in range(NUM_HEADS)
            }
            downstream_contexts[(layer_idx, head_idx)] = {
                "block_samples": block_replay_samples[layer_idx],
                "value_scales": per_head_value_scales,
                "attn_v_scale": max(per_head_value_scales.values()),
                "out_proj_scale": out_proj_scale,
                "out_proj_weight": layer.weight.detach().cpu().numpy().astype(np.float32),
                "out_proj_bias": layer.bias.detach().cpu().numpy().astype(np.float32),
                "twin_uniform_mode": (
                    twin_uniform_mode
                    if (
                        not twin_uniform_disable_hessian
                        and twin_uniform_mode != "off"
                        and twin_uniform_softmax_blocks is not None
                        and layer_idx in twin_uniform_softmax_blocks
                    )
                    else "off"
                ),
            }

    softmax_max_probs, qkt_max_abs, softmax_debug = calibrate_softmax_scales(
        compile_model_ref,
        sample_inputs,
        mode=softmax_mode,
        percentile=softmax_percentile,
        min_prob=softmax_min_prob,
        max_prob=softmax_max_prob,
        search_heads=softmax_search_heads,
        search_objective=softmax_search_objective,
        downstream_contexts=downstream_contexts,
    )
    attn_v_downstream_contexts = {}
    if attn_v_mode == "search" and _uses_downstream_attn_v_objective(attn_v_search_objective):
        value_scales_by_head = default_value_head_scales(
            calibration,
            qkv_head_scales=qkv_head_scales,
            value_head_scales=value_head_scales,
        )
        out_proj_scale = default_block_input_scale(calibration)
        for layer_idx in attn_v_search_blocks:
            layer = compile_model_ref.vit.encoder.layer[layer_idx].attention.output.dense
            attn_v_downstream_contexts[layer_idx] = {
                "block_samples": block_replay_samples[layer_idx],
                "default_head_probs": {
                    h: softmax_max_probs[(layer_idx, h)]
                    for h in range(NUM_HEADS)
                },
                "value_scales": {
                    h: value_scales_by_head[(layer_idx, h)]
                    for h in range(NUM_HEADS)
                },
                "out_proj_scale": out_proj_scale,
                "out_proj_weight": layer.weight.detach().cpu().numpy().astype(np.float32),
                "out_proj_bias": layer.bias.detach().cpu().numpy().astype(np.float32),
            }
    attn_v_block_scales = {}
    attn_v_debug = {}
    if attn_v_mode != "off":
        attn_v_block_scales, attn_v_debug = calibrate_attn_v_scales(
            compile_model_ref,
            sample_inputs,
            mode=attn_v_mode,
            percentile=attn_v_percentile,
            safety_margin=attn_v_safety_margin,
            default_block_scales=default_attn_v_block_scales(
                calibration,
                qkv_head_scales=qkv_head_scales,
                value_head_scales=value_head_scales,
            ),
            search_blocks=attn_v_search_blocks,
            search_objective=attn_v_search_objective,
            downstream_contexts=attn_v_downstream_contexts,
        )
    fused_attn_v_head_scales = {}
    fused_attn_v_head_debug = {}
    if fused_softmax_attnv_accum_out_proj and fused_softmax_attnv_blocks is not None:
        value_scales_by_head = default_value_head_scales(
            calibration,
            qkv_head_scales=qkv_head_scales,
            value_head_scales=value_head_scales,
        )
        default_block_scales = default_attn_v_block_scales(
            calibration,
            qkv_head_scales=qkv_head_scales,
            value_head_scales=value_head_scales,
        )
        out_proj_scale = default_block_input_scale(calibration)
        fused_head_contexts = {}
        for layer_idx in sorted(fused_softmax_attnv_blocks):
            layer = compile_model_ref.vit.encoder.layer[layer_idx].attention.output.dense
            fused_head_contexts[layer_idx] = {
                "default_attn_v_scale": float(attn_v_block_scales.get(layer_idx, default_block_scales[layer_idx])),
                "default_head_probs": {
                    h: softmax_max_probs[(layer_idx, h)]
                    for h in range(NUM_HEADS)
                },
                "value_scales": {
                    h: value_scales_by_head[(layer_idx, h)]
                    for h in range(NUM_HEADS)
                },
                "out_proj_scale": out_proj_scale,
                "out_proj_weight": layer.weight.detach().cpu().numpy().astype(np.float32),
                "out_proj_bias": layer.bias.detach().cpu().numpy().astype(np.float32),
            }
        fused_attn_v_head_scales, fused_attn_v_head_debug = calibrate_fused_attn_v_head_scales(
            block_replay_samples,
            fused_head_contexts,
            search_objective=attn_v_search_objective,
        )
    gelu_block_scales = {}
    gelu_debug = {}
    if gelu_output_mode == "search":
        fc2_scale = default_block_input_scale(calibration)
        for block_idx in sorted(gelu_search_blocks):
            layer = compile_model_ref.vit.encoder.layer[block_idx].output.dense
            best_scale, best_debug = select_best_gelu_output_scale(
                block_replay_samples[block_idx],
                default_scale=calibration.scales.get(
                    f"vit.encoder.layer.{block_idx}.intermediate.intermediate_act_fn",
                    calibration.scales.get(f"vit.encoder.layer.{block_idx}.intermediate.dense", 6.0 / 127.0),
                ),
                fc2_scale=fc2_scale,
                fc2_weight=layer.weight.detach().cpu().numpy().astype(np.float32),
                fc2_bias=layer.bias.detach().cpu().numpy().astype(np.float32),
                search_objective=gelu_search_objective,
                twin_uniform_mode=(
                    twin_uniform_mode
                    if (
                        not twin_uniform_disable_hessian
                        and twin_uniform_mode != "off"
                        and twin_uniform_gelu_blocks is not None
                        and block_idx in twin_uniform_gelu_blocks
                    )
                    else "off"
                ),
            )
            gelu_block_scales[block_idx] = best_scale
            gelu_debug[block_idx] = best_debug
    residual1_block_scales = {}
    if dequant_add_residual1_blocks is not None:
        residual1_block_scales = calibrate_residual1_block_scales(
            block_replay_samples,
            dequant_add_residual1_blocks,
            mode=dequant_add_residual1_scale_mode,
            default_scale=default_block_input_scale(calibration),
            percentile=dequant_add_residual1_scale_percentile,
            blend_alpha=dequant_add_residual1_scale_alpha,
        )
    final_logit_scale_override = None
    final_logit_debug = {}
    if final_logit_mode == "search":
        final_logit_samples = collect_final_logit_tensors(compile_model_ref, processor, sample_images)
        classifier_weight = compile_state_dict["classifier.weight"].detach().cpu().numpy().astype(np.float32)
        classifier_bias_tensor = compile_state_dict.get("classifier.bias")
        if classifier_bias_tensor is None:
            classifier_bias = np.zeros(classifier_weight.shape[0], dtype=np.float32)
        else:
            classifier_bias = classifier_bias_tensor.detach().cpu().numpy().astype(np.float32)
        final_logit_scale_override, final_logit_debug = select_best_final_logit_scale(
            final_logit_samples,
            default_scale=calibration.scales.get("vit.layernorm", 6.0 / 127.0),
            classifier_weight=classifier_weight,
            classifier_bias=classifier_bias,
        )
    activation_scale_overrides = {}
    if activation_percentile_targets:
        for node_name, target in activation_percentile_targets.items():
            scale = calibration.percentile_scale(target["module_name"], target["percentile"])
            for scale_key in target["scale_keys"]:
                activation_scale_overrides[scale_key] = scale
        shown = " ".join(
            f"{node_name}=p{target['percentile']:.1f}->{activation_scale_overrides[target['scale_keys'][0]]:.5f}"
            for node_name, target in activation_percentile_targets.items()
        )
        if shown:
            print(f"  Activation percentile overrides: {shown}")
    cal_scales = build_calibration_scales(
        calibration,
        softmax_max_probs,
        qkt_max_abs,
        attn_v_block_scales=attn_v_block_scales,
        fused_attn_v_head_scales=fused_attn_v_head_scales,
        qkv_head_scales=qkv_head_scales,
        value_head_scales=value_head_scales,
        gelu_block_scales=gelu_block_scales,
        residual1_block_scales=residual1_block_scales,
        final_ln_scale_override=final_logit_scale_override,
        activation_scale_overrides=activation_scale_overrides,
    )
    embed_scale = cal_scales.get("pos_embed_add", 14.0 / 127.0)
    print(f"  Got {len(cal_scales)} calibration entries; "
          f"block0_ln1={cal_scales.get('block0_ln1', 'N/A'):.5f}  "
          f"pos_embed_add={embed_scale:.5f}")
    print(
        "  Softmax calibration: "
        f"mode={softmax_mode} percentile={softmax_percentile:.1f} "
        f"clip=[{softmax_min_prob:.3f},{softmax_max_prob:.3f}]"
    )
    # Show per-head max probs and QKT scales for block 0
    b0_probs = [softmax_max_probs.get((0, h), 0.20) for h in range(NUM_HEADS)]
    print(f"  Softmax max_prob block0: " + " ".join(f"h{h}={p:.3f}" for h, p in enumerate(b0_probs)))
    b0_qkt = [qkt_max_abs.get((0, h), 6.0) for h in range(NUM_HEADS)]
    print(f"  QKT max_abs   block0: " + " ".join(f"h{h}={v:.2f}" for h, v in enumerate(b0_qkt)))
    if softmax_mode == "search":
        heads = sorted(softmax_search_heads) if softmax_search_heads is not None else sorted(softmax_debug)
        shown = [head for head in heads if head in softmax_debug][:4]
        if shown:
            print(
                "  Softmax search picks: "
                + " ".join(
                    f"b{layer}h{head}={softmax_debug[(layer, head)].get('label', 'n/a')}"
                    for layer, head in shown
                )
            )
            print(f"  Softmax search objective: {softmax_search_objective}")
    if attn_v_mode == "off":
        print("  attn@V calibration: off (using V projection scales)")
    else:
        print(
            "  attn@V scales block0/block5/block11: "
            f"{attn_v_block_scales.get(0, cal_scales.get('block0_concat', 0.0)):.5f} "
            f"{attn_v_block_scales.get(5, cal_scales.get('block5_concat', 0.0)):.5f} "
            f"{attn_v_block_scales.get(11, cal_scales.get('block11_concat', 0.0)):.5f}"
        )
        if attn_v_mode == "search":
            blocks = sorted(attn_v_search_blocks) if attn_v_search_blocks is not None else list(range(DEPTH))
            shown = [b for b in blocks if b in attn_v_debug][:4]
            if shown:
                print(
                    "  attn@V search picks: "
                    + " ".join(f"b{b}={attn_v_debug[b].get('label', 'n/a')}" for b in shown)
                )
                print(f"  attn@V search objective: {attn_v_search_objective}")
    if per_head_qkv_calibration:
        print(
            "  Per-head Q/K/V scales block0: "
            + " ".join(
                f"h{h}=({cal_scales.get(f'block0_head{h}_query', 0.0):.5f},"
                f"{cal_scales.get(f'block0_head{h}_key', 0.0):.5f},"
                f"{cal_scales.get(f'block0_head{h}_value', 0.0):.5f})"
                for h in range(NUM_HEADS)
            )
        )
    if value_head_mode != "off":
        print(
            "  Value-head scales block0: "
            + " ".join(
                f"h{h}={cal_scales.get(f'block0_head{h}_value', 0.0):.5f}"
                for h in range(NUM_HEADS)
            )
        )
        print(
            "  Value-head search block0: "
            + " ".join(
                f"h{h}={value_head_debug.get((0, h), {}).get('label', 'n/a')}"
                for h in range(NUM_HEADS)
            )
        )
    if gelu_output_mode == "search":
        shown = [b for b in sorted(gelu_search_blocks) if b in gelu_debug][:4]
        if shown:
            print(
                "  GELU search picks: "
                + " ".join(f"b{b}={gelu_debug[b].get('label', 'n/a')}" for b in shown)
            )
    if smoothquant_factors:
        print(f"  SmoothQuant groups: {len(smoothquant_factors)}")
    if final_logit_mode == "search":
        print(
            "  Final-logit search pick: "
            f"{final_logit_debug.get('label', 'n/a')} "
            f"scale={final_logit_debug.get('scale', cal_scales.get('final_ln', 0.0)):.5f} "
            f"mean={final_logit_debug.get('mean_classifier_cosine', 0.0):.4f} "
            f"p10={final_logit_debug.get('p10_classifier_cosine', 0.0):.4f} "
            f"min={final_logit_debug.get('min_classifier_cosine', 0.0):.4f} "
            f"top1={final_logit_debug.get('top1_agreement', 0.0):.3f}"
        )
    if requant_pc_qkv:
        if requant_pc_qkv_selection is None:
            print("  REQUANT_PC: enabled for all per-head Q/K/V projections (experimental)")
        else:
            total = DEPTH * len(QKV_PROJECTIONS) * NUM_HEADS
            selected = len(requant_pc_qkv_selection)
            sample = " ".join(
                f"b{layer}:{proj[0].upper()}h{head}"
                for layer, proj, head in sorted(requant_pc_qkv_selection)[:6]
            )
            print(
                f"  REQUANT_PC: enabled for {selected}/{total} Q/K/V projections "
                f"(experimental){' [' + sample + (' ...' if selected > 6 else '') + ']' if sample else ''}"
            )
    if requant_pc_out_proj:
        shown_out_proj_blocks = (
            ",".join(str(block_idx) for block_idx in sorted(requant_pc_out_proj_blocks))
            if requant_pc_out_proj_blocks is not None else "all"
        )
        print(
            f"  REQUANT_PC: enabled for out_proj matmuls blocks={shown_out_proj_blocks} (experimental)"
        )
    if requant_pc_fc1:
        shown_fc1_blocks = (
            ",".join(str(block_idx) for block_idx in sorted(requant_pc_fc1_blocks))
            if requant_pc_fc1_blocks is not None else "all"
        )
        print(f"  REQUANT_PC: enabled for FC1 matmuls blocks={shown_fc1_blocks} (experimental)")
    if requant_pc_fc2:
        shown_fc2_blocks = (
            ",".join(str(block_idx) for block_idx in sorted(requant_pc_fc2_blocks))
            if requant_pc_fc2_blocks is not None else "all"
        )
        print(f"  REQUANT_PC: enabled for FC2 matmuls blocks={shown_fc2_blocks} (experimental)")
    if fused_softmax_attnv_blocks is not None:
        print(
            "  Fused SOFTMAX_ATTNV blocks: "
            + ",".join(str(block_idx) for block_idx in sorted(fused_softmax_attnv_blocks))
            + " (experimental)"
        )
        if fused_softmax_attnv_accum_out_proj:
            print("  Fused out_proj accumulation: enabled (experimental)")
            shown = []
            for block_idx in sorted(fused_softmax_attnv_blocks):
                for head_idx in range(NUM_HEADS):
                    key = (block_idx, head_idx)
                    if key in fused_attn_v_head_debug:
                        shown.append(
                            f"b{block_idx}h{head_idx}={fused_attn_v_head_debug[key].get('label', 'n/a')}"
                        )
            if shown:
                print("  Fused attn_v head scales: " + " ".join(shown[:6]) + (" ..." if len(shown) > 6 else ""))
    if dequant_add_residual1_blocks is not None:
        print(
            "  DEQUANT_ADD residual1 blocks: "
            + ",".join(str(block_idx) for block_idx in sorted(dequant_add_residual1_blocks))
            + " (experimental)"
        )
        print(
            "  DEQUANT_ADD residual1 scale mode: "
            f"{dequant_add_residual1_scale_mode}"
            + (
                f" percentile={dequant_add_residual1_scale_percentile:.1f}"
                if dequant_add_residual1_scale_mode == "percentile" else ""
            )
            + (
                f" alpha={dequant_add_residual1_scale_alpha:.2f}"
                if dequant_add_residual1_scale_mode == "blend" else ""
            )
        )
        shown = " ".join(
            f"b{block_idx}={cal_scales.get(f'block{block_idx}_residual1', 0.0):.5f}"
            for block_idx in sorted(dequant_add_residual1_blocks)
        )
        if shown:
            print(f"  Residual1 scales: {shown}")

    weight_quantization_overrides = {}
    if (
        output_aware_clipping_fc1_blocks is not None
        or adaround_fc1_blocks is not None
        or output_aware_clipping_fc2_blocks is not None
        or adaround_fc2_blocks is not None
        or output_aware_clipping_classifier
    ):
        weight_quantization_overrides.update(build_fc1_weight_quantization_overrides(
            compile_model_ref,
            sample_inputs,
            output_aware_clipping_fc1_blocks,
            adaround_block_indices=adaround_fc1_blocks,
            requant_pc_fc1=requant_pc_fc1,
            requant_pc_fc1_blocks=requant_pc_fc1_blocks,
            n_candidates=output_aware_clipping_candidates,
            alpha_min=output_aware_clipping_alpha_min,
        ))
        weight_quantization_overrides.update(build_block_dense_weight_quantization_overrides(
            compile_model_ref,
            sample_inputs,
            output_aware_clipping_fc2_blocks,
            module_suffix="output.dense",
            requant_pc_enabled=requant_pc_fc2,
            requant_pc_blocks=requant_pc_fc2_blocks,
            adaround_block_indices=adaround_fc2_blocks,
            n_candidates=output_aware_clipping_candidates,
            alpha_min=output_aware_clipping_alpha_min,
        ))
        weight_quantization_overrides.update(build_classifier_weight_quantization_override(
            compile_model_ref,
            sample_inputs,
            enabled=output_aware_clipping_classifier,
            n_candidates=output_aware_clipping_candidates,
            alpha_min=output_aware_clipping_alpha_min,
        ))
        if output_aware_clipping_fc1_blocks is not None:
            shown = ",".join(str(block_idx) for block_idx in sorted(output_aware_clipping_fc1_blocks))
            print(
                "  Output-aware clipping FC1 blocks="
                f"{shown} candidates={output_aware_clipping_candidates} alpha_min={output_aware_clipping_alpha_min:.2f}"
            )
            if requant_pc_fc1:
                pc_blocks = (
                    set(range(DEPTH)) if requant_pc_fc1_blocks is None else set(requant_pc_fc1_blocks)
                )
                mode_summary = " ".join(
                    f"b{block_idx}={'pc' if block_idx in pc_blocks else 'tensor'}"
                    for block_idx in sorted(output_aware_clipping_fc1_blocks)
                )
                print(f"  Output-aware clipping modes: {mode_summary}")
        if adaround_fc1_blocks is not None:
            shown = ",".join(str(block_idx) for block_idx in sorted(adaround_fc1_blocks))
            print(f"  AdaRound FC1 blocks={shown} (experimental)")
        if output_aware_clipping_fc2_blocks is not None:
            shown = ",".join(str(block_idx) for block_idx in sorted(output_aware_clipping_fc2_blocks))
            print(
                "  Output-aware clipping FC2 blocks="
                f"{shown} candidates={output_aware_clipping_candidates} alpha_min={output_aware_clipping_alpha_min:.2f}"
            )
            if requant_pc_fc2:
                pc_blocks = (
                    set(range(DEPTH)) if requant_pc_fc2_blocks is None else set(requant_pc_fc2_blocks)
                )
                mode_summary = " ".join(
                    f"b{block_idx}={'pc' if block_idx in pc_blocks else 'tensor'}"
                    for block_idx in sorted(output_aware_clipping_fc2_blocks)
                )
                print(f"  Output-aware clipping FC2 modes: {mode_summary}")
        if adaround_fc2_blocks is not None:
            shown = ",".join(str(block_idx) for block_idx in sorted(adaround_fc2_blocks))
            print(f"  AdaRound FC2 blocks={shown} (experimental)")
        if output_aware_clipping_classifier:
            print("  Output-aware clipping classifier: enabled")

    bias_correction_targets = []
    bias_corrections = None
    if bias_correction:
        bias_correction_targets = resolve_bias_correction_targets(
            compile_state_dict,
            bias_correction_layers,
        )
        print(
            "  Bias correction targets: "
            + ", ".join(bias_correction_targets[:4])
            + (" ..." if len(bias_correction_targets) > 4 else "")
        )
        bias_corrections = compute_bias_corrections(
            compile_model_ref,
            compile_state_dict,
            quantize_weights(
                compile_state_dict,
                quantization_overrides=weight_quantization_overrides,
            ),
            cal_scales,
            sample_inputs,
            bias_correction_targets,
        )
        if bias_corrections:
            corrected_biases = sorted(bias_corrections)
            summaries = sorted(
                (
                    bias_name,
                    float(np.mean(np.abs(correction))),
                    float(np.max(np.abs(correction))),
                )
                for bias_name, correction in bias_corrections.items()
            )
            shown = " ".join(
                f"{name}:mean|corr|={mean_abs:.4e},max={max_abs:.4e}"
                for name, mean_abs, max_abs in summaries[:3]
            )
            print(
                f"  Bias correction computed for {len(corrected_biases)} biases"
                + (f" [{shown}{' ...' if len(summaries) > 3 else ''}]" if shown else "")
            )

    compiler = Compiler()
    program = compiler.compile(
        compile_state_dict,
        calibration=type('C', (), {'scales': cal_scales})(),
        bias_corrections=bias_corrections,
        weight_quantization_overrides=weight_quantization_overrides,
        gelu_from_accum=gelu_from_accum,
        gelu_from_accum_blocks=gelu_from_accum_blocks,
        dequant_add_residual1_blocks=dequant_add_residual1_blocks,
        fused_softmax_attnv_blocks=fused_softmax_attnv_blocks,
        fused_softmax_attnv_accum_out_proj=fused_softmax_attnv_accum_out_proj,
        requant_pc_qkv=requant_pc_qkv,
        requant_pc_qkv_selection=requant_pc_qkv_selection,
        requant_pc_fc1=requant_pc_fc1,
        requant_pc_fc1_blocks=requant_pc_fc1_blocks,
        requant_pc_fc2=requant_pc_fc2,
        requant_pc_fc2_blocks=requant_pc_fc2_blocks,
        requant_pc_out_proj=requant_pc_out_proj,
        requant_pc_out_proj_blocks=requant_pc_out_proj_blocks,
    )
    program.compiler_manifest = dict(program.compiler_manifest or {})
    runtime_twin_uniform = build_runtime_twin_uniform_manifest(
        program,
        softmax_max_probs=softmax_max_probs,
        cal_scales=cal_scales,
        twin_uniform_softmax_blocks=twin_uniform_softmax_blocks,
        twin_uniform_gelu_blocks=twin_uniform_gelu_blocks,
        twin_uniform_mode=twin_uniform_mode,
        block_replay_samples=block_replay_samples,
    )
    program.compiler_manifest["runtime_twin_uniform"] = runtime_twin_uniform
    program.compiler_manifest["compare_golden_compile"] = {
        "sample_image_count": int(len(sample_images)),
        "softmax": {
            "mode": softmax_mode,
            "percentile": float(softmax_percentile),
            "min_prob": float(softmax_min_prob),
            "max_prob": float(softmax_max_prob),
            "search_heads": (
                [
                    {"block": int(block), "head": int(head)}
                    for block, head in sorted(softmax_search_heads)
                ]
                if softmax_search_heads else []
            ),
            "search_objective": softmax_search_objective,
        },
        "attn_v": {
            "mode": attn_v_mode,
            "percentile": float(attn_v_percentile),
            "safety_margin": float(attn_v_safety_margin),
            "search_blocks": (
                sorted(int(block_idx) for block_idx in attn_v_search_blocks)
                if attn_v_search_blocks is not None else []
            ),
            "search_objective": attn_v_search_objective,
        },
        "gelu_output": {
            "mode": gelu_output_mode,
            "search_blocks": (
                sorted(int(block_idx) for block_idx in gelu_search_blocks)
                if gelu_search_blocks is not None else []
            ),
            "search_objective": gelu_search_objective,
        },
        "ptq4vit": {
            "hessian_calibration_images": int(hessian_calibration_images),
            "hessian_target_nodes": hessian_target_nodes,
            "twin_uniform_softmax_blocks": (
                sorted(int(block_idx) for block_idx in twin_uniform_softmax_blocks)
                if twin_uniform_softmax_blocks is not None else []
            ),
            "twin_uniform_gelu_blocks": (
                sorted(int(block_idx) for block_idx in twin_uniform_gelu_blocks)
                if twin_uniform_gelu_blocks is not None else []
            ),
            "twin_uniform_mode": twin_uniform_mode,
            "twin_uniform_disable_hessian": bool(twin_uniform_disable_hessian),
            "runtime_twin_uniform": {
                "softmax_pc_count": int(len(runtime_twin_uniform.get("softmax", {}))),
                "gelu_pc_count": int(len(runtime_twin_uniform.get("gelu", {}))),
            },
        },
        "smoothquant": {
            "targets": smoothquant_targets,
            "alpha": float(smoothquant_alpha),
            "blocks": (
                sorted(int(block_idx) for block_idx in smoothquant_blocks)
                if smoothquant_blocks is not None else None
            ),
            "group_count": int(len(smoothquant_factors)),
        },
        "bias_correction": {
            "enabled": bool(bias_correction),
            "layers": bias_correction_layers,
            "target_weights": list(bias_correction_targets),
            "corrected_biases": (
                sorted(bias_corrections)
                if bias_corrections is not None else []
            ),
        },
        "activation_percentiles": {
            node_name: {
                "percentile": float(target["percentile"]),
                "module_name": target["module_name"],
                "applied_scale_keys": list(target["scale_keys"]),
                "scale": float(activation_scale_overrides[target["scale_keys"][0]]),
            }
            for node_name, target in activation_percentile_targets.items()
        },
        "output_aware_clipping": {
            "fc1_blocks": (
                sorted(int(block_idx) for block_idx in output_aware_clipping_fc1_blocks)
                if output_aware_clipping_fc1_blocks is not None else []
            ),
            "candidate_count": int(output_aware_clipping_candidates),
            "alpha_min": float(output_aware_clipping_alpha_min),
            "adaround_fc1_blocks": (
                sorted(int(block_idx) for block_idx in adaround_fc1_blocks)
                if adaround_fc1_blocks is not None else []
            ),
            "target_weights": sorted(weight_quantization_overrides),
        },
        "value_head_calibration": value_head_mode,
        "per_head_qkv_calibration": bool(per_head_qkv_calibration),
        "gelu_from_accum": bool(gelu_from_accum),
        "gelu_from_accum_blocks": (
            sorted(int(block_idx) for block_idx in gelu_from_accum_blocks)
            if gelu_from_accum_blocks is not None else None
        ),
        "dequant_add_residual1_blocks": (
            sorted(int(block_idx) for block_idx in dequant_add_residual1_blocks)
            if dequant_add_residual1_blocks is not None else None
        ),
        "dequant_add_residual1_scale_mode": dequant_add_residual1_scale_mode,
        "dequant_add_residual1_scale_percentile": float(dequant_add_residual1_scale_percentile),
        "dequant_add_residual1_scale_alpha": float(dequant_add_residual1_scale_alpha),
        "fused_softmax_attnv_blocks": (
            sorted(int(block_idx) for block_idx in fused_softmax_attnv_blocks)
            if fused_softmax_attnv_blocks is not None else None
        ),
        "fused_softmax_attnv_accum_out_proj": bool(fused_softmax_attnv_accum_out_proj),
        "requant_pc_qkv": bool(requant_pc_qkv),
        "requant_pc_qkv_selection": (
            [
                {"block": int(block), "projection": proj, "head": int(head)}
                for block, proj, head in sorted(requant_pc_qkv_selection)
            ]
            if requant_pc_qkv_selection is not None else None
        ),
        "requant_pc_fc1": bool(requant_pc_fc1),
        "requant_pc_fc1_blocks": (
            sorted(int(block_idx) for block_idx in requant_pc_fc1_blocks)
            if requant_pc_fc1_blocks is not None else None
        ),
        "requant_pc_out_proj": bool(requant_pc_out_proj),
    }
    return program, cal_scales


def main():
    parser = argparse.ArgumentParser(description="Compare FP32 vs golden model inference")
    parser.add_argument(
        "--diagnostic-preset",
        choices=sorted(DIAGNOSTIC_PRESETS),
        default="",
        help="Apply a canonical diagnostics preset (overrides benchmark/compiler trace knobs)",
    )
    parser.add_argument(
        "--benchmark-dataset",
        choices=["frozen_coco", "cats_dogs_local", "local_flat"],
        default="frozen_coco",
        help="Benchmark dataset source",
    )
    parser.add_argument("--max-images", type=int, default=len(FROZEN_EVAL_IMAGE_IDS))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--benchmark-image-source",
        choices=["local", "download"],
        default="local",
        help="Use the local frozen image cache or re-download COCO benchmark images",
    )
    parser.add_argument(
        "--local-benchmark-image-dir",
        default=LOCAL_FROZEN_IMAGE_DIR,
        help="Directory holding the local frozen benchmark image cache",
    )
    parser.add_argument(
        "--cats-dogs-image-dir",
        default=CATS_DOGS_IMAGE_DIR,
        help="Directory holding the flat local cats/dogs dataset",
    )
    parser.add_argument(
        "--populate-local-benchmark-cache",
        action="store_true",
        help="Download missing frozen benchmark images into the local cache before running",
    )
    parser.add_argument(
        "--calibration-images",
        type=int,
        default=0,
        help="Number of collected images to use for calibration (0 = use all collected images)",
    )
    parser.add_argument(
        "--softmax-calibration",
        choices=["max", "percentile", "search"],
        default="max",
        help="Softmax head calibration mode",
    )
    parser.add_argument(
        "--softmax-percentile",
        type=float,
        default=99.0,
        help="Percentile used when --softmax-calibration=percentile",
    )
    parser.add_argument(
        "--softmax-min-prob",
        type=float,
        default=1e-4,
        help="Minimum prob used for softmax scale clipping",
    )
    parser.add_argument(
        "--softmax-max-prob",
        type=float,
        default=1.0,
        help="Maximum prob used for softmax scale clipping",
    )
    parser.add_argument(
        "--final-logit-calibration",
        choices=["off", "search"],
        default="off",
        help="Experimental final_ln/cls_extract activation-scale search for classifier-logit fidelity",
    )
    parser.add_argument(
        "--bias-correction",
        action="store_true",
        help="Experimental analytical bias correction for selected linear layers",
    )
    parser.add_argument(
        "--bias-correction-layers",
        default="classifier",
        help="Comma-separated bias-correction selectors (classifier,late_out_proj,late_fc2,all or explicit *.weight names)",
    )
    parser.add_argument(
        "--act-percentile-nodes",
        default="",
        help="Comma-separated activation percentile overrides like final_ln:99.9,block9_ln2:99.5",
    )
    parser.add_argument(
        "--output-aware-clipping-fc1-blocks",
        default="",
        help="Comma-separated FC1 block indices to quantize with output-aware clipping",
    )
    parser.add_argument(
        "--output-aware-clipping-fc2-blocks",
        default="",
        help="Comma-separated FC2 block indices to quantize with output-aware clipping",
    )
    parser.add_argument(
        "--output-aware-clipping-classifier",
        action="store_true",
        help="Apply output-aware clipping to classifier weights",
    )
    parser.add_argument(
        "--output-aware-clipping-candidates",
        type=int,
        default=25,
        help="Number of alpha candidates searched by output-aware clipping",
    )
    parser.add_argument(
        "--output-aware-clipping-alpha-min",
        type=float,
        default=0.5,
        help="Minimum alpha/max_abs fraction searched by output-aware clipping",
    )
    parser.add_argument(
        "--adaround-fc1-blocks",
        default="",
        help="Comma-separated FC1 block indices to refine with greedy AdaRound after clipping/rounding",
    )
    parser.add_argument(
        "--adaround-fc2-blocks",
        default="",
        help="Comma-separated FC2 block indices to refine with greedy AdaRound after clipping/rounding",
    )
    parser.add_argument(
        "--softmax-search-heads",
        default="",
        help="Comma-separated block:head pairs to search when --softmax-calibration=search",
    )
    parser.add_argument(
        "--softmax-search-objective",
        choices=["local_prob", "downstream_out_proj", "tail_out_proj", "tail_attn_v", "hessian_prob"],
        default="local_prob",
        help="Objective used when --softmax-calibration=search",
    )
    parser.add_argument(
        "--attn-v-calibration",
        choices=["off", "max", "percentile", "search"],
        default="off",
        help="attn@V calibration mode",
    )
    parser.add_argument(
        "--attn-v-percentile",
        type=float,
        default=99.0,
        help="Percentile used when --attn-v-calibration=percentile",
    )
    parser.add_argument(
        "--attn-v-safety-margin",
        type=float,
        default=1.10,
        help="Safety margin applied to attn@V max_abs before quantization",
    )
    parser.add_argument(
        "--attn-v-search-blocks",
        default="",
        help="Comma-separated block indices to search when --attn-v-calibration=search",
    )
    parser.add_argument(
        "--attn-v-search-objective",
        choices=["local_attn_v", "downstream_out_proj", "tail_out_proj", "tail_attn_v"],
        default="local_attn_v",
        help="Objective used when --attn-v-calibration=search",
    )
    parser.add_argument(
        "--gelu-from-accum",
        action="store_true",
        help="Use the experimental GELU-from-ACCUM strip-mined path",
    )
    parser.add_argument(
        "--gelu-from-accum-blocks",
        default="",
        help="Comma-separated block indices that use GELU-from-ACCUM (default: all GELU-enabled blocks)",
    )
    parser.add_argument(
        "--fused-softmax-attnv",
        action="store_true",
        help="Use the experimental fused softmax->attn@V path for all attention blocks",
    )
    parser.add_argument(
        "--fused-softmax-attnv-blocks",
        default="",
        help="Comma-separated block indices that use the experimental fused softmax->attn@V path",
    )
    parser.add_argument(
        "--fused-softmax-attnv-accum-out-proj",
        action="store_true",
        help="For fused softmax->attn@V blocks, accumulate out_proj directly from per-head outputs before requant",
    )
    parser.add_argument(
        "--per-head-qkv-calibration",
        action="store_true",
        help="Calibrate Q/K/V activation scales per head instead of per layer",
    )
    parser.add_argument(
        "--value-head-calibration",
        choices=["off", "max", "search"],
        default="off",
        help="Calibrate only the per-head value projection scales",
    )
    parser.add_argument(
        "--gelu-output-calibration",
        choices=["off", "search"],
        default="off",
        help="Experimental GELU output-scale calibration mode",
    )
    parser.add_argument(
        "--gelu-search-blocks",
        default="9,10,11",
        help="Comma-separated block indices used by --gelu-output-calibration=search",
    )
    parser.add_argument(
        "--gelu-search-objective",
        choices=["downstream_residual2", "hessian_output"],
        default="downstream_residual2",
        help="Objective used when --gelu-output-calibration=search",
    )
    parser.add_argument(
        "--hessian-calibration-images",
        type=int,
        default=0,
        help="Optional limit on calibration images used for Hessian-guided replay objectives (0 = all)",
    )
    parser.add_argument(
        "--hessian-target-nodes",
        default="",
        help="Optional label or explicit node list for Hessian-guided experiments (metadata only in v1)",
    )
    parser.add_argument(
        "--twin-uniform-softmax-blocks",
        default="",
        help="Comma-separated block indices that use PTQ4ViT twin-uniform softmax emulation during search",
    )
    parser.add_argument(
        "--twin-uniform-gelu-blocks",
        default="",
        help="Comma-separated block indices that use PTQ4ViT twin-uniform GELU emulation during search",
    )
    parser.add_argument(
        "--twin-uniform-mode",
        choices=["off", "paper_exact"],
        default="off",
        help="Twin-uniform software emulation mode for PTQ4ViT-inspired search",
    )
    parser.add_argument(
        "--twin-uniform-disable-hessian",
        action="store_true",
        help="Disable twin-uniform emulation inside Hessian-guided search ablations",
    )
    parser.add_argument(
        "--smoothquant-targets",
        choices=["off", "ln1_qkv", "ln2_fc1", "both"],
        default="off",
        help="Experimental SmoothQuant target set",
    )
    parser.add_argument(
        "--smoothquant-alpha",
        type=float,
        default=0.5,
        help="SmoothQuant alpha used when --smoothquant-targets is enabled",
    )
    parser.add_argument(
        "--smoothquant-blocks",
        default="",
        help="Comma-separated block indices to SmoothQuant (default: all blocks)",
    )
    parser.add_argument(
        "--requant-pc-qkv",
        action="store_true",
        help="Experimental: use REQUANT_PC for per-head Q/K/V projections",
    )
    parser.add_argument(
        "--requant-pc-qkv-blocks",
        default="",
        help="Comma-separated block indices for selective Q/K/V REQUANT_PC rollout (default: all blocks)",
    )
    parser.add_argument(
        "--requant-pc-qkv-heads",
        default="",
        help="Comma-separated head indices for selective Q/K/V REQUANT_PC rollout (default: all heads)",
    )
    parser.add_argument(
        "--requant-pc-qkv-projections",
        default="all",
        help="Comma-separated Q/K/V projections for selective REQUANT_PC rollout (query,key,value or all)",
    )
    parser.add_argument(
        "--requant-pc-qkv-exclude",
        default="",
        help="Comma-separated block:projection:head triplets to exclude from Q/K/V REQUANT_PC",
    )
    parser.add_argument(
        "--requant-pc-fc1",
        action="store_true",
        help="Experimental: use REQUANT_PC for FC1 matmuls",
    )
    parser.add_argument(
        "--requant-pc-fc1-blocks",
        default="",
        help="Comma-separated block indices for selective FC1 REQUANT_PC rollout (default: all blocks)",
    )
    parser.add_argument(
        "--requant-pc-fc2",
        action="store_true",
        help="Experimental: use REQUANT_PC for FC2 matmuls",
    )
    parser.add_argument(
        "--requant-pc-fc2-blocks",
        default="",
        help="Comma-separated block indices for selective FC2 REQUANT_PC rollout (default: all blocks)",
    )
    parser.add_argument(
        "--requant-pc-out-proj",
        action="store_true",
        help="Experimental: use REQUANT_PC for out_proj matmuls",
    )
    parser.add_argument(
        "--requant-pc-out-proj-blocks",
        default="",
        help="Comma-separated block indices for selective out_proj REQUANT_PC rollout (default: all blocks)",
    )
    parser.add_argument(
        "--dequant-add-residual1-blocks",
        default=None,
        help="Comma-separated block indices that use DEQUANT_ADD for residual1 (default: disabled)",
    )
    parser.add_argument(
        "--dequant-add-residual1-scale-mode",
        choices=["default", "max", "percentile", "blend"],
        default="max",
        help="How to choose the output scale for DEQUANT_ADD residual1 blocks",
    )
    parser.add_argument(
        "--dequant-add-residual1-scale-percentile",
        type=float,
        default=99.9,
        help="Percentile used when --dequant-add-residual1-scale-mode=percentile",
    )
    parser.add_argument(
        "--dequant-add-residual1-scale-alpha",
        type=float,
        default=1.0,
        help="Blend weight between default and max residual1 scale when mode=blend",
    )
    parser.add_argument(
        "--output",
        default="golden_comparison.json",
        help="Path for JSON results output (default: golden_comparison.json)",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Print extra per-image diagnostics and mismatch severity",
    )
    parser.add_argument(
        "--trace-worst-k",
        type=int,
        default=0,
        help="After the benchmark, trace the lowest-cosine evaluation images",
    )
    parser.add_argument(
        "--trace-image-ids",
        default="",
        help="Comma-separated evaluation sample ids to trace explicitly",
    )
    parser.add_argument(
        "--trace-output",
        default="",
        help="Optional path for trace diagnostics JSON",
    )
    parser.add_argument(
        "--fold-cls-pos-embed",
        action="store_true",
        help="Experiment: fold CLS position embedding on the host like patch rows",
    )
    parser.add_argument(
        "--replay-early-attn",
        action="store_true",
        help="Replay traced early attention blocks to attribute softmax/value/output-scale error",
    )
    parser.add_argument(
        "--replay-blocks",
        default="0,1",
        help="Comma-separated block indices used by --replay-early-attn",
    )
    parser.add_argument(
        "--replay-late-attn",
        action="store_true",
        help="Replay traced late attention blocks for per-head downstream attribution",
    )
    parser.add_argument(
        "--replay-late-mlp",
        action="store_true",
        help="Replay traced late MLP blocks through GELU, FC2, and residual2",
    )
    parser.add_argument(
        "--replay-attn-blocks",
        default="9,10,11",
        help="Comma-separated block indices used by --replay-late-attn",
    )
    parser.add_argument(
        "--replay-mlp-blocks",
        default="9,10,11",
        help="Comma-separated block indices used by --replay-late-mlp",
    )
    args = parser.parse_args()
    explicit_overrides = explicit_cli_dest_overrides()
    preset = apply_diagnostic_preset(args, explicit_overrides=explicit_overrides)

    if args.benchmark_dataset == "cats_dogs_local" and "local_benchmark_image_dir" not in explicit_overrides:
        args.local_benchmark_image_dir = args.cats_dogs_image_dir

    if args.replay_early_attn and args.trace_worst_k <= 0:
        parser.error("--replay-early-attn requires --trace-worst-k > 0")
    if args.replay_late_attn and args.trace_worst_k <= 0:
        parser.error("--replay-late-attn requires --trace-worst-k > 0")
    if args.replay_late_mlp and args.trace_worst_k <= 0:
        parser.error("--replay-late-mlp requires --trace-worst-k > 0")

    try:
        requant_pc_qkv_selection = build_requant_pc_qkv_selection(
            blocks_text=args.requant_pc_qkv_blocks,
            heads_text=args.requant_pc_qkv_heads,
            projections_text=args.requant_pc_qkv_projections,
            exclude_text=args.requant_pc_qkv_exclude,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.requant_pc_qkv and requant_pc_qkv_selection is not None and not requant_pc_qkv_selection:
        parser.error("Q/K/V REQUANT_PC selection is empty after applying filters/exclusions")
    requant_pc_fc1_blocks = parse_csv_int_set(args.requant_pc_fc1_blocks)
    if requant_pc_fc1_blocks is not None and not args.requant_pc_fc1:
        parser.error("--requant-pc-fc1-blocks requires --requant-pc-fc1")
    if requant_pc_fc1_blocks is not None:
        invalid_requant_pc_fc1_blocks = sorted(
            block_idx for block_idx in requant_pc_fc1_blocks if not (0 <= block_idx < DEPTH)
        )
        if invalid_requant_pc_fc1_blocks:
            parser.error(f"--requant-pc-fc1-blocks out of range: {invalid_requant_pc_fc1_blocks}")
    requant_pc_fc2_blocks = parse_csv_int_set(args.requant_pc_fc2_blocks)
    if requant_pc_fc2_blocks is not None and not args.requant_pc_fc2:
        parser.error("--requant-pc-fc2-blocks requires --requant-pc-fc2")
    if requant_pc_fc2_blocks is not None:
        invalid_requant_pc_fc2_blocks = sorted(
            block_idx for block_idx in requant_pc_fc2_blocks if not (0 <= block_idx < DEPTH)
        )
        if invalid_requant_pc_fc2_blocks:
            parser.error(f"--requant-pc-fc2-blocks out of range: {invalid_requant_pc_fc2_blocks}")
    requant_pc_out_proj_blocks = parse_csv_int_set(args.requant_pc_out_proj_blocks)
    if requant_pc_out_proj_blocks is not None and not args.requant_pc_out_proj:
        parser.error("--requant-pc-out-proj-blocks requires --requant-pc-out-proj")
    if requant_pc_out_proj_blocks is not None:
        invalid_requant_pc_out_proj_blocks = sorted(
            block_idx for block_idx in requant_pc_out_proj_blocks if not (0 <= block_idx < DEPTH)
        )
        if invalid_requant_pc_out_proj_blocks:
            parser.error(
                f"--requant-pc-out-proj-blocks out of range: {invalid_requant_pc_out_proj_blocks}"
            )
    output_aware_clipping_fc1_blocks = parse_csv_int_set(args.output_aware_clipping_fc1_blocks)
    if output_aware_clipping_fc1_blocks is not None:
        invalid_output_aware_fc1_blocks = sorted(
            block_idx for block_idx in output_aware_clipping_fc1_blocks if not (0 <= block_idx < DEPTH)
        )
        if invalid_output_aware_fc1_blocks:
            parser.error(f"--output-aware-clipping-fc1-blocks out of range: {invalid_output_aware_fc1_blocks}")
    output_aware_clipping_fc2_blocks = parse_csv_int_set(args.output_aware_clipping_fc2_blocks)
    if output_aware_clipping_fc2_blocks is not None:
        invalid_output_aware_fc2_blocks = sorted(
            block_idx for block_idx in output_aware_clipping_fc2_blocks if not (0 <= block_idx < DEPTH)
        )
        if invalid_output_aware_fc2_blocks:
            parser.error(f"--output-aware-clipping-fc2-blocks out of range: {invalid_output_aware_fc2_blocks}")
    try:
        activation_percentile_nodes = parse_activation_percentile_overrides(args.act_percentile_nodes)
        resolve_activation_percentile_targets(activation_percentile_nodes)
    except ValueError as exc:
        parser.error(str(exc))
    adaround_fc1_blocks = parse_csv_int_set(args.adaround_fc1_blocks)
    if adaround_fc1_blocks is not None:
        invalid_adaround_fc1_blocks = sorted(
            block_idx for block_idx in adaround_fc1_blocks if not (0 <= block_idx < DEPTH)
        )
        if invalid_adaround_fc1_blocks:
            parser.error(f"--adaround-fc1-blocks out of range: {invalid_adaround_fc1_blocks}")
    adaround_fc2_blocks = parse_csv_int_set(args.adaround_fc2_blocks)
    if adaround_fc2_blocks is not None:
        invalid_adaround_fc2_blocks = sorted(
            block_idx for block_idx in adaround_fc2_blocks if not (0 <= block_idx < DEPTH)
        )
        if invalid_adaround_fc2_blocks:
            parser.error(f"--adaround-fc2-blocks out of range: {invalid_adaround_fc2_blocks}")
    if args.output_aware_clipping_candidates < 1:
        parser.error("--output-aware-clipping-candidates must be >= 1")
    if not (0.0 < args.output_aware_clipping_alpha_min <= 1.0):
        parser.error("--output-aware-clipping-alpha-min must be in (0, 1]")

    width = 72
    def header(title):
        print("\n" + "=" * width)
        print(f"  {title}")
        print("=" * width)

    # Load model
    header("Loading DeiT-tiny model")
    model, state_dict = load_model()
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    id2label = model.config.id2label
    print(f"  Loaded {sum(p.numel() for p in model.parameters()):,} parameters")

    trace_image_tokens = parse_csv_token_list(args.trace_image_ids)
    gelu_from_accum_blocks = parse_csv_int_set(args.gelu_from_accum_blocks)
    if gelu_from_accum_blocks is not None:
        invalid_gelu_from_accum_blocks = sorted(
            block_idx for block_idx in gelu_from_accum_blocks if not (0 <= block_idx < DEPTH)
        )
        if invalid_gelu_from_accum_blocks:
            parser.error(f"--gelu-from-accum-blocks out of range: {invalid_gelu_from_accum_blocks}")
    dequant_add_residual1_blocks = parse_csv_int_set(args.dequant_add_residual1_blocks)
    if dequant_add_residual1_blocks is not None:
        invalid_dequant_add_blocks = sorted(
            block_idx for block_idx in dequant_add_residual1_blocks if not (0 <= block_idx < DEPTH)
        )
        if invalid_dequant_add_blocks:
            parser.error(f"--dequant-add-residual1-blocks out of range: {invalid_dequant_add_blocks}")
    if not (0.0 <= args.dequant_add_residual1_scale_alpha <= 1.0):
        parser.error("--dequant-add-residual1-scale-alpha must be in [0, 1]")
    if not (0.0 < args.dequant_add_residual1_scale_percentile <= 100.0):
        parser.error("--dequant-add-residual1-scale-percentile must be in (0, 100]")
    if args.fused_softmax_attnv and args.fused_softmax_attnv_blocks:
        parser.error("--fused-softmax-attnv cannot be combined with --fused-softmax-attnv-blocks")
    fused_softmax_attnv_blocks = (
        set(range(DEPTH))
        if args.fused_softmax_attnv
        else parse_csv_int_set(args.fused_softmax_attnv_blocks)
    )
    if fused_softmax_attnv_blocks is not None:
        invalid_fused_softmax_blocks = sorted(
            block_idx for block_idx in fused_softmax_attnv_blocks if not (0 <= block_idx < DEPTH)
        )
        if invalid_fused_softmax_blocks:
            parser.error(f"--fused-softmax-attnv-blocks out of range: {invalid_fused_softmax_blocks}")
    if args.fused_softmax_attnv_accum_out_proj and fused_softmax_attnv_blocks is None:
        parser.error("--fused-softmax-attnv-accum-out-proj requires fused softmax->attn@V blocks")
    twin_uniform_softmax_blocks = parse_csv_int_set(args.twin_uniform_softmax_blocks)
    if twin_uniform_softmax_blocks is not None:
        invalid_twin_softmax_blocks = sorted(
            block_idx for block_idx in twin_uniform_softmax_blocks if not (0 <= block_idx < DEPTH)
        )
        if invalid_twin_softmax_blocks:
            parser.error(f"--twin-uniform-softmax-blocks out of range: {invalid_twin_softmax_blocks}")
    twin_uniform_gelu_blocks = parse_csv_int_set(args.twin_uniform_gelu_blocks)
    if twin_uniform_gelu_blocks is not None:
        invalid_twin_gelu_blocks = sorted(
            block_idx for block_idx in twin_uniform_gelu_blocks if not (0 <= block_idx < DEPTH)
        )
        if invalid_twin_gelu_blocks:
            parser.error(f"--twin-uniform-gelu-blocks out of range: {invalid_twin_gelu_blocks}")
    smoothquant_blocks = parse_csv_int_set(args.smoothquant_blocks)
    if smoothquant_blocks is not None:
        invalid_smoothquant_blocks = sorted(
            block_idx for block_idx in smoothquant_blocks if not (0 <= block_idx < DEPTH)
        )
        if invalid_smoothquant_blocks:
            parser.error(f"--smoothquant-blocks out of range: {invalid_smoothquant_blocks}")

    if preset is not None:
        eval_image_ids = list(preset["benchmark"]["eval_image_ids"])
        calibration_image_ids = list(preset["benchmark"]["calibration_image_ids"])
    else:
        if args.benchmark_dataset == "frozen_coco":
            if args.max_images > len(LOCAL_FROZEN_EVAL_IMAGE_IDS):
                parser.error(
                    f"--max-images must be <= {len(LOCAL_FROZEN_EVAL_IMAGE_IDS)} for the frozen benchmark"
                )
            if args.calibration_images > len(LOCAL_FROZEN_CALIBRATION_IMAGE_IDS):
                parser.error(
                    "--calibration-images must be <= "
                    f"{len(LOCAL_FROZEN_CALIBRATION_IMAGE_IDS)} for the frozen benchmark"
                )
            eval_image_ids = list(LOCAL_FROZEN_EVAL_IMAGE_IDS[:args.max_images])
            if args.calibration_images <= 0:
                calibration_image_ids = list(LOCAL_FROZEN_CALIBRATION_IMAGE_IDS)
            else:
                calibration_image_ids = list(LOCAL_FROZEN_CALIBRATION_IMAGE_IDS[:args.calibration_images])
        elif args.benchmark_dataset == "cats_dogs_local":
            available_ids = list(_discover_cats_dogs_sample_ids(args.local_benchmark_image_dir))
            if not available_ids:
                parser.error(
                    f"No local cats/dogs images found in {args.local_benchmark_image_dir}"
                )
            if "max_images" not in explicit_overrides:
                args.max_images = len(available_ids)
            if args.max_images > len(available_ids):
                parser.error(
                    f"--max-images must be <= {len(available_ids)} for the local cats/dogs dataset"
                )
            if args.calibration_images > len(available_ids):
                parser.error(
                    "--calibration-images must be <= "
                    f"{len(available_ids)} for the local cats/dogs dataset"
                )
            eval_image_ids = list(available_ids[:args.max_images])
            if args.calibration_images <= 0:
                calibration_image_ids = list(available_ids)
            else:
                calibration_image_ids = list(available_ids[:args.calibration_images])
        else:
            available_ids = list(_discover_local_flat_sample_ids(args.local_benchmark_image_dir))
            if not available_ids:
                parser.error(
                    f"No local benchmark images found in {args.local_benchmark_image_dir}"
                )
            if "max_images" not in explicit_overrides:
                args.max_images = len(available_ids)
            if args.max_images > len(available_ids):
                parser.error(
                    f"--max-images must be <= {len(available_ids)} for the local flat dataset"
                )
            if args.calibration_images > len(available_ids):
                parser.error(
                    "--calibration-images must be <= "
                    f"{len(available_ids)} for the local flat dataset"
                )
            eval_image_ids = list(available_ids[:args.max_images])
            if args.calibration_images <= 0:
                calibration_image_ids = list(available_ids)
            else:
                calibration_image_ids = list(available_ids[:args.calibration_images])

    resolved_trace_ids, invalid_trace_ids = resolve_explicit_sample_ids(trace_image_tokens, eval_image_ids)
    if invalid_trace_ids:
        parser.error(
            "--trace-image-ids must be a subset of the selected evaluation sample ids; "
            f"invalid ids: {invalid_trace_ids}"
        )

    if args.populate_local_benchmark_cache and args.benchmark_dataset != "frozen_coco":
        parser.error("--populate-local-benchmark-cache is only supported for the frozen COCO benchmark")

    if args.populate_local_benchmark_cache:
        cache_ids = sorted(set(eval_image_ids).union(calibration_image_ids))
        header(f"Populating local frozen benchmark cache ({len(cache_ids)} images)")
        populate_local_image_cache(
            cache_ids,
            "benchmark",
            image_root=args.local_benchmark_image_dir,
        )

    if args.benchmark_dataset in {"cats_dogs_local", "local_flat"} and args.benchmark_image_source != "local":
        parser.error("--benchmark-image-source=download is not supported for local flat-folder datasets")

    if args.benchmark_dataset == "frozen_coco":
        image_loader = load_local_images if args.benchmark_image_source == "local" else collect_images
        header(f"Collecting {len(eval_image_ids)} frozen evaluation images")
        eval_pairs = image_loader(
            eval_image_ids,
            "evaluation",
            image_root=args.local_benchmark_image_dir,
        ) if image_loader is load_local_images else image_loader(eval_image_ids, "evaluation")
        images = [{"sample_id": img_id, "dataset_label": None, "image_path": None, "image": img}
                  for img_id, img in eval_pairs]
        print(f"  Got {len(images)} evaluation images")

        header(f"Collecting {len(calibration_image_ids)} frozen calibration images")
        calibration_pairs = image_loader(
            calibration_image_ids,
            "calibration",
            image_root=args.local_benchmark_image_dir,
        ) if image_loader is load_local_images else image_loader(calibration_image_ids, "calibration")
        calibration_records = [{"sample_id": img_id, "dataset_label": None, "image_path": None, "image": img}
                               for img_id, img in calibration_pairs]
    elif args.benchmark_dataset == "cats_dogs_local":
        header(f"Collecting {len(eval_image_ids)} local cats/dogs evaluation images")
        images = load_flat_local_images(
            eval_image_ids,
            "evaluation",
            image_root=args.local_benchmark_image_dir,
        )
        print(f"  Got {len(images)} evaluation images")

        header(f"Collecting {len(calibration_image_ids)} local cats/dogs calibration images")
        calibration_records = load_flat_local_images(
            calibration_image_ids,
            "calibration",
            image_root=args.local_benchmark_image_dir,
        )
    else:
        header(f"Collecting {len(eval_image_ids)} local flat evaluation images")
        images = load_flat_local_images(
            eval_image_ids,
            "evaluation",
            image_root=args.local_benchmark_image_dir,
        )
        print(f"  Got {len(images)} evaluation images")

        header(f"Collecting {len(calibration_image_ids)} local flat calibration images")
        calibration_records = load_flat_local_images(
            calibration_image_ids,
            "calibration",
            image_root=args.local_benchmark_image_dir,
        )

    calibration_samples = [sample["image"] for sample in calibration_records]
    print(f"  Got {len(calibration_samples)} calibration images")

    # Compile model (now using calibration from actual images)
    header("Compiling model to INT8 program")
    t0 = time.time()
    print(f"  Using calibration sample ids: {calibration_image_ids}")
    program, cal_scales = compile_model(
        model,
        state_dict,
        calibration_samples,
        processor,
        softmax_mode=args.softmax_calibration,
        softmax_percentile=args.softmax_percentile,
        softmax_min_prob=args.softmax_min_prob,
        softmax_max_prob=args.softmax_max_prob,
        final_logit_mode=args.final_logit_calibration,
        bias_correction=args.bias_correction,
        bias_correction_layers=args.bias_correction_layers,
        activation_percentile_nodes=activation_percentile_nodes,
        output_aware_clipping_fc1_blocks=output_aware_clipping_fc1_blocks,
        output_aware_clipping_fc2_blocks=output_aware_clipping_fc2_blocks,
        output_aware_clipping_classifier=args.output_aware_clipping_classifier,
        output_aware_clipping_candidates=args.output_aware_clipping_candidates,
        output_aware_clipping_alpha_min=args.output_aware_clipping_alpha_min,
        adaround_fc1_blocks=adaround_fc1_blocks,
        adaround_fc2_blocks=adaround_fc2_blocks,
        softmax_search_heads=(
            {
                tuple(int(part) for part in item.split(":", 1))
                for item in args.softmax_search_heads.split(",")
                if item.strip()
            }
            if args.softmax_search_heads else None
        ),
        softmax_search_objective=args.softmax_search_objective,
        attn_v_mode=args.attn_v_calibration,
        attn_v_percentile=args.attn_v_percentile,
        attn_v_safety_margin=args.attn_v_safety_margin,
        attn_v_search_blocks=(
            {int(part) for part in args.attn_v_search_blocks.split(",") if part.strip()}
            if args.attn_v_search_blocks else None
        ),
        attn_v_search_objective=args.attn_v_search_objective,
        gelu_output_mode=args.gelu_output_calibration,
        gelu_search_blocks=(
            {int(part) for part in args.gelu_search_blocks.split(",") if part.strip()}
            if args.gelu_search_blocks else None
        ),
        gelu_search_objective=args.gelu_search_objective,
        hessian_calibration_images=args.hessian_calibration_images,
        hessian_target_nodes=args.hessian_target_nodes,
        twin_uniform_softmax_blocks=twin_uniform_softmax_blocks,
        twin_uniform_gelu_blocks=twin_uniform_gelu_blocks,
        twin_uniform_mode=args.twin_uniform_mode,
        twin_uniform_disable_hessian=args.twin_uniform_disable_hessian,
        smoothquant_targets=args.smoothquant_targets,
        smoothquant_alpha=args.smoothquant_alpha,
        smoothquant_blocks=smoothquant_blocks,
        requant_pc_qkv=args.requant_pc_qkv,
        requant_pc_qkv_selection=requant_pc_qkv_selection,
        requant_pc_fc1=args.requant_pc_fc1,
        requant_pc_fc1_blocks=requant_pc_fc1_blocks,
        requant_pc_fc2=args.requant_pc_fc2,
        requant_pc_fc2_blocks=requant_pc_fc2_blocks,
        requant_pc_out_proj=args.requant_pc_out_proj,
        requant_pc_out_proj_blocks=requant_pc_out_proj_blocks,
        value_head_mode=args.value_head_calibration,
        per_head_qkv_calibration=args.per_head_qkv_calibration,
        gelu_from_accum=args.gelu_from_accum,
        gelu_from_accum_blocks=gelu_from_accum_blocks,
        dequant_add_residual1_blocks=dequant_add_residual1_blocks,
        dequant_add_residual1_scale_mode=args.dequant_add_residual1_scale_mode,
        dequant_add_residual1_scale_percentile=args.dequant_add_residual1_scale_percentile,
        dequant_add_residual1_scale_alpha=args.dequant_add_residual1_scale_alpha,
        fused_softmax_attnv_blocks=fused_softmax_attnv_blocks,
        fused_softmax_attnv_accum_out_proj=args.fused_softmax_attnv_accum_out_proj,
    )
    run_config = build_run_config(
        args,
        eval_image_ids=eval_image_ids,
        calibration_image_ids=calibration_image_ids,
        smoothquant_blocks=smoothquant_blocks,
        requant_pc_qkv_selection=requant_pc_qkv_selection,
    )
    embed_scale = cal_scales.get("pos_embed_add", 14.0 / 127.0)
    dt = time.time() - t0
    print(f"  {program.insn_count} instructions, {len(program.data):,} bytes data")
    print(f"  Compiled in {dt:.1f}s")

    # Run comparison
    header(f"Running FP32 vs Golden Model on {len(images)} images")
    top1_matches = 0
    top5_matches = 0
    results = []

    for idx, sample in enumerate(images, 1):
        sample_id = sample["sample_id"]
        img = sample["image"]
        dataset_label = sample.get("dataset_label")
        sample_banner = f"COCO id={sample_id}" if args.benchmark_dataset == "frozen_coco" else f"sample={sample_id}"
        if dataset_label:
            sample_banner += f" ({dataset_label})"
        print(f"\n  --- Image {idx}/{len(images)}: {sample_banner} ---")

        # FP32 reference
        logits_fp32 = fp32_inference(model, processor, img)
        fp32_top = np.argsort(logits_fp32)[::-1][:args.top_k]

        # Patch embed + golden model
        patches_int8, cls_int8, act_scale = patch_embed_int8(
            model,
            processor,
            img,
            embed_scale,
            fold_cls_pos_embed=args.fold_cls_pos_embed,
        )
        logits_golden, insn_count, cycles, _ = golden_inference(program, patches_int8, cls_int8=cls_int8)
        golden_top = np.argsort(logits_golden)[::-1][:args.top_k]

        # Compare
        top1_match = fp32_top[0] == golden_top[0]
        top5_set_fp32 = set(fp32_top[:5])
        top5_set_golden = set(golden_top[:5])
        top5_overlap = len(top5_set_fp32 & top5_set_golden) / 5.0

        if top1_match:
            top1_matches += 1
        top5_matches += top5_overlap

        # Normalize golden logits for cosine similarity
        # (golden logits are INT32 accumulator values, different scale from FP32)
        norm_fp32 = np.linalg.norm(logits_fp32)
        norm_golden = np.linalg.norm(logits_golden)
        if norm_fp32 > 0 and norm_golden > 0:
            cosine_sim = np.dot(logits_fp32, logits_golden) / (norm_fp32 * norm_golden)
        else:
            cosine_sim = 0.0

        # Top-1 margin is a useful stability signal for near-tie label flips.
        fp32_top1_logit = float(logits_fp32[fp32_top[0]])
        fp32_top2_logit = float(logits_fp32[fp32_top[1]]) if len(fp32_top) > 1 else fp32_top1_logit
        golden_top1_logit = float(logits_golden[golden_top[0]])
        golden_top2_logit = float(logits_golden[golden_top[1]]) if len(golden_top) > 1 else golden_top1_logit
        fp32_margin = fp32_top1_logit - fp32_top2_logit
        golden_margin = golden_top1_logit - golden_top2_logit
        margin_delta = golden_margin - fp32_margin

        # Rank correlation (Spearman-like on top predictions)
        fp32_ranks = np.argsort(np.argsort(logits_fp32)[::-1])
        golden_ranks = np.argsort(np.argsort(logits_golden)[::-1])

        sym = "MATCH" if top1_match else "MISMATCH"

        print(f"  FP32   top-1: class {fp32_top[0]:>4} = {id2label.get(int(fp32_top[0]), '?')}")
        print(f"  Golden top-1: class {golden_top[0]:>4} = {id2label.get(int(golden_top[0]), '?')}")
        print(f"  Result: {sym}  |  cosine_sim={cosine_sim:.4f}  |  top5_overlap={top5_overlap*100:.0f}%")
        print(f"  Golden: {insn_count} insns, {cycles:,} cycles")
        if args.diagnostics:
            print(
                "  Diag: "
                f"fp32_margin={fp32_margin:.3f} "
                f"golden_margin={golden_margin:.3f} "
                f"margin_delta={margin_delta:.3f}"
            )

        print(f"\n  {'Rank':<6} {'FP32':>6} {'FP32 label':<30} {'Golden':>6} {'Golden label':<30}")
        print(f"  {'-'*6} {'-'*6} {'-'*30} {'-'*6} {'-'*30}")
        for r in range(args.top_k):
            fp32_cls = int(fp32_top[r])
            golden_cls = int(golden_top[r])
            fp32_label = id2label.get(fp32_cls, str(fp32_cls))[:30]
            golden_label = id2label.get(golden_cls, str(golden_cls))[:30]
            match_marker = " <=" if fp32_cls == golden_cls else ""
            print(f"  {r+1:<6} {fp32_cls:>6} {fp32_label:<30} {golden_cls:>6} {golden_label:<30}{match_marker}")

        results.append({
            "sample_id": sample_id,
            "img_id": sample_id,
            "dataset_label": dataset_label,
            "top1_match": bool(top1_match),
            "top5_overlap": float(top5_overlap),
            "cosine_sim": float(cosine_sim),
            "fp32_top5": [int(x) for x in fp32_top[:5]],
            "golden_top5": [int(x) for x in golden_top[:5]],
            "cycles": int(cycles),
            "fp32_top1_logit": fp32_top1_logit,
            "golden_top1_logit": golden_top1_logit,
            "fp32_margin": fp32_margin,
            "golden_margin": golden_margin,
            "margin_delta": margin_delta,
        })

    # Summary
    header("Summary")
    n = len(results)
    summary = summarize_results(results)
    print(f"  Images tested     : {n}")
    print(f"  Benchmark dataset : {args.benchmark_dataset}")
    print(f"  Eval sample ids   : {eval_image_ids}")
    print(f"  Calib sample ids  : {calibration_image_ids}")
    print(f"  Top-1 agreement   : {top1_matches}/{n} ({summary['top1_agreement']*100:.1f}%)")
    print(f"  Top-5 overlap     : {summary['top5_overlap_avg']*100:.1f}% average")
    print(f"  Cosine similarity : {summary['cosine_sim_avg']:.4f} average")
    print(f"  Cosine p10 / min  : {summary['cosine_sim_p10']:.4f} / {summary['cosine_sim_min']:.4f}")
    print(f"  Avg sim cycles    : {summary['avg_cycles']:,.0f}")

    if top1_matches < n:
        print(f"\n  Mismatched images:")
        for r in results:
            if not r["top1_match"]:
                fp32_cls = r["fp32_top5"][0]
                golden_cls = r["golden_top5"][0]
                print(f"    id={str(r['sample_id']):<12}  fp32={fp32_cls} {id2label.get(fp32_cls, '?')}"
                      f"  ->  golden={golden_cls} {id2label.get(golden_cls, '?')}")

    trace_report = None
    trace_targets = select_trace_image_ids(results, explicit_ids=resolved_trace_ids, trace_worst_k=args.trace_worst_k)
    if trace_targets:
        trace_node_order = default_trace_node_order()
        traced = []
        image_map = {sample["sample_id"]: sample["image"] for sample in images}
        replay_blocks = (
            [int(part) for part in args.replay_blocks.split(",") if part.strip()]
            if args.replay_early_attn else []
        )
        replay_reports = []
        replay_attn_blocks = (
            [int(part) for part in args.replay_attn_blocks.split(",") if part.strip()]
            if args.replay_late_attn else []
        )
        replay_late_attn_reports = []
        replay_mlp_blocks = (
            [int(part) for part in args.replay_mlp_blocks.split(",") if part.strip()]
            if args.replay_late_mlp else []
        )
        replay_late_mlp_reports = []

        header(f"Tracing {len(trace_targets)} Images")
        first_drop_counts = {}
        aggregate = {}
        for img_id in trace_targets:
            img = image_map[img_id]
            trace_banner = f"COCO id={img_id}" if args.benchmark_dataset == "frozen_coco" else f"sample={img_id}"
            print(f"\n  --- Trace image {trace_banner} ---")
            _, fp32_traces = fp32_trace(model, processor, img)
            patches_int8, cls_int8, _ = patch_embed_int8(
                model,
                processor,
                img,
                embed_scale,
                fold_cls_pos_embed=args.fold_cls_pos_embed,
            )
            _, _, _, golden_trace = golden_inference(
                program,
                patches_int8,
                cls_int8=cls_int8,
                trace_nodes=trace_node_order,
            )
            node_metrics = compare_trace_tensors(fp32_traces, golden_trace, trace_node_order)
            first_drop = first_major_trace_drop(node_metrics)
            first_drop_node = first_drop["node"] if first_drop else None
            if first_drop_node:
                first_drop_counts[first_drop_node] = first_drop_counts.get(first_drop_node, 0) + 1

            for metric in node_metrics:
                bucket = aggregate.setdefault(metric["node"], {
                    "cosine": [],
                    "delta": [],
                    "saturation": [],
                    "zero_fraction": [],
                    "max_abs": [],
                    "qdq_cosine": [],
                })
                bucket["cosine"].append(metric["cosine_sim"])
                bucket["saturation"].append(metric["saturation_rate"])
                bucket["zero_fraction"].append(metric["zero_fraction"])
                bucket["max_abs"].append(metric["max_abs_error"])
                bucket["qdq_cosine"].append(metric["qdq_cosine_sim"])
                if metric["delta_from_prev"] is not None:
                    bucket["delta"].append(metric["delta_from_prev"])

            print(
                "  First major drop: "
                + (
                    f"{first_drop_node} (cos={first_drop['cosine_sim']:.4f}, delta={first_drop['delta_from_prev']:.4f})"
                    if first_drop
                    else "none"
                )
            )
            top_drops = sorted(
                [m for m in node_metrics if m["delta_from_prev"] is not None],
                key=lambda item: item["delta_from_prev"],
            )[:8]
            for metric in top_drops:
                print(
                    f"    {metric['node']:<24} cos={metric['cosine_sim']:.4f} "
                    f"delta={metric['delta_from_prev']:.4f} "
                    f"qdq={metric['qdq_cosine_sim']:.4f} "
                    f"step={metric['quant_step']:.5f} "
                    f"zero={metric['zero_fraction']*100:.2f}% "
                    f"sat={metric['saturation_rate']*100:.2f}% "
                    f"max_abs={metric['max_abs_error']:.4f}"
                )

            early_replay = None
            if args.replay_early_attn:
                early_replay = replay_early_attention(
                    model,
                    fp32_traces,
                    golden_trace,
                    cal_scales,
                    block_indices=replay_blocks,
                )
                replay_reports.append(early_replay)
                print("  Early-attention replay:")
                for block in early_replay["blocks"]:
                    worst_head = next(
                        head for head in block["heads"] if head["head_idx"] == block["worst_head_idx"]
                    )
                    golden_head = worst_head.get("golden_attn_v_metrics") or {}
                    block0 = block["block_variants"]
                    print(
                        f"    block{block['block_idx']} worst_head=h{block['worst_head_idx']} "
                        f"golden_attn_v={golden_head.get('cosine_sim', 0.0):.4f} "
                        f"softmax_qdq={worst_head['variants']['qdq_softmax']['attn_v_qdq_metrics']['cosine_sim']:.4f} "
                        f"value_qdq={worst_head['variants']['qdq_value']['attn_v_qdq_metrics']['cosine_sim']:.4f} "
                        f"both_qdq={worst_head['variants']['qdq_softmax_value']['attn_v_qdq_metrics']['cosine_sim']:.4f}"
                    )
                    golden_out = block.get("golden_block_metrics", {}).get("out_proj_metrics", {})
                    print(
                        f"      out_proj golden={golden_out.get('cosine_sim', 0.0):.4f} "
                        f"softmax_qdq={block0['qdq_softmax']['out_proj_qdq_metrics']['cosine_sim']:.4f} "
                        f"value_qdq={block0['qdq_value']['out_proj_qdq_metrics']['cosine_sim']:.4f} "
                        f"both_qdq={block0['qdq_softmax_value']['out_proj_qdq_metrics']['cosine_sim']:.4f}"
                    )

            late_attn_replay = None
            if args.replay_late_attn:
                late_attn_replay = replay_late_attention(
                    model,
                    fp32_traces,
                    golden_trace,
                    cal_scales,
                    block_indices=replay_attn_blocks,
                )
                replay_late_attn_reports.append(late_attn_replay)
                print("  Late-attention replay:")
                for block in late_attn_replay["blocks"]:
                    top_head = min(
                        block["heads"],
                        key=lambda item: (
                            item["golden_attn_v_metrics"]["cosine_sim"]
                            if item["golden_attn_v_metrics"] is not None else 1.0
                        ),
                    )
                    golden_out = block.get("golden_block_metrics", {}).get("out_proj_metrics", {})
                    golden_head = top_head.get("golden_attn_v_metrics") or {}
                    isolated = top_head["isolated_block_variants"]
                    print(
                        f"    block{block['block_idx']} weakest_head=h{top_head['head_idx']} "
                        f"golden_attn_v={golden_head.get('cosine_sim', 0.0):.4f} "
                        f"golden_out={golden_out.get('cosine_sim', 0.0):.4f} "
                        f"softmax_out={isolated['qdq_softmax']['out_proj_qdq_metrics']['cosine_sim']:.4f} "
                        f"value_out={isolated['qdq_value']['out_proj_qdq_metrics']['cosine_sim']:.4f} "
                        f"both_out={isolated['qdq_softmax_value']['out_proj_qdq_metrics']['cosine_sim']:.4f}"
                    )

            late_mlp_replay = None
            if args.replay_late_mlp:
                late_mlp_replay = replay_late_mlp(
                    model,
                    fp32_traces,
                    golden_trace,
                    cal_scales,
                    block_indices=replay_mlp_blocks,
                )
                replay_late_mlp_reports.append(late_mlp_replay)
                print("  Late-MLP replay:")
                for block in late_mlp_replay["blocks"]:
                    golden_res = block.get("golden_metrics", {}).get("residual2_metrics", {})
                    print(
                        f"    block{block['block_idx']} golden_residual2={golden_res.get('cosine_sim', 0.0):.4f} "
                        f"fc1_only={block['variants']['qdq_fc1']['residual2_metrics']['cosine_sim']:.4f} "
                        f"gelu_only={block['variants']['qdq_gelu_out']['residual2_metrics']['cosine_sim']:.4f} "
                        f"both={block['variants']['qdq_fc1_gelu_out']['residual2_metrics']['cosine_sim']:.4f}"
                    )

            traced.append({
                "sample_id": img_id,
                "img_id": img_id,
                "first_major_drop": first_drop,
                "node_metrics": node_metrics,
                "early_attention_replay": early_replay,
                "late_attention_replay": late_attn_replay,
                "late_mlp_replay": late_mlp_replay,
            })

        aggregate_nodes = []
        for node_name, bucket in aggregate.items():
            aggregate_nodes.append({
                "node": node_name,
                "mean_cosine": float(np.mean(bucket["cosine"])),
                "min_cosine": float(np.min(bucket["cosine"])),
                "mean_delta_from_prev": float(np.mean(bucket["delta"])) if bucket["delta"] else 0.0,
                "mean_qdq_cosine": float(np.mean(bucket["qdq_cosine"])),
                "mean_saturation_rate": float(np.mean(bucket["saturation"])),
                "mean_zero_fraction": float(np.mean(bucket["zero_fraction"])),
                "mean_max_abs_error": float(np.mean(bucket["max_abs"])),
                "first_major_drop_count": int(first_drop_counts.get(node_name, 0)),
            })
        aggregate_nodes.sort(key=lambda item: (item["first_major_drop_count"], -item["mean_delta_from_prev"]), reverse=True)

        print("\n  Trace node ranking:")
        for item in aggregate_nodes[:10]:
            print(
                f"    {item['node']:<24} first_drop={item['first_major_drop_count']} "
                f"mean_cos={item['mean_cosine']:.4f} "
                f"mean_qdq={item['mean_qdq_cosine']:.4f} "
                f"mean_delta={item['mean_delta_from_prev']:.4f} "
                f"zero={item['mean_zero_fraction']*100:.2f}% "
                f"sat={item['mean_saturation_rate']*100:.2f}%"
            )

        replay_aggregate = None
        if args.replay_early_attn:
            replay_aggregate = summarize_early_attention_replay(replay_reports)
            print("\n  Early-attention replay aggregate:")
            for item in replay_aggregate:
                print(
                    f"    block{item['block_idx']} golden_out={item['mean_golden_out_proj_cosine']:.4f} "
                    f"golden_worst_head={item['mean_golden_worst_head_attn_v_cosine']:.4f} "
                    f"softmax_out={item['variant_mean_out_proj_qdq_cosine'].get('qdq_softmax', 0.0):.4f} "
                    f"value_out={item['variant_mean_out_proj_qdq_cosine'].get('qdq_value', 0.0):.4f} "
                    f"both_out={item['variant_mean_out_proj_qdq_cosine'].get('qdq_softmax_value', 0.0):.4f}"
                )

        late_attention_aggregate = None
        if args.replay_late_attn:
            late_attention_aggregate = summarize_late_attention_replay(replay_late_attn_reports)
            print("\n  Late-attention replay aggregate:")
            for item in late_attention_aggregate:
                weakest = min(
                    item["per_head"],
                    key=lambda head: head["mean_golden_attn_v_cosine"],
                )
                print(
                    f"    block{item['block_idx']} golden_out={item['mean_golden_out_proj_cosine']:.4f} "
                    f"weakest_h={weakest['head_idx']} "
                    f"golden_attn_v={weakest['mean_golden_attn_v_cosine']:.4f} "
                    f"softmax_out={weakest['variant_mean_isolated_out_proj_qdq_cosine'].get('qdq_softmax', 0.0):.4f} "
                    f"value_out={weakest['variant_mean_isolated_out_proj_qdq_cosine'].get('qdq_value', 0.0):.4f} "
                    f"both_out={weakest['variant_mean_isolated_out_proj_qdq_cosine'].get('qdq_softmax_value', 0.0):.4f}"
                )

        late_mlp_aggregate = None
        if args.replay_late_mlp:
            late_mlp_aggregate = summarize_late_mlp_replay(replay_late_mlp_reports)
            print("\n  Late-MLP replay aggregate:")
            for item in late_mlp_aggregate:
                print(
                    f"    block{item['block_idx']} golden_residual2={item['mean_golden_residual2_cosine']:.4f} "
                    f"fc1_only={item['variant_mean_residual2_cosine'].get('qdq_fc1', 0.0):.4f} "
                    f"gelu_only={item['variant_mean_residual2_cosine'].get('qdq_gelu_out', 0.0):.4f} "
                    f"both={item['variant_mean_residual2_cosine'].get('qdq_fc1_gelu_out', 0.0):.4f}"
                )

        trace_report = {
            "traced_image_ids": trace_targets,
            "per_image": traced,
            "aggregate_nodes": aggregate_nodes,
            "early_attention_replay_aggregate": replay_aggregate,
            "late_attention_replay_aggregate": late_attention_aggregate,
            "late_mlp_replay_aggregate": late_mlp_aggregate,
        }

    # Save results
    out = args.output
    with open(out, "w") as f:
        json.dump({
            "diagnostic_preset": args.diagnostic_preset or None,
            "run_config": run_config,
            "program_manifest": program.compiler_manifest,
            "summary": summary,
            "eval_image_ids": eval_image_ids,
            "calibration_image_ids": calibration_image_ids,
            "per_image": results,
            "trace_report": trace_report,
        }, f, indent=2)
    print(f"\n  Results saved to {out}\n")

    if args.trace_output:
        with open(args.trace_output, "w") as f:
            json.dump(trace_report or {}, f, indent=2)
        print(f"  Trace diagnostics saved to {args.trace_output}\n")


if __name__ == "__main__":
    main()
