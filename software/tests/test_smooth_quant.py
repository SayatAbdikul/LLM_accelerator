"""Tests for SmoothQuant preprocessing helpers."""

import torch
from torch import nn

from taccel.compiler.graph_extract import DEPTH
from taccel.quantizer.smooth_quant import apply_smooth_quant, compute_smooth_factors


class _DummyAttentionProj(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)


class _DummyAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attention = _DummyAttentionProj(dim)


class _DummyIntermediate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dense = nn.Linear(dim, dim)


class _DummyBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.layernorm_before = nn.LayerNorm(dim)
        self.layernorm_after = nn.LayerNorm(dim)
        self.attention = _DummyAttention(dim)
        self.intermediate = _DummyIntermediate(dim)

    def forward(self, x):
        ln1 = self.layernorm_before(x)
        attn_mix = (
            self.attention.attention.query(ln1)
            + self.attention.attention.key(ln1)
            + self.attention.attention.value(ln1)
        ) / 3.0
        residual = x + attn_mix
        ln2 = self.layernorm_after(residual)
        fc1 = self.intermediate.dense(ln2)
        return residual + 0.1 * fc1


class _DummyEncoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.layer = nn.ModuleList([_DummyBlock(dim) for _ in range(DEPTH)])


class _DummyVit(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.encoder = _DummyEncoder(dim)


class _DummyModel(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.vit = _DummyVit(dim)

    def forward(self, pixel_values):
        x = pixel_values
        for block in self.vit.encoder.layer:
            x = block(x)
        return x


def _clone_state_dict(state_dict):
    return {name: tensor.detach().clone() for name, tensor in state_dict.items()}


class TestSmoothQuant:
    def test_compute_smooth_factors_returns_shared_qkv_and_fc1_specs(self):
        torch.manual_seed(0)
        model = _DummyModel(dim=8)
        sample_inputs = [{"pixel_values": torch.randn(2, 3, 8)} for _ in range(2)]

        factors = compute_smooth_factors(model, sample_inputs, alpha=0.5, targets="both")

        assert len(factors) == DEPTH * 2
        ln1_spec = factors["block0_ln1_qkv"]
        ln2_spec = factors["block0_ln2_fc1"]
        assert ln1_spec["smooth"].shape == torch.Size([8])
        assert torch.all(ln1_spec["smooth"] > 0)
        assert ln1_spec["linear_weights"] == [
            "vit.encoder.layer.0.attention.attention.query.weight",
            "vit.encoder.layer.0.attention.attention.key.weight",
            "vit.encoder.layer.0.attention.attention.value.weight",
        ]
        assert ln2_spec["linear_weights"] == ["vit.encoder.layer.0.intermediate.dense.weight"]

    def test_compute_smooth_factors_can_filter_specific_blocks(self):
        torch.manual_seed(0)
        model = _DummyModel(dim=8)
        sample_inputs = [{"pixel_values": torch.randn(2, 3, 8)} for _ in range(2)]

        factors = compute_smooth_factors(
            model,
            sample_inputs,
            alpha=0.5,
            targets="ln2_fc1",
            blocks={2, 3},
        )

        assert set(factors) == {"block2_ln2_fc1", "block3_ln2_fc1"}

    def test_apply_smooth_quant_preserves_fp32_outputs(self):
        torch.manual_seed(1)
        model = _DummyModel(dim=8)
        sample_inputs = [{"pixel_values": torch.randn(2, 4, 8)} for _ in range(3)]

        factors = compute_smooth_factors(model, sample_inputs, alpha=0.5, targets="both")
        smoothed_state = _clone_state_dict(model.state_dict())
        apply_smooth_quant(smoothed_state, factors)

        smoothed_model = _DummyModel(dim=8)
        smoothed_model.load_state_dict(smoothed_state)
        smoothed_model.eval()
        model.eval()

        with torch.no_grad():
            for inp in sample_inputs:
                original = model(**inp)
                smoothed = smoothed_model(**inp)
                assert torch.allclose(original, smoothed, atol=1e-5, rtol=1e-4)

    def test_apply_smooth_quant_does_not_mutate_original_state_dict_when_cloned(self):
        torch.manual_seed(2)
        model = _DummyModel(dim=8)
        sample_inputs = [{"pixel_values": torch.randn(2, 3, 8)}]
        factors = compute_smooth_factors(model, sample_inputs, alpha=0.6, targets="ln1_qkv")

        original_state = _clone_state_dict(model.state_dict())
        smoothed_state = _clone_state_dict(model.state_dict())
        apply_smooth_quant(smoothed_state, factors)

        for name, tensor in model.state_dict().items():
            assert torch.equal(tensor, original_state[name])

        changed = [
            name for name, tensor in smoothed_state.items()
            if not torch.equal(tensor, original_state[name])
        ]
        assert "vit.encoder.layer.0.layernorm_before.weight" in changed
        assert "vit.encoder.layer.0.attention.attention.query.weight" in changed

    def test_compute_smooth_factors_rejects_out_of_range_blocks(self):
        torch.manual_seed(0)
        model = _DummyModel(dim=8)
        sample_inputs = [{"pixel_values": torch.randn(2, 3, 8)}]

        try:
            compute_smooth_factors(
                model,
                sample_inputs,
                alpha=0.5,
                targets="ln2_fc1",
                blocks={DEPTH},
            )
        except ValueError as exc:
            assert "out of range" in str(exc)
        else:
            raise AssertionError("Expected out-of-range SmoothQuant blocks to fail")
