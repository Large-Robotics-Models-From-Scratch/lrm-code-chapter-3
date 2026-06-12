"""Unit tests for attention rollout (pure torch, no model download)."""

import torch

from ch03.viz_attention import attention_rollout, patch_saliency


def _random_attention(num_layers, heads, n):
    # already row-normalized per head, like real softmax attention
    return [
        torch.softmax(torch.rand(1, heads, n, n), dim=-1)
        for _ in range(num_layers)
    ]


def test_rollout_rows_are_distributions():
    # composing row-stochastic layers must stay row-stochastic
    roll = attention_rollout(_random_attention(3, 4, 9))
    assert roll.shape == (1, 9, 9)
    assert torch.allclose(roll.sum(dim=-1), torch.ones(1, 9), atol=1e-5)


def test_rollout_handles_single_layer():
    roll = attention_rollout(_random_attention(1, 2, 5))
    assert roll.shape == (1, 5, 5)


def test_patch_saliency_grid_shape():
    rollout = torch.softmax(torch.rand(196, 196), dim=-1)
    saliency = patch_saliency(rollout)
    assert saliency.shape == (14, 14)


def test_discard_ratio_sharpens_without_crashing():
    roll = attention_rollout(_random_attention(2, 4, 16), discard_ratio=0.5)
    assert roll.shape == (1, 16, 16)
    assert torch.isfinite(roll).all()
