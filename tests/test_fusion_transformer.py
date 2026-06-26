"""Unit tests for FusionTransformer (pure torch, no model download)."""

import torch

from ch03.fusion_transformer import FusionTransformer


def test_causal_mask_is_upper_triangular():
    mask = FusionTransformer.causal_mask(4, torch.device("cpu"))
    # True blocks attention; row i may see columns 0..i only.
    assert mask.dtype == torch.bool
    assert not mask[0, 0]  # token 0 sees itself
    assert mask[0, 1]  # token 0 cannot see the future
    assert not mask[3].any()  # last token sees everything


def test_output_shape_matches_input():
    fusion = FusionTransformer(hidden_dim=576, num_layers=6, num_heads=9)
    tokens = torch.rand(2, 50, 576)
    out = fusion(tokens)
    assert out.shape == tokens.shape


def test_output_attentions_shape():
    fusion = FusionTransformer(
        hidden_dim=576, num_layers=6, num_heads=9
    ).eval()
    tokens = torch.rand(2, 50, 576)
    _, attentions = fusion(tokens, output_attentions=True)
    assert len(attentions) == 6
    assert attentions[0].shape == (2, 9, 50, 50)


def test_gradients_flow():
    fusion = FusionTransformer()
    tokens = torch.rand(2, 50, 576)
    fusion(tokens).mean().backward()
    grad = fusion.blocks[0].attn.in_proj_weight.grad
    assert grad is not None and grad.abs().sum() > 0


def test_padding_mask_accepted():
    fusion = FusionTransformer().eval()
    tokens = torch.rand(2, 50, 576)
    mask = torch.zeros(2, 50, dtype=torch.bool)
    mask[:, -5:] = True  # pad the last five positions
    out = fusion(tokens, key_padding_mask=mask)
    assert out.shape == tokens.shape


def test_causal_mask_blocks_future_influence():
    """Behavioral causality: changing a future token must not change
    the outputs at earlier positions."""
    torch.manual_seed(0)
    fusion = FusionTransformer(num_layers=2, dropout=0.0).eval()
    tokens = torch.rand(1, 6, 576)
    perturbed = tokens.clone()
    perturbed[:, -1] = torch.rand(576)  # only the last (future) token
    with torch.no_grad():
        base = fusion(tokens)
        changed = fusion(perturbed)
    assert torch.allclose(base[:, :-1], changed[:, :-1], atol=1e-5)


def test_padding_mask_zeroes_pad_influence():
    """Behavioral masking: values at padded positions must not affect
    a later non-padded token's output."""
    torch.manual_seed(0)
    fusion = FusionTransformer(num_layers=2, dropout=0.0).eval()
    tokens = torch.rand(1, 10, 576)
    mask = torch.zeros(1, 10, dtype=torch.bool)
    mask[:, 5:7] = True  # positions 5-6 are padding (mid-sequence)
    perturbed = tokens.clone()
    perturbed[:, 5:7] = torch.rand(2, 576)
    with torch.no_grad():
        base = fusion(tokens, key_padding_mask=mask)
        changed = fusion(perturbed, key_padding_mask=mask)
    # the last token attends to all earlier keys but must ignore padded
    assert torch.allclose(base[:, -1], changed[:, -1], atol=1e-5)
