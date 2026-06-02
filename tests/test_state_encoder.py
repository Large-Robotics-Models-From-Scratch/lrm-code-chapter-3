"""Unit tests for StateEncoder (pure torch, no model download)."""

import torch
import torch.nn as nn

from ch03.state_encoder import STATE_DIM, StateEncoder


def test_default_state_dim_is_six():
    assert STATE_DIM == 6


def test_output_shape_is_one_token(dummy_state):
    encoder = StateEncoder(state_dim=6, hidden_dim=512)
    out = encoder(dummy_state)
    assert out.shape == (2, 1, 512)


def test_has_nonlinearity():
    encoder = StateEncoder()
    has_gelu = any(
        isinstance(m, nn.GELU) for m in encoder.modules()
    )
    assert has_gelu


def test_is_trainable():
    encoder = StateEncoder()
    assert all(p.requires_grad for p in encoder.parameters())


def test_gradients_flow():
    encoder = StateEncoder()
    out = encoder(torch.rand(2, 6))
    out.mean().backward()
    grad = encoder.mlp[0].weight.grad
    assert grad is not None and grad.abs().sum() > 0
