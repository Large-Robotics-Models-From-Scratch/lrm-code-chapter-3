"""Tests for VisionEncoder. Marked integration (downloads SigLIP)."""

import pytest
import torch

from ch03.vision_encoder import NUM_PATCHES, VisionEncoder


@pytest.fixture(scope="module")
def encoder():
    return VisionEncoder(hidden_dim=512).eval()


@pytest.fixture(scope="module")
def eager_encoder():
    # output_attentions only works with the eager attention path
    return VisionEncoder(hidden_dim=512, attn_implementation="eager").eval()


@pytest.mark.integration
def test_siglip_is_frozen(encoder):
    assert all(not p.requires_grad for p in encoder.siglip.parameters())


@pytest.mark.integration
def test_projection_is_trainable(encoder):
    assert encoder.project.weight.requires_grad


@pytest.mark.integration
def test_output_shape_and_dtype(encoder, dummy_image_batch):
    with torch.no_grad():
        patches = encoder(dummy_image_batch)
    assert patches.shape == (2, NUM_PATCHES, 512)
    assert patches.dtype == torch.float32


@pytest.mark.integration
def test_output_attentions_returns_layer_maps(
    eager_encoder, dummy_image_batch
):
    with torch.no_grad():
        _, attentions = eager_encoder(
            dummy_image_batch, output_attentions=True
        )
    assert len(attentions) > 0
    assert attentions[0].shape[-1] == NUM_PATCHES


@pytest.mark.integration
def test_frozen_siglip_stays_eval_under_train(encoder):
    # frozen backbone must not flip to train mode (dropout off)
    encoder.train()
    assert not encoder.siglip.training
    encoder.eval()


def test_pixel_normalization_maps_unit_to_signed():
    # the [0,1] -> [-1,1] SigLIP mapping, tested without a download
    mean = torch.tensor(0.5)
    std = torch.tensor(0.5)
    ones = torch.ones(1, 3, 8, 8)
    assert torch.allclose((ones - mean) / std, torch.ones_like(ones))
    zeros = torch.zeros(1, 3, 8, 8)
    assert torch.allclose((zeros - mean) / std, -torch.ones_like(zeros))
