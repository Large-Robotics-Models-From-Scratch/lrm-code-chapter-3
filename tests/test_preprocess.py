"""Unit tests for preprocess_image (pure torch, no model download)."""

import torch

from ch03.preprocess import SIGLIP_INPUT_SIZE, preprocess_image


def test_resizes_batched_native_frame(dummy_raw_image_batch):
    out = preprocess_image(dummy_raw_image_batch)
    assert out.shape == (2, 3, SIGLIP_INPUT_SIZE, SIGLIP_INPUT_SIZE)


def test_resizes_single_frame():
    out = preprocess_image(torch.rand(3, 480, 640))
    assert out.shape == (3, SIGLIP_INPUT_SIZE, SIGLIP_INPUT_SIZE)


def test_keeps_values_in_unit_range():
    out = preprocess_image(torch.rand(2, 3, 480, 640))
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_already_correct_size_is_idempotent_shape():
    img = torch.rand(2, 3, 224, 224)
    assert preprocess_image(img).shape == img.shape
