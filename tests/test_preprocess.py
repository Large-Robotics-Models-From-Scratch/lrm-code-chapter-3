"""Unit tests for preprocess_image (pure torch, no model download).

The parity tests at the bottom pin the resize itself: bicubic with
antialiasing, and one shared implementation between the standalone
``preprocess_image`` helper and ``VisionEncoder``'s internal resize.
"""

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import ch03.preprocess as preprocess_module
import ch03.vision_encoder as vision_encoder_module
from ch03.preprocess import SIGLIP_INPUT_SIZE, preprocess_image
from ch03.vision_encoder import VisionEncoder


def _interpolate(image, **kwargs):
    return F.interpolate(
        image, size=(SIGLIP_INPUT_SIZE, SIGLIP_INPUT_SIZE), **kwargs
    )


def _read(module) -> str:
    return Path(module.__file__).read_text(encoding="utf-8")


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


def test_resize_is_bicubic_and_antialiased(dummy_raw_image_batch):
    # the manuscript pins the resize (listing 3.1 annotation E), so
    # match the exact interpolation rather than merely the shape
    expected = _interpolate(
        dummy_raw_image_batch,
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    assert torch.equal(preprocess_image(dummy_raw_image_batch), expected)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "bilinear", "align_corners": False, "antialias": True},
        {"mode": "bicubic", "align_corners": False, "antialias": False},
        {"mode": "bicubic", "align_corners": True, "antialias": True},
    ],
)
def test_resize_differs_from_other_interpolations(
    dummy_raw_image_batch, kwargs
):
    # a silent revert to bilinear, or dropping antialias, must fail here
    other = _interpolate(dummy_raw_image_batch, **kwargs)
    assert not torch.equal(preprocess_image(dummy_raw_image_batch), other)


def test_vision_encoder_reuses_the_one_resize_implementation():
    # the encoder must call this helper, not repeat the interpolation
    assert vision_encoder_module.preprocess_image is preprocess_image
    encoder_source = _read(vision_encoder_module)
    assert "preprocess_image(images)" in encoder_source
    assert "interpolate" not in encoder_source
    # and the single implementation is the pinned one
    preprocess_source = _read(preprocess_module)
    assert preprocess_source.count("F.interpolate") == 1
    for setting in (
        'mode="bicubic"',
        "align_corners=False",
        "antialias=True",
    ):
        assert setting in preprocess_source


@pytest.mark.integration
def test_encoder_internal_resize_matches_preprocess_image():
    # feeding native frames and pre-resized frames must be identical:
    # the encoder's internal path IS preprocess_image
    encoder = VisionEncoder(hidden_dim=576).eval()
    native = torch.rand(1, 3, 480, 640)
    with torch.no_grad():
        from_native = encoder.siglip_features(native)
        from_resized = encoder.siglip_features(preprocess_image(native))
    assert torch.equal(from_native, from_resized)
