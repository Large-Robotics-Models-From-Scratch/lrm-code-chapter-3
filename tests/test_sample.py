"""Tests for the real Chapter 2 sample shipped with the package.

The shape/dtype tests are cheap (they only read PNGs). The end-to-end
test runs the assembled backbone on the real sample, so it is marked
integration like the rest of the backbone suite.
"""

import pytest
import torch

from ch03 import UnifiedEmbeddingBackbone, load_sample, preprocess_image
from ch03.vla_backbone import IMAGE_TOKENS, SMOLLM_WIDTH

INSTRUCTION = "pink lego brick into the transparent box"


def test_load_sample_shapes_and_dtypes():
    # Chapter 2's contract: two camera views at native 480x640 in
    # [0, 1] float32, and a 6-dim state. The dataset stores state as
    # float64; load_sample casts it, because the linear layers are
    # float32.
    images, state, instruction = load_sample()
    assert images.shape == (1, 2, 3, 480, 640)
    assert images.dtype == torch.float32
    assert 0.0 <= float(images.min()) and float(images.max()) <= 1.0
    assert state.shape == (1, 6)
    assert state.dtype == torch.float32
    assert instruction == INSTRUCTION


def test_load_sample_is_a_real_frame_pair():
    # Real camera views, not noise and not the same frame twice.
    images, _, _ = load_sample()
    up, side = images[0, 0], images[0, 1]
    assert not torch.equal(up, side)
    # The PNGs are lossless uint8, so every value is an exact k/255.
    quantized = images * 255
    assert torch.equal(quantized, quantized.round())


@pytest.mark.integration
def test_backbone_runs_on_the_real_sample():
    # The whole point of shipping the sample: the chapter's final
    # contract check runs on real data instead of torch.rand.
    images, state, instruction = load_sample()
    backbone = UnifiedEmbeddingBackbone().eval()
    text_ids = backbone.tokenize_instruction(instruction)
    sequence_ids = torch.tensor([backbone.build_sequence_ids(text_ids)])
    with torch.no_grad():
        hidden = backbone(preprocess_image(images[0]).unsqueeze(0),
                          sequence_ids, state)
    assert hidden.shape == (
        1,
        IMAGE_TOKENS + len(text_ids) + 1,
        SMOLLM_WIDTH,
    )
    assert hidden.dtype == torch.float32
