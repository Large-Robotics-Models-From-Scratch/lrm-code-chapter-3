"""Tests for the real Chapter 2 sample shipped with the package.

The shape/dtype tests are cheap (they only read PNGs). The end-to-end
test runs the assembled backbone on the real sample, so it is marked
integration like the rest of the backbone suite.
"""

import pytest
import torch

from ch03 import VLABackbone, load_sample
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
    # contract check (listing 3.7) runs on real data instead of
    # torch.rand. The raw 480x640 frames go in directly; the vision
    # module resizes internally.
    images, state, instruction = load_sample()
    backbone = VLABackbone().eval()
    tokens = backbone.tokenizer(
        [instruction], return_tensors="pt", padding=True
    )
    L = tokens.input_ids.shape[1]
    with torch.no_grad():
        emb, mask, pos = backbone.embed_inputs(
            images, tokens.input_ids, state, tokens.attention_mask
        )
        hidden = backbone.contextualize(emb, mask, pos)
    assert emb.shape == (1, IMAGE_TOKENS + L + 1, SMOLLM_WIDTH)
    assert hidden.shape == emb.shape
    assert hidden.dtype == torch.float32
    # The identical shapes hide the chapter's central point: the
    # vectors changed. Contextualization must actually do something.
    assert not torch.allclose(hidden, emb.to(hidden.dtype), atol=1e-2)
