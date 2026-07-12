"""Tests for UnifiedEmbeddingBackbone.

Marked integration (downloads SigLIP and SmolLM2).
"""

import pytest
import torch

from ch03 import UnifiedEmbeddingBackbone
from ch03.vla_backbone import IMAGE_TOKENS, SMOLLM_VOCAB


@pytest.fixture(scope="module")
def backbone():
    return UnifiedEmbeddingBackbone().eval()


def _two_camera_batch(batch_size: int) -> torch.Tensor:
    """[B, 2, 3, 224, 224] random frames in [0, 1] (two camera views)."""
    return torch.rand(batch_size, 2, 3, 224, 224)


@pytest.mark.integration
def test_ch4_contract_shape(backbone, dummy_state):
    # Hidden states are [B, 392 + L + 1, 576]; Chapter 4 reads from this.
    text_ids = [101, 102, 103, 104]  # stand-in for a tokenized instruction
    L = len(text_ids)
    sequence_ids = torch.tensor([backbone.build_sequence_ids(text_ids)] * 2)
    images = _two_camera_batch(2)
    with torch.no_grad():
        hidden = backbone(images, sequence_ids, dummy_state)
    assert hidden.shape == (2, IMAGE_TOKENS + L + 1, 576)


@pytest.mark.integration
def test_image_block_is_392(backbone):
    # Two cameras at 196 patches each give 392 image-placeholder rows.
    row = backbone.build_sequence_ids([5, 6, 7])
    assert row.count(backbone.image_id) == IMAGE_TOKENS == 392
    assert row.count(backbone.state_id) == 1


@pytest.mark.integration
def test_template_order_is_image_text_state(backbone):
    # Order must be [image (392), text (L), state (1)].
    text_ids = [5, 6, 7]
    row = backbone.build_sequence_ids(text_ids)
    assert row[:IMAGE_TOKENS] == [backbone.image_id] * IMAGE_TOKENS
    assert row[IMAGE_TOKENS : IMAGE_TOKENS + len(text_ids)] == text_ids
    assert row[-1] == backbone.state_id


@pytest.mark.integration
def test_no_vocab_expansion(backbone):
    # Chapter 3 never calls the Chapter 4 vocabulary expansion. The
    # underlying language model config keeps the native vocab size; the
    # two inert placeholder rows are spliced over, not real tokens.
    assert backbone.language_backbone.config.vocab_size == SMOLLM_VOCAB


@pytest.mark.integration
def test_longer_instruction_contract(backbone):
    # A genuinely longer instruction still gives 392 + L + 1.
    text_ids = list(range(100, 112))  # 12 text tokens
    L = len(text_ids)
    sequence_ids = torch.tensor([backbone.build_sequence_ids(text_ids)] * 2)
    images = _two_camera_batch(2)
    state = torch.rand(2, 6)
    with torch.no_grad():
        hidden = backbone(images, sequence_ids, state)
    assert hidden.shape == (2, IMAGE_TOKENS + L + 1, 576)


@pytest.mark.integration
def test_malformed_template_rejected(backbone):
    # masked_scatter would silently drop tokens on a malformed
    # template; forward must reject it instead.
    text_ids = [5, 6, 7]
    good = backbone.build_sequence_ids(text_ids)
    images = _two_camera_batch(1)
    state = torch.rand(1, 6)

    missing_image = torch.tensor([[good[0]] * 0 + good[1:]])
    with pytest.raises(ValueError, match="image placeholders"):
        backbone(images, missing_image, state)

    no_state = torch.tensor([good[:-1] + [5]])
    with pytest.raises(ValueError, match="state"):
        backbone(images, no_state, state)
