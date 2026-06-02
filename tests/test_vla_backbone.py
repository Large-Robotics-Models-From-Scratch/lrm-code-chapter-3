"""Tests for VLABackbone. Marked integration (downloads both models)."""

import pytest
import torch

from ch03 import VLABackbone


@pytest.fixture(scope="module")
def backbone():
    return VLABackbone(hidden_dim=512).eval()


@pytest.mark.integration
def test_ch4_contract_shape(
    backbone, dummy_raw_image_batch, dummy_instructions, dummy_state
):
    # Hidden states are [B, 196 + L + 1, 512]; Chapter 4 reads from this.
    # tokenize() gives L without a full SmolLM forward pass.
    _, lang_mask = backbone.language_backbone.tokenize(dummy_instructions)
    expected_len = 196 + lang_mask.shape[1] + 1
    with torch.no_grad():
        hidden = backbone(
            dummy_raw_image_batch, dummy_instructions, dummy_state
        )
    assert hidden.shape == (2, expected_len, 512)


@pytest.mark.integration
def test_no_vocab_expansion(backbone):
    # Chapter 3 leaves the tokenizer native; Chapter 4 expands it. Compare
    # the model embedding table to the tokenizer rather than hardcoding a
    # number, so a SmolLM revision surfaces as a clear mismatch.
    lang = backbone.language_backbone
    assert lang.model.config.vocab_size == lang.tokenizer.vocab_size
    assert len(lang.tokenizer) == lang.model.config.vocab_size


@pytest.mark.integration
def test_mixed_length_batch_contract(backbone):
    # genuinely different instruction lengths must still give 196 + L + 1
    instructions = [
        "go",
        "pick up the small red cube on the far left of the table",
    ]
    image = torch.rand(2, 3, 480, 640)
    state = torch.rand(2, 6)
    _, lang_mask = backbone.language_backbone.tokenize(instructions)
    expected_len = 196 + lang_mask.shape[1] + 1
    with torch.no_grad():
        hidden = backbone(image, instructions, state)
    assert hidden.shape == (2, expected_len, 512)


@pytest.mark.integration
def test_output_attentions_path(
    backbone, dummy_raw_image_batch, dummy_instructions, dummy_state
):
    with torch.no_grad():
        _, attentions = backbone(
            dummy_raw_image_batch,
            dummy_instructions,
            dummy_state,
            output_attentions=True,
        )
    assert len(attentions) == 6
