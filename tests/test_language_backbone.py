"""Tests for LanguageBackbone. Marked integration (downloads SmolLM)."""

import pytest
import torch

from ch03.language_backbone import SMOLLM_VOCAB, LanguageBackbone


@pytest.fixture(scope="module")
def backbone():
    return LanguageBackbone(hidden_dim=512).eval()


@pytest.mark.integration
def test_native_vocab_size(backbone):
    # Catches a silent SmolLM revision and proves no vocab expansion
    # happened in Chapter 3 (that is Chapter 4's first step).
    assert backbone.tokenizer.vocab_size == SMOLLM_VOCAB
    assert backbone.model.config.vocab_size == SMOLLM_VOCAB
    # vocab_size is fixed at load; len() also counts added tokens, so this
    # catches an accidental add_tokens (vocab expansion is Chapter 4's job).
    assert len(backbone.tokenizer) == SMOLLM_VOCAB


@pytest.mark.integration
def test_tokenize_shapes(backbone, dummy_instructions):
    input_ids, mask = backbone.tokenize(dummy_instructions)
    assert input_ids.shape == mask.shape
    assert input_ids.shape[0] == 2


@pytest.mark.integration
def test_forward_projects_to_512(backbone, dummy_instructions):
    with torch.no_grad():
        hidden, mask = backbone(dummy_instructions)
    assert hidden.shape[0] == 2
    assert hidden.shape[-1] == 512
    assert hidden.shape[1] == mask.shape[1]


@pytest.mark.integration
def test_projection_is_trainable(backbone):
    assert backbone.project.weight.requires_grad
