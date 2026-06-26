"""Tests for LanguageBackbone. Marked integration (downloads SmolLM2)."""

import pytest
import torch
import torch.nn as nn

import ch03.language_backbone as lb
from ch03.language_backbone import (
    SMOLLM_VOCAB,
    SMOLLM_WIDTH,
    LanguageBackbone,
)


class _FakeTokenizer:
    """Minimal stand-in for SmolLM2's tokenizer: no pad, has an EOS."""

    _EOS_ID = 2

    def __init__(self):
        self.eos_token = "</s>"
        self.eos_token_id = self._EOS_ID
        self.pad_token = None
        self.pad_token_id = None

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        # Mirror HF behavior: setting pad_token to EOS gives it the EOS id.
        if name == "pad_token" and value == getattr(
            self, "eos_token", None
        ):
            super().__setattr__("pad_token_id", self._EOS_ID)


class _FakeConfig:
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
        self.pad_token_id = None


class _FakeModel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.config = _FakeConfig(hidden_size)


def _patch_loaders(monkeypatch, hidden_size):
    monkeypatch.setattr(
        lb.AutoTokenizer,
        "from_pretrained",
        lambda *a, **k: _FakeTokenizer(),
    )
    monkeypatch.setattr(
        lb.AutoModel,
        "from_pretrained",
        lambda *a, **k: _FakeModel(hidden_size),
    )


def test_pad_fallback_and_width_contract(monkeypatch):
    # Unit test: no SmolLM2 download. Stubbed loaders prove the pad-token
    # fallback wires through to the model config and the 576 contract.
    _patch_loaders(monkeypatch, SMOLLM_WIDTH)
    backbone = LanguageBackbone()
    assert backbone.tokenizer.pad_token == backbone.tokenizer.eos_token
    assert backbone.tokenizer.pad_token_id is not None
    assert (
        backbone.model.config.pad_token_id
        == backbone.tokenizer.pad_token_id
    )


def test_width_contract_rejects_wrong_hidden_size(monkeypatch):
    # A model whose native width is not 576 must fail loudly (a raised
    # error, not an assert that -O strips).
    _patch_loaders(monkeypatch, 512)
    with pytest.raises(ValueError):
        LanguageBackbone()


@pytest.fixture(scope="module")
def backbone():
    return LanguageBackbone().eval()


@pytest.mark.integration
def test_native_vocab_size(backbone):
    # Catches a silent SmolLM2 revision and proves no vocab expansion
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
def test_forward_is_native_576(backbone, dummy_instructions):
    # SmolLM2's native width is the common width, so no projection is
    # applied: the per-token hidden states come out at 576 directly.
    with torch.no_grad():
        hidden, mask = backbone(dummy_instructions)
    assert hidden.shape[0] == 2
    assert hidden.shape[-1] == SMOLLM_WIDTH
    assert hidden.shape[1] == mask.shape[1]


@pytest.mark.integration
def test_model_stays_trainable(backbone):
    # The language backbone is trainable, unlike the frozen vision encoder.
    assert any(p.requires_grad for p in backbone.model.parameters())
