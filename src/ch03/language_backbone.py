"""Language backbone: SmolLM-135M with its native tokenizer.

The instruction ("pick up the red cube") is encoded by a small,
pre-trained decoder language model. SmolLM-135M is frozen-friendly,
runs on a laptop, and ships a 49,152-token byte-pair tokenizer that we
use unchanged. Vocabulary expansion for action tokens is Chapter 4's
first step, not this chapter's concern.

The model's 576-dim hidden states are projected to the book's common
512-dim width so the language tokens sit in the same space as the image
and state tokens.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

SMOLLM_MODEL = "HuggingFaceTB/SmolLM-135M"
SMOLLM_WIDTH = 576
SMOLLM_VOCAB = 49152
MAX_INSTRUCTION_TOKENS = 64


class LanguageBackbone(nn.Module):
    """SmolLM-135M text encoder with a 576 to 512 projection."""

    def __init__(
        self,
        model_name: str = SMOLLM_MODEL,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            # SmolLM ships no pad token; reuse EOS so batches of
            # different-length instructions can be padded and masked.
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # SmolLM stays trainable (unlike the frozen vision encoder): a
        # small language model benefits from fine-tuning on instructions,
        # and chapter 4 trains it end-to-end with the action head.
        self.model = AutoModel.from_pretrained(model_name)
        assert self.model.config.hidden_size == SMOLLM_WIDTH, (
            f"SmolLM hidden_size is {self.model.config.hidden_size}, "
            f"expected {SMOLLM_WIDTH}; update the projection width."
        )
        self.project = nn.Linear(SMOLLM_WIDTH, hidden_dim)

    def tokenize(
        self,
        instructions: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of strings to padded ``input_ids`` and a mask."""
        encoded = self.tokenizer(
            instructions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INSTRUCTION_TOKENS,
        )
        return encoded["input_ids"], encoded["attention_mask"]

    def forward(
        self,
        instructions: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode instructions to ``[B, L, 512]`` hidden states and a mask.

        The returned attention mask (``[B, L]``, 1 for real tokens, 0 for
        padding) lets the fusion transformer ignore padded positions.
        """
        input_ids, attention_mask = self.tokenize(instructions)
        device = self.project.weight.device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden = self.project(outputs.last_hidden_state)
        return hidden, attention_mask
