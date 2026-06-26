"""Language backbone: SmolLM2-135M with its native tokenizer.

The instruction ("pick up the red cube") is encoded by a small,
pre-trained decoder language model. SmolLM2-135M runs on a laptop and
ships a 49,152-token byte-pair tokenizer that we use unchanged.
Vocabulary expansion for action tokens is Chapter 4's first step, not
this chapter's concern.

SmolLM2's native hidden width is 576, the book's common width, so the
language stream needs no projection at all; its per-token hidden states
sit beside the image and state tokens as they come out of the model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

SMOLLM_MODEL = "HuggingFaceTB/SmolLM2-135M"
SMOLLM_WIDTH = 576
SMOLLM_VOCAB = 49152
MAX_INSTRUCTION_TOKENS = 64


class LanguageBackbone(nn.Module):
    """SmolLM2-135M language backbone at its native 576-dim width."""

    def __init__(
        self,
        model_name: str = SMOLLM_MODEL,
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            # SmolLM2 ships no pad token; reuse EOS so batches of
            # different-length instructions can be padded and masked.
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # SmolLM2 stays trainable (unlike the frozen vision encoder): a
        # small language model benefits from fine-tuning on instructions,
        # and chapter 4 trains it end-to-end with the action head.
        self.model = AutoModel.from_pretrained(model_name)
        # Keep HF internals consistent: the pad id must match the token
        # we padding with, or attention masking silently disagrees.
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        if self.model.config.hidden_size != SMOLLM_WIDTH:
            raise ValueError(
                f"SmolLM2 hidden_size is "
                f"{self.model.config.hidden_size}, expected "
                f"{SMOLLM_WIDTH}; the language stream is native, so "
                f"its width is fixed and not configurable."
            )

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
        """Encode instructions to ``[B, L, 576]`` hidden states and a mask.

        The per-token hidden states come out at the native 576, so no
        projection is applied. The returned attention mask (``[B, L]``,
        1 for real tokens, 0 for padding) lets downstream fusion ignore
        padded positions.
        """
        input_ids, attention_mask = self.tokenize(instructions)
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden = outputs.last_hidden_state
        return hidden, attention_mask
