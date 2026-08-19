"""VLA backbone: vision, language, and state in one sequence.

This is the chapter's deliverable and the interface Chapter 4 builds
on. ``VLABackbone`` encodes each input stream to the language
backbone's width and concatenates the streams into one multimodal
sequence, which the pretrained SmolLM2 contextualizes. There is no
separate fusion transformer: the language backbone is the fuser.

The fixed observation-prefix order is

    [image (392), text/language (L), state (1)]

so the contract Chapter 4 reads is ``[B, 392 + L + 1, 576]``.

The backbone exposes two representation levels:

- ``embed_inputs`` stops before SmolLM2's Transformer layers and
  returns the multimodal input embeddings together with the attention
  mask and position ids that describe them.
- ``contextualize`` runs SmolLM2 over a prepared embedding sequence
  through the ``inputs_embeds`` interface and returns contextualized
  hidden states of the same shape.

``forward`` chains the two. Downstream heads that only read hidden
states call ``forward``; heads that extend the sequence with extra
positions before the Transformer (Chapter 4's parallel action head)
call ``embed_inputs`` and ``contextualize`` separately.

Only the language stream carries vocabulary ids. The image and state
streams enter as vectors straight from their encoders, so no
placeholder ids and no embedding-table changes are needed; SmolLM2's
tokenizer, embedding table, and ``config.vocab_size`` all stay native.

Padding: batched instructions of different lengths are padded by the
tokenizer (``pad_token`` reuses SmolLM2's end-of-text token, adding no
vocabulary entry). Because the state embedding sits after the padded
text block, position ids are derived from the attention mask so padded
slots never shift the state position's rotary index.

Do not rename ``VLABackbone`` or change the ``embed_inputs`` /
``contextualize`` / ``forward`` signatures: they are the contract
Chapter 4 reads from. ``input_ids`` here is Hugging Face's own
tokenizer output (text only, ``[B, L]``); the observation prefix is
assembled internally.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from ch03.state_encoder import StateEncoder
from ch03.vision_encoder import NUM_PATCHES, VisionEncoder

SMOLLM_MODEL = "HuggingFaceTB/SmolLM2-135M"
SMOLLM_VOCAB = 49152  # SmolLM2 native vocabulary
SMOLLM_WIDTH = 576
NUM_CAMERAS = 2
IMAGE_TOKENS = NUM_CAMERAS * NUM_PATCHES  # 392 = 2 x 196


class VLABackbone(nn.Module):
    """Encode vision, language, and state into one input sequence."""

    def __init__(self) -> None:
        super().__init__()
        # Listing 3.1's encoder, reused verbatim: frozen SigLIP, the
        # 768 to 576 projection, the resize to 224, and SigLIP's
        # [0,1] -> [-1,1] pixel normalization all live inside it, so
        # the backbone owns no second copy of any of them.
        self.vision_encoder = VisionEncoder(hidden_dim=SMOLLM_WIDTH)
        self.state_encoder = StateEncoder(6, SMOLLM_WIDTH)
        self.tokenizer = AutoTokenizer.from_pretrained(SMOLLM_MODEL)
        # SmolLM2 ships no dedicated padding token; reuse end-of-text.
        # This maps an existing id, so the vocabulary does not grow.
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.language_backbone = AutoModel.from_pretrained(SMOLLM_MODEL)

    def embed_inputs(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        state: torch.Tensor,
        text_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the multimodal input-embedding sequence.

        ``images`` is ``[B, 2, 3, H, W]`` in ``[0, 1]`` (any spatial
        size; the vision encoder resizes to 224); ``input_ids`` is the
        tokenizer's text ids ``[B, L]``; ``state`` is ``[B, 6]``;
        ``text_attention_mask`` marks real (1) vs padded (0) text
        positions and defaults to all-real.

        Returns ``(input_embeddings, attention_mask, position_ids)``,
        each laid out as ``[image (392), text (L), state (1)]`` along
        the sequence dimension, so downstream heads that extend the
        sequence have everything they need.
        """
        if images.dim() != 5 or images.shape[1] != NUM_CAMERAS:
            raise ValueError(
                f"images must be [B, {NUM_CAMERAS}, 3, H, W]; got "
                f"{tuple(images.shape)}."
            )
        embed_tokens = self.language_backbone.get_input_embeddings()
        device = embed_tokens.weight.device
        images = images.to(device)
        input_ids = input_ids.to(device)
        state = state.to(device)

        B = images.shape[0]
        image_embeddings = self.vision_encoder(
            images.flatten(0, 1)
        ).reshape(B, IMAGE_TOKENS, SMOLLM_WIDTH)  # overhead then side
        text_embeddings = embed_tokens(input_ids)  # [B, L, 576]
        state_embedding = self.state_encoder(state)  # [B, 1, 576]

        dtype = text_embeddings.dtype
        image_embeddings = image_embeddings.to(dtype=dtype)
        state_embedding = state_embedding.to(dtype=dtype)

        input_embeddings = torch.cat(
            [image_embeddings, text_embeddings, state_embedding],
            dim=1,
        )  # [B, 392 + L + 1, 576]

        if text_attention_mask is None:
            text_attention_mask = torch.ones_like(input_ids)
        else:
            text_attention_mask = text_attention_mask.to(device)
        image_mask = torch.ones(
            B,
            IMAGE_TOKENS,
            device=device,
            dtype=text_attention_mask.dtype,
        )
        state_mask = torch.ones(
            B, 1, device=device, dtype=text_attention_mask.dtype
        )
        attention_mask = torch.cat(
            [image_mask, text_attention_mask, state_mask], dim=1
        )  # [B, 392 + L + 1]

        # Padded text slots sit between the text block and the state
        # embedding, so positions must be counted over VALID slots
        # only: the state position's rotary index is 392 + L_valid for
        # every row, however much padding the batch carries. The
        # stock Llama path would instead number positions by physical
        # index, padding included.
        position_ids = attention_mask.long().cumsum(dim=1) - 1
        position_ids = position_ids.masked_fill(attention_mask == 0, 0)

        return input_embeddings, attention_mask, position_ids

    def contextualize(
        self,
        input_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run SmolLM2 over a prepared embedding sequence.

        Takes ``[B, N, 576]`` input embeddings (plus the mask and
        position ids that describe them) and returns contextualized
        hidden states of the same shape. Vocabulary ids play no role
        here: the backbone runs through ``inputs_embeds``.
        """
        outputs = self.language_backbone(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return outputs.last_hidden_state

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        state: torch.Tensor,
        text_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode, concatenate, and contextualize one observation.

        Returns ``[B, 392 + L + 1, 576]`` contextualized hidden
        states. They are the representation Chapter 4's action
        architectures build on: a head may read one or more of these
        positions, or it may extend the observation prefix through
        ``embed_inputs`` and ``contextualize`` instead.
        """
        embeddings, mask, position_ids = self.embed_inputs(
            images, input_ids, state, text_attention_mask
        )
        return self.contextualize(embeddings, mask, position_ids)
