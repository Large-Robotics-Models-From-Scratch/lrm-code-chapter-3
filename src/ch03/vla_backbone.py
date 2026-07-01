"""VLA backbone: vision, language, and state fused in one sequence.

This is the chapter's deliverable and the frozen interface Chapter 4
builds on. ``UnifiedEmbeddingBackbone`` projects the two camera views
into the language backbone's own token stream and lets the pretrained
attention do the fusion. There is no separate fusion transformer: the
language backbone is the fuser.

The image and state tokens are spliced into reserved placeholder rows of
the backbone's input embeddings with ``masked_scatter``. The token order
the template builds is

    [image (392), text/language (L), state (1)]

so the contract Chapter 4 reads is ``[B, 392 + L + 1, 576]``.

Do not rename ``UnifiedEmbeddingBackbone`` or change the ``forward``
signature ``(images, sequence_ids, state)``: the ``[B, 392 + L + 1, 576]``
output is the contract Chapter 4 reads from. ``sequence_ids`` is OUR
fused layout (image + text + state); it is distinct from Hugging Face's
``input_ids``, which stay the tokenizer/model API name unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, SiglipVisionModel

from ch03.state_encoder import StateEncoder
from ch03.vision_encoder import NUM_PATCHES, SIGLIP_MODEL

SMOLLM_MODEL = "HuggingFaceTB/SmolLM2-135M"
SMOLLM_VOCAB = 49152  # SmolLM2 native vocabulary
SMOLLM_WIDTH = 576
SIGLIP_WIDTH = 768
NUM_CAMERAS = 2
IMAGE_TOKENS = NUM_CAMERAS * NUM_PATCHES  # 392 = 2 x 196

# Two placeholder ids, just past the native vocab, mark the rows the
# splice overwrites. Their embeddings are never read (masked_scatter
# replaces them before the backbone runs), so this is not the Chapter 4
# vocabulary expansion: it reserves two inert splice markers, not real
# tokens.
DEFAULT_IMAGE_ID = SMOLLM_VOCAB
DEFAULT_STATE_ID = SMOLLM_VOCAB + 1


class FrozenSiglipFeatures(nn.Module):
    """Frozen SigLIP that returns raw ``[B, 196, 768]`` patch features.

    Same frozen SigLIP as the ``VisionEncoder`` of section 3.2, but
    without the 768 to 576 projection: that projection lives once in the
    backbone's ``img_proj``, with the consumer, not here. SigLIP expects
    pixels in ``[-1, 1]``; the buffers keep that detail inside the module
    so callers keep passing ``[0, 1]`` frames.
    """

    def __init__(self, model_name: str = SIGLIP_MODEL) -> None:
        super().__init__()
        self.siglip = SiglipVisionModel.from_pretrained(model_name)
        self.siglip.eval()
        for param in self.siglip.parameters():
            param.requires_grad = False
        self.register_buffer("pixel_mean", torch.tensor(0.5))
        self.register_buffer("pixel_std", torch.tensor(0.5))

    def train(self, mode: bool = True) -> "FrozenSiglipFeatures":
        super().train(mode)
        self.siglip.eval()  # frozen: never toggle out of eval
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pixels = (images - self.pixel_mean) / self.pixel_std
        return self.siglip(pixel_values=pixels).last_hidden_state


def load_siglip(model_name: str = SIGLIP_MODEL) -> nn.Module:
    """Return the frozen SigLIP feature extractor (768-dim patches).

    Same frozen SigLIP as listing 3.1, but without the 768 to 576
    projection: it takes ``[0, 1]`` frames (normalization handled
    inside) and returns their ``[B, 196, 768]`` patch features. The
    projection lives once in the backbone's ``img_proj``, with the
    consumer, so this is the bare feature extractor it consumes.
    """
    return FrozenSiglipFeatures(model_name)


class UnifiedEmbeddingBackbone(nn.Module):
    """Fuse vision, language, and state in the backbone's own stream."""

    def __init__(
        self,
        image_id: int = DEFAULT_IMAGE_ID,
        state_id: int = DEFAULT_STATE_ID,
    ) -> None:
        super().__init__()
        # The placeholder ids must sit past the native vocab (so they are
        # reserved, not real tokens) and be distinct (so the two splices
        # never collide).
        if image_id < SMOLLM_VOCAB or state_id < SMOLLM_VOCAB:
            raise ValueError(
                f"placeholder ids must be reserved (>= {SMOLLM_VOCAB}); "
                f"got image_id={image_id}, state_id={state_id}."
            )
        if image_id == state_id:
            raise ValueError(
                f"image_id and state_id must differ; both are {image_id}."
            )
        self.image_id = image_id
        self.state_id = state_id
        self.vision_encoder = load_siglip()  # frozen, 768-dim
        self.img_proj = nn.Linear(SIGLIP_WIDTH, SMOLLM_WIDTH)  # 768 -> 576
        self.state_proj = StateEncoder(6, SMOLLM_WIDTH)
        self.tokenizer = AutoTokenizer.from_pretrained(SMOLLM_MODEL)
        self.language_backbone = AutoModel.from_pretrained(SMOLLM_MODEL)
        # Grow the input embedding table just far enough that both
        # placeholder ids index validly. These rows are inert: the splice
        # overwrites them before the backbone runs, so this is not the
        # Chapter 4 vocabulary expansion (no new vocabulary, no tokenizer
        # change). For the defaults this is SMOLLM_VOCAB + 2.
        self.embed_tokens = self._grow_embeddings(
            self.language_backbone.get_input_embeddings(),
            max(image_id, state_id) + 1,
        )

    @staticmethod
    def _grow_embeddings(
        embed: nn.Embedding, new_size: int
    ) -> nn.Embedding:
        """Copy an embedding table into a slightly larger one."""
        if embed.num_embeddings >= new_size:
            return embed
        grown = nn.Embedding(
            new_size, embed.embedding_dim, padding_idx=embed.padding_idx
        )
        with torch.no_grad():
            grown.weight[: embed.num_embeddings] = embed.weight
        return grown

    def tokenize_instruction(self, instruction: str) -> list[int]:
        """Return the L native SmolLM2 text ids for one instruction.

        HF's tokenizer hands back ``input_ids``; we pull the single
        row out as a plain ``list[int]`` so ``build_sequence_ids`` can
        splice the text between the image and state placeholders. No
        padding or truncation here: batching pads in ``forward``.
        """
        encoded = self.tokenizer(instruction, return_tensors="pt")
        return encoded["input_ids"][0].tolist()

    def build_sequence_ids(self, text_ids: list[int]) -> list[int]:
        """Template one row: image (392) + text (L) + state (1).

        The 392 image placeholder ids (196 for camera 0, then 196 for
        camera 1) come first, then the tokenized text, then the single
        state placeholder id, matching the ``392 + L + 1`` contract.
        """
        image_ids = [self.image_id] * IMAGE_TOKENS
        state_ids = [self.state_id]
        return image_ids + text_ids + state_ids

    def forward(
        self,
        images: torch.Tensor,
        sequence_ids: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Encode, splice, and run the backbone as an encoder.

        ``images`` is ``[B, 2, 3, 224, 224]`` in ``[0, 1]``;
        ``sequence_ids`` is ``[B, N]`` with 392 image-placeholder rows,
        L text rows, and 1 state-placeholder row per the template
        (N = 392 + L + 1); ``state`` is ``[B, 6]``. Returns
        ``[B, 392 + L + 1, 576]`` hidden states.
        """
        B = images.shape[0]
        siglip_features = self.vision_encoder(
            images.flatten(0, 1)
        )  # [B*2, 196, 768]
        img = self.img_proj(siglip_features).reshape(
            B, -1, SMOLLM_WIDTH
        )  # [B, 392, 576] cam0 then cam1
        emb = self.embed_tokens(sequence_ids)  # [B, N, 576]
        img = img.to(emb.dtype)
        state_tok = self.state_proj(state).to(emb.dtype)
        img_mask = (sequence_ids == self.image_id).unsqueeze(-1)
        emb = emb.masked_scatter(img_mask, img)  # fill 392 slots
        st_mask = (sequence_ids == self.state_id).unsqueeze(-1)
        emb = emb.masked_scatter(st_mask, state_tok)  # fill 1 slot
        h = self.language_backbone(inputs_embeds=emb).last_hidden_state
        return h  # [B, 392 + L + 1, 576]
