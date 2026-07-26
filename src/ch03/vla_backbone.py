"""VLA backbone: vision, language, and state fused in one sequence.

This is the chapter's deliverable and the frozen interface Chapter 4
builds on. ``VLABackbone`` projects the two camera views into the
language backbone's own token stream and lets the pretrained attention
do the fusion. There is no separate fusion transformer: the language
backbone is the fuser.

The image and state tokens are spliced into reserved placeholder rows of
the backbone's input embeddings with ``masked_scatter``. The token order
the template builds is

    [image (392), text/language (L), state (1)]

so the contract Chapter 4 reads is ``[B, 392 + L + 1, 576]``.

The backbone composes rather than rebuilds. The vision path is the
``VisionEncoder`` of section 3.2, used as-is: frozen SigLIP, the 768 to
576 projection, and SigLIP's ``[0, 1]`` to ``[-1, 1]`` pixel
normalization all stay inside it. The grown input embedding table is
handed back to the language backbone, so the whole model holds exactly
one embedding table and one image projection.

Do not rename ``VLABackbone`` or change the ``forward`` signature
``(images, sequence_ids, state)``: the ``[B, 392 + L + 1, 576]`` output
is the contract Chapter 4 reads from. ``sequence_ids`` is OUR fused
layout (image + text + state); it is distinct from Hugging Face's
``input_ids``, which stay the tokenizer/model API name unchanged.
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

# Two placeholder ids, just past the native vocab, mark the rows the
# splice overwrites. Their embeddings are never read (masked_scatter
# replaces them before the backbone runs), so this is not the Chapter 4
# vocabulary expansion: it reserves two inert splice markers, not real
# tokens.
DEFAULT_IMAGE_ID = SMOLLM_VOCAB
DEFAULT_STATE_ID = SMOLLM_VOCAB + 1


class VLABackbone(nn.Module):
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
        # Listing 3.1's encoder, reused verbatim: frozen SigLIP, the
        # 768 to 576 projection, and SigLIP's [0,1] -> [-1,1] pixel
        # normalization all live inside it, so the backbone owns no
        # second copy of any of them.
        self.vision_encoder = VisionEncoder(hidden_dim=SMOLLM_WIDTH)
        self.state_encoder = StateEncoder(6, SMOLLM_WIDTH)
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
        # Hand the grown table back so the model holds exactly ONE
        # embedding table. Without this the language backbone would keep
        # its original table as well, ~28M trainable parameters that
        # forward never reads (it passes inputs_embeds). Swapping the
        # table in directly, rather than through Hugging Face's
        # token-resize helper, is what keeps config.vocab_size at 49152:
        # the resize helper rewrites the config, and that number is a
        # Chapter 3 invariant.
        self.language_backbone.set_input_embeddings(self.embed_tokens)

    @staticmethod
    def _grow_embeddings(
        embed: nn.Embedding, new_size: int
    ) -> nn.Embedding:
        """Copy an embedding table into a slightly larger one.

        The new table inherits the source table's dtype and device.
        ``nn.Embedding`` would otherwise default to float32 on the CPU,
        and a backbone loaded in bfloat16 (what recent Transformers
        releases do for SmolLM2) would then fail in ``forward`` with a
        dtype mismatch.
        """
        if embed.num_embeddings >= new_size:
            return embed
        grown = nn.Embedding(
            new_size,
            embed.embedding_dim,
            padding_idx=embed.padding_idx,
            dtype=embed.weight.dtype,
            device=embed.weight.device,
        )
        with torch.no_grad():
            grown.weight[: embed.num_embeddings] = embed.weight
        return grown

    def tokenize_instruction(self, instruction: str) -> list[int]:
        """Return the L native SmolLM2 text ids for one instruction.

        HF's tokenizer hands back ``input_ids``; we pull the single
        row out as a plain ``list[int]`` so ``build_sequence_ids`` can
        splice the text between the image and state placeholders. No
        padding or truncation here: callers building variable-L
        batches pad the assembled ``sequence_ids`` rows themselves
        (Chapter 4's collate does this).
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
        patches = self.vision_encoder(
            images.flatten(0, 1)
        )  # [B*2, 196, 576] frozen SigLIP, already projected
        img = patches.reshape(
            B, -1, SMOLLM_WIDTH
        )  # [B, 392, 576] cam0 then cam1
        emb = self.embed_tokens(sequence_ids)  # [B, N, 576]
        img = img.to(emb.dtype)
        state_tok = self.state_encoder(state).to(emb.dtype)
        img_mask = (sequence_ids == self.image_id).unsqueeze(-1)
        st_mask = (sequence_ids == self.state_id).unsqueeze(-1)
        # Guard the splice (repo hardening; the book listing omits
        # it): masked_scatter fills only as many slots as the mask
        # has True positions, so a template with the wrong number of
        # placeholders would silently drop part of img/state_tok yet
        # still return the expected shape.
        img_counts = img_mask.sum(dim=(1, 2))
        if not bool((img_counts == IMAGE_TOKENS).all()):
            raise ValueError(
                f"every sequence_ids row needs exactly {IMAGE_TOKENS} "
                f"image placeholders (id {self.image_id}); got "
                f"{img_counts.tolist()}."
            )
        st_counts = st_mask.sum(dim=(1, 2))
        if not bool((st_counts == 1).all()):
            raise ValueError(
                f"every sequence_ids row needs exactly 1 state "
                f"placeholder (id {self.state_id}); got "
                f"{st_counts.tolist()}."
            )
        emb = emb.masked_scatter(img_mask, img)  # fill 392 slots
        emb = emb.masked_scatter(st_mask, state_tok)  # fill 1 slot
        h = self.language_backbone(inputs_embeds=emb).last_hidden_state
        return h  # [B, 392 + L + 1, 576]
