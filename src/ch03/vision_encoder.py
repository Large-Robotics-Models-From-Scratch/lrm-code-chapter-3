"""Vision encoder: a frozen SigLIP-base/16 plus a trainable projection.

SigLIP arrives pre-trained on billions of image-text pairs. We treat it
as a fixed feature extractor: every SigLIP parameter is frozen, and the
only thing trained is a single linear layer that projects SigLIP's
768-dim patch tokens down to the book's common 512-dim width.

A 224x224 image at patch size 16 yields a 14x14 grid, so the encoder
returns 196 patch tokens per frame.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import SiglipVisionModel

SIGLIP_MODEL = "google/siglip-base-patch16-224"
SIGLIP_WIDTH = 768
NUM_PATCHES = 196


class VisionEncoder(nn.Module):
    """Frozen SigLIP patch encoder with a 768 to 512 projection."""

    def __init__(
        self,
        model_name: str = SIGLIP_MODEL,
        hidden_dim: int = 512,
        attn_implementation: str = "sdpa",
    ) -> None:
        super().__init__()
        # SDPA is the fast default for training. The attention-rollout
        # visualization needs the per-layer maps, which only the "eager"
        # path exposes, so that one figure loads the encoder with
        # attn_implementation="eager".
        self.attn_implementation = attn_implementation
        self.siglip = SiglipVisionModel.from_pretrained(
            model_name,
            attn_implementation=attn_implementation,
        )
        self.siglip.eval()  # frozen: never run dropout, even under .train()
        for param in self.siglip.parameters():
            param.requires_grad = False
        self.project = nn.Linear(SIGLIP_WIDTH, hidden_dim)
        # SigLIP expects pixel values normalized to [-1, 1]. The buffers
        # keep that detail inside the encoder so callers pass [0, 1].
        self.register_buffer("pixel_mean", torch.tensor(0.5))
        self.register_buffer("pixel_std", torch.tensor(0.5))

    def train(self, mode: bool = True) -> "VisionEncoder":
        """Keep the frozen SigLIP in eval mode when the parent trains."""
        super().train(mode)
        self.siglip.eval()
        return self

    def forward(
        self,
        images: torch.Tensor,
        output_attentions: bool = False,
    ):
        """Encode ``[B, 3, 224, 224]`` images to ``[B, 196, 512]`` tokens.

        When ``output_attentions`` is set, also returns the per-layer
        SigLIP self-attention tensors for attention-rollout visualization.
        This requires the encoder loaded with ``attn_implementation=
        "eager"``; the fast SDPA default does not expose attention maps.
        """
        if output_attentions and self.attn_implementation != "eager":
            raise ValueError(
                "output_attentions=True needs the encoder loaded with "
                'attn_implementation="eager"; the default "sdpa" backend '
                "does not expose per-layer attention maps."
            )
        pixel_values = (images - self.pixel_mean) / self.pixel_std
        outputs = self.siglip(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
        )
        patches = self.project(outputs.last_hidden_state)
        if output_attentions:
            return patches, outputs.attentions
        return patches
