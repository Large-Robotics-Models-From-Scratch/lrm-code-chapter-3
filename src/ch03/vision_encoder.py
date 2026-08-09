"""Vision encoder: a frozen SigLIP-base/16 plus a trainable projection.

SigLIP arrives pre-trained on billions of image-text pairs. We treat it
as a fixed feature extractor: every SigLIP parameter is frozen, and the
only thing trained is a single linear layer that projects SigLIP's
768-dim patch tokens up to the book's common 576-dim width.

A 224x224 image at patch size 16 yields a 14x14 grid, so the encoder
returns 196 patch tokens per frame.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import SiglipVisionModel

from ch03.preprocess import SIGLIP_INPUT_SIZE, preprocess_image

SIGLIP_MODEL = "google/siglip-base-patch16-224"
SIGLIP_WIDTH = 768
NUM_PATCHES = 196


class VisionEncoder(nn.Module):
    """Frozen SigLIP patch encoder with a 768 to 576 projection."""

    def __init__(
        self,
        model_name: str = SIGLIP_MODEL,
        hidden_dim: int = 576,
    ) -> None:
        super().__init__()
        self.siglip = SiglipVisionModel.from_pretrained(model_name)
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

    def siglip_features(self, images: torch.Tensor) -> torch.Tensor:
        """Raw frozen SigLIP patch features, ``[B, 196, 768]``.

        These are the features before our projection. The self-similarity
        figure reads them directly: they are what the frozen encoder
        represents, with the untrained projection out of the way.

        Frames at any spatial size are accepted: anything that is not
        already 224 x 224 goes through the same ``preprocess_image``
        resize the probe path uses, so the whole preprocessing path
        lives inside the vision module.
        """
        if images.shape[-2:] != (SIGLIP_INPUT_SIZE, SIGLIP_INPUT_SIZE):
            images = preprocess_image(images)
        pixel_values = (images - self.pixel_mean) / self.pixel_std
        return self.siglip(pixel_values=pixel_values).last_hidden_state

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode ``[B, 3, H, W]`` to ``[B, 196, 576]`` patch tokens.

        Frames not already at 224 x 224 are resized internally.
        """
        return self.project(self.siglip_features(images))
