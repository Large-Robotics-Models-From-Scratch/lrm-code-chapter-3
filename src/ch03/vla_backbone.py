"""VLA backbone: vision, language, and state fused into one sequence.

This is the chapter's deliverable and the frozen interface Chapter 4
builds on. The backbone encodes an image, an instruction, and the robot
state, lays their tokens out as a single sequence in the order

    [image patches (196), language tokens (L), state (1)]

and runs the causal fusion transformer over them. The output is a
sequence of contextualized hidden states ready for an action head.

Do not rename ``VLABackbone`` or change the ``forward`` signature: the
shape ``[B, 196 + L + 1, 512]`` is the contract Chapter 4 reads from.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ch03.fusion_transformer import FusionTransformer
from ch03.language_backbone import LanguageBackbone
from ch03.preprocess import preprocess_image
from ch03.state_encoder import StateEncoder
from ch03.vision_encoder import NUM_PATCHES, VisionEncoder


class VLABackbone(nn.Module):
    """Compose vision, language, and state into fused hidden states."""

    def __init__(
        self,
        hidden_dim: int = 512,
        state_dim: int = 6,
        num_fusion_layers: int = 6,
        num_fusion_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vision_encoder = VisionEncoder(hidden_dim=hidden_dim)
        self.language_backbone = LanguageBackbone(hidden_dim=hidden_dim)
        self.state_encoder = StateEncoder(
            state_dim=state_dim, hidden_dim=hidden_dim
        )
        self.fusion_transformer = FusionTransformer(
            hidden_dim=hidden_dim,
            num_layers=num_fusion_layers,
            num_heads=num_fusion_heads,
            dropout=dropout,
        )

    def forward(
        self,
        image: torch.Tensor,
        instruction: list[str],
        state: torch.Tensor,
        output_attentions: bool = False,
    ):
        """Fuse one batch into ``[B, 196 + L + 1, 512]`` hidden states.

        ``image`` is a ``[B, 3, H, W]`` batch in ``[0, 1]`` (resized to
        224x224 here); ``instruction`` is a length-B list of strings;
        ``state`` is ``[B, state_dim]``. With ``output_attentions`` the
        fusion transformer's per-layer attention maps are also returned.
        """
        image = preprocess_image(image)
        image_tokens = self.vision_encoder(image)
        lang_tokens, lang_mask = self.language_backbone(instruction)
        state_token = self.state_encoder(state)

        tokens = torch.cat(
            [image_tokens, lang_tokens, state_token], dim=1
        )
        key_padding_mask = self._build_padding_mask(
            batch_size=tokens.size(0),
            lang_mask=lang_mask,
            device=tokens.device,
        )
        return self.fusion_transformer(
            tokens,
            key_padding_mask=key_padding_mask,
            output_attentions=output_attentions,
        )

    @staticmethod
    def _build_padding_mask(
        batch_size: int,
        lang_mask: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Mark padded language positions; image and state never pad."""
        image_pad = torch.zeros(
            batch_size, NUM_PATCHES, dtype=torch.bool, device=device
        )
        lang_pad = lang_mask == 0
        state_pad = torch.zeros(
            batch_size, 1, dtype=torch.bool, device=device
        )
        return torch.cat([image_pad, lang_pad, state_pad], dim=1)
