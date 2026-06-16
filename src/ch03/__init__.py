"""ch03 — the VLA backbone built in Chapter 3.

Re-exports the Ch 4 contract symbols at the package root; the set grows
as each section PR lands. This PR adds the state encoder and fusion.
"""

from ch03.fusion_transformer import FusionTransformer
from ch03.language_backbone import LanguageBackbone
from ch03.preprocess import preprocess_image
from ch03.state_encoder import StateEncoder
from ch03.vision_encoder import VisionEncoder

__all__ = [
    "VisionEncoder",
    "LanguageBackbone",
    "StateEncoder",
    "FusionTransformer",
    "preprocess_image",
]
