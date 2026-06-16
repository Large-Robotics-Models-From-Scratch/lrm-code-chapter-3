"""ch03 — the VLA backbone built in Chapter 3.

Re-exports the Ch 4 contract symbols at the package root; the set grows
as each section PR lands. This PR adds the language backbone.
"""

from ch03.language_backbone import LanguageBackbone
from ch03.preprocess import preprocess_image
from ch03.vision_encoder import VisionEncoder

__all__ = ["VisionEncoder", "LanguageBackbone", "preprocess_image"]
