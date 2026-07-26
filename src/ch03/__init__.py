"""ch03 - the VLA backbone built in Chapter 3.

Re-exports the Ch 4 contract symbols at the package root so downstream
chapters can ``from ch03 import VLABackbone`` without knowing the
submodule layout.

``FusionTransformer`` is the optional separate-encoder fusion of exercise
3.4, not on the main path, so it is not exported here; readers who build
it import it directly with
``from ch03.fusion_transformer import FusionTransformer``.
"""

from ch03.language_backbone import LanguageBackbone
from ch03.preprocess import preprocess_image
from ch03.sample import load_sample
from ch03.state_encoder import StateEncoder
from ch03.vision_encoder import VisionEncoder
from ch03.vla_backbone import VLABackbone

__all__ = [
    "VLABackbone",
    "VisionEncoder",
    "LanguageBackbone",
    "StateEncoder",
    "preprocess_image",
    "load_sample",
]
