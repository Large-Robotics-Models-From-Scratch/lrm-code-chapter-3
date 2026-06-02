"""ch03 package — the VLA backbone built in Chapter 3.

Re-exports the Ch 4 contract symbols at the package root so downstream
chapters can `from ch03 import VLABackbone` without knowing the submodule
layout.

Modules land here as PRs merge:
- PR 2: vision_encoder (SigLIP + project) + viz_similarity
- PR 3: language_backbone (SmolLM, native tokenizer)
- PR 4: state_encoder, fusion_transformer
- PR 5: vla_backbone (the integration + Ch 4 contract) + preprocess
- PR 6: viz_similarity tracking grid (figure 3.6) + notebook
"""

from ch03.fusion_transformer import FusionTransformer
from ch03.language_backbone import LanguageBackbone
from ch03.preprocess import preprocess_image
from ch03.state_encoder import StateEncoder
from ch03.vision_encoder import VisionEncoder
from ch03.vla_backbone import VLABackbone

__all__ = [
    "VLABackbone",
    "VisionEncoder",
    "LanguageBackbone",
    "StateEncoder",
    "FusionTransformer",
    "preprocess_image",
]
