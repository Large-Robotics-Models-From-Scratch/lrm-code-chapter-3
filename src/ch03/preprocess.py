"""Image preprocessing: Chapter 2 dataloader to SigLIP input.

Chapter 2's ``make_pickplace_dataloader`` yields camera frames at the
dataset's native resolution, ``[B, 3, 480, 640]`` float32 in ``[0, 1]``.
SigLIP-base/16 expects ``224 x 224``. This module bridges the two with a
resize and nothing more: bicubic interpolation with antialiasing, the
resampling the pinned SigLIP image processor also uses (this matches its
resize and normalization settings, not every step of it). SigLIP's
own ``[-1, 1]`` normalization lives inside ``VisionEncoder`` so the
boundary here stays the ``[0, 1]`` convention Chapter 2 established.

This is the only resize in Chapter 3. ``VisionEncoder`` calls it rather
than repeating the interpolation, so the probe path and the encoder path
cannot drift apart.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

SIGLIP_INPUT_SIZE = 224


def preprocess_image(image: torch.Tensor) -> torch.Tensor:
    """Resize a ``[0, 1]`` image to SigLIP's ``224 x 224`` input.

    Accepts ``[3, H, W]`` or ``[B, 3, H, W]`` and returns the same rank
    with the spatial dimensions resized to ``224 x 224`` by bicubic
    interpolation with antialiasing. Pixel values stay in ``[0, 1]``;
    SigLIP normalization is applied later, inside the vision encoder.
    """
    single = image.ndim == 3
    if single:
        image = image.unsqueeze(0)
    resized = F.interpolate(
        image,
        size=(SIGLIP_INPUT_SIZE, SIGLIP_INPUT_SIZE),
        mode="bicubic",
        align_corners=False,
        antialias=True,  # avoid aliasing when downsampling 480/640 -> 224
    )
    return resized.squeeze(0) if single else resized
