#!/usr/bin/env python
"""Render SigLIP attention-rollout overlays on a real SO-101 frame.

Decodes one frame straight from the cached LeRobot video (no lerobot
import), runs it through the eager VisionEncoder, composes the rollout,
and writes grayscale overlays at two discard ratios so the residual-
diffuse map and the sharpened map can be compared side by side.
"""

import sys
from pathlib import Path

import imageio.v3 as iio
import torch

from ch03.preprocess import preprocess_image
from ch03.vision_encoder import VisionEncoder
from ch03.viz_attention import attention_rollout, overlay_heatmap

CACHE = Path.home() / ".cache/huggingface/lerobot/hub"
MP4 = next(CACHE.rglob("observation.images.up/chunk-000/file-000.mp4"))
OUT = Path(__file__).resolve().parent.parent / "figures"
FRAME_IDX = 120


def main() -> int:
    OUT.mkdir(exist_ok=True)
    frame = iio.imread(MP4, index=FRAME_IDX, plugin="pyav")  # HWC uint8
    img = torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0
    img224 = preprocess_image(img)  # [3,224,224] in [0,1]

    enc = VisionEncoder(attn_implementation="eager").eval()
    with torch.no_grad():
        _, attentions = enc(img224.unsqueeze(0), output_attentions=True)

    for ratio in (0.0, 0.9):
        rollout = attention_rollout(attentions, discard_ratio=ratio)
        path = OUT / f"ch3_attn_up_f{FRAME_IDX}_discard{ratio}.png"
        overlay_heatmap(
            img224, rollout[0], save_path=str(path),
            title=f"SigLIP rollout (discard={ratio})",
        )
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
