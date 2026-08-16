"""One real sample from the Chapter 2 dataset, shipped in the repo.

Chapter 2's ``make_pickplace_dataloader`` is the real source of batches,
but it does not install alongside Chapter 3 under this repo's pins
(lerobot 0.5.1 wants huggingface-hub>=1.0 while transformers<5.0 wants
huggingface-hub<1.0), and decoding the dataset's videos needs a system
FFmpeg. So that the chapter never has to demonstrate the backbone on
``torch.rand`` noise, one real frame pair from
``lerobot/svla_so101_pickplace`` travels with the package: the two camera
views as lossless PNGs, plus the z-scored 6-dim state and the episode's
task string in ``assets/sample.json``.

The PNGs hold the exact uint8 pixels the dataloader decodes, so dividing
by 255 reproduces the dataloader's ``[0, 1]`` float32 frames bit for bit.
Frames stay at the dataset's native ``480 x 640``: resizing to SigLIP's
``224 x 224`` is ``preprocess_image``'s job, and the caller still does
it, exactly as it would with a Chapter 2 batch.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ASSETS = Path(__file__).parent / "assets"
# Camera order matches the observation prefix: camera 0 is the overhead
# view (observation.images.up), camera 1 the side view.
CAMERA_FRAMES = ("frame_up.png", "frame_side.png")
METADATA = "sample.json"


def _load_frame(path: Path) -> torch.Tensor:
    """Read one PNG as a ``[3, 480, 640]`` float32 frame in ``[0, 1]``."""
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    frame = torch.from_numpy(pixels).permute(2, 0, 1)  # HWC -> CHW
    return frame.to(torch.float32) / 255.0


def load_sample() -> tuple[torch.Tensor, torch.Tensor, str]:
    """Return one real ``(images, state, instruction)`` sample.

    ``images`` is ``[1, 2, 3, 480, 640]`` float32 in ``[0, 1]``, the two
    camera views of a single timestep at the dataset's native resolution
    (run ``preprocess_image`` on it before the vision encoder).
    ``state`` is ``[1, 6]`` float32, the z-scored SO-101 state (the
    dataset stores it as float64, so this casts it to the float32 the
    model's linear layers expect). ``instruction`` is the episode's own
    task string.
    """
    meta = json.loads((ASSETS / METADATA).read_text())
    frames = [_load_frame(ASSETS / name) for name in CAMERA_FRAMES]
    images = torch.stack(frames).unsqueeze(0)  # [1, 2, 3, 480, 640]
    state = torch.tensor(meta["state"], dtype=torch.float32).unsqueeze(
        0
    )  # [1, 6] float32, cast down from the dataset's float64
    return images, state, meta["instruction"]
