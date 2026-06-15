#!/usr/bin/env python
"""Experiment: patch-to-patch self-similarity (DINOv2-style).

Pick one query patch on the arm and one on the cube, take cosine
similarity of that patch's SigLIP embedding against all 196 patches.
If the features are semantically coherent, the whole object lights up
(its outline), not just the queried point. Image-only, no text.

Also renders the canonical DINOv2 PCA-of-patches map: first 3 PCA
components of the patch features as RGB.
"""

import sys
from pathlib import Path

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, SiglipVisionModel

MODEL = "google/siglip-base-patch16-224"
CACHE = Path.home() / ".cache/huggingface/lerobot/hub"
MP4 = next(CACHE.rglob("observation.images.up/chunk-000/file-000.mp4"))
OUT = Path(__file__).resolve().parent.parent / "figures" / "ch3_patch_self_sim.png"
FRAME_IDX = 120
GRID, PATCH = 14, 16


def pick_queries(rgb224):
    """Heuristic query pixels: darkest center patch (arm), reddest
    bottom-center patch (cube). Returns (x, y) in 224 space each."""
    r, g, b = rgb224[..., 0], rgb224[..., 1], rgb224[..., 2]
    bright = rgb224.mean(-1)
    # arm: darkest patch within the central region
    arm_mask = np.zeros_like(bright); arm_mask[40:170, 70:150] = 1
    arm_idx = np.argmin(np.where(arm_mask > 0, bright, 1e9))
    ay, ax = np.unravel_index(arm_idx, bright.shape)
    # cube: reddest patch in the bottom-center band
    redness = r - 0.5 * g - b
    cube_mask = np.zeros_like(redness); cube_mask[150:215, 80:160] = 1
    cube_idx = np.argmax(np.where(cube_mask > 0, redness, -1e9))
    cy, cx = np.unravel_index(cube_idx, redness.shape)
    return (int(ax), int(ay)), (int(cx), int(cy))


def patch_index(x, y):
    return (y // PATCH) * GRID + (x // PATCH)


def sim_map(patches, qidx):
    pn = F.normalize(patches, dim=-1)
    sim = pn @ pn[qidx]
    m = sim.reshape(GRID, GRID)
    return (m - m.min()) / (m.max() - m.min() + 1e-8)


def pca_rgb(patches):
    x = patches - patches.mean(0, keepdim=True)
    _, _, v = torch.pca_lowrank(x, q=3)
    proj = x @ v                                  # [196,3]
    proj = (proj - proj.min(0).values) / (proj.max(0).values - proj.min(0).values + 1e-8)
    return proj.reshape(GRID, GRID, 3).numpy()


def overlay(ax, rgb, heat, title, qxy=None):
    ax.imshow(rgb)
    up = F.interpolate(
        torch.tensor(heat)[None, None], size=rgb.shape[:2],
        mode="bilinear", align_corners=False,
    )[0, 0].numpy()
    ax.imshow(up, cmap="turbo", alpha=0.5)
    if qxy is not None:
        ax.scatter([qxy[0]], [qxy[1]], c="white", s=120, edgecolors="black", marker="o")
    ax.set_title(title, fontsize=10); ax.axis("off")


def main() -> int:
    proc = AutoProcessor.from_pretrained(MODEL)
    model = SiglipVisionModel.from_pretrained(MODEL).eval()
    raw = iio.imread(MP4, index=FRAME_IDX, plugin="pyav")
    pil = Image.fromarray(raw).resize((224, 224))
    rgb = np.asarray(pil).astype(np.float32) / 255.0

    img_in = proc(images=pil, return_tensors="pt")
    with torch.no_grad():
        patches = model(pixel_values=img_in["pixel_values"]).last_hidden_state[0]

    arm_xy, cube_xy = pick_queries(rgb)
    print("arm query px:", arm_xy, "cube query px:", cube_xy)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(rgb)
    axes[0].scatter(*zip(arm_xy, cube_xy), c=["cyan", "red"], s=120, edgecolors="black")
    axes[0].set_title("frame (cyan=arm q, red=cube q)", fontsize=10); axes[0].axis("off")
    overlay(axes[1], rgb, sim_map(patches, patch_index(*arm_xy)), "self-sim: arm patch", arm_xy)
    overlay(axes[2], rgb, sim_map(patches, patch_index(*cube_xy)), "self-sim: cube patch", cube_xy)
    axes[3].imshow(np.asarray(Image.fromarray((pca_rgb(patches) * 255).astype("uint8")).resize((224, 224), Image.NEAREST)))
    axes[3].set_title("patch PCA (top-3 → RGB)", fontsize=10); axes[3].axis("off")

    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
