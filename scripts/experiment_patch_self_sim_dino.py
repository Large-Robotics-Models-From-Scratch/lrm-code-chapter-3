#!/usr/bin/env python
"""Same patch self-similarity + PCA experiment, on DINOv2.

DINOv2 is trained self-supervised for dense features, so the same code
that gave noise on SigLIP should give clean object masks here. Direct
backbone comparison: identical frame, queries, and readouts.
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
from transformers import AutoImageProcessor, AutoModel

MODEL = "facebook/dinov2-base"   # patch14, 1 CLS token, no registers
N_PREFIX = 1                     # drop CLS
SIZE, PATCH = 224, 14            # -> 16x16 = 256 patches
GRID = SIZE // PATCH
CACHE = Path.home() / ".cache/huggingface/lerobot/hub"
MP4 = next(CACHE.rglob("observation.images.up/chunk-000/file-000.mp4"))
OUT = Path(__file__).resolve().parent.parent / "figures" / "ch3_patch_self_sim_dino.png"
FRAME_IDX = 120


def pick_queries(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    bright = rgb.mean(-1)
    arm_mask = np.zeros_like(bright); arm_mask[40:170, 70:150] = 1
    ay, ax = np.unravel_index(np.argmin(np.where(arm_mask > 0, bright, 1e9)), bright.shape)
    redness = r - 0.5 * g - b
    cube_mask = np.zeros_like(redness); cube_mask[150:215, 80:160] = 1
    cy, cx = np.unravel_index(np.argmax(np.where(cube_mask > 0, redness, -1e9)), redness.shape)
    return (int(ax), int(ay)), (int(cx), int(cy))


def patch_index(x, y):
    return (y // PATCH) * GRID + (x // PATCH)


def sim_map(patches, qidx):
    pn = F.normalize(patches, dim=-1)
    m = (pn @ pn[qidx]).reshape(GRID, GRID)
    return (m - m.min()) / (m.max() - m.min() + 1e-8)


def pca_rgb(patches):
    x = patches - patches.mean(0, keepdim=True)
    _, _, v = torch.pca_lowrank(x, q=3)
    proj = x @ v
    proj = (proj - proj.min(0).values) / (proj.max(0).values - proj.min(0).values + 1e-8)
    return proj.reshape(GRID, GRID, 3).numpy()


def overlay(ax, rgb, heat, title, qxy):
    ax.imshow(rgb)
    up = F.interpolate(torch.tensor(heat)[None, None], size=rgb.shape[:2],
                       mode="bilinear", align_corners=False)[0, 0].numpy()
    ax.imshow(up, cmap="turbo", alpha=0.5)
    ax.scatter([qxy[0]], [qxy[1]], c="white", s=120, edgecolors="black")
    ax.set_title(title, fontsize=10); ax.axis("off")


def main() -> int:
    proc = AutoImageProcessor.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).eval()
    raw = iio.imread(MP4, index=FRAME_IDX, plugin="pyav")
    pil = Image.fromarray(raw).resize((SIZE, SIZE))
    rgb = np.asarray(pil).astype(np.float32) / 255.0

    img_in = proc(images=pil, return_tensors="pt", do_resize=False, do_center_crop=False)
    with torch.no_grad():
        patches = model(pixel_values=img_in["pixel_values"]).last_hidden_state[0, N_PREFIX:]
    print("n_patches:", patches.shape[0], "grid:", GRID)

    arm_xy, cube_xy = pick_queries(rgb)
    print("arm query px:", arm_xy, "cube query px:", cube_xy)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(rgb)
    axes[0].scatter(*zip(arm_xy, cube_xy), c=["cyan", "red"], s=120, edgecolors="black")
    axes[0].set_title("DINOv2 frame (cyan=arm, red=cube)", fontsize=10); axes[0].axis("off")
    overlay(axes[1], rgb, sim_map(patches, patch_index(*arm_xy)), "self-sim: arm patch", arm_xy)
    overlay(axes[2], rgb, sim_map(patches, patch_index(*cube_xy)), "self-sim: cube patch", cube_xy)
    axes[3].imshow(np.asarray(Image.fromarray((pca_rgb(patches) * 255).astype("uint8")).resize((SIZE, SIZE), Image.NEAREST)))
    axes[3].set_title("patch PCA (top-3 -> RGB)", fontsize=10); axes[3].axis("off")

    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
