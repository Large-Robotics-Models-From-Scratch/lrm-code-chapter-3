#!/usr/bin/env python
"""Experiment: does the named object light up via image-text patch cosine?

Two readouts on SigLIP, same SO-101 frame:
  naive   - cosine(last_hidden_state patch, pooled text embed)
  dense   - MaskCLIP-style: push each patch through the pooling head's
            value+output projection first (puts patches in the text-
            aligned space), then cosine with the pooled text embed.

Writes a grid overlay so naive vs dense can be compared per prompt.
"""

import sys
from pathlib import Path

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, SiglipModel

MODEL = "google/siglip-base-patch16-224"
CACHE = Path.home() / ".cache/huggingface/lerobot/hub"
MP4 = next(CACHE.rglob("observation.images.up/chunk-000/file-000.mp4"))
OUT = Path(__file__).resolve().parent.parent / "figures" / "ch3_text_patch_sim.png"
FRAME_IDX = 120
PROMPTS = ["the orange block", "the robot gripper arm", "the table surface"]


def dense_patch_features(vision_model, patches):
    """MaskCLIP trick: per-patch features in the text-aligned space.

    Apply the pooling head's value projection and output projection to
    every patch independently (identity attention), then the head's
    layernorm+mlp residual, mirroring what the head does to the pooled
    token. Result lives in the same space as the pooled image embed.
    """
    head = vision_model.head
    mha = head.attention
    e = patches.shape[-1]
    w_v = mha.in_proj_weight[2 * e : 3 * e]
    b_v = mha.in_proj_bias[2 * e : 3 * e]
    v = patches @ w_v.T + b_v
    out = F.linear(v, mha.out_proj.weight, mha.out_proj.bias)
    hs = head.layernorm(out)
    return out + head.mlp(hs)


def sim_grid(sim_col, grid=14):
    """One [196] similarity column -> normalized [14,14]."""
    m = sim_col.reshape(grid, grid)
    return (m - m.min()) / (m.max() - m.min() + 1e-8)


def overlay(ax, rgb, heat, title):
    ax.imshow(rgb)
    up = F.interpolate(
        heat[None, None], size=rgb.shape[:2], mode="bilinear",
        align_corners=False,
    )[0, 0]
    ax.imshow(up.numpy(), cmap="turbo", alpha=0.5)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def main() -> int:
    proc = AutoProcessor.from_pretrained(MODEL)
    model = SiglipModel.from_pretrained(MODEL).eval()

    frame = iio.imread(MP4, index=FRAME_IDX, plugin="pyav")
    pil = Image.fromarray(frame).resize((224, 224))
    rgb = (torch.from_numpy(iio.imread(MP4, index=FRAME_IDX, plugin="pyav"))
           .float() / 255.0)
    rgb = F.interpolate(
        rgb.permute(2, 0, 1)[None], size=(224, 224), mode="bilinear",
        align_corners=False,
    )[0].permute(1, 2, 0).numpy()

    img_in = proc(images=pil, return_tensors="pt")
    txt_in = proc(
        text=PROMPTS, return_tensors="pt", padding="max_length", max_length=64
    )
    with torch.no_grad():
        vout = model.vision_model(pixel_values=img_in["pixel_values"])
        patches = vout.last_hidden_state[0]            # [196,768]
        dense = dense_patch_features(model.vision_model, patches)  # [196,768]
        tfeat = model.get_text_features(**txt_in)      # [P,768]

    pn = F.normalize(patches, dim=-1)
    dn = F.normalize(dense, dim=-1)
    tn = F.normalize(tfeat, dim=-1)
    sim_naive = pn @ tn.T                              # [196,P]
    sim_dense = dn @ tn.T                              # [196,P]

    fig, axes = plt.subplots(2, len(PROMPTS) + 1, figsize=(4 * (len(PROMPTS) + 1), 8))
    axes[0, 0].imshow(rgb); axes[0, 0].set_title("frame", fontsize=10); axes[0, 0].axis("off")
    axes[1, 0].imshow(rgb); axes[1, 0].set_title("frame", fontsize=10); axes[1, 0].axis("off")
    for j, prompt in enumerate(PROMPTS):
        overlay(axes[0, j + 1], rgb, sim_grid(sim_naive[:, j]), f"naive: {prompt}")
        overlay(axes[1, j + 1], rgb, sim_grid(sim_dense[:, j]), f"dense: {prompt}")
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
