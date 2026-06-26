"""Patch self-similarity: which regions does the frozen encoder group?

Pick one patch (say, on the brick), take its frozen SigLIP feature
vector, and measure cosine similarity to every other patch. Reshape to
the 14x14 grid and the patches the encoder represents like the query
light up: the rest of the brick for a brick query, the arm for an arm
query. This reads the frozen features directly, so it shows what the
encoder represents with no training and no attention machinery.

A more familiar probe, attention rollout, composes the per-layer
attention maps. It reads cleanly off a class token, which SigLIP does
not have, and after a dozen layers of token mixing the attention weights
no longer point back to specific input patches. Reading the patch
features sidesteps both problems and is less code.
"""

from __future__ import annotations

import matplotlib.figure
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from ch03.preprocess import preprocess_image

GRID = 14  # 196 patches arranged as a 14 x 14 grid


def patch_self_similarity(
    features: torch.Tensor,
    query_row: int,
    query_col: int,
) -> torch.Tensor:
    """Cosine similarity of one query patch to all patches, as ``[14, 14]``.

    ``features`` is ``[196, 768]`` (or ``[1, 196, 768]``) raw SigLIP
    features for one frame. The query patch is named by its ``(row, col)``
    position in the 14x14 grid.
    """
    if features.dim() == 3:
        if features.shape[0] != 1:
            raise ValueError(
                f"patch_self_similarity needs a single frame; got a "
                f"batch of {features.shape[0]}. Index one frame first."
            )
        features = features[0]
    if not (0 <= query_row < GRID and 0 <= query_col < GRID):
        raise ValueError(
            f"query (row, col)=({query_row}, {query_col}) is outside "
            f"the {GRID}x{GRID} grid [0, {GRID})."
        )
    if features.shape[0] != GRID * GRID:
        raise ValueError(
            f"expected {GRID * GRID} patches for a {GRID}x{GRID} grid; "
            f"got {features.shape[0]}."
        )
    normed = F.normalize(features, dim=-1)
    query = normed[query_row * GRID + query_col]
    return (normed @ query).reshape(GRID, GRID)


def _overlay(ax, frame: torch.Tensor, simmap: torch.Tensor) -> None:
    heat = F.interpolate(
        simmap[None, None],
        size=frame.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    ax.imshow(frame.detach().cpu().permute(1, 2, 0).numpy())
    ax.imshow(heat.detach().cpu().numpy(), cmap="inferno", alpha=0.6)
    ax.axis("off")


def similarity_grid(
    vision_encoder,
    image: torch.Tensor,
    queries: list[tuple[int, int, str]],
    save_path: str | None = None,
) -> matplotlib.figure.Figure:
    """Show self-similarity for one or more query patches, side by side.

    ``image`` is one ``[3, H, W]`` frame in ``[0, 1]``. ``queries`` is a
    list of ``(row, col, label)`` grid positions. The first panel marks
    each query on the frame; the rest show that query's self-similarity.
    Pass a frame at any resolution; the encoder preprocesses to 224x224.
    """
    if not queries:
        raise ValueError("queries must be a non-empty list.")
    device = next(vision_encoder.parameters()).device
    frame = preprocess_image(image).unsqueeze(0).to(device)
    was_training = vision_encoder.training
    vision_encoder.eval()
    try:
        with torch.no_grad():
            features = vision_encoder.siglip_features(frame)[0]
    finally:
        vision_encoder.train(was_training)

    height, width = image.shape[-2:]
    fig, axes = plt.subplots(
        1, len(queries) + 1, figsize=(4 * (len(queries) + 1), 4)
    )
    axes[0].imshow(image.detach().cpu().permute(1, 2, 0).numpy())
    axes[0].set_title("query patches", fontsize=10)
    axes[0].axis("off")
    for row, col, _ in queries:
        px = (col + 0.5) / GRID * width
        py = (row + 0.5) / GRID * height
        axes[0].scatter(
            [px], [py], s=60, c="white", edgecolors="black", linewidths=1.5
        )

    for ax, (row, col, label) in zip(axes[1:], queries):
        _overlay(ax, image, patch_self_similarity(features, row, col))
        ax.set_title(label, fontsize=10)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def tracking_grid(
    vision_encoder,
    frames: list[torch.Tensor],
    queries: list[tuple[int, int]],
    labels: list[str] | None = None,
    save_path: str | None = None,
) -> matplotlib.figure.Figure:
    """Self-similarity of the object's patch in each of several frames.

    ``frames`` is a list of ``[3, H, W]`` frames; ``queries`` gives the
    object's ``(row, col)`` patch in each frame (it moves, so the query
    moves with it). The highlight tracks the object across positions,
    showing the frozen encoder localizes it wherever it spawns.
    """
    if not frames:
        raise ValueError("frames must be a non-empty list.")
    if len(queries) != len(frames):
        raise ValueError("one (row, col) query per frame is required.")
    if labels is not None and len(labels) != len(frames):
        raise ValueError(
            f"got {len(labels)} labels for {len(frames)} frames; one "
            f"label per frame is required so no panel is dropped."
        )
    labels = labels or [f"frame {i}" for i in range(len(frames))]
    device = next(vision_encoder.parameters()).device
    fig, axes = plt.subplots(1, len(frames), figsize=(4 * len(frames), 4))
    if len(frames) == 1:
        axes = [axes]
    was_training = vision_encoder.training
    vision_encoder.eval()
    try:
        with torch.no_grad():
            for ax, frame, (row, col), label in zip(
                axes, frames, queries, labels
            ):
                feats = vision_encoder.siglip_features(
                    preprocess_image(frame).unsqueeze(0).to(device)
                )[0]
                _overlay(ax, frame, patch_self_similarity(feats, row, col))
                ax.set_title(label, fontsize=10)
    finally:
        vision_encoder.train(was_training)
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
