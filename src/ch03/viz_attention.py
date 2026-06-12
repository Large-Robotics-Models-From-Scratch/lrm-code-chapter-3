"""Attention rollout, built from scratch for SigLIP.

Attention rollout (Abnar and Zuidema, 2020) turns a stack of per-layer
attention maps into a single map that approximates how much each input
patch influences the encoder's output. The recipe per layer: average the
heads, add the identity to account for residual connections, normalize
each row back to a distribution, then multiply the layers together.

SigLIP has no class token, so there is no single CLS row to read the map
from. Instead the per-patch saliency is the average attention each patch
receives across all queries, reshaped to the 14x14 patch grid.
"""

from __future__ import annotations

import matplotlib.figure
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def attention_rollout(
    attentions: list[torch.Tensor],
    discard_ratio: float = 0.0,
) -> torch.Tensor:
    """Compose per-layer attention into one ``[B, N, N]`` rollout map.

    ``attentions`` is one tensor per layer, each ``[B, H, N, N]``. The
    ``discard_ratio`` optionally zeroes the lowest-weight connections per
    layer before composing, which sharpens the map.
    """
    if not attentions:
        raise ValueError("attentions must be a non-empty list of layers.")
    device = attentions[0].device
    dtype = attentions[0].dtype
    num_tokens = attentions[0].size(-1)
    eye = torch.eye(num_tokens, device=device, dtype=dtype)
    result = eye.unsqueeze(0).expand(attentions[0].size(0), -1, -1).clone()

    for layer in attentions:
        heads_avg = layer.mean(dim=1)
        if discard_ratio > 0.0:
            heads_avg = _discard_lowest(heads_avg, discard_ratio)
        augmented = heads_avg + eye
        augmented = augmented / augmented.sum(dim=-1, keepdim=True)
        result = augmented @ result
    return result


def _discard_lowest(
    attn: torch.Tensor,
    ratio: float,
) -> torch.Tensor:
    """Zero the lowest-weight ``ratio`` fraction of each attention row."""
    num_tokens = attn.size(-1)
    k = int(num_tokens * ratio)
    if k == 0:
        return attn
    flat = attn.clone()
    threshold = flat.kthvalue(k, dim=-1, keepdim=True).values
    return torch.where(flat < threshold, torch.zeros_like(flat), flat)


def patch_saliency(rollout: torch.Tensor) -> torch.Tensor:
    """Reduce an ``[N, N]`` rollout to a ``[grid, grid]`` saliency map.

    Saliency is the mean attention each patch receives across queries.
    """
    received = rollout.mean(dim=-2)
    grid = int(received.numel() ** 0.5)
    return received.reshape(grid, grid)


def overlay_heatmap(
    image: torch.Tensor,
    rollout: torch.Tensor,
    save_path: str | None = None,
    title: str | None = None,
) -> matplotlib.figure.Figure:
    """Overlay a rollout saliency map on one ``[3, 224, 224]`` image.

    ``rollout`` is one ``[N, N]`` map. A ``[1, N, N]`` batch is accepted
    and squeezed; a larger batch raises, since one overlay needs one map.
    """
    if rollout.dim() == 3:
        if rollout.size(0) != 1:
            raise ValueError(
                "overlay_heatmap draws one image; pass a single "
                "[N, N] map (e.g. rollout[i]), not a [B, N, N] batch."
            )
        rollout = rollout[0]
    saliency = patch_saliency(rollout)
    heat = F.interpolate(
        saliency[None, None],
        size=image.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)

    rgb = image.detach().cpu().permute(1, 2, 0).numpy()
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(rgb)
    ax.imshow(heat.detach().cpu().numpy(), cmap="gray", alpha=0.6)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
