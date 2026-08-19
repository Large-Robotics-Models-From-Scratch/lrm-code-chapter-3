"""Tests for patch self-similarity viz."""

import matplotlib
import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from matplotlib.patches import Rectangle

matplotlib.use("Agg")

from ch03.viz_similarity import (
    GRID,
    patch_self_similarity,
    similarity_grid,
    tracking_grid,
)


class _LocalFeatureEncoder(nn.Module):
    """Deterministic frozen features without a Hugging Face download."""

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        features = torch.zeros(GRID * GRID, 2)
        features[:, 1] = 1.0
        features[0] = torch.tensor([1.0, 0.0])
        features[1] = torch.tensor([-1.0, 0.0])
        features[2 * GRID + 3] = torch.tensor([0.0, 1.0])
        self.register_buffer("features", features)

    def siglip_features(self, images):
        return self.features.unsqueeze(0).expand(images.shape[0], -1, -1)


def _frame():
    return torch.zeros(3, 140, 140)


def test_query_is_most_similar_to_itself():
    # pure unit: cosine sim of a patch with itself is 1.0 and is the max
    feats = torch.rand(GRID * GRID, 768)
    sim = patch_self_similarity(feats, 7, 7)
    assert sim.shape == (GRID, GRID)
    assert torch.isclose(sim[7, 7], torch.tensor(1.0), atol=1e-5)
    # the query cell is the argmax of its own similarity map (an
    # exact-equality compare on floats is flaky; use argmax instead)
    row, col = divmod(int(sim.argmax()), GRID)
    assert (row, col) == (7, 7)


def test_accepts_batched_features():
    feats = torch.rand(1, GRID * GRID, 768)
    assert patch_self_similarity(feats, 0, 0).shape == (GRID, GRID)


def test_patch_self_similarity_rejects_invalid_query_position():
    with pytest.raises(ValueError, match="outside"):
        patch_self_similarity(torch.rand(GRID * GRID, 2), GRID, 0)


def test_patch_self_similarity_rejects_wrong_patch_count():
    with pytest.raises(ValueError, match="expected 196 patches"):
        patch_self_similarity(torch.rand(GRID * GRID - 1, 2), 0, 0)


def test_similarity_grid_rejects_empty_queries():
    with pytest.raises(ValueError, match="non-empty"):
        similarity_grid(None, _frame(), [])


def test_similarity_grid_keeps_raw_maps_on_one_shared_scale():
    """Independent min-max scaling would turn the -1 cell into zero."""
    fig = similarity_grid(
        _LocalFeatureEncoder(),
        _frame(),
        [(0, 0, "first"), (2, 3, "second")],
        similarity_range=(-0.1, 1.0),
    )
    heatmaps = [axis.images[-1] for axis in fig.axes[1:3]]

    assert [heatmap.get_clim() for heatmap in heatmaps] == [
        (-0.1, 1.0),
        (-0.1, 1.0),
    ]
    assert float(heatmaps[0].get_array().min()) < -0.1
    plt.close(fig)


def test_similarity_grid_outlines_each_query_patch_in_grid_coordinates():
    fig = similarity_grid(
        _LocalFeatureEncoder(),
        _frame(),
        [(5, 6, "brick"), (1, 2, "arm")],
    )
    outlines = [
        patch
        for patch in fig.axes[0].patches
        if isinstance(patch, Rectangle)
    ]

    assert [(patch.get_x(), patch.get_y()) for patch in outlines] == [
        (60.0, 50.0),
        (20.0, 10.0),
    ]
    sizes = [(patch.get_width(), patch.get_height()) for patch in outlines]
    assert sizes == [
        (10.0, 10.0),
        (10.0, 10.0),
    ]
    plt.close(fig)


def test_three_queries_render_a_2_by_2_grid_with_shared_colorbar():
    fig = similarity_grid(
        _LocalFeatureEncoder(),
        _frame(),
        [
            (11, 4, "brick query patch"),
            (5, 6, "arm query patch"),
            (12, 8, "tabletop query patch"),
        ],
    )
    content_axes = fig.axes[:4]

    assert [axis.get_title() for axis in content_axes] == [
        "source frame",
        "brick query patch",
        "arm query patch",
        "tabletop query patch",
    ]
    assert len(fig.axes) == 5
    assert fig.axes[-1].get_ylabel() == "cosine similarity to query patch"
    assert [axis.images[-1].get_clim() for axis in content_axes[1:]] == [
        (-0.1, 1.0),
        (-0.1, 1.0),
        (-0.1, 1.0),
    ]
    assert [
        (axis.get_subplotspec().rowspan.start,
         axis.get_subplotspec().colspan.start)
        for axis in content_axes
    ] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    plt.close(fig)


@pytest.mark.integration
def test_similarity_grid_panels():
    from ch03.vision_encoder import VisionEncoder

    ve = VisionEncoder().eval()
    fig = similarity_grid(
        ve, torch.rand(3, 480, 640), [(5, 6, "cube"), (3, 7, "arm")]
    )
    # One frame panel plus one per query. Count content panels rather
    # than len(fig.axes): the shared colorbar contributes an axes of its
    # own, and that colorbar is what makes the maps comparable, so it
    # must not be dropped just to keep an old count passing.
    content = [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]
    assert len(content) == 3
    assert len(fig.axes) == 4
    plt.close(fig)


@pytest.mark.integration
def test_tracking_grid_one_query_per_frame():
    from ch03.vision_encoder import VisionEncoder

    ve = VisionEncoder().eval()
    frames = [torch.rand(3, 480, 640) for _ in range(3)]
    fig = tracking_grid(ve, frames, [(5, 5), (5, 8), (8, 6)])
    assert len(fig.axes) == 3


def test_tracking_grid_validates_query_count():
    # one query per frame; a mismatch is caught before any model call,
    # so a None encoder is never touched
    with pytest.raises(ValueError):
        tracking_grid(None, [torch.rand(3, 8, 8)], [(0, 0), (1, 1)])
