"""Tests for patch self-similarity viz."""

import matplotlib
import pytest
import torch

matplotlib.use("Agg")

from ch03.viz_similarity import (
    GRID,
    patch_self_similarity,
    similarity_grid,
    tracking_grid,
)


def test_query_is_most_similar_to_itself():
    # pure unit: cosine sim of a patch with itself is 1.0 and is the max
    feats = torch.rand(GRID * GRID, 768)
    sim = patch_self_similarity(feats, 7, 7)
    assert sim.shape == (GRID, GRID)
    assert torch.isclose(sim[7, 7], torch.tensor(1.0), atol=1e-5)
    assert sim.max() == sim[7, 7]


def test_accepts_batched_features():
    feats = torch.rand(1, GRID * GRID, 768)
    assert patch_self_similarity(feats, 0, 0).shape == (GRID, GRID)


@pytest.mark.integration
def test_similarity_grid_panels():
    from ch03.vision_encoder import VisionEncoder

    ve = VisionEncoder().eval()
    fig = similarity_grid(
        ve, torch.rand(3, 480, 640), [(5, 6, "cube"), (3, 7, "arm")]
    )
    # one frame panel + one per query
    assert len(fig.axes) == 3


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
