"""Shared pytest fixtures for the ch03 test suite.

Module-specific fixtures land in each test_<module>.py.
Cross-module fixtures (dummy image batch, instructions, state) live here.
"""

import pytest
import torch


@pytest.fixture
def dummy_image_batch():
    """[2, 3, 224, 224] random tensor in [0, 1] — SigLIP input shape."""
    return torch.rand(2, 3, 224, 224)


@pytest.fixture
def dummy_instructions():
    """List of 2 strings — matches batch size of dummy_image_batch."""
    return ["pick up the cube", "place it in the target"]


@pytest.fixture
def dummy_state(request):
    """[2, state_dim] random tensor; defaults to state_dim=7 (Ch 2 hand-off)."""
    state_dim = getattr(request, "param", 7)
    return torch.rand(2, state_dim)
