"""State encoder: the robot's proprioception as a single embedding.

The SO-101 reports a 6-dim proprioceptive state vector (five arm joint
positions plus the gripper position). A two-layer MLP lifts those six
numbers into one 576-dim state embedding that occupies the state position
alongside the visual and language positions in the observation prefix.

Going from 6 dims up to 576 benefits from a nonlinearity in the middle,
so this is a ``Linear -> GELU -> Linear`` MLP rather than a single
linear projection.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# The SO-101's proprioceptive width, named for tests and for callers
# that build a StateEncoder from the dataset contract rather than from
# the default. Listing 3.3 spells the default out as 6.
STATE_DIM = 6


class StateEncoder(nn.Module):
    """Project ``[B, state_dim]`` state to a ``[B, 1, 576]`` embedding."""

    def __init__(self, state_dim=6, width=576):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Encode ``[B, state_dim]`` to one ``[B, 1, 576]`` state embedding.

        The ``unsqueeze(1)`` adds the sequence dimension, so the
        output concatenates directly with the ``[B, 392, 576]`` image
        block and the ``[B, L, 576]`` text block.
        """
        return self.net(state).unsqueeze(1)
