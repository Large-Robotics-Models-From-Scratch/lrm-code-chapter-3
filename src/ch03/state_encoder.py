"""State encoder: the robot's proprioception as a single token.

The SO-101 reports a 6-dim proprioceptive state vector (five arm joint
positions plus the gripper position). A two-layer MLP lifts those six
numbers into one 576-dim token that sits alongside the image and
language tokens in the fused sequence.

Going from 6 dims up to 576 benefits from a nonlinearity in the middle,
so this is a ``Linear -> GELU -> Linear`` MLP rather than a single
linear projection.
"""

from __future__ import annotations

import torch
import torch.nn as nn

STATE_DIM = 6


class StateEncoder(nn.Module):
    """Project ``[B, state_dim]`` state to one ``[B, 576]`` state token."""

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        width: int = 576,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Encode ``[B, state_dim]`` to one ``[B, 576]`` state token."""
        return self.net(state)
