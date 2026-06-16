"""State encoder: the robot's proprioception as a single token.

The SO-100 reports a 6-dim proprioceptive state vector (five arm joint
positions plus the gripper position). A two-layer MLP lifts those six
numbers into one 512-dim token that sits alongside the image and
language tokens in the fused sequence.

Going from 6 dims up to 512 benefits from a nonlinearity in the middle,
so this is a ``Linear -> GELU -> Linear`` MLP rather than a single
linear projection.
"""

from __future__ import annotations

import torch
import torch.nn as nn

STATE_DIM = 6


class StateEncoder(nn.Module):
    """Project ``[B, state_dim]`` state to one ``[B, 1, 512]`` token."""

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Encode ``[B, state_dim]`` to a single ``[B, 1, 512]`` token."""
        token = self.mlp(state)
        return token.unsqueeze(1)
