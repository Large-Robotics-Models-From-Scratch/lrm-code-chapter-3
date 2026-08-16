"""Separate-encoder fusion: an optional, standalone alternative.

This module is the exercise-only separate-encoder fuser. It is not part
of the chapter's main path and is not imported by the main backbone; it
exists for the chapter's optional fusion exercise, where the reader
bolts a from-scratch fusion Transformer onto the frozen streams and
compares it against the direct concatenation the chapter ships.

In separate-encoder fusion each stream is encoded on its own and a
dedicated module attends over the concatenated visual, language, and
state input embeddings. This module is that dedicated fuser: a stack of
pre-norm self-attention blocks that mixes the three streams into one
sequence of contextualized hidden states.

Attention is causal (each position attends only to itself and earlier
positions), matching the VLA backbone so the comparison is like for
like and Chapter 4's autoregressive action prediction drops on
without architecture surgery.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FusionBlock(nn.Module):
    """One pre-norm self-attention block with a feed-forward layer."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_mlp = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        attn_mask: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        normed = self.norm_attn(tokens)
        attended, weights = self.attn(
            normed,
            normed,
            normed,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        tokens = tokens + attended
        tokens = tokens + self.mlp(self.norm_mlp(tokens))
        return tokens, weights


class FusionTransformer(nn.Module):
    """A causal pre-norm transformer over the concatenated sequence."""

    def __init__(
        self,
        hidden_dim: int = 576,
        num_layers: int = 6,
        num_heads: int = 9,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            FusionBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        )
        self.norm_out = nn.LayerNorm(hidden_dim)

    @staticmethod
    def causal_mask(num_tokens: int, device: torch.device) -> torch.Tensor:
        """Boolean upper-triangular mask; ``True`` blocks future tokens."""
        ones = torch.ones(
            num_tokens, num_tokens, dtype=torch.bool, device=device
        )
        return torch.triu(ones, diagonal=1)

    def forward(
        self,
        tokens: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Mix ``[B, N, 576]`` tokens; returns the same shape.

        ``key_padding_mask`` is ``[B, N]`` with ``True`` at padded
        positions (the fusion ignores them). With ``output_attentions``,
        also returns a list of per-layer ``[B, H, N, N]`` attention maps.
        """
        attn_mask = self.causal_mask(tokens.size(1), tokens.device)
        attentions = []
        for block in self.blocks:
            tokens, weights = block(
                tokens,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=output_attentions,
            )
            if output_attentions:
                attentions.append(weights)
        tokens = self.norm_out(tokens)
        if output_attentions:
            return tokens, attentions
        return tokens
