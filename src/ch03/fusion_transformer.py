"""Fusion transformer: the one component built from scratch.

The vision encoder and language backbone are inherited pre-trained. The
fusion transformer is the part the reader builds: a stack of pre-norm
self-attention blocks that mixes image, language, and state tokens into
a single contextualized sequence.

Attention is causal (each token attends only to itself and earlier
positions). That choice is deliberate. Chapter 4 appends action tokens
to the right end of the sequence and trains them to be predicted one at
a time, left to right; a causal mask here means no architecture surgery
later.
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
    """A causal pre-norm transformer over the fused token sequence."""

    def __init__(
        self,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
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
        """Mix ``[B, N, 512]`` tokens; returns the same shape.

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
