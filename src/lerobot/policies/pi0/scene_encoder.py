"""
SceneAnchor — multi-view scene prior encoder for PI0.

MVCNN-style approach: 3D scene → multi-view rendered images → SigLIP features
→ compressed scene tokens in the same embedding space as observation images.

Why multi-view rendering instead of 3D point clouds?
  - Scene tokens and observation image tokens share the **same SigLIP feature space**,
    so PaliGemma's self-attention naturally learns scene ↔ observation associations.
  - No heterogeneous 3D encoder needed — reuses the existing SigLIP vision backbone.
  - Offline SigLIP encoding: zero additional compute at training/inference time.

Pipeline overview:
  Offline:  GLB → render N views → SigLIP encode → (N_views, 256, D) → .pt
  Online:   load .pt → SceneAnchor (compress + view PE) → (B, K, D) scene tokens
                                                              ↓
            prefix_embs = [img_tokens, scene_tokens, lang_tokens]
                                 ↕
            PaliGemma self-attention establishes scene ↔ observation links

Two compression modes:
  - ``pool``:  per-view adaptive pooling + concat → lightweight, no learnable params
               beyond LayerNorm + view PE. Best when ``n_tokens = n_views * pool_k``.
  - ``perceiver``: learnable query tokens cross-attend to all view features →
               produces exactly ``n_tokens`` outputs regardless of view count.
               Heavier but more expressive; recommended when n_views is large.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SceneAnchor(nn.Module):
    """Compresses pre-encoded multi-view SigLIP features into scene tokens.

    Input:  (B, N_views * N_img_tokens, D)  — offline SigLIP features from rendered views.
    Output: (B, n_tokens, D)                — compressed scene tokens for prefix.

    Since the input is already in SigLIP space (same as observation image tokens),
    only a lightweight compression + positional encoding is needed.

    Args:
        embed_dim:    SigLIP / PaliGemma hidden dimension (e.g. 2048).
        n_tokens:     number of output scene tokens (default 64).
        compress_mode: ``"pool"`` or ``"perceiver"``.
        perceiver_depth: number of cross-attention layers (only for perceiver mode).
    """

    def __init__(
        self,
        embed_dim: int,
        n_tokens: int = 64,
        compress_mode: str = "perceiver",
        perceiver_depth: int = 2,
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.compress_mode = compress_mode

        # Input normalization — aligns potential distribution shift between
        # offline-encoded scene features and online observation features.
        self.input_norm = nn.LayerNorm(embed_dim)

        if compress_mode == "pool":
            # Lightweight: adaptive pool → linear
            self.proj = nn.Linear(embed_dim, embed_dim)
        elif compress_mode == "perceiver":
            # Learnable queries cross-attend to multi-view features
            self.query_tokens = nn.Parameter(torch.randn(1, n_tokens, embed_dim) * 0.02)
            self.layers = nn.ModuleList([
                _CrossAttentionBlock(embed_dim) for _ in range(perceiver_depth)
            ])
            self.output_norm = nn.LayerNorm(embed_dim)
        else:
            raise ValueError(f"Unknown compress_mode: {compress_mode!r}")

        # Learnable positional embedding for output scene tokens
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, embed_dim) * 0.02)

    def forward(self, scene_features: torch.Tensor) -> torch.Tensor:
        """Compress multi-view SigLIP features into scene tokens.

        Args:
            scene_features: (B, N_total, D) pre-encoded SigLIP features from
                multiple rendered views, concatenated along the token dimension.
                N_total = n_views * tokens_per_view (e.g. 8 * 256 = 2048).

        Returns:
            (B, n_tokens, D) compressed scene tokens.
        """
        x = self.input_norm(scene_features)  # (B, N_total, D)

        if self.compress_mode == "pool":
            # Adaptive pool along token dim: (B, N_total, D) → (B, n_tokens, D)
            x = x.transpose(1, 2)  # (B, D, N_total)
            x = F.adaptive_avg_pool1d(x, self.n_tokens)  # (B, D, n_tokens)
            x = x.transpose(1, 2)  # (B, n_tokens, D)
            x = self.proj(x)
        else:  # perceiver
            queries = self.query_tokens.expand(x.shape[0], -1, -1)
            for layer in self.layers:
                queries = layer(queries, x)
            x = self.output_norm(queries)

        return x + self.pos_embed


class _CrossAttentionBlock(nn.Module):
    """Cross-attention + FFN block for Perceiver-style compression."""

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, query: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.cross_attn(query, kv, kv)
        x = self.norm1(query + attn_out)
        return self.norm2(x + self.ffn(x))
