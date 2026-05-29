"""Patch-token transformer IDM for action recovery from encoder-latent chunks.

The IDM takes a chunk of ``k+1`` encoder-latent frames spanning ``k`` action
steps and predicts the GT action chunk of length ``k``. It operates directly
on the native (B, T, H, W, C) grid produced by the autoencoder's ``encode``
— no pooling. A learned per-step CLS readout token aggregates spatial evidence
through self-attention and decodes to the action chunk.

The same architecture is used for every encoder; only the input linear
projection differs in input channel dim.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_IDM_ACTION_DIM = 7  # pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, gripper


class _SDPASelfAttention(nn.Module):
    """Self-attention using ``F.scaled_dot_product_attention`` (FlashAttention-capable)."""

    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = float(dropout)
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        qkv = self.qkv_proj(x).reshape(b, t, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(b, t, d)
        return self.out_proj(out)


class _IDMBlock(nn.Module):
    """Pre-norm transformer block with SDPA self-attention and GELU MLP."""

    def __init__(
        self, dim: int, num_heads: int, mlp_ratio: int, dropout: float
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _SDPASelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerIDM(nn.Module):
    """Patch-token transformer IDM with per-step CLS action-readout queries.

    Parameters
    ----------
    feature_dim : int
        Encoder latent channel dim (C).
    spatial_h : int
        Latent grid height (typically 16, or 8 for VAE w/ patch_size=2).
    spatial_w : int
        Latent grid width.
    horizon : int
        Action chunk length k. The model expects ``k+1`` latent frames as
        input and predicts ``k`` actions.
    action_dim : int
        Number of action dims to predict (default 7).
    model_dim : int
        Internal transformer width.
    num_layers : int
        Number of transformer encoder layers.
    num_heads : int
        Attention heads.
    dropout : float
    max_horizon : int, optional
        Upper bound on temporal positional embedding length. Must be ``>= horizon+1``.
        Defaults to ``horizon+1``.
    """

    def __init__(
        self,
        feature_dim: int,
        spatial_h: int,
        spatial_w: int,
        horizon: int,
        action_dim: int = _IDM_ACTION_DIM,
        model_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_horizon: Optional[int] = None,
        patch_size: int = 1,
    ) -> None:
        super().__init__()
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if model_dim % num_heads != 0:
            raise ValueError(
                f"model_dim={model_dim} must be divisible by num_heads={num_heads}"
            )
        if patch_size < 1:
            raise ValueError(f"patch_size must be >= 1, got {patch_size}")
        if spatial_h % patch_size != 0 or spatial_w % patch_size != 0:
            raise ValueError(
                f"spatial grid ({spatial_h}, {spatial_w}) must be divisible by patch_size={patch_size}"
            )

        if max_horizon is None:
            max_horizon = horizon
        if max_horizon < horizon:
            raise ValueError(f"max_horizon={max_horizon} must be >= horizon={horizon}")

        self.feature_dim = int(feature_dim)
        self.spatial_h = int(spatial_h)
        self.spatial_w = int(spatial_w)
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.model_dim = int(model_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.max_horizon = int(max_horizon)
        self.patch_size = int(patch_size)

        self.grid_h = self.spatial_h // self.patch_size
        self.grid_w = self.spatial_w // self.patch_size
        n_patches = self.grid_h * self.grid_w
        proj_in = feature_dim * self.patch_size * self.patch_size

        self.input_norm = nn.RMSNorm(proj_in, elementwise_affine=False, eps=1e-6)
        self.input_proj = nn.Linear(proj_in, model_dim)
        self.temporal_pos_emb = nn.Parameter(torch.zeros(max_horizon + 1, model_dim))
        self.spatial_pos_emb = nn.Parameter(torch.zeros(n_patches, model_dim))
        self.cls_queries = nn.Parameter(torch.zeros(horizon, model_dim))

        self.transformer = nn.ModuleList(
            [
                _IDMBlock(
                    dim=model_dim,
                    num_heads=num_heads,
                    mlp_ratio=4,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(model_dim)
        self.head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, action_dim),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.temporal_pos_emb, std=0.02)
        nn.init.normal_(self.spatial_pos_emb, std=0.02)
        nn.init.normal_(self.cls_queries, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict an action chunk from ``k+1`` latent frames.

        Parameters
        ----------
        x : (B, k+1, H, W, C) encoder-latent grid spanning ``k`` transitions.

        Returns
        -------
        (B, k, action_dim)
        """
        if x.dim() != 5:
            raise ValueError(f"Expected (B, T, H, W, C) got {tuple(x.shape)}")
        b, t, h, w, c = x.shape
        k = self.horizon
        if t != k + 1:
            raise ValueError(
                f"Expected T={k+1} latent frames for horizon={k}, got T={t}"
            )
        if h != self.spatial_h or w != self.spatial_w:
            raise ValueError(
                f"Expected spatial grid ({self.spatial_h}, {self.spatial_w}), got ({h}, {w})"
            )
        if c != self.feature_dim:
            raise ValueError(f"Expected feature_dim={self.feature_dim}, got {c}")

        p = self.patch_size
        gh, gw = self.grid_h, self.grid_w
        # Patchify spatial grid: (B, T, H, W, C) -> (B, T, gh, gw, p*p*C)
        if p > 1:
            tokens = x.reshape(b, t, gh, p, gw, p, c)
            tokens = tokens.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
            tokens = tokens.reshape(b, t, gh * gw, p * p * c)
        else:
            tokens = x.reshape(b, t, gh * gw, c)
        tokens = self.input_proj(self.input_norm(tokens))

        # Add positional embeddings
        temporal = self.temporal_pos_emb[:t].view(1, t, 1, self.model_dim)
        spatial = self.spatial_pos_emb.view(1, 1, gh * gw, self.model_dim)
        tokens = tokens + temporal + spatial
        tokens = tokens.reshape(b, t * gh * gw, self.model_dim)

        # Prepend k CLS queries
        queries = self.cls_queries.view(1, k, self.model_dim).expand(b, -1, -1)
        seq = torch.cat([queries, tokens], dim=1)

        for block in self.transformer:
            seq = block(seq)
        cls_out = self.final_norm(seq[:, :k])
        return self.head(cls_out)


def create_idm(
    feature_dim: int,
    spatial_h: int,
    spatial_w: int,
    horizon: int,
    action_dim: int = _IDM_ACTION_DIM,
    model_dim: int = 512,
    num_layers: int = 6,
    num_heads: int = 8,
    dropout: float = 0.1,
    max_horizon: Optional[int] = None,
    patch_size: int = 1,
) -> TransformerIDM:
    """Factory for the patch-token Transformer IDM."""
    return TransformerIDM(
        feature_dim=feature_dim,
        spatial_h=spatial_h,
        spatial_w=spatial_w,
        horizon=horizon,
        action_dim=action_dim,
        model_dim=model_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        max_horizon=max_horizon,
        patch_size=patch_size,
    )
