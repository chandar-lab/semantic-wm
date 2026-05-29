"""Trajectory success probes and feature extraction utilities."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.model import SwiGLU


@torch.no_grad()
def extract_features(
    frames: torch.Tensor,
    autoencoder,
    adapter=None,
    feature_space: str = "adapter",
) -> torch.Tensor:
    """Encode pixel frames to latent features."""
    z = autoencoder.encode(frames)

    if feature_space == "encoder" or adapter is None:
        return z

    z_adapted = adapter.encode(z)
    if isinstance(z_adapted, tuple):
        z_adapted = z_adapted[0]
    return z_adapted


def _pool_patches(x: torch.Tensor, pool_mode: str) -> torch.Tensor:
    """Pool spatial patch dimensions."""
    if pool_mode == "mean":
        return x.mean(dim=(2, 3))

    if pool_mode == "super_patch_4x4":
        bsz, timesteps, height, width, channels = x.shape
        x = x.reshape(bsz * timesteps, height, width, channels).permute(0, 3, 1, 2)
        x = F.adaptive_avg_pool2d(x, (4, 4))
        return x.permute(0, 2, 3, 1).reshape(bsz, timesteps, 16, channels)

    if pool_mode == "super_patch_8x8":
        bsz, timesteps, height, width, channels = x.shape
        x = x.reshape(bsz * timesteps, height, width, channels).permute(0, 3, 1, 2)
        x = F.adaptive_avg_pool2d(x, (8, 8))
        return x.permute(0, 2, 3, 1).reshape(bsz, timesteps, 64, channels)

    raise ValueError(f"Unknown pool_mode: {pool_mode}")


class ProbeSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.dropout = float(dropout)
        head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, elementwise_affine=False)
        self.k_norm = nn.RMSNorm(head_dim, elementwise_affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(x.shape[0], x.shape[1], self.num_heads, -1).transpose(1, 2)
        k = k.view(x.shape[0], x.shape[1], self.num_heads, -1).transpose(1, 2)
        v = v.view(x.shape[0], x.shape[1], self.num_heads, -1).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
        return self.out_proj(out)


class ProbeCrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.dropout = float(dropout)
        head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, elementwise_affine=False)
        self.k_norm = nn.RMSNorm(head_dim, elementwise_affine=False)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context is None:
            return torch.zeros_like(x)

        context = context.to(dtype=x.dtype)
        q = self.q_proj(x)
        k = self.k_proj(context)
        v = self.v_proj(context)

        q = q.view(x.shape[0], x.shape[1], self.num_heads, -1).transpose(1, 2)
        k = k.view(context.shape[0], context.shape[1], self.num_heads, -1).transpose(
            1, 2
        )
        v = v.view(context.shape[0], context.shape[1], self.num_heads, -1).transpose(
            1, 2
        )
        q = self.q_norm(q)
        k = self.k_norm(k)

        attn_mask = None
        if context_mask is not None:
            safe_mask = context_mask.to(dtype=torch.bool)
            if safe_mask.ndim != 2:
                raise ValueError(
                    f"context_mask must be (B, L), got {tuple(safe_mask.shape)}"
                )
            no_context = ~safe_mask.any(dim=-1)
            if no_context.any():
                safe_mask = safe_mask.clone()
                safe_mask[no_context, 0] = True
                context = context.clone()
                context[no_context, 0] = 0.0
                k = (
                    self.k_proj(context)
                    .view(context.shape[0], context.shape[1], self.num_heads, -1)
                    .transpose(1, 2)
                )
                v = (
                    self.v_proj(context)
                    .view(context.shape[0], context.shape[1], self.num_heads, -1)
                    .transpose(1, 2)
                )
                k = self.k_norm(k)
            attn_mask = safe_mask[:, None, None, :]

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
        return self.out_proj(out)


class ProbeBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        n_patches: int,
        n_frames: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_patches = int(n_patches)
        self.n_frames = int(n_frames)
        self.spatial_norm = nn.RMSNorm(dim, elementwise_affine=False, eps=1e-6)
        self.temporal_norm = nn.RMSNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross_norm = nn.RMSNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ffn_norm = nn.RMSNorm(dim, elementwise_affine=False, eps=1e-6)
        self.spatial_attn = ProbeSelfAttention(dim, num_heads, dropout=dropout)
        self.temporal_attn = ProbeSelfAttention(dim, num_heads, dropout=dropout)
        self.cross_attn = ProbeCrossAttention(dim, num_heads, dropout=dropout)
        self.ffn = SwiGLU(in_features=dim, hidden_features=dim * 4)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        text_tokens: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cls_token = x[:, :1]
        patches = x[:, 1:]
        bsz, _, dim = patches.shape
        patches = patches.reshape(bsz, self.n_frames, self.n_patches, dim)

        spatial = patches.reshape(bsz * self.n_frames, self.n_patches, dim)
        spatial = spatial + self.spatial_attn(self.spatial_norm(spatial))
        patches = spatial.reshape(bsz, self.n_frames, self.n_patches, dim)

        temporal = patches.permute(0, 2, 1, 3).reshape(
            bsz * self.n_patches, self.n_frames, dim
        )
        temporal = temporal + self.temporal_attn(self.temporal_norm(temporal))
        patches = temporal.reshape(bsz, self.n_patches, self.n_frames, dim).permute(
            0, 2, 1, 3
        )

        x = torch.cat(
            [cls_token, patches.reshape(bsz, self.n_frames * self.n_patches, dim)],
            dim=1,
        )
        x = x + self.cross_attn(self.cross_norm(x), text_tokens, context_mask=text_mask)
        x = x + self.ffn_dropout(self.ffn(self.ffn_norm(x)))
        return x


class LinearProbe(nn.Module):
    def __init__(
        self, feature_dim: int, n_frames: int, pool_mode: str = "mean", **_: dict
    ):
        super().__init__()
        self.pool_mode = pool_mode
        if pool_mode == "mean":
            in_dim = n_frames * feature_dim
        elif pool_mode == "super_patch_4x4":
            in_dim = n_frames * 16 * feature_dim
        elif pool_mode == "super_patch_8x8":
            in_dim = n_frames * 64 * feature_dim
        else:
            raise ValueError(f"Unknown pool_mode: {pool_mode}")
        self.classifier = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor, **_: dict) -> torch.Tensor:
        x = _pool_patches(x, self.pool_mode)
        return self.classifier(x.flatten(1)).squeeze(-1)


class TemporalProbe(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        n_frames: int,
        n_heads: int = 8,
        pool_mode: str = "mean",
        **_: dict,
    ):
        super().__init__()
        self.pool_mode = pool_mode

        if pool_mode in ("super_patch_4x4", "super_patch_8x8"):
            n_patches = 16 if pool_mode == "super_patch_4x4" else 64
            self.spatial_proj = nn.Linear(n_patches * feature_dim, feature_dim)
        else:
            self.spatial_proj = None

        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, n_frames + 1, feature_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=n_heads,
            dim_feedforward=feature_dim * 4,
            batch_first=True,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.head = nn.Linear(feature_dim, 1)

    def forward(self, x: torch.Tensor, **_: dict) -> torch.Tensor:
        x = _pool_patches(x, self.pool_mode)

        if self.spatial_proj is not None:
            bsz, timesteps = x.shape[:2]
            x = x.reshape(bsz, timesteps, -1)
            x = self.spatial_proj(x)

        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.transformer(x)
        return self.head(x[:, 0]).squeeze(-1)


class SpatiotemporalProbe(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        n_frames: int,
        n_heads: int = 8,
        pool_mode: str = "super_patch_4x4",
        num_layers: int = 6,
        model_dim: int = 384,
        text_dim: int = 1152,
        text_dropout: float = 0.1,
        dropout: float = 0.1,
    ):
        super().__init__()
        if pool_mode == "super_patch_4x4":
            self.n_patches = 16
        elif pool_mode == "super_patch_8x8":
            self.n_patches = 64
        else:
            raise ValueError(
                f"SpatiotemporalProbe requires super_patch pool_mode, got: {pool_mode}"
            )
        self.pool_mode = pool_mode
        self.n_frames = n_frames
        self.model_dim = model_dim
        self.input_norm = nn.RMSNorm(feature_dim, elementwise_affine=False, eps=1e-6)
        self.input_proj = nn.Linear(feature_dim, model_dim)
        self.text_proj = nn.Linear(text_dim, model_dim)
        self.text_norm = nn.RMSNorm(model_dim, elementwise_affine=False, eps=1e-6)
        self.text_dropout = nn.Dropout(text_dropout)
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim))
        self.temporal_embed = nn.Parameter(torch.randn(1, n_frames, 1, model_dim))
        self.spatial_embed = nn.Parameter(torch.randn(1, 1, self.n_patches, model_dim))
        self.blocks = nn.ModuleList(
            [
                ProbeBlock(
                    dim=model_dim,
                    num_heads=n_heads,
                    n_patches=self.n_patches,
                    n_frames=n_frames,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.RMSNorm(model_dim, elementwise_affine=False, eps=1e-6)
        self.head = nn.Linear(model_dim, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.temporal_embed, std=0.02)
        nn.init.normal_(self.spatial_embed, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        text_tokens: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = _pool_patches(x, self.pool_mode)
        x = self.input_proj(self.input_norm(x))
        x = x + self.temporal_embed + self.spatial_embed
        x = x.reshape(x.shape[0], self.n_frames * self.n_patches, self.model_dim)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)

        if text_tokens is not None:
            text_tokens = self.text_proj(text_tokens.to(dtype=x.dtype))
            text_tokens = self.text_dropout(self.text_norm(text_tokens))

        for block in self.blocks:
            x = block(x, text_tokens=text_tokens, text_mask=text_mask)
        x = self.final_norm(x)
        # The factorized blocks only mix video tokens with other video tokens, so
        # pooling the patch tokens is the stable way to expose visual evidence.
        return self.head(x[:, 1:].mean(dim=1)).squeeze(-1)


class ProgressRegressor(nn.Module):
    def __init__(self, feature_dim: int, pool_mode: str = "mean", **_: dict):
        super().__init__()
        self.pool_mode = pool_mode
        self.head = nn.Linear(feature_dim, 1)

    def forward(self, x: torch.Tensor, **_: dict) -> torch.Tensor:
        x = _pool_patches(x, self.pool_mode)
        if x.ndim == 4:
            x = x.mean(dim=2)
        return self.head(x).squeeze(-1)


def create_probe(
    probe_type: str,
    feature_dim: int,
    n_frames: int,
    pool_mode: str = "mean",
    n_heads: int = 8,
    num_layers: int = 6,
    model_dim: int = 384,
    text_dim: int = 1152,
    text_dropout: float = 0.1,
    dropout: float = 0.1,
) -> nn.Module:
    if probe_type == "linear":
        return LinearProbe(feature_dim, n_frames, pool_mode=pool_mode)
    if probe_type == "temporal":
        return TemporalProbe(
            feature_dim, n_frames, n_heads=n_heads, pool_mode=pool_mode
        )
    if probe_type == "spatiotemporal":
        return SpatiotemporalProbe(
            feature_dim=feature_dim,
            n_frames=n_frames,
            n_heads=n_heads,
            pool_mode=pool_mode,
            num_layers=num_layers,
            model_dim=model_dim,
            text_dim=text_dim,
            text_dropout=text_dropout,
            dropout=dropout,
        )
    if probe_type == "progress":
        return ProgressRegressor(feature_dim, pool_mode=pool_mode)
    raise ValueError(f"Unknown probe_type: {probe_type}")
