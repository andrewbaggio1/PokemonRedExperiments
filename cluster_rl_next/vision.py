from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MobileViTBlock, SpatialTransformerBlock
from .config import VisionConfig


def _group_norm(channels: int, groups: int = 32) -> nn.GroupNorm:
    g = min(groups, channels)
    while channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, channels)


class PreActResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, se_reduction: int = 16):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.GELU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.skip = None
        squeeze_hidden = max(out_channels // se_reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, squeeze_hidden, 1),
            nn.GELU(),
            nn.Conv2d(squeeze_hidden, out_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = out * self.se(out)
        if self.skip is not None:
            identity = self.skip(identity)
        return out + identity


class DualStreamEncoder(nn.Module):
    """
    Two-path encoder: hi-res branch keeps spatial acuity, low-res captures layout.
    """

    def __init__(self, obs_shape: Tuple[int, int, int], cfg: VisionConfig):
        super().__init__()
        in_channels = obs_shape[0]
        self.cfg = cfg
        self.hi_stem = nn.Sequential(
            nn.Conv2d(in_channels, cfg.hi_res_channels, kernel_size=3, stride=1, padding=1, bias=False),
            _group_norm(cfg.hi_res_channels),
            nn.GELU(),
        )
        self.lo_stem = nn.Sequential(
            nn.Conv2d(in_channels, cfg.lo_res_channels, kernel_size=5, stride=2, padding=2, bias=False),
            _group_norm(cfg.lo_res_channels),
            nn.GELU(),
        )
        hi_blocks = []
        for _ in range(cfg.res_blocks_hi):
            hi_blocks.append(PreActResBlock(cfg.hi_res_channels, cfg.hi_res_channels, se_reduction=cfg.se_reduction))
        self.hi_blocks = nn.Sequential(*hi_blocks)

        lo_blocks = []
        for idx in range(cfg.res_blocks_lo):
            stride = 2 if idx == 0 else 1
            in_c = cfg.lo_res_channels if idx == 0 else cfg.lo_res_channels
            lo_blocks.append(PreActResBlock(in_c, cfg.lo_res_channels, stride=stride, se_reduction=cfg.se_reduction))
        self.lo_blocks = nn.Sequential(*lo_blocks)

        fusion_channels = cfg.hi_res_channels + cfg.lo_res_channels
        self.fusion_proj = nn.Sequential(
            nn.Conv2d(fusion_channels, fusion_channels, kernel_size=1, bias=False),
            _group_norm(fusion_channels),
            nn.GELU(),
        )
        attn_blocks = []
        for _ in range(max(0, cfg.attention_depth)):
            if cfg.attention_type == "mobilevit":
                attn_blocks.append(MobileViTBlock(fusion_channels, cfg.attention_dropout))
            else:
                attn_blocks.append(SpatialTransformerBlock(fusion_channels, cfg.attention_heads, cfg.attention_dropout))
        self.attn_blocks = nn.Sequential(*attn_blocks)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(cfg.feature_dropout)
        self.feature_dim = fusion_channels
        self.film_channels = fusion_channels

    def forward(
        self,
        obs: torch.Tensor,
        film_params: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        hi = self.hi_stem(obs)
        hi = self.hi_blocks(hi)
        lo = self.lo_stem(obs)
        lo = self.lo_blocks(lo)
        lo = F.interpolate(lo, size=hi.shape[-2:], mode="bilinear", align_corners=False)
        fused = torch.cat([hi, lo], dim=1)
        if film_params is not None:
            gamma, beta = film_params
            fused = fused * (1 + gamma.view(gamma.size(0), -1, 1, 1)) + beta.view(beta.size(0), -1, 1, 1)
        fused = self.fusion_proj(fused)
        fused = self.attn_blocks(fused)
        pooled = self.global_pool(fused).flatten(1)
        return self.dropout(pooled)
