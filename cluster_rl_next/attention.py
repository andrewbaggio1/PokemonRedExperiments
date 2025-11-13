from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpatialSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.view(b, c, h * w).transpose(1, 2)  # [B, HW, C]
        tokens = self.norm(tokens)
        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        attn_out = self.dropout(attn_out)
        attn_out = attn_out.transpose(1, 2).reshape(b, c, h, w)
        return x + attn_out


class SpatialTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn = SpatialSelfAttention(dim, num_heads, dropout)
        self.norm = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, hidden_dim=max(dim * 4, 128), dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = self.attn(x)
        tokens = x.view(b, c, h * w).transpose(1, 2)
        tokens = tokens + self.ffn(self.norm(tokens))
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class MobileViTBlock(nn.Module):
    """
    Lightweight attention alternative for when MHSA is too expensive.
    """

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.pw_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(dim)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        x = self.norm(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return residual + x
