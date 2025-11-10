from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        scale = self.pool(x).view(b, c)
        scale = self.fc(scale).view(b, c, 1, 1)
        return x * scale


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, reduction: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SqueezeExcite(out_channels, reduction=reduction)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = F.relu(out + identity, inplace=True)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.view(b, c, h * w).transpose(1, 2)
        tokens = self.norm(tokens)
        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        attn_out = self.dropout(attn_out)
        attn_out = attn_out.transpose(1, 2).reshape(b, c, h, w)
        return x + attn_out


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_eps", torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_eps", torch.empty(out_features))
        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_eps.copy_(eps_out.ger(eps_in))
        self.bias_eps.copy_(eps_out)

    def forward(self, input):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_eps
            bias = self.bias_mu + self.bias_sigma * self.bias_eps
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return nn.functional.linear(input, weight, bias)


class ClusterDQN(nn.Module):
    """
    Unified, efficient model:
    - Conv stem + 2-3 residual blocks (+ optional spatial attention)
    - Global pooling; fuse with compact context MLP
    - Single LSTM for temporal memory
    - Dueling quantile head with NoisyLinear
    - Auxiliary head reconstructs context features
    """

    def __init__(
        self,
        obs_shape: Tuple[int, int, int],
        context_dim: int,
        n_actions: int,
        *,
        use_spatial_attention: bool = True,
        lstm_hidden_size: int = 512,
        num_quantiles: int = 51,
    ):
        super().__init__()
        c, h, w = obs_shape
        self.n_actions = int(n_actions)
        self.num_quantiles = int(num_quantiles)

        stem_channels = 96
        self.stem = nn.Sequential(
            nn.Conv2d(c, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, stem_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )
        self.residual = nn.Sequential(
            ResidualBlock(stem_channels, stem_channels),
            ResidualBlock(stem_channels, stem_channels),
        )
        self.use_spatial_attention = bool(use_spatial_attention)
        if self.use_spatial_attention:
            self.spatial_attn = SpatialAttention(stem_channels, num_heads=4, dropout=0.1)
        self.post_attn = nn.Sequential(
            nn.Conv2d(stem_channels, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dropout = nn.Dropout(p=0.1)

        self.context_net = nn.Sequential(
            nn.Linear(context_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )

        fused_dim = 256 + 128
        self.lstm_hidden_size = int(lstm_hidden_size)
        self.lstm = nn.LSTM(fused_dim, self.lstm_hidden_size, batch_first=True)

        duel_hidden = 384
        self.advantage = nn.Sequential(
            NoisyLinear(self.lstm_hidden_size, duel_hidden),
            nn.ReLU(),
            NoisyLinear(duel_hidden, n_actions * num_quantiles),
        )
        self.value = nn.Sequential(
            NoisyLinear(self.lstm_hidden_size, duel_hidden),
            nn.ReLU(),
            NoisyLinear(duel_hidden, num_quantiles),
        )
        self.aux_head = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, context_dim),
        )

    def init_hidden(self, batch_size: int, device: torch.device):
        h0 = torch.zeros(1, batch_size, self.lstm_hidden_size, device=device)
        c0 = torch.zeros(1, batch_size, self.lstm_hidden_size, device=device)
        return (h0, c0)

    def forward(self, obs, context, hidden=None):
        batch = obs.size(0)
        if hidden is None:
            hidden = self.init_hidden(batch, obs.device)

        features = self.stem(obs)
        features = self.residual(features)
        if self.use_spatial_attention:
            features = self.spatial_attn(features)
        features = self.post_attn(features)
        features = self.global_pool(features).view(batch, -1)
        features = self.feature_dropout(features)

        ctx = self.context_net(context)
        fused = torch.cat([features, ctx], dim=1).unsqueeze(1)

        lstm_out, (h1, c1) = self.lstm(fused, hidden)
        output = lstm_out.squeeze(1)
        next_hidden = (h1, c1)

        adv = self.advantage(output).view(batch, self.n_actions, self.num_quantiles)
        val = self.value(output).view(batch, 1, self.num_quantiles)
        q_quantiles = val + adv - adv.mean(dim=1, keepdim=True)
        aux_pred = self.aux_head(output)
        return q_quantiles, next_hidden, aux_pred

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


