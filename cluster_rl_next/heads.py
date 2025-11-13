from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AuxHeadConfig, HeadsConfig


class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
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
        bound = 1.0 / (self.in_features ** 0.5)
        self.weight_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.sigma_init / (self.in_features ** 0.5))
        self.bias_mu.data.uniform_(-bound, bound)
        self.bias_sigma.data.fill_(self.sigma_init / (self.out_features ** 0.5))

    def _scale_noise(self, size: int):
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_eps.copy_(torch.outer(eps_out, eps_in))
        self.bias_eps.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_eps
            bias = self.bias_mu + self.bias_sigma * self.bias_eps
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class DuelingQuantileHead(nn.Module):
    def __init__(self, latent_dim: int, n_actions: int, cfg: HeadsConfig):
        super().__init__()
        duel_hidden = cfg.dueling.hidden
        self.n_actions = int(n_actions)
        self.num_quantiles = int(cfg.num_quantiles)
        self.advantage = nn.Sequential(
            NoisyLinear(latent_dim, duel_hidden),
            nn.GELU(),
            NoisyLinear(duel_hidden, self.n_actions * self.num_quantiles),
        )
        self.value = nn.Sequential(
            NoisyLinear(latent_dim, duel_hidden),
            nn.GELU(),
            NoisyLinear(duel_hidden, self.num_quantiles),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        batch = latent.size(0)
        adv = self.advantage(latent).view(batch, self.n_actions, self.num_quantiles)
        val = self.value(latent).view(batch, 1, self.num_quantiles)
        q = val + adv - adv.mean(dim=1, keepdim=True)
        return q

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class AuxiliaryHeads(nn.Module):
    def __init__(self, latent_dim: int, context_dim: int, n_actions: int, cfg: AuxHeadConfig):
        super().__init__()
        self.cfg = cfg
        self.context_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Linear(latent_dim // 2, context_dim),
        )
        self.inverse_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Linear(latent_dim // 2, n_actions),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Linear(latent_dim // 2, 1),
        )

    def forward(self, latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "context_recon": self.context_head(latent),
            "inverse_dynamics": self.inverse_head(latent),
            "reward_forecast": self.reward_head(latent),
        }
