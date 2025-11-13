from __future__ import annotations

import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .config import ClusterNextConfig, ContextConfig
from .heads import AuxiliaryHeads, DuelingQuantileHead
from .temporal import TemporalFusionCore
from .vision import DualStreamEncoder


class ContextFusion(nn.Module):
    def __init__(self, context_dim: int, cfg: ContextConfig, film_channels: int):
        super().__init__()
        self.context_dim = context_dim
        self.cfg = cfg
        self.static_mlp = nn.Sequential(
            nn.Linear(context_dim, cfg.mlp_hidden),
            nn.LayerNorm(cfg.mlp_hidden),
            nn.GELU(),
            nn.Linear(cfg.mlp_hidden, cfg.static_dim),
            nn.GELU(),
        )
        self.event_mlp = nn.Sequential(
            nn.Linear(context_dim, cfg.mlp_hidden),
            nn.LayerNorm(cfg.mlp_hidden),
            nn.GELU(),
            nn.Linear(cfg.mlp_hidden, cfg.event_dim),
            nn.GELU(),
        )
        self.use_film = bool(cfg.use_film)
        if self.use_film:
            self.film_gamma = nn.Linear(cfg.event_dim, film_channels)
            self.film_beta = nn.Linear(cfg.event_dim, film_channels)
        self.output_dim = cfg.static_dim + cfg.event_dim

    def forward(self, context: torch.Tensor) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        static_feat = self.static_mlp(context)
        event_feat = self.event_mlp(context)
        film = None
        if self.use_film:
            gamma = self.film_gamma(event_feat)
            beta = self.film_beta(event_feat)
            film = (gamma, beta)
        fused_context = torch.cat([static_feat, event_feat], dim=1)
        return fused_context, film


def _apply_legacy_overrides(cfg: ClusterNextConfig, overrides: dict) -> ClusterNextConfig:
    updated = copy.deepcopy(cfg)
    if "use_spatial_attention" in overrides:
        updated.vision.attention_depth = 2 if overrides["use_spatial_attention"] else 0
    if "lstm_hidden_size" in overrides:
        updated.temporal.lstm_hidden = int(overrides["lstm_hidden_size"])
    if "num_quantiles" in overrides:
        updated.heads.num_quantiles = int(overrides["num_quantiles"])
    return updated


class ClusterNextModel(nn.Module):
    def __init__(
        self,
        obs_shape: Tuple[int, int, int],
        context_dim: int,
        n_actions: int,
        cfg: Optional[ClusterNextConfig] = None,
        **legacy_overrides,
    ):
        super().__init__()
        base_cfg = cfg or ClusterNextConfig()
        if legacy_overrides:
            base_cfg = _apply_legacy_overrides(base_cfg, legacy_overrides)
        self.cfg = base_cfg
        self.vision = DualStreamEncoder(obs_shape, self.cfg.vision)
        self.context_fuser = ContextFusion(context_dim, self.cfg.context, self.vision.film_channels)
        fused_dim = self.vision.feature_dim + self.context_fuser.output_dim
        self.temporal = TemporalFusionCore(fused_dim, self.cfg.temporal)
        self.dueling = DuelingQuantileHead(self.temporal.latent_dim, n_actions, self.cfg.heads)
        self.aux = AuxiliaryHeads(self.temporal.latent_dim, context_dim, n_actions, self.cfg.heads.aux)

    def init_hidden(self, batch_size: int, device: torch.device):
        return self.temporal.init_hidden(batch_size, device)

    def forward(self, obs: torch.Tensor, context: torch.Tensor, hidden=None):
        batch = obs.size(0)
        if hidden is None:
            hidden = self.init_hidden(batch, obs.device)
        ctx_embed, film = self.context_fuser(context)
        vision_feat = self.vision(obs, film)
        fused = torch.cat([vision_feat, ctx_embed], dim=1)
        latent, next_hidden = self.temporal(fused, hidden)
        q_quantiles = self.dueling(latent)
        aux = self.aux(latent)
        return q_quantiles, next_hidden, aux

    def reset_noise(self):
        self.dueling.reset_noise()


class ClusterDQN(ClusterNextModel):
    """
    Compatibility alias for existing training code.
    """
