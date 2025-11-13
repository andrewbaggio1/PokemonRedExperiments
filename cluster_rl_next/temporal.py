from __future__ import annotations

import torch
import torch.nn as nn

from .config import TemporalConfig


class TemporalFusionCore(nn.Module):
    """
    Combines GRU, LSTM, and a short Transformer encoder over the fused embeddings.
    Only the LSTM carries explicit hidden state to keep API compatibility with actors.
    """

    def __init__(self, input_dim: int, cfg: TemporalConfig):
        super().__init__()
        self.cfg = cfg
        self.gru = nn.GRU(input_dim, cfg.gru_hidden, batch_first=True)
        self.lstm = nn.LSTM(input_dim, cfg.lstm_hidden, batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.transformer.d_model,
            nhead=cfg.transformer.nhead,
            dim_feedforward=cfg.transformer.dim_feedforward,
            dropout=cfg.transformer.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.transformer.depth)
        self.transformer_in = nn.Linear(cfg.gru_hidden + cfg.lstm_hidden, cfg.transformer.d_model)
        self.transformer_out = nn.Linear(cfg.transformer.d_model, cfg.transformer.d_model)
        concat_dim = cfg.gru_hidden + cfg.lstm_hidden + cfg.transformer.d_model
        self.latent_proj = nn.Sequential(
            nn.Linear(concat_dim, cfg.latent_target_dim),
            nn.LayerNorm(cfg.latent_target_dim),
            nn.GELU(),
        )

    @property
    def latent_dim(self) -> int:
        return self.cfg.latent_target_dim

    def init_hidden(self, batch_size: int, device: torch.device):
        h0 = torch.zeros(1, batch_size, self.cfg.lstm_hidden, device=device)
        c0 = torch.zeros(1, batch_size, self.cfg.lstm_hidden, device=device)
        return (h0, c0)

    def forward(self, fused: torch.Tensor, hidden):
        seq = fused.unsqueeze(1)  # [B,1,D]
        gru_out, _ = self.gru(seq)
        lstm_out, next_hidden = self.lstm(seq, hidden)
        transformer_in = self.transformer_in(torch.cat([gru_out, lstm_out], dim=2))
        trans_out = self.transformer(transformer_in)
        trans_out = self.transformer_out(trans_out)
        concat = torch.cat([gru_out, lstm_out, trans_out], dim=2).squeeze(1)
        latent = self.latent_proj(concat)
        return latent, next_hidden
