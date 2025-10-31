from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class VisionBranchConfig:
    in_channels: int = 4
    conv_channels: Tuple[int, ...] = (32, 64, 128, 128)
    kernel_sizes: Tuple[int, ...] = (5, 3, 3, 3)
    strides: Tuple[int, ...] = (2, 2, 2, 1)
    gru_hidden_size: int = 256
    embedding_dim: int = 256
    attention_dropout: float = 0.1
    linear_dropout: float = 0.1
    activation: str = "silu"


def _get_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation '{name}'")


class SpatialAttentionPooling(nn.Module):
    """Light-weight spatial attention to focus on salient tokens."""

    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(channels, 1, kernel_size=1),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: (B, C, H, W)
        attn_logits = self.project(feats)  # (B, 1, H, W)
        attn_logits = attn_logits.flatten(start_dim=2)  # (B, 1, HW)
        weights = F.softmax(attn_logits, dim=-1)  # (B, 1, HW)

        tokens = feats.flatten(start_dim=2)  # (B, C, HW)
        pooled = torch.bmm(tokens, weights.transpose(1, 2)).squeeze(-1)  # (B, C)
        return pooled


class VisionBranch(nn.Module):
    """CNN → Attention → GRU pipeline that yields visual embeddings."""

    def __init__(self, config: Optional[VisionBranchConfig] = None) -> None:
        super().__init__()
        self.config = config or VisionBranchConfig()

        conv_layers: Iterable[nn.Module] = self._build_conv_stack()
        self.cnn = nn.Sequential(*conv_layers)
        self.attention = SpatialAttentionPooling(
            self.config.conv_channels[-1], dropout=self.config.attention_dropout
        )
        self.pre_gru = nn.Linear(self.config.conv_channels[-1], self.config.gru_hidden_size)
        self.pre_gru_act = _get_activation(self.config.activation)
        self.gru_cell = nn.GRUCell(
            self.config.gru_hidden_size, self.config.gru_hidden_size
        )
        self.post_linear = nn.Sequential(
            nn.Linear(self.config.gru_hidden_size, self.config.embedding_dim),
            _get_activation(self.config.activation),
            nn.Dropout(self.config.linear_dropout),
        )

    def _build_conv_stack(self) -> Iterable[nn.Module]:
        layers = []
        in_ch = self.config.in_channels
        for out_ch, k, stride in zip(
            self.config.conv_channels,
            self.config.kernel_sizes,
            self.config.strides,
        ):
            padding = k // 2
            layers.append(
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=k,
                    stride=stride,
                    padding=padding,
                )
            )
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(_get_activation(self.config.activation))
            in_ch = out_ch
        return layers

    @property
    def embedding_dim(self) -> int:
        return self.config.embedding_dim

    @property
    def hidden_size(self) -> int:
        return self.config.gru_hidden_size

    def forward(
        self,
        frames: torch.Tensor,
        state_in: torch.Tensor,
        seq_lens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            frames: Tensor[N, T, C, H, W] or Tensor[N, C, H, W]
            state_in: Tensor[N, hidden]
            seq_lens: optional Tensor[N] with valid lengths.
        Returns:
            embedding: Tensor[N * T, embedding_dim]
            normalized_embedding: L2-normalized copy of embeddings.
            state_out: Tensor[N, hidden]
        """
        single_step = frames.dim() == 4
        if single_step:
            frames = frames.unsqueeze(1)

        batch, time, channels, height, width = frames.shape
        x = frames.view(batch * time, channels, height, width)
        x = self.cnn(x)
        pooled = self.attention(x)  # (batch*time, channels_last)
        pooled = pooled.view(batch, time, -1)

        gru_inp = self.pre_gru_act(self.pre_gru(pooled))

        if state_in.dim() == 1:
            state_in = state_in.unsqueeze(0)
        if state_in.shape[0] == 1 and batch > 1:
            state = state_in.expand(batch, -1).contiguous()
        elif state_in.shape[0] != batch:
            state = state_in[:batch]
        else:
            state = state_in

        if seq_lens is not None:
            seq_lens = seq_lens.to(gru_inp.device)

        outputs = []
        for t in range(time):
            step_input = gru_inp[:, t, :]
            prev_state = state
            new_state = self.gru_cell(step_input, state)
            if seq_lens is not None:
                mask = (seq_lens > t).float().unsqueeze(-1)
                state = mask * new_state + (1 - mask) * prev_state
            else:
                state = new_state
            outputs.append(state.unsqueeze(1))

        outputs_tensor = torch.cat(outputs, dim=1)  # (batch, time, hidden)
        embeddings = self.post_linear(outputs_tensor)  # (batch, time, emb)
        normalized = F.normalize(embeddings, dim=-1)

        if single_step:
            embeddings = embeddings.squeeze(1)
            normalized = normalized.squeeze(1)
        else:
            embeddings = embeddings.view(batch * time, -1)
            normalized = normalized.view(batch * time, -1)

        return embeddings, normalized, state
