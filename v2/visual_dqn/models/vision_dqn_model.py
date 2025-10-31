from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from gymnasium import spaces
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override

from .vision_branch import VisionBranch, VisionBranchConfig


def _get_space_shape(space: spaces.Space) -> Tuple[int, ...]:
    if isinstance(space, spaces.Box):
        return space.shape
    raise ValueError(f"Unsupported space type for vision branch: {space}")


class VisionDQNModel(TorchModelV2, torch.nn.Module):
    """RLlib Torch model that applies a VisionBranch and masked DQN head."""

    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        num_outputs: int,
        model_config: Dict,
        name: str,
        **kwargs,
    ) -> None:
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        torch.nn.Module.__init__(self)

        if hasattr(obs_space, "original_space") and isinstance(
            obs_space.original_space, spaces.Dict
        ):
            obs_space = obs_space.original_space
        if hasattr(action_space, "original_space"):
            action_space = action_space.original_space

        assert isinstance(
            action_space, spaces.Discrete
        ), "VisionDQNModel currently supports discrete action spaces."
        self.action_dim = action_space.n

        custom_config = model_config.get("custom_model_config", {})
        if not isinstance(obs_space, spaces.Dict):
            raise TypeError(
                "VisionDQNModel expects Dict observation space with keys 'obs' and 'action_mask'."
            )
        obs_shape = _get_space_shape(obs_space.spaces["obs"])
        in_channels = obs_shape[0]
        branch_cfg = VisionBranchConfig(
            in_channels=in_channels,
            conv_channels=tuple(custom_config.get("conv_channels", (32, 64, 128, 128))),
            kernel_sizes=tuple(custom_config.get("kernel_sizes", (5, 3, 3, 3))),
            strides=tuple(custom_config.get("strides", (2, 2, 2, 1))),
            gru_hidden_size=custom_config.get("gru_hidden_size", 256),
            embedding_dim=custom_config.get("embedding_dim", 256),
            attention_dropout=custom_config.get("attention_dropout", 0.1),
            linear_dropout=custom_config.get("linear_dropout", 0.1),
            activation=custom_config.get("activation", "silu"),
        )

        self.vision_branch = VisionBranch(branch_cfg)
        q_hidden_dims: Sequence[int] = custom_config.get("q_hidden_dims", (256, 256))
        q_dropout = custom_config.get("q_dropout", 0.1)

        q_layers: List[torch.nn.Module] = []
        dim_in = self.vision_branch.config.embedding_dim
        for hidden_dim in q_hidden_dims:
            q_layers.append(torch.nn.Linear(dim_in, hidden_dim))
            q_layers.append(torch.nn.SiLU(inplace=True))
            if q_dropout > 0:
                q_layers.append(torch.nn.Dropout(q_dropout))
            dim_in = hidden_dim
        q_layers.append(torch.nn.Linear(dim_in, self.action_dim))
        self.q_head = torch.nn.Sequential(*q_layers)

        self._value_out = torch.zeros(1)
        self._last_normalized: Optional[torch.Tensor] = None
        self.time_major = model_config.get("_time_major", False)

    @property
    def embedding_dim(self) -> int:
        return self.vision_branch.embedding_dim

    @override(ModelV2)
    def get_initial_state(self) -> List[torch.Tensor]:
        return []

    @override(ModelV2)
    def forward(
        self,
        input_dict: Dict[str, torch.Tensor],
        state: List[torch.Tensor],
        seq_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        obs = input_dict["obs"]
        frames = obs["obs"]
        mask = obs.get("action_mask")

        frames = frames.float()
        if frames.max() > 1.0:
            frames = frames / 255.0

        if frames.dim() == 4:
            frames = frames.unsqueeze(1)
        elif frames.dim() == 5:
            pass
        else:
            raise ValueError(f"Unexpected frame tensor shape: {frames.shape}")

        time_steps = frames.shape[1]

        if seq_lens is None:
            seq_lens = torch.ones(
                frames.shape[0], dtype=torch.int32, device=frames.device
            )
        else:
            seq_lens = seq_lens.to(device=frames.device, dtype=torch.int32)

        if self.time_major:
            frames = frames.transpose(0, 1)
            time_steps = frames.shape[0]

        hidden_state = torch.zeros(
            frames.shape[0], self.vision_branch.hidden_size, device=frames.device
        )

        embeddings, normalized, state_out = self.vision_branch(
            frames, hidden_state, seq_lens
        )
        self._last_normalized = normalized.detach()

        q_values = self.q_head(embeddings)

        if mask is not None:
            mask = mask.float()
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).expand(-1, time_steps, -1)
            elif mask.dim() == 3:
                if mask.shape[1] != time_steps:
                    raise ValueError(
                        f"Action mask time dimension {mask.shape[1]} does not match frames {time_steps}"
                    )
            else:
                raise ValueError(f"Unexpected action mask shape: {mask.shape}")
            mask_reshaped = mask.reshape(q_values.shape[0], -1)
            invalid = mask_reshaped < 0.5
            q_values = q_values.masked_fill(invalid, -1e9)

        self._value_out = torch.zeros(
            q_values.shape[0], device=q_values.device, dtype=q_values.dtype
        )

        return q_values, []

    @override(ModelV2)
    def value_function(self) -> torch.Tensor:
        return self._value_out

    @torch.no_grad()
    def encode_observations(
        self,
        frames: np.ndarray,
        state: Optional[np.ndarray] = None,
        normalize: bool = True,
    ) -> np.ndarray:
        device = next(self.parameters()).device
        tensor = torch.as_tensor(frames, dtype=torch.float32, device=device)
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() != 4:
            raise ValueError("Expected frames with shape (batch, C, H, W)")
        if tensor.max().item() > 1.0:
            tensor = tensor / 255.0
        tensor = tensor.unsqueeze(1)  # add time dimension

        if state is None:
            state_tensor = torch.zeros(
                tensor.shape[0], self.vision_branch.hidden_size, device=device
            )
        else:
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)

        embeddings, normalized, _ = self.vision_branch(tensor, state_tensor)
        selected = normalized if normalize else embeddings
        return selected.detach().cpu().numpy()

    def get_last_normalized_embedding(self) -> Optional[torch.Tensor]:
        return self._last_normalized
