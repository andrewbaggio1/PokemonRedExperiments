from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple

import numpy as np
from skimage.transform import resize


@dataclass
class ActionMaskCacheConfig:
    downsample_shape: Tuple[int, int] = (16, 16)
    mse_threshold: float = 1e-3
    max_history: Optional[int] = None
    grayscale_last_only: bool = True


class ActionMaskCache:
    """Caches invalid actions for states determined via frame similarity."""

    def __init__(self, config: Optional[ActionMaskCacheConfig] = None) -> None:
        self.config = config or ActionMaskCacheConfig()
        self._invalid: Dict[str, Set[int]] = {}

    def reset(self) -> None:
        self._invalid.clear()

    @staticmethod
    def _prepare_frames(frames: np.ndarray) -> np.ndarray:
        arr = frames.astype(np.float32)
        if arr.max() > 1.0:
            arr /= 255.0
        if arr.ndim == 3:
            # allow both HWK and KHW layouts
            if arr.shape[0] <= 4:  # assume channel-first
                arr = np.transpose(arr, (1, 2, 0))
        elif arr.ndim == 2:
            arr = arr[..., None]
        return arr

    def _downsample(self, frames: np.ndarray, last_only: bool = False) -> np.ndarray:
        arr = self._prepare_frames(frames)
        if last_only and arr.shape[-1] > 1:
            arr = arr[..., -1:]
        grayscale = arr.mean(axis=-1)
        scaled = resize(
            grayscale,
            self.config.downsample_shape,
            mode="reflect",
            anti_aliasing=True,
            preserve_range=True,
        ).astype(np.float32)
        return scaled

    def compute_state_key(self, frames: np.ndarray) -> str:
        downsampled = self._downsample(
            frames, last_only=self.config.grayscale_last_only
        )
        quantized = np.clip(np.round(downsampled * 255), 0, 255).astype(np.uint8)
        digest = hashlib.sha1(quantized.tobytes()).hexdigest()
        return digest

    def get_mask(self, state_key: str, num_actions: int) -> np.ndarray:
        mask = np.ones(num_actions, dtype=np.int8)
        invalid = self._invalid.get(state_key)
        if invalid:
            mask[list(invalid)] = 0
        return mask

    def _compute_mse(self, prev_obs: np.ndarray, next_obs: np.ndarray) -> float:
        prev_down = self._downsample(prev_obs, last_only=True)
        next_down = self._downsample(next_obs, last_only=True)
        diff = prev_down - next_down
        mse = float(np.mean(diff * diff))
        return mse

    def record_transition(
        self,
        prev_key: Optional[str],
        action: int,
        prev_obs: np.ndarray,
        next_obs: np.ndarray,
    ) -> None:
        if prev_key is None:
            return

        mse = self._compute_mse(prev_obs, next_obs)
        if mse <= self.config.mse_threshold:
            invalid = self._invalid.setdefault(prev_key, set())
            invalid.add(int(action))
