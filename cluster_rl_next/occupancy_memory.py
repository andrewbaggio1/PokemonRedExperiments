from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np


@dataclass
class OccupancyConfig:
    map_size: int = 256
    crop_radius: int = 10  # agent-centered crop half-size (crop is (2r+1)^2)


class ClusterEnv(gym.Env):
    """
    Wraps the PokemonRedEnv to add:
    - Persistent per-map occupancy counts
    - Discovered saliency marks (doors/stairs inferred at map transitions)
    - Agent-centered occupancy+saliency crop in info
    - Episode savestate bytes for deterministic replay (captured on reset)
    """

    metadata = {"render_modes": []}

    def __init__(self, base_env: gym.Env, *, cfg: Optional[OccupancyConfig] = None) -> None:
        super().__init__()
        self.env = base_env
        self.cfg = cfg or OccupancyConfig()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self._map_occ: Dict[int, np.ndarray] = {}
        self._map_sal: Dict[int, np.ndarray] = {}
        self._last_map: Optional[int] = None
        self._last_coords: Optional[Tuple[int, int]] = None
        self._episode_savestate: Optional[bytes] = None

    @property
    def pyboy(self):
        env = self.env
        while env is not None:
            if hasattr(env, "pyboy"):
                return getattr(env, "pyboy")
            env = getattr(env, "env", None)
        raise AttributeError("Underlying environment does not expose a PyBoy instance.")

    def _ensure_maps(self, map_id: int) -> None:
        s = self.cfg.map_size
        if map_id not in self._map_occ:
            self._map_occ[map_id] = np.zeros((s, s), dtype=np.float32)
        if map_id not in self._map_sal:
            self._map_sal[map_id] = np.zeros((s, s), dtype=np.float32)

    def _update_maps(self, info: Dict) -> None:
        map_id = int(info.get("map_id") or 0)
        coords = info.get("agent_coords") or (0, 0)
        x, y = int(coords[0]), int(coords[1])
        self._ensure_maps(map_id)
        occ = self._map_occ[map_id]
        if 0 <= x < occ.shape[1] and 0 <= y < occ.shape[0]:
            occ[y, x] += 1.0
        # Detect map transitions and mark door/stairs saliency at previous map
        if self._last_map is not None and map_id != self._last_map and self._last_coords is not None:
            px, py = self._last_coords
            self._ensure_maps(self._last_map)
            sal = self._map_sal[self._last_map]
            if 0 <= px < sal.shape[1] and 0 <= py < sal.shape[0]:
                sal[py, px] = max(sal[py, px], 1.0)
        self._last_map = map_id
        self._last_coords = (x, y)

    def _crop_centered(self, array: np.ndarray, center: Tuple[int, int], radius: int) -> np.ndarray:
        cx, cy = int(center[0]), int(center[1])
        y0, y1 = cy - radius, cy + radius + 1
        x0, x1 = cx - radius, cx + radius + 1
        # Pad edges
        pad_top = max(0, -y0)
        pad_left = max(0, -x0)
        pad_bottom = max(0, y1 - array.shape[0])
        pad_right = max(0, x1 - array.shape[1])
        if any(v > 0 for v in (pad_top, pad_bottom, pad_left, pad_right)):
            array = np.pad(array, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant")
            y0 += pad_top
            y1 += pad_top
            x0 += pad_left
            x1 += pad_left
        return array[y0:y1, x0:x1]

    def _attach_occupancy_info(self, info: Dict) -> Dict:
        map_id = int(info.get("map_id") or 0)
        coords = info.get("agent_coords") or (0, 0)
        self._ensure_maps(map_id)
        occ = self._map_occ[map_id]
        sal = self._map_sal[map_id]
        crop_r = max(0, int(self.cfg.crop_radius))
        occ_crop = self._crop_centered(occ, (int(coords[0]), int(coords[1])), crop_r)
        sal_crop = self._crop_centered(sal, (int(coords[0]), int(coords[1])), crop_r)
        info = dict(info)
        info["occupancy_total_unique"] = float(np.count_nonzero(occ))
        info["occupancy_steps_sum"] = float(np.sum(occ))
        info["saliency_total_marks"] = float(np.count_nonzero(sal))
        info["occupancy_crop"] = occ_crop.astype(np.float32)
        info["saliency_crop"] = sal_crop.astype(np.float32)
        info["occupancy_crop_radius"] = crop_r
        return info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Capture savestate bytes for replay
        try:
            from io import BytesIO
            buf = BytesIO()
            self.pyboy.save_state(buf)  # type: ignore[attr-defined]
            self._episode_savestate = buf.getvalue()
        except Exception:
            self._episode_savestate = None
        self._last_map = None
        self._last_coords = None
        self._update_maps(info)
        info = self._attach_occupancy_info(info)
        info["episode_savestate"] = self._episode_savestate
        return obs, info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._update_maps(info)
        info = self._attach_occupancy_info(info)
        return obs, reward, terminated, truncated, info

    def close(self):
        self.env.close()


