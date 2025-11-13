from __future__ import annotations

import os
import time
from typing import Optional, Tuple

from .config import ClusterConfig
from .utils import ensure_dir


class EmulatorManager:
    """
    Handles savestate persistence and flatline detection for long-lived actor runs.
    """

    def __init__(self, cfg: ClusterConfig, actor_id: int):
        self.cfg = cfg
        self.actor_id = actor_id
        self.snapshot_dir = os.path.join(cfg.run_dir, cfg.snapshots_dir)
        ensure_dir(self.snapshot_dir)
        try:
            ddp_rank = int(os.environ.get("RANK", "0"))
        except Exception:
            ddp_rank = 0
        self.rank_prefix = f"r{ddp_rank}_"
        self.latest_path = os.path.join(self.snapshot_dir, f"{self.rank_prefix}actor{actor_id}_latest.state")
        self.flatline_threshold = max(1, int(cfg.emulator.flatline_steps_threshold))
        self.min_delta_pos = int(cfg.emulator.min_delta_pos)
        self.min_delta_reward = float(cfg.emulator.min_delta_reward)
        self._last_progress: Optional[Tuple[int, int, int]] = None
        self._last_reward: float = 0.0
        self._stalled_steps: int = 0

    def resume_state_path(self) -> Optional[str]:
        return self.latest_path if os.path.exists(self.latest_path) else None

    def write_snapshot_bytes(self, data: bytes) -> None:
        ts = int(time.time())
        tmp = os.path.join(self.snapshot_dir, f"{self.rank_prefix}actor{self.actor_id}_{ts}.tmp")
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, self.latest_path)

    def update_progress(self, position: Tuple[int, int, int], reward_delta: float) -> bool:
        """
        Track agent movement/reward. Returns True if the actor is deemed stuck.
        """
        if self._last_progress is None:
            self._last_progress = position
            self._last_reward = reward_delta
            self._stalled_steps = 0
            return False
        dx = abs(position[0] - self._last_progress[0])
        dy = abs(position[1] - self._last_progress[1])
        same_area = position[2] == self._last_progress[2]
        reward_change = abs(reward_delta - self._last_reward)
        moved = (dx + dy) >= self.min_delta_pos or not same_area
        rewarded = reward_change >= self.min_delta_reward
        if moved or rewarded:
            self._stalled_steps = 0
            self._last_progress = position
            self._last_reward = reward_delta
            return False
        self._stalled_steps += 1
        return self._stalled_steps >= self.flatline_threshold
