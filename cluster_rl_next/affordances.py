from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .utils import obs_token_hash


Action = int


@dataclass
class AffordanceStats:
    # Track per-action failure counts for a given UI signature
    failures: Dict[Action, int] = field(default_factory=dict)
    # Boolean allow mask for actions (True means allowed)
    allow_mask: Dict[Action, bool] = field(default_factory=dict)


class AffordanceMemory:
    """
    Tracks which actions are effective in a given UI/menu context and masks
    repeatedly ineffective actions to prevent menu thrashing.
    """

    def __init__(
        self,
        *,
        num_actions: int,
        fail_threshold: int = 3,
        decay: float = 0.995,
    ) -> None:
        self.num_actions = int(num_actions)
        self.fail_threshold = max(1, int(fail_threshold))
        self.decay = float(decay)
        self._memory: Dict[str, AffordanceStats] = {}

    @staticmethod
    def _signature_from_obs_info(obs: np.ndarray, info: Dict) -> str:
        in_battle = "B1" if info.get("in_battle") else "B0"
        map_id = int(info.get("map_id") or 0)
        hud_hash = obs_token_hash(obs, stride=4, region="bottom")
        return f"{in_battle}|M{map_id:02X}|H{hud_hash[:12]}"

    def get_allow_mask(self, obs: np.ndarray, info: Dict) -> List[bool]:
        sig = self._signature_from_obs_info(obs, info)
        stats = self._memory.get(sig)
        if stats is None:
            # Default: allow all actions
            return [True] * self.num_actions
        mask = [stats.allow_mask.get(a, True) for a in range(self.num_actions)]
        return mask

    def report_effect(self, prev_obs: np.ndarray, prev_info: Dict, action: Action, next_obs: np.ndarray, next_info: Dict) -> None:
        sig = self._signature_from_obs_info(prev_obs, prev_info)
        stats = self._memory.get(sig)
        if stats is None:
            stats = AffordanceStats()
            self._memory[sig] = stats
        # Initialize defaults
        if not stats.allow_mask:
            for a in range(self.num_actions):
                stats.allow_mask[a] = True
                stats.failures[a] = 0
        # Heuristic: action ineffective if both obs and key info didn't change much
        ineffective = self._is_ineffective(prev_obs, prev_info, next_obs, next_info)
        if ineffective:
            stats.failures[action] = stats.failures.get(action, 0) + 1
            if stats.failures[action] >= self.fail_threshold:
                stats.allow_mask[action] = False
        else:
            # Decay failures on success, and re-allow the action soon after
            stats.failures[action] = max(0, int(stats.failures.get(action, 0) * self.decay))
            if stats.failures[action] == 0:
                stats.allow_mask[action] = True

    @staticmethod
    def _is_ineffective(prev_obs: np.ndarray, prev_info: Dict, next_obs: np.ndarray, next_info: Dict) -> bool:
        try:
            # Visual difference check
            if prev_obs.shape == next_obs.shape:
                diff = np.mean(np.abs(prev_obs.astype(np.float32) - next_obs.astype(np.float32)))
                if diff > 1.5:
                    return False
            # Key info differences
            coords_changed = prev_info.get("agent_coords") != next_info.get("agent_coords")
            map_changed = prev_info.get("map_id") != next_info.get("map_id")
            battle_state_changed = prev_info.get("in_battle") != next_info.get("in_battle")
            if coords_changed or map_changed or battle_state_changed:
                return False
            # Menu selection change signal: battle_result or caught flags change
            if prev_info.get("battle_result") != next_info.get("battle_result"):
                return False
        except Exception:
            pass
        return True


def apply_action_mask(q_values: np.ndarray, allow_mask: Iterable[bool]) -> int:
    """
    Given per-action Q-values and a boolean allow mask, pick the best allowed action.
    If all are masked, fall back to argmax.
    """
    mask = np.array(list(allow_mask), dtype=bool)
    if mask.shape[0] != q_values.shape[0]:
        return int(np.argmax(q_values))
    if np.any(mask):
        masked = np.where(mask, q_values, -1e9)
        return int(np.argmax(masked))
    return int(np.argmax(q_values))


