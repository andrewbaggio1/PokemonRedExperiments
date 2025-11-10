from __future__ import annotations

from typing import Dict


class MapStayPenalty:
    """Penalty for spending too long on a particular map without leaving."""

    def __init__(
        self,
        *,
        map_id: int = 0,
        interval: int = 200,
        penalty: float = -5.0,
    ) -> None:
        self.map_id = int(map_id)
        self.interval = max(1, int(interval))
        self.penalty = float(penalty)
        self._ticks = 0

    def reset(self) -> None:
        self._ticks = 0

    def compute(self, obs, info: Dict) -> float:
        current_map = int(info.get("map_id", -1))
        if current_map != self.map_id:
            self._ticks = 0
            return 0.0
        self._ticks += 1
        if self._ticks % self.interval == 0:
            return self.penalty
        return 0.0

