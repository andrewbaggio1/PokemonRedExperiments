from __future__ import annotations

from typing import Dict, Iterable, Set


class MapStayPenalty:
    """Penalty for spending too long inside a set of maps without leaving."""

    def __init__(
        self,
        *,
        map_id: int | None = 0,
        map_ids: Iterable[int] | None = None,
        interval: int = 200,
        penalty: float = -5.0,
        name: str | None = None,
    ) -> None:
        target_maps: Set[int] = set(int(mid) for mid in map_ids or [])
        if map_id is not None:
            target_maps.add(int(map_id))
        if not target_maps:
            raise ValueError("MapStayPenalty requires map_id or map_ids.")
        self._target_maps: Set[int] = target_maps
        self.name = name or "map_stay"
        self.interval = max(1, int(interval))
        self.penalty = float(penalty)
        self._ticks = 0

    def reset(self) -> None:
        self._ticks = 0

    def compute(self, obs, info: Dict) -> float:
        current_map = int(info.get("map_id", -1))
        if current_map not in self._target_maps:
            self._ticks = 0
            return 0.0
        self._ticks += 1
        if self._ticks % self.interval == 0:
            return self.penalty
        return 0.0
