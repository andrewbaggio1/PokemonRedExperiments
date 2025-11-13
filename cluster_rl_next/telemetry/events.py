from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class TelemetryEvent:
    ts: float
    run: str
    actor_id: int
    episode: int
    env_step: int
    map_id: int
    x: int
    y: int
    reward_delta: float
    epsilon: float
    segment_idx: int


def event_to_json(event: TelemetryEvent) -> Dict[str, Any]:
    return asdict(event)
