from __future__ import annotations

from typing import Dict, Iterable, List, Tuple


class QuestReward:
    """Rewards custom milestones such as reaching coordinates, maps, or story flags."""

    def __init__(self, quests: Iterable[Dict] | None = None) -> None:
        self.quests: List[Dict] = []
        for quest in quests or []:
            if not isinstance(quest, dict):
                continue
            processed = {
                "name": quest.get("name", "quest"),
                "map_id": quest.get("map_id"),
                "reward": float(quest.get("reward", 50.0)),
                "coords": quest.get("coords"),
                "story_flag": quest.get("story_flag"),
                "once": bool(quest.get("once", True)),
            }
            if processed["map_id"] is None and processed["story_flag"] is None:
                continue
            self.quests.append(processed)
        self._completed: set[str] = set()

    def reset(self) -> None:
        # Keep progress across episodes unless quest explicitly marked repeatable.
        pass

    def compute(self, obs, info: Dict) -> float:
        if not self.quests:
            return 0.0
        reward = 0.0
        coords = info.get("agent_coords")
        map_id = info.get("map_id")
        story_flags = info.get("story_flags") or {}

        for quest in self.quests:
            name = quest["name"]
            if quest["once"] and name in self._completed:
                continue
            map_match = quest["map_id"] is None or quest["map_id"] == map_id
            coord_match = True
            if quest["coords"] and coords:
                coord_match = tuple(coords) == tuple(quest["coords"])
            flag_match = True
            flag_key = quest["story_flag"]
            if flag_key:
                flag_match = bool(story_flags.get(flag_key, False))
            if map_match and coord_match and flag_match:
                reward += quest["reward"]
                if quest["once"]:
                    self._completed.add(name)
        return reward
