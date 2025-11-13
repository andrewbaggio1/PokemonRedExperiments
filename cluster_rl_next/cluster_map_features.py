from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

# Lightweight region categories (mirrors subset of PokemonRedStreaming.map_features)
REGION_UNKNOWN = 0
REGION_TOWN = 1
REGION_ROUTE = 2
REGION_INTERIOR = 3
REGION_GYM = 4
REGION_DUNGEON = 5

_REGION_LOOKUP: Dict[int, int] = {
    0x00: REGION_TOWN,  # Pallet Town
    0x05: REGION_TOWN,  # Viridian City
    0x06: REGION_GYM,
    0x0A: REGION_ROUTE,
    0x0B: REGION_ROUTE,
    0x0C: REGION_DUNGEON,  # Viridian Forest
    0x11: REGION_GYM,
    0x14: REGION_GYM,
    0x18: REGION_GYM,
    0x1C: REGION_ROUTE,
    0x1D: REGION_ROUTE,
    0x1E: REGION_DUNGEON,
    0x20: REGION_TOWN,
    0x24: REGION_GYM,
    0x30: REGION_ROUTE,
    0x33: REGION_GYM,
    0x34: REGION_GYM,
    0x63: REGION_DUNGEON,  # Elite Four suite
    0x64: REGION_DUNGEON,
    0x65: REGION_DUNGEON,
    0x66: REGION_DUNGEON,
    0x67: REGION_DUNGEON,
}

_REGION_COUNT = 6


def region_one_hot(map_id: int) -> np.ndarray:
    vec = np.zeros(_REGION_COUNT, dtype=np.float32)
    region = _REGION_LOOKUP.get(map_id, REGION_UNKNOWN)
    vec[region] = 1.0
    return vec


def _normalize_coords(coords: Tuple[int, int]) -> Tuple[float, float]:
    x, y = coords
    return float(x) / 255.0, float(y) / 255.0


def build_context_features(info: Dict) -> np.ndarray:
    """
    Build a compact context vector for the policy that excludes HP values.
    We include occupancy and saliency crop summaries for local spatial memory.
    """
    coords = info.get("agent_coords") or (0, 0)
    norm_x, norm_y = _normalize_coords((int(coords[0]), int(coords[1])))
    map_id = int(info.get("map_id") or 0)
    region_vec = region_one_hot(map_id)
    outdoor_flag = 1.0 if _REGION_LOOKUP.get(map_id, REGION_UNKNOWN) in {REGION_TOWN, REGION_ROUTE} else 0.0
    badge_count = float(info.get("badge_count", 0)) / 8.0
    champion_flag = 1.0 if info.get("champion_defeated") else 0.0
    story_flags = info.get("story_flags") or {}
    story_progress = 0.0
    if isinstance(story_flags, dict) and story_flags:
        on_flags = sum(1 for v in story_flags.values() if v)
        story_progress = on_flags / float(len(story_flags))
    key_item_ids = info.get("key_item_ids") or []
    key_item_count = min(len(key_item_ids), 20) / 20.0
    pokedex_owned = float(info.get("pokedex_owned_count") or 0) / 151.0
    in_battle = 1.0 if info.get("in_battle") else 0.0
    battle_type = float(info.get("battle_type") or 0) / 10.0
    recent_catch = 1.0 if info.get("last_battle_result") == "caught" else 0.0
    # Occupancy/saliency summaries
    total_unique = float(info.get("occupancy_total_unique") or 0.0)
    steps_sum = float(info.get("occupancy_steps_sum") or 0.0)
    sal_marks = float(info.get("saliency_total_marks") or 0.0)
    total_unique_norm = min(total_unique / 4096.0, 1.0)
    steps_sum_norm = min(steps_sum / 10000.0, 1.0)
    sal_marks_norm = min(sal_marks / 500.0, 1.0)
    # Local crop: flatten small occupancy+saliency patch
    occ_crop = info.get("occupancy_crop")
    sal_crop = info.get("saliency_crop")
    crop_vec = np.zeros((0,), dtype=np.float32)
    if isinstance(occ_crop, np.ndarray) and isinstance(sal_crop, np.ndarray):
        # Normalize per-crop for scale robustness
        occ_norm = occ_crop / max(1.0, float(occ_crop.max() if occ_crop.size > 0 else 1.0))
        sal_norm = np.clip(sal_crop, 0.0, 1.0)
        crop_vec = np.concatenate([occ_norm.flatten(), sal_norm.flatten()]).astype(np.float32)
    core = np.array(
        [
            norm_x,
            norm_y,
            float(map_id) / 255.0,
            outdoor_flag,
            badge_count,
            champion_flag,
            story_progress,
            key_item_count,
            pokedex_owned,
            in_battle,
            battle_type,
            recent_catch,
            total_unique_norm,
            steps_sum_norm,
            sal_marks_norm,
        ],
        dtype=np.float32,
    )
    return np.concatenate([core, region_vec, crop_vec], axis=0)


