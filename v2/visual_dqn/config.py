from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _resolve_path(value: str, base: Optional[Path]) -> str:
    path = Path(value).expanduser()
    if base is not None and not path.is_absolute():
        path = (base / path).expanduser()
    return str(path.resolve())


def load_curriculum_config(path: Optional[Path]) -> Optional[List[Dict[str, Any]]]:
    if path is None:
        return None
    curriculum_path = Path(path).expanduser()
    with curriculum_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Curriculum config must be a list of stage definitions.")
    return data


def normalize_curriculum_paths(
    curriculum: Optional[List[Dict[str, Any]]],
    *,
    state_base: Optional[Path] = None,
    save_base: Optional[Path] = None,
) -> Optional[List[Dict[str, Any]]]:
    if curriculum is None:
        return None

    normalized: List[Dict[str, Any]] = []
    for stage in curriculum:
        stage_cfg = dict(stage)
        if "state_path" in stage_cfg:
            stage_cfg["state_path"] = _resolve_path(stage_cfg["state_path"], state_base)
        if "save_path" in stage_cfg and stage_cfg["save_path"] is not None:
            stage_cfg["save_path"] = _resolve_path(stage_cfg["save_path"], save_base)
        normalized.append(stage_cfg)
    return normalized

