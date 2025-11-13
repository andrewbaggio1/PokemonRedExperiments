from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def serialize_dataclass(obj) -> Dict[str, Any]:
    if is_dataclass(obj):
        return asdict(obj)
    raise TypeError("Expected dataclass instance")


def save_json(path: str, data: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def obs_token_hash(obs: np.ndarray, stride: int = 4, region: str = "bottom") -> str:
    """Small hash of a HUD-like region for UI signatures."""
    arr = obs
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    h, w = arr.shape[:2]
    if region == "top":
        arr = arr[: h // 3, :]
    elif region == "bottom":
        arr = arr[-h // 3 :, :]
    else:
        arr = arr
    sample = arr[::stride, ::stride].astype(np.uint8)
    return hashlib.sha1(sample.tobytes()).hexdigest()


class SimpleVideoWriter:
    """Lightweight MP4 writer using OpenCV if available; otherwise, no-op."""

    def __init__(self, path: str, fps: int, frame_shape: Tuple[int, int]):
        self.path = path
        self.fps = fps
        self.frame_shape = frame_shape
        self._writer = None
        try:
            import cv2
        except Exception:
            cv2 = None
        self._cv2 = cv2
        if cv2 is not None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            h, w = frame_shape
            self._writer = cv2.VideoWriter(path, fourcc, fps, (w, h), True)

    def write_rgb(self, frame: np.ndarray) -> None:
        if self._writer is None or self._cv2 is None:
            return
        arr = frame
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        arr = np.ascontiguousarray(arr[:, :, ::-1])  # RGB -> BGR
        self._writer.write(arr)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


def soft_update(dst: torch.nn.Module, src: torch.nn.Module, tau: float) -> None:
    with torch.no_grad():
        for p, q in zip(dst.parameters(), src.parameters()):
            p.data.mul_(1.0 - tau).add_(q.data, alpha=tau)


def maybe_load_state_dict(module: torch.nn.Module, path: Optional[str]) -> bool:
    if not path or not os.path.exists(path):
        return False
    state = torch.load(path, map_location="cpu")
    target = module.module if hasattr(module, "module") else module
    target.load_state_dict(state)
    return True


def atomic_write_bytes(path: str, data: bytes) -> None:
    ensure_dir(os.path.dirname(path))
    tmp = f"{path}.tmp.{int(time.time()*1000)}"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)

