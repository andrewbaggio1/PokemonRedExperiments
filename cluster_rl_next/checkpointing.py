from __future__ import annotations

import io
import os
from typing import Any, Dict, List, Optional

import torch

from .utils import atomic_write_bytes, ensure_dir


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    replay_state: Optional[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> None:
    ensure_dir(os.path.dirname(path))
    payload: Dict[str, Any] = {
        "model": model.state_dict(),
        "metadata": metadata,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if replay_state is not None:
        payload["replay"] = replay_state
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    atomic_write_bytes(path, buffer.getvalue())


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload


def list_checkpoints(directory: str) -> List[str]:
    if not os.path.isdir(directory):
        return []
    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".pt")
    ]
    files.sort()
    return files


def prune_checkpoints(directory: str, max_keep: int) -> None:
    if max_keep <= 0:
        return
    ckpts = list_checkpoints(directory)
    excess = max(0, len(ckpts) - max_keep)
    for path in ckpts[:excess]:
        try:
            os.remove(path)
        except FileNotFoundError:
            continue
