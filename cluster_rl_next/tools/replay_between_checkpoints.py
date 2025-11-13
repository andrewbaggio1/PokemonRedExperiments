from __future__ import annotations

import argparse
import json
import os
from typing import List

import numpy as np


def _parse_args():
    parser = argparse.ArgumentParser(description="Build a replay summary between two checkpoints.")
    parser.add_argument("--from-ckpt", required=True, help="Starting checkpoint path.")
    parser.add_argument("--to-ckpt", required=True, help="Ending checkpoint path.")
    parser.add_argument("--logs-dir", required=True, help="Directory containing actor *.npz segment logs.")
    parser.add_argument("--out", required=True, help="Output npz path for merged actions.")
    return parser.parse_args()


def _select_segments(logs_dir: str, start_mtime: float, end_mtime: float) -> List[str]:
    segments = []
    for root, _, files in os.walk(logs_dir):
        for fname in files:
            if not fname.endswith(".npz"):
                continue
            path = os.path.join(root, fname)
            mtime = os.path.getmtime(path)
            if start_mtime <= mtime <= end_mtime:
                segments.append(path)
    segments.sort()
    return segments


def main():
    args = _parse_args()
    start_mtime = os.path.getmtime(args.from_ckpt)
    end_mtime = os.path.getmtime(args.to_ckpt)
    if end_mtime <= start_mtime:
        raise ValueError("Destination checkpoint must be newer than source.")
    segments = _select_segments(args.logs_dir, start_mtime, end_mtime)
    if not segments:
        raise RuntimeError("No segments found in the specified time window.")
    merged_actions: List[np.ndarray] = []
    records = []
    first_savestate = None
    for seg_path in segments:
        data = np.load(seg_path, allow_pickle=False)
        actions = data.get("actions")
        if actions is None:
            continue
        merged_actions.append(actions.astype(np.int16, copy=False))
        savestate = data.get("savestate")
        if first_savestate is None and savestate is not None:
            first_savestate = np.copy(savestate)
        records.append({"segment": seg_path, "num_actions": int(actions.shape[0])})
    if not merged_actions:
        raise RuntimeError("Segments did not contain action arrays.")
    concat_actions = np.concatenate(merged_actions, axis=0)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(
        args.out,
        actions=concat_actions,
        savestate=first_savestate if first_savestate is not None else np.array([], dtype=np.uint8),
    )
    summary_path = f"{os.path.splitext(args.out)[0]}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump({"segments": records, "total_actions": int(concat_actions.shape[0])}, fh, indent=2)
    print(f"Replay bundle saved to {args.out} with {concat_actions.shape[0]} actions.")


if __name__ == "__main__":
    main()
