from __future__ import annotations

import argparse
import io
import os
from typing import Optional

import numpy as np

from PokemonRedStreaming.env_pokemon import PokemonRedEnv
from PokemonRedStreaming.epsilon_env import EpsilonEnv

from .config import load_config
from .occupancy_memory import ClusterEnv, OccupancyConfig
from .utils import SimpleVideoWriter, ensure_dir

try:  # pragma: no cover - optional dependency
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def _parse_args():
    p = argparse.ArgumentParser(description="Offline replayer to encode training episodes into MP4 and occupancy maps.")
    p.add_argument("--rom", required=True, help="Path to pokemon_red.gb")
    p.add_argument("--logs", required=True, help="Directory with episode .npz logs")
    p.add_argument("--out", required=True, help="Output video directory")
    p.add_argument("--map-out", default=None, help="Directory for map heatmaps (defaults to <out>/maps)")
    p.add_argument("--fps", type=int, default=30, help="Output frames per second")
    p.add_argument("--config", default=None, help="Optional cluster config to mirror env settings")
    return p.parse_args()


def _load_episode(path: str):
    data = np.load(path, allow_pickle=False)
    actions = data["actions"].astype(np.int32).tolist()
    savestate_arr = data["savestate"].astype(np.uint8)
    savestate = savestate_arr.tobytes()
    return actions, savestate


def _build_env(args, cfg) -> ClusterEnv:
    frame_skip = cfg.frame_skip if cfg else 4
    boot_steps = cfg.boot_steps if cfg else 120
    max_no_input = cfg.max_no_input_frames if cfg else 600
    input_spacing = cfg.input_spacing_frames if cfg else 1
    delete_sav = cfg.delete_sav_on_reset if cfg else True
    base = PokemonRedEnv(
        rom_path=args.rom,
        show_display=False,
        frame_skip=frame_skip,
        boot_steps=boot_steps,
        max_no_input_frames=max_no_input,
        input_spacing_frames=input_spacing,
        delete_sav_on_reset=delete_sav,
        emulation_speed=0,
    )
    epsilon = EpsilonEnv(base, reward_modules=[])
    cluster_env = ClusterEnv(epsilon, cfg=OccupancyConfig())
    return cluster_env


def _seed_occupancy(env: ClusterEnv) -> None:
    inner_env = env.env.env  # PokemonRedEnv
    _ = inner_env._obs()
    info = inner_env._gather_info()
    env._update_maps(info)
    env._attach_occupancy_info(info)


def _save_map_image(env: ClusterEnv, path: str, episode_label: str) -> None:
    if plt is None:
        return
    aggregate = None
    for occ in env._map_occ.values():
        if aggregate is None:
            aggregate = occ.copy()
        else:
            aggregate = np.maximum(aggregate, occ)
    if aggregate is None or not np.any(aggregate):
        return
    vmax = max(1.0, float(aggregate.max()))
    plt.figure(figsize=(5, 5))
    plt.imshow(aggregate, cmap="plasma", origin="lower", vmin=0.0, vmax=vmax)
    plt.title(f"Episode {episode_label} Occupancy")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    args = _parse_args()
    ensure_dir(args.out)
    cfg = load_config(args.config) if args.config else None
    map_dir = args.map_out or os.path.join(args.out, "maps")
    if plt is not None:
        ensure_dir(map_dir)
    elif args.map_out:
        print("[replayer] matplotlib not available; skipping map generation.")
    # Scan logs
    ep_files = sorted([os.path.join(args.logs, f) for f in os.listdir(args.logs) if f.endswith(".npz")])
    if not ep_files:
        print("No episode logs found.")
        return
    env = _build_env(args, cfg)
    screen = env.pyboy.screen
    frame_shape = screen.ndarray.shape[:2]
    for ep_path in ep_files:
        actions, savestate = _load_episode(ep_path)
        writer = SimpleVideoWriter(os.path.join(args.out, os.path.basename(ep_path).replace(".npz", ".mp4")), args.fps, frame_shape)
        env.reset()
        if savestate:
            buf = io.BytesIO(savestate)
            env.pyboy.load_state(buf)
        env._map_occ.clear()
        env._map_sal.clear()
        env._last_map = None
        env._last_coords = None
        _seed_occupancy(env)
        frame = np.array(screen.ndarray, copy=True)
        writer.write_rgb(frame)
        for action in actions:
            obs, _, terminated, truncated, info = env.step(int(action))
            frame = info.get("raw_frame")
            if frame is None:
                frame = np.array(screen.ndarray, copy=True)
            writer.write_rgb(frame)
            if terminated or truncated:
                break
        writer.close()
        if plt is not None:
            map_path = os.path.join(map_dir, os.path.basename(ep_path).replace(".npz", ".png"))
            _save_map_image(env, map_path, os.path.basename(ep_path).replace(".npz", ""))
    env.close()


if __name__ == "__main__":
    main()

