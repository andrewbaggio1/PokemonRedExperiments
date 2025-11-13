from __future__ import annotations

import argparse
import io
import os
from typing import Optional, Dict, List, Tuple
import re

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
    p.add_argument("--split", action="store_true", help="Emit one video per segment (legacy mode). Default is one video per actor.")
    p.add_argument("--ext", choices=["mp4", "mov"], default="mp4", help="Output container/extension (default: mp4).")
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


def _get_pyboy_screen(pyboy):
    """Return a screen interface compatible with both old and new PyBoy versions."""
    try:
        return pyboy.screen
    except AttributeError:
        return pyboy.botsupport_manager().screen()


def _screen_frame(screen) -> np.ndarray:
    """Extract an RGB frame from a PyBoy screen object across versions."""
    if hasattr(screen, "screen_ndarray"):
        frame = screen.screen_ndarray()
    elif hasattr(screen, "buffer"):
        frame = np.asarray(screen.buffer, dtype=np.uint8).reshape(144, 160, 3)
    else:
        frame = np.asarray(screen.screen_image(), dtype=np.uint8)
    return np.array(frame, copy=True)

def _group_segments_by_actor(paths: List[str]) -> Dict[int, List[Tuple[int, str]]]:
    """
    Group and sort segment files by actor id.
    Returns mapping: actor_id -> list of (segment_index, path) sorted by index.
    """
    groups: Dict[int, List[Tuple[int, str]]] = {}
    pat = re.compile(r"actor(\d+)_(?:seg|ep)(\d+)\.npz$")
    for p in paths:
        name = os.path.basename(p)
        m = pat.search(name)
        if not m:
            continue
        actor_id = int(m.group(1))
        seg_idx = int(m.group(2))
        groups.setdefault(actor_id, []).append((seg_idx, p))
    for aid in groups:
        groups[aid].sort(key=lambda t: t[0])
    return groups

def _merge_occupancy(dst: Optional[np.ndarray], src: Dict[int, np.ndarray]) -> Optional[np.ndarray]:
    """Aggregate occupancy maps across all tiles by max merge into a single array."""
    aggregate = dst.copy() if dst is not None else None
    for occ in src.values():
        if aggregate is None:
            aggregate = occ.copy()
        else:
            # Pad if shapes differ (should not in practice)
            h = max(aggregate.shape[0], occ.shape[0])
            w = max(aggregate.shape[1], occ.shape[1])
            if aggregate.shape != (h, w):
                pad_agg = np.zeros((h, w), dtype=np.float32)
                pad_agg[: aggregate.shape[0], : aggregate.shape[1]] = aggregate
                aggregate = pad_agg
            if occ.shape != (h, w):
                pad_occ = np.zeros((h, w), dtype=np.float32)
                pad_occ[: occ.shape[0], : occ.shape[1]] = occ
                occ = pad_occ
            aggregate = np.maximum(aggregate, occ)
    return aggregate

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
    plt.title(f"{episode_label} Occupancy")
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
    screen = _get_pyboy_screen(env.pyboy)
    frame_shape = _screen_frame(screen).shape[:2]

    if args.split:
        print(f"[replayer] Split mode: writing one file per segment ({len(ep_files)} segments).")
        for i, ep_path in enumerate(ep_files, 1):
            actions, savestate = _load_episode(ep_path)
            out_name = os.path.basename(ep_path).replace(".npz", f".{args.ext}")
            writer = SimpleVideoWriter(os.path.join(args.out, out_name), args.fps, frame_shape)
            env.reset()
            if savestate:
                buf = io.BytesIO(savestate)
                env.pyboy.load_state(buf)
            env._map_occ.clear()
            env._map_sal.clear()
            env._last_map = None
            env._last_coords = None
            _seed_occupancy(env)
            frame = _screen_frame(screen)
            writer.write_rgb(frame)
            for action in actions:
                obs, _, terminated, truncated, info = env.step(int(action))
                frame = info.get("raw_frame")
                if frame is None:
                    frame = _screen_frame(screen)
                writer.write_rgb(frame)
                if terminated or truncated:
                    break
            writer.close()
            if plt is not None:
                map_path = os.path.join(map_dir, os.path.basename(ep_path).replace(".npz", ".png"))
                _save_map_image(env, map_path, os.path.basename(ep_path).replace(".npz", ""))
            if i % 10 == 0 or i == len(ep_files):
                print(f"[replayer] processed {i}/{len(ep_files)} segments")
        env.close()
        return

    # Continuous mode per actor
    groups = _group_segments_by_actor(ep_files)
    if not groups:
        print("No actor-labeled logs found (expected files like actor0_seg00000.npz).")
        env.close()
        return
    print(f"[replayer] Continuous mode: {len(groups)} actor groups found.")
    for actor_id, items in sorted(groups.items(), key=lambda kv: kv[0]):
        total = len(items)
        out_path = os.path.join(args.out, f"actor{actor_id}.{args.ext}")
        writer = SimpleVideoWriter(out_path, args.fps, frame_shape)
        actor_agg_occ: Optional[np.ndarray] = None
        for idx, (seg_idx, ep_path) in enumerate(items, 1):
            actions, savestate = _load_episode(ep_path)
            env.reset()
            if savestate:
                buf = io.BytesIO(savestate)
                env.pyboy.load_state(buf)
            env._map_occ.clear()
            env._map_sal.clear()
            env._last_map = None
            env._last_coords = None
            _seed_occupancy(env)
            frame = _screen_frame(screen)
            writer.write_rgb(frame)
            for action in actions:
                obs, _, terminated, truncated, info = env.step(int(action))
                frame = info.get("raw_frame")
                if frame is None:
                    frame = _screen_frame(screen)
                writer.write_rgb(frame)
                if terminated or truncated:
                    break
            # Merge occupancy for this segment into actor aggregate
            actor_agg_occ = _merge_occupancy(actor_agg_occ, env._map_occ)
            if idx % 10 == 0 or idx == total:
                print(f"[replayer] actor {actor_id}: {idx}/{total} segments")
        writer.close()
        if plt is not None and actor_agg_occ is not None and np.any(actor_agg_occ):
            vmax = max(1.0, float(actor_agg_occ.max()))
            plt.figure(figsize=(5, 5))
            plt.imshow(actor_agg_occ, cmap="plasma", origin="lower", vmin=0.0, vmax=vmax)
            plt.title(f"Actor {actor_id} Occupancy")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.tight_layout()
            map_path = os.path.join(map_dir, f"actor{actor_id}.png")
            plt.savefig(map_path, dpi=150)
            plt.close()
        print(f"[replayer] actor {actor_id}: wrote {out_path}")
    env.close()


if __name__ == "__main__":
    main()

