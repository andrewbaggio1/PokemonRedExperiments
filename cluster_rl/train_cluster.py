from __future__ import annotations

import argparse
import signal
import os
import time
from multiprocessing import Event, Process, Queue, set_start_method
import torch

from .actor import actor_process
from .config import ClusterConfig, load_config
from .learner import learner_loop
from .utils import ensure_dir, load_json, save_json, serialize_dataclass, set_global_seed


def _parse_args():
    parser = argparse.ArgumentParser(description="Cluster-friendly Pokémon Red training (actor-learner).")
    parser.add_argument("--config", default=None, help="Path to JSON config (cluster_config.json).")
    parser.add_argument("--rom", default=None, help="Override ROM path.")
    parser.add_argument("--save-dir", default=None, help="Override save directory.")
    parser.add_argument("--run-name", default=None, help="Override run name.")
    return parser.parse_args()


def main():
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    args = _parse_args()
    cfg = load_config(args.config)
    if args.rom:
        cfg.rom_path = args.rom
    if args.save_dir:
        cfg.save_dir = args.save_dir
    if args.run_name:
        cfg.run_name = args.run_name
    run_dir = cfg.run_dir
    ensure_dir(run_dir)
    save_json(os.path.join(run_dir, "config.json"), serialize_dataclass(cfg))
    set_global_seed(cfg.base_seed)
    # Queues
    ep_queue: Queue = Queue(maxsize=cfg.num_actors * 2)
    stop_event = Event()
    # Paths for param broadcast and logs
    param_path = os.path.join(run_dir, "online_params.pt")
    replay_log_dir = os.path.join(run_dir, cfg.replay_log_dir)
    ensure_dir(replay_log_dir)
    progress_path = os.path.join(run_dir, "progress.json")
    initial_env_steps = 0
    if os.path.exists(progress_path):
        try:
            progress_data = load_json(progress_path)
            initial_env_steps = int(progress_data.get("env_steps", 0))
        except Exception:
            initial_env_steps = 0
    # Bootstrap one env to probe shapes (on CPU, single step)
    from PokemonRedStreaming.env_pokemon import PokemonRedEnv
    from PokemonRedStreaming.epsilon_env import EpsilonEnv
    from PokemonRedStreaming.rewards.efficiency_penalty import EfficiencyPenalty
    from .occupancy_memory import ClusterEnv
    from .cluster_map_features import build_context_features
    base = PokemonRedEnv(
        rom_path=cfg.rom_path,
        show_display=False,
        frame_skip=cfg.frame_skip,
        boot_steps=cfg.boot_steps,
        max_no_input_frames=cfg.max_no_input_frames,
        input_spacing_frames=cfg.input_spacing_frames,
        delete_sav_on_reset=cfg.delete_sav_on_reset,
        emulation_speed=0,
    )
    shaped = EpsilonEnv(base, [EfficiencyPenalty()])
    env = ClusterEnv(shaped)
    obs, info = env.reset(seed=cfg.base_seed)
    obs_shape = env.observation_space.shape
    c, h, w = (1, obs_shape[0], obs_shape[1]) if len(obs_shape) == 2 else (obs_shape[2], obs_shape[0], obs_shape[1])
    context_dim = build_context_features(info).shape[0]
    n_actions = env.action_space.n
    env.close()
    # Start learner
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
    learner = Process(
        target=learner_loop,
        args=(cfg, param_path, ep_queue, device, (c, h, w), context_dim, stop_event, progress_path, initial_env_steps, n_actions),
        daemon=True,
    )
    learner.start()
    # Start actors
    actors = []
    for rank in range(cfg.num_actors):
        p = Process(
            target=actor_process,
            args=(rank, cfg, param_path, ep_queue, stop_event, replay_log_dir),
            daemon=True,
        )
        p.start()
        actors.append(p)
    # Graceful shutdown on SIGINT/SIGTERM
    def _signal_handler(signum, frame):
        stop_event.set()
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    # Monitor runtime and wait for completion
    start_time = time.monotonic()
    try:
        while learner.is_alive():
            learner.join(timeout=5.0)
            if not learner.is_alive():
                break
            if cfg.max_runtime_seconds and time.monotonic() - start_time >= cfg.max_runtime_seconds:
                stop_event.set()
                break
    except KeyboardInterrupt:
        stop_event.set()
    # Ensure stop event is set and wait for processes
    stop_event.set()
    learner.join()
    for p in actors:
        p.join()


if __name__ == "__main__":
    main()


