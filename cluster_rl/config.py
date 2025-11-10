from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass
class ClusterConfig:
    # Paths
    rom_path: str
    save_dir: str
    run_name: str
    # Training
    total_episodes: int
    target_total_steps: int
    max_steps_per_episode: int
    num_actors: int
    unroll_length: int
    burn_in: int
    batch_size: int
    gamma: float
    n_step: int
    learning_rate: float
    target_sync_interval: int
    learn_start_steps: int
    train_frequency: int
    prioritized_alpha: float
    prioritized_beta_frames: int
    # Exploration
    epsilon_actor_start: float
    epsilon_actor_end: float
    epsilon_decay_steps: int
    # Replay capacity (in transitions)
    replay_capacity_transitions: int
    # Environment
    frame_skip: int
    boot_steps: int
    max_no_input_frames: int
    input_spacing_frames: int
    headless: bool
    delete_sav_on_reset: bool
    # Visualization/logging (no live training video; only metrics)
    log_interval_steps: int
    checkpoint_every_episodes: int
    # Model
    use_spatial_attention: bool
    lstm_hidden_size: int
    num_quantiles: int
    # Replay logging (for offline video)
    log_episodes_for_replay: bool
    replay_log_dir: str
    # Randomness
    base_seed: int
    # Affordances
    affordance_fail_threshold: int
    affordance_decay: float
    # Sticky actions / action durations
    action_duration_options: tuple[int, ...]
    # Occupancy crop
    occupancy_crop_radius: int
    # Runtime management
    max_runtime_seconds: Optional[int]

    @property
    def run_dir(self) -> str:
        return os.path.join(self.save_dir, self.run_name)

    @staticmethod
    def default(rom_dir: Optional[str] = None) -> "ClusterConfig":
        rom_path = "pokemon_red.gb"
        if rom_dir:
            rom_path = os.path.join(rom_dir, rom_path)
        return ClusterConfig(
            rom_path=rom_path,
            save_dir="cluster_runs",
            run_name="run1",
            total_episodes=0,
            target_total_steps=25_000_000,
            max_steps_per_episode=2000,
            num_actors=4,
            unroll_length=80,
            burn_in=40,
            batch_size=64,
            gamma=0.99,
            n_step=5,
            learning_rate=2.5e-4,
            target_sync_interval=4000,
            learn_start_steps=10000,
            train_frequency=4,
            prioritized_alpha=0.6,
            prioritized_beta_frames=200_000,
            epsilon_actor_start=1.0,
            epsilon_actor_end=0.05,
            epsilon_decay_steps=1_000_000,
            replay_capacity_transitions=1_000_000,
            frame_skip=4,
            boot_steps=120,
            max_no_input_frames=600,
            input_spacing_frames=1,
            headless=True,
            delete_sav_on_reset=True,
            log_interval_steps=2000,
            checkpoint_every_episodes=50,
            use_spatial_attention=True,
            lstm_hidden_size=512,
            num_quantiles=51,
            log_episodes_for_replay=True,
            replay_log_dir="episode_logs",
            base_seed=7,
            affordance_fail_threshold=3,
            affordance_decay=0.995,
            action_duration_options=(1, 2, 4, 8),
            occupancy_crop_radius=10,
            max_runtime_seconds=13_500,  # ~3.75 hours
        )


def load_config(path: Optional[str]) -> ClusterConfig:
    if not path:
        return ClusterConfig.default()
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    defaults = asdict(ClusterConfig.default())
    merged: Dict[str, Any] = {**defaults, **data}
    # Handle tuple conversion for action durations
    act_dur = merged.get("action_duration_options", (1, 2, 4, 8))
    if isinstance(act_dur, list):
        merged["action_duration_options"] = tuple(int(x) for x in act_dur)
    if merged.get("max_runtime_seconds") is None:
        merged["max_runtime_seconds"] = None
    return ClusterConfig(**merged)  # type: ignore[arg-type]


