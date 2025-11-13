from __future__ import annotations

import os
from typing import Optional

import queue
import time
from io import BytesIO

import numpy as np
import torch
from torch.multiprocessing import Event, Queue

from PokemonRedStreaming.env_pokemon import PokemonRedEnv
from PokemonRedStreaming.epsilon_env import EpsilonEnv
from PokemonRedStreaming.rewards.battle_outcome import BattleOutcomeReward
from PokemonRedStreaming.rewards.battle_damage_reward import BattleDamageReward
from PokemonRedStreaming.rewards.badge_reward import BadgeReward
from PokemonRedStreaming.rewards.story_flag_reward import StoryFlagReward
from PokemonRedStreaming.rewards.champion_reward import ChampionReward
from PokemonRedStreaming.rewards.item_collection import ItemCollectionReward
from PokemonRedStreaming.rewards.pokedex_reward import PokedexReward
from PokemonRedStreaming.rewards.trainer_tier_reward import TrainerBattleReward
from PokemonRedStreaming.rewards.map_exploration import MapExplorationReward
from PokemonRedStreaming.rewards.novelty import NoveltyReward
from PokemonRedStreaming.rewards.map_visit_reward import MapVisitReward
from PokemonRedStreaming.rewards.exploration_frontier_reward import ExplorationFrontierReward
from PokemonRedStreaming.rewards.efficiency_penalty import EfficiencyPenalty
from PokemonRedStreaming.rewards.safety_penalty import SafetyPenalty
from PokemonRedStreaming.rewards.resource_reward import ResourceManagementReward
from PokemonRedStreaming.rewards.latent_event_reward import LatentEventReward
from PokemonRedStreaming.rewards.map_stay_penalty import MapStayPenalty

from .affordances import AffordanceMemory, apply_action_mask
from .cluster_map_features import build_context_features
from .config import ClusterConfig
from .occupancy_memory import ClusterEnv, OccupancyConfig
from .utils import ensure_dir


def preprocess_obs(obs: np.ndarray) -> np.ndarray:
    arr = obs.astype(np.float32) / 255.0
    if arr.ndim == 3:
        arr = np.transpose(arr, (2, 0, 1))
    return arr


def build_rewards(cfg: ClusterConfig):
    modules = []
    # Balanced mixture; most weights are small and annealing handled by learner if desired
    modules.append(MapExplorationReward(base_reward=0.5, neighbor_radius=1, neighbor_weight=0.08, distance_weight=0.6, min_reward=0.05, persist_across_episodes=False))
    modules.append(NoveltyReward(base_reward=0.4, decay=0.85, min_reward=0.02, sample_stride=4, quantisation=32, persist_across_episodes=False))
    modules.append(MapVisitReward(map_reward=20.0))
    modules.append(ExplorationFrontierReward(distance_reward=1.0, min_gain=2))
    modules.append(BattleOutcomeReward(win_reward=40.0, loss_penalty=-30.0))
    modules.append(BattleDamageReward(damage_scale=4.0, escape_penalty=-8.0))
    modules.append(BadgeReward(reward_per_badge=250.0))
    modules.append(StoryFlagReward(flag_definitions=None, default_reward=180.0))
    modules.append(ChampionReward(reward=1000.0))
    modules.append(PokedexReward(new_species_reward=10.0, milestone_rewards=[(10, 50.0), (30, 150.0), (60, 350.0)]))
    modules.append(ItemCollectionReward(item_reward=1.0, key_item_reward=12.0, key_item_ids=[]))
    modules.append(EfficiencyPenalty(step_penalty=-0.0005, idle_penalty=-0.2, idle_threshold=30))
    modules.append(SafetyPenalty(loss_penalty=-30.0, blackout_penalty=-60.0, low_hp_threshold=0.15, low_hp_penalty=-3.0))
    modules.append(ResourceManagementReward(map_keywords=["pokemon center", "pokémon center", "poke center"], map_reward=0.0, utility_item_ids=[13, 14, 34], item_reward=0.0))
    modules.append(LatentEventReward(base_reward=0.0, revisit_decay=0.5))
    modules.append(MapStayPenalty(map_id=0, interval=180, penalty=-12.0))
    return modules


def make_env(cfg: ClusterConfig, show_display: bool = False, resume_state_path: Optional[str] = None) -> ClusterEnv:
    base = PokemonRedEnv(
        rom_path=cfg.rom_path,
        show_display=bool(show_display and not cfg.headless),
        frame_skip=cfg.frame_skip,
        boot_steps=cfg.boot_steps,
        max_no_input_frames=cfg.max_no_input_frames,
        state_path=resume_state_path,
        story_flag_defs=None,
        track_visit_stats=True,
        delete_sav_on_reset=cfg.delete_sav_on_reset,
        input_spacing_frames=cfg.input_spacing_frames,
        emulation_speed=0 if cfg.headless else 1,
    )
    shaped = EpsilonEnv(base, build_rewards(cfg))
    occ = ClusterEnv(shaped, cfg=OccupancyConfig(crop_radius=cfg.occupancy_crop_radius))
    return occ


def epsilon_by_step(step: int, cfg: ClusterConfig) -> float:
    if cfg.epsilon_decay_steps <= 0:
        return cfg.epsilon_actor_end
    fraction = min(1.0, step / cfg.epsilon_decay_steps)
    return cfg.epsilon_actor_start + fraction * (cfg.epsilon_actor_end - cfg.epsilon_actor_start)


def actor_process(
    rank: int,
    cfg: ClusterConfig,
    param_path: str,
    out_queue: Queue,
    stop_event: Event,
    log_dir: Optional[str] = None,
) -> None:
    """
    Actor loop:
      - periodically loads latest model params from param_path (if present)
      - runs episodes headless, applies affordance masks
      - sends completed episode transitions to learner via out_queue
      - logs minimal replay data if enabled (one file per episode)
    """
    device = torch.device("cpu")
    # Resume from per-actor snapshot if enabled and present
    resume_state_path: Optional[str] = None
    if getattr(cfg, "resume_from_snapshot", False):
        snap_dir = os.path.join(cfg.run_dir, getattr(cfg, "snapshots_dir", "snapshots"))
        latest = os.path.join(snap_dir, f"actor{rank}_latest.state")
        if os.path.exists(latest):
            resume_state_path = latest
    env = make_env(cfg, show_display=False, resume_state_path=resume_state_path)
    afford = AffordanceMemory(num_actions=env.action_space.n, fail_threshold=cfg.affordance_fail_threshold, decay=cfg.affordance_decay)
    # Initialize a tiny model to shape action choices when parameters are available
    obs_shape = env.observation_space.shape
    c, h, w = (1, obs_shape[0], obs_shape[1]) if len(obs_shape) == 2 else (obs_shape[2], obs_shape[0], obs_shape[1])
    # Probe one reset to get context size
    obs0, info0 = env.reset(seed=cfg.base_seed + rank)
    context_dim = build_context_features(info0).shape[0]
    from .model import ClusterDQN
    policy = ClusterDQN((c, h, w), context_dim, env.action_space.n, use_spatial_attention=True, lstm_hidden_size=cfg.lstm_hidden_size, num_quantiles=cfg.num_quantiles).to(device)
    hidden = policy.init_hidden(1, device)
    last_param_mtime = 0.0
    step_counter = 0
    episode = 0
    segment_idx = 0
    segment_start_state: Optional[bytes] = None
    last_snapshot_time = time.monotonic()
    snapshot_interval_s = max(1, int(getattr(cfg, "snapshot_interval_minutes", 10))) * 60
    snapshots_dir = os.path.join(cfg.run_dir, getattr(cfg, "snapshots_dir", "snapshots"))
    ensure_dir(snapshots_dir)
    last_seg_log = -1
    max_episodes = cfg.total_episodes if cfg.total_episodes > 0 else None
    try:
        while not stop_event.is_set():
            if max_episodes is not None and episode >= max_episodes:
                break
            transitions = []
            # Reload latest parameters if available
            try:
                if os.path.exists(param_path):
                    mtime = os.path.getmtime(param_path)
                    if mtime > last_param_mtime:
                        state = torch.load(param_path, map_location="cpu")
                        policy.load_state_dict(state)
                        last_param_mtime = mtime
            except Exception:
                pass
            obs, info = env.reset(seed=cfg.base_seed + rank + episode)
            ep_savestate = info.get("episode_savestate")
            if segment_start_state is None:
                # Seed the first segment with a starting savestate (either from reset or resume)
                try:
                    if ep_savestate:
                        segment_start_state = bytes(ep_savestate)
                    else:
                        buf = BytesIO()
                        env.pyboy.save_state(buf)  # type: ignore[attr-defined]
                        segment_start_state = buf.getvalue()
                except Exception:
                    segment_start_state = b""
            hidden = policy.init_hidden(1, device)
            done = False
            ep_actions = []
            prev_obs_proc = preprocess_obs(obs)
            prev_info = info
            max_steps = cfg.max_steps_per_episode if not getattr(cfg, "continuous_mode", False) else 1_000_000_000
            for t in range(max_steps):
                if stop_event.is_set():
                    break
                step_counter += 1
                epsilon = epsilon_by_step(step_counter, cfg)
                context = build_context_features(info)
                obs_tensor = torch.tensor(prev_obs_proc, dtype=torch.float32, device=device).unsqueeze(0)
                ctx_tensor = torch.tensor(context, dtype=torch.float32, device=device).unsqueeze(0)
                policy.reset_noise()
                with torch.no_grad():
                    q_quantiles, next_hidden, _ = policy(obs_tensor, ctx_tensor, hidden)
                    q_vals = q_quantiles.mean(dim=2).squeeze(0).cpu().numpy()
                allow_mask = afford.get_allow_mask(prev_obs_proc.transpose(1, 2, 0), info)
                if np.random.rand() < epsilon:
                    # Epsilon action among allowed set
                    allowed_indices = [i for i, ok in enumerate(allow_mask) if ok]
                    if allowed_indices:
                        action = int(np.random.choice(allowed_indices))
                    else:
                        action = int(np.random.randint(env.action_space.n))
                else:
                    action = apply_action_mask(q_vals, allow_mask)
                # Execute with unit duration; sticky/durations can be added by repeating steps
                next_obs, reward, terminated, truncated, next_info = env.step(action)
                next_obs_proc = preprocess_obs(next_obs)
                next_context = build_context_features(next_info)
                afford.report_effect(prev_obs_proc.transpose(1, 2, 0), prev_info, action, next_obs, next_info)
                discount = cfg.gamma if not (terminated or truncated) else 0.0
                transitions.append((prev_obs_proc, context, action, float(reward), float(discount), next_obs_proc, next_context, bool(terminated or truncated)))
                hidden = next_hidden
                ep_actions.append(int(action))
                prev_obs_proc, prev_info = next_obs_proc, next_info
                # Periodic actor-side snapshots for resuming long runs
                now = time.monotonic()
                if now - last_snapshot_time >= snapshot_interval_s:
                    try:
                        buf = BytesIO()
                        env.pyboy.save_state(buf)  # type: ignore[attr-defined]
                        tmp = os.path.join(snapshots_dir, f"actor{rank}_{int(now)}.state.tmp")
                        final = os.path.join(snapshots_dir, f"actor{rank}_{int(now)}.state")
                        with open(tmp, "wb") as fh:
                            fh.write(buf.getvalue())
                        os.replace(tmp, final)
                        # Update latest pointer
                        latest = os.path.join(snapshots_dir, f"actor{rank}_latest.state")
                        try:
                            os.replace(final, latest)
                        except Exception:
                            # If replace failed because latest exists, copy bytes into latest
                            with open(final, "rb") as src, open(latest, "wb") as dst:
                                dst.write(src.read())
                        try:
                            print(f"[actor {rank}] snapshot saved at t={int(now - last_snapshot_time)}s")
                        except Exception:
                            pass
                    except Exception:
                        pass
                    last_snapshot_time = now
                # In continuous mode, flush segments without resetting
                if getattr(cfg, "continuous_mode", False) and len(transitions) >= max(1, int(getattr(cfg, "segment_length", 2000))):
                    # Ship segment
                    while not stop_event.is_set():
                        try:
                            out_queue.put(("episode", transitions), timeout=1.0)
                            break
                        except queue.Full:
                            continue
                    # Log segment for offline replay
                    if cfg.log_episodes_for_replay and log_dir:
                        try:
                            ensure_dir(log_dir)
                            seg_path = os.path.join(log_dir, f"actor{rank}_seg{segment_idx:05d}.npz")
                            start_bytes = segment_start_state or b""
                            np.savez_compressed(seg_path, actions=np.asarray(ep_actions, dtype=np.int16), savestate=np.frombuffer(start_bytes, dtype=np.uint8))
                        except Exception:
                            pass
                    segment_idx += 1
                    # Prepare next segment starting point
                    try:
                        buf = BytesIO()
                        env.pyboy.save_state(buf)  # type: ignore[attr-defined]
                        segment_start_state = buf.getvalue()
                    except Exception:
                        segment_start_state = b""
                    transitions = []
                    ep_actions = []
                    if segment_idx % 10 == 0 and last_seg_log != segment_idx:
                        try:
                            print(f"[actor {rank}] segments flushed: {segment_idx}")
                            last_seg_log = segment_idx
                        except Exception:
                            pass
                if terminated or truncated:
                    break
            if stop_event.is_set():
                break
            # Flush tail either as an episode (default) or as the final segment chunk
            if transitions:
                while not stop_event.is_set():
                    try:
                        out_queue.put(("episode", transitions), timeout=1.0)
                        break
                    except queue.Full:
                        continue
                if cfg.log_episodes_for_replay and log_dir:
                    try:
                        ensure_dir(log_dir)
                        if getattr(cfg, "continuous_mode", False):
                            seg_path = os.path.join(log_dir, f"actor{rank}_seg{segment_idx:05d}.npz")
                            start_bytes = segment_start_state or b""
                            np.savez_compressed(seg_path, actions=np.asarray(ep_actions, dtype=np.int16), savestate=np.frombuffer(start_bytes, dtype=np.uint8))
                            segment_idx += 1
                            # Prepare next segment start
                            try:
                                buf = BytesIO()
                                env.pyboy.save_state(buf)  # type: ignore[attr-defined]
                                segment_start_state = buf.getvalue()
                            except Exception:
                                segment_start_state = b""
                        else:
                            ep_path = os.path.join(log_dir, f"actor{rank}_ep{episode:05d}.npz")
                            savestate_bytes = ep_savestate if isinstance(ep_savestate, (bytes, bytearray)) else b""
                            np.savez_compressed(ep_path, actions=np.asarray(ep_actions, dtype=np.int16), savestate=np.frombuffer(savestate_bytes, dtype=np.uint8))
                    except Exception:
                        pass
            episode += 1
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


