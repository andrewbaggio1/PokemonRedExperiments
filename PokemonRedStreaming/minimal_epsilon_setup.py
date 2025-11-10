# All imports are local to epsilon/
from __future__ import annotations

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import random
from collections import deque, namedtuple
from dataclasses import dataclass
import json

import numpy as np
import torch
import torch.nn.functional as F

from env_pokemon import PokemonRedEnv
from epsilon_env import EpsilonEnv
from map_features import extract_map_features
from rewards.battle_outcome import BattleOutcomeReward
from rewards.battle_damage_reward import BattleDamageReward
from rewards.badge_reward import BadgeReward
from rewards.story_flag_reward import StoryFlagReward
from rewards.champion_reward import ChampionReward
from rewards.item_collection import ItemCollectionReward
from rewards.pokedex_reward import PokedexReward
from rewards.trainer_tier_reward import TrainerBattleReward
from rewards.map_exploration import MapExplorationReward
from rewards.novelty import NoveltyReward
from rewards.learned_map_embedding import LearnedMapEmbeddingReward
from rewards.map_visit_reward import MapVisitReward
from rewards.exploration_frontier_reward import ExplorationFrontierReward
from rewards.quest_reward import QuestReward
from rewards.efficiency_penalty import EfficiencyPenalty
from rewards.safety_penalty import SafetyPenalty
from rewards.resource_reward import ResourceManagementReward
from rewards.latent_event_reward import LatentEventReward
from rewards.map_stay_penalty import MapStayPenalty
from simple_dqn import SimpleDQN
from visualization import RouteMapVisualizer, MultiRouteMapVisualizer, GameplayGridVisualizer


Transition = namedtuple(
    "Transition",
    "obs map_feat action reward discount next_obs next_map_feat done",
)


def _coerce_int_list(value) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        if not value.strip():
            return []
        return [int(part.strip(), 0) for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [int(v) for v in value]
    return [int(value)]


def _coerce_float_tuple_list(value) -> list[tuple[int, float]]:
    if value is None:
        return []
    result: list[tuple[int, float]] = []
    if isinstance(value, str):
        # Expect JSON-style string: "count:reward,count:reward"
        if not value.strip():
            return []
        pairs = value.split(",")
        for pair in pairs:
            if ":" in pair:
                count_str, reward_str = pair.split(":", 1)
                try:
                    result.append((int(count_str.strip(), 0), float(reward_str.strip())))
                except ValueError:
                    continue
        return result
    for item in value:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            try:
                count = int(item[0])
                reward = float(item[1])
                result.append((count, reward))
            except (TypeError, ValueError):
                continue
    return result


def _coerce_str_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        if not value.strip():
            return []
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    return [str(value)]


@dataclass
class TrainConfig:
    rom_path: str
    episodes: int
    max_steps_per_episode: int
    gamma: float
    learning_rate: float
    buffer_size: int
    batch_size: int
    train_frequency: int
    learning_starts: int
    target_sync_interval: int
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_steps: int
    device: torch.device
    headless: bool
    save_dir: str
    save_every: int
    seed: int
    frame_skip: int
    emulation_speed: int
    boot_steps: int
    max_no_input_frames: int
    input_spacing_frames: int
    state_path: str | None
    render_map: bool
    map_refresh: int
    # MapExplorationReward hyperparameters
    mapexplore_base: float
    mapexplore_neighbor_radius: int
    mapexplore_neighbor_weight: float
    mapexplore_distance_weight: float
    mapexplore_min_reward: float
    mapexplore_persist: bool
    persist_map: bool
    save_map_images: bool
    map_image_every: int
    novelty_base: float
    novelty_decay: float
    novelty_min_reward: float
    novelty_stride: int
    novelty_quantisation: int
    novelty_persist: bool
    embedding_base: float
    embedding_decay: float
    embedding_min_reward: float
    embedding_include_map: bool
    embedding_persist: bool
    badge_reward: float
    story_flag_default_reward: float
    story_flags: dict | None
    champion_reward: float
    num_envs: int
    aggregate_map_refresh: int
    vectorized: bool
    battle_win_reward: float
    battle_loss_penalty: float
    n_step: int
    show_env_maps: bool
    log_interval: int
    verbose_logs: bool
    auxiliary_loss_coef: float
    progress_interval: int
    display_envs: int
    expose_visit_features: bool
    delete_sav_on_reset: bool
    battle_damage_scale: float
    battle_escape_penalty: float
    revisit_penalty_base: float
    revisit_penalty_excess: float
    revisit_penalty_ratio: float
    latent_event_reward: float
    latent_event_revisit_decay: float
    item_reward: float
    key_item_reward: float
    key_item_ids: list[int] | None
    pokedex_new_species_reward: float
    pokedex_milestones: list[tuple[int, float]] | None
    trainer_wild_reward: float
    trainer_trainer_reward: float
    trainer_gym_reward: float
    trainer_elite_reward: float
    gym_map_ids: list[int] | None
    elite_map_ids: list[int] | None
    quest_definitions: list[dict] | None
    pallet_penalty_map_id: int
    pallet_penalty_interval: int
    pallet_penalty: float
    frontier_reward: float
    frontier_min_gain: int
    map_visit_reward: float
    step_penalty: float
    idle_penalty: float
    idle_threshold: int
    loss_penalty: float
    blackout_penalty: float
    low_hp_threshold: float
    low_hp_penalty: float
    resource_map_keywords: list[str] | None
    resource_map_reward: float
    resource_item_ids: list[int] | None
    resource_item_reward: float
    curriculum_goals: list[dict] | None


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
    ):
        self.capacity = capacity
        self.storage = []
        self.position = 0
        self.alpha = alpha
        self.beta = beta_start
        self.beta_increment = (1.0 - beta_start) / max(1, beta_frames)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0

    def push(
        self,
        obs: np.ndarray,
        map_feat: np.ndarray,
        action: int,
        reward: float,
        discount: float,
        next_obs: np.ndarray,
        next_map_feat: np.ndarray,
        done: bool,
    ) -> None:
        if len(self.storage) < self.capacity:
            self.storage.append(None)
        self.storage[self.position] = Transition(
            np.array(obs, copy=True),
            np.array(map_feat, copy=True),
            int(action),
            float(reward),
            float(discount),
            np.array(next_obs, copy=True),
            np.array(next_map_feat, copy=True),
            bool(done),
        )
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Transition:
        size = len(self.storage)
        if size == 0:
            raise ValueError("Cannot sample from empty buffer")
        priorities = self.priorities[:size]
        if not np.any(priorities):
            priorities = np.ones(size, dtype=np.float32)
        scaled = priorities ** self.alpha
        probs = scaled / scaled.sum()
        indices = np.random.choice(size, batch_size, p=probs)
        samples = [self.storage[i] for i in indices]

        weights = (size * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)

        batch = Transition(*zip(*samples))
        return batch, weights.astype(np.float32), indices

    def __len__(self) -> int:
        return len(self.storage)

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            value = float(abs(priority) + 1e-6)
            self.priorities[idx] = value
            self.max_priority = max(self.max_priority, value)


class CatchPokemonReward:
    """Encourages catching a wild encounter on Route 1."""

    def __init__(
        self,
        catch_bonus: float = 150.0,
        encounter_bonus: float = 1.0,
        step_penalty: float = -0.02,
        off_route_penalty: float = -0.2,
        escape_penalty: float = -5.0,
        target_map: str | int = "route 1",
    ):
        self.catch_bonus = catch_bonus
        self.encounter_bonus = encounter_bonus
        self.step_penalty = step_penalty
        self.off_route_penalty = off_route_penalty
        self.escape_penalty = escape_penalty
        self.target_map = (
            target_map.lower() if isinstance(target_map, str) else target_map
        )
        self._catch_awarded = False

    def compute(self, obs, info) -> float:
        reward = self.step_penalty
        # Removed off-route penalty per user preference.
        if info.get("in_battle"):
            reward += self.encounter_bonus
        if info.get("battle_result") == "escaped":
            reward += self.escape_penalty
        if info.get("caught_pokemon") and not self._catch_awarded:
            self._catch_awarded = True
            reward += self.catch_bonus
        return reward

    def reset(self) -> None:
        self._catch_awarded = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preprocess_obs(obs: np.ndarray) -> np.ndarray:
    obs = obs.astype(np.float32) / 255.0
    if obs.ndim == 3:
        obs = np.transpose(obs, (2, 0, 1))  # (C, H, W)
    return obs


def epsilon_by_step(step: int, cfg: TrainConfig) -> float:
    if cfg.epsilon_decay_steps <= 0:
        return cfg.epsilon_end
    fraction = min(1.0, step / cfg.epsilon_decay_steps)
    return cfg.epsilon_start + fraction * (cfg.epsilon_end - cfg.epsilon_start)


def batch_to_tensors(batch: Transition, device: torch.device):
    obs = torch.tensor(np.stack(batch.obs), dtype=torch.float32, device=device)
    map_feat = torch.tensor(
        np.stack(batch.map_feat), dtype=torch.float32, device=device
    )
    actions = torch.tensor(batch.action, dtype=torch.long, device=device)
    rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device)
    discounts = torch.tensor(batch.discount, dtype=torch.float32, device=device)
    next_obs = torch.tensor(np.stack(batch.next_obs), dtype=torch.float32, device=device)
    next_map_feat = torch.tensor(
        np.stack(batch.next_map_feat), dtype=torch.float32, device=device
    )
    dones = torch.tensor(batch.done, dtype=torch.float32, device=device)
    return obs, map_feat, actions, rewards, discounts, next_obs, next_map_feat, dones


def build_reward_modules(cfg: TrainConfig):
    modules: list = []

    if cfg.mapexplore_base > 0.0:
        modules.append(
            MapExplorationReward(
                base_reward=cfg.mapexplore_base,
                neighbor_radius=cfg.mapexplore_neighbor_radius,
                neighbor_weight=cfg.mapexplore_neighbor_weight,
                distance_weight=cfg.mapexplore_distance_weight,
                min_reward=cfg.mapexplore_min_reward,
                persist_across_episodes=cfg.mapexplore_persist,
            )
        )

    if cfg.novelty_base > 0.0:
        modules.append(
            NoveltyReward(
                base_reward=cfg.novelty_base,
                decay=cfg.novelty_decay,
                min_reward=cfg.novelty_min_reward,
                sample_stride=cfg.novelty_stride,
                quantisation=cfg.novelty_quantisation,
                persist_across_episodes=cfg.novelty_persist,
            )
        )

    if cfg.embedding_base > 0.0:
        modules.append(
            LearnedMapEmbeddingReward(
                base_reward=cfg.embedding_base,
                decay=cfg.embedding_decay,
                min_reward=cfg.embedding_min_reward,
                include_map_id=cfg.embedding_include_map,
                persist_across_episodes=cfg.embedding_persist,
            )
        )

    modules.append(CatchPokemonReward())

    modules.append(
        BattleOutcomeReward(
            win_reward=cfg.battle_win_reward,
            loss_penalty=cfg.battle_loss_penalty,
        )
    )
    if cfg.battle_damage_scale != 0.0 or cfg.battle_escape_penalty != 0.0:
        modules.append(
            BattleDamageReward(
                damage_scale=cfg.battle_damage_scale,
                escape_penalty=cfg.battle_escape_penalty,
            )
        )

    modules.append(BadgeReward(reward_per_badge=cfg.badge_reward))
    modules.append(
        StoryFlagReward(cfg.story_flags, default_reward=cfg.story_flag_default_reward)
    )
    modules.append(ChampionReward(reward=cfg.champion_reward))

    if cfg.map_visit_reward > 0.0:
        modules.append(MapVisitReward(map_reward=cfg.map_visit_reward))

    if cfg.frontier_reward > 0.0:
        modules.append(
            ExplorationFrontierReward(
                distance_reward=cfg.frontier_reward,
                min_gain=cfg.frontier_min_gain,
            )
        )

    if cfg.quest_definitions:
        modules.append(QuestReward(cfg.quest_definitions))

    if cfg.pallet_penalty != 0.0:
        modules.append(
            MapStayPenalty(
                map_id=cfg.pallet_penalty_map_id,
                interval=cfg.pallet_penalty_interval,
                penalty=cfg.pallet_penalty,
            )
        )

    modules.append(
        PokedexReward(
            new_species_reward=cfg.pokedex_new_species_reward,
            milestone_rewards=cfg.pokedex_milestones,
        )
    )
    modules.append(
        ItemCollectionReward(
            item_reward=cfg.item_reward,
            key_item_reward=cfg.key_item_reward,
            key_item_ids=cfg.key_item_ids,
        )
    )

    modules.append(
        EfficiencyPenalty(
            step_penalty=cfg.step_penalty,
            idle_penalty=cfg.idle_penalty,
            idle_threshold=cfg.idle_threshold,
        )
    )
    modules.append(
        SafetyPenalty(
            loss_penalty=cfg.loss_penalty,
            blackout_penalty=cfg.blackout_penalty,
            low_hp_threshold=cfg.low_hp_threshold,
            low_hp_penalty=cfg.low_hp_penalty,
        )
    )

    if cfg.resource_map_reward or cfg.resource_item_reward:
        modules.append(
            ResourceManagementReward(
                map_keywords=cfg.resource_map_keywords,
                map_reward=cfg.resource_map_reward,
                utility_item_ids=cfg.resource_item_ids,
                item_reward=cfg.resource_item_reward,
            )
        )

    if cfg.latent_event_reward:
        modules.append(
            LatentEventReward(
                base_reward=cfg.latent_event_reward,
                revisit_decay=cfg.latent_event_revisit_decay,
            )
        )

    modules.append(
        TrainerBattleReward(
            wild_reward=cfg.trainer_wild_reward,
            trainer_reward=cfg.trainer_trainer_reward,
            gym_reward=cfg.trainer_gym_reward,
            elite_reward=cfg.trainer_elite_reward,
            gym_map_ids=cfg.gym_map_ids,
            elite_map_ids=cfg.elite_map_ids,
        )
    )

    return modules


def make_env(cfg: TrainConfig) -> PokemonRedEnv:
    return PokemonRedEnv(
        rom_path=cfg.rom_path,
        show_display=not cfg.headless,
        frame_skip=cfg.frame_skip,
        emulation_speed=cfg.emulation_speed,
        boot_steps=cfg.boot_steps,
        max_no_input_frames=cfg.max_no_input_frames,
        state_path=cfg.state_path,
        story_flag_defs=cfg.story_flags,
        track_visit_stats=cfg.expose_visit_features,
        delete_sav_on_reset=cfg.delete_sav_on_reset,
        input_spacing_frames=cfg.input_spacing_frames,
    )

def make_env_instance(cfg: TrainConfig, env_index: int) -> PokemonRedEnv:
    # Show a PyBoy window for a limited number of environments unless headless is requested.
    show_display = not cfg.headless and env_index < max(0, cfg.display_envs)
    return PokemonRedEnv(
        rom_path=cfg.rom_path,
        show_display=show_display,
        frame_skip=cfg.frame_skip,
        emulation_speed=cfg.emulation_speed,
        boot_steps=cfg.boot_steps,
        max_no_input_frames=cfg.max_no_input_frames,
        state_path=cfg.state_path,
        story_flag_defs=cfg.story_flags,
        track_visit_stats=cfg.expose_visit_features,
        delete_sav_on_reset=cfg.delete_sav_on_reset,
        input_spacing_frames=cfg.input_spacing_frames,
    )


def compute_td_loss(
    online_net: SimpleDQN,
    target_net: SimpleDQN,
    optimizer: torch.optim.Optimizer,
    batch_tensors,
    weights: torch.Tensor,
    auxiliary_coef: float,
) -> tuple[float, torch.Tensor]:
    online_net.reset_noise()
    target_net.reset_noise()
    obs, map_feat, actions, rewards, discounts, next_obs, next_map_feat, dones = (
        batch_tensors
    )
    quantiles, _, aux_pred = online_net(obs, map_feat)
    batch_size = quantiles.size(0)
    num_quantiles = quantiles.size(-1)
    action_index = actions.unsqueeze(-1).unsqueeze(-1).expand(
        batch_size, 1, num_quantiles
    )
    chosen_quantiles = quantiles.gather(1, action_index).squeeze(1)

    with torch.no_grad():
        next_quantiles_online, _, _ = online_net(next_obs, next_map_feat)
        next_actions = next_quantiles_online.mean(dim=2).argmax(dim=1)
        next_action_index = next_actions.unsqueeze(-1).unsqueeze(-1).expand(
            batch_size, 1, num_quantiles
        )
        next_quantiles_target, _, _ = target_net(next_obs, next_map_feat)
        next_chosen_quantiles = next_quantiles_target.gather(1, next_action_index).squeeze(1)
        target_quantiles = rewards.unsqueeze(1) + discounts.unsqueeze(1) * (1.0 - dones.unsqueeze(1)) * next_chosen_quantiles

    tau = (torch.arange(num_quantiles, device=obs.device, dtype=torch.float32) + 0.5) / num_quantiles
    tau = tau.view(1, num_quantiles, 1)

    diff = target_quantiles.unsqueeze(1) - chosen_quantiles.unsqueeze(2)
    huber = torch.where(diff.abs() <= 1.0, 0.5 * diff.pow(2), diff.abs() - 0.5)
    quantile_loss = torch.abs(tau - (diff < 0).float()) * huber
    quantile_loss = quantile_loss.mean(dim=2).sum(dim=1)

    td_loss = (weights * quantile_loss).mean()
    aux_loss = F.mse_loss(aux_pred, map_feat, reduction="none").mean(dim=1)
    aux_loss = (weights * aux_loss).mean()
    loss = td_loss + auxiliary_coef * aux_loss
    td_errors = (target_quantiles.mean(dim=1) - chosen_quantiles.mean(dim=1))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online_net.parameters(), 10.0)
    optimizer.step()
    return float(loss.item()), td_errors.detach()


def _move_hidden_to_device(hidden, device):
    if hidden is None:
        return None
    gru = hidden.get("gru")
    lstm = hidden.get("lstm")
    if gru is None or lstm is None:
        return None
    lstm_h, lstm_c = lstm
    return {
        "gru": gru.to(device),
        "lstm": (lstm_h.to(device), lstm_c.to(device)),
    }


def _detach_hidden(hidden):
    if hidden is None:
        return None
    gru = hidden.get("gru")
    lstm = hidden.get("lstm")
    if gru is None or lstm is None:
        return None
    lstm_h, lstm_c = lstm
    return {
        "gru": gru.detach(),
        "lstm": (lstm_h.detach(), lstm_c.detach()),
    }


def select_action_eps(
    model: SimpleDQN,
    obs: np.ndarray,
    map_feat: np.ndarray,
    epsilon: float,
    action_space,
    device: torch.device,
    hidden_state=None,
) -> tuple[int, dict | None]:
    model.reset_noise()
    if np.random.rand() < epsilon:
        return action_space.sample(), hidden_state
    obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
    map_tensor = torch.from_numpy(map_feat).float().unsqueeze(0).to(device)
    hidden = _move_hidden_to_device(hidden_state, device)
    with torch.no_grad():
        quantiles, next_hidden, _ = model(obs_tensor, map_tensor, hidden)
    q_mean = quantiles.mean(dim=2)
    action = int(torch.argmax(q_mean, dim=1).item())
    return action, _detach_hidden(next_hidden)


def resolve_device(device_spec: str) -> torch.device:
    if device_spec != "auto":
        return torch.device(device_spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def maybe_save(model: SimpleDQN, path: str) -> None:
    torch.save(model.state_dict(), path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a DQN with epsilon-greedy policy to catch a Pokemon on Route 1."
    )
    parser.set_defaults(story_flags=None)
    default_rom = os.path.join(os.path.dirname(__file__), "pokemon_red.gb")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to JSON config file (defaults to training_config.json if present).",
    )
    parser.add_argument("--rom", default=default_rom, help="Path to Pokemon Red ROM.")
    parser.add_argument("--episodes", type=int, default=400, help="Training episodes.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=2000,
        help="Max environment steps per episode.",
    )
    parser.add_argument(
        "--buffer-size", type=int, default=100000, help="Replay buffer capacity."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Mini-batch size for updates."
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Optimizer learning rate.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument(
        "--learning-starts",
        type=int,
        default=5000,
        help="Number of steps before training begins.",
    )
    parser.add_argument(
        "--train-frequency",
        type=int,
        default=4,
        help="How often (in steps) to apply a gradient update.",
    )
    parser.add_argument(
        "--n-step",
        type=int,
        default=3,
        help="Number of steps to accumulate for n-step returns.",
    )
    parser.add_argument(
        "--target-sync",
        type=int,
        default=4000,
        help="Sync interval (in steps) for the target network.",
    )
    parser.add_argument(
        "--epsilon-start", type=float, default=1.0, help="Initial epsilon value."
    )
    parser.add_argument(
        "--epsilon-end", type=float, default=0.05, help="Final epsilon value."
    )
    parser.add_argument(
        "--epsilon-decay-steps",
        type=int,
        default=200000,
        help="Number of steps over which epsilon decays.",
    )
    parser.add_argument("--seed", type=int, default=7, help="PRNG seed.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device spec (e.g. cpu, cuda, mps) or 'auto'.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run PyBoy in headless mode to avoid rendering overhead.",
    )
    parser.add_argument(
        "--save-dir",
        default=os.path.join(os.path.dirname(__file__), "checkpoints"),
        help="Directory to store checkpoints.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="Episode interval for periodic checkpointing.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        help="How often to print per-step logs during training (in environment steps).",
    )
    parser.add_argument(
        "--headless-mode",
        action="store_true",
        default=False,
        help="Convenience flag: disable rendering/logging and force headless PyBoy for overnight runs.",
    )
    parser.add_argument(
        "--no-logs",
        action="store_false",
        dest="verbose_logs",
        help="Disable verbose episode/step logging (useful for headless overnight runs).",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=10,
        help="Episode interval for always printing a summary even when verbose logs are disabled.",
    )
    parser.add_argument(
        "--display-envs",
        type=int,
        default=1,
        help="Number of PyBoy windows to display when rendering gameplay.",
    )
    parser.add_argument(
        "--auxiliary-loss-coef",
        type=float,
        default=0.05,
        help="Weight applied to the auxiliary representation reconstruction loss.",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=4,
        help="Number of emulator frames to advance per environment step.",
    )
    parser.add_argument(
        "--emulation-speed",
        type=float,
        default=1.0,
        help="Speed multiplier for PyBoy when a display window is shown (set 0 for unlimited).",
    )
    parser.add_argument(
        "--input-spacing-frames",
        type=int,
        default=0,
        help="Number of blank frames to insert after each action to avoid rapid button mashing.",
    )
    parser.add_argument(
        "--boot-steps",
        type=int,
        default=120,
        help="Steps to fast-forward after reset before control begins.",
    )
    parser.add_argument(
        "--no-input-timeout",
        type=int,
        default=600,
        help="Terminate an episode if no inputs were pressed for this many frames.",
    )
    parser.add_argument(
        "--state-path",
        default=None,
        help="Optional PyBoy state file to load on reset (starts near Route 1).",
    )
    parser.add_argument(
        "--render-map",
        action="store_true",
        help="Render a live Route 1 occupancy map alongside gameplay.",
    )
    parser.add_argument(
        "--map-refresh",
        type=int,
        default=4,
        help="Update the map visual every N environment steps.",
    )
    parser.add_argument(
        "--persist-map",
        action="store_true",
        help="Do not clear the occupancy map between episodes.",
    )
    parser.add_argument(
        "--save-map-images",
        action="store_true",
        help="Save map PNGs periodically to the checkpoint directory.",
    )
    parser.add_argument(
        "--map-image-every",
        type=int,
        default=10,
        help="Episode interval for saving map images when enabled.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of parallel environments to run for training.",
    )
    parser.add_argument(
        "--aggregate-map-refresh",
        type=int,
        default=8,
        help="Update frequency (steps) for the aggregate map visual in parallel mode.",
    )
    parser.add_argument(
        "--show-env-maps",
        action="store_true",
        default=True,
        help="Display per-environment map panels alongside the aggregate view.",
    )
    parser.add_argument(
        "--no-show-env-maps",
        action="store_false",
        dest="show_env_maps",
        help="Hide individual environment map panels and show only the aggregate view.",
    )
    parser.add_argument(
        "--vectorized",
        action="store_true",
        help="Enable multiprocessing vectorized environment workers.",
    )
    parser.add_argument(
        "--story-flags-json",
        default=None,
        help="Optional JSON file describing story flag memory locations and rewards.",
    )
    parser.add_argument(
        "--badge-reward",
        type=float,
        default=200.0,
        help="Reward applied when a new gym badge is obtained.",
    )
    parser.add_argument(
        "--story-flag-default-reward",
        type=float,
        default=150.0,
        help="Default reward applied when a configured story flag becomes true.",
    )
    parser.add_argument(
        "--champion-reward",
        type=float,
        default=1000.0,
        help="Reward applied when the Champion is defeated.",
    )
    # Novelty reward controls
    parser.add_argument(
        "--novelty-base",
        type=float,
        default=1.0,
        help="Base reward for novel screens before decay.",
    )
    parser.add_argument(
        "--novelty-decay",
        type=float,
        default=0.9,
        help="Multiplicative decay applied on each revisit of the same token.",
    )
    parser.add_argument(
        "--novelty-min-reward",
        type=float,
        default=0.0,
        help="Minimum novelty reward after decay.",
    )
    parser.add_argument(
        "--novelty-stride",
        type=int,
        default=4,
        help="Stride used when downsampling observations for novelty hashing.",
    )
    parser.add_argument(
        "--novelty-quantisation",
        type=int,
        default=32,
        help="Quantisation bucket size for novelty hashing (higher = more tolerant).",
    )
    # Learned map embedding controls
    parser.add_argument(
        "--embedding-base",
        type=float,
        default=1.0,
        help="Base reward for new coordinate embeddings.",
    )
    parser.add_argument(
        "--embedding-decay",
        type=float,
        default=0.9,
        help="Decay applied each time an embedding is revisited.",
    )
    parser.add_argument(
        "--embedding-min-reward",
        type=float,
        default=0.0,
        help="Minimum embedding reward after decay.",
    )
    parser.add_argument(
        "--embedding-include-map",
        action="store_true",
        default=True,
        help="Include map_id in the embedding key (can disable with --no-embedding-include-map).",
    )
    parser.add_argument(
        "--no-embedding-include-map",
        action="store_false",
        dest="embedding_include_map",
        help="Do not include map_id in the embedding key.",
    )
    # MapExplorationReward knobs
    parser.add_argument(
        "--mapexplore-base", type=float, default=1.0,
        help="Base reward for a brand new coordinate."
    )
    parser.add_argument(
        "--mapexplore-neighbor-radius", type=int, default=1,
        help="Chebyshev radius for neighbor density penalty."
    )
    parser.add_argument(
        "--mapexplore-neighbor-weight", type=float, default=0.15,
        help="Penalty per visited neighbor within radius."
    )
    parser.add_argument(
        "--mapexplore-distance-weight", type=float, default=0.5,
        help="Bonus scale for distance from episode start."
    )
    parser.add_argument(
        "--mapexplore-min-reward", type=float, default=0.05,
        help="Minimum reward floor for a new coordinate."
    )
    parser.add_argument(
        "--battle-win-reward",
        type=float,
        default=20.0,
        help="Bonus reward when the agent wins a trainer battle.",
    )
    parser.add_argument(
        "--battle-loss-penalty",
        type=float,
        default=-15.0,
        help="Penalty applied when a trainer battle is lost or results in blackout.",
    )
    args = parser.parse_args()

    defaults: dict[str, object] = {}
    for name in vars(args):
        try:
            defaults[name] = parser.get_default(name)
        except (AttributeError, KeyError):
            continue

    additional_defaults = {
        "item_reward": 1.0,
        "key_item_reward": 5.0,
        "pokedex_new_species_reward": 10.0,
        "trainer_wild_reward": 5.0,
        "trainer_trainer_reward": 20.0,
        "trainer_gym_reward": 100.0,
        "trainer_elite_reward": 250.0,
        "frontier_reward": 1.0,
        "frontier_min_gain": 1,
        "map_visit_reward": 5.0,
        "step_penalty": -0.001,
        "idle_penalty": -0.1,
        "idle_threshold": 20,
        "loss_penalty": -25.0,
        "blackout_penalty": -50.0,
        "low_hp_threshold": 0.1,
        "low_hp_penalty": -2.0,
        "resource_map_reward": 5.0,
        "resource_item_reward": 2.0,
        "auxiliary_loss_coef": 0.05,
        "latent_event_reward": 25.0,
        "latent_event_revisit_decay": 0.5,
        "verbose_logs": True,
        "headless_mode": False,
        "progress_interval": 10,
        "display_envs": 1,
        "mapexplore_persist": True,
        "novelty_persist": True,
        "embedding_persist": True,
        "expose_visit_features": True,
        "delete_sav_on_reset": True,
        "battle_damage_scale": 6.0,
        "battle_escape_penalty": -12.0,
        "revisit_penalty_base": 0.02,
        "revisit_penalty_excess": 0.01,
        "revisit_penalty_ratio": 0.015,
        "pallet_penalty_map_id": 0,
        "pallet_penalty_interval": 200,
        "pallet_penalty": 0.0,
    }
    list_defaults = {
        "key_item_ids": [],
        "pokedex_milestones": [],
        "gym_map_ids": [],
        "elite_map_ids": [],
        "quest_definitions": [],
        "resource_map_keywords": ["pokemon center"],
        "resource_item_ids": [],
        "curriculum_goals": [],
    }

    for key, value in additional_defaults.items():
        if key not in defaults:
            defaults[key] = value
        if not hasattr(args, key):
            setattr(args, key, value)

    for key, value in list_defaults.items():
        if key not in defaults:
            defaults[key] = value
        if not hasattr(args, key):
            setattr(args, key, list(value) if isinstance(value, (list, tuple)) else value)
    config_path = args.config
    config_dir = None
    if not config_path:
        candidate = os.path.join(os.path.dirname(__file__), "training_config.json")
        if os.path.exists(candidate):
            config_path = candidate

    if config_path:
        try:
            with open(config_path, "r", encoding="utf-8") as fh:
                config_data = json.load(fh)
        except Exception as exc:
            print(f"[config] Failed to load {config_path}: {exc}")
        else:
            config_dir = os.path.dirname(os.path.abspath(config_path))
            path_keys = {"rom", "save_dir", "state_path", "story_flags_json"}
            for key, value in config_data.items():
                if key not in defaults:
                    continue
                if key in path_keys and isinstance(value, str) and value:
                    if not os.path.isabs(value):
                        value = os.path.normpath(os.path.join(config_dir, value))
                if getattr(args, key) == defaults[key]:
                    setattr(args, key, value)
            args.config = config_path
            print(f"[config] Loaded parameters from {config_path}")

    # Apply headless-mode presets if requested (via CLI or config).
    headless_mode = getattr(args, "headless_mode", False)
    if headless_mode:
        setattr(args, "headless", True)
        setattr(args, "render_map", False)
        setattr(args, "save_map_images", False)
        # Only force quiet logging if the user did not override it.
        verbose_default = defaults.get("verbose_logs", True)
        if getattr(args, "verbose_logs", verbose_default) == verbose_default:
            setattr(args, "verbose_logs", False)

    # Load story flag definitions from JSON file if supplied.
    story_flags = getattr(args, "story_flags", None)
    story_flags_json = getattr(args, "story_flags_json", None)
    if story_flags_json:
        json_path = story_flags_json
        if isinstance(json_path, str) and not os.path.isabs(json_path) and config_dir:
            json_path = os.path.normpath(os.path.join(config_dir, json_path))
        try:
            with open(json_path, "r", encoding="utf-8") as fh:
                story_flags = json.load(fh)
        except Exception as exc:
            print(f"[config] Failed to load story flags from {story_flags_json}: {exc}")
        else:
            setattr(args, "story_flags", story_flags)

    if story_flags is None:
        setattr(args, "story_flags", None)

    return args


def build_config(args) -> TrainConfig:
    key_item_ids = _coerce_int_list(getattr(args, "key_item_ids", []))
    pokedex_milestones = _coerce_float_tuple_list(getattr(args, "pokedex_milestones", []))
    gym_map_ids = _coerce_int_list(getattr(args, "gym_map_ids", []))
    elite_map_ids = _coerce_int_list(getattr(args, "elite_map_ids", []))
    resource_map_keywords = _coerce_str_list(getattr(args, "resource_map_keywords", []))
    resource_item_ids = _coerce_int_list(getattr(args, "resource_item_ids", []))
    quest_definitions = getattr(args, "quest_definitions", []) or []
    curriculum_goals = getattr(args, "curriculum_goals", []) or []
    return TrainConfig(
        rom_path=args.rom,
        episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        gamma=args.gamma,
        learning_rate=args.lr,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        train_frequency=args.train_frequency,
        learning_starts=args.learning_starts,
        target_sync_interval=args.target_sync,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        device=resolve_device(args.device),
        headless=args.headless,
        save_dir=args.save_dir,
        save_every=args.save_every,
        seed=args.seed,
        frame_skip=args.frame_skip,
        emulation_speed=max(0, int(round(getattr(args, "emulation_speed", 1)))),
        input_spacing_frames=max(0, int(getattr(args, "input_spacing_frames", 0))),
        boot_steps=args.boot_steps,
        max_no_input_frames=args.no_input_timeout,
        state_path=args.state_path if args.state_path else None,
        render_map=args.render_map,
        map_refresh=max(1, args.map_refresh),
        mapexplore_base=args.mapexplore_base,
        mapexplore_neighbor_radius=args.mapexplore_neighbor_radius,
        mapexplore_neighbor_weight=args.mapexplore_neighbor_weight,
        mapexplore_distance_weight=args.mapexplore_distance_weight,
        mapexplore_min_reward=args.mapexplore_min_reward,
        mapexplore_persist=bool(getattr(args, "mapexplore_persist", True)),
        persist_map=args.persist_map,
        save_map_images=args.save_map_images,
        map_image_every=max(1, args.map_image_every),
        novelty_base=args.novelty_base,
        novelty_decay=args.novelty_decay,
        novelty_min_reward=args.novelty_min_reward,
        novelty_stride=args.novelty_stride,
        novelty_quantisation=args.novelty_quantisation,
        novelty_persist=bool(getattr(args, "novelty_persist", True)),
        embedding_base=args.embedding_base,
        embedding_decay=args.embedding_decay,
        embedding_min_reward=args.embedding_min_reward,
        embedding_include_map=args.embedding_include_map,
        embedding_persist=bool(getattr(args, "embedding_persist", True)),
        battle_win_reward=args.battle_win_reward,
        battle_loss_penalty=args.battle_loss_penalty,
        badge_reward=args.badge_reward,
        story_flag_default_reward=args.story_flag_default_reward,
        story_flags=getattr(args, "story_flags", None),
        champion_reward=args.champion_reward,
        num_envs=max(1, args.num_envs),
        aggregate_map_refresh=max(1, args.aggregate_map_refresh),
        vectorized=args.vectorized,
        n_step=max(1, args.n_step),
        show_env_maps=bool(args.show_env_maps),
        log_interval=max(1, args.log_interval),
        verbose_logs=bool(getattr(args, "verbose_logs", True)),
        auxiliary_loss_coef=float(getattr(args, "auxiliary_loss_coef", 0.05)),
        progress_interval=max(1, int(getattr(args, "progress_interval", 10))),
        display_envs=max(0, int(getattr(args, "display_envs", 1))),
        expose_visit_features=bool(getattr(args, "expose_visit_features", True)),
        delete_sav_on_reset=bool(getattr(args, "delete_sav_on_reset", True)),
        battle_damage_scale=float(getattr(args, "battle_damage_scale", 6.0)),
        battle_escape_penalty=float(getattr(args, "battle_escape_penalty", -12.0)),
        revisit_penalty_base=float(getattr(args, "revisit_penalty_base", 0.02)),
        revisit_penalty_excess=float(getattr(args, "revisit_penalty_excess", 0.01)),
        revisit_penalty_ratio=float(getattr(args, "revisit_penalty_ratio", 0.015)),
        latent_event_reward=float(getattr(args, "latent_event_reward", 25.0)),
        latent_event_revisit_decay=float(getattr(args, "latent_event_revisit_decay", 0.5)),
        item_reward=float(getattr(args, "item_reward", 1.0)),
        key_item_reward=float(getattr(args, "key_item_reward", 5.0)),
        key_item_ids=key_item_ids,
        pokedex_new_species_reward=float(getattr(args, "pokedex_new_species_reward", 10.0)),
        pokedex_milestones=pokedex_milestones,
        trainer_wild_reward=float(getattr(args, "trainer_wild_reward", 5.0)),
        trainer_trainer_reward=float(getattr(args, "trainer_trainer_reward", 20.0)),
        trainer_gym_reward=float(getattr(args, "trainer_gym_reward", 100.0)),
        trainer_elite_reward=float(getattr(args, "trainer_elite_reward", 250.0)),
        gym_map_ids=gym_map_ids,
        elite_map_ids=elite_map_ids,
        quest_definitions=quest_definitions,
        pallet_penalty_map_id=int(getattr(args, "pallet_penalty_map_id", 0)),
        pallet_penalty_interval=max(1, int(getattr(args, "pallet_penalty_interval", 200))),
        pallet_penalty=float(getattr(args, "pallet_penalty", 0.0)),
        frontier_reward=float(getattr(args, "frontier_reward", 1.0)),
        frontier_min_gain=max(1, int(getattr(args, "frontier_min_gain", 1))),
        map_visit_reward=float(getattr(args, "map_visit_reward", 5.0)),
        step_penalty=float(getattr(args, "step_penalty", -0.001)),
        idle_penalty=float(getattr(args, "idle_penalty", -0.1)),
        idle_threshold=max(1, int(getattr(args, "idle_threshold", 20))),
        loss_penalty=float(getattr(args, "loss_penalty", -25.0)),
        blackout_penalty=float(getattr(args, "blackout_penalty", -50.0)),
        low_hp_threshold=float(getattr(args, "low_hp_threshold", 0.1)),
        low_hp_penalty=float(getattr(args, "low_hp_penalty", -2.0)),
        resource_map_keywords=resource_map_keywords,
        resource_map_reward=float(getattr(args, "resource_map_reward", 5.0)),
        resource_item_ids=resource_item_ids,
        resource_item_reward=float(getattr(args, "resource_item_reward", 2.0)),
        curriculum_goals=curriculum_goals,
    )


def train(cfg: TrainConfig) -> None:
    os.makedirs(cfg.save_dir, exist_ok=True)
    set_seed(cfg.seed)
    num_envs = max(1, cfg.num_envs)
    envs = [
        EpsilonEnv(make_env_instance(cfg, idx), build_reward_modules(cfg))
        for idx in range(num_envs)
    ]

    probe_obs, probe_info = envs[0].reset(seed=cfg.seed)
    obs_shape = (probe_obs.shape[2], probe_obs.shape[0], probe_obs.shape[1]) if probe_obs.ndim == 3 else probe_obs.shape
    map_feat_dim = extract_map_features(probe_info).shape[0]
    n_actions = envs[0].action_space.n
    map_viz = None
    gameplay_viz = None
    if cfg.render_map:
        try:
            map_viz = MultiRouteMapVisualizer(num_envs, show_env_panels=cfg.show_env_maps)
        except RuntimeError as exc:
            print(f"[visualization] Disabling map view: {exc}")
            map_viz = None

    device = cfg.device
    online_net = SimpleDQN(obs_shape, map_feat_dim, n_actions).to(device)
    target_net = SimpleDQN(obs_shape, map_feat_dim, n_actions).to(device)
    target_net.load_state_dict(online_net.state_dict())
    optimizer = torch.optim.Adam(online_net.parameters(), lr=cfg.learning_rate)
    replay_buffer = ReplayBuffer(cfg.buffer_size)
    n_step = max(1, cfg.n_step)
    gamma = cfg.gamma
    nstep_buffers = [deque() for _ in range(num_envs)]
    actor_hidden = [online_net.init_hidden(1, device) for _ in range(num_envs)]

    def flush_nstep(idx: int, force: bool = False) -> None:
        queue = nstep_buffers[idx]
        while queue and (force or len(queue) >= n_step):
            reward_sum = 0.0
            steps = 0
            done_final = False
            next_obs_final = queue[0][4]
            next_map_final = queue[0][5]
            for obs_tuple in list(queue)[:n_step]:
                _, _, _, r, nxt_obs, nxt_map, done_flag = obs_tuple
                reward_sum += (gamma ** steps) * r
                steps += 1
                next_obs_final = nxt_obs
                next_map_final = nxt_map
                if done_flag:
                    done_final = True
                    break
            discount = 0.0 if done_final else gamma ** steps
            first = queue[0]
            replay_buffer.push(
                first[0],
                first[1],
                first[2],
                reward_sum,
                discount,
                next_obs_final,
                next_map_final,
                done_final,
            )
            queue.popleft()
            if not force and len(queue) < n_step:
                break

    global_step = 0
    reward_window = deque(maxlen=25)
    best_reward = -float("inf")
    last_loss = None
    best_model_path = os.path.join(cfg.save_dir, "dqn_route1_best.pt")
    map_refresh = max(1, cfg.map_refresh)
    aggregate_refresh = max(1, cfg.aggregate_map_refresh)
    log_interval = max(1, cfg.log_interval)

    print(f"Starting training on device {device}. ROM: {cfg.rom_path}")
    try:
        for episode in range(cfg.episodes):
            obs_list = []
            info_list = []
            done = [False] * num_envs
            episode_rewards = [0.0] * num_envs
            env_steps = [0] * num_envs
            for idx, env in enumerate(envs):
                seed = cfg.seed + episode * num_envs + idx
                obs, info = env.reset(seed=seed)
                raw_frame = info.get("raw_frame")
                obs_list.append(preprocess_obs(obs))
                info_list.append(info)
                actor_hidden[idx] = online_net.init_hidden(1, device)
                if cfg.verbose_logs:
                    print(
                        f"[episode {episode + 1:04d}] env {idx + 1}: map={info.get('map_name')} "
                        f"coords={info.get('agent_coords')}",
                        flush=True,
                    )
                if cfg.render_map and gameplay_viz is None:
                    frame_shape = raw_frame.shape if isinstance(raw_frame, np.ndarray) else obs.shape
                    try:
                        gameplay_viz = GameplayGridVisualizer(num_envs, frame_shape=frame_shape)
                    except RuntimeError as exc:
                        print(f"[visualization] Disabling gameplay grid: {exc}")
                        gameplay_viz = None
                    except Exception as exc:
                        print(f"[visualization] Failed to initialise gameplay grid: {exc}")
                        gameplay_viz = None
                if map_viz:
                    map_viz.new_episode(idx, episode + 1)
                    if not cfg.persist_map:
                        map_viz.reset(idx)
                    map_viz.update(idx, info, reward=0.0, terminal=False, update_aggregate=True)
                if gameplay_viz:
                    gameplay_viz.new_episode(idx, episode + 1)
                    frame = raw_frame if isinstance(raw_frame, np.ndarray) else obs
                    gameplay_viz.update(idx, frame, info=info, reward=0.0, terminal=False)

            for step in range(cfg.max_steps_per_episode):
                active = False
                for idx, env in enumerate(envs):
                    if done[idx]:
                        continue
                    active = True
                    map_feat = extract_map_features(info_list[idx])
                    epsilon = epsilon_by_step(global_step, cfg)
                    action, new_hidden = select_action_eps(
                        online_net,
                        obs_list[idx],
                        map_feat,
                        epsilon,
                        env.action_space,
                        device,
                        actor_hidden[idx],
                    )
                    actor_hidden[idx] = new_hidden
                    next_obs, reward, terminated, truncated, next_info = env.step(action)
                    raw_frame = next_info.get("raw_frame")
                    next_obs_proc = preprocess_obs(next_obs)
                    next_map_feat = extract_map_features(next_info)
                    done_flag = bool(terminated or truncated)
                    nstep_buffers[idx].append(
                        (
                            obs_list[idx],
                            map_feat,
                            action,
                            reward,
                            next_obs_proc,
                            next_map_feat,
                            done_flag,
                        )
                    )
                    flush_nstep(idx)
                    obs_list[idx] = next_obs_proc
                    info_list[idx] = next_info
                    episode_rewards[idx] += reward
                    global_step += 1
                    env_steps[idx] += 1

                    coords = next_info.get("agent_coords")
                    map_name = next_info.get("map_name")
                    if cfg.verbose_logs and ((env_steps[idx] % log_interval == 0) or done_flag):
                        print(
                            f"[episode {episode + 1:04d}] step {step:05d} "
                            f"env {idx + 1}: reward={reward:+7.3f} "
                            f"map={map_name} coords={coords}",
                            flush=True,
                        )

                    if map_viz:
                        update_map = (env_steps[idx] % map_refresh == 0) or done_flag
                        update_agg = (env_steps[idx] % aggregate_refresh == 0) or done_flag
                        if update_map:
                            map_viz.update(
                                idx,
                                next_info,
                                reward=reward,
                                terminal=done_flag,
                                update_aggregate=update_agg,
                            )

                    if gameplay_viz:
                        update_frame = (env_steps[idx] % map_refresh == 0) or done_flag
                        if update_frame:
                            gameplay_viz.update(
                                idx,
                                raw_frame if isinstance(raw_frame, np.ndarray) else next_obs,
                                info=next_info,
                                reward=reward,
                                terminal=done_flag,
                            )

                    if (
                        len(replay_buffer) >= cfg.batch_size
                        and global_step >= cfg.learning_starts
                        and global_step % cfg.train_frequency == 0
                    ):
                        batch, weights, indices = replay_buffer.sample(cfg.batch_size)
                        batch_tensors = batch_to_tensors(batch, device)
                        weights_tensor = torch.tensor(
                            weights, dtype=torch.float32, device=device
                        )
                        loss_val, td_errors = compute_td_loss(
                            online_net,
                            target_net,
                            optimizer,
                            batch_tensors,
                            weights_tensor,
                            cfg.auxiliary_loss_coef,
                        )
                        replay_buffer.update_priorities(
                            indices, td_errors.abs().cpu().numpy()
                        )
                        last_loss = loss_val

                    if (
                        global_step >= cfg.learning_starts
                        and global_step % cfg.target_sync_interval == 0
                    ):
                        target_net.load_state_dict(online_net.state_dict())

                    if done_flag:
                        flush_nstep(idx, force=True)
                        actor_hidden[idx] = online_net.init_hidden(1, device)
                        done[idx] = True

                if all(done) or not active:
                    break
            for idx in range(num_envs):
                flush_nstep(idx, force=True)

            mean_reward = float(np.mean(episode_rewards))
            reward_window.append(mean_reward)
            running_avg = float(np.mean(reward_window))
            epsilon = epsilon_by_step(global_step, cfg)
            loss_val = last_loss if last_loss is not None else float("nan")
            rewards_str = ", ".join(f"{r:8.2f}" for r in episode_rewards)
            summary_due = cfg.verbose_logs or ((episode + 1) % cfg.progress_interval == 0) or (episode == 0) or (episode + 1 == cfg.episodes)
            if summary_due:
                print(
                    f"Episode {episode + 1:04d} | Rewards [{rewards_str}] | Mean {mean_reward:8.2f} | "
                    f"Avg(25) {running_avg:8.2f} | Steps {global_step:7d} | "
                    f"Epsilon {epsilon:6.3f} | Loss {loss_val:8.4f}",
                    flush=True,
                )
            if map_viz and cfg.save_map_images and ((episode + 1) % cfg.map_image_every == 0):
                os.makedirs(cfg.save_dir, exist_ok=True)
                map_path = os.path.join(cfg.save_dir, f"dqn_route1_ep{episode + 1:04d}.pt.map.png")
                try:
                    map_viz.save(map_path)
                    print(f"[map] Saved {map_path}")
                except Exception as exc:
                    print(f"[map] Failed to save map image: {exc}")

            if mean_reward > best_reward:
                best_reward = mean_reward
                maybe_save(online_net, best_model_path)

            if (episode + 1) % cfg.save_every == 0:
                checkpoint_path = os.path.join(
                    cfg.save_dir, f"dqn_route1_ep{episode + 1:04d}.pt"
                )
                maybe_save(online_net, checkpoint_path)
    except KeyboardInterrupt:
        print("\n[training] Interrupted by user. Saving latest checkpoint and shutting down...")
        latest_path = os.path.join(cfg.save_dir, "dqn_route1_latest.pt")
        maybe_save(online_net, latest_path)
        try:
            if 'map_viz' in locals() and map_viz and cfg.save_map_images:
                os.makedirs(cfg.save_dir, exist_ok=True)
                map_path = os.path.join(cfg.save_dir, "route_map_interrupt.png")
                map_viz.save(map_path)
                print(f"[map] Saved interrupt map to {map_path}")
        except Exception:
            pass
    finally:
        if map_viz:
            map_viz.close()
        if gameplay_viz:
            gameplay_viz.close()
        for env in envs:
            env.close()


if __name__ == "__main__":
    arguments = parse_args()
    config = build_config(arguments)
    train(config)
