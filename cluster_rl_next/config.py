from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None


# ---------------------------------------------------------------------------
# Dataclasses


@dataclass
class VisionConfig:
    hi_res_channels: int = 128
    lo_res_channels: int = 160
    stem_type: str = "preact"  # preact | basic
    res_blocks_hi: int = 4
    res_blocks_lo: int = 5
    se_reduction: int = 8
    deformable: bool = False
    attention_type: str = "mhsa"  # mhsa | mobilevit
    attention_heads: int = 4
    attention_depth: int = 2
    attention_dropout: float = 0.1
    feature_dropout: float = 0.1


@dataclass
class ContextConfig:
    static_dim: int = 128
    event_dim: int = 64
    mlp_hidden: int = 192
    use_film: bool = True


@dataclass
class TransformerConfig:
    d_model: int = 512
    nhead: int = 4
    depth: int = 2
    dim_feedforward: int = 1024
    dropout: float = 0.1


@dataclass
class TemporalConfig:
    gru_hidden: int = 384
    lstm_hidden: int = 768
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    latent_target_dim: int = 1344


@dataclass
class DuelingConfig:
    hidden: int = 512


@dataclass
class AuxHeadConfig:
    context_recon_weight: float = 0.25
    inverse_dynamics_weight: float = 0.1
    reward_forecast_weight: float = 0.1


@dataclass
class HeadsConfig:
    dueling: DuelingConfig = field(default_factory=DuelingConfig)
    num_quantiles: int = 51
    aux: AuxHeadConfig = field(default_factory=AuxHeadConfig)


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 3e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.01
    eps: float = 1e-8


@dataclass
class MultiGPUConfig:
    mode: str = "dp"  # none | dp | ddp
    world_size: int = 1
    device_ids: Optional[List[int]] = None
    nccl_timeout_s: int = 180


@dataclass
class ActorConfig:
    per_gpu: int = 4
    total: int = 16
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 1_000_000
    action_duration_options: Tuple[int, ...] = (1, 2, 4, 8)


@dataclass
class TrainingConfig:
    batch_size: int = 256
    unroll_length: int = 128
    burn_in: int = 32
    gamma: float = 0.997
    n_step: int = 5
    prioritized_alpha: float = 0.6
    prioritized_beta_frames: int = 1_000_000
    learn_start_steps: int = 40_000
    train_frequency: int = 2
    target_sync_interval_updates: int = 2000
    grad_clip: float = 1.0
    amp: bool = True
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    multi_gpu: MultiGPUConfig = field(default_factory=MultiGPUConfig)
    actors: ActorConfig = field(default_factory=ActorConfig)
    target_total_steps: int = 50_000_000
    max_steps_per_episode: int = 4000


@dataclass
class ReplayConfig:
    capacity_transitions: int = 2_000_000
    save_stats: bool = True


@dataclass
class CheckpointConfig:
    interval_minutes: int = 10
    max_keep: int = 6
    include_replay_stats: bool = True
    include_actor_snapshots: bool = True


@dataclass
class ResumeConfig:
    resume_policy: str = "latest"  # latest | specific
    resume_from_id: Optional[str] = None


@dataclass
class EmulatorConfig:
    frame_skip: int = 4
    boot_steps: int = 60
    input_spacing_frames: int = 1
    continuous_mode: bool = True
    snapshots_dir: str = "snapshots"
    flatline_steps_threshold: int = 800
    min_delta_pos: int = 1
    min_delta_reward: float = 0.1
    action_reset: Tuple[str, int] = ("hold_A", 12)
    max_no_input_frames: int = 600
    headless: bool = True
    delete_sav_on_reset: bool = False
    snapshot_interval_minutes: int = 5
    occupancy_crop_radius: int = 10


@dataclass
class TelemetryConfig:
    enabled: bool = True
    flush_interval_s: float = 2.0
    max_queue: int = 2048
    events_file: str = "events.jsonl"
    throttle_hz: float = 5.0


@dataclass
class PathsConfig:
    rom_path: str = "PokemonRed.gb"
    save_dir: str = "cluster_runs"
    run_name: str = "run_next"
    logs_dir: str = "logs"

    @property
    def run_dir(self) -> str:
        return os.path.join(self.save_dir, self.run_name)


@dataclass
class ClusterNextConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    heads: HeadsConfig = field(default_factory=HeadsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    checkpointing: CheckpointConfig = field(default_factory=CheckpointConfig)
    resume: ResumeConfig = field(default_factory=ResumeConfig)
    emulator: EmulatorConfig = field(default_factory=EmulatorConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    log_interval_steps: int = 4000
    seed: int = 42
    log_episodes_for_replay: bool = True
    replay_log_dir: str = "episode_logs"
    base_seed: int = 7
    affordance_fail_threshold: int = 3
    affordance_decay: float = 0.995
    max_runtime_seconds: Optional[int] = None
    segment_length: int = 2000
    resume_from_snapshot: bool = True
    total_episodes: int = 0

    _MIGRATION_FIELDS: ClassVar[Dict[str, str]] = {
        "rom_path": "paths.rom_path",
        "save_dir": "paths.save_dir",
        "run_name": "paths.run_name",
        "logs_dir": "paths.logs_dir",
        "gamma": "training.gamma",
        "n_step": "training.n_step",
        "learning_rate": "training.optimizer.lr",
        "target_sync_interval": "training.target_sync_interval_updates",
        "burn_in": "training.burn_in",
        "unroll_length": "training.unroll_length",
        "batch_size": "training.batch_size",
        "learn_start_steps": "training.learn_start_steps",
        "train_frequency": "training.train_frequency",
        "prioritized_alpha": "training.prioritized_alpha",
        "prioritized_beta_frames": "training.prioritized_beta_frames",
        "epsilon_actor_start": "training.actors.epsilon_start",
        "epsilon_actor_end": "training.actors.epsilon_end",
        "epsilon_decay_steps": "training.actors.epsilon_decay_steps",
        "action_duration_options": "training.actors.action_duration_options",
        "num_actors": "training.actors.total",
        "replay_capacity_transitions": "replay.capacity_transitions",
        "frame_skip": "emulator.frame_skip",
        "boot_steps": "emulator.boot_steps",
        "max_no_input_frames": "emulator.max_no_input_frames",
        "input_spacing_frames": "emulator.input_spacing_frames",
        "headless": "emulator.headless",
        "delete_sav_on_reset": "emulator.delete_sav_on_reset",
        "occupancy_crop_radius": "emulator.occupancy_crop_radius",
        "continuous_mode": "emulator.continuous_mode",
        "snapshot_interval_minutes": "emulator.snapshot_interval_minutes",
        "snapshots_dir": "emulator.snapshots_dir",
        "target_total_steps": "training.target_total_steps",
        "max_steps_per_episode": "training.max_steps_per_episode",
        "total_episodes": "total_episodes",
        "lstm_hidden_size": "temporal.lstm_hidden",
        "use_spatial_attention": "vision.attention_depth",
        "num_quantiles": "heads.num_quantiles",
    }

    def __getattr__(self, name: str):
        alias = self._MIGRATION_FIELDS.get(name)
        if alias is None:
            raise AttributeError(name)
        value = _resolve_alias(self, alias)
        if name == "use_spatial_attention":
            return bool(value)
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        fields_dict = getattr(self, "__dataclass_fields__", {})
        if name in fields_dict:
            return super().__setattr__(name, value)
        alias = self._MIGRATION_FIELDS.get(name)
        if alias:
            parent, attr = _resolve_alias_parent(self, alias)
            setattr(parent, attr, value)
        else:
            super().__setattr__(name, value)

    @property
    def run_dir(self) -> str:
        return self.paths.run_dir


# ---------------------------------------------------------------------------
# Helpers


def _load_raw(path: str) -> Dict[str, Any]:
    _, ext = os.path.splitext(path)
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    if ext.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML configs.")
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("config file must map to a JSON/YAML object")
    return data


def _merge_dataclass(default: Any, overrides: Dict[str, Any]) -> Any:
    kwargs = {}
    sentinel = object()
    for f in fields(default):
        current_value = getattr(default, f.name)
        override_value = overrides.get(f.name, sentinel)
        if is_dataclass(current_value):
            nested_overrides = override_value if isinstance(override_value, dict) else {}
            kwargs[f.name] = _merge_dataclass(current_value, nested_overrides)
        else:
            kwargs[f.name] = current_value if override_value is sentinel else override_value
    return type(default)(**kwargs)


def _merge_config(default: ClusterNextConfig, overrides: Dict[str, Any]) -> ClusterNextConfig:
    merged_kwargs: Dict[str, Any] = {}
    for f in fields(default):
        field_name = f.name
        field_value = getattr(default, field_name)
        if is_dataclass(field_value):
            sub_overrides = overrides.get(field_name, {})
            merged_kwargs[field_name] = _merge_dataclass(field_value, sub_overrides)
        else:
            merged_kwargs[field_name] = overrides.get(field_name, field_value)
    cfg = ClusterNextConfig(**merged_kwargs)
    for legacy_name, attr_path in ClusterNextConfig._MIGRATION_FIELDS.items():
        if legacy_name in overrides and legacy_name not in merged_kwargs:
            parent, attr = _resolve_alias_parent(cfg, attr_path)
            setattr(parent, attr, overrides[legacy_name])
    return cfg


def _resolve_alias(cfg: ClusterNextConfig, path: str):
    parts = path.split(".")
    obj: Any = cfg
    for part in parts:
        obj = getattr(obj, part)
    return obj


def _resolve_alias_parent(cfg: ClusterNextConfig, path: str):
    parts = path.split(".")
    obj: Any = cfg
    for part in parts[:-1]:
        obj = getattr(obj, part)
    return obj, parts[-1]


# ---------------------------------------------------------------------------
# Public API


def default_config() -> ClusterNextConfig:
    return ClusterNextConfig()


def load_config(path: Optional[str]) -> ClusterNextConfig:
    base = default_config()
    if not path:
        return base
    overrides = _load_raw(path)
    return _merge_config(base, overrides)


def save_config(cfg: ClusterNextConfig, path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(asdict(cfg), fh, indent=2)


# Backwards compatibility alias
ClusterConfig = ClusterNextConfig
