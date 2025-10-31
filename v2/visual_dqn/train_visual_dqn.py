from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict
import sys

os.environ.setdefault("NUMPY_SKIP_MAC_OS_CHECK", "1")

import ray
from ray.rllib.algorithms.dqn import DQNConfig
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog

PROJECT_ROOT = Path(__file__).resolve().parents[2]
V2_DIR = PROJECT_ROOT / "v2"
for path in (PROJECT_ROOT, V2_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from red_gym_env_v2 import RedGymEnv  # noqa: E402
from visual_dqn.callbacks import CurriculumManagerCallback  # noqa: E402
from visual_dqn.config import load_curriculum_config, normalize_curriculum_paths  # noqa: E402
from visual_dqn.models import VisionDQNModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train visual DQN on Pokémon Red.")
    parser.add_argument("--num-workers", type=int, default=2, help="RLlib rollout workers.")
    parser.add_argument("--num-gpus", type=float, default=0.0, help="Number of GPUs for training.")
    parser.add_argument("--stop-iters", type=int, default=1000, help="Training iterations to run.")
    parser.add_argument("--logdir", type=Path, default=Path("runs_visual_dqn"), help="Log directory.")
    parser.add_argument("--beta", type=float, default=0.1, help="Intrinsic reward scale.")
    parser.add_argument("--memory-capacity", type=int, default=4096, help="Novelty memory capacity.")
    parser.add_argument("--knn-k", type=int, default=15, help="k-NN neighbors for novelty.")
    parser.add_argument("--checkpoint-interval", type=int, default=25, help="Iterations between checkpoints.")
    parser.add_argument("--curriculum-config", type=Path, default=None, help="Optional curriculum JSON definition.")
    parser.add_argument("--curriculum-start", type=int, default=0, help="Starting stage index for curriculum.")
    return parser.parse_args()


def build_env_config(log_dir: Path, curriculum=None, curriculum_start: int = 0) -> Dict:
    log_dir.mkdir(exist_ok=True, parents=True)
    env_config = {
        "headless": True,
        "save_final_state": False,
        "early_stop": False,
        "action_freq": 24,
        "init_state": str((PROJECT_ROOT / "init.state").resolve()),
        "max_steps": 4096,
        "print_rewards": False,
        "save_video": False,
        "fast_video": True,
        "session_path": log_dir,
        "gb_path": str((PROJECT_ROOT / "PokemonRed.gb").resolve()),
        "debug": False,
        "reward_scale": 0.5,
        "explore_weight": 0.25,
        "include_action_mask": True,
        "frame_stacks": 4,
        "no_change_mse_threshold": 1e-3,
    }
    if curriculum:
        env_config["curriculum"] = curriculum
        env_config["curriculum_start_index"] = curriculum_start
    return env_config


def configured_callback(beta: float, capacity: int, knn_k: int, curriculum=None):
    class _Callback(CurriculumManagerCallback):
        curriculum_blueprint = curriculum

        def __init__(self) -> None:
            super().__init__(beta=beta, memory_capacity=capacity, k=knn_k)

    return _Callback


def main() -> None:
    args = parse_args()
    log_dir = V2_DIR / args.logdir
    curriculum_raw = load_curriculum_config(args.curriculum_config)
    curriculum = normalize_curriculum_paths(
        curriculum_raw, state_base=V2_DIR, save_base=log_dir
    )
    env_config = build_env_config(log_dir, curriculum=curriculum, curriculum_start=args.curriculum_start)

    env_name = "RedGymVisualDQNEnv"
    try:
        register_env(env_name, lambda cfg: RedGymEnv(cfg))
    except ValueError:
        pass
    try:
        ModelCatalog.register_custom_model("vision_dqn", VisionDQNModel)
    except ValueError:
        pass

    config = (
        DQNConfig()
        .environment(env_name, env_config=env_config)
        .framework("torch")
        .resources(num_gpus=args.num_gpus)
        .rollouts(
            num_rollout_workers=args.num_workers,
            rollout_fragment_length=128,
            num_envs_per_worker=1,
            enable_connectors=False,
        )
        .training(
            gamma=0.997,
            lr=3e-4,
            n_step=5,
            train_batch_size=1024,
            dueling=False,
            hiddens=[],
            double_q=True,
            grad_clip=40.0,
            target_network_update_freq=10000,
            replay_buffer_config={
                "type": "MultiAgentReplayBuffer",
                "capacity": int(2e4),
                "storage_unit": "timesteps",
            },
        )
        .exploration(
            exploration_config={
                "type": "EpsilonGreedy",
                "initial_epsilon": 1.0,
                "final_epsilon": 0.01,
                "epsilon_timesteps": int(1e6),
            }
        )
    )

    config.model = {
        "custom_model": "vision_dqn",
        "max_seq_len": 32,
        "fcnet_hiddens": [],
        "fcnet_activation": "relu",
        "custom_model_config": {
            "in_channels": env_config.get("frame_stacks", 4),
            "embedding_dim": 256,
            "gru_hidden_size": 256,
            "conv_channels": (32, 64, 128, 128),
            "kernel_sizes": (5, 3, 3, 3),
            "strides": (2, 2, 2, 1),
            "attention_dropout": 0.1,
            "linear_dropout": 0.1,
            "q_hidden_dims": (256, 256),
            "q_dropout": 0.1,
        },
    }

    config = config.callbacks(
        configured_callback(
            args.beta,
            args.memory_capacity,
            args.knn_k,
            curriculum=curriculum,
        )
    )

    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath_entries = [str(PROJECT_ROOT), str(V2_DIR)]
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    runtime_env = {
        "working_dir": str(PROJECT_ROOT),
        "excludes": [
            "runs",
            ".venv",
            "__pycache__",
            ".git",
            "visualization/poke_map/pokemap_full_rough.psd",
        ],
        "env_vars": {"PYTHONPATH": ":".join(pythonpath_entries)},
    }
    ray.init(ignore_reinit_error=True, runtime_env=runtime_env)
    algo = config.build()

    try:
        for it in range(1, args.stop_iters + 1):
            result = algo.train()
            print(
                f"[Iteration {it}] "
                f"reward_mean={result['episode_reward_mean']:.3f} "
                f"timesteps_total={result.get('timesteps_total', 0)}"
            )

            if args.checkpoint_interval and it % args.checkpoint_interval == 0:
                checkpoint_path = algo.save(checkpoint_dir=str(log_dir))
                print(f"Checkpoint saved at {checkpoint_path}")
    finally:
        algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
