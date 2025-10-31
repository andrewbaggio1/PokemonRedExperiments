from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("NUMPY_SKIP_MAC_OS_CHECK", "1")

import numpy as np
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
V2_DIR = PROJECT_ROOT / "v2"
for path in (PROJECT_ROOT, V2_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from red_gym_env_v2 import RedGymEnv  # noqa: E402
from visual_dqn.config import load_curriculum_config, normalize_curriculum_paths  # noqa: E402
from visual_dqn.models import VisionDQNModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play a trained visual DQN agent.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="RLlib checkpoint path.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run.")
    parser.add_argument("--headless", action="store_true", help="Run without rendering window.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional per-episode step cap.")
    parser.add_argument("--curriculum-config", type=Path, default=None, help="Override curriculum JSON.")
    parser.add_argument("--curriculum-stage", type=int, default=None, help="Force curriculum stage index.")
    return parser.parse_args()


def build_env_config(
    base_config: dict,
    *,
    headless: bool,
    session_path: Path,
    max_steps: int | None,
    curriculum,
    curriculum_stage: int | None,
) -> dict:
    session_path.mkdir(exist_ok=True, parents=True)
    config = dict(base_config)
    config["headless"] = headless
    config["session_path"] = session_path
    if max_steps is not None:
        config["max_steps"] = max_steps
    if curriculum is not None:
        config["curriculum"] = curriculum
    if curriculum_stage is not None:
        config["curriculum_start_index"] = curriculum_stage
    return config


def main() -> None:
    args = parse_args()
    log_dir = Path(__file__).resolve().parent / "runs_visual_dqn_play"
    env_name = "RedGymVisualDQNEnv"
    try:
        register_env(env_name, lambda cfg: RedGymEnv(cfg))
    except ValueError:
        pass
    try:
        ModelCatalog.register_custom_model("vision_dqn", VisionDQNModel)
    except ValueError:
        pass

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
    ray.init(ignore_reinit_error=True, include_dashboard=False, runtime_env=runtime_env)
    algo = Algorithm.from_checkpoint(str(args.checkpoint))
    policy = algo.get_policy()

    base_env_config = dict(algo.config.get("env_config", {}))
    override_curriculum = load_curriculum_config(args.curriculum_config)
    curriculum = normalize_curriculum_paths(
        override_curriculum,
        state_base=Path.cwd(),
        save_base=log_dir,
    )
    env_config = build_env_config(
        base_env_config,
        headless=args.headless,
        session_path=log_dir,
        max_steps=args.max_steps,
        curriculum=curriculum if curriculum is not None else base_env_config.get("curriculum"),
        curriculum_stage=args.curriculum_stage,
    )

    env = RedGymEnv(env_config)
    total_rewards = []
    for episode in range(args.episodes):
        obs, _ = env.reset()
        state = policy.get_initial_state()
        done = False
        truncated = False
        ep_reward = 0.0

        while not (done or truncated):
            action, state, _ = algo.compute_single_action(
                obs, state=state, explore=False, full_fetch=True
            )
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            if not args.headless:
                env.render()

        print(f"Episode {episode + 1}: reward={ep_reward:.2f}")
        total_rewards.append(ep_reward)

    if total_rewards:
        mean_reward = float(np.mean(total_rewards))
        print(f"Average reward over {len(total_rewards)} episodes: {mean_reward:.2f}")

    if hasattr(env, "close"):
        env.close()
    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
