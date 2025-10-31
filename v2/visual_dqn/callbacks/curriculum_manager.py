from __future__ import annotations

import math
from collections import deque
from typing import Dict, Iterable, List, Optional

from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID

from .episodic_novelty import EpisodicNoveltyCallback


class _StageTracker:
    def __init__(self, stage_config: Dict[str, object]) -> None:
        self.window_size = int(stage_config.get("window_size", 20))
        self.min_episodes = int(stage_config.get("min_episodes", self.window_size))
        self.success_rate_threshold = float(stage_config.get("success_rate_threshold", 0.8))
        mean_steps = stage_config.get("mean_steps_threshold")
        self.mean_steps_threshold: Optional[float] = None if mean_steps is None else float(mean_steps)
        self.auto_advance = bool(stage_config.get("auto_advance", True))
        self.successes: deque[float] = deque(maxlen=self.window_size)
        self.success_steps: deque[float] = deque(maxlen=self.window_size)
        self.total_episodes = 0
        self.saved = False
        self.advanced = False

    def update(self, successes: Iterable[float], success_steps: Iterable[float]) -> None:
        added = 0
        for val in successes:
            self.successes.append(float(val))
            added += 1
        self.total_episodes += added

        for steps in success_steps:
            if steps is None:
                continue
            if isinstance(steps, float) and math.isnan(steps):
                continue
            self.success_steps.append(float(steps))

    def success_rate(self) -> float:
        if not self.successes:
            return 0.0
        return float(sum(self.successes) / len(self.successes))

    def mean_success_steps(self) -> float:
        if not self.success_steps:
            return float("inf")
        return float(sum(self.success_steps) / len(self.success_steps))

    def ready(self) -> bool:
        return self.total_episodes >= self.min_episodes and len(self.successes) >= min(
            self.min_episodes, self.window_size
        )

    def should_advance(self) -> bool:
        if self.advanced:
            return False
        if not self.ready():
            return False
        if self.success_rate() < self.success_rate_threshold:
            return False
        if self.mean_steps_threshold is not None:
            if not self.success_steps:
                return False
            if self.mean_success_steps() > self.mean_steps_threshold:
                return False
        return True


class CurriculumManagerCallback(EpisodicNoveltyCallback):
    curriculum_blueprint: Optional[List[Dict[str, object]]] = None

    def __init__(
        self,
        beta: float = 0.1,
        memory_capacity: int = 4096,
        k: int = 15,
        normalize_distances: bool = False,
    ) -> None:
        super().__init__(beta=beta, memory_capacity=memory_capacity, k=k, normalize_distances=normalize_distances)
        self._stage_trackers: Dict[int, _StageTracker] = {}
        self._active_stage = 0

    def _get_curriculum(self, algorithm=None) -> Optional[List[Dict[str, object]]]:
        if self.curriculum_blueprint is not None:
            return self.curriculum_blueprint
        if algorithm is None:
            return None
        env_config = algorithm.config.get("env_config", {})
        return env_config.get("curriculum")

    def _ensure_tracker(self, stage_idx: int, stage_cfg: Dict[str, object]) -> _StageTracker:
        tracker = self._stage_trackers.get(stage_idx)
        if tracker is None:
            tracker = _StageTracker(stage_cfg)
            self._stage_trackers[stage_idx] = tracker
        return tracker

    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs) -> None:
        super().on_episode_end(worker=worker, base_env=base_env, policies=policies, episode=episode, **kwargs)

        info = episode.last_info_for(DEFAULT_POLICY_ID)
        if not info:
            return
        curriculum_info = info.get("curriculum")
        if not curriculum_info:
            return

        stage_idx = int(curriculum_info.get("stage_index", 0))
        success = 1.0 if curriculum_info.get("success") else 0.0
        first_success_step = curriculum_info.get("first_success_step")
        step_value = float(first_success_step) if first_success_step is not None else float("nan")

        episode.hist_data.setdefault(f"stage_{stage_idx}_success", []).append(success)
        episode.hist_data.setdefault(f"stage_{stage_idx}_success_steps", []).append(step_value)

    def on_train_result(self, *, algorithm, result: Dict, **kwargs) -> None:
        super().on_train_result(algorithm=algorithm, result=result, **kwargs)

        curriculum = self._get_curriculum(algorithm)
        if not curriculum:
            return

        if self._active_stage >= len(curriculum):
            return

        stage_cfg = curriculum[self._active_stage]
        tracker = self._ensure_tracker(self._active_stage, stage_cfg)

        hist_stats = result.get("hist_stats", {})
        success_key = f"stage_{self._active_stage}_success"
        steps_key = f"stage_{self._active_stage}_success_steps"
        successes = hist_stats.get(success_key, [])
        steps = hist_stats.get(steps_key, [])

        if successes or steps:
            tracker.update(successes, steps)

        result.setdefault("custom_metrics", {})
        result["custom_metrics"][f"curriculum_stage_{self._active_stage}_success_rate"] = tracker.success_rate()
        result["custom_metrics"][f"curriculum_stage_{self._active_stage}_mean_steps"] = tracker.mean_success_steps()
        result["custom_metrics"]["curriculum_active_stage"] = self._active_stage

        if tracker.should_advance():
            if not tracker.saved:
                save_results = algorithm.workers.foreach_env(lambda env: env.save_curriculum_state())
                tracker.saved = any(save_results) or stage_cfg.get("save_path") is None

            if tracker.auto_advance and tracker.saved:
                algorithm.workers.foreach_env(lambda env: env.advance_curriculum())
                tracker.advanced = True
                self._active_stage += 1
