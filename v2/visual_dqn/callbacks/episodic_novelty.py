from __future__ import annotations

from typing import Dict

import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy.sample_batch import SampleBatch

from ..modules.episodic_memory import EpisodicMemory, EpisodicMemoryConfig


class EpisodicNoveltyCallback(DefaultCallbacks):
    """Adds episodic novelty rewards computed from visual embeddings."""

    def __init__(
        self,
        beta: float = 0.1,
        memory_capacity: int = 4096,
        k: int = 15,
        normalize_distances: bool = False,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.memory_capacity = memory_capacity
        self.k = k
        self.normalize_distances = normalize_distances
        self._memory_key = "episodic_memory"

    def _get_or_create_memory(
        self, policy_model, episode
    ) -> EpisodicMemory:
        memory = episode.user_data.get(self._memory_key)
        if memory is None:
            config = EpisodicMemoryConfig(
                dim=policy_model.embedding_dim,
                max_size=self.memory_capacity,
                k=self.k,
                normalize_distances=self.normalize_distances,
            )
            memory = EpisodicMemory(config)
            episode.user_data[self._memory_key] = memory
        return memory

    def on_episode_start(self, *, episode, **kwargs) -> None:
        memory = episode.user_data.get(self._memory_key)
        if memory is not None:
            memory.reset()

    def on_postprocess_trajectory(
        self,
        *,
        worker,
        episode,
        agent_id,
        policy_id,
        policies,
        postprocessed_batch,
        original_batches,
        **kwargs,
    ) -> None:
        policy = policies[policy_id]
        model = policy.model
        memory = self._get_or_create_memory(model, episode)

        frames_array = None

        if agent_id in original_batches:
            _, original = original_batches[agent_id]
            raw_obs = original[SampleBatch.OBS]
            if isinstance(raw_obs, dict):
                frames_array = np.asarray(raw_obs.get("obs"))
            elif isinstance(raw_obs, (list, tuple, np.ndarray)) and len(raw_obs) > 0:
                first = raw_obs[0]
                if isinstance(first, dict) and "obs" in first:
                    frames_array = np.asarray([item["obs"] for item in raw_obs])

        if frames_array is None:
            return

        embeddings = model.encode_observations(frames_array, normalize=True)
        novelty = memory.novelty_score(embeddings)
        novelty = np.nan_to_num(novelty, nan=0.0, posinf=10.0, neginf=0.0)
        novelty = np.clip(novelty, 0.0, 10.0)
        memory.add(embeddings)

        rewards = postprocessed_batch["rewards"]
        postprocessed_batch["rewards"] = rewards + self.beta * novelty
