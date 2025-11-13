from __future__ import annotations

import random
from collections import deque, namedtuple
from typing import Any, Deque, Dict, List, Tuple

import numpy as np

Transition = namedtuple(
    "Transition",
    "obs context action reward discount next_obs next_context done",
)


class EpisodeStorage:
    def __init__(self):
        self.transitions: List[Transition] = []

    def append(self, t: Transition) -> None:
        self.transitions.append(t)

    def __len__(self) -> int:
        return len(self.transitions)


class PrioritizedSequenceReplay:
    """
    Simple prioritized replay over sequences for recurrent Q-learning.
    Stores episodes and samples random sub-sequences with (unroll + burn-in).
    Priorities are maintained per start-index key.
    """

    def __init__(
        self,
        capacity_transitions: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 200_000,
    ) -> None:
        self.capacity_transitions = max(1, int(capacity_transitions))
        self.total_transitions = 0
        self.episodes: Dict[int, EpisodeStorage] = {}
        self.order: Deque[int] = deque()
        self.alpha = float(alpha)
        self.beta = float(beta_start)
        self.beta_increment = (1.0 - beta_start) / max(1, int(beta_frames))
        self.priorities: Dict[Tuple[int, int], float] = {}  # (episode_idx, start_idx) -> priority
        self.max_priority: float = 1.0
        self._next_episode_index = 0

    def add_episode(self, transitions: List[Transition]) -> None:
        if not transitions:
            return
        ep = EpisodeStorage()
        for t in transitions:
            # Compress storage to reduce RAM: obs/next_obs -> uint8 [0..255], context -> float16
            obs_arr = t.obs
            if isinstance(obs_arr, np.ndarray) and obs_arr.dtype != np.uint8:
                # obs is expected in [0,1] float; convert to [0,255] uint8
                obs_arr = np.clip(np.round(obs_arr * 255.0), 0, 255).astype(np.uint8, copy=False)
            next_obs_arr = t.next_obs
            if isinstance(next_obs_arr, np.ndarray) and next_obs_arr.dtype != np.uint8:
                next_obs_arr = np.clip(np.round(next_obs_arr * 255.0), 0, 255).astype(np.uint8, copy=False)
            ctx_arr = t.context
            if isinstance(ctx_arr, np.ndarray) and ctx_arr.dtype != np.float16:
                ctx_arr = ctx_arr.astype(np.float16, copy=False)
            next_ctx_arr = t.next_context
            if isinstance(next_ctx_arr, np.ndarray) and next_ctx_arr.dtype != np.float16:
                next_ctx_arr = next_ctx_arr.astype(np.float16, copy=False)
            ep.append(Transition(obs_arr, ctx_arr, int(t.action), float(t.reward), float(t.discount), next_obs_arr, next_ctx_arr, bool(t.done)))
        ep_id = self._next_episode_index
        self._next_episode_index += 1
        self.episodes[ep_id] = ep
        self.order.append(ep_id)
        self.total_transitions += len(ep)
        self._evict_excess()

    def __len__(self) -> int:
        return self.total_transitions

    def _evict_excess(self) -> None:
        while self.total_transitions > self.capacity_transitions and self.order:
            old_id = self.order.popleft()
            ep = self.episodes.pop(old_id, None)
            if ep is None:
                continue
            self.total_transitions = max(0, self.total_transitions - len(ep))
            drop_keys = [k for k in list(self.priorities.keys()) if k[0] == old_id]
            for k in drop_keys:
                self.priorities.pop(k, None)

    def update_priorities(self, keys: List[Tuple[int, int]], td_errors: np.ndarray) -> None:
        td_errors = np.asarray(td_errors, dtype=np.float32).flatten()
        for k, e in zip(keys, td_errors):
            val = float(abs(e) + 1e-6)
            self.priorities[k] = val
            self.max_priority = max(self.max_priority, val)

    def sample_sequences(
        self,
        batch_size: int,
        unroll: int,
        burn_in: int,
    ):
        """
        Returns:
            batch (list[list[Transition]]), weights (np.ndarray), keys (list[(ep_idx, start_idx)])
        """
        candidates: List[Tuple[int, int, int]] = []  # (ep_id, start_idx, length)
        for ep_id in self.order:
            ep = self.episodes.get(ep_id)
            if ep is None:
                continue
            ep_len = len(ep)
            total_len = burn_in + unroll
            if ep_len >= total_len:
                for start in range(0, ep_len - total_len + 1):
                    candidates.append((ep_id, start, total_len))
        if not candidates:
            raise ValueError("Not enough data to sample sequences")
        # Build probabilities from priorities
        probs = []
        keys = []
        for (eid, start, _) in candidates:
            key = (eid, start)
            p = self.priorities.get(key, self.max_priority)
            probs.append(p ** self.alpha)
            keys.append(key)
        probs = np.asarray(probs, dtype=np.float32)
        probs /= probs.sum()
        idxs = np.random.choice(len(candidates), size=batch_size, p=probs)
        sampled_keys = [keys[i] for i in idxs]
        weights = (len(candidates) * probs[idxs]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)
        # Assemble transitions
        batch = []
        for (eid, start) in sampled_keys:
            ep = self.episodes[eid]
            seq = ep.transitions[start : start + burn_in + unroll]
            batch.append(seq)
        return batch, weights.astype(np.float32), sampled_keys

    def state_dict(self) -> Dict[str, Any]:
        return {
            "capacity_transitions": self.capacity_transitions,
            "total_transitions": self.total_transitions,
            "episodes": {
                str(eid): [t._asdict() for t in ep.transitions] for eid, ep in self.episodes.items()
            },
            "order": list(self.order),
            "alpha": self.alpha,
            "beta": self.beta,
            "beta_increment": self.beta_increment,
            "priorities": {f"{k[0]}:{k[1]}": v for k, v in self.priorities.items()},
            "max_priority": self.max_priority,
            "_next_episode_index": self._next_episode_index,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.capacity_transitions = int(state.get("capacity_transitions", self.capacity_transitions))
        self.total_transitions = int(state.get("total_transitions", 0))
        episodes_state = state.get("episodes", {})
        self.episodes = {}
        for eid_str, transitions in episodes_state.items():
            ep = EpisodeStorage()
            for t in transitions:
                ep.append(
                    Transition(
                        t["obs"],
                        t["context"],
                        int(t["action"]),
                        float(t["reward"]),
                        float(t["discount"]),
                        t["next_obs"],
                        t["next_context"],
                        bool(t["done"]),
                    )
                )
            self.episodes[int(eid_str)] = ep
        self.order = deque(state.get("order", []))
        self.alpha = float(state.get("alpha", self.alpha))
        self.beta = float(state.get("beta", self.beta))
        self.beta_increment = float(state.get("beta_increment", self.beta_increment))
        self.priorities = {}
        for key, val in state.get("priorities", {}).items():
            eid_str, start_str = key.split(":")
            self.priorities[(int(eid_str), int(start_str))] = float(val)
        self.max_priority = float(state.get("max_priority", 1.0))
        self._next_episode_index = int(state.get("_next_episode_index", len(self.episodes)))
