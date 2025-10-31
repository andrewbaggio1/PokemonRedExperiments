from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np

try:
    import hnswlib
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "hnswlib is required for EpisodicMemory. Install with `pip install hnswlib`."
    ) from exc


@dataclass
class EpisodicMemoryConfig:
    dim: int
    max_size: int = 4096
    k: int = 15
    space: str = "l2"
    ef_construction: int = 200
    ef_runtime: int = 200
    novelty_default: float = 1.0
    normalize_distances: bool = False
    min_distance: float = 0.0
    max_distance: float = 25.0


class EpisodicMemory:
    """k-NN episodic memory that tracks visual embeddings for intrinsic rewards."""

    def __init__(self, config: EpisodicMemoryConfig) -> None:
        self.config = config
        self._index = self._build_index()
        self._id_fifo: deque[int] = deque()
        self._next_id = 0
        self._size = 0

    def _build_index(self) -> "hnswlib.Index":
        index = hnswlib.Index(space=self.config.space, dim=self.config.dim)
        index.init_index(
            max_elements=self.config.max_size,
            ef_construction=self.config.ef_construction,
            M=16,
        )
        index.set_ef(self.config.ef_runtime)
        return index

    def reset(self) -> None:
        self._index = self._build_index()
        self._id_fifo.clear()
        self._next_id = 0
        self._size = 0

    def _maybe_evict(self) -> None:
        if self._size < self.config.max_size:
            return
        oldest = self._id_fifo.popleft()
        self._index.mark_deleted(oldest)
        self._size -= 1

    def add(self, embeddings: np.ndarray) -> None:
        if embeddings.size == 0:
            return
        emb = np.asarray(embeddings, dtype=np.float32)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)

        for vector in emb:
            self._maybe_evict()
            idx = self._next_id
            self._index.add_items(vector, idx)
            self._id_fifo.append(idx)
            self._next_id += 1
            self._size += 1

    def _query(self, embeddings: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        return self._index.knn_query(embeddings, k=k, num_threads=-1)

    def novelty_score(self, embeddings: np.ndarray) -> np.ndarray:
        emb = np.asarray(embeddings, dtype=np.float32)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        if emb.shape[1] != self.config.dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.config.dim}, got {emb.shape[1]}"
            )

        if self._size == 0:
            return np.full(emb.shape[0], self.config.novelty_default, dtype=np.float32)

        query_k = min(self.config.k, self._size)
        labels, distances = self._query(emb, query_k)
        min_dist = distances.min(axis=1)
        novelty = np.sqrt(np.maximum(min_dist, 0.0)).astype(np.float32)

        if self.config.normalize_distances:
            novelty = np.clip(
                (novelty - self.config.min_distance)
                / (self.config.max_distance - self.config.min_distance + 1e-8),
                0.0,
                1.0,
            )

        return novelty

