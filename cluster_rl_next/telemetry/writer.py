from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from typing import Deque, Optional

from .events import TelemetryEvent, event_to_json


class EventStreamWriter:
    def __init__(self, path: str, flush_interval_s: float = 2.0, max_queue: int = 2048):
        self.path = path
        self.flush_interval = max(0.1, float(flush_interval_s))
        self.max_queue = max_queue
        self._queue: Deque[TelemetryEvent] = deque()
        self._lock = threading.Lock()
        self._last_flush = time.monotonic()

    def enqueue(self, event: TelemetryEvent) -> None:
        with self._lock:
            self._queue.append(event)
            if len(self._queue) >= self.max_queue or (time.monotonic() - self._last_flush) >= self.flush_interval:
                self._flush_locked()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._queue:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            while self._queue:
                ev = self._queue.popleft()
                fh.write(json.dumps(event_to_json(ev)) + "\n")
        self._last_flush = time.monotonic()
