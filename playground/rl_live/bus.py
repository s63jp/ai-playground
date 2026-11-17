"""Fanout bus between the training thread and SSE clients.

Publish sides (training thread, refresher thread) never block or fail on
slow/gone subscribers: each subscriber holds

  - a small bounded event queue (dropped oldest when full), and
  - ONE pending frame slot with latest-wins semantics (the viewer wants the
    freshest pose of the rig, not a backlog of stale ones).

Frame rate is decimated at the bus — MuJoCo decisions arrive at 50 Hz;
viewers get the configurable default 30 Hz.
"""

from __future__ import annotations

import threading
import time
from collections import deque

EVENT_QUEUE_MAX = 200


class Subscriber:
    def __init__(self):
        self.cond = threading.Condition()
        self.events: deque[tuple[str, dict]] = deque(maxlen=EVENT_QUEUE_MAX)
        self._pending_frame: dict | None = None
        self._dropped_frames = 0

    def push_event(self, kind, payload):
        with self.cond:
            self.events.append((kind, payload))
            self.cond.notify_all()

    def push_frame(self, frame):
        with self.cond:
            if self._pending_frame is not None:
                self._dropped_frames += 1
            self._pending_frame = frame
            self.cond.notify_all()

    def drain(self) -> tuple[list[tuple[str, dict]], dict | None, int]:
        with self.cond:
            events = list(self.events)
            self.events.clear()
            frame, self._pending_frame = self._pending_frame, None
            return events, frame, self._dropped_frames


class Bus:
    def __init__(self, frame_hz: float = 30.0):
        self.frame_hz = frame_hz
        self._subs: list[Subscriber] = []
        self._lock = threading.Lock()
        self._last_frame_t = 0.0
        self._published_frames = 0

    def subscribe(self) -> Subscriber:
        sub = Subscriber()
        with self._lock:
            self._subs.append(sub)
        return sub

    def unsubscribe(self, sub: Subscriber) -> None:
        with self._lock:
            if sub in self._subs:
                self._subs.remove(sub)

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subs)

    def publish_frame(self, frame: dict) -> None:
        """Decimate by wall-clock; latest-wins per subscriber."""
        now = time.monotonic()
        min_interval = 1.0 / self.frame_hz
        with self._lock:
            if now - self._last_frame_t < min_interval:
                return
            self._last_frame_t = now
            subs = list(self._subs)
        for sub in subs:
            sub.push_frame(frame)
        self._published_frames += 1

    def publish_route(self, msg: dict) -> None:
        """The tap's single sink: route by the message's own `t` field."""
        kind = msg.get("t") or "spec"
        if kind == "frame":
            self.publish_frame(msg)
        else:
            self.publish_event(kind, msg)

    def publish_event(self, kind, payload: dict) -> None:
        with self._lock:
            subs = list(self._subs)
        for sub in subs:
            sub.push_event(kind, payload)

    def stats(self) -> dict:
        with self._lock:
            return {"subscribers": len(self._subs),
                    "frames_published": self._published_frames,
                    "frame_hz": self.frame_hz}
